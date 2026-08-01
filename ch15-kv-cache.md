# 第 15 章 KV 缓存原理与实现

## 本章导读

KV 缓存是大模型推理中最重要的数据结构。没有它，每生成一个 token 都要重新处理全部历史，速度会慢到不可用。这章我们拆解 ds4 的 KV 缓存实现，包括滑动窗口、压缩缓存、FP8 量化。

读完本章你会理解：
- KV 缓存为什么能省大量重复计算
- 滑动窗口缓存的工作方式
- 压缩 KV 缓存如何支持超长上下文
- KV 缓存的内存占用计算

**前置知识**：第 11、14 章

---

## 15.1 为什么需要 KV 缓存

回顾注意力的计算：每个 token 要和**所有历史 token** 算注意力分数。

```
生成第 1 个 token: Q₁ 要和 K₀ 算
生成第 2 个 token: Q₂ 要和 K₀, K₁ 算
生成第 3 个 token: Q₃ 要和 K₀, K₁, K₂ 算
...
生成第 N 个 token: Q_N 要和 K₀, K₁, ..., K_{N-1} 算
```

每次生成都要重新算 K 和 V 吗？**不需要**--因为 K 和 V 只取决于输入 token 和权重，与位置无关（RoPE 之前的 K/V）。所以可以**缓存起来**：

```
没有 KV 缓存:
  生成第 N 个 token: 重新算 K₀..K_{N-1}, V₀..V_{N-1}  → O(N²) 总计算量
  生成 1000 个 token: ~50 万次重复计算

有 KV 缓存:
  生成第 N 个 token: 只算 K_N, V_N，存入缓存
  注意力时从缓存取 K₀..K_{N-1}, V₀..V_{N-1}
  → O(N) 总计算量（每个 token 只算一次 K/V）
  生成 1000 个 token: 1000 次 K/V 计算
```

> **省了什么**：不是省了注意力计算（注意力还是要扫描所有历史 KV），而是省了**重复计算 K/V 的投影**。每个 token 的 K/V 只算一次，之后从缓存读取。

---

## 15.2 ds4 的缓存结构

`ds4_layer_cache`（`ds4.c:12058`）定义了每层的缓存：

```c
typedef struct {
    // 原始 KV 缓存（滑动窗口）
    float *raw_kv;          // 原始 KV 行数组
    uint32_t n_raw;         // 当前行数
    uint32_t cap_raw;       // 容量 (= SWA 窗口大小)

    // 压缩 KV 缓存
    uint32_t compress_ratio;// 压缩比 (0=不压缩, 2 或 4)
    uint32_t comp_cap;      // 压缩缓存容量
    uint32_t n_comp;        // 当前压缩行数
    float *attn_comp_kv;    // 压缩 KV 行
    float *attn_state_kv;   // 压缩器状态 (KV)
    float *attn_state_score;// 压缩器状态 (分数)

    // 索引器压缩缓存 (ratio=4 时)
    uint32_t n_index_comp;
    float *index_comp_kv;
    float *index_state_kv;
    float *index_state_score;
} ds4_layer_cache;
```

整个模型的缓存（`ds4.c:12078`）：

```c
typedef struct {
    ds4_layer_cache layer[DS4_MAX_LAYER];  // 43 层，每层一个
    uint32_t head_dim;
} ds4_kv_cache;
```

---

## 15.3 原始 KV 缓存：滑动窗口

### 滑动窗口的原理

`raw_kv` 是一个**环形缓冲区**，容量为 `cap_raw = DS4_N_SWA = 128`。它只保留最近 128 个 token 的 KV：

```
cap_raw = 128 (SWA 窗口大小)

初始状态 (n_raw < cap_raw): 不断追加
  [KV₀] [KV₁] [KV₂] ... [KV₁₂₇]  n_raw=128, 满了

满了之后 (n_raw >= cap_raw): 滑动！
  丢弃最旧的, 整体前移, 新的放最后
  
  [KV₁] [KV₂] ... [KV₁₂₇] [KV₁₂₈]  ← KV₀ 被丢弃
  [KV₂] [KV₃] ... [KV₁₂₈] [KV₁₂₉]  ← KV₁ 被丢弃
  ...
```

### 源码：kv_cache_push_raw

`kv_cache_push_raw()`（`ds4.c:12321`）：

```c
static void kv_cache_push_raw(ds4_layer_cache *cache, const float *kv) {
    if (cache->n_raw < cache->cap_raw) {
        // 还没满：直接追加
        float *dst = cache->raw_kv + cache->n_raw * DS4_N_HEAD_DIM;
        for (uint32_t i = 0; i < DS4_N_HEAD_DIM; i++)
            dst[i] = f16_to_f32(f32_to_f16(kv[i]));  // 存为 F16 省空间
        cache->n_raw++;
        return;
    }

    // 满了：滑动窗口
    memmove(cache->raw_kv,
            cache->raw_kv + DS4_N_HEAD_DIM,           // 从第 1 行开始
            (cache->cap_raw - 1) * DS4_N_HEAD_DIM * sizeof(float));  // 前移
    float *dst = cache->raw_kv + (cache->cap_raw - 1) * DS4_N_HEAD_DIM;
    for (uint32_t i = 0; i < DS4_N_HEAD_DIM; i++)
        dst[i] = f16_to_f32(f32_to_f16(kv[i]));  // 新行放最后
}
```

### F16 存储

```c
dst[i] = f16_to_f32(f32_to_f16(kv[i]));
```

KV 缓存用 F16 存储而不是 F32，内存减半。精度损失可以接受（KV 只是注意力的中间结果，不是最终输出）。

```
F32 存储: 128 行 × 512 维 × 4 字节 = 256 KB / 每层
F16 存储: 128 行 × 512 维 × 2 字节 = 128 KB / 每层
43 层: 43 × 128 KB = 5.5 MB  (原始 SWA 缓存)
```

### 滑动窗口的代价

丢弃旧 KV 意味着注意力只能"看到"最近 128 个 token。但信息没有完全丢失：

```
Layer 0:  能看到最近 128 个 token
Layer 1:  Layer 0 的输出已经融合了 128 个 token 的信息
          Layer 1 看到 Layer 0 的输出 + 最近 128 个 -> 间接看到 256 个
Layer 2:  间接看到更多
...
Layer 42: 间接看到很远的 token
```

加上压缩缓存（下一节），远处的信息以压缩形式保留。

---

## 15.4 压缩 KV 缓存

对于有 `compress_ratio != 0` 的层，旧 KV 不会直接丢弃，而是被**压缩**。

### 压缩原理

```
compress_ratio = 2 的层:

每 2 个 token，把它们的 KV 压缩成 1 个压缩行:
  [KV₀, KV₁] -> 压缩 -> [CompKV₀]
  [KV₂, KV₃] -> 压缩 -> [CompKV₁]
  ...

压缩比 2:1，128 个原始 token -> 64 个压缩行
```

### kv_cache_push_comp

`kv_cache_push_comp()`（`ds4.c:12336`）：

```c
static void kv_cache_push_comp(float *rows, uint32_t *n_rows,
                               uint32_t cap_rows, uint32_t row_dim,
                               const float *kv) {
    if (*n_rows >= cap_rows)
        ds4_die("compressed KV cache capacity exceeded");
    float *dst = rows + (*n_rows) * row_dim;
    for (uint32_t i = 0; i < row_dim; i++)
        dst[i] = f16_to_f32(f32_to_f16(kv[i]));  // 同样用 F16 存储
    (*n_rows)++;
}
```

压缩缓存的容量（`comp_cap`）在 `kv_cache_init` 中计算（`ds4.c:12278`）：

```c
const uint32_t comp_cap = ctx_size / ratio + 2;
```

上下文长度 32768，ratio=2：压缩缓存容量 = 16384 + 2 行。

### 混合注意力

有压缩的层，注意力计算扫描**两部分** KV（第 14 章阶段 5）：

```
注意力输入 = 原始 SWA 缓存 (最近 128 行)
            + 压缩缓存 (远处的压缩行)
            + 索引器选择 (ratio=4 时，只看部分压缩行)
```

```
                     时间线 ──────────────────>
                    
  [压缩KV₀][压缩KV₁]...[压缩KV_N] [原始KV₀..KV₁₂₇] [当前]
   ↑ 远处的压缩表示                      ↑ 最近128个     ↑ Q
   ↑ 索引器选择 top-512 个参与注意力
```

---

## 15.5 缓存初始化

`kv_cache_init()`（`ds4.c:12264`）为 43 层各分配缓存：

```c
static void kv_cache_init(ds4_kv_cache *cache, uint32_t ctx_size,
                          uint32_t raw_cap) {
    if (raw_cap == 0) raw_cap = ds4_default_raw_cap(ctx_size);  // 默认 128
    cache->head_dim = DS4_N_HEAD_DIM;

    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        const uint32_t ratio = ds4_layer_compress_ratio(il);

        // 原始 SWA 缓存
        cache->layer[il].cap_raw = raw_cap;  // 128
        cache->layer[il].raw_kv = xmalloc_zeroed(raw_cap * DS4_N_HEAD_DIM, sizeof(float));

        if (ratio != 0) {
            // 压缩缓存
            const uint32_t comp_cap = ctx_size / ratio + 2;
            cache->layer[il].comp_cap = comp_cap;
            cache->layer[il].attn_comp_kv = xmalloc_zeroed(comp_cap * DS4_N_HEAD_DIM, sizeof(float));
            // 压缩器状态
            cache->layer[il].attn_state_kv = xmalloc_zeroed(...);
            cache->layer[il].attn_state_score = xmalloc(...);
            // 初始化分数为 -inf
            for (...) cache->layer[il].attn_state_score[i] = DS4_NEG_INF;

            if (ratio == 4) {
                // 索引器缓存（额外的一层压缩）
                cache->layer[il].index_comp_kv = xmalloc_zeroed(...);
                // ...
            }
        }
    }
}
```

---

## 15.6 内存占用计算

### 理论 KV 缓存大小

```
每层原始缓存:
  128 行 × 512 维 × 2 字节(F16) = 128 KB

每层压缩缓存 (ratio=2, ctx=32768):
  16384 行 × 512 维 × 2 字节 = 16 MB

每层压缩缓存 (ratio=4, ctx=32768):
  8192 行 × 512 维 × 2 字节 = 8 MB

43 层总计 (假设全部 ratio=2):
  43 × (128 KB + 16 MB) ≈ 700 MB
```

> **对比 GQA 的效果**：如果没有 GQA（n_head_kv=64），KV 缓存会是 64 倍：
> ```
> 43 × 64 × (128 KB + 16 MB) ≈ 44 GB
> ```
> 有了 GQA 64:1，只需要 ~700 MB。这就是 GQA 对长上下文的关键意义。

### FP8 量化进一步压缩

在 `layer_forward_raw_swa_one` 中（`ds4.c:13519`）：

```c
dsv4_fp8_kv_quantize_row_inplace_cpu(scratch->kv, DS4_N_HEAD_DIM, DS4_N_ROT);
```

KV 在存入缓存前还会做 FP8 量化（第 18 章），把 F16 的 2 字节进一步压缩到 1 字节。Metal 路径上 KV 缓存用 FP8 存储，内存再减半。

```
F32 -> F16 -> FP8:
  4 字节 -> 2 字节 -> 1 字节
  
原始 SWA 缓存 (FP8):
  128 × 512 × 1 = 64 KB / 每层
  43 层 = 2.75 MB
```

---

## 15.7 KV 缓存的持久化

ds4 支持 KV 缓存的**保存和加载**（检查点）。这在多轮对话中很有用：保存当前 KV 缓存，下次对话时直接加载，不用重新 prefill。

相关函数在 `ds4.h:380-400`：

```c
int ds4_session_save_payload(ds4_session *s, FILE *fp, ...);
int ds4_session_load_payload(ds4_session *s, FILE *fp, ...);
int ds4_session_save_snapshot(ds4_session *s, ds4_session_snapshot *snap, ...);
int ds4_session_load_snapshot(ds4_session *s, const ds4_session_snapshot *snap, ...);
```

### 检查点复用

`ds4_session_sync_internal`（第 3 章）的增量复用就是基于 checkpoint：

```c
if (s->checkpoint_valid &&
    prompt->len >= s->checkpoint.len &&
    ds4_tokens_starts_with(prompt, &s->checkpoint))
{
    // 新 prompt 和旧的有公共前缀
    // 只处理新增部分，复用已有 KV 缓存
    for (int i = s->checkpoint.len; i < prompt->len; i++) {
        forward_token_raw_swa_cpu_decode_scratch(...);
        token_vec_push(&s->checkpoint, prompt->v[i]);
    }
}
```

> **多轮对话场景**：用户发了第一条消息，prefill 算了 1000 个 token。用户发第二条消息时，前 1000 个 token 不用重算，只算新增部分。这让连续对话非常快。

---

## 15.8 完整缓存架构

```
ds4_kv_cache (整个模型)
  │
  ├─ layer[0] (ds4_layer_cache)
  │   ├─ raw_kv[128 × 512]       ← 滑动窗口 (最近 128 token)
  │   ├─ attn_comp_kv[N × 512]   ← 压缩缓存 (远处 token 的压缩表示)
  │   ├─ attn_state_kv           ← 压缩器内部状态
  │   ├─ index_comp_kv[N × 128]  ← 索引器压缩 (ratio=4 时)
  │   └─ index_state_kv          ← 索引器内部状态
  │
  ├─ layer[1]
  │   └─ ... 同上 ...
  │
  └─ ...
      └─ layer[42]
          └─ ... 同上 ...

注意力计算时 (每层):
  Q 扫描: raw_kv (128行) + attn_comp_kv (被索引器选中的行)
  → 即使上下文 10 万 token，实际扫描的 KV 行数有限
```

---

## 本章小结

- KV 缓存避免重复计算历史 token 的 K/V 投影，把 O(N²) 降到 O(N)
- `raw_kv` 是滑动窗口缓存：保留最近 128 个 token，满了就滑动
- 压缩缓存把旧 KV 压缩成更少的行，保留远处信息
- 索引器（ratio=4）从压缩缓存中选择 top-K 行参与注意力
- KV 用 F16/FP8 存储，GQA 64:1 让缓存缩小 64 倍
- KV 检查点支持多轮对话的增量复用
- 32768 上下文的 KV 缓存约 700MB（有 GQA + 压缩）

## 动手实验

1. 在 `ds4.c:12058` 查看 `ds4_layer_cache` 结构，理解每个字段的含义
2. 在 `ds4.c:12321` 阅读 `kv_cache_push_raw`，理解滑动窗口的 memmove 逻辑
3. 计算：上下文 128K，ratio=2，43 层，KV 缓存需要多少内存？
4. 思考：为什么滑动窗口满了用 memmove 而不是环形索引？（提示：缓存局部性）

## 下一章预告

第 16 章，Prefill 与 Decode 两条路径--为什么处理输入和生成输出要用不同的策略？
