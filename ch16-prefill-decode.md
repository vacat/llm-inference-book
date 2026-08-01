# 第 16 章 Prefill 与 Decode：两条推理路径

## 本章导读

第 1 章我们提到推理分 Prefill 和 Decode 两阶段。这章我们深入源码，看这两条路径在实现上的根本区别，以及为什么需要不同的策略。

读完本章你会理解：
- Prefill 为什么用"层优先"（layer-major）策略
- Decode 为什么用"token 优先"（token-major）策略
- 两种策略的并行性和内存访问模式差异
- prefill 的分批处理

**前置知识**：第 14-15 章

---

## 16.1 两种策略的本质区别

```
Prefill: 一次处理 N 个 token (N 可能几千)
  -> 可以把 N 个 token 的计算并行化
  -> 关键：最大化吞吐量

Decode: 每次处理 1 个 token
  -> 天然串行 (依赖上一个 token)
  -> 关键：最小化延迟
```

### 层优先 vs Token 优先

```
Token 优先 (Decode 路径):
  for token in [当前 token]:
    for layer in [0..42]:
      处理这一层
  
  每次只处理 1 个 token，跑完全部 43 层
  -> forward_token_raw_swa_cpu_decode_scratch()

层优先 (Prefill 路径):
  for layer in [0..42]:
    for token in [所有 prompt token]:
      处理这一层的所有 token
  
  一次处理所有 token，但一层一层来
  -> prefill_layer_major_cpu()
```

> **为什么 Prefill 用层优先？** 因为注意力需要同一层所有 token 的 KV。层优先策略在处理每一层时，把该层所有 token 的 Q/K/V 一次性算好，然后做批量注意力。这样矩阵乘法可以高效并行。

---

## 16.2 Decode 路径回顾

Decode 路径我们在第 14 章已经详细讲过。核心是 `forward_token_raw_swa_cpu_decode_scratch()`（`ds4.c:13635`）：

```c
// 处理 1 个 token，跑 43 层
for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
    layer_forward_raw_swa_one(next, ..., cur, il, pos, token, ...);
    swap(cur, next);
}
output_logits_one_decode_scratch(logits, ..., cur, ...);
```

### Decode 的特点

```
优点:
  - 每次只处理 1 个 token，内存占用小
  - KV 缓存只需读不需要写（只写当前 token 的 KV）
  - 适合 GPU 的低延迟执行

缺点:
  - 矩阵乘法是矩阵-向量乘法 (matvec)，计算密度低
  - 内存带宽瓶颈：要读整个模型权重，但只算 1 个 token
  - 无法利用批量并行
```

---

## 16.3 Prefill 路径：层优先

`prefill_layer_major_cpu()`（`ds4.c:13695`）是 Prefill 的核心：

```c
static void prefill_layer_major_cpu(
        float *logits, const ds4_model *model, const ds4_weights *weights,
        ds4_kv_cache *cache, const token_vec *prompt, ...) {

    const uint64_t n_tok = prompt->len;  // prompt 的 token 数
    float *cur = xmalloc(n_tok * hc_dim * sizeof(float));   // 所有 token 的输入
    float *next = xmalloc(n_tok * hc_dim * sizeof(float));  // 所有 token 的输出
    float *attn = xmalloc(n_tok * hc_dim * sizeof(float));  // 注意力中间结果

    // 1. 所有 token 的 Embedding
    for (uint64_t t = 0; t < n_tok; t++) {
        embed_token_f16(model, weights, prompt->v[t], plain);
        hc_from_plain_embedding(cur + t * hc_dim, plain, DS4_N_EMBD, DS4_N_HC);
    }

    // 2. 逐层处理，每层处理所有 token
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        // 批量注意力：所有 token 同时算 Q/K/V + 注意力
        layer_attention_raw_swa_batch(attn, model, &weights->layer[il],
                                      &cache->layer[il], cur, n_tok, il, ...);

        // 批量 FFN：所有 token 同时做 MoE
        layer_ffn_shared_batch(next, model, &weights->layer[il],
                               attn, prompt->v, n_tok, il, ...);

        swap(cur, next);  // 交换缓冲
    }

    // 3. 最后一层 -> logits
    output_logits_one_decode_scratch(logits, ..., cur + (n_tok-1) * hc_dim, ...);
}
```

### 层优先的数据流

```
Prompt: [token₀, token₁, token₂, ..., token_N]

第 1 步: 全部 Embedding
  cur = [emb₀, emb₁, emb₂, ..., emb_N]

第 2 步: 第 0 层 (所有 token 同时)
  ┌─────────────────────────────────────────┐
  │ layer 0:                                │
  │   批量 Q/KV 投影:  [emb₀..emb_N] -> Q/KV│
  │   批量注意力:      Q × KV_cache         │
  │   批量 MoE:        [attn₀..attn_N]      │
  └─────────────────────────────────────────┘
  cur = [layer0_out₀, ..., layer0_out_N]

第 3 步: 第 1 层 (所有 token 同时)
  ... 同上 ...

... 直到第 42 层 ...

最后: 取最后一个 token 的输出 -> logits
```

### 批量注意力的优势

```
Decode (1 个 token):
  matvec(W, x)  -> 矩阵 × 向量
  GPU 利用率低（大量线程闲置）

Prefill (N 个 token):
  matmul(W, X)  -> 矩阵 × 矩阵
  GPU 利用率高（所有线程都在算）
  
  N=1000 时，计算量是 Decode 的 1000 倍
  但时间只多了 ~50 倍（并行加速）
```

---

## 16.4 Prefill 的分批处理

如果 prompt 很长（几千 token），一次性处理可能内存不够。ds4 支持**分批 prefill**：

```
prefill_chunk = 512 (可配置)

Prompt: [token₀, ..., token₅₁₁] [token₅₁₂, ..., token₁₀₂₃] [token₁₀₂₄, ...]
         ↑ 第 1 批 (512)          ↑ 第 2 批 (512)            ↑ 第 3 批
```

每批处理 512 个 token，逐批推进。每批结束后更新 KV 缓存，下一批可以看到前面的 KV。

在 `ds4_session_sync_internal`（`ds4.c:58093`）中，prefill 通过 `prefill_chunk` 参数控制批大小。`ds4_effective_prefill_chunk()`（`ds4.c:12091`）计算有效的 chunk 大小。

> **进度显示**：你在终端看到的 `ds4: prefill layer 5/43` 就是层优先 prefill 的进度。每处理完一层，打印一次进度。

---

## 16.5 两条路径的 KV 缓存交互

### Prefill 阶段

Prefill 时，每层的批量注意力会**一次性写入所有 token 的 KV**：

```
layer_attention_raw_swa_batch():
  for token in [0..N]:
    算 Q, KV
    kv_cache_push_raw(cache, KV)  ← 逐个写入
    注意力: Q 看前面所有已写入的 KV
```

Prefill 结束后，KV 缓存有了所有 prompt token 的 KV。

### Decode 阶段

Decode 时，每次只写 1 个 token 的 KV：

```
layer_forward_raw_swa_one():
  算 Q, KV
  kv_cache_push_raw(cache, KV)  ← 写 1 个
  注意力: Q 看缓存中的所有 KV
```

### 增量 Sync（复用 KV 缓存）

如果新 prompt 和旧的有公共前缀（第 3 章），ds4 只 prefill 新增部分：

```c
// ds4.c:58123
if (s->checkpoint_valid &&
    ds4_tokens_starts_with(prompt, &s->checkpoint))
{
    // 只处理新增 token，用 decode 路径逐个走
    for (int i = s->checkpoint.len; i < prompt->len; i++) {
        forward_token_raw_swa_cpu_decode_scratch(...);
    }
}
```

> **有趣的设计**：增量 sync 不走 prefill（批量），而是走 decode（逐个）。因为新增 token 通常很少，批量优势不明显，而且 decode 路径可以精确复用已有 KV 缓存。

---

## 16.6 两条路径的性能对比

```
                    Prefill              Decode
─────────────────────────────────────────────────────
处理 token 数       N (几百~几千)         1
矩阵运算            矩阵×矩阵 (matmul)    矩阵×向量 (matvec)
GPU 利用率          高 (>80%)             低 (~5%)
瓶颈                计算                   内存带宽
并行性              高 (token 间独立)     低 (依赖上一步)
速度                ~2000 tok/s (Metal)   ~65 tok/s (Metal)
```

### 为什么 Decode 慢？

```
Decode 每个 token:
  - 读全部模型权重 (~86GB 中按需读取)
  - 但只做 1 个 token 的计算
  - 大部分时间在等数据从内存到计算单元

  计算量: ~37B 激活参数 × 2 = 74 GFLOP
  内存读取: ~15GB (激活的权重)
  
  算术强度 = 74G / 15G = ~5 FLOP/byte
  GPU 需要 >100 FLOP/byte 才能打满计算单元
  -> 内存带宽瓶颈！
```

> **这就是推测解码（第 22 章）的动机**：既然 Decode 时 GPU 计算单元闲着，不如让它同时猜几个 token，猜对了就赚了。

---

## 16.7 Metal 路径的差异

在 Metal/GPU 路径上，Prefill 和 Decode 有不同的图（graph）实现：

```
Metal Prefill:
  metal_graph_prefill_layer_major()
  -> 编译好的 Metal 着色器批量执行
  -> 所有 token 共享一份着色器调用

Metal Decode:
  metal_graph_eval_token_raw_swa()
  -> 每次调用执行一层的着色器
  -> 43 次调用 = 43 层
```

Metal 路径把 CPU 参考实现中的循环展开成 GPU 着色器调用，每个着色器处理大量并行数据。

---

## 本章小结

- Prefill 用层优先策略：逐层处理所有 token，最大化批量并行
- Decode 用 token 优先策略：逐 token 跑 43 层，最小化延迟
- Prefill 做矩阵×矩阵乘法，GPU 利用率高；Decode 做矩阵×向量乘法，GPU 利用率低
- Decode 的瓶颈是内存带宽（读权重的时间 > 计算时间）
- 分批 prefill 支持超长 prompt，每批 `prefill_chunk` 个 token
- 增量 sync 复用 KV 缓存，只 prefill 新增 token
- Prefill ~2000 tok/s，Decode ~65 tok/s（Metal 估算）

## 动手实验

1. 在 `ds4.c:13695` 阅读 `prefill_layer_major_cpu`，对比 `forward_token_raw_swa_cpu_decode_scratch`
2. 运行 `./ds4 -p "$(cat speed-bench/promessi_sposi.txt)" -n 1`，观察 prefill 进度和首字时间
3. 思考：为什么增量 sync 用 decode 路径而不是 prefill？（提示：批量大小 + KV 精确复用）

## 下一章预告

第 17 章，采样策略--模型输出 logits 后，怎么决定下一个 token？Temperature、Top-P、Min-P 各是什么？
