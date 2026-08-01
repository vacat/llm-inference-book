# 第 11 章 Attention：注意力机制

## 本章导读

注意力机制是 Transformer 的核心。这一章我们把注意力拆成四步：Q/K/V 投影 -> 点积打分 -> Softmax 归一化 -> 加权求和。每一步都对照 ds4 的 CPU 参考代码逐行讲解。

读完本章你会理解：
- Q/K/V 各自代表什么
- 点积注意力为什么有效
- 数值稳定的 Softmax 实现
- ds4 的 attention sinks（注意力池）是什么
- GQA 64:1 如何在代码中体现

**前置知识**：第 8-10 章

---

## 11.1 注意力的直觉

注意力机制模拟人类阅读时的行为：**在看当前词时，会"注意"上下文中相关的词**。

```
句子: "The cat sat on the mat because it was tired"
                                            ↑ "it" 指代谁？

注意力计算后:
  "it" 对 "cat" 的注意力权重 = 0.7  (高)
  "it" 对 "mat" 的注意力权重 = 0.1  (低)
  "it" 对 "sat" 的注意力权重 = 0.15 (中)
  ...
```

实现方式：每个位置生成三个向量--Query（查询）、Key（键）、Value（值），然后：

```
Query  = "我想找什么"
Key    = "我有什么" 
Value  = "我的内容"

注意力分数 = Query · Key  (点积 = 相似度)
输出 = Σ(注意力分数 × Value)  (加权求和)
```

> **类比**：你去图书馆找书。Query 是你的搜索词，Key 是每本书的标签，Value 是书的内容。搜索词和标签越匹配（点积越大），你越关注那本书的内容。

---

## 11.2 Q/K/V 投影

在注意力之前，输入向量要通过三个线性变换（矩阵乘法），分别生成 Q、K、V。

### 标准注意力

```
Q = W_q × x    (x 是 4096 维, Q 是 32768 维 = 64头 × 512维)
K = W_k × x
V = W_v × x
```

### DeepSeek V4 的 MLA 低秩投影

DeepSeek V4 用低秩分解来压缩 Q 和 KV：

```
Q 的生成 (两步):
  压缩:  q_compressed = W_q_a × x     (4096 -> 1024 维, attn_q_a)
  还原:  Q = W_q_b × q_compressed     (1024 -> 32768 维, attn_q_b)

KV 的生成 (两步):
  压缩:  kv_compressed = W_kv × x     (4096 -> 压缩维, attn_kv)
  还原:  K = W_k_b × kv_compressed
         V = W_v_b × kv_compressed
```

### 源码：Q 投影

`layer_q_projection_with_lora_one_decode_scratch()`（`ds4.c:10120`）：

```c
static void layer_q_projection_with_lora_one_decode_scratch(
        const ds4_model *model, const ds4_layer_weights *layer,
        const float *norm, float *q, float *qr_norm,
        ds4_cpu_decode_scratch *scratch) {

    const float *q_a_norm = tensor_data(model, layer->attn_q_a_norm);

    // 第 1 步：压缩 x -> qr (4096 -> 1024)
    matvec_q8_0_decode_scratch(scratch->qr, model, layer->attn_q_a,
                               norm, scratch);

    // 第 2 步：对压缩结果做 RMSNorm
    rms_norm_weight(qr_norm, scratch->qr, q_a_norm,
                    DS4_N_LORA_Q, DS4_RMS_EPS);

    // 第 3 步：还原 qr_norm -> q (1024 -> 32768)
    matvec_q8_0_decode_scratch(q, model, layer->attn_q_b,
                               qr_norm, scratch);

    // 第 4 步：对每个头独立做 RMSNorm
    head_rms_norm_inplace(q, DS4_N_HEAD, DS4_N_HEAD_DIM, DS4_RMS_EPS);
}
```

`matvec_q8_0_decode_scratch` 是量化矩阵-向量乘法（第 18 章详讲）。这里你只需知道它做的是 `output = weight × input`。

注意 Q 投影有**两次 RMSNorm**：一次在压缩后（对 1024 维），一次在还原后（逐头 512 维）。这是 MLA 的设计，保证低秩分解后的数值稳定性。

### 源码：KV 投影

`layer_kv_projection_normed_one_decode_scratch()`（`ds4.c:10135`）：

```c
static void layer_kv_projection_normed_one_decode_scratch(
        const ds4_model *model, const ds4_layer_weights *layer,
        const float *normed, float *kv,
        ds4_cpu_decode_scratch *scratch) {

    const float *kv_norm = tensor_data(model, layer->attn_kv_a_norm);

    // 压缩 x -> kv_raw (4096 -> head_dim=512)
    matvec_q8_0_decode_scratch(scratch->kv_raw, model, layer->attn_kv,
                               normed, scratch);

    // RMSNorm
    rms_norm_weight(kv, scratch->kv_raw, kv_norm,
                    DS4_N_HEAD_DIM, DS4_RMS_EPS);
}
```

> **GQA 64:1 的体现**：注意 KV 投影只生成 **一个** `kv` 向量（512 维），而不是 64 个。64 个查询头共享这同一个 KV。这就是 `n_head_kv = 1` 的含义--所有头用同一份 Key 和 Value。

---

## 11.3 注意力计算核心

注意力计算在 `layer_attention_rows_one()`（`ds4.c:10369`）。这是全书最重要的函数之一：

```c
static void layer_attention_rows_one(
        float *out_heads,           // 输出：64 个头的注意力结果
        const ds4_model *model,
        const ds4_layer_weights *layer,
        const float *q,             // Query: 64 头 × 512 维
        const float *kv_rows,       // KV 缓存: n_kv 行 × 512 维
        uint32_t n_kv) {            // KV 行数（历史 token 数 + 1）

    const float *sinks = tensor_data(model, layer->attn_sinks);
    const float kq_scale = 1.0f / sqrtf((float)DS4_N_HEAD_DIM);
    // ...
```

### 第 1 步：逐头计算注意力分数

```c
    for (uint32_t h = 0; h < DS4_N_HEAD; h++) {
        const float *qh = q + (uint64_t)h * DS4_N_HEAD_DIM;  // 第 h 个头的 Q

        float max_score = sinks[h];  // 初始最大值 = sink 值
        for (uint32_t r = 0; r < n_kv; r++) {
            const float *kv = kv_rows + (uint64_t)r * DS4_N_HEAD_DIM;
            score[r] = dot_f32(qh, kv, DS4_N_HEAD_DIM) * kq_scale;
            //       ↑ Q 和 KV 的点积   ↑ 缩放 1/sqrt(d)
            if (score[r] > max_score) max_score = score[r];
        }
```

对每个头 `h`：
1. 取出该头的 Query 向量 `qh`（512 维）
2. 遍历所有历史 KV 行，计算点积 `qh · kv`（相似度）
3. 乘以 `kq_scale = 1/sqrt(512)` 防止分数过大
4. 记录最大值 `max_score`（用于数值稳定）

**为什么除以 sqrt(d)？** 当维度 d 很大时，点积的值会很大（因为是 d 个乘积之和），导致 Softmax 饱和（梯度消失）。除以 `sqrt(d)` 把分数拉回合理范围。

### dot_f32：点积计算

`dot_f32()`（`ds4.c:10310`）用了 ARM NEON 向量化：

```c
static inline float dot_f32(const float *a, const float *b, uint32_t n) {
#if defined(__ARM_NEON)
    // NEON 版本：一次处理 8 个 float
    float32x4_t acc0 = vdupq_n_f32(0.0f);
    float32x4_t acc1 = vdupq_n_f32(0.0f);
    for (; i + 8 <= n; i += 8) {
        acc0 = vfmaq_f32(acc0, vld1q_f32(a + i),     vld1q_f32(b + i));
        acc1 = vfmaq_f32(acc1, vld1q_f32(a + i + 4), vld1q_f32(b + i + 4));
    }
    // 累加两个向量的和
    float acc = vaddvq_f32(vaddq_f32(acc0, acc1));
    // 处理剩余元素
    for (; i < n; i++) acc += a[i] * b[i];
    return acc;
#else
    // 标量版本
    float acc = 0.0f;
    for (uint32_t i = 0; i < n; i++) acc += a[i] * b[i];
    return acc;
#endif
}
```

NEON 是 ARM 的 SIMD 指令集，一次处理 4 个 float（用两个寄存器并行就是 8 个）。`vfmaq_f32` 是融合乘加（FMA）指令，在一个时钟周期内完成 `acc += a × b`。

### 第 2 步：数值稳定的 Softmax

```c
        float *oh = out_heads + (uint64_t)h * DS4_N_HEAD_DIM;
        memset(oh, 0, DS4_N_HEAD_DIM * sizeof(oh[0]));

        float denom = expf(sinks[h] - max_score);  // sink 的权重
        for (uint32_t r = 0; r < n_kv; r++) {
            const float weight = expf(score[r] - max_score);  // ← 减去最大值！
            const float *kv = kv_rows + (uint64_t)r * DS4_N_HEAD_DIM;
            denom += weight;
            axpy_f32(oh, kv, weight, DS4_N_HEAD_DIM);
            // oh += weight × kv  (加权累加)
        }

        const float inv = 1.0f / denom;
        scale_f32(oh, inv, DS4_N_HEAD_DIM);
        // oh *= 1/denom  (归一化)
```

### Softmax 的数学

标准 Softmax：

```
weight[i] = exp(score[i]) / Σ exp(score[j])
```

**数值稳定版**：先减去最大值 `max_score`：

```
weight[i] = exp(score[i] - max) / Σ exp(score[j] - max)
```

为什么这样做？因为 `exp()` 在输入很大时会溢出（`exp(1000) = inf`）。减去最大值后，最大的指数变成 `exp(0) = 1`，其他的都小于 1，不会溢出。数学上结果完全相同（分子分母同除以 `exp(max)`）。

### 第 3 步：加权求和

`axpy_f32()`（`ds4.c:10329`）做 `y += a × x`：

```c
static inline void axpy_f32(float *y, const float *x, float a, uint32_t n) {
    for (uint32_t i = 0; i < n; i++) y[i] += a * x[i];
}
```

对每个 KV 行：`output += weight × kv_value`。最后除以 `denom` 归一化。

```
output = Σ weight[r] × kv[r] / Σ weight[r]
       = Σ (softmax_weight[r] × kv[r])
```

这就是注意力的输出：**按注意力权重加权平均所有 Value**。

---

## 11.4 Attention Sinks（注意力池）

注意代码中的 `sinks`：

```c
const float *sinks = tensor_data(model, layer->attn_sinks);
// ...
float max_score = sinks[h];
// ...
float denom = expf(sinks[h] - max_score);
```

`attn_sinks` 是一个可学习的参数（每个头一个值），代表一个"虚拟的"注意力目标。

```
普通注意力:
  weight[token] = softmax(Q·K[token])
  Σ weight[token] = 1

带 sink 的注意力:
  weight[sink] = softmax(sink_value)     ← 一个虚拟的"池"
  weight[token] = softmax(Q·K[token])
  Σ (weight[sink] + weight[token]) = 1
```

> **Sink 的作用**：在滑动窗口注意力中，窗口外的旧 token 被丢弃。但研究发现，模型会向"第一个 token"倾注大量注意力（即使它已经不在窗口内）。Sink 相当于一个"替身"，吸收那些本该给被丢弃 token 的注意力，避免信息丢失。

---

## 11.5 完整注意力流程

把投影和注意力计算串起来：

```
输入向量 x (4096维)
    │
    ├─ RMSNorm(x, attn_norm) -> norm (4096维)
    │
    ├─ Q 投影:
    │   matvec(attn_q_a, norm) -> qr (1024维)     [压缩]
    │   RMSNorm(qr, q_a_norm) -> qr_norm (1024维)
    │   matvec(attn_q_b, qr_norm) -> q (32768维)  [还原]
    │   head_rms_norm(q) -> q (32768维)            [逐头归一化]
    │   RoPE(q) -> q                               [位置编码]
    │
    ├─ KV 投影:
    │   matvec(attn_kv, norm) -> kv_raw (512维)    [压缩]
    │   RMSNorm(kv_raw, kv_norm) -> kv (512维)
    │   RoPE(kv) -> kv                             [位置编码]
    │   存入 KV 缓存                                [更新缓存]
    │
    ├─ 注意力计算 (layer_attention_rows_one):
    │   for 每个头 h (共64个):
    │     scores[r] = (Q[h] · KV_cache[r]) / sqrt(512)
    │     weights[r] = softmax(scores[r], sinks[h])
    │     output[h] = Σ weights[r] × KV_cache[r]
    │
    ├─ 输出投影:
    │   分组投影 (attn_output_a + attn_output_b)   [低秩还原]
    │   反向 RoPE (如果需要)
    │
    └─ 残差连接: x = x + attention_output
```

---

## 11.6 性能分析

### Decode 阶段的瓶颈

Decode 时每次只有 1 个 token，但注意力要扫描所有历史 KV：

```
n_kv = 上下文长度 (可能几千到几万)
每个头: n_kv 次点积 (每次 512 维) + n_kv 次加权累加 (每次 512 维)
64 个头: 64 × n_kv × 1024 次浮点运算

如果 n_kv = 4096:
  64 × 4096 × 1024 = 2.68 亿次浮点运算
  -> 这在 decode 时是大头之一
```

但更大的瓶颈是**内存带宽**：要把所有 KV 行从内存读到 CPU/GPU：

```
KV 缓存大小 = n_kv × 512 × 4 字节 = 4096 × 2048 = 8MB (每个头)
64 个头共享 1 个 KV 头 -> 仍然是 8MB
每次 decode 都要读 8MB -> 内存带宽瓶颈
```

这就是为什么 Flash Attention（第 20 章）和 KV 缓存压缩（第 15 章）如此重要。

### GQA 的优势

```
没有 GQA (n_head_kv = 64):
  KV 缓存 = 64 × n_kv × 512 × 4 = 512MB (n_kv=4096时)

有 GQA 64:1 (n_head_kv = 1):
  KV 缓存 = 1 × n_kv × 512 × 4 = 8MB (n_kv=4096时)
  
内存减少 64 倍！
```

---

## 本章小结

- 注意力 = Q·K 打分 -> Softmax 归一化 -> 加权 V
- DeepSeek V4 用 MLA 低秩投影压缩 Q/KV，再用 GQA 64:1 共享 KV
- Q 投影：压缩(4096→1024) -> RMSNorm -> 还原(1024→32768) -> 逐头 RMSNorm -> RoPE
- KV 投影：压缩(4096→512) -> RMSNorm -> RoPE -> 存缓存
- Softmax 减去最大值保证数值稳定
- Attention Sinks 吸收被丢弃 token 的注意力
- `dot_f32` 用 NEON 向量化加速
- Decode 瓶颈在 KV 缓存的内存带宽

## 动手实验

1. 在 `ds4.c:10369` 阅读 `layer_attention_rows_one`，画出数据流图
2. 手算：2 个 KV 行，分数 [3.0, 1.0]，手动做 Softmax（先减最大值）
3. 在 `ds4.c:10310` 查看 `dot_f32`，对比 NEON 版和标量版的区别

## 下一章预告

第 12 章，FFN（前馈网络）与 SwiGLU 激活--注意力之后的另一半。我们会看到专家网络的结构和门控激活机制。
