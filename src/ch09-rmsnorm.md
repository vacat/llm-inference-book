# 第 9 章 RMSNorm：归一化

## 本章导读

在 Transformer 的每一层中，注意力机制和 FFN 之前都要做一次**归一化（Normalization）**。DeepSeek V4 用的是 RMSNorm--一种简单高效的归一化方法。这章我们逐行拆解它的实现。

读完本章你会理解：
- 为什么需要归一化
- RMSNorm 比 LayerNorm 简单在哪里
- 代码中三种 RMSNorm 变体的区别

**前置知识**：第 8 章

---

## 9.1 为什么需要归一化

神经网络在训练和推理时，数据的"尺度"（magnitude）会不断变化。如果不加控制：

```
第 1 层输入:  [0.1, 0.2, 0.3, ...]     ← 尺度正常
第 2 层输入:  [1.2, 3.4, 5.6, ...]      ← 开始变大
第 3 层输入:  [12, 34, 56, ...]         ← 越来越大
...
第 43 层输入: [inf, inf, inf, ...]      ← 数值溢出，模型崩溃
```

归一化的作用是**把每层的输入拉回合理的范围**，防止数值爆炸或消失。

---

## 9.2 LayerNorm vs RMSNorm

### LayerNorm（传统方法）

```
给定向量 x = [x0, x1, ..., xn]

1. 算均值: mean = (x0 + x1 + ... + xn) / n
2. 算方差: var = ((x0-mean)² + (x1-mean)² + ... + (xn-mean)²) / n
3. 归一化: x'i = (xi - mean) / sqrt(var + eps)
4. 缩放:   y_i = x'i * weight[i]
```

### RMSNorm（DeepSeek V4 用的方法）

```
给定向量 x = [x0, x1, ..., xn]

1. 算均方根: rms = sqrt((x0² + x1² + ... + xn²) / n + eps)
2. 归一化:   x'i = xi / rms
3. 缩放:     y_i = x'i * weight[i]
```

**区别**：RMSNorm 不减均值！只除以均方根。这省去了一次遍历（算均值），计算量减半。

> **为什么可以不减均值？** 研究发现，减均值对 Transformer 的效果影响很小，但减均值需要额外计算。RMSNorm 保留了归一化的核心功能（控制尺度），去掉了不必要的计算，在推理时更高效。

---

## 9.3 源码解读：rms_norm_weight

带学习权重的 RMSNorm 在 `rms_norm_weight()`（`ds4.c:6701`）：

```c
static void rms_norm_weight(float *out, const float *x,
                            const float *weight, uint64_t n, float eps) {
    // 1. 算平方和
    double ss = 0.0;
    for (uint64_t i = 0; i < n; i++)
        ss += (double)x[i] * x[i];

    // 2. 算缩放因子: 1 / sqrt(ss/n + eps)
    const float scale = 1.0f / sqrtf((float)(ss / (double)n) + eps);

    // 3. 归一化 + 乘以学习权重
    for (uint64_t i = 0; i < n; i++)
        out[i] = x[i] * scale * weight[i];
}
```

### 逐行解读

**第 1 步：算平方和**
```c
double ss = 0.0;
for (uint64_t i = 0; i < n; i++)
    ss += (double)x[i] * x[i];
```
遍历整个向量，累加每个元素的平方。注意用了 `double`（64 位浮点）而不是 `float`，因为累加很多个数时，`float` 的精度可能不够。

> **为什么用 double？** `n` 可能是 4096，累加 4096 个 `float` 的平方时，如果值较大，`float` 的 23 位尾数可能丢失精度。`double` 有 52 位尾数，安全得多。

**第 2 步：算缩放因子**
```c
const float scale = 1.0f / sqrtf((float)(ss / (double)n) + eps);
```
对应数学公式：

```
scale = 1 / sqrt(Σ(xi²)/n + eps)
```

`eps` 是一个很小的数（`DS4_RMS_EPS`，通常是 1e-6），防止除以零。

**第 3 步：归一化 + 缩放**
```c
for (uint64_t i = 0; i < n; i++)
    out[i] = x[i] * scale * weight[i];
```
每个元素：先除以 RMS（乘以 scale），再乘以学习到的权重 `weight[i]`。

`weight` 是模型训练时学到的参数，允许每个维度有不同的缩放。这给模型"微调"每个维度尺度的能力。

---

## 9.4 三种 RMSNorm 变体

ds4 中有三种 RMSNorm，用于不同场景：

### 1. rms_norm_weight（标准版）

```c
// ds4.c:6701
static void rms_norm_weight(float *out, const float *x,
                            const float *weight, uint64_t n, float eps);
```
有学习权重。用于注意力前（`attn_norm`）和 FFN 前（`ffn_norm`）的归一化。

### 2. rms_norm_no_weight（无权重版）

```c
// ds4.c:6692
static void rms_norm_no_weight(float *out, const float *x, uint64_t n, float eps) {
    double ss = 0.0;
    for (uint64_t i = 0; i < n; i++) ss += (double)x[i] * x[i];
    const float scale = 1.0f / sqrtf((float)(ss / (double)n) + eps);
    for (uint64_t i = 0; i < n; i++) out[i] = x[i] * scale;
    //                                      ↑ 没有 * weight[i]
}
```
没有学习权重。用于 HyperConnection（HC）控制向量的归一化（`ds4.c:9771`）。

### 3. head_rms_norm_inplace（逐头归一化）

```c
// ds4.c:6710
static void head_rms_norm_inplace(float *x, uint32_t n_head,
                                  uint32_t head_dim, float eps) {
    for (uint32_t h = 0; h < n_head; h++) {
        float *head = x + (uint64_t)h * head_dim;
        double ss = 0.0;
        for (uint32_t i = 0; i < head_dim; i++)
            ss += (double)head[i] * head[i];
        const float scale = 1.0f / sqrtf((float)(ss / (double)head_dim) + eps);
        for (uint32_t i = 0; i < head_dim; i++)
            head[i] *= scale;
    }
}
```

**关键区别**：不是对整个向量做一次归一化，而是**对每个注意力头独立做归一化**。

```
标准 RMSNorm: 对 4096 维整体算一次 RMS
  x = [head0(512维) | head1(512维) | ... | head63(512维)]
  → 算一次 RMS(4096 维) → 整体缩放

Head RMSNorm: 对每个头分别算 RMS
  head0: 算 RMS(512 维) → 缩放
  head1: 算 RMS(512 维) → 缩放
  ...
  head63: 算 RMS(512 维) → 缩放
```

这用在 Q 投影之后（`ds4.c:10080`），让每个注意力头有独立的尺度控制。

> **为什么逐头归一化？** 不同注意力头可能关注不同的特征，尺度差异大。逐头归一化让每个头独立调整，避免某个头的尺度影响其他头。

---

## 9.5 RMSNorm 在模型中的位置

每一层有两处用 RMSNorm：

```
输入向量 x (4096维)
    │
    ├─ rms_norm_weight(x, attn_norm)  ← 注意力前归一化
    │   │
    │   └─ 注意力计算
    │       └─ head_rms_norm_inplace(q)  ← Q 投影后逐头归一化
    │       └─ rms_norm_weight(kv, kv_norm)  ← KV 归一化
    │
    ├─ 残差连接
    │
    ├─ rms_norm_weight(x, ffn_norm)  ← FFN 前归一化
    │   │
    │   └─ MoE / FFN 计算
    │
    └─ 残差连接
```

> **Pre-Norm 结构**：DeepSeek V4 用的是 Pre-Norm（归一化在子层之前），不是 Post-Norm（归一化在子层之后）。Pre-Norm 训练更稳定，是现代 Transformer 的标准选择。

---

## 9.6 性能特点

### 计算量

RMSNorm 需要两次遍历向量：
1. 第一次：算平方和（`ss += x[i] * x[i]`）
2. 第二次：归一化 + 缩放（`out[i] = x[i] * scale * weight[i]`）

对于 4096 维向量，这是 8192 次浮点运算--在所有算子中算很少的。

### 内存访问

两次遍历都要读整个向量。4096 × 4 = 16KB 的数据，在 L1 cache 里，访问很快。

### GPU 优化

在 Metal/CUDA 中，RMSNorm 是一个简单的 kernel：每个线程处理几个元素，用共享内存做归约求和。代码在 `metal/norm.metal` 中。

> **融合优化**：GPU 路径经常把 RMSNorm 和后面的矩阵乘法融合在一起，减少一次全局内存读写。这在 `ds4_metal.m` 的图执行路径中实现。

---

## 本章小结

- RMSNorm = 不减均值的 LayerNorm：只除以均方根，计算量减半
- `rms_norm_weight()`：标准版，有学习权重，用于注意力/FFN 前
- `rms_norm_no_weight()`：无权重版，用于 HC 控制向量
- `head_rms_norm_inplace()`：逐头归一化，用于 Q 投影后
- 用 `double` 累加保证精度，`eps` 防止除零
- RMSNorm 计算量小，通常不是瓶颈

## 动手实验

1. 在 `ds4.c:6701` 阅读 `rms_norm_weight`，手算一个 4 维向量的 RMSNorm 结果
2. 对比 `rms_norm_weight` 和 `rms_norm_no_weight`，确认唯一区别是是否乘 `weight[i]`
3. 思考：为什么 `head_rms_norm_inplace` 叫"inplace"？（提示：看输出参数）

## 下一章预告

第 10 章，RoPE（旋转位置编码）--让模型理解 token 顺序的关键技术。数学稍多，但我们会用图解讲清楚。
