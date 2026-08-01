# 第 12 章 FFN 与 SwiGLU：前馈网络

## 本章导读

每一层 Transformer 有两半：注意力（第 11 章）和 FFN（前馈网络，本章）。注意力负责"看上下文"，FFN 负责"做变换"。这章我们拆解 FFN 的结构，特别是 DeepSeek V4 用的 SwiGLU 激活函数。

读完本章你会理解：
- FFN 在 Transformer 中的作用
- SwiGLU 的门控机制
- 共享专家就是一个标准 FFN
- clamp 防止数值爆炸

**前置知识**：第 8-11 章

---

## 12.1 FFN 的作用

如果只有注意力，模型只能"在已有信息之间搬来搬去"。FFN 给了模型**非线性变换**的能力--把信息"加工"成新的表示。

```
注意力: "从上下文中找到相关信息"  -> 信息搬运
FFN:    "对信息做非线性变换"     -> 信息加工
```

标准 FFN 的结构：

```
输入 x (4096 维)
  │
  ├─ gate = W_gate × x    (4096 -> 2048 维)
  ├─ up   = W_up × x      (4096 -> 2048 维)
  │
  ├─ 激活: mid = activation(gate) × up    (逐元素)
  │
  └─ 输出 = W_down × mid  (2048 -> 4096 维)
```

两个线性变换把维度先升后降（4096 -> 2048 -> 4096），中间加非线性激活。这是"瓶颈"结构，让信息在更高维空间做变换后再投影回来。

---

## 12.2 SwiGLU 激活

DeepSeek V4 用的是 **SwiGLU**（Swish-Gated Linear Unit），它是 GLU（门控线性单元）的变体。

### 激活函数的演进

```
ReLU:    f(x) = max(0, x)                    ← 最简单，但有"死神经元"问题
GELU:    f(x) = x × Φ(x)                     ← 平滑版 ReLU
Swish:   f(x) = x × sigmoid(x)               ← 又叫 SiLU
GLU:     f(gate, up) = activation(gate) × up ← 门控机制
SwiGLU:  f(gate, up) = silu(gate) × up       ← Swish + 门控
```

### SwiGLU 的直觉

```
gate = W_gate × x    -> "哪些信息要通过"（门控信号）
up   = W_up × x      -> "通过的信息内容"

output = silu(gate) × up
         ↑               ↑
         门控（0~1之间）   内容
         
silu(gate) = gate × sigmoid(gate)
           ≈ gate × (0~1)
           
当 gate > 0: silu(gate) ≈ gate   (门打开，信息通过)
当 gate < 0: silu(gate) ≈ 0      (门关闭，信息阻断)
```

门控让模型可以**选择性地通过信息**--对有用的特征放大，对无用的抑制。

---

## 12.3 源码解读

### silu 函数

`silu()`（`ds4.c:10484`）就是 Swish 激活：

```c
static float silu(float x) {
    return x * sigmoid_stable(x);
}
```

`sigmoid_stable()`（`ds4.c:10354`）是数值稳定的 sigmoid：

```c
static float sigmoid_stable(float x) {
    if (x >= 0.0f) {
        const float e = expf(-x);
        return 1.0f / (1.0f + e);
    } else {
        const float e = expf(x);
        return e / (1.0f + e);
    }
}
```

> **为什么要分正负？** 当 `x` 很大时，`exp(-x)` 接近 0，`1/(1+0) = 1`，没问题。当 `x` 很负时，`exp(-x)` 会溢出（`exp(1000) = inf`）。所以负数时用 `exp(x) / (1 + exp(x))`，此时 `exp(x)` 接近 0，不会溢出。

### swiglu 函数

`swiglu()`（`ds4.c:10494`）是 SwiGLU 的核心：

```c
static void swiglu(float *out, const float *gate, const float *up,
                   uint64_t n, float clamp) {
    for (uint64_t i = 0; i < n; i++) {
        float g = gate[i];
        float u = up[i];

        // clamp：防止极端值
        if (clamp > 1.0e-6f) {
            if (g > clamp) g = clamp;
            if (u > clamp) u = clamp;
            if (u < -clamp) u = -clamp;
        }

        out[i] = silu(g) * u;   // SwiGLU: silu(gate) × up
    }
}
```

### clamp 的作用

```c
if (clamp > 1.0e-6f) {
    if (g > clamp) g = clamp;      // 限制 gate 上界
    if (u > clamp) u = clamp;      // 限制 up 上界
    if (u < -clamp) u = -clamp;    // 限制 up 下界
}
```

`DS4_SWIGLU_CLAMP_EXP` 是一个很小的正数（如 30.0），防止 `silu(g)` 或 `u` 过大导致 `exp()` 溢出。这是一种防御性编程，正常情况下不会触发，但能防止极端输入导致 NaN。

### 共享专家 FFN

`layer_shared_ffn_one_decode_scratch()`（`ds4.c:10542`）是共享专家的 FFN：

```c
static void layer_shared_ffn_one_decode_scratch(
        float *out, const ds4_model *model,
        const ds4_layer_weights *layer,
        const float *x, ds4_cpu_decode_scratch *scratch) {

    // 第 1 步：gate 和 up 投影（同时算两个矩阵乘法）
    matvec_q8_0_pair_decode_scratch(
        scratch->shared_gate,       // 输出: gate (2048维)
        scratch->shared_up,         // 输出: up (2048维)
        model,
        layer->ffn_gate_shexp,      // 权重: gate 矩阵
        layer->ffn_up_shexp,        // 权重: up 矩阵
        x,                          // 输入 (4096维)
        scratch);

    // 第 2 步：SwiGLU 激活
    swiglu(scratch->shared_mid,     // 输出: mid (2048维)
           scratch->shared_gate,
           scratch->shared_up,
           DS4_N_FF_EXP,            // 2048
           DS4_SWIGLU_CLAMP_EXP);

    // 第 3 步：down 投影
    matvec_q8_0_decode_scratch(
        out,                        // 输出 (4096维)
        model,
        layer->ffn_down_shexp,      // 权重: down 矩阵
        scratch->shared_mid,        // 输入 (2048维)
        scratch);
}
```

> **优化点**：`matvec_q8_0_pair_decode_scratch` 一次算两个矩阵乘法（gate 和 up），共享输入向量 x 的量化。如果分开算，x 要量化两次。合并后只量化一次，省一半预处理时间。

---

## 12.4 FFN 的数据流

```
输入 x (4096 维)
    │
    ├──> matvec(ffn_gate_shexp, x) ──> gate (2048 维)
    │                                  │
    ├──> matvec(ffn_up_shexp, x) ────> up (2048 维)
    │                                  │
    │                    ┌─────────────┘
    │                    ▼
    │              swiglu(gate, up) = silu(gate) × up
    │                    │
    │                    ▼
    │              mid (2048 维)
    │                    │
    │                    ▼
    │              matvec(ffn_down_shexp, mid)
    │                    │
    │                    ▼
    └──────────────> out (4096 维)
```

### 维度变化

```
x:    4096 维  ──gate──> 2048 维
      4096 维  ──up────> 2048 维
                        ↓ swiglu
                    2048 维  ──down──> 4096 维
```

先从 4096 升到 2048（实际上是"降到"更低维做变换），再升回 4096。`n_ff_exp = 2048` 就是中间层的宽度。

---

## 12.5 共享专家 vs 路由专家

DeepSeek V4 每层的 FFN 由两部分组成：

```
FFN 总输出 = 共享专家输出 + Σ(路由专家输出 × 权重)
```

| | 共享专家 | 路由专家 |
|---|---|---|
| 数量 | 1 个 | 256 个，每次选 6 个 |
| 激活方式 | **每次都激活** | 由路由门控选择 |
| 权重 | `ffn_gate_shexp` / `ffn_up_shexp` / `ffn_down_shexp` | `ffn_gate_exps` / `ffn_up_exps` / `ffn_down_exps` |
| 作用 | 处理通用信息 | 处理特定领域信息 |
| 结构 | 标准 SwiGLU FFN | 标准 SwiGLU FFN（结构相同） |

共享专家就是本章讲的这个 FFN。它每个 token 都参与计算，负责"通用知识"。路由专家的结构完全相同，只是每次只选 6 个来算（第 13 章详讲）。

---

## 12.6 性能分析

### 共享专家的计算量

```
gate: 4096 × 2048 = 8.4M 乘加
up:   4096 × 2048 = 8.4M 乘加
down: 2048 × 4096 = 8.4M 乘加
总计: ~25M 乘加 / 每层 / 每 token
43 层: ~1.08G 乘加
```

### 路由专家的计算量

```
6 个专家 × (gate + up + down) = 6 × 25M = 150M 乘加 / 每层
43 层: ~6.45G 乘加
```

路由专家的计算量是共享专家的 6 倍。但如果没有 MoE（用 dense FFN），计算量会是 256 倍。这就是 MoE 的价值：用 6 倍的计算量获得 256 倍的参数容量。

### 量化加速

共享专家和路由专家的权重都用 Q8_0 量化。`matvec_q8_0` 利用 int8 运算，比 F32 快约 4 倍（详见第 18 章）。`matvec_q8_0_pair` 进一步合并两个矩阵乘法，减少输入向量的重复读取。

---

## 本章小结

- FFN = 线性变换 -> 非线性激活 -> 线性变换，负责信息加工
- SwiGLU = `silu(gate) × up`，门控机制让模型选择性通过信息
- `silu(x) = x × sigmoid(x)`，数值稳定的 sigmoid 分正负两种情况
- 共享专家 = 标准 SwiGLU FFN，每次都激活，处理通用信息
- gate 和 up 投影合并计算（`matvec_q8_0_pair`），共享输入量化
- clamp 防止极端值导致 exp 溢出
- 路由专家结构相同，只是每次选 6 个

## 动手实验

1. 在 `ds4.c:10494` 阅读 `swiglu`，手算：gate=2.0, up=3.0, clamp=0 时的输出
2. 在 `ds4.c:10354` 阅读 `sigmoid_stable`，理解为什么分正负两种情况
3. 思考：为什么 gate 和 up 用同一个输入 x，而不是两个不同的输入？

## 下一章预告

第 13 章，MoE（混合专家）路由--256 个专家怎么选 6 个？路由门控怎么打分？hash 路由和学习路由有什么区别？
