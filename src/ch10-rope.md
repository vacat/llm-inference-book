# 第 10 章 RoPE：旋转位置编码

## 本章导读

Transformer 本身是"无序"的--如果你打乱输入 token 的顺序，注意力计算的结果不会自动改变。为了让模型理解 token 的先后顺序，需要**位置编码**。DeepSeek V4 用的是 RoPE（Rotary Position Embedding，旋转位置编码）。

这章可能是全书数学最多的一章，但别怕--我们会用旋转的直觉来理解它。

读完本章你会理解：
- 为什么需要位置编码
- RoPE 的"旋转"直觉
- ds4 中 RoPE 只作用于"尾部"的设计
- YaRN 长度外推的原理

**前置知识**：第 8-9 章

---

## 10.1 为什么需要位置编码

注意力机制的核心是：**每个 token 看其他所有 token，算相似度，加权求和**。这个过程不考虑顺序。

```
没有位置编码:
  "猫 追 狗" 和 "狗 追 猫" 的注意力计算完全一样
  ↑ 模型无法区分主语和宾语！

有位置编码:
  给每个 token 加上位置信息
  "猫(pos=0) 追(pos=1) 狗(pos=2)" ≠ "狗(pos=0) 追(pos=1) 猫(pos=2)"
```

位置编码就是给每个 token 的向量注入"你在第几个位置"的信息。

---

## 10.2 位置编码的演进

```
1. 绝对位置编码 (原始 Transformer)
   pos_enc(pos) = [sin(pos/1000^(2i/d)), cos(pos/1000^(2i/d))]
   直接加到 Embedding 上: x = embedding(token) + pos_enc(pos)
   问题：位置信息在加法后被后续计算稀释

2. 可学习位置编码 (GPT-2)
   每个位置有一个可学习的向量，直接加到 Embedding 上
   问题：训练时最大长度有限，超过就无法处理

3. ALiBi (偏置注意力)
   在注意力分数上加一个与距离成正比的偏置
   优点：天然支持长上下文外推

4. RoPE (旋转位置编码) ← DeepSeek V4 用的
   不是加到向量上，而是"旋转"向量
   优点：相对位置编码 + 支持外推 + 计算高效
```

---

## 10.3 RoPE 的旋转直觉

RoPE 的核心思想：**用旋转来编码位置**。

### 二维的情况

想象一个二维向量 `[x0, x1]`，把它看作平面上的一个点。RoPE 把它旋转一个角度，角度大小取决于位置：

```
位置 0: 旋转 0°
位置 1: 旋转 θ°
位置 2: 旋转 2θ°
位置 p: 旋转 p×θ°

旋转矩阵:
  [cos(pθ)  -sin(pθ)] [x0]   [x0·cos(pθ) - x1·sin(pθ)]
  [sin(pθ)   cos(pθ)] [x1] = [x0·sin(pθ) + x1·cos(pθ)]
```

### 为什么旋转能编码位置？

关键性质：**两个向量的点积（注意力分数）只取决于它们的相对角度差**。

```
位置 p 的向量旋转了 pθ
位置 q 的向量旋转了 qθ

它们的点积 ∝ cos(pθ - qθ) = cos((p-q)θ)
                         ↑ 只取决于相对位置 p-q
```

这意味着注意力分数自动包含了"两个 token 相距多远"的信息--这就是相对位置编码。

### 扩展到高维

512 维的向量怎么旋转？答案是：**拆成 256 对二维向量，每对用不同的频率旋转**。

```
head_dim = 512 的向量:
  [x0, x1] ← 第 0 对，频率最高（旋转最快）
  [x2, x3] ← 第 1 对
  [x4, x5] ← 第 2 对
  ...
  [x510, x511] ← 第 255 对，频率最低（旋转最慢）

每对的旋转角度:
  θ_i = pos × base^(-2i/n_rot)    (i = 0, 1, ..., n_rot/2-1)

  base = 10000 (默认)
  i 越大 -> 频率越低 -> 旋转越慢
```

不同频率捕捉不同粒度的位置信息：高频对捕捉近距离关系，低频对捕捉远距离关系。

---

## 10.4 源码解读：rope_tail_ext_inplace

ds4 的 RoPE 实现在 `rope_tail_ext_inplace()`（`ds4.c:10166`）。"tail"是因为它只旋转向量的尾部：

```c
static void rope_tail_ext_inplace(
        float *x,               // 输入/输出向量
        uint32_t n_head,        // 头数 (64)
        uint32_t head_dim,      // 每头维度 (512)
        uint32_t n_rot,         // 旋转维度 (64) ← 只旋转前 64 维！
        uint32_t pos,           // 当前位置
        uint64_t n_ctx_orig,    // 原始训练长度
        float freq_base,        // 频率基数 (10000)
        float freq_scale,       // 频率缩放
        float ext_factor,       // 外推因子
        float attn_factor,      // 注意力缩放
        float beta_fast,        // YaRN 快速边界
        float beta_slow,        // YaRN 慢速边界
        bool inverse) {         // 是否反向旋转

    const uint32_t n_nope = head_dim - n_rot;  // 不旋转的维度 (512-64=448)
    const float theta_scale = powf(freq_base, -2.0f / (float)n_rot);
    const float sin_sign = inverse ? -1.0f : 1.0f;
    // ...
```

### 关键设计：只旋转尾部

```c
const uint32_t n_nope = head_dim - n_rot;  // 448
```

`n_rot = 64` 意味着每个 512 维的头中，只有最后 64 维被 RoPE 旋转，前 448 维不旋转。

```
一个 head_dim=512 的头:
  [维度 0-447: 不旋转] [维度 448-511: 旋转 64 维]
   ↑ n_nope = 448        ↑ n_rot = 64
```

> **为什么只旋转一部分？** 这是 MLA（多头潜在注意力）的设计。不旋转的维度用于存储"内容"信息（不受位置影响），旋转的维度用于编码"位置"信息。这种分离让模型在压缩 KV 缓存时更高效--压缩的是内容部分，位置部分保持原始形式。

### 旋转循环

```c
for (uint32_t h = 0; h < n_head; h++) {
    float *tail = x + (uint64_t)h * head_dim + n_nope;  // 指向第 h 个头的尾部
    float theta_extrap = (float)pos;                     // 初始角度 = pos

    for (uint32_t i = 0; i < n_rot; i += 2) {
        // YaRN 长度外推的混合（如果 ext_factor != 0）
        const float theta_interp = freq_scale * theta_extrap;
        float theta = theta_interp;
        float mscale = attn_factor;
        if (ext_factor != 0.0f) {
            // ... YaRN 混合逻辑 ...
        }

        // 旋转！这就是 2D 旋转矩阵
        const float c = cosf(theta) * mscale;
        const float s = sin_sign * sinf(theta) * mscale;
        const float x0 = tail[i + 0];
        const float x1 = tail[i + 1];
        tail[i + 0] = x0 * c - x1 * s;   // 新 x0 = x0·cos - x1·sin
        tail[i + 1] = x0 * s + x1 * c;   // 新 x1 = x0·sin + x1·cos

        theta_extrap *= theta_scale;      // 更新角度（频率递减）
    }
}
```

### 逐行理解旋转

```c
const float c = cosf(theta) * mscale;
const float s = sin_sign * sinf(theta) * mscale;
const float x0 = tail[i + 0];
const float x1 = tail[i + 1];
tail[i + 0] = x0 * c - x1 * s;
tail[i + 1] = x0 * s + x1 * c;
```

这正好是二维旋转矩阵：

```
[x0']   [cos(θ)  -sin(θ)] [x0]
[x1'] = [sin(θ)   cos(θ)] [x1]
```

`sin_sign` 控制旋转方向：正向旋转用于 Q/K，反向旋转（`inverse=true`）用于注意力输出的"反旋转"。

### 频率递减

```c
theta_extrap *= theta_scale;  // theta_scale = base^(-2/n_rot)
```

每对维度的旋转角度递减：

```
第 0 对: θ = pos × base^0     = pos          （频率最高）
第 1 对: θ = pos × base^(-2/64) = pos × 0.933
第 2 对: θ = pos × base^(-4/64) = pos × 0.871
...
第 31 对: θ = pos × base^(-62/64) ≈ pos × 0.108  （频率最低）
```

高频对旋转快，对位置变化敏感；低频对旋转慢，对位置变化不敏感。这样不同维度捕捉不同粒度的位置信息。

---

## 10.5 YaRN 长度外推

当推理时的上下文长度超过训练时的长度（`n_ctx_orig`），RoPE 的角度会超出训练范围，导致质量下降。**YaRN** 是一种长度外推技术，让模型在更长上下文中仍能工作。

### 原理

```
训练长度: n_ctx_orig = 4096
推理长度: n_ctx = 32768 (8 倍)

不外推:
  θ = pos × base^(-2i/n_rot)
  pos 可以到 32768，超出训练范围 -> 质量下降

YaRN 外推:
  1. 对高频维度 (i 小): 纯外推，直接用 pos
     -> 高频对位置变化不敏感，外推安全
  2. 对低频维度 (i 大): 插值，把 pos 缩放到训练范围
     -> θ = pos / 8 × base^(-2i/n_rot)
     -> 等效于在训练长度内
  3. 中间维度: 外推和插值的混合
```

代码中的混合逻辑：

```c
if (ext_factor != 0.0f) {
    const float ramp_mix = rope_yarn_ramp(corr_dims[0], corr_dims[1], (int)i)
                         * ext_factor;
    theta = theta_interp * (1.0f - ramp_mix) + theta_extrap * ramp_mix;
    //      ↑ 插值（缩放到训练范围）   ↑ 外推（直接用 pos）
    mscale *= 1.0f + 0.1f * logf(1.0f / freq_scale);
}
```

`rope_yarn_ramp()`（`ds4.c:10147`）计算一个 0 到 1 的渐变值：高频维度（i 小）接近 0（用插值），低频维度（i 大）接近 1（用外推）。

> **直觉**：高频维度旋转快，位置稍微变化角度就差很多，插值会丢失信息，所以用外推。低频维度旋转慢，外推时角度变化太小没有区分度，所以用插值压缩到训练范围。

---

## 10.6 反向旋转（inverse）

`inverse=true` 时 `sin_sign = -1.0f`，旋转方向反转。这用于**注意力输出投影**之前：把旋转过的向量"转回来"。

```
Q 和 K 都做了 RoPE 旋转
  -> 注意力分数 = Q·K 自动包含相对位置
  -> 但输出向量也带了旋转信息
  -> 在输出投影前反向旋转，去除位置信息
  -> 让后续层拿到"干净"的内容向量
```

这是 DeepSeek V4 / MLA 架构特有的设计。

---

## 10.7 不同层的不同 RoPE 频率

`layer_rope_freq_base()`（`ds4.c:10217`）根据层是否压缩返回不同的频率基数：

```c
static float layer_rope_freq_base(uint32_t il) {
    return ds4_layer_compress_ratio(il) != 0 && DS4_COMPRESS_ROPE_FREQ_BASE > 0.0f
        ? DS4_COMPRESS_ROPE_FREQ_BASE
        : DS4_ROPE_FREQ_BASE;
}
```

压缩层（有 KV 压缩的层）用不同的 RoPE 频率，因为压缩后的 KV 表示需要不同的位置编码策略。

---

## 本章小结

- RoPE 用旋转编码位置：每对维度旋转不同频率的角度
- 注意力分数 ∝ cos(相对角度)，自动包含相对位置信息
- ds4 只旋转每个头的尾部 64 维（`n_rot=64`），前 448 维不旋转
- 旋转矩阵：`x0' = x0·cos - x1·sin`，`x1' = x0·sin + x1·cos`
- YaRN 长度外推：高频维度外推，低频维度插值，中间混合
- `inverse` 模式反向旋转，用于注意力输出投影前
- 不同层可能用不同的 RoPE 频率基数

## 动手实验

1. 在 `ds4.c:10166` 阅读 `rope_tail_ext_inplace`，确认旋转矩阵的实现
2. 手算：pos=2, theta_scale=0.9, 第 0 对维度 [1.0, 0.0] 旋转后的结果
3. 思考：为什么高频维度用外推而低频维度用插值？（提示：想想角度变化率）

## 下一章预告

第 11 章，Attention（注意力机制）--Transformer 的灵魂。我们会拆解 Q/K/V 投影、点积注意力、softmax、加权求和的完整过程。
