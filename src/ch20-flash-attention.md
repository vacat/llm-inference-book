# 第 20 章 Flash Attention：重新设计注意力

## 本章导读

标准注意力的内存读写量随序列长度二次增长。Flash Attention 通过重新组织计算顺序，把内存读写降到线性，同时不改变数学结果。这章我们讲原理和 ds4 的实现。

读完本章你会理解：
- 标准注意力的内存瓶颈
- Flash Attention 的分块计算思想
- 在线 Softmax 的数学技巧
- ds4 中 Flash Attention 的使用

**前置知识**：第 11、19 章

---

## 20.1 标准注意力的问题

标准注意力（第 11 章）的计算：

```
1. S = Q × K^T          (得分矩阵, n×n)
2. P = softmax(S)        (概率矩阵, n×n)
3. O = P × V             (输出矩阵, n×d)

n = 序列长度, d = 头维度
```

### 内存问题

```
S 矩阵大小: n × n × 4 字节
  n=4096:  4096² × 4 = 64 MB
  n=32768: 32768² × 4 = 4 GB  ← 写入 GPU 全局内存！
  n=131072: 131072² × 4 = 64 GB ← 不可能存下

每步都要写 GPU 全局内存再读回来:
  写 S -> 读 S -> 写 P -> 读 P -> 写 O
  内存读写量: O(n²)
```

标准注意力把中间结果（S、P 矩阵）写回全局内存，导致：
1. 内存占用 O(n²)
2. 读写带宽消耗 O(n²)

---

## 20.2 Flash Attention 的核心思想

Flash Attention 的关键洞察：**中间矩阵 S 和 P 不需要写回全局内存，可以在寄存器/共享内存中完成全部计算**。

### 分块计算

```
标准注意力: 一次性算整个 n×n 矩阵
  S = Q × K^T  (n×n, 写回内存)
  P = softmax(S) (n×n, 写回内存)
  O = P × V (n×d, 写回内存)

Flash Attention: 分块算
  for each block of Q (大小 B):
    for each block of K, V (大小 B):
      算 S_block = Q_block × K_block^T  (B×B, 在寄存器里)
      算 P_block = softmax(S_block)      (B×B, 在寄存器里)
      累加 O_block += P_block × V_block  (B×d, 在寄存器里)
    写回 O_block (只写一次输出)
```

```
                    Q 分块
  ┌───┬───┬───┬───┐
  │Q₀ │Q₁ │Q₂ │Q₃ │
  └───┴───┴───┴───┘
  
  对每个 Q 块:
    遍历所有 K/V 块, 累加输出
    ┌───┬───┬───┬───┐
    │K₀V₀│K₁V₁│K₂V₂│K₃V₃│
    └───┴───┴───┴───┘
    每次只在寄存器里放 B×B 的小矩阵
```

### 内存改善

```
标准: O(n²) 全局内存
Flash: O(B²) 寄存器/共享内存 (B 通常 64-128)

  n=32768:
    标准: 4 GB 全局内存
    Flash: 128² × 4 = 64 KB 共享内存 (完全可以)
```

---

## 20.3 在线 Softmax 的数学技巧

分块计算有个问题：**Softmax 需要看到所有分数才能归一化**。分块时只看到一部分分数，怎么算正确的 Softmax？

### 在线 Softmax 算法

```
传统 Softmax (需要全部分数):
  max_val = max(all_scores)
  exp_scores = [exp(s - max_val) for s in all_scores]
  sum_exp = sum(exp_scores)
  probs = [e / sum_exp for e in exp_scores]

在线 Softmax (逐块更新):
  维护两个累积量:
    m = 当前最大值 (初始 -inf)
    l = 当前指数和 (初始 0)

  处理一个新块 scores_block:
    m_new = max(m, max(scores_block))
    l = l * exp(m - m_new) + sum(exp(scores_block - m_new))
    m = m_new

  最终: probs = exp(scores - m) / l
```

### 关键数学

```
已有: m_old, l_old (基于前面所有块)
新块: scores_new, m_new = max(m_old, max(scores_new))

修正旧和: l_old_new = l_old × exp(m_old - m_new)
  ↑ 因为最大值变了, 旧的指数和需要重新缩放

新和: l_new = l_old_new + Σ exp(scores_new - m_new)
```

> **直觉**：当你看到更大的分数时，之前的指数和需要"缩小"（因为分母变大了）。在线 Softmax 让你不需要一次性看到所有分数就能逐步计算正确的归一化。

---

## 20.4 ds4 中的 Flash Attention

ds4 的 Flash Attention 实现在 `metal/flash_attn.metal`（51KB，最大的 Metal 文件之一）。

### 使用场景

```
Decode (单 token):
  Q 只有 1 行, 不需要 Flash Attention
  直接遍历 KV 缓存算点积 (第 11 章的实现)

Prefill (多 token):
  Q 有 N 行, 需要算 N × n_kv 的注意力矩阵
  -> 用 Flash Attention 避免巨大的中间矩阵
```

### Metal 着色器

`flash_attn.metal` 开头定义了多种 Flash Attention 变体（`metal/flash_attn.metal:1-10`）：

```metal
#define FC_FLASH_ATTN_EXT_PAD 100
#define FC_FLASH_ATTN_EXT_BLK 200
#define FC_FLASH_ATTN_EXT 300
#define FC_FLASH_ATTN_EXT_VEC 400
#define FC_FLASH_ATTN_EXT_VEC_REDUCE 500
```

不同的编号对应不同的分块策略，针对不同的 Q/KV 维度组合优化。

### 参数

```metal
#define OP_FLASH_ATTN_EXT_NQPSG 8    // Q 分组并行度
#define OP_FLASH_ATTN_EXT_NCPSG 64   // KV 分组并行度
```

这些参数控制 GPU 线程如何分块处理 Q 和 KV，影响共享内存使用和并行度。

---

## 20.5 Flash Attention 的变体

ds4 根据场景使用不同的注意力实现：

```
1. 标准 Attention (CPU 参考, 第 11 章):
   layer_attention_rows_one()
   -> 适合: decode (1 个 token), 简单直接

2. Flash Attention (Metal, prefill):
   flash_attn.metal 中的 kernel
   -> 适合: prefill (多 token), 避免 O(n²) 内存

3. 混合 Attention (有压缩缓存的层):
   layer_attention_mixed_one_decode_scratch()
   -> SWA 原始缓存 + 压缩缓存
   -> decode 时扫描两部分 KV
```

---

## 20.6 性能影响

```
标准 Attention (n=32768):
  中间矩阵: 4 GB
  读写量: ~12 GB (写S + 读S写P + 读P写O)
  时间: ~50ms (内存带宽限制)

Flash Attention (n=32768):
  中间矩阵: 0 (在寄存器里)
  读写量: ~0.5 GB (只读写输入输出)
  时间: ~5ms (10 倍加速)
```

### 对长上下文的意义

```
没有 Flash Attention:
  32K 上下文: 注意力占 50ms/token
  128K 上下文: 注意力占 800ms/token (不可用)

有 Flash Attention:
  32K 上下文: 注意力占 5ms/token
  128K 上下文: 注意力占 20ms/token (可用)
```

> **配合 SWA + 压缩缓存**：Flash Attention + 滑动窗口(128) + 压缩缓存(512) 让 ds4 即使在 128K 上下文下也能保持合理的注意力计算量。Flash Attention 处理窗口内的 128 个 token，压缩缓存的 512 个 token 也用类似方式高效计算。

---

## 本章小结

- 标准注意力写回 n×n 中间矩阵，内存 O(n²)
- Flash Attention 分块计算，中间结果留在寄存器，内存 O(B²)
- 在线 Softmax 逐块更新最大值和指数和，不需要一次性看到所有分数
- ds4 在 prefill 路径用 Flash Attention，decode 路径用直接遍历
- `metal/flash_attn.metal` 有多种分块策略，针对不同维度优化
- Flash Attention 对长上下文至关重要：10 倍以上加速

## 动手实验

1. 查看 `metal/flash_attn.metal` 文件大小（51KB），理解为什么它是最大的 Metal 文件
2. 回顾第 11 章的 `layer_attention_rows_one`，思考为什么 decode 不需要 Flash Attention
3. 思考：在线 Softmax 中，为什么 `l_old` 需要乘以 `exp(m_old - m_new)`？

## 下一章预告

第 21 章，SSD 流式加载--内存不够时怎么跑 86GB 的模型？
