# 第 8 章 Embedding：词嵌入查表

## 本章导读

推理的第一步是把 token id 变成向量。这个过程叫 Embedding（嵌入），原理极其简单--查表。但这张"表"是模型最大的张量之一，理解它的存储方式对后续学习很重要。

**前置知识**：第 4-5 章

---

## 8.1 什么是 Embedding

Token id 是一个整数（如 9906），模型无法直接对整数做矩阵运算。Embedding 就是把整数映射成向量：

```
token id: 9906
    │
    ▼
查表 (Embedding Table)
    │
    ▼
向量: [0.12, -0.34, 0.56, ..., 0.78]  (4096 维)
```

这张表的大小是 `词表大小 × 嵌入维度 = 129280 × 4096`。每个 token 对应表中的一行（4096 维向量）。

> **直觉理解**：可以把 Embedding 向量想成 token 的"身份证号"--一个独特的数字指纹。语义相近的 token（如"猫"和"狗"），它们的 Embedding 向量在空间中也比较近。这个空间分布是训练时学出来的。

---

## 8.2 源码解读：embed_token_f16

ds4 的 Embedding 实现在 `embed_token_f16()`（`ds4.c:6633`）。函数很短：

```c
static void embed_token_f16(const ds4_model *m, const ds4_weights *w,
                            int token, float *out) {
    ds4_tensor *te = w->token_embd;
    // 验证类型和维度
    if (te->type != DS4_TENSOR_F16 || te->ndim != 2) {
        ds4_die("expected a 2D F16 token embedding tensor");
    }
    if (token < 0 || (uint64_t)token >= te->dim[1]) {
        ds4_die("token id is outside the embedding table");
    }

    // 计算行地址
    const uint16_t *base = tensor_data(m, te);       // 表的起始地址
    const uint64_t stride = te->dim[0];               // 每行的元素数 (4096)
    const uint16_t *row = base + (uint64_t)token * stride;  // 第 token 行

    // 把 F16 转成 F32
    for (uint64_t i = 0; i < stride; i++) {
        out[i] = f16_to_f32(row[i]);
    }
}
```

### 逐步解读

**第 1 步：验证**
```c
if (te->type != DS4_TENSOR_F16 || te->ndim != 2)
```
确保嵌入表是 2D 的 F16 张量。如果模型文件不对，立即报错。

**第 2 步：定位行**
```c
const uint16_t *base = tensor_data(m, te);  // mmap 中嵌入表的起始地址
const uint16_t *row = base + token * stride;
```
Embedding 表在内存中是一个二维数组。`token * stride` 就是偏移量--跳过前面 `token` 行，每行 `stride`（4096）个元素。

```
Embedding 表布局 (F16):
base -> [token 0 的 4096 个 F16 值]
        [token 1 的 4096 个 F16 值]
        ...
        [token 9906 的 4096 个 F16 值]  <- row 指向这里
        ...
        [token 129279 的 4096 个 F16 值]
```

**第 3 步：F16 -> F32 转换**
```c
for (uint64_t i = 0; i < stride; i++) {
    out[i] = f16_to_f32(row[i]);
}
```
模型权重用 F16（半精度浮点）存储以节省空间，但计算时用 F32（单精度浮点）以保证精度。`f16_to_f32()` 把每个 16 位浮点数转成 32 位。

### F16 vs F32

```
F32 (32 位浮点): 1 符号位 + 8 指数位 + 23 尾数位
F16 (16 位浮点): 1 符号位 + 5 指数位 + 10 尾数位

F16 的范围更小（±65504），精度更低（约 3 位有效十进制数）
但占一半空间！

Embedding 表用 F16 存储:
  129280 × 4096 × 2 字节 = 1.06 GB
如果用 F32:
  129280 × 4096 × 4 字节 = 2.12 GB
```

---

## 8.3 Q8_0 量化版本的 Embedding

有些模型的 Embedding 表用 Q8_0 量化存储。`embed_token_q8_0()`（`ds4.c:6651`）处理这种情况：

```c
static void embed_token_q8_0(const ds4_model *m, const ds4_weights *w,
                             int token, float *out) {
    const uint64_t n = te->dim[0];                    // 4096
    const uint64_t blocks = (n + 31) / 32;            // ⌈n/32⌉ 向上取整 = 128 个块（每块 32 元素）
    const uint8_t *row = (const uint8_t *)tensor_data(m, te) +
                         token * blocks * 34;          // 每块 34 字节

    for (uint64_t b = 0; b < blocks; b++) {
        // 每块开头 2 字节是缩放因子 (F16)
        uint16_t scale_bits;
        memcpy(&scale_bits, row + b * 34, sizeof(scale_bits));
        const float scale = f16_to_f32(scale_bits);

        // 接下来 32 字节是量化值 (int8)
        const int8_t *qs = (const int8_t *)(row + b * 34 + 2);
        for (uint64_t i = 0; i < 32; i++) {
            out[b * 32 + i] = scale * (float)qs[i];  // 反量化
        }
    }
}
```

**为什么是 `(n + 31) / 32` 而不是 `n / 32`？**

这是 C/C++ 里把"整数除法"变成"向上取整"（ceiling division）的经典写法：

| n | `n / 32`（整除，向下） | `(n + 31) / 32`（向上） | 应该的块数 |
|---|---|---|---|
| 4096 | 128 | 128 | 128 |
| 4097 | **128** ❌ 少一块 | 129 ✓ | 129 |
| 33   | **1** ❌ 少一块 | 2 ✓ | 2 |
| 32   | 1 | 1 | 1 |

- 整除 `/ 32` 默认**向下取整**（floor），会丢掉余数。如果 `n=4097`，除完得 128，但实际需要 129 块（最后一块只有 1 个元素）。
- 加 `(32 - 1) = 31` 再除，等价于"把不足一块的部分也凑成一整块"——这就是 `⌈n / 32⌉`。
- 通用公式：**`⌈a / b⌉ = (a + b - 1) / b`**（仅限非负整数）。

> 本例 `n=4096` 恰好整除 32，两种写法结果相同；但 `n` 是从 GGUF 张量元数据读出来的，运行时不一定都是 32 的倍数，所以代码必须用向上取整的写法兜底，否则最后一组元素会丢。


Q8_0 的结构：每 32 个元素为一组，存一个 F16 缩放因子 + 32 个 int8 值。反量化时：`实际值 = 缩放因子 × int8值`。

```
Q8_0 块结构 (34 字节):
  [scale (F16, 2字节)] [q0 q1 q2 ... q31 (int8, 32字节)]

反量化: value[i] = scale * q[i]
```

> **DeepSeek V4 Flash 的 Embedding 用的是什么？** 根据文件名 `...OutQ8...`，输出头是 Q8。Embedding 表的具体类型取决于 GGUF 文件。`embed_token_any()`（`ds4.c:6677`）会根据张量类型自动选择 F16 或 Q8_0 路径。

---

## 8.4 Embedding 的"统一分发"函数

`embed_token_any()`（`ds4.c:6677`）是个分发器：

```c
static void embed_token_any(const ds4_model *m, const ds4_weights *w,
                            int token, float *out) {
    switch (w->token_embd->type) {
    case DS4_TENSOR_F16:
        embed_token_f16(m, w, token, out);
        break;
    case DS4_TENSOR_Q8_0:
        embed_token_q8_0(m, w, token, out);
        break;
    default:
        ds4_die("unsupported token embedding tensor type");
    }
}
```

根据 Embedding 张量的量化类型，选择对应的解码函数。这是 C 语言中常见的"类型分发"模式。

---

## 8.5 Embedding 的性能特点

Embedding 是所有算子中最简单的，但也有值得注意的性能点：

### 优点：几乎没有计算

Embedding 就是查表 + 类型转换，没有乘加运算。4096 次内存读取 + F16->F32 转换，对现代 CPU/GPU 来说微不足道。

### 注意：内存访问模式

```
Prefill (处理 N 个 token):
  访问 Embedding 表的 N 行
  -> 如果 N 个 token 相近，可能命中 cache
  -> 通常不是瓶颈

Decode (处理 1 个 token):
  访问 Embedding 表的 1 行
  -> 4096 × 2 = 8KB 的数据
  -> 完全在 L1 cache 里，极快
```

### Embedding 表的内存占用

```
F16 存储: 129280 × 4096 × 2 = 1.06 GB
Q8_0 存储: 129280 × 4096 × 8.5/8 ≈ 0.56 GB
```

Embedding 表是模型中较大的张量之一，但它只在推理开始时访问一次，之后不再使用。

---

## 8.6 输出头的"逆 Embedding"

模型的最后一层（output head）做的事情和 Embedding 相反：把 4096 维向量映射回 129280 维的 logits。这本质上是一个矩阵乘法：

```
Embedding (输入端):   token id -> 查表 -> 4096 维向量
Output head (输出端): 4096 维向量 -> 矩阵乘 -> 129280 维 logits
```

有趣的是，有些模型采用**权重绑定（weight tying）**：输出头的权重和 Embedding 表共享同一份数据。DeepSeek V4 Flash 中，output 权重是独立的（文件名中的 `OutQ8` 表示输出头用 Q8 量化）。

---

## 本章小结

- Embedding = 查表：token id 对应表中的一行向量
- `embed_token_f16()` 从 F16 表中取一行，转成 F32
- `embed_token_q8_0()` 处理 Q8_0 量化表：每 32 元素一组，反量化为 F32
- `embed_token_any()` 根据类型自动分发
- Embedding 几乎没有计算，不是性能瓶颈
- 输出头是"逆 Embedding"：4096 维 -> 129280 维 logits

## 动手实验

1. 在 `ds4.c:6633` 阅读 `embed_token_f16`，确认你理解指针运算 `base + token * stride`
2. 在 `ds4.c:6651` 阅读 `embed_token_q8_0`，理解 Q8_0 块结构（34 字节 = 2 + 32）
3. 思考：如果词表有 129280 个 token，每个 4096 维 F16，Embedding 表占多少内存？

## 下一章预告

第 9 章，我们看 RMSNorm--一种比传统 BatchNorm 更简单、更适合推理的归一化方法。
