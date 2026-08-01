# 第 19 章 GPU 内核优化基础

## 本章导读

前面我们读的都是 CPU 参考实现--逻辑清晰但速度慢。真正跑起来用的是 GPU 内核（Metal/CUDA）。这章我们讲 GPU 优化的基本思路，以及 ds4 如何把 CPU 逻辑映射到 GPU。

读完本章你会理解：
- CPU 和 GPU 的根本区别
- Metal 着色器（shader）的基本结构
- 算子融合为什么能加速
- GPU 图（graph）执行模式

**前置知识**：第 14、18 章

---

## 19.1 CPU vs GPU

```
CPU: 少量核心 (8-16), 每个核心很强
  适合: 串行逻辑、复杂控制流、低延迟
  矩阵乘法: 逐行计算, 一次一行

GPU: 大量核心 (数千-数万), 每个核心较弱
  适合: 大规模并行、简单计算、高吞吐
  矩阵乘法: 所有行同时计算
```

```
CPU matvec (4096 -> 2048):
  for row in [0..2047]:        ← 串行
    for col in [0..4095]:
      out[row] += W[row][col] * x[col]
  时间: ~5ms

GPU matvec (4096 -> 2048):
  2048 个线程同时执行 ← 并行
  每个线程算一行
  时间: ~0.1ms (50 倍加速)
```

---

## 19.2 Metal 着色器基础

Metal 是 Apple 的 GPU 编程框架。ds4 的 GPU 内核在 `metal/*.metal` 文件中。

### Metal 着色器的结构

```metal
// metal/dense.metal 中的简化示例

// 每个线程执行一次这个函数
kernel void matvec_q8_0(
    device const float *x,        // 输入向量
    device const uint8_t *w,      // 量化权重
    device float *out,            // 输出
    uint row [[thread_position_in_grid]],  // 当前线程编号 = 行号
    uint blocks [[buffer(4)]])    // 块数
{
    float acc = 0.0;
    for (uint b = 0; b < blocks; b++) {
        // 读取缩放因子和量化值
        float scale = f16_to_f32(w[row * blocks * 34 + b * 34]);
        // ... 反量化 + 点积累加 ...
    }
    out[row] = acc;
}
```

### 线程组织

```
GPU 执行模型:
  Grid (网格) = 多个 Thread Group
  Thread Group = 多个 Thread

  matvec 2048 行:
    Grid: 2048 个线程
    每个 Thread Group: 64 个线程
    Grid 大小: 32 个 Thread Group

  GPU 同时调度所有 Thread Group
  每个 Thread 独立计算一行
```

---

## 19.3 算子融合

算子融合是 GPU 优化最重要的技术：**把多个小算子合成一个大算子**。

### 不融合的问题

```
不融合 (CPU 参考实现):
  1. matvec(attn_q_a, x) -> qr       ← 读 x, 写 qr
  2. rms_norm(qr, weight) -> qr_norm  ← 读 qr, 写 qr_norm
  3. matvec(attn_q_b, qr_norm) -> q   ← 读 qr_norm, 写 q
  4. head_rms_norm(q) -> q            ← 读 q, 写 q

  每步都读写全局内存, 4 次内存往返
```

### 融合后

```
融合 (GPU 内核):
  一个 kernel 同时做 matvec + rms_norm + matvec + head_rms_norm
  中间结果存在 GPU 寄存器/共享内存, 不写回全局内存

  只读写 1 次全局内存 (输入 x, 输出 q)
  内存往返从 4 次减少到 1 次
```

### ds4 中的融合

在 `ds4_metal.m` 中，Metal 图执行路径把多个操作融合成一个着色器调用：

```
CPU 参考实现 (第 14 章):
  hc_pre -> rms_norm -> q_proj -> kv_proj -> rope -> kv_store
  -> attention -> inv_rope -> out_proj -> hc_post -> moe

Metal 融合实现:
  metal_graph_eval_token_raw_swa()
  -> 几个大的融合 kernel 完成上述所有步骤
  -> 中间数据尽量留在 GPU 显存
```

> **融合的代价**：融合后的代码可读性差，调试困难。所以 ds4 保留 CPU 参考实现用于验证正确性，GPU 融合实现用于追求性能。

---

## 19.4 图执行模式

ds4 的 Metal 路径不是逐个算子调用 GPU，而是构建一个**执行图（graph）**，一次性提交所有操作。

### 图执行 vs 逐个调用

```
逐个调用 (慢):
  调用 kernel 1 -> 等 GPU 完成 -> 调用 kernel 2 -> 等 -> ...

图执行 (快):
  构建图: [kernel1] -> [kernel2] -> [kernel3] -> ...
  一次性提交给 GPU
  GPU 自行调度, 无需 CPU 等待
```

### metal_graph_eval_token_raw_swa

`metal_graph_eval_token_raw_swa()`（`ds4.c:29334`）是 Metal decode 路径的主函数。它构建并执行一层的完整图：

```
metal_graph_eval_token_raw_swa(graph, model, weights, token, pos, logits)
  │
  ├─ Q 投影 (融合: matvec + rms_norm + matvec + rms_norm)
  ├─ KV 投影 (融合: matvec + rms_norm)
  ├─ RoPE (旋转)
  ├─ KV 存储 (写入缓存)
  ├─ 注意力 (flash attention kernel)
  ├─ 输出投影 (融合: matvec + 分组)
  ├─ HC 后处理
  └─ MoE (融合: route + 6 experts + shared expert)
```

所有这些步骤在一次图提交中完成，GPU 连续执行，CPU 只在开始和结束时参与。

---

## 19.5 Metal 着色器文件

ds4 的 Metal 着色器按功能分文件：

| 文件 | 功能 |
|------|------|
| `dense.metal` | 稠密矩阵乘法（Q/KV/输出投影） |
| `flash_attn.metal` | Flash Attention |
| `moe.metal` | MoE 路由 + 专家计算 |
| `norm.metal` | RMSNorm |
| `dsv4_rope.metal` | RoPE 旋转 |
| `dsv4_hc.metal` | HC 预处理/后处理 |
| `dsv4_kv.metal` | KV 缓存操作 |
| `dsv4_misc.metal` | 杂项操作 |
| `softmax.metal` | Softmax |
| `glu.metal` | SwiGLU 激活 |

每个文件包含多个 kernel 函数，在引擎启动时编译（`ds4_metal.m` 中的编译逻辑）。

---

## 19.6 GPU 内存的零拷贝

ds4 利用 Metal 的 `MTLBuffer` 实现零拷贝：

```
传统方式:
  模型文件 -> CPU 内存 -> 复制到 GPU 显存
  ↑ 86GB 的复制, 非常慢

零拷贝方式 (ds4):
  模型文件 mmap -> MAP_SHARED
  Metal 把 mmap 区域包装成 MTLBuffer (无复制)
  GPU 直接从 mmap 区域读取权重
  ↑ 操作系统按需把磁盘数据加载到 GPU 可见的内存
```

这就是为什么 `model_open` 中 Metal 路径用 `MAP_SHARED`（第 6 章）--只有 shared mapping 才能被 Metal 包装为 GPU buffer。

---

## 19.7 性能对比

```
CPU decode (Apple M5 Max):
  ~215 ms/token  → ~5 tok/s

Metal decode (Apple M5 Max):
  ~15 ms/token   → ~65 tok/s
  加速比: ~14x

Metal prefill (Apple M5 Max):
  ~0.5 ms/token  → ~2000 tok/s
```

### 为什么 GPU decode 仍有瓶颈？

```
Decode 每个 token:
  读取 ~15GB 激活权重 (从 mmap)
  计算 ~74 GFLOP

Apple M5 Max:
  内存带宽: ~400 GB/s
  理论时间: 15GB / 400GB/s = 37.5ms (纯读取)
  实际: ~15ms (部分权重在 GPU 缓存中)

  算术强度: 74G / 15G = ~5 FLOP/byte
  GPU 需要 >50 FLOP/byte 才能打满计算
  -> 仍然是内存带宽瓶颈
```

---

## 本章小结

- GPU 用数千个线程并行计算，适合大规模矩阵运算
- Metal 着色器是 GPU 程序，每个线程处理一个数据单元
- 算子融合把多个小 kernel 合成大 kernel，减少内存往返
- 图执行模式一次性提交所有操作，减少 CPU-GPU 同步开销
- 零拷贝：Metal 直接从 mmap 区域读取权重，无需复制
- CPU 参考实现用于验证正确性，GPU 融合实现用于追求性能
- Decode 仍有内存带宽瓶颈，这是推测解码的动机

## 动手实验

1. 查看 `metal/` 目录，对比每个 `.metal` 文件对应第 14 章的哪个阶段
2. 在 `ds4.c:29334` 查看 `metal_graph_eval_token_raw_swa` 的调用，理解图执行
3. 运行 `DS4_DECODE_PROFILE_DETAIL=1 ./ds4 -p "test" -n 5`，对比 CPU 和 Metal 的耗时

## 下一章预告

第 20 章，Flash Attention--重新设计注意力计算，减少内存读写，让长上下文不再昂贵。
