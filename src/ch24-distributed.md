# 第 24 章 分布式推理

## 本章导读

当一台机器的内存装不下整个模型时，可以把模型拆到多台机器上。ds4 用**流水线并行**（Pipeline Parallelism）实现分布式推理。这章我们讲它的原理和实现。

读完本章你会理解：
- 流水线并行的原理
- coordinator 和 worker 的分工
- 层切片（layer slice）机制
- 分布式推理的网络开销

**前置知识**：第 3、23 章

---

## 24.1 为什么需要分布式

```
单机限制:
  DeepSeek V4 Flash: 86GB
  Mac Studio 最大内存: 192GB (够)
  MacBook Pro 最大内存: 128GB (紧张)
  普通服务器: 64-128GB (不够)

分布式:
  机器 A (64GB) + 机器 B (64GB) = 128GB 总内存
  -> 模型拆成两半, 每台放一半
  -> 两台机器协作完成推理
```

---

## 24.2 流水线并行

流水线并行把模型的**层**分配到不同机器上：

```
机器 A (层 0-21)          机器 B (层 22-42)
┌──────────────────┐     ┌──────────────────┐
│ Layer 0          │     │ Layer 22         │
│ Layer 1          │     │ Layer 23         │
│ ...              │     │ ...              │
│ Layer 21 ────────┼────>│ Layer 42         │
│                  │ 网络 │                  │
│ Embedding        │     │ Output Head      │
└──────────────────┘     └──────────────────┘

数据流:
  输入 token -> 机器 A 算 Layer 0-21
             -> 中间结果通过网络传给机器 B
             -> 机器 B 算 Layer 22-42
             -> logits 传回机器 A
             -> 采样得到输出 token
```

### 为什么叫"流水线"？

```
Token 1: [A 算 L0-21] -> [B 算 L22-42] -> 输出
Token 2:              [A 算 L0-21] -> [B 算 L22-42] -> 输出
Token 3:                           [A 算 L0-21] -> [B 算 L22-42] -> 输出

像流水线一样, A 和 B 可以交替工作
但 decode 阶段每次只有 1 个 token, 流水线难以填满
```

> **注意**：流水线并行在 decode 时效率不高（每次只 1 个 token，网络延迟占比大）。它主要用于解决内存不够的问题，不是为了加速。

---

## 24.3 角色分工

ds4 的分布式推理有两个角色（`ds4.h:47-50`）：

```c
typedef enum {
    DS4_DISTRIBUTED_NONE = 0,
    DS4_DISTRIBUTED_COORDINATOR,  // 协调者
    DS4_DISTRIBUTED_WORKER,       // 工作者
} ds4_distributed_role;
```

### Coordinator（协调者）

```
职责:
  - 接收用户输入 (prompt)
  - 处理分配给自己的层 (如 Layer 0-21)
  - 把中间结果发给 Worker
  - 接收 Worker 的结果
  - 做最终采样, 输出 token

类比: 流水线的第一站, 也是最终汇总者
```

### Worker（工作者）

```
职责:
  - 接收 Coordinator 发来的中间结果
  - 处理分配给自己的层 (如 Layer 22-42)
  - 把结果发回 Coordinator

类比: 流水线的中间站
```

### 层分配

在 `ds4.h:52-57` 中：

```c
typedef struct {
    uint32_t start;      // 起始层
    uint32_t end;        // 结束层
    bool has_output;     // 是否包含输出头
    bool set;            // 是否已设置
} ds4_distributed_layers;
```

Coordinator 和 Worker 各自知道自己处理哪些层。例如：

```
Coordinator: layers.start=0, layers.end=21, has_output=false
Worker:      layers.start=22, layers.end=42, has_output=true
```

---

## 24.4 层切片执行

分布式推理的核心是 `ds4_session_eval_layer_slice()`（`ds4.h:340-360`）：

```c
int ds4_session_eval_layer_slice(
    ds4_session *s,
    const int *tokens,        // 输入 token
    uint32_t n_tokens,        // token 数
    uint32_t pos0,            // 起始位置
    uint32_t layer_start,     // 起始层
    uint32_t layer_end,       // 结束层
    const float *input_hc,    // 输入隐藏状态 (从前一台机器传来的)
    float *output_hc,         // 输出隐藏状态 (传给下一台机器)
    bool output_logits,       // 是否输出 logits
    float *logits,            // logits 输出
    char *err, size_t errlen);
```

这个函数只执行 `layer_start` 到 `layer_end` 之间的层，接收输入隐藏状态，输出隐藏状态。

### 分布式数据流

```
Coordinator:
  1. embed_token(token) -> hc
  2. ds4_session_eval_layer_slice(hc, 0, 21, hc_out)
  3. 网络发送 hc_out 给 Worker

Worker:
  4. 接收 hc_out
  5. ds4_session_eval_layer_slice(hc_out, 22, 42, hc_final)
  6. ds4_session_eval_output_head_from_hc(hc_final, logits)
  7. 网络发送 logits 给 Coordinator

Coordinator:
  8. 接收 logits
  9. sample(logits) -> 下一个 token
```

---

## 24.5 网络传输

### 传输内容

```
每层之间的隐藏状态:
  hc_dim = 4 × 4096 = 16384 维
  每维 F32: 4 字节
  单次传输: 16384 × 4 = 64 KB

43 层分成 2 段:
  每段只在边界传输一次 (64 KB)
  网络开销很小
```

### 传输方式

```
TCP:
  通用, 跨平台
  延迟: ~1ms (局域网)

如果分更多段 (3+ 台机器):
  每段边界都要传输
  但每段内部的计算可以并行
```

> **对比张量并行**：分布式（流水线并行）传输的是层边界的隐藏状态（64KB），而张量并行（第 25 章）每层都要传输部分和（更大）。所以流水线并行的网络开销更小，但无法像张量并行那样加速单层计算。

---

## 24.6 配置分布式推理

### Coordinator 端

```bash
./ds4-server \
  --distributed coordinator \
  --listen 0.0.0.0:9000 \
  --layers 0-21 \
  --port 8080
```

- `--distributed coordinator`：角色为协调者
- `--listen 0.0.0.0:9000`：监听 Worker 连接
- `--layers 0-21`：处理第 0-21 层
- `--port 8080`：HTTP 服务端口

### Worker 端

```bash
./ds4 \
  --distributed worker \
  --coordinator 192.168.1.100:9000 \
  --layers 22-42 \
  --has-output
```

- `--distributed worker`：角色为工作者
- `--coordinator 192.168.1.100:9000`：连接协调者
- `--layers 22-42`：处理第 22-42 层
- `--has-output`：包含输出头

### 分布式选项

在 `ds4.h:59-70` 中：

```c
typedef struct {
    ds4_distributed_role role;
    ds4_distributed_layers layers;
    const char *listen_host;
    int listen_port;
    const char *coordinator_host;
    int coordinator_port;
    uint32_t prefill_chunk;    // prefill 分块大小
    uint32_t prefill_window;   // prefill 窗口
    uint32_t activation_bits;  // 激活值精度
    bool replay_check;         // 重放检查
    bool debug;
} ds4_distributed_options;
```

---

## 24.7 多机扩展

```
3 台机器:
  机器 A: Layer 0-14 (Coordinator)
  机器 B: Layer 15-28 (Worker)
  机器 C: Layer 29-42 (Worker, has_output)

数据流:
  A (L0-14) -> B (L15-28) -> C (L29-42) -> logits 回传 A

每台机器只需要 1/3 的模型权重
-> 29GB / 台 (vs 86GB 单机)
```

### 限制

```
1. 网络延迟累积:
   每多一台机器, 多一次网络传输 (~1ms)
   3 台机器: 额外 ~3ms/token

2. 串行依赖:
   A 算完才能传给 B, B 算完才能传给 C
   无法像张量并行那样真正并行计算

3. KV 缓存分布:
   每台机器只存自己层的 KV 缓存
   不需要同步, 但 checkpoint 需要协调
```

---

## 24.8 分布式 vs SSD 流式

两种"小内存跑大模型"的策略对比：

```
                    分布式 (流水线)        SSD 流式
─────────────────────────────────────────────────────
需要多台机器?       是                     否
网络依赖            高 (每次传输)          低 (本地 SSD)
延迟                +1ms/段                +10ms/专家未命中
适用场景            有多台机器             单机小内存
部署复杂度          高                     低
```

> **实践建议**：如果只有一台机器，用 SSD 流式。如果有多台机器且网络好（如 10GbE），用分布式。两者也可以组合：每台机器用 SSD 流式 + 多机流水线。

---

## 本章小结

- 流水线并行把模型层分配到多台机器，解决单机内存不够的问题
- Coordinator 负责输入/输出/采样，Worker 负责中间层
- `ds4_session_eval_layer_slice` 执行指定层范围，输入/输出隐藏状态
- 网络传输量小（64KB/层边界），但串行依赖限制加速效果
- 分布式主要用于扩展内存容量，不是为了加速
- 与 SSD 流式互补：分布式适合多机，SSD 流式适合单机

## 动手实验

1. 在 `ds4.h:340-360` 查看 `ds4_session_eval_layer_slice` 的接口
2. 在 `ds4_distributed.h` 查看分布式选项结构
3. 思考：为什么流水线并行在 decode 时效率不高？（提示：串行依赖 + 网络延迟）

## 下一章预告

第 25 章，张量并行--两台机器同时算同一层，用 RDMA 交换中间结果。
