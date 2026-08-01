# 第 25 章 张量并行

## 本章导读

流水线并行（第 24 章）把不同层放到不同机器，但每层仍由一台机器算。张量并行更进一步：**两台机器同时算同一层**，各自算一半，然后交换结果。这章我们讲 ds4 的张量并行实现。

读完本章你会理解：
- 张量并行与流水线并行的区别
- Leader/Worker 锁步协议
- 门控交换（gate exchange）机制
- RDMA 传输的优势

**前置知识**：第 24 章

---

## 25.1 张量并行 vs 流水线并行

```
流水线并行 (第 24 章):
  机器 A: Layer 0-21    机器 B: Layer 22-42
  A 算完 -> 传给 B -> B 算
  串行, 每层只一台机器算

张量并行:
  机器 A: Layer 0-42 (一半权重)    机器 B: Layer 0-42 (另一半权重)
  A 和 B 同时算同一层, 各算一半
  每层结束后交换中间结果
  并行, 每层两台机器同时算
```

### 加速效果

```
单机 decode: 15ms/token

流水线并行 (2 台):
  机器 A 算 21 层 (7.5ms) + 传输 (1ms) + 机器 B 算 22 层 (7.5ms)
  总计: ~16ms/token (没有加速, 还多了网络开销)
  -> 流水线并行不加速 decode, 只解决内存

张量并行 (2 台):
  机器 A 算一半 (7.5ms) + 交换 (1ms) + 继续
  机器 B 同时算另一半 (7.5ms)
  总计: ~8.5ms/token (接近 2 倍加速!)
  -> 张量并行真正加速 decode
```

---

## 25.2 分工策略

`ds4_tp.h` 的注释（`ds4_tp.h:7-15`）描述了分工：

```
两台机器运行相同的逻辑模型:
  - 每台保留一半的路由专家 (experts)
  - 稠密权重 (注意力、共享专家) 两台都有 (复制)
  - Leader 负责输入/采样
  - Worker 镜像 Leader 的每个操作 (锁步)
```

### 权重分配

```
路由专家 (256 个):
  机器 A: 专家 0-127 (前一半)
  机器 B: 专家 128-255 (后一半)
  -> 每台只需一半专家权重的内存

稠密权重 (注意力投影、共享专家、Embedding):
  机器 A: 完整副本
  机器 B: 完整副本
  -> 两台都有, 不省这部分内存
```

> **为什么稠密权重要复制？** 因为稠密权重每次都参与计算，无法像路由专家那样只算一半。复制虽然多用内存，但避免了每层都交换中间结果。

---

## 25.3 Leader/Worker 锁步协议

张量并行要求两台机器**完全同步**执行：

```
Leader (Rank 0):                Worker (Rank 1):
  session_sync(prompt)     ───>  session_sync(prompt)    // 镜像
  session_eval(token)     ───>  session_eval(token)      // 镜像
  
  for 每层:
    算注意力 (一半)               算注意力 (另一半)
    gate_exchange(ATTN)    <──>  gate_exchange(ATTN)     // 交换部分和
    合并 -> 继续                  合并 -> 继续
    
    算 MoE (自己的专家)           算 MoE (自己的专家)
    gate_exchange(FFN)     <──>  gate_exchange(FFN)      // 交换部分和
    合并 -> 继续                  合并 -> 继续
  
  argmax(logits)           ───>  (不采样, 等 Leader)
  输出 token                     等待下一个 eval
```

### ds4_session_eval_probe_tp

在 `ds4.c:59885` 中，`ds4_session_eval_probe_tp` 实现了锁步协议：

```c
static int ds4_session_eval_probe_tp(ds4_session *s, int token, ...) {
    if (ds4_session_tp_leader(s)) {
        // Leader: 发送 eval 指令给 Worker
        ds4_tp_send_eval(e->tp.ctx, s->tp_session_id, ++e->tp.eval_seq, token);
    }
    // 两台机器同时执行
    int rc = ds4_session_eval_internal(s, token, probe_mtp, err, errlen);
    
    // Vocab-split: 合并 logits 的两半
    if (rc == 0 && s->engine->tp.active && s->engine->tp.vocab_split) {
        if (s->engine->tp.rank == 0) {
            // Leader 接收 Worker 的 logits 后半
            ds4_tp_recv_logits_half(s->engine->tp.ctx, s->logits + vhalf, vhalf);
        } else {
            // Worker 发送 logits 后半
            ds4_tp_send_logits_half(s->engine->tp.ctx, s->logits + vhalf, vhalf);
        }
    }
    return rc;
}
```

---

## 25.4 门控交换

每层有两个"门控"（gate）需要交换数据（`ds4_tp.h:24-27`）：

```c
enum {
    DS4_TP_GATE_ATTN = 0,    // 注意力门控
    DS4_TP_GATE_FFN = 1,     // FFN 门控
    DS4_TP_GATES_PER_LAYER = 2,
};
```

### 注意力门控

```
注意力计算:
  机器 A 算 32 个头 (前一半)
  机器 B 算 32 个头 (后一半)
  
  gate_exchange(ATTN):
    A 发送自己的 32 个头结果给 B
    B 发送自己的 32 个头结果给 A
    两台都得到完整的 64 个头结果
```

### FFN 门控

```
MoE 计算:
  机器 A 算选中的专家中属于自己的一半
  机器 B 算选中的专家中属于自己的一半
  
  gate_exchange(FFN):
    A 发送自己的 FFN 部分和给 B
    B 发送自己的 FFN 部分和给 A
    两台都得到完整的 FFN 结果
```

---

## 25.5 传输方式

`ds4_tp.h:3-6` 描述了两种传输方式：

```c
typedef enum {
    DS4_TP_TRANSPORT_AUTO = 0,  // 自动选择
    DS4_TP_TRANSPORT_RDMA,      // RDMA (远程直接内存访问)
    DS4_TP_TRANSPORT_TCP,       // TCP
} ds4_tp_transport;
```

### RDMA vs TCP

```
TCP:
  数据: 机器 A -> 内核 -> 网卡 -> 网络 -> 网卡 -> 内核 -> 机器 B
  延迟: ~1-2ms (经过内核协议栈)
  CPU 开销: 有 (内核参与)

RDMA (Remote Direct Memory Access):
  数据: 机器 A 内存 -> 网卡 -> 网络 -> 网卡 -> 机器 B 内存
  延迟: ~0.1ms (绕过内核)
  CPU 开销: 无 (网卡直接读写内存)
```

> **Thunderbolt RDMA**：ds4 支持通过 Thunderbolt 网络做 RDMA。两台 Mac 用 Thunderbolt 线直连，延迟极低，适合张量并行。

### 配置

```bash
# Leader (机器 A)
./ds4 --tensor-parallel leader \
      --listen 0.0.0.0:9000 \
      --transport rdma \
      --rdma-device rdma0

# Worker (机器 B)
./ds4 --tensor-parallel worker \
      --leader 192.168.1.100:9000 \
      --transport rdma \
      --rdma-device rdma0
```

---

## 25.6 推测解码与张量并行

张量并行和推测解码可以组合使用。`ds4_session_tp_spec_cycle()`（`ds4.h:330`）实现了 TP worker 的推测验证：

```c
int ds4_session_tp_spec_cycle(ds4_session *s, const int *drafts, int draft_n,
                              char *err, size_t errlen);
```

```
Leader 驱动推测解码:
  1. MTP 起草草稿
  2. Leader 发送 eval 镜像给 Worker
  3. 两台同时验证草稿
  4. Leader 决定接受/回滚
  5. Worker 遵循 Leader 的决定
```

---

## 25.7 身份验证

两台机器配对时，先交换身份信息确保匹配（`ds4_tp.h:30-40`）：

```c
typedef struct {
    uint64_t gguf_bytes;    // 模型文件大小 (必须一致)
    uint32_t model_id;      // 模型 ID
    uint32_t n_layer;       // 层数
    uint32_t n_embd;        // 嵌入维度
    uint32_t n_vocab;       // 词表大小
    uint32_t quant_bits;    // 量化位数
    uint32_t ctx_size;      // 上下文大小
    // 门控调度信息
    uint32_t gate_slot_start;
    uint32_t gate_slot_step;
    uint32_t gates_per_token;
} ds4_tp_identity;
```

如果两台机器的模型不匹配（大小、架构、量化不同），连接会被拒绝。

---

## 25.8 性能预期

```
两台 M5 Max (Thunderbolt RDMA):
  单机 decode: ~15 ms/token (~65 tok/s)
  TP decode:   ~9 ms/token  (~110 tok/s)
  加速比: ~1.7x (接近理论 2x, 网络开销占 ~1.5ms)

两台 M3 Ultra (RDMA):
  类似加速比

TCP (千兆网):
  网络延迟 ~1ms/门控 × 2 门控/层 × 43 层 = 86ms
  -> TCP 张量并行不可行 (太慢)
  -> 必须 RDMA
```

> **结论**：张量并行需要 RDMA 级别的低延迟网络（Thunderbolt 或 InfiniBand），普通 TCP 网络的延迟太大。

---

## 本章小结

- 张量并行：两台机器同时算同一层，各算一半，真正加速 decode
- 路由专家按半分配，稠密权重复制
- Leader/Worker 锁步协议：Leader 驱动，Worker 镜像每个操作
- 每层两个门控交换：注意力门控 + FFN 门控
- RDMA 绕过内核，延迟比 TCP 低 10 倍
- 张量并行 + 推测解码可以组合
- 需要 RDMA 网络，TCP 延迟太大不可行
- 典型加速比 1.7x（两台机器）

## 动手实验

1. 在 `ds4_tp.h` 阅读头文件注释，理解锁步协议
2. 在 `ds4.c:59885` 查看 `ds4_session_eval_probe_tp`，理解 Leader/Worker 同步
3. 思考：为什么稠密权重要复制而路由专家可以拆分？

## 下一章预告

第 26 章，性能分析与调优实战--用 profiling 工具定位瓶颈，做数据驱动的优化。
