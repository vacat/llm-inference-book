# 第 22 章 推测解码：猜对了就赚

## 本章导读

Decode 阶段每次只处理 1 个 token，GPU 算力大量闲置。推测解码（Speculative Decoding）利用这些闲置算力，让一个小模型"猜"几个 token，再用大模型"验证"。猜对了就白赚几个 token 的速度。

读完本章你会理解：
- 推测解码的起草-验证状态机
- MTP（Multi-Token Prediction）的工作方式
- DSpark 增强推测解码
- 为什么推测解码是"正确性安全"的

**前置知识**：第 16-17 章

---

## 22.1 问题：Decode 的算力浪费

```
Decode 每个 token:
  读 ~15GB 权重 (内存带宽瓶颈)
  算 ~74 GFLOP (计算量不大)
  
  GPU 计算单元利用率: ~5%
  -> 95% 的算力在闲着！
```

如果能一次验证多个 token，就能利用闲置算力：

```
普通 decode:
  token₁ -> token₂ -> token₃ -> token₄  (4 次串行)

推测解码:
  1. 小模型猜: token₂, token₃, token₄  (一次并行猜 3 个)
  2. 大模型验证: 一次前向同时验证 3 个
  3. 假设前 2 个猜对: 接受 token₂, token₃, 拒绝 token₄
  4. 输出 token₂, token₃ (一次得到了 2 个 token)
  
  理论加速: 接受 N 个 = N 倍速度
```

---

## 22.2 推测解码的状态机

推测解码是一个"起草-验证-接受/回滚"的状态机：

```
                    ┌──────────┐
                    │  正常     │
                    │  Decode   │
                    │  得到 T₁  │
                    └────┬─────┘
                         │
                         ▼
                ┌────────────────┐
                │ MTP 起草        │
                │ 猜 T₂, T₃, T₄ │
                └───────┬────────┘
                        │
                        ▼
                ┌────────────────┐
                │ 目标模型验证    │
                │ 批量前向 T₂~T₄ │
                └───────┬────────┘
                        │
              ┌─────────┼─────────┐
              ▼         ▼         ▼
         ┌────────┐ ┌────────┐ ┌────────┐
         │全部接受│ │部分接受│ │全部拒绝│
         │T₂T₃T₄ │ │T₂T₃   │ │回滚    │
         └───┬────┘ └───┬────┘ └───┬────┘
             │          │          │
             └──────────┼──────────┘
                        ▼
                   回到正常 Decode
```

### 一个周期

```
周期开始: 刚 decode 出 token T₁

1. 起草 (Draft):
   MTP 模型根据 T₁ 猜: T₂=cat, T₃=sat, T₄=on
   (猜 3 个, 成本很低)

2. 验证 (Verify):
   目标模型一次性前向 T₁, T₂, T₃, T₄ (4 个 token 的批量)
   得到目标模型自己的预测:
     T₁ -> 目标说下一个应该是 "the"  (不是 "cat")
     T₂ -> 目标说下一个应该是 "sat"  (猜对了!)
     T₃ -> 目标说下一个应该是 "on"   (猜对了!)

3. 接受/回滚:
   T₂ 猜的 "cat" ≠ 目标的 "the" -> 拒绝 T₂, 用目标的 "the"
   -> 接受 0 个草稿, 输出 "the"
   -> 回滚 T₃, T₄ 的 KV 缓存

   或者:
   T₂ 猜的 "cat" = 目标的 "cat" -> 接受
   T₃ 猜的 "sat" = 目标的 "sat" -> 接受
   T₄ 猜的 "on" ≠ 目标的 "mat" -> 拒绝, 用 "mat"
   -> 接受 2 个草稿 + 1 个修正 = 3 个 token
   -> 一次得到了 3 个 token!
```

---

## 22.3 源码：ds4_session_eval_speculative_argmax

`ds4_session_eval_speculative_argmax()`（`ds4.c:64129`）是推测解码的入口。核心注释（`ds4.c:64288`）说得很清楚：

```c
/*
 * MTP in DeepSeek V4 is a speculative drafter, not a replacement sampler.
 * The target model still defines the exact output stream.  A cycle starts
 * by accepting one normal target token, then asks the MTP block to propose
 * a short suffix.  The suffix is useful only if the target model can verify
 * several proposed positions together.
 */
```

### 关键设计：正确性安全

```
推测解码不改变输出结果:
  - 草稿只是"猜测", 不直接输出
  - 每个草稿 token 都被目标模型验证
  - 只有和目标模型一致的才接受
  - 不一致的用目标模型的结果替换
  -> 最终输出 = 纯目标模型的输出 (只是更快)
```

### 参数

在 `ds4_engine_options` 中：

```c
int mtp_draft_tokens;    // 每次猜几个 token (1-16)
float mtp_margin;        // 接受草稿的余量 (默认 3.0)
```

- `mtp_draft_tokens = 3`：每次猜 3 个 token
- `mtp_margin`：控制接受草稿的保守程度。margin 越高越保守（只接受高置信度的草稿），接受率低但更安全

---

## 22.4 MTP 模型

MTP（Multi-Token Prediction）是 DeepSeek V4 的推测解码起草器。它是一个小模型，有自己的权重（`ds4_mtp_weights`，`ds4.c:4110`）。

### MTP 的结构

```c
typedef struct {
    ds4_tensor *e_proj;      // 嵌入投影
    ds4_tensor *h_proj;      // 隐藏状态投影
    ds4_tensor *enorm;       // 嵌入归一化
    ds4_tensor *hnorm;       // 隐藏状态归一化
    ds4_tensor *norm;        // 最终归一化
    ds4_tensor *hc_head_base;// HC 头基础
    ds4_tensor *hc_head_fn;  // HC 头函数
    ds4_tensor *hc_head_scale;// HC 头缩放
    ds4_layer_weights block; // 一个 Transformer 层 (复用层结构)
} ds4_mtp_weights;
```

MTP 模型很轻量：只有一个 Transformer 层 + 投影头。它接收目标模型的隐藏状态，预测下一个 token。

### MTP 的输入

```
目标模型 decode token T₁ -> 得到隐藏状态 h₁

MTP 起草:
  MTP(T₁, h₁) -> 预测 T₂
  MTP(T₂, h₂) -> 预测 T₃  (用上一步的隐藏状态)
  MTP(T₃, h₃) -> 预测 T₄
  -> 草稿: [T₂, T₃, T₄]
```

> **MTP 有自己的 KV 缓存**：MTP 的草稿使用独立的 SWA 缓存（`mtp_n_raw`），验证后用计数器控制可见行数，而非物理回滚。这让回滚操作很快（只改计数器，不移动数据）。

---

## 22.5 DSpark：增强推测解码

DSpark 是 DeepSeek V4 的增强推测解码技术，比基础 MTP 更强。

### DSpark vs MTP

```
MTP (基础):
  - 单层 Transformer 起草
  - 简单 argmax 猜测
  - 草稿命中率一般

DSpark (增强):
  - 多阶段 Markov 预测
  - 置信度评估
  - 自适应草稿长度
  - 草稿命中率更高
```

### DSpark 的参数

```c
typedef struct {
    uint32_t stages;          // 预测阶段数
    uint32_t block_size;      // 块大小
    uint32_t markov_rank;     // Markov 预测阶数
    uint32_t noise_token_id;  // 噪声 token
    uint32_t target_layer_count;
    uint32_t target_layers[DS4_DSPARK_MAX_TARGET_LAYERS];
    bool has_confidence_head; // 有置信度头
    bool has_final_head;      // 有最终预测头
    // ...
} ds4_dspark_summary;
```

DSpark 用 Markov 模型和多阶段预测来提高草稿质量。`confidence_head` 评估草稿的置信度，`dspark_confidence_threshold` 控制是否采纳低置信度草稿。

### 使用 DSpark

```bash
./ds4 --dspark --mtp gguf/DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf \
       -p "你好" -n 100
```

- `--dspark`：启用 DSpark
- `--mtp FILE`：指定 MTP/DSpark 支持模型文件

---

## 22.6 接受率与加速

### 接受率

```
接受率 = 被接受的草稿 token 数 / 总草稿 token 数

MTP (mtp_draft_tokens=3):
  假设每个草稿的独立命中率 60%
  全部 3 个命中: 0.6³ = 21.6%
  至少 2 个命中: 64.8%
  至少 1 个命中: 93.6%
  平均接受: ~1.6 个/周期

理论加速:
  普通 decode: 1 token/周期
  推测 decode: 1 + 1.6 = 2.6 token/周期
  -> 2.6 倍加速 (理想)
  实际: ~1.5-2 倍 (验证有开销)
```

### 观察 MTP 命中率

```bash
DS4_MTP_PROBE=1 ./ds4 -p "test" -n 50
# 输出会显示: hit=X/Y (Z%)
```

### 调优

```
mtp_draft_tokens:
  1: 最保守, 几乎不增加开销
  3: 平衡 (默认)
  8: 激进, 验证开销大但接受多时收益大

mtp_margin:
  3.0 (默认): 平衡
  更高: 更保守, 只接受高置信度
  更低: 更激进, 接受更多但可能有回滚
```

---

## 22.7 推测解码的代价

```
代价:
  1. MTP 模型加载: 额外的模型文件 (~1-2GB)
  2. 起草计算: MTP 前向传播 (~2ms/草稿)
  3. 验证开销: 批量前向比单 token 慢
  4. 回滚开销: KV 缓存回滚 (用计数器, 很小)

收益:
  每个接受的草稿 token = 省一次完整 decode (~15ms)
  起草成本 ~2ms vs decode ~15ms
  -> 接受 1 个就赚 13ms
```

> **什么时候推测解码不划算？**
> - 接受率很低时（草稿质量差）
> - 上下文极短时（decode 本身很快）
> - 模型输出高度不可预测时（创意写作可能比代码生成的接受率低）

---

## 本章小结

- 推测解码利用 decode 时的闲置算力，用小模型猜、大模型验
- 状态机：正常 decode -> MTP 起草 -> 批量验证 -> 接受/回滚
- 正确性安全：草稿只加速不改变输出，目标模型定义最终输出流
- MTP 模型只有一层 Transformer，轻量快速
- DSpark 用 Markov 预测 + 置信度评估提升草稿质量
- MTP 有独立 KV 缓存，回滚只改计数器不改数据
- 接受率决定加速效果，典型 1.5-2 倍加速
- `DS4_MTP_PROBE=1` 观察命中率

## 动手实验

1. 运行 `DS4_MTP_PROBE=1 ./ds4 -p "1+1=" -n 20`，观察 MTP 命中率
2. 尝试不同 `--mtp-draft 1/2/3`，对比速度
3. 思考：为什么推测解码在代码生成中比创意写作中更有效？

## 下一章预告

第五部分优化篇结束了。从第 23 章开始进入第六部分"工程篇"--服务化部署、分布式推理、张量并行、性能调优。
