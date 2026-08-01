# 第 17 章 采样策略：从概率到文字

## 本章导读

模型输出的 logits 是一组原始分数（129280 个），不是文字。怎么从这组分数中选出一个 token？这就是采样策略要解决的问题。这章我们讲解 ds4 支持的几种采样方法。

读完本章你会理解：
- Temperature 怎么控制随机性
- Top-P（核采样）的原理
- Min-P 采样的优势
- 贪心解码 vs 采样的取舍

**前置知识**：第 3、16 章

---

## 17.1 从 logits 到概率

模型最后一层输出 logits（一组原始分数）。第一步是把它变成概率分布：

```
logits = [2.1, 0.3, -1.5, 5.8, 0.1, ...]  (129280 个分数)

经过 softmax:
  probs[i] = exp(logits[i]) / Σ exp(logits[j])

probs = [0.02, 0.003, 0.0005, 0.85, 0.002, ...]  (和为 1 的概率)
```

分数最高的 token（logits=5.8 对应的）概率最大（0.85），但不一定选它--采样策略决定了怎么选。

---

## 17.2 贪心解码（Argmax）

最简单的策略：**永远选概率最高的 token**。

```c
// ds4_engine_generate_argmax 中的逻辑
int token = ds4_session_argmax(s);  // 找 logits 最大的位置
```

```
logits = [2.1, 0.3, -1.5, 5.8, 0.1, ...]
                            ↑ 最大
选择 token 3
```

### 特点

```
优点:
  - 确定性：同样的输入永远得到同样的输出
  - 简单：不需要随机数
  - 适合：事实性问答、代码生成、数学推理

缺点:
  - 重复倾向：容易陷入循环（"你好你好你好..."）
  - 缺乏创造性：每次回答都一样
```

> **ds4 的默认行为**：`temperature=0`（即 `-t 0`）时走 argmax 路径。CLI 中 `ds4_engine_generate_argmax` 是优化过的快速路径，不需要创建完整 session（除非有 TP 或 MTP）。

---

## 17.3 Temperature（温度）

Temperature 控制概率分布的"平坦程度"：

```
probs[i] = exp(logits[i] / T) / Σ exp(logits[j] / T)

T = 1.0: 原始分布
T = 0.5: 分布更尖锐（高概率的更高，低概率的更低）
T = 2.0: 分布更平坦（概率更均匀）
T = 0.0: 完全贪心（等于 argmax）
```

### 直觉

```
T = 0.1 (很冷):
  几乎总是选最高概率的 token
  输出非常确定，但可能重复

T = 1.0 (正常):
  按原始概率分布随机选
  平衡确定性和创造性

T = 2.0 (很热):
  概率分布很平坦，低概率 token 也有机会
  输出更随机、更有创造性，但也更容易出错
```

### Temperature 的数学效果

```
logits = [1.0, 2.0, 3.0]

T=1.0:  exp([1, 2, 3]) = [2.72, 7.39, 20.09]
  probs = [0.09, 0.24, 0.67]  ← 第三个占 67%

T=0.5:  exp([2, 4, 6]) = [7.39, 54.6, 403.4]
  probs = [0.016, 0.12, 0.86]  ← 第三个占 86%（更集中）

T=2.0:  exp([0.5, 1.0, 1.5]) = [1.65, 2.72, 4.48]
  probs = [0.19, 0.31, 0.50]  ← 第三个只占 50%（更均匀）
```

---

## 17.4 Top-P（核采样 / Nucleus Sampling）

Temperature 的问题：即使调高温度，极低概率的 token 也可能被选中，导致输出荒谬。Top-P 解决这个问题。

### 原理

```
1. 按概率从高到低排序
2. 累加概率，直到达到阈值 P（如 0.9）
3. 只在累加达到 P 的那些 token 中采样

P = 0.9 的例子:
  排序后: [0.85, 0.07, 0.03, 0.02, 0.01, 0.005, ...]
  累加:   [0.85, 0.92, ...]  ← 到第二个就超过 0.9
  只在前 2 个中采样
```

```
                    全部 token (129280 个)
  ┌──────────────────────────────────────┐
  │ 0.85  0.07  0.03  0.02  │ 0.01  0.005 ... │
  │  ←── 累加 ≥ 0.9 ──→     │  ← 被排除 →     │
  │  ← 在这些中采样 →        │                 │
  └──────────────────────────────────────┘
```

### 效果

```
P = 1.0: 等于纯 temperature 采样（不排除任何 token）
P = 0.9: 排除概率很低的 token，保留 90% 概率质量的候选
P = 0.1: 只选最高概率的少数 token（非常保守）
```

> **为什么叫"核"采样？** 因为保留的 token 集合是概率分布的"核心"（概率质量最集中的部分）。

---

## 17.5 Min-P 采样

Top-P 的问题是：阈值 P 是**绝对概率**，但不同 token 的概率分布差异很大。Min-P 用**相对概率**做阈值。

### 原理

```
1. 找到最高概率: max_prob = max(probs)
2. 阈值 = max_prob × min_p
3. 只保留概率 ≥ 阈值的 token

min_p = 0.05 的例子:
  最高概率 = 0.85
  阈值 = 0.85 × 0.05 = 0.0425
  保留: [0.85, 0.07]  (0.03 < 0.0425, 被排除)
```

### Min-P vs Top-P

```
场景 1: 模型很确定 (max_prob = 0.95)
  Top-P 0.9: 保留前 1-2 个 token
  Min-P 0.05: 阈值 = 0.0475, 保留前 1-2 个 token
  → 效果类似

场景 2: 模型不确定 (max_prob = 0.15)
  Top-P 0.9: 保留很多 token (要累加到 0.9)
  Min-P 0.05: 阈值 = 0.0075, 保留概率 > 0.0075 的 token
  → Min-P 更合理（根据模型的不确定程度自适应）
```

> **ds4 的默认值**：`DS4_DEFAULT_MIN_P = 0.05f`（`ds4.h:38`）。Min-P 是 ds4 的主要采样策略，因为它对不同概率分布的适应性更好。

---

## 17.6 ds4 中的采样实现

在 `run_sampled_generation()`（`ds4_cli.c:521`）中，采样调用：

```c
token = ds4_session_sample(session,
                           cfg->gen.temperature,  // 温度
                           0,                      // top_k (未使用)
                           cfg->gen.top_p,         // top-p
                           cfg->gen.min_p,         // min-p
                           &rng);                  // 随机数生成器
```

### 采样流程

`ds4_session_sample` 内部的逻辑（简化）：

```
1. 取 logits (129280 个)
2. 如果 temperature <= 0: 直接 argmax (贪心)
3. 否则:
   a. logits /= temperature  (应用温度)
   b. softmax -> probs
   c. 如果 top_p < 1.0: 排除累加超过 top_p 的 token
   d. 如果 min_p > 0: 排除概率 < max_prob × min_p 的 token
   e. 在剩余 token 中按概率随机采样
```

### 随机数生成

```c
uint64_t rng = cfg->gen.seed ? cfg->gen.seed :
    ((uint64_t)time(NULL) ^ ((uint64_t)getpid() << 32) ^ (uint64_t)clock());
```

如果指定了 seed（`--seed`），输出可复现。否则用时间 + PID + clock 生成随机种子。

---

## 17.7 采样参数的选择建议

| 场景 | Temperature | Top-P | Min-P | 说明 |
|------|------------|-------|-------|------|
| 代码生成 | 0 (贪心) | - | - | 确定性高，避免随机 |
| 数学推理 | 0 (贪心) | - | - | 需要精确 |
| 事实问答 | 0.1-0.3 | 0.9 | 0.05 | 低随机性 |
| 创意写作 | 0.7-1.0 | 0.95 | 0.05 | 平衡创造性和连贯性 |
| 头脑风暴 | 1.0-1.3 | 0.97 | 0.03 | 高随机性 |

> **实践建议**：
> - 需要正确性的任务用 `temperature=0`（贪心）
> - 需要创造性的任务用 `temperature=0.7-1.0 + min_p=0.05`
> - 不要把 temperature 调太高（>1.5），容易产生乱码
> - Min-P 通常比 Top-P 更稳定，推荐优先使用

---

## 17.8 停止条件

采样循环什么时候停止？在 `run_sampled_generation()` 中：

```c
while (generated < max_tokens && !cli_interrupt_requested()) {
    token = ds4_session_sample(session, ...);

    // 条件 1: 遇到停止 token (EOS)
    if (ds4_token_is_stop_for_think_mode(engine, token, think_mode))
        break;

    // 条件 2: 达到最大 token 数
    // (while 条件: generated < max_tokens)

    // 条件 3: 上下文窗口用完
    // (while 条件: ds4_session_pos < ctx_size)

    // 条件 4: 用户 Ctrl+C
    // (cli_interrupt_requested())
}
```

### 停止 token

- **EOS**（End of Sequence）：模型生成的特殊 token，表示"我说完了"
- **Think mode 标记**：如 `<|im_end|>`，表示当前对话轮结束

---

## 17.9 采样对输出的影响

同样的 prompt，不同采样参数的输出对比：

```
Prompt: "写一首关于秋天的诗"

Temperature=0 (贪心):
  "秋风起落叶飞，天高云淡雁南归。"
  (每次运行结果一样)

Temperature=0.7, Min-P=0.05:
  "金色的叶子在风中起舞，像一首无声的诗。"
  (每次运行可能不同)

Temperature=1.5, Min-P=0.01:
  "秋天的苹果树！金黄！紫色的风，叶子跳舞..."
  (更随机，可能不太连贯)
```

---

## 本章小结

- Logits 经 softmax 变成概率分布
- 贪心解码（Argmax）：选概率最高的，确定性但可能重复
- Temperature：控制分布平坦程度，T 越低越确定
- Top-P：保留累加概率达到 P 的 token，排除长尾
- Min-P：保留概率 ≥ max_prob × min_p 的 token，自适应性强
- ds4 默认用 Min-P=0.05，Temperature=1.0
- seed 控制可复现性
- 停止条件：EOS / max_tokens / 上下文窗口 / 用户中断

## 动手实验

1. 运行 `./ds4 -p "写一首诗" -t 0 -n 50` 和 `./ds4 -p "写一首诗" -t 0.8 -n 50`，对比输出差异
2. 用相同 seed 运行两次：`./ds4 -p "你好" -t 0.8 --seed 42 -n 20`，确认结果一致
3. 思考：为什么 Min-P 比 Top-P 更能适应不同的概率分布？

## 下一章预告

第四部分结束了。从第 18 章开始进入第五部分"优化篇"--量化、GPU 内核、Flash Attention、SSD 流式、推测解码。让推理更快更省。
