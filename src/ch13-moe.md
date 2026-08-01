# 第 13 章 MoE：混合专家路由

## 本章导读

MoE（Mixture of Experts，混合专家）是 DeepSeek V4 最核心的架构创新。256 个专家中每次只选 6 个来计算，用更少的算力获得更强的能力。这章我们拆解路由门控的打分、选择、加权全过程。

读完本章你会理解：
- 路由门控如何给 256 个专家打分
- Top-K 选择算法的实现
- Hash 路由 vs 学习路由的区别
- 专家权重的归一化
- 选中专家的计算流程

**前置知识**：第 8-12 章

---

## 13.1 MoE 路由全流程

回顾第 5 章，每层的 FFN 是 MoE 结构：

```
输入 x (4096 维)
    │
    ├──────────────────────────────────────────┐
    │                                           │
    ▼                                           ▼
┌─ 路由门控 ──────────────┐          ┌─ 共享专家 ──────────┐
│  gate(x) -> 256 个分数  │          │  FFN(x)             │
│  选 top-6               │          │  (每次都算)          │
│  计算专家权重            │          └────────┬────────────┘
└────────────┬────────────┘                   │
             │                                 │
             ▼                                 │
┌─ 6 个路由专家 ──────────┐                    │
│  每个: FFN(x) × 权重    │                    │
│  累加 6 个专家的输出     │                    │
└────────────┬────────────┘                   │
             │                                 │
             ▼                                 ▼
         Σ(路由专家输出)  +  Σ(共享专家输出)
             │
             ▼
        FFN 总输出 (4096 维)
```

---

## 13.2 路由门控打分

路由门控是一个小线性层，给每个专家打一个分数。

`layer_router_probs_one()`（`ds4.c:10652`）：

```c
static void layer_router_probs_one(
        float probs[DS4_MAX_EXPERT],
        const ds4_model *model,
        const ds4_layer_weights *layer,
        const float *x) {
    float logits[DS4_MAX_EXPERT];

    // 矩阵乘法: logits = ffn_gate_inp × x
    // ffn_gate_inp 是 (4096 -> 256) 的权重矩阵
    matvec_any(logits, model, layer->ffn_gate_inp, x);

    // 对每个专家的 logit 做 softplus + sqrt
    for (uint32_t i = 0; i < DS4_N_EXPERT; i++) {
        probs[i] = sqrtf(softplus_stable(logits[i]));
    }
}
```

### 为什么用 softplus + sqrt？

标准做法是 softmax，但 DeepSeek V4 用了不同的激活：

```
标准 softmax:  probs[i] = exp(logits[i]) / Σ exp(logits[j])
DeepSeek V4:   probs[i] = sqrt(softplus(logits[i]))
```

`softplus_stable()`（`ds4.c:10489`）：

```c
static float softplus_stable(float x) {
    if (x > 20.0f) return x;          // 大正数：softplus(x) ≈ x
    if (x < -20.0f) return expf(x);   // 大负数：softplus(x) ≈ exp(x) ≈ 0
    return log1pf(expf(x));           // 正常：log(1 + exp(x))
}
```

```
softplus(x) = log(1 + exp(x))
sqrt(softplus(x))

效果:
  logit 很大 -> prob ≈ sqrt(logit)    (增长缓慢，防止某个专家独大)
  logit 很小 -> prob ≈ sqrt(exp(x))   (快速趋零)
  logit = 0  -> prob = sqrt(log(2))   (合理中间值)
```

> **设计意图**：`sqrt(softplus(x))` 是一个比 softmax 更平缓的激活。它让分数差异不那么极端，避免路由过于集中到少数专家上（负载均衡）。

---

## 13.3 Top-K 选择

选 6 个分数最高的专家。`layer_topk_selected_experts_from_probs()`（`ds4.c:10710`）：

```c
static void layer_topk_selected_experts_from_probs(
        int selected[DS4_MAX_EXPERT_USED],       // 输出：选中的专家编号
        float expert_weight[DS4_MAX_EXPERT_USED], // 输出：专家权重
        const ds4_model *model,
        const ds4_layer_weights *layer,
        const float probs[DS4_MAX_EXPERT]) {

    float selection[DS4_MAX_EXPERT];
    memcpy(selection, probs, sizeof(selection));

    // 偏置项：给某些专家加分（训练时学到的偏好）
    if (layer->ffn_exp_probs_b) {
        const float *bias = tensor_data(model, layer->ffn_exp_probs_b);
        for (uint32_t i = 0; i < DS4_N_EXPERT; i++)
            selection[i] += bias[i];
    }

    // 选 top-6
    topk_desc(selection, DS4_N_EXPERT, DS4_N_EXPERT_USED, selected);

    // 计算权重（注意：用原始 probs，不是加了 bias 的 selection）
    float sum = 0.0f;
    for (uint32_t i = 0; i < DS4_N_EXPERT_USED; i++) {
        expert_weight[i] = probs[selected[i]];  // 用原始概率
        sum += expert_weight[i];
    }
    if (sum < 6.103515625e-5f) sum = 6.103515625e-5f;  // 防除零
    for (uint32_t i = 0; i < DS4_N_EXPERT_USED; i++) {
        expert_weight[i] = expert_weight[i] / sum * DS4_EXPERT_WEIGHT_SCALE;
    }
}
```

### 偏置的作用

```c
if (layer->ffn_exp_probs_b) {
    for (uint32_t i = 0; i < DS4_N_EXPERT; i++)
        selection[i] += bias[i];
}
```

`ffn_exp_probs_b` 是一个可学习的偏置向量（256 维）。它给某些专家"加分"，让它们更容易被选中。这是训练时学到的专家使用偏好。

> **重要细节**：选择用 `selection`（加了 bias），但计算权重用 `probs`（没加 bias）。bias 只影响"选谁"，不影响"给多大权重"。这防止 bias 过度影响专家输出的相对权重。

### topk_desc：选择算法

`topk_desc()`（`ds4.c:10675`）是简单的插入排序 top-k：

```c
static void topk_desc(const float *score, int n, int k, int *idx) {
    for (int i = 0; i < k; i++) idx[i] = -1;

    for (int i = 0; i < n; i++) {           // 遍历所有 256 个专家
        for (int j = 0; j < k; j++) {        // 找插入位置
            if (idx[j] < 0 || score[i] > score[idx[j]]) {
                // 插入：后面的后移
                for (int m = k - 1; m > j; m--) idx[m] = idx[m - 1];
                idx[j] = i;
                break;
            }
        }
    }
}
```

维护一个长度为 6 的有序数组，遍历所有 256 个分数，把比当前最小值大的插入。时间复杂度 O(n×k) = O(256×6) = O(1536)，对于 256 选 6 足够快。

### 权重归一化

```c
expert_weight[i] = probs[selected[i]] / sum * DS4_EXPERT_WEIGHT_SCALE;
```

选中的 6 个专家的权重归一化后乘以 `DS4_EXPERT_WEIGHT_SCALE`（DeepSeek V4 Flash 是 1.5）。这个 scale 是训练时定的，控制 MoE 输出的整体强度。

---

## 13.4 Hash 路由

前 3 层（`n_hash_layer = 3`）用 hash 路由，不走门控打分：

在 `layer_routed_moe_one()`（`ds4.c:10761`）中：

```c
if (layer->ffn_gate_tid2eid) {
    // Hash 路由：根据 token id 直接映射到专家
    layer_hash_selected_experts(selected, model, layer, token);
    layer_hash_router_weights_one(expert_weight, model, layer, x, selected);
} else {
    // 学习路由：门控打分 + top-k
    layer_topk_selected_experts(selected, expert_weight, model, layer, x);
}
```

### Hash 路由 vs 学习路由

```
学习路由:
  gate(x) -> 256 个分数 -> top-6
  需要计算矩阵乘法 (4096 × 256)
  路由质量高，但有计算开销

Hash 路由:
  hash(token_id) -> 直接映射到 6 个专家
  不需要矩阵乘法，极快
  路由质量略低（不依赖输入内容）
```

> **为什么前 3 层用 hash？** 前几层在做初步特征提取，输入还没分化出明显模式，学习路由的收益不大。用 hash 路由省计算，把算力留给后面的层。`ffn_gate_tid2eid` 是一个预计算的 token-id 到专家的映射表。

### Hash 路由的权重

`layer_hash_router_weights_from_probs()`（`ds4.c:10665`）：

```c
static void layer_hash_router_weights_from_probs(
        float weights_out[DS4_MAX_EXPERT_USED],
        const float probs[DS4_MAX_EXPERT],
        const int selected[DS4_MAX_EXPERT_USED]) {
    float sum = 0.0f;
    for (uint32_t i = 0; i < DS4_N_EXPERT_USED; i++) {
        weights_out[i] = probs[selected[i]];  // 用门控概率做权重
        sum += weights_out[i];
    }
    if (sum < 6.103515625e-5f) sum = 6.103515625e-5f;
    for (uint32_t i = 0; i < DS4_N_EXPERT_USED; i++) {
        weights_out[i] = weights_out[i] / sum * DS4_EXPERT_WEIGHT_SCALE;
    }
}
```

即使选择用 hash，权重仍用门控概率计算。这样 hash 路由的专家权重质量有保证。

---

## 13.5 选中专家的计算

选好 6 个专家后，对每个专家做 FFN 计算。在 `layer_routed_moe_one()`（`ds4.c:10761`）中，核心循环（简化）：

```c
memset(out, 0, DS4_N_EMBD * sizeof(out[0]));  // 输出清零

// 量化输入（一次量化，6 个专家共用）
ds4_quantize_row_q8_K(x, xq, expert_in_dim);

for (int e = 0; e < DS4_N_EXPERT_USED; e++) {  // 6 个专家
    int eid = selected[e];
    float w = expert_weight[e];

    // 每个专家的 FFN:
    // gate = ffn_gate_exps[eid] × x
    // up   = ffn_up_exps[eid] × x
    // mid  = swiglu(gate, up)
    // down = ffn_down_exps[eid] × mid

    // ... 计算专家 e 的输出 ...
    
    // 累加: out += w × expert_output
    for (uint32_t i = 0; i < DS4_N_EMBD; i++)
        out[i] += w * expert_out[i];
}
```

### 关键优化：量化一次，六次复用

```c
ds4_quantize_row_q8_K(x, xq, expert_in_dim);
```

输入 x 被量化成 Q8_K 格式**一次**，然后 6 个专家都用这个量化结果做矩阵乘法。如果每个专家分别量化，要量化 6 次。这是重要的性能优化。

### 专家权重的存储

256 个专家的权重打包在三个大张量里：

```
ffn_gate_exps: [256 个专家的 gate 矩阵]  (256 × 4096 × 2048)
ffn_up_exps:   [256 个专家的 up 矩阵]    (256 × 4096 × 2048)
ffn_down_exps: [256 个专家的 down 矩阵]  (256 × 2048 × 4096)
```

计算第 `eid` 个专家时，从大张量中"抽取"第 `eid` 块。这种打包方式让 SSD 流式加载可以按需读取单个专家（第 21 章）。

---

## 13.6 MoE 的负载均衡问题

如果所有 token 都选同几个专家，其他专家就浪费了。这叫**负载不均衡**。

DeepSeek V4 通过以下机制缓解：

1. **偏置项（bias）**：训练时动态调整 bias，给冷门专家加分
2. **softplus + sqrt 激活**：让分数差异不那么极端
3. **共享专家**：无论路由结果如何，共享专家都参与计算，保证基础能力

---

## 13.7 完整 MoE 流程图

```
输入 x (4096 维)
    │
    ├──> 路由门控:
    │    matvec(ffn_gate_inp, x) -> logits (256 维)
    │    probs[i] = sqrt(softplus(logits[i]))
    │    selection[i] = probs[i] + bias[i]
    │    topk_desc(selection, 256, 6) -> selected[6]
    │    expert_weight[i] = probs[selected[i]] / sum × 1.5
    │
    ├──> 量化输入:
    │    xq = quantize_q8_K(x)
    │
    ├──> 6 个路由专家:
    │    for e in [0..5]:
    │      eid = selected[e]
    │      gate = matvec(ffn_gate_exps[eid], xq)
    │      up   = matvec(ffn_up_exps[eid], xq)
    │      mid  = swiglu(gate, up)
    │      down = matvec(ffn_down_exps[eid], mid)
    │      out += expert_weight[e] × down
    │
    ├──> 共享专家:
    │    gate = matvec(ffn_gate_shexp, x)
    │    up   = matvec(ffn_up_shexp, x)
    │    mid  = swiglu(gate, up)
    │    shexp_out = matvec(ffn_down_shexp, mid)
    │
    └──> 总输出 = 路由专家累加 + 共享专家
         (4096 维)
```

---

## 本章小结

- 路由门控 = 小线性层给 256 个专家打分，用 `sqrt(softplus(logit))` 激活
- Top-K 选择：加了 bias 的分数选 top-6，但权重用原始概率
- 前 3 层用 hash 路由（按 token id 映射），后面用学习路由
- 输入量化一次，6 个专家复用，省 5 次量化
- 专家权重打包存储，支持按需加载
- 负载均衡靠 bias + softplus/sqrt + 共享专家

## 动手实验

1. 在 `ds4.c:10652` 阅读 `layer_router_probs_one`，理解 `sqrt(softplus(x))` 的效果
2. 在 `ds4.c:10675` 阅读 `topk_desc`，手推 [3, 1, 4, 1, 5] 选 top-3 的过程
3. 在 `ds4.c:10761` 阅读 `layer_routed_moe_one`，理解 hash 路由和学习路由的分支
4. 思考：为什么权重用原始 probs 而不是加了 bias 的 selection？

## 下一章预告

第三部分算子篇结束了。从第 14 章开始进入第四部分"推理篇"，我们把所有算子串成完整的一层前向传播。
