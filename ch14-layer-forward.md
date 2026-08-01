# 第 14 章 单层前向传播全流程

## 本章导读

前面六章我们逐个学习了算子：Embedding、RMSNorm、RoPE、Attention、FFN、MoE。这章把它们串起来，看看一个 token 从输入到输出，在**一层**中经历了什么，然后看 43 层是怎么叠在一起的。

读完本章你会拥有完整的"一层推理"心理模型。

**前置知识**：第 8-13 章

---

## 14.1 从 token 到 logits 的完整路径

`forward_token_raw_swa_cpu_decode_scratch()`（`ds4.c:13635`）是 CPU decode 路径的主函数。它做的事情：

```c
static void forward_token_raw_swa_cpu_decode_scratch(
        float *logits, const ds4_model *model, const ds4_weights *weights,
        ds4_kv_cache *cache, int token, uint32_t pos, ...) {

    float *cur = scratch->cur;
    float *next = scratch->next;

    // 1. Token -> Embedding
    embed_token_f16(model, weights, token, scratch->plain);
    hc_from_plain_embedding(cur, scratch->plain, DS4_N_EMBD, DS4_N_HC);

    // 2. 43 层前向传播
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        layer_forward_raw_swa_one(next, model, &weights->layer[il],
                                  &cache->layer[il], cur, il, pos, token, ...);
        // 交换 cur 和 next（双缓冲）
        float *tmp = cur; cur = next; next = tmp;
    }

    // 3. 最终层 -> logits
    if (logits) {
        output_logits_one_decode_scratch(logits, model, weights, cur, scratch);
    }
}
```

### 双缓冲设计

```c
float *cur = scratch->cur;
float *next = scratch->next;

for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
    layer_forward_raw_swa_one(next, ..., cur, ...);
    float *tmp = cur; cur = next; next = tmp;
}
```

用两个缓冲区交替使用：第 0 层读 `cur` 写 `next`，交换后第 1 层读 `cur`（原来的 next）写 `next`（原来的 cur）。避免每层都分配新内存。

### HC 维度

```c
hc_from_plain_embedding(cur, scratch->plain, DS4_N_EMBD, DS4_N_HC);
```

注意输入不是直接用 4096 维的 plain embedding，而是先扩展成 HC 维度（`4096 × 4 = 16384` 维）。这是 HyperConnection 的设计--把输入复制成 4 份，每份对应一个 HC 分组，让不同分组有不同的残差路径。

---

## 14.2 一层内部：layer_forward_raw_swa_one

`layer_forward_raw_swa_one()`（`ds4.c:13459`）是单层前向的核心。它有详细的 profiling 支持（`DS4_DECODE_PROFILE_DETAIL`），我们按执行顺序逐步拆解。

### 阶段 1：HC 预处理

```c
// ds4.c:13496
memcpy(scratch->attn_residual, inp_hc, n_hc * DS4_N_EMBD * sizeof(float));
hc_pre_from_state_one_scratch(model,
    layer->hc_attn_fn, layer->hc_attn_scale, layer->hc_attn_base,
    scratch->attn_residual, scratch->attn_cur, post, comb,
    scratch->hc_flat, false);
```

HC 预处理（HyperConnection Pre）做的是：根据 HC 控制向量（`hc_attn_fn/scale/base`），把输入从 HC 维度（16384）变换回标准维度（4096），同时计算 `post` 和 `comb` 系数用于后处理。

### 阶段 2：注意力归一化 + Q/KV 投影

```c
// ds4.c:13503
layer_attn_norm_one(scratch->attn_norm, model, layer, scratch->attn_cur);

// Q 投影 (第 11 章)
layer_q_projection_with_lora_one_decode_scratch(model, layer,
    scratch->attn_norm, scratch->q, scratch->qr_norm, scratch);

// KV 投影 (第 11 章)
layer_kv_projection_normed_one_decode_scratch(model, layer,
    scratch->attn_norm, scratch->kv, scratch);
```

### 阶段 3：RoPE + KV 缓存

```c
// ds4.c:13514
// RoPE 位置编码 (第 10 章)
rope_tail_layer_inplace(scratch->q, DS4_N_HEAD, DS4_N_HEAD_DIM, DS4_N_ROT, pos, il, false);
rope_tail_layer_inplace(scratch->kv, DS4_N_HEAD_KV, DS4_N_HEAD_DIM, DS4_N_ROT, pos, il, false);

// FP8 量化 KV (第 18 章)
dsv4_fp8_kv_quantize_row_inplace_cpu(scratch->kv, DS4_N_HEAD_DIM, DS4_N_ROT);

// 存入 KV 缓存 (第 15 章)
kv_cache_push_raw(cache, scratch->kv);
```

### 阶段 4：压缩 + 索引器（如果该层有压缩）

```c
// ds4.c:13522
if (ratio != 0) {
    // KV 压缩
    compressor_decode_one_decode_scratch(scratch->comp, model,
        layer->attn_compressor_kv, layer->attn_compressor_gate, ...);
    kv_cache_push_comp(cache->attn_comp_kv, &cache->n_comp, ...);

    // 索引器压缩
    if (ratio == 4) {
        compressor_decode_one_decode_scratch(scratch->index_comp, model,
            layer->indexer_compressor_kv, ...);
        kv_cache_push_comp(cache->index_comp_kv, ...);
    }
}

// 索引器选择
if (ratio == 4) {
    comp_allowed = indexer_allowed_decode_one_decode_scratch(model, layer, ...);
}
```

不是所有层都有压缩（`ratio = 0` 表示不压缩）。有压缩的层会把旧 KV 压缩成更少的槽位，用索引器选择哪些压缩槽位参与注意力计算。

### 阶段 5：注意力计算

```c
// ds4.c:13550
if (ratio != 0) {
    // 混合注意力：原始 KV + 压缩 KV
    layer_attention_mixed_one_decode_scratch(scratch->heads, model, layer,
        scratch->q, cache->raw_kv, cache->n_raw,
        cache->attn_comp_kv, cache->n_comp, comp_allowed, scratch);
} else {
    // 纯原始 KV 注意力
    layer_attention_rows_one(scratch->heads, model, layer,
        scratch->q, cache->raw_kv, cache->n_raw);
}
```

### 阶段 6：反向 RoPE + 输出投影

```c
// ds4.c:13559
// 反向 RoPE（第 10 章）
rope_tail_layer_inplace(scratch->heads, DS4_N_HEAD, DS4_N_HEAD_DIM, DS4_N_ROT, pos, il, true);

// 分组输出投影
layer_grouped_out_one_decode_scratch(scratch->attn_out, model, layer,
    scratch->heads, scratch);
```

### 阶段 7：HC 后处理 + 残差

```c
// ds4.c:13564
hc_post_one(scratch->after_attn_hc, scratch->attn_out, scratch->attn_residual,
            post, comb, DS4_N_EMBD, n_hc);
```

HC 后处理把注意力输出和残差合并，加上阶段 1 计算的 `post` 和 `comb` 系数。

### 阶段 8：FFN / MoE

```c
// ds4.c:13567
layer_ffn_one_decode_scratch(out_hc, model, layer, scratch->after_attn_hc,
    il, token, steering_dirs, steering_ffn_scale, scratch);
```

这一步执行 MoE 路由 + 6 个路由专家 + 1 个共享专家（第 12-13 章），产出 FFN 输出。

### Profiling 输出

如果设置 `DS4_DECODE_PROFILE_DETAIL=1`，每层会打印各阶段耗时：

```
ds4: decode detail layer 0 attn hc=0.12 q=0.34 kv=0.11 rope=0.05 
  compress=0.00 indexer=0.00 attn_rows=2.31 inv_rope=0.04 out=0.28 
  post=0.03 ffn=1.67 total=4.95 ms
```

这帮助定位瓶颈：是注意力（`attn_rows`）还是 MoE（`ffn`）慢。

---

## 14.3 完整一层的流程图

```
输入 inp_hc (16384 维 = 4 × 4096)
    │
    ├─ HC 预处理: inp_hc -> attn_cur (4096 维) + post/comb 系数
    │   保存残差: attn_residual = inp_hc
    │
    ├─ RMSNorm(attn_cur, attn_norm) -> attn_norm (4096 维)
    │
    ├─ Q 投影: attn_norm -> q (32768 维)
    │   压缩(4096→1024) → RMSNorm → 还原(1024→32768) → 逐头RMSNorm
    │
    ├─ KV 投影: attn_norm -> kv (512 维)
    │   压缩(4096→512) → RMSNorm
    │
    ├─ RoPE: q, kv 旋转位置编码
    │
    ├─ FP8 量化 kv
    │
    ├─ 存入 KV 缓存: kv_cache_push_raw(cache, kv)
    │
    ├─ [如果该层有压缩] KV 压缩 + 索引器
    │
    ├─ 注意力计算:
    │   heads = attention(q, raw_kv + compressed_kv)
    │   64 个头 × SWA 窗口 + 压缩槽位
    │
    ├─ 反向 RoPE: heads 旋转回来
    │
    ├─ 输出投影: heads -> attn_out (4096 维)
    │   分组: 8 组 × (512→1024→4096)
    │
    ├─ HC 后处理: attn_out + attn_residual -> after_attn_hc (16384 维)
    │
    ├─ FFN / MoE:
    │   ├─ RMSNorm(after_attn_hc, ffn_norm)
    │   ├─ 路由门控: 256 选 6
    │   ├─ 6 个路由专家 × SwiGLU
    │   ├─ 1 个共享专家 × SwiGLU
    │   └─ 加权求和 -> ffn_out (16384 维)
    │
    └─ 输出 out_hc (16384 维) -> 下一层输入
```

---

## 14.4 输出层：从隐藏状态到 logits

43 层跑完后，最后一层的输出通过 `output_logits_one_decode_scratch()` 转成 logits：

```
最后一层输出 (4096 维)
    │
    ├─ HC 逆变换 -> plain (4096 维)
    ├─ RMSNorm(output_norm)
    └─ matvec(output_weight, norm) -> logits (129280 维)
```

`output_weight` 是输出头权重（`w->output`），把 4096 维映射到 129280 维（词表大小）。这个 logits 就是每个 token 的原始分数，后面经过采样变成下一个 token（第 17 章）。

---

## 14.5 一次 Decode 的完整时间线

```
ds4_session_eval(token, pos)
    │
    ├─ embed_token_f16(token)         ~0.01ms  (查表)
    ├─ hc_from_plain_embedding()      ~0.01ms  (复制扩展)
    │
    ├─ for 43 层:
    │   ├─ HC 预处理                   ~0.1ms
    │   ├─ RMSNorm + Q/KV 投影         ~0.5ms
    │   ├─ RoPE + KV 存储              ~0.1ms
    │   ├─ 注意力计算                   ~2.3ms  (随上下文增长)
    │   ├─ 输出投影                     ~0.3ms
    │   ├─ HC 后处理                    ~0.1ms
    │   └─ MoE (6专家+1共享)           ~1.7ms
    │   ─────────────────────────────────
    │   每层 ~5ms × 43 层 ≈ 215ms (CPU)
    │
    └─ 输出层 -> logits                ~0.5ms

总计: ~215ms/token (CPU), ~15ms/token (Metal GPU)
```

> **注意**：以上是 CPU 路径的粗略估算。Metal GPU 路径会快 10-15 倍，因为矩阵乘法和注意力计算高度并行化。

---

## 本章小结

- `forward_token_raw_swa_cpu_decode_scratch` 是 CPU decode 主函数
- 双缓冲设计：`cur` 和 `next` 交替使用，避免每层分配内存
- 一层内部 8 个阶段：HC预处理 -> RMSNorm -> Q/KV投影 -> RoPE+KV缓存 -> 注意力 -> 反RoPE+输出投影 -> HC后处理 -> MoE
- KV 缓存 + 压缩缓存 + 索引器共同支撑长上下文注意力
- `DS4_DECODE_PROFILE_DETAIL=1` 可以看到每层每阶段的耗时
- 43 层 × 每层 ~5ms ≈ CPU 上每 token 约 215ms

## 动手实验

1. 在 `ds4.c:13459` 阅读 `layer_forward_raw_swa_one`，对照本章流程图确认每个阶段
2. 运行 `DS4_DECODE_PROFILE_DETAIL=1 ./ds4 -p "test" -n 5`，观察每层各阶段耗时
3. 思考：为什么用双缓冲而不是每层 malloc/free？（提示：性能 + 内存碎片）

## 下一章预告

第 15 章，KV 缓存--推理中最重要的数据结构。看看它怎么存储、怎么压缩、怎么支持滑动窗口。
