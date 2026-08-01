# 附录 A 关键数据结构索引

本附录列出 ds4 中最重要的数据结构及其源码位置，方便快速查找。

## 核心类型

| 结构 | 位置 | 说明 |
|------|------|------|
| `ds4_engine` | `ds4.c:21808` | 引擎：加载好的模型 + GPU 状态 |
| `ds4_session` | `ds4.c:23259` | 会话：一次推理的 KV 缓存 + 状态 |
| `ds4_engine_options` | `ds4.h:95` | 引擎配置参数 |
| `ds4_shape` | `ds4.c:500` | 模型架构参数（层数、维度等） |
| `DS4_SHAPE_FLASH` | `ds4.c:535` | DeepSeek V4 Flash 架构常量 |
| `DS4_SHAPE_PRO` | `ds4.c:580` | DeepSeek V4 Pro 架构常量 |
| `DS4_SHAPE_GLM52` | `ds4.c:610` | GLM 5.2 架构常量 |

## 权重结构

| 结构 | 位置 | 说明 |
|------|------|------|
| `ds4_weights` | `ds4.c:4104` | 全局权重（Embedding + 输出头 + 各层） |
| `ds4_layer_weights` | `ds4.c:4040` | 单层权重（注意力 + MoE + HC） |
| `ds4_mtp_weights` | `ds4.c:4110` | MTP 推测解码模型权重 |
| `ds4_tensor` | `ds4.c:2057` | 张量描述（名字、维度、类型、偏移） |
| `ds4_model` | `ds4.c:2085` | 模型文件（mmap + 元数据 + 张量目录） |

## KV 缓存

| 结构 | 位置 | 说明 |
|------|------|------|
| `ds4_kv_cache` | `ds4.c:12078` | 全模型 KV 缓存（43 层） |
| `ds4_layer_cache` | `ds4.c:12058` | 单层 KV 缓存（原始 + 压缩 + 索引） |

## 关键函数索引

### 模型加载

| 函数 | 位置 | 说明 |
|------|------|------|
| `model_open` | `ds4.c:2428` | 打开 GGUF 文件，mmap 加载 |
| `parse_metadata` | `ds4.c:2342` | 解析 GGUF 元数据 |
| `parse_tensors` | `ds4.c:2372` | 解析张量目录 |
| `model_find_tensor` | `ds4.c:2767` | 按名字查找张量 |
| `tensor_data` | `ds4.c:3048` | 获取张量数据指针 |
| `ds4_engine_open` | `ds4.c:55454` | 引擎打开入口 |
| `ds4_session_create` | `ds4.c:56773` | 创建推理会话 |

### 推理主流程

| 函数 | 位置 | 说明 |
|------|------|------|
| `ds4_session_sync` | `ds4.c:58035` | Prefill 入口 |
| `ds4_session_sync_internal` | `ds4.c:58093` | Prefill 实现（含增量复用） |
| `ds4_session_eval` | `ds4.c:59928` | Decode 入口 |
| `ds4_session_eval_internal` | `ds4.c:59721` | Decode 实现 |
| `ds4_session_eval_speculative_argmax` | `ds4.c:64129` | 推测解码 |
| `prefill_layer_major_cpu` | `ds4.c:13695` | Prefill 层优先策略 |
| `forward_token_raw_swa_cpu_decode_scratch` | `ds4.c:13635` | Decode CPU 主函数 |
| `layer_forward_raw_swa_one` | `ds4.c:13459` | 单层前向（含 profiling） |

### 算子

| 函数 | 位置 | 说明 |
|------|------|------|
| `embed_token_f16` | `ds4.c:6633` | Embedding 查表 (F16) |
| `embed_token_q8_0` | `ds4.c:6651` | Embedding 查表 (Q8_0) |
| `rms_norm_weight` | `ds4.c:6701` | RMSNorm（带权重） |
| `rms_norm_no_weight` | `ds4.c:6692` | RMSNorm（无权重） |
| `head_rms_norm_inplace` | `ds4.c:6710` | 逐头 RMSNorm |
| `rope_tail_ext_inplace` | `ds4.c:10166` | RoPE 旋转位置编码 |
| `layer_q_projection_with_lora_one_decode_scratch` | `ds4.c:10120` | Q 投影（MLA 低秩） |
| `layer_kv_projection_normed_one_decode_scratch` | `ds4.c:10135` | KV 投影 |
| `layer_attention_rows_one` | `ds4.c:10369` | 注意力计算核心 |
| `dot_f32` | `ds4.c:10310` | F32 点积（NEON 优化） |
| `swiglu` | `ds4.c:10494` | SwiGLU 激活 |
| `silu` | `ds4.c:10484` | SiLU 激活 |
| `layer_shared_ffn_one_decode_scratch` | `ds4.c:10542` | 共享专家 FFN |
| `layer_router_probs_one` | `ds4.c:10652` | MoE 路由门控打分 |
| `layer_topk_selected_experts_from_probs` | `ds4.c:10710` | Top-K 专家选择 |
| `layer_routed_moe_one` | `ds4.c:10761` | MoE 路由专家计算 |

### KV 缓存

| 函数 | 位置 | 说明 |
|------|------|------|
| `kv_cache_init` | `ds4.c:12264` | 初始化 KV 缓存 |
| `kv_cache_push_raw` | `ds4.c:12321` | 写入原始 KV（滑动窗口） |
| `kv_cache_push_comp` | `ds4.c:12336` | 写入压缩 KV |

### 量化

| 函数 | 位置 | 说明 |
|------|------|------|
| `dot_q8_0_row_f32_ref` | `ds4.c:7546` | Q8_0 反量化点积 |
| `ds4_vec_dot_q2_K_q8_K` | `ds4.c:3346` | Q2_K 点积 |
| `ds4_vec_dot_iq2_xxs_f32` | `ds4.c:3779` | IQ2_XXS 点积 |
| `matvec_q8_0` | `ds4.c:7534` | Q8_0 矩阵-向量乘法 |

### 分词器

| 函数 | 位置 | 说明 |
|------|------|------|
| `bpe_rank` | `ds4.c:35686` | BPE 合并优先级查询 |
| `bpe_emit_piece` | `ds4.c:35703` | BPE 合并算法 |
| `bpe_tokenize_text` | `ds4.c:36114` | DeepSeek 预分词 + BPE |
| `bpe_tokenize_text_glm4` | `ds4.c:35979` | GLM 预分词 + BPE |
| `vocab_load` | `ds4.c:36206` | 加载词表 |
