# 附录 B 环境变量速查表

## 性能分析

| 变量 | 作用 | 示例 |
|------|------|------|
| `DS4_DECODE_PROFILE_DETAIL=1` | 逐层逐算子计时，定位 decode 瓶颈 | `DS4_DECODE_PROFILE_DETAIL=1 ./ds4 -p "test" -n 10` |
| `DS4_TOKEN_TIMING=1` | 每个 token 的生成时间（仅 argmax 路径，需 `--temp 0`） | `DS4_TOKEN_TIMING=1 ./ds4 -p "你好" -n 20 --temp 0` |
| `DS4_MTP_PROBE=1` | 观察 MTP 草稿命中率 | `DS4_MTP_PROBE=1 ./ds4 -p "test" -n 50` |
| `DS4_MTP_TIMING=1` | MTP 推测解码各阶段计时 | `DS4_MTP_TIMING=1 ./ds4 -p "test" -n 50` |
| `DS4_METAL_STREAMING_SELECTED_READAHEAD_PROFILE=1` | 专家预取耗时 profile | 配合 `--ssd-streaming` 使用 |

## 数值诊断

| 变量 | 作用 | 示例 |
|------|------|------|
| `DS4_METAL_MATH_SAFE=1` | 关闭 fastMath，定位数值漂移 | `DS4_METAL_MATH_SAFE=1 ./ds4 -p "test" -n 5` |

## 推测解码控制

| 变量 | 作用 | 说明 |
|------|------|------|
| `DS4_MTP_SPEC_DISABLE` | 禁用 MTP 推测解码 | 设置任意值即禁用 |
| `DS4_GLM_MTP_PROBE` | GLM MTP 命中率探测 | GLM 模型专用 |
| `DS4_DSPARK_SPEC_LOG` | DSpark 推测日志 | 调试 DSpark 用 |
| `DS4_CUDA_GREEDY_TOP1` | CUDA 贪心 top-1 路径 | 默认开启 |
| `DS4_CUDA_TP_OUTPUT_WAYS` | CUDA TP 输出路径数 | 默认 8 |

## 性能调优

| 变量 | 作用 | 说明 |
|------|------|------|
| `DS4_NO_BATCHED_ATTN` | 禁用批量注意力 | 调试用 |
| `DS4_BATCHED_FFN` | 启用批量 FFN | prefill 优化 |
| `DS4_PARALLEL_FFN` | 启用并行 FFN | CPU 多线程 |
| `DS4_NO_SHARED_BATCH_FFN` | 禁用共享专家批量 FFN | 调试用 |
| `DS4_PREFILL_BATCH` | prefill FFN 批大小 | 如 `DS4_PREFILL_BATCH=256` |
| `DS4_CLI_FORCE_SESSION` | 强制使用 session 路径 | TP 验证用 |

## GPU 配置

| 变量 | 作用 | 说明 |
|------|------|------|
| `DS4_GGUF_DIR` | GGUF 模型目录 | `download_model.sh` 使用 |
| `HF_TOKEN` | HuggingFace 下载令牌 | 下载模型时需要 |

## 使用建议

- **定位 decode 慢**：`DS4_DECODE_PROFILE_DETAIL=1` + `DS4_TOKEN_TIMING=1`（后者需配 `--temp 0` 才生效）
- **验证推测解码**：`DS4_MTP_PROBE=1` 看命中率
- **诊断数值问题**：`DS4_METAL_MATH_SAFE=1` 排除 fastMath 影响
- **性能对比**：先跑基线，再开优化，用 `ds4-bench --csv` 对比
