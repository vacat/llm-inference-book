# 附录 D 进一步学习路线

## 本书覆盖的知识地图

```
第一部分 基础篇 (第1-3章)
  ✅ 推理概念、ds4 环境、推理全貌

第二部分 模型篇 (第4-7章)
  ✅ GGUF 格式、架构、加载、分词

第三部分 算子篇 (第8-13章)
  ✅ Embedding、RMSNorm、RoPE、Attention、FFN、MoE

第四部分 推理篇 (第14-17章)
  ✅ 层前向、KV 缓存、Prefill/Decode、采样

第五部分 优化篇 (第18-22章)
  ✅ 量化、GPU 内核、Flash Attention、SSD 流式、推测解码

第六部分 工程篇 (第23-26章)
  ✅ 服务化、分布式、张量并行、性能调优
```

## 推荐的进阶学习路线

### 路线 1：深入 GPU 内核优化

学完本书后，如果想写自己的 Metal/CUDA 内核：

1. **精读 `metal/*.metal`**：每个文件对应一个算子，对照 CPU 参考实现理解
2. **学习 Metal Shading Language**：Apple 官方文档 + WWDC 视频
3. **学习 CUDA 编程**：NVIDIA 官方教程 + 《Programming Massively Parallel Processors》
4. **实践**：尝试修改一个 kernel 的 tile 大小，用 `ds4-eval` 验证正确性，用 `ds4-bench` 测性能

### 路线 2：深入量化技术

1. **阅读 `gguf-tools/quants.c`**：量化工具的实现
2. **学习 imatrix**：`gguf-tools/imatrix/README.md`
3. **研究混合量化策略**：不同层用不同量化格式的实验
4. **实践**：用量化工具生成不同精度的模型，对比质量和速度

### 路线 3：深入推理系统设计

1. **对比 llama.cpp**：ds4 参考了 llama.cpp，对比两者的设计差异
2. **学习 vLLM**：了解 PagedAttention 等服务端优化
3. **阅读论文**：
   - Flash Attention (Dao et al.)
   - Mixture of Experts (Shazeer et al.)
   - Speculative Decoding (Leviathan et al.)
   - DeepSeek V2/V3/V4 技术报告
4. **实践**：尝试实现一个简单的推理引擎

### 路线 4：分布式系统

1. **精读 `ds4_distributed.c`**：流水线并行实现
2. **精读 `ds4_tp.c`**：张量并行实现
3. **学习 RDMA 编程**：libibverbs 教程
4. **实践**：用两台机器搭建张量并行

## 关键源码阅读顺序

```
入门 (建立全局观):
  1. ds4.h           ← 公共 API
  2. ds4_cli.c:2058  ← main 函数
  3. ds4.c:55454     ← engine_open
  4. ds4.c:58035     ← session_sync (prefill)
  5. ds4.c:59928     ← session_eval (decode)

进阶 (理解算子):
  6. ds4.c:6633      ← embed_token
  7. ds4.c:6701      ← rms_norm
  8. ds4.c:10166     ← rope
  9. ds4.c:10369     ← attention
  10. ds4.c:10494    ← swiglu
  11. ds4.c:10652    ← moe router
  12. ds4.c:13459    ← layer_forward (串起所有算子)

高级 (优化与工程):
  13. ds4.c:7546     ← quantization
  14. ds4.c:12058    ← kv_cache
  15. ds4.c:13695    ← prefill_layer_major
  16. ds4.c:64129    ← speculative decoding
  17. ds4_metal.m    ← GPU 实现
  18. ds4_server.c   ← HTTP 服务
  19. ds4_tp.c       ← 张量并行
  20. ds4_distributed.c ← 分布式
```

## 实践项目建议

1. **添加 profiling**：给一个还没有计时埋点的算子加上 `DS4_DECODE_PROFILE_DETAIL` 计时
2. **优化 CPU 算子**：用 NEON 向量化重写一个标量循环（如某处 norm/sum），用 `make test` 验证
3. **调参实验**：对比不同 `--mtp-draft`、`--power`、`--quality` 对吞吐的影响，用 CSV 量化
4. **GPU kernel 优化**：读 `metal/*.metal` 找到可改善的 tile 大小或融合点，改后用 `ds4-eval` 验证
5. **实现新功能**：给 ds4-server 加一个简单的 API 端点（如 `/v1/models`）

## 社区与资源

- **ds4 仓库**：https://github.com/antirez/ds4
- **llama.cpp**：https://github.com/ggml-org/llama.cpp（ds4 的参考来源）
- **GGML 文档**：GGUF 格式和量化格式的权威来源
- **DeepSeek 官方**：模型架构和技术报告

## 学习节奏建议

```
入门阶段 (1-2 周):
  读完第一、二部分，跑通 ds4，理解全局架构

核心阶段 (2-4 周):
  精读第三、四部分，逐个算子对照源码，手算关键步骤

进阶阶段 (2-4 周):
  读第五部分，理解优化技术，尝试调参实验

实战阶段 (持续):
  读第六部分，做性能优化，参与开源贡献
```

> **最重要的一条建议**：每个概念都对照源码读一遍，每个算子都手算一个小例子。不要只看不动手。ds4 的代码就是最好的教材。
