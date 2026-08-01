# Summary

## 前言

- [关于本书](./preface.md)

## 第一部分 · 基础篇 — 建立全局观

- [第 1 章 什么是大模型推理](./ch01-what-is-llm-inference.md)
- [第 2 章 ds4 项目概览与环境搭建](./ch02-ds4-overview-setup.md)
- [第 3 章 一次推理的全貌：从输入到输出](./ch03-inference-overview.md)

## 第二部分 · 模型篇 — 理解模型的结构与加载

- [第 4 章 GGUF 模型文件格式](./ch04-gguf-format.md)
- [第 5 章 DeepSeek V4 架构详解](./ch05-architecture.md)
- [第 6 章 模型加载与内存映射](./ch06-model-loading.md)
- [第 7 章 分词器：从文本到 Token](./ch07-tokenizer.md)

## 第三部分 · 算子篇 — 逐个吃透核心算子

- [第 8 章 Embedding：词嵌入查表](./ch08-embedding.md)
- [第 9 章 RMSNorm：归一化](./ch09-rmsnorm.md)
- [第 10 章 RoPE：旋转位置编码](./ch10-rope.md)
- [第 11 章 Attention：注意力机制](./ch11-attention.md)
- [第 12 章 FFN 与 SwiGLU：前馈网络](./ch12-ffn.md)
- [第 13 章 MoE：混合专家路由](./ch13-moe.md)

## 第四部分 · 推理篇 — 把算子串成完整推理

- [第 14 章 单层前向传播全流程](./ch14-layer-forward.md)
- [第 15 章 KV 缓存原理与实现](./ch15-kv-cache.md)
- [第 16 章 Prefill 与 Decode：两条推理路径](./ch16-prefill-decode.md)
- [第 17 章 采样策略：从概率到文字](./ch17-sampling.md)

## 第五部分 · 优化篇 — 让推理更快更省

- [第 18 章 量化技术：用更少比特存权重](./ch18-quantization.md)
- [第 19 章 GPU 内核优化基础](./ch19-gpu-kernels.md)
- [第 20 章 Flash Attention：重新设计注意力](./ch20-flash-attention.md)
- [第 21 章 SSD 流式加载与内存管理](./ch21-ssd-streaming.md)
- [第 22 章 推测解码：猜对了就赚](./ch22-speculative-decoding.md)

## 第六部分 · 工程篇 — 从引擎到产品

- [第 23 章 服务化部署：ds4-server](./ch23-server.md)
- [第 24 章 分布式推理](./ch24-distributed.md)
- [第 25 章 张量并行](./ch25-tensor-parallel.md)
- [第 26 章 性能分析与调优实战](./ch26-profiling.md)

## 附录

- [附录 A 关键数据结构索引](./appendix-a-data-structures.md)
- [附录 B 环境变量速查表](./appendix-b-env-vars.md)
- [附录 C 术语表](./appendix-c-glossary.md)
- [附录 D 进一步学习路线](./appendix-d-learning-path.md)
