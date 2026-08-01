# 附录 C 术语表

## A

- **Argmax**：选概率最大的 token。贪心解码策略，确定性输出。
- **Attention**（注意力）：Transformer 的核心机制，让每个 token "关注"其他 token，加权聚合信息。
- **Attention Sinks**（注意力池）：可学习的虚拟注意力目标，吸收被丢弃 token 的注意力，防止信息丢失。

## B

- **BPE**（Byte-Pair Encoding，字节对编码）：分词算法，从字节开始贪心合并最常见的相邻对。
- **BOS**（Beginning of Sequence）：序列起始标记 token。

## C

- **Decode**（解码）：逐个生成 token 的推理阶段，每次处理 1 个 token。
- **DSpark**：DeepSeek V4 的增强推测解码技术，用 Markov 预测和多阶段提升草稿质量。

## E

- **Embedding**（嵌入）：把 token id 映射成向量的查表操作。
- **EOS**（End of Sequence）：序列结束标记 token，模型生成到此停止。

## F

- **FFN**（Feed-Forward Network，前馈网络）：Transformer 中注意力之后的非线性变换层。
- **Flash Attention**：重新组织注意力计算，避免写回 n×n 中间矩阵，内存降至线性。
- **FP8**：8 位浮点格式，用于 KV 缓存量化，内存减半。

## G

- **GQA**（Grouped-Query Attention，分组查询注意力）：多个查询头共享同一组 KV，大幅减少 KV 缓存。DeepSeek V4 用 64:1 压缩。

## H

- **HC**（HyperConnection，超连接）：DeepSeek V4 的头通道分组机制，把输入复制成多份走不同残差路径。
- **Hash 路由**：按 token id 的 hash 值直接映射到专家，不依赖输入内容，速度快但精度低。

## I

- **imatrix**（Importance Matrix，重要性矩阵）：训练时统计的权重重要性，用于指导量化精度分配。
- **IQ2_XXS**：2 位重要性矩阵量化，每元素约 2.06 bit，最激进的压缩格式。
- **Indexer**（索引器）：DeepSeek V4 的 KV 缓存压缩机制，从压缩缓存中选择 top-K 行参与注意力。

## K

- **K-quant**：llama.cpp 的量化格式系列（Q2_K, Q4_K, Q5_K, Q6_K, Q8_K），用两级缩放提高精度。
- **KV 缓存**：存储历史 token 的 Key 和 Value 向量，避免重复计算。
- **kq_scale**：注意力分数的缩放因子 1/sqrt(d)，防止点积过大导致 softmax 饱和。

## L

- **logits**：模型输出的原始分数（未经 softmax），每个 token 对应一个分数。
- **LoRA**（Low-Rank Adaptation）：低秩矩阵分解，用两个小矩阵代替大矩阵，减少参数和计算。

## M

- **MLA**（Multi-head Latent Attention，多头潜在注意力）：DeepSeek 的注意力压缩技术，用低秩分解压缩 Q/KV。
- **MoE**（Mixture of Experts，混合专家）：每层有多个专家网络，每次只激活部分专家。
- **MTP**（Multi-Token Prediction）：DeepSeek V4 的推测解码起草器，用小模型猜后续 token。
- **mmap**（内存映射文件）：把文件映射到虚拟内存，按需加载页面。
- **Min-P**：相对概率采样策略，保留概率 ≥ max_prob × min_p 的 token。

## P

- **Prefill**（预填充）：一次性处理输入 prompt 的推理阶段，批量并行计算。
- **Prefill Chunk**：分批 prefill 的批大小，避免超长 prompt 内存不足。

## Q

- **Q8_0**：8 位量化格式，每 32 元素一组，1 个 F16 缩放 + 32 个 int8。
- **Q2_K**：2 位 K-quant，每 256 元素一组，两级缩放，约 2.6 bit/元素。

## R

- **RDMA**（Remote Direct Memory Access）：远程直接内存访问，绕过内核，网络延迟极低。
- **RMSNorm**（Root Mean Square Normalization）：不减均值的归一化，比 LayerNorm 计算量小。
- **RoPE**（Rotary Position Embedding，旋转位置编码）：用旋转向量编码位置，注意力分数自动包含相对位置。

## S

- **Softmax**：把分数转成概率分布（和为 1）的函数。数值稳定版先减最大值。
- **SSD 流式**：把不常用的专家权重放 SSD，用到时才加载，让小内存机器跑大模型。
- **SWA**（Sliding Window Attention，滑动窗口注意力）：只看最近 N 个 token 的注意力，计算量固定。
- **SwiGLU**：Swish 门控线性单元，`silu(gate) × up`，DeepSeek V4 的 FFN 激活函数。
- **Sinkhorn 迭代**：HC 中的矩阵归一化迭代，使矩阵行列和为 1。

## T

- **Tensor Parallelism**（张量并行）：两台机器同时算同一层，各算一半，用 RDMA 交换结果。
- **Token**：模型处理的最小文本单元，由分词器从文本切分得到。
- **Top-K**：选分数最高的 K 个，用于 MoE 专家选择。
- **Top-P**（核采样）：保留累加概率达到 P 的 token，排除长尾。
- **Temperature**：控制概率分布平坦程度的参数，T 越低越确定。

## Y

- **YaRN**：RoPE 的长度外推技术，高频维度外推、低频维度插值，支持超长上下文。
