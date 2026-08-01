# 关于本书

> **副标题**：从原理到工程实践 -- 基于 ds4 (DwarfStar) 源码的逐行解读
>
> **面向读者**：有一定编程基础、想系统掌握大模型推理技术的工程师与技术人员
>
> **配套源码**：[antirez/ds4](https://github.com/antirez/ds4)（DeepSeek V4 推理引擎，约 18 万行 C 代码）

---

## 对应的 ds4 版本

本书基于以下 ds4 版本编写，书中所有源码行号均对应此版本：

| 项目 | 值 |
|------|-----|
| 仓库 | [antirez/ds4](https://github.com/antirez/ds4) |
| 分支 | `main` |
| Commit | [`54b36ed`](https://github.com/antirez/ds4/commit/54b36ed9ba42da31b24f2d1a5feb075c2475dbb1) |
| 提交日期 | 2026-07-28 |
| 提交说明 | Merge pull request #617 from kyuz0/fix/rocm-distributed-glm |

> **注意**：ds4 是一个快速迭代的项目，代码变更频繁。如果你使用的是更新版本的 ds4，部分行号可能已偏移。遇到对不上的情况，请用 `git checkout 54b36ed` 切换到本书对应的版本，或用函数名搜索定位。

```bash
# 获取本书对应的 ds4 版本
git clone https://github.com/antirez/ds4.git
cd ds4
git checkout 54b36ed
```

---

## 如何使用本书

- **读法**：按章节顺序阅读，每章先看"本章导读"建立预期，再读原理讲解，最后对照源码逐行理解。
- **实践**：每章末尾有"动手实验"，建议在 macOS + Apple Silicon 环境下用 ds4 实际操作。
- **源码引用**：书中所有代码引用均标注 `文件名:行号`，可直接在编辑器中跳转。
- **图示**：关键概念配有 ASCII 图表与流程图，帮助建立直觉。

## 全书结构

| 部分 | 章节 | 主题 |
|------|------|------|
| 第一部分 基础篇 | 第 1-3 章 | 推理概念、环境搭建、推理全貌 |
| 第二部分 模型篇 | 第 4-7 章 | GGUF 格式、架构、加载、分词 |
| 第三部分 算子篇 | 第 8-13 章 | Embedding、RMSNorm、RoPE、Attention、FFN、MoE |
| 第四部分 推理篇 | 第 14-17 章 | 层前向、KV 缓存、Prefill/Decode、采样 |
| 第五部分 优化篇 | 第 18-22 章 | 量化、GPU 内核、Flash Attention、SSD 流式、推测解码 |
| 第六部分 工程篇 | 第 23-26 章 | 服务化、分布式、张量并行、性能调优 |
| 附录 | A-D | 数据结构索引、环境变量、术语表、学习路线 |

## 配套源码

本书基于 ds4 仓库：https://github.com/antirez/ds4

```bash
git clone https://github.com/antirez/ds4.git
cd ds4
git checkout 54b36ed    # 切换到本书对应的版本
make
./ds4 --inspect
```
