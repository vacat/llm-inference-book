# 第 2 章 ds4 项目概览与环境搭建

## 本章导读

上一章我们理解了推理的概念，这一章要把 ds4 跑起来。我们会：

- 了解 ds4 代码库的整体结构
- 在 macOS 上编译 ds4
- 用 `--inspect` 命令查看模型内部结构
- 跑通第一次推理

**前置知识**：第 1 章（推理的基本概念）

---

## 2.1 代码库全景

ds4 的代码组织非常清晰，每个文件职责明确。先看整体：

```
ds4/
├── ds4.c          ← 核心引擎：GGUF 加载 + CPU 算子 + Metal 图驱动 + 分词器
├── ds4.h          ← 公共 API 头文件（engine + session 接口）
├── ds4_metal.m    ← Metal GPU 实现（Apple Silicon）
├── ds4_cuda.cu    ← CUDA GPU 实现（NVIDIA）
├── ds4_rocm.cu    ← ROCm GPU 实现（AMD Strix Halo）
├── ds4_cli.c      ← 命令行交互界面
├── ds4_server.c   ← HTTP 服务器（OpenAI 兼容 API）
├── ds4_agent.c    ← AI Agent（工具调用 + 编程助手）
├── ds4_bench.c    ← 性能基准测试
├── ds4_eval.c     ← 质量评测（检测 logits 漂移）
├── ds4_distributed.c ← 分布式推理（流水线并行）
├── ds4_tp.c       ← 张量并行
├── ds4_kvstore.c  ← KV 缓存持久化（检查点）
├── ds4_ssd.c      ← SSD 流式加载接口
├── ds4_gpu.h      ← GPU 抽象层（Metal/CUDA 统一接口）
├── metal/         ← Metal 着色器源码（.metal 文件）
│   ├── flash_attn.metal  ← Flash Attention 内核
│   ├── moe.metal         ← MoE 路由 + 专家计算内核
│   ├── dense.metal       ← 稠密矩阵乘法内核
│   ├── norm.metal        ← RMSNorm 内核
│   ├── dsv4_rope.metal   ← RoPE 位置编码内核
│   └── ...
├── rocm/          ← ROCm HIP 内核源码（.cuh 文件）
├── gguf-tools/    ← GGUF 生成、量化、质量测试工具
├── tests/         ← 测试代码
├── speed-bench/   ← 性能测试用的文本素材
├── Makefile       ← 构建脚本
└── README.md      ← 项目文档
```

### 核心文件关系图

```
           ds4.h (公共 API)
              │
    ┌─────────┼─────────────┐
    │         │             │
 ds4_cli.c  ds4_server.c  ds4_agent.c    ← 上层应用
    │         │             │
    └─────────┼─────────────┘
              │
          ds4.c (核心引擎)                  ← 本书的重点
         ╱    │    ╲
        ╱     │     ╲
ds4_metal.m  CPU  ds4_cuda.cu              ← 计算后端
    │              │
 metal/*.metal   (CUDA 内联在 .cu 中)
```

> **阅读建议**：本书大部分时间在 `ds4.c` 中。它是"垂直一体化"的--一个文件包含了从文件加载到算子实现的所有逻辑。虽然 6 万多行很长，但每个函数都做了实事，注释也很清楚。

---

## 2.2 三种计算后端

ds4 支持三种计算后端，由编译时选择：

| 后端 | 平台 | 文件 | 特点 |
|------|------|------|------|
| **Metal** | Apple Silicon (M 系列) | `ds4_metal.m` + `metal/*.metal` | 主力路径，96GB+ Mac 上最快 |
| **CUDA** | NVIDIA GPU | `ds4_cuda.cu` | 多卡支持，DGX Spark 优化 |
| **CPU** | 任意 | `ds4.c` 内的参考实现 | 最慢但最易读，适合学习 |

在 macOS 上默认编译 Metal 路径。本书的代码讲解以 **CPU 参考实现为主**（因为最易读），辅以 Metal GPU 实现的对比。

> **为什么先读 CPU 代码？** GPU 代码为了性能做了大量优化（线程调度、共享内存、向量化），可读性较差。CPU 参考实现是"教学版"，逻辑清晰，是理解 GPU 路径的蓝本。ds4 的 GPU 路径在功能上与 CPU 路径一一对应。

---

## 2.3 编译 ds4

### 环境要求

- **macOS + Apple Silicon**（推荐）：Xcode Command Line Tools
- **Linux + NVIDIA**：CUDA Toolkit
- **任意平台 CPU-only**：只需 C 编译器

### 编译步骤（macOS）

```bash
# 1. 克隆代码
git clone https://github.com/antirez/ds4.git
cd ds4

# 2. 编译（默认 Metal 路径）
make
```

编译成功后会生成 5 个可执行文件：

```
ds4        ← 交互式命令行推理
ds4-server ← HTTP 服务器
ds4-bench  ← 性能基准测试
ds4-eval   ← 质量评测
ds4-agent  ← AI 编程助手
```

### 编译 CPU-only 版本（用于学习）

如果你想读 CPU 参考实现的代码并对照运行：

```bash
make cpu
```

这会编译不含 GPU 依赖的版本，速度慢但逻辑最清晰。

### Makefile 中的关键配置

打开 `Makefile`，前几行是编译选项（`Makefile:1-20`）：

```makefile
CC ?= cc
CFLAGS ?= -O3 -ffast-math -g -mcpu=native -Wall -Wextra -std=c99
```

- `-O3`：最高优化级别
- `-ffast-math`：允许数学优化（可能牺牲精度，用 `DS4_METAL_MATH_SAFE=1` 可关闭）
- `-mcpu=native`：使用当前 CPU 的全部指令集

> **注意**：在 macOS 上，`make` 会自动检测到是 Apple 平台，使用 Metal 路径编译。核心目标文件包括 `ds4.o`、`ds4_metal.o`、`ds4_distributed.o`、`ds4_tp.o`、`ds4_ssd.o`、`ds4_layer_pack.o`（见 `Makefile:56-58`）。

---

## 2.4 下载模型

ds4 不自带模型文件，需要单独下载。DeepSeek V4 Flash 的量化版约 86GB。

```bash
# 使用项目提供的下载脚本
./download_model.sh
```

这个脚本（`download_model.sh:1-30`）会从 HuggingFace 下载 GGUF 格式的模型文件。下载的文件包括：

- **主模型**：`DeepSeek-V4-Flash-IQ2XXS-...-imatrix.gguf`（约 86GB）
- **MTP 模型**：用于推测解码的小模型（可选）
- **DSpark 支持模型**：用于增强推测解码（可选）

如果你没有 86GB 内存，ds4 支持 **SSD 流式加载**，可以从硬盘按需读取模型权重。

> **本书约定**：后续章节假设模型文件已下载，路径为 `ds4flash.gguf`（通常是个符号链接，指向实际 GGUF 文件）。

---

## 2.5 第一次运行：--inspect

编译好之后，第一件事不是跑推理，而是**窥探模型内部**：

```bash
./ds4 --inspect
```

`--inspect` 模式只加载模型元数据，不执行推理。它会打印出模型的完整架构信息。在 `ds4_cli.c:2010` 可以看到这个参数的解析：

```c
} else if (!strcmp(arg, "--inspect")) {
    c.inspect = true;
```

然后在 `ds4_cli.c:2073`，inspect 标志被传给引擎选项：

```c
cfg.engine.inspect_only = cfg.inspect;
```

`inspect_only = true` 时，引擎只读 GGUF 头部、验证张量布局、打印架构参数，然后退出。不分配推理用的 KV 缓存，不编译 GPU 着色器。

### 你会看到什么

`--inspect` 的输出大致长这样（简化）：

```
ds4: model name: DeepSeek V4 Flash
ds4: architecture: DeepSeek V4
ds4: n_layer = 43
ds4: n_embd = 4096
ds4: n_vocab = 129280
ds4: n_head = 64
ds4: n_head_kv = 1
ds4: n_head_dim = 512
ds4: n_expert = 256
ds4: n_expert_used = 6
ds4: n_expert_shared = 1
ds4: n_swa = 128
ds4: ...
ds4: tensors: NNN tensors loaded
ds4: model size: XX GB
```

这些数字就是模型的"身份证"。我们在第 5 章会逐个解释每个参数的含义。现在你只需要知道：

- **43 层**：模型有 43 个 Transformer 层
- **256 个专家**：每层有 256 个"专家网络"，每次只用 6 个（这就是 MoE）
- **64 个注意力头，但只有 1 个 KV 头**：这是 GQA（分组查询注意力），大幅省内存

---

## 2.6 跑通第一次推理

确认 `--inspect` 正常后，来跑一次真正的推理：

```bash
./ds4 -p "用三句话解释什么是大模型推理"
```

- `-p`：指定 prompt（提示词），生成后退出（不进入交互模式）

你会看到模型逐字输出文字，同时可能有计时信息。这就是一次完整的推理：Prefill（处理你的输入）→ Decode（逐字生成回答）。

### 观察推理日志

加一些环境变量可以看到更多信息：

```bash
# 显示每个 token 的生成时间（注意：必须加 --temp 0 走 argmax 路径才生效）
DS4_TOKEN_TIMING=1 ./ds4 -p "你好" -n 20 --temp 0

# 限制生成 20 个 token（-n 参数）
# 观察首字延迟和后续每个 token 的耗时差异
```

你会发现：
- 第一个 token 最慢（包含 Prefill 时间）
- 后续 token 相对快且稳定（纯 Decode 时间）

这正好印证了第 1 章讲的"两阶段"理论。

---

## 2.7 五个可执行文件各做什么

| 命令 | 源文件 | 用途 |
|------|--------|------|
| `ds4` | `ds4_cli.c` | 交互式命令行推理，支持 `-p` 单次、`-i` 交互 |
| `ds4-server` | `ds4_server.c` | HTTP 服务器，提供 OpenAI 兼容 API，支持多用户 |
| `ds4-bench` | `ds4_bench.c` | 性能测试，测量 prefill/decode 吞吐 |
| `ds4-eval` | `ds4_eval.c` | 质量测试，确保优化没破坏输出正确性 |
| `ds4-agent` | `ds4_agent.c` | AI 编程助手，支持工具调用 |

日常学习和开发用 `ds4`（命令行）就够了。部署给别人用用 `ds4-server`。

---

## 2.8 实例锁：一个要注意的细节

ds4 启动时会在 `/tmp/ds4.lock` 创建一个文件锁，保证同一时刻只有一个 ds4 进程使用模型。这是因为模型占用的内存/显存很大，多个实例同时跑会导致内存不足。

如果你之前异常退出（比如 Ctrl+C 后没清理），锁可能残留，导致新的 ds4 无法启动。遇到这种情况：

```bash
# 检查是否有残留锁
ls -la /tmp/ds4.lock

# 如果确认没有其他 ds4 在运行，可以删除
rm /tmp/ds4.lock
```

> **原理**：文件锁用的是 `flock()`，锁的持有者是文件描述符（fd）而非进程 ID。进程正常退出时操作系统会自动释放 fd 从而释放锁；但如果进程被 `kill -9` 或崩溃，fd 可能残留。详见第 6 章模型加载部分。

---

## 本章小结

- ds4 代码库结构清晰：`ds4.c` 是核心，辅以 Metal/CUDA 后端和上层应用
- 三种后端：Metal（Apple）、CUDA（NVIDIA）、CPU（学习用）
- `make` 编译，`--inspect` 查看模型，`-p` 跑推理
- 五个可执行文件各有分工：推理、服务、测试、评测、Agent
- 注意实例锁机制，避免多实例内存冲突

## 动手实验

1. 编译 ds4（`make`），运行 `./ds4 --inspect`，记下模型的 `n_layer`、`n_expert`、`n_head` 三个值
2. 运行 `DS4_TOKEN_TIMING=1 ./ds4 -p "1+1等于几" -n 10 --temp 0`，观察首字延迟和后续 token 速度的差异（注意：`--temp 0` 让 CLI 走 argmax 路径，`DS4_TOKEN_TIMING` 只在 argmax 路径生效；默认 temperature=1.0 会走 session 路径，该变量不生效）
3. 运行 `./ds4-bench --help`（如果有的话）或查看 `ds4_bench.c:554` 的 main 函数，了解基准测试支持哪些参数

## 下一章预告

第 3 章，我们会把"一次推理从头到尾发生了什么"用代码主线串起来。从你输入文字到看到输出，每一步对应哪个函数，画出完整的推理流程图。
