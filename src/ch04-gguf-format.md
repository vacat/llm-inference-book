# 第 4 章 GGUF 模型文件格式

## 本章导读

86GB 的模型文件里到底装了什么？这章我们打开 GGUF 文件的"黑盒"，看看它的二进制结构。你会理解：

- GGUF 文件的布局：头部、元数据、张量目录、数据区
- 每种量化类型的含义和压缩比
- ds4 如何用 mmap 高效加载大文件

**前置知识**：第 2-3 章

---

## 4.1 为什么有 GGUF

大模型的本质是一堆**数字（权重）**。训练好的模型需要存到硬盘上，但直接存原始浮点数太大了。比如 DeepSeek V4 Flash 有约 570 亿参数，如果用 32 位浮点数（F32）存储，需要 228GB。

GGUF（GPT-Generated Unified Format）是 llama.cpp 社区创建的模型文件格式，核心思想是：

1. **用量化压缩**：把 32 位浮点数压缩成 2 位、4 位、8 位，大幅减小体积
2. **统一格式**：一个文件包含所有信息（架构参数 + 权重 + 分词器），不依赖外部文件
3. **mmap 友好**：文件布局设计为可以直接内存映射，按需加载

> **ds4 与 GGUF 的关系**：ds4 不链接 llama.cpp/GGML，但兼容 GGUF 格式。文件头解析、量化块布局等逻辑参考了 llama.cpp（`ds4.c:1-10` 的文件头注释明确说明了这一点）。

---

## 4.2 GGUF 文件布局

一个 GGUF 文件从字节层面看是这样的：

```
┌─────────────────────────────────────────────────┐  偏移 0
│  Magic: "GGUF" (4 字节, 0x46554747)              │
├─────────────────────────────────────────────────┤  偏移 4
│  Version: 3 (4 字节)                              │
├─────────────────────────────────────────────────┤  偏移 8
│  n_tensors: 张量数量 (8 字节)                      │
├─────────────────────────────────────────────────┤  偏移 16
│  n_kv: 元数据键值对数量 (8 字节)                    │
├─────────────────────────────────────────────────┤  偏移 24
│                                                   │
│  元数据区 (Metadata)                               │
│  ┌───────────────────────────────────────────┐   │
│  │ key: "general.name" = "DeepSeek V4 Flash" │   │
│  │ key: "llm.n_layer" = 43                   │   │
│  │ key: "llm.n_embd" = 4096                  │   │
│  │ key: "tokenizer.ggml.tokens" = [...]      │   │
│  │ ... 几十个键值对 ...                        │   │
│  └───────────────────────────────────────────┘   │
├─────────────────────────────────────────────────┤
│                                                   │
│  张量目录 (Tensor Directory)                       │
│  ┌───────────────────────────────────────────┐   │
│  │ tensor[0]: name="token_embd.weight"       │   │
│  │   ndim=2, dim=[4096, 129280]              │   │
│  │   type=Q8_0, offset=数据区偏移             │   │
│  │ tensor[1]: name="blk.0.attn_norm.weight"  │   │
│  │   ...                                     │   │
│  │ ... 几百个张量 ...                          │   │
│  └───────────────────────────────────────────┘   │
├─────────────────────────────────────────────────┤
│                                                   │
│  数据区 (Tensor Data)                              │
│  ┌───────────────────────────────────────────┐   │
│  │ [token_embd.weight 的实际字节数据]          │   │
│  │ [blk.0.attn_norm.weight 的数据]            │   │
│  │ [blk.0.attn_q_a.weight 的数据]             │   │
│  │ ... 所有张量的数据按顺序排列 ...            │   │
│  │ ... 这部分占了文件的 99%+ ...              │   │
│  └───────────────────────────────────────────┘   │
└─────────────────────────────────────────────────┘
```

关键点：**元数据和张量目录很小（几百 KB），真正的体积全在数据区**。ds4 只需要读前两部分就能知道模型的一切信息，不需要扫描整个文件。

---

## 4.3 源码解读：model_open

ds4 解析 GGUF 的入口是 `model_open()`（`ds4.c:2428`）。我们逐段看：

### 第 1 步：打开文件并 mmap

```c
// ds4.c:2430-2440
int fd = open(path, O_RDONLY);
struct stat st;
fstat(fd, &st);

// mmap：把整个文件映射到虚拟内存
const int mmap_flags = metal_mapping ? MAP_SHARED : MAP_PRIVATE;
void *map = mmap(NULL, st.st_size, PROT_READ, mmap_flags, fd, 0);

m->fd = fd;
m->map = map;
m->size = st.st_size;
```

**mmap 是什么？** 它告诉操作系统："请把这个文件的内容映射到我的进程内存空间。" 之后你就可以像访问数组一样访问文件内容，操作系统负责在背后按需读取磁盘。

```
没有 mmap:                        有 mmap:
  读文件 -> malloc -> memcpy       文件直接映射到虚拟内存
  ↑ 整个文件复制到内存              ↑ 只访问到的页面才加载到物理内存
  ↑ 内存不够就失败                  ↑ 内存不够时用 SSD 兜底
```

> **Metal vs CPU 的区别**（`ds4.c:2437`）：Metal 路径用 `MAP_SHARED`，因为 Metal 需要把映射的内存直接包装成 GPU buffer（零拷贝）。CPU 路径用 `MAP_PRIVATE`，避免 Darwin 内核的一个 VM bug。

### 第 2 步：读文件头

```c
// ds4.c:2456-2465
ds4_cursor c = cursor_at(m, 0);   // 创建一个读取游标
uint32_t magic;
cursor_u32(&c, &magic);            // 读 4 字节: magic
if (magic != DS4_GGUF_MAGIC)       // 检查是不是 "GGUF"
    ds4_die("model is not a GGUF file");

cursor_u32(&c, &m->version);       // 读 4 字节: 版本号
cursor_u64(&c, &m->n_tensors);     // 读 8 字节: 张量数量
cursor_u64(&c, &m->n_kv);          // 读 8 字节: 元数据数量

if (m->version != 3)
    ds4_die("only GGUF v3 is supported");
```

`DS4_GGUF_MAGIC` 定义在 `ds4.c:1035`：

```c
#define DS4_GGUF_MAGIC 0x46554747u /* "GGUF", little endian. */
```

`0x46554747` 在小端序下就是 ASCII 字符 `G` `G` `U` `F`。

### 第 3 步：解析元数据和张量目录

```c
// ds4.c:2467-2468
parse_metadata(m, &c);   // 读 n_kv 个键值对
parse_tensors(m, &c);    // 读 n_tensors 个张量描述
```

`parse_metadata()`（`ds4.c:2342`）读取所有元数据键值对，比如 `n_layer=43`、`n_embd=4096`、词表等。`parse_tensors()`（`ds4.c:2372`）读取每个张量的名字、维度、类型和在文件中的偏移量。

解析完后，ds4 知道了：
- 模型有哪些张量（名字 + 维度 + 类型）
- 每个张量的数据在文件的哪个位置
- 但**还没有读取任何张量的实际数据**--那要等推理时按需读取

---

## 4.4 张量的数据结构

每个张量用 `ds4_tensor` 结构描述（`ds4.c:2057-2069`）：

```c
typedef struct {
    ds4_str name;          // 张量名，如 "blk.0.attn_q_a.weight"
    uint32_t ndim;         // 维度数（1 或 2）
    uint64_t dim[DS4_MAX_DIMS];  // 各维度大小
    uint32_t type;         // 量化类型（F32/Q8_0/Q2_K/IQ2_XXS 等）
    uint64_t rel_offset;   // 在数据区中的相对偏移
    uint64_t abs_offset;   // 在文件中的绝对偏移
    uint64_t elements;     // 元素总数
    uint64_t bytes;        // 占用字节数
} ds4_tensor;
```

张量名遵循约定：`blk.{层号}.{组件名}.weight`。比如：

| 张量名 | 含义 |
|--------|------|
| `token_embd.weight` | 词嵌入表 |
| `blk.0.attn_norm.weight` | 第 0 层的注意力归一化权重 |
| `blk.0.attn_q_a.weight` | 第 0 层的 Q 投影低秩矩阵 A |
| `blk.0.attn_q_b.weight` | 第 0 层的 Q 投影低秩矩阵 B |
| `blk.0.moe_gate.weight` | 第 0 层的 MoE 路由门控 |
| `blk.0.experts.{i}.weight` | 第 0 层第 i 个专家的权重 |
| `output_norm.weight` | 最终归一化 |
| `output.weight` | 输出头（映射到词表） |

---

## 4.5 量化类型详解

GGUF 支持多种量化类型。ds4 用到的几种定义在 `ds4.c:2041-2051`：

```c
enum {
    DS4_TENSOR_F32     = 0,   // 32 位浮点
    DS4_TENSOR_F16     = 1,   // 16 位浮点
    DS4_TENSOR_Q4_0    = 2,   // 4 位量化
    DS4_TENSOR_Q8_0    = 8,   // 8 位量化
    DS4_TENSOR_Q2_K    = 10,  // 2 位 K-quant
    DS4_TENSOR_Q4_K    = 12,  // 4 位 K-quant
    DS4_TENSOR_Q5_K    = 13,  // 5 位 K-quant
    DS4_TENSOR_Q6_K    = 14,  // 6 位 K-quant
    DS4_TENSOR_Q8_K    = 15,  // 8 位 K-quant
    DS4_TENSOR_IQ2_XXS = 16,  // 2 位重要性矩阵量化
    DS4_TENSOR_I32     = 26,  // 32 位整数
};
```

### 量化类型参数表

`gguf_types[]` 数组（`ds4.c:2008`）定义了每种类型的基本参数：

```c
static const gguf_type_info gguf_types[] = {
    [0]  = {"f32",      1,   4},  // 每元素 4 字节
    [1]  = {"f16",      1,   2},  // 每元素 2 字节
    [8]  = {"q8_0",    32,  34},  // 32 个元素占 34 字节 ≈ 8.5 bit/元素
    [10] = {"q2_k",   256,  84},  // 256 个元素占 84 字节 ≈ 2.6 bit/元素
    [12] = {"q4_k",   256, 144},  // 256 个元素占 144 字节 ≈ 4.5 bit/元素
    [16] = {"iq2_xxs",256,  66},  // 256 个元素占 66 字节 ≈ 2.06 bit/元素
    // ...
};
```

三个字段含义：`{名称, block_elems（每块元素数）, block_bytes（每块字节数）}`。

### 压缩比对比

以 570 亿参数的 DeepSeek V4 Flash 为例：

| 类型 | 每参数比特 | 理论大小 | 压缩比 |
|------|-----------|---------|--------|
| F32 | 32 bit | 228 GB | 1x |
| F16 | 16 bit | 114 GB | 2x |
| Q8_0 | 8.5 bit | 60 GB | 3.8x |
| Q4_K | 4.5 bit | 32 GB | 7.1x |
| Q2_K | 2.6 bit | 18 GB | 12.3x |
| IQ2_XXS | 2.06 bit | 15 GB | 15.5x |

> **DeepSeek V4 Flash 用的是混合量化**：文件名 `IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8` 表示：
> - 路由专家权重：IQ2_XXS（最激进压缩，~2 bit）
> - 下投影权重：Q2_K
> - 注意力投影：Q8_0（精度要求高，用 8 bit）
> - 共享专家：Q8_0
> - 输出头：Q8_0
>
> 混合量化的思路是：**对精度敏感的用高精度，对精度不敏感的用低精度**，在体积和质量间取平衡。量化原理在第 18 章详讲。

---

## 4.6 怎么访问张量数据

有了张量描述后，访问数据只需要一个指针运算。`tensor_data()` 函数（`ds4.c:3048`）：

```c
static const void *tensor_data(const ds4_model *m, const ds4_tensor *t) {
    return (const uint8_t *)m->map + t->abs_offset;
}
```

就是 `mmap 基地址 + 张量偏移`。因为整个文件已经 mmap 了，所以每个张量的数据"天然就在内存里"（虚拟内存层面），实际物理读取由操作系统的页面调度机制按需完成。

```
m->map (mmap 基地址)
  │
  ├── [文件头 + 元数据 + 张量目录]  ← 启动时已读
  │
  ├── tensor[0].abs_offset ──→ token_embd 的数据
  │                              ↑ 访问时才从磁盘加载
  ├── tensor[1].abs_offset ──→ blk.0.attn_norm 的数据
  │
  ├── ...
  │
  └── tensor[N].abs_offset ──→ output 的数据
```

---

## 4.7 为什么 GGUF 设计成这样

回顾 GGUF 的设计，有三个聪明之处：

### 1. 头部在前，数据在后

元数据和张量目录在文件开头，很小。读完后就知道整个模型的"目录"，不需要扫描 86GB 的数据区。

### 2. 数据区连续排列

所有张量数据按顺序排列在文件尾部，偏移量精确记录。mmap 后通过简单的指针运算就能访问任意张量。

### 3. 量化块对齐

每种量化类型把元素分成固定大小的块（如 Q8_0 每 32 个元素一块，Q2_K 每 256 个一块）。块内独立存储缩放因子，使得：
- 可以随机访问任意位置（不需要从头解码）
- GPU 内核可以按块并行处理
- 压缩和解压都是块级的，简单高效

---

## 本章小结

- GGUF 文件 = 文件头 + 元数据 + 张量目录 + 数据区
- `model_open()` 用 mmap 把整个文件映射到虚拟内存，只解析头部和目录
- 张量数据按需读取，由操作系统的页面调度管理
- ds4 用混合量化：重要权重 Q8_0，非关键权重 IQ2_XXS/Q2_K
- `tensor_data()` 就是指针运算：`mmap 基地址 + 偏移量`

## 动手实验

1. 运行 `./ds4 --inspect`，找到输出中显示的量化类型信息，对照本章的量化表
2. 在 `ds4.c:2428` 阅读 `model_open` 完整函数，理解 mmap 的两种模式（SHARED vs PRIVATE）为什么不同
3. 思考：如果模型文件在 NFS 网络盘上，mmap 会有什么问题？（提示：页面缺失时的网络延迟）

## 下一章预告

第 5 章，我们深入 DeepSeek V4 的架构：43 层、256 个专家、64 个注意力头--这些数字背后是怎样的网络结构？
