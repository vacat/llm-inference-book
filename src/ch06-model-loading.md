# 第 6 章 模型加载与内存映射

## 本章导读

第 4 章讲了 GGUF 文件的物理结构，第 5 章讲了模型的逻辑架构。这章把它们连起来：ds4 怎么从 GGUF 文件里的 `blk.0.attn_q_a.weight` 找到代码中的 `layer[0].attn_q_a` 指针。

读完本章你会理解：
- 张量名到代码指针的绑定过程
- 权重验证（为什么加载时要检查每个张量的维度和类型）
- mmap 的延迟加载特性
- 实例锁的作用与陷阱

**前置知识**：第 4-5 章

---

## 6.1 从文件到指针：model_find_tensor

加载模型时，ds4 需要把 GGUF 文件中的张量"绑定"到 `ds4_weights` 结构体。核心函数是 `model_find_tensor()`（`ds4.c:2767`）：

```c
static ds4_tensor *model_find_tensor(const ds4_model *m, const char *name) {
    const size_t len = strlen(name);
    for (uint64_t i = 0; i < m->n_tensors; i++) {
        if (m->tensors[i].name.len == len &&
            memcmp(m->tensors[i].name.ptr, name, len) == 0) {
            return &m->tensors[i];
        }
    }
    return NULL;  // 没找到
}
```

逻辑很简单：**遍历所有张量描述，按名字匹配**。比如找 `"blk.0.attn_q_a.weight"`，就在张量目录里找到名字相同的那个，返回它的指针。

> **性能注意**：这是线性查找。但模型加载只做一次，且张量数量（几百个）不大，所以不是瓶颈。

---

## 6.2 权重绑定过程

加载时，ds4 为每一层调用绑定函数，把文件中的张量指针填入 `ds4_layer_weights` 结构。大致流程：

```
ds4_engine_open_internal()
  │
  ├─ model_open()          ← mmap 文件，解析头部和目录
  │
  ├─ 绑定全局权重:
  │    w->token_embd  = model_find_tensor(m, "token_embd.weight")
  │    w->output_norm = model_find_tensor(m, "output_norm.weight")
  │    w->output      = model_find_tensor(m, "output.weight")
  │
  ├─ 循环 43 层，绑定每层权重:
  │    for (il = 0; il < 43; il++) {
  │        char name[128];
  │        sprintf(name, "blk.%d.attn_norm.weight", il);
  │        l->attn_norm = model_find_tensor(m, name);
  │
  │        sprintf(name, "blk.%d.attn_q_a.weight", il);
  │        l->attn_q_a = model_find_tensor(m, name);
  │        // ... 几十个张量 ...
  │    }
  │
  └─ 验证所有张量的维度和类型
```

绑定后，`w->layer[0].attn_q_a` 就是一个指向 `ds4_tensor` 的指针。访问数据时用 `tensor_data(m, w->layer[0].attn_q_a)`，返回 mmap 区域中对应位置的地址。

> **关键点**：绑定只是存了指针，没有复制数据。实际数据还在 mmap 映射的虚拟内存里，等到推理时访问到才由操作系统加载到物理内存。

---

## 6.3 权重验证：为什么不能跳过

绑定之后，ds4 会验证每个张量的**维度和类型**是否正确。函数是 `tensor_expect_layout()`（`ds4.c:4228`）。

验证逻辑大致是：

```c
static void tensor_expect_layout(const ds4_tensor *t, uint32_t expected_type,
                                 int ndim, uint64_t d0, uint64_t d1, uint64_t d2) {
    if (!t) ds4_die("tensor not found");
    if (t->type != expected_type) ds4_die("wrong quant type");
    if (t->ndim != ndim) ds4_die("wrong number of dimensions");
    if (t->dim[0] != d0) ds4_die("wrong dimension 0");
    // ...
}
```

调用示例（`ds4.c:4895` 附近）：

```c
// 验证 output_norm 是 F32 类型、1 维、大小为 n_embd
tensor_expect_layout(w->output_norm, DS4_TENSOR_F32, 1, DS4_N_EMBD, 0, 0);

// 验证每层的 attn_norm
tensor_expect_layout(l->attn_norm, DS4_TENSOR_F32, 1, DS4_N_EMBD, 0, 0);

// 验证 Q 压缩矩阵
tensor_expect_layout(l->attn_q_a, ..., 2, DS4_N_LORA_Q, DS4_N_EMBD, 0);

// 验证 MoE 路由门控
tensor_expect_layout(l->ffn_gate_inp, DS4_TENSOR_F32, 2, DS4_N_EMBD, DS4_N_EXPERT, 0);
```

### 为什么要验证？

1. **提前失败**：如果模型文件和代码不匹配（比如你用一个 Llama 的 GGUF 来跑 ds4），立即报错，而不是推理时产生莫名其妙的 segfault
2. **防止精度问题**：如果路由门控权重不是 F32 而是 Q8_0，路由计算会出问题
3. **调试辅助**：报错信息告诉你哪个张量不对，方便定位问题

> **ds4 的设计哲学**：它只支持 DeepSeek V4 / GLM 5.2 的特定布局，不做通用 GGUF 运行器。所以验证可以很严格--不符合就直接退出。这是"专精"带来的好处：代码更简单、更安全。

---

## 6.4 mmap 延迟加载的原理

这是 ds4 能在内存不够时跑大模型的关键。让我们深入理解 mmap 的工作方式。

### 传统读取 vs mmap

```
传统读取:
  open() -> read(buf, size) -> 数据从磁盘复制到 buf
  ↑ 一次性把数据读到内存
  ↑ 86GB 文件需要 86GB 内存

mmap 读取:
  open() -> mmap(map, size) -> 建立虚拟内存映射
  ↑ 没有实际读取数据！
  ↑ 访问 map[offset] 时才触发缺页中断
  ↑ 操作系统从磁盘读取对应页面（通常 4KB）
```

### 缺页中断（Page Fault）

```
程序访问 map[1000000]
    │
    ▼
CPU 检查：这个虚拟地址对应的物理页面在内存里吗？
    │
    ├─ 在 -> 直接访问（快，纳秒级）
    │
    └─ 不在 -> 触发缺页中断
                 │
                 ▼
        操作系统从磁盘读取对应的 4KB 页面
                 │
                 ▼
        映射到物理内存，恢复程序执行
```

**关键点**：只有被访问到的数据才会真正占用物理内存。如果推理只用到 256 个专家中的 6 个，那只有这 6 个专家的权重页面会被加载。

### 预读（Prefetch）

`model_open()` 最后有一行（`ds4.c:2469`）：

```c
if (!metal_mapping && prefetch_cpu) model_prefetch_cpu_mapping(m);
```

`model_prefetch_cpu_mapping` 会建议操作系统预读整个文件。这对 CPU 路径有用（提前把数据读进来），但对 Metal 路径不调用（因为 Metal 路径由 GPU 按需访问）。

> **SSD 流式模式**（第 21 章）是 mmap 延迟加载的进阶版：它不仅依赖操作系统的页面调度，还主动控制哪些专家在内存、哪些在 SSD，实现更精细的内存管理。

---

## 6.5 实例锁

`ds4_engine_open_internal()` 在加载模型前会调用 `ds4_acquire_instance_lock()`（`ds4.c:55524` 附近）。这个锁保证同一时刻只有一个 ds4 进程在运行。

### 为什么需要实例锁

```
没有锁:
  进程 A 加载 86GB 模型 -> 占用 86GB 内存
  进程 B 也加载 86GB 模型 -> 再占 86GB 内存
  → 总共 172GB，内存溢出，系统崩溃

有锁:
  进程 A 启动 -> 获取锁
  进程 B 启动 -> 获取锁失败 -> 报错退出
  → 内存安全
```

### 锁的实现

锁用的是 `flock()`（文件锁），锁文件在 `/tmp/ds4.lock`：

```c
// 简化逻辑
int fd = open("/tmp/ds4.lock", O_CREAT | O_RDWR, 0644);
if (flock(fd, LOCK_EX | LOCK_NB) == -1) {
    // 获取锁失败，说明已有实例在运行
    fprintf(stderr, "ds4: another instance is already running\n");
    exit(1);
}
```

### 锁残留问题

正常情况下，进程退出时操作系统自动释放 `flock`。但如果进程被 `kill -9` 或崩溃，可能出现锁残留。

**但注意**：`flock` 的持有者是**文件描述符（fd）**，不是进程。进程崩溃后，fd 会被操作系统关闭，锁也应该自动释放。真正的残留问题通常是因为有子进程继承了 fd，或者 `flock` 实现的某些边缘情况。

如果遇到锁问题，可以手动清理：

```bash
# 检查是否真的有 ds4 在运行
ps aux | grep ds4

# 如果没有，删除锁文件
rm /tmp/ds4.lock
```

---

## 6.6 模型加载的完整流程图

```
ds4_engine_open() [ds4.c:55454]
    │
    ▼
ds4_engine_open_internal() [ds4.c:55459]
    │
    ├─ 1. 分配 engine 结构体，复制配置
    │
    ├─ 2. ds4_acquire_instance_lock()
    │      └─ flock(/tmp/ds4.lock)
    │
    ├─ 3. model_open() [ds4.c:2428]
    │      ├─ open() + fstat()
    │      ├─ mmap() → 整个文件映射到虚拟内存
    │      ├─ 读文件头 (magic, version, n_tensors, n_kv)
    │      ├─ parse_metadata() → 架构参数
    │      ├─ parse_tensors() → 张量目录
    │      └─ 可选: prefetch
    │
    ├─ 4. 绑定权重指针
    │      ├─ w->token_embd = find_tensor("token_embd.weight")
    │      ├─ w->output = find_tensor("output.weight")
    │      └─ for 43 layers: 绑定每层 ~40 个张量
    │
    ├─ 5. 验证张量布局
    │      └─ tensor_expect_layout() 检查每个张量的维度和类型
    │
    ├─ 6. 初始化 GPU
    │      ├─ Metal: 编译 metal/*.metal 着色器
    │      └─ CUDA: 初始化 CUDA 上下文
    │
    ├─ 7. 加载支持模型（MTP/DSpark，可选）
    │
    └─ 8. 返回 engine 指针
         此时：权重数据还在磁盘上，等推理时按需加载
```

---

## 本章小结

- `model_find_tensor()` 按名字在张量目录中查找，返回指针
- 权重绑定只存指针，不复制数据（mmap 延迟加载）
- `tensor_expect_layout()` 验证每个张量的维度和类型，提前发现不匹配
- mmap 通过缺页中断实现按需加载，只有访问到的数据才占物理内存
- 实例锁防止多进程同时加载大模型导致内存溢出

## 动手实验

1. 在 `ds4.c:2767` 阅读 `model_find_tensor`，理解线性查找逻辑
2. 在 `ds4.c:4895` 附近查看权重验证代码，数一数每层验证了多少个张量
3. 运行 `./ds4 --inspect`，观察加载过程中是否有"按需加载"的迹象（首字节时间 vs 全量读取时间）

## 下一章预告

第 7 章，我们看分词器：你的输入文字是怎么变成 token id 的？BPE 算法的贪心合并过程是怎样的？
