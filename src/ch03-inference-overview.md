# 第 3 章 一次推理的全貌：从输入到输出

## 本章导读

这是全书最重要的一章之一。我们会把"一次推理从头到尾发生了什么"用代码主线串起来。你输入一句"你好"，到模型吐出回复，中间经过了哪些函数？数据是怎么流动的？

读完本章你会拥有一张**推理流程地图**，后面所有章节都是对这张地图某个部分的放大。

**前置知识**：第 1-2 章

---

## 3.1 两个核心概念：Engine 与 Session

在深入代码之前，先理解 ds4 最重要的两个数据结构。它们定义在 `ds4.h:1-20` 的注释里：

```
┌─────────────────────────────────────────────────────┐
│  ds4_engine（引擎 = 加载好的模型）                    │
│                                                       │
│  - 模型权重（从 GGUF 文件 mmap 进来的）                │
│  - GPU 状态（Metal/CUDA 上下文、编译好的着色器）        │
│  - 配置参数（context_size, prefill_chunk 等）          │
│  - 生命周期：打开一次，可以创建多个 session             │
│                                                       │
│  类比：一台装好了软件的电脑                            │
├─────────────────────────────────────────────────────┤
│  ds4_session（会话 = 一次推理时间线）                  │
│                                                       │
│  - KV 缓存（存放到目前为止的注意力中间结果）            │
│  - 已处理的 token 序列（checkpoint）                   │
│  - logits 输出缓冲区（词表大小的概率分数）             │
│  - 生命周期：每次对话创建一个，用完释放                 │
│                                                       │
│  类比：在这台电脑上打开的一个文档                      │
└─────────────────────────────────────────────────────┘
```

一个 engine 可以服务多个 session（多用户场景），每个 session 有自己独立的对话历史和 KV 缓存。

> **源码位置**：`ds4.h:91-92` 声明了这两个类型：
> ```c
> typedef struct ds4_engine ds4_engine;
> typedef struct ds4_session ds4_session;
> ```
> 它们的完整定义在 `ds4.c` 内部（分别是 `ds4.c:21808` 和 `ds4.c:23259` 附近），对外只暴露不透明指针，这是 C 语言常见的封装手法。

---

## 3.2 推理的五步主线

一次完整的推理，从你在终端敲下命令到看到输出，经过五步：

```
第 1 步：解析参数 + 加载模型
  main() → ds4_engine_open()
  读 GGUF 文件、mmap 权重、编译 GPU 着色器

第 2 步：创建会话
  ds4_session_create()
  分配 KV 缓存、logits 缓冲区

第 3 步：Prefill（处理输入）
  ds4_session_sync()
  把你的 prompt 一次性灌进去，算出 KV 缓存 + 第一个 token

第 4 步：Decode（逐个生成）
  循环调用 ds4_session_eval()
  每次喂 1 个 token，得到下一个 token 的 logits

第 5 步：采样 + 输出
  从 logits 中选出 token，打印出来
  如果没到结束条件，回到第 4 步
```

用代码表示就是：

```c
// 第 1 步：加载模型
ds4_engine_open(&engine, &options);          // ds4.c:55454

// 第 2 步：创建会话
ds4_session_create(&session, engine, ctx);   // ds4.c:56773

// 第 3 步：Prefill
ds4_session_sync(session, &prompt_tokens, err, errlen);  // ds4.c:58035

// 第 4-5 步：Decode 循环
while (还没结束) {
    ds4_session_eval(session, token, err, errlen);  // ds4.c:59928
    token = sample(session->logits);  // 从概率分布中选 token
    print(token);
}
```

下面逐步拆解。

---

## 3.3 第 1 步：加载模型

入口在 `ds4_cli.c:2058` 的 `main()` 函数。核心调用在 `ds4_cli.c:2112`：

```c
} else if (ds4_engine_open(&engine, &cfg.engine) != 0) {
    // 加载失败，退出
    return 1;
}
```

`ds4_engine_open()` 定义在 `ds4.c:55454`，它是个薄包装，实际调用 `ds4_engine_open_internal()`（`ds4.c:55459`）。这个函数做了很多事情：

```
ds4_engine_open_internal()
  │
  ├─ 分配 ds4_engine 结构体
  ├─ 复制配置参数（backend, context_size, prefill_chunk 等）
  ├─ ds4_acquire_instance_lock()     ← 获取实例锁（防止多实例）
  ├─ model_open()                    ← 打开 GGUF 文件，mmap 加载
  │    （ds4.c:2428）
  ├─ 解析模型形状（n_layer, n_embd, n_expert 等）
  ├─ 验证张量布局（tensor_expect_layout）
  ├─ 初始化 GPU 后端
  │    ├─ Metal: 编译 metal/*.metal 着色器
  │    └─ CUDA:  初始化 CUDA 上下文
  └─ 加载 MTP/DSpark 支持模型（如果配置了）
```

> **关键设计**：模型加载用的是 **mmap**（内存映射文件），不是一次性读入内存。操作系统按需把文件内容映射到虚拟内存，只有真正访问到的页面才会加载到物理内存。这意味着 86GB 的模型文件在内存不够时也能用（配合 SSD 流式，第 21 章详讲）。

---

## 3.4 第 2 步：创建会话

模型加载好后，需要创建一个会话来持有推理状态。在 CLI 的 `run_generation()` 函数（`ds4_cli.c:1169`）中，最终会走到 session 创建。

`ds4_session_create()` 定义在 `ds4.c:56773`。它根据后端走不同路径：

```c
int ds4_session_create(ds4_session **out, ds4_engine *e, int ctx_size) {
    if (e->backend == DS4_BACKEND_CPU) {
        // CPU 路径：分配 CPU 用的 KV 缓存和 scratch 空间
        ds4_session *s = xcalloc(1, sizeof(*s));
        s->engine = e;
        s->ctx_size = ctx_size;
        kv_cache_init(&s->cpu_cache, ctx_size, 0);          // KV 缓存
        cpu_decode_scratch_init(&s->cpu_scratch, ctx_size);  // 临时计算空间
        s->logits = xmalloc(DS4_N_VOCAB * sizeof(float));   // 输出缓冲区
        s->sample_probs = xmalloc(DS4_N_VOCAB * sizeof(float));
        *out = s;
        return 0;
    }
    // GPU 路径（Metal/CUDA）：分配 GPU 显存中的 KV 缓存
    // ...
}
```

注意 `s->logits` 的大小是 `DS4_N_VOCAB * sizeof(float)`。DeepSeek V4 Flash 的词表有 129280 个词，所以这个数组有约 0.5MB。每次推理后，模型输出的概率分数就写在这里。

---

## 3.5 第 3 步：Prefill（处理输入）

当你输入一段 prompt，ds4 需要先把它"消化"一遍。这个步骤叫 **sync**（同步），入口是 `ds4_session_sync()`（`ds4.c:58035`）：

```c
int ds4_session_sync(ds4_session *s, const ds4_tokens *prompt,
                     char *err, size_t errlen) {
    int rc = ds4_session_sync_internal(s, prompt, err, errlen);
    // ... 清理工作
    return rc;
}
```

`ds4_session_sync_internal()`（`ds4.c:58093`）是核心。它做了一件聪明的事--**增量复用**：

```
情况 1：新 prompt 和上次的有公共前缀
  → 只处理新增的 token，复用已有的 KV 缓存
  → 这就是为什么连续对话时，后续消息的 prefill 很快

情况 2：完全不同的 prompt
  → 重置 KV 缓存，从头处理整个 prompt
```

代码里的判断逻辑（简化）：

```c
// ds4.c:58123 附近
if (s->checkpoint_valid &&
    prompt->len >= s->checkpoint.len &&
    ds4_tokens_starts_with(prompt, &s->checkpoint))
{
    // 有公共前缀！只处理新增部分
    for (int i = s->checkpoint.len; i < prompt->len; i++) {
        forward_token_raw_swa_cpu_decode_scratch(...);  // 逐个处理
        token_vec_push(&s->checkpoint, prompt->v[i]);
    }
    return 0;
}

// 没有公共前缀，从头来
session_cpu_reset_cache(s);
prefill_layer_major_cpu(s->logits, &e->model, &e->weights,
                        &s->cpu_cache, prompt, ...);  // 批量处理
```

> **关键区别**：
> - 有公共前缀时，逐个 token 调用 `forward_token_raw_swa_cpu_decode_scratch()`（走 decode 路径）
> - 全新 prompt 时，调用 `prefill_layer_major_cpu()`（走 prefill 路径，可以批量并行）
>
> 第 16 章会详细讲这两条路径的区别。

### Prefill 的产出

Prefill 结束后，session 里有了：
1. **KV 缓存**：每一层都存好了 prompt 中每个位置的 Key 和 Value 向量
2. **logits**：最后一个 token 位置算出的概率分布，可以直接用来采样第一个输出 token

---

## 3.6 第 4 步：Decode（逐个生成）

Prefill 之后，就进入 decode 循环。核心函数是 `ds4_session_eval()`（`ds4.c:59928`）：

```c
int ds4_session_eval(ds4_session *s, int token, char *err, size_t errlen) {
    bool probe_mtp = true;
    // ...
    return ds4_session_eval_probe_tp(s, token, probe_mtp, err, errlen);
}
```

它最终调用 `ds4_session_eval_internal()`（`ds4.c:59721`）。以 CPU 路径为例（`ds4.c:59733`）：

```c
if (ds4_session_is_cpu(s)) {
    ds4_engine *e = s->engine;
    forward_token_raw_swa_cpu_decode_scratch(
        s->logits,           // 输出：logits
        &e->model,           // 模型
        &e->weights,         // 权重
        &s->cpu_cache,       // KV 缓存（会更新）
        token,               // 输入：当前 token
        (uint32_t)s->checkpoint.len,  // 当前位置
        ...
    );
    token_vec_push(&s->checkpoint, token);  // 记录已处理
    s->checkpoint_valid = true;
    return 0;
}
```

Metal/GPU 路径类似（`ds4.c:59875`）：

```c
metal_graph_eval_token_raw_swa(&s->graph, &e->model, &e->weights,
                               token, s->checkpoint.len, s->logits);
```

每次 eval 做的事情：
1. 把输入 token 转成 embedding 向量
2. 跑 43 层 Transformer（每层：Attention + MoE/FFN）
3. 每层把新的 K/V 存入缓存，用历史 K/V 算注意力
4. 最后一层输出经过 output head，得到 logits

```
输入 token (1个)
    │
    ▼
┌──────────────────────────────────┐
│  Layer 0                          │
│  ┌─────────┐    ┌──────────────┐ │
│  │Attention│ →  │ MoE / FFN    │ │
│  │ + KV缓存│    │ (256选6+1)   │ │
│  └─────────┘    └──────────────┘ │
├──────────────────────────────────┤
│  Layer 1                          │
│  ... 同上 ...                     │
├──────────────────────────────────┤
│  ...                              │
├──────────────────────────────────┤
│  Layer 42                         │
│  ... 同上 ...                     │
├──────────────────────────────────┤
│  Output Head (线性层 + softmax)   │
└──────────────────────────────────┘
    │
    ▼
logits (129280个分数) → 采样 → 下一个 token
```

---

## 3.7 第 5 步：采样与输出

得到 logits 后，需要从中选出下一个 token。这个过程叫**采样**。

在 CLI 中，有两种模式（`ds4_cli.c:1245` 附近）：

### 模式 1：贪心解码（Argmax）

当 temperature = 0 时，直接选 logits 最大的 token：

```c
rc = ds4_engine_generate_argmax(engine, &prompt, n_predict, ctx_size,
                                print_generated_token, ...);
```

`print_generated_token` 是个回调函数，每生成一个 token 就被调用一次，把 token 对应的文字打印到屏幕。

### 模式 2：采样解码

当 temperature > 0 时，走 `run_sampled_generation()`，用 temperature/top-p/min-p 等策略采样。

采样的详细原理在第 17 章讲。现在你只需要知道：**logits 是一组分数，采样策略决定怎么从分数变成一个 token**。

---

## 3.8 完整流程图

把五步合在一起：

```
用户输入: ./ds4 -p "你好"
    │
    ▼
┌─ main() [ds4_cli.c:2058] ─────────────────────────────┐
│  parse_options()  → 解析 -p, -n 等参数                  │
│  ds4_engine_open()  → 加载模型 [ds4.c:55454]           │
│    ├─ model_open()  → mmap GGUF 文件                   │
│    └─ 编译 GPU 着色器（Metal/CUDA）                     │
│  run_generation()  → [ds4_cli.c:1169]                  │
│    ├─ build_prompt()  → 文本 → token 序列               │
│    │                                                     │
│    ├─ ds4_session_create()  → [ds4.c:56773]            │
│    │    └─ 分配 KV 缓存 + logits 缓冲区                 │
│    │                                                     │
│    ├─ ds4_session_sync()  → Prefill [ds4.c:58035]      │
│    │    ├─ 检查能否复用上次 KV 缓存                      │
│    │    ├─ prefill_layer_major_cpu()  或  逐 token 处理 │
│    │    └─ 产出：KV 缓存 + 第一个 token 的 logits       │
│    │                                                     │
│    └─ Decode 循环:                                      │
│         ┌──────────────────────────────────┐           │
│         │ ds4_session_eval() [ds4.c:59928] │           │
│         │  → forward 43 layers             │           │
│         │  → 更新 KV 缓存                   │           │
│         │  → 得到 logits                    │           │
│         ├──────────────────────────────────┤           │
│         │ sample(logits) → token           │           │
│         │ print(token)                     │           │
│         │ 没结束? → 继续循环                │           │
│         └──────────────────────────────────┘           │
│                                                          │
│  ds4_engine_close()  → 释放资源                         │
└──────────────────────────────────────────────────────────┘
    │
    ▼
屏幕输出: "你好！有什么可以帮你的吗？"
```

---

## 3.9 小结：一句话记住主线

```
engine_open → session_create → session_sync(预填充) → loop { session_eval(解码) → sample }
```

这五个函数就是推理的骨架。后面所有章节都是对这条主线的某个环节做深入：

- 第 4-6 章：深入 `engine_open` 里的模型加载
- 第 7 章：深入 `build_prompt` 里的分词
- 第 8-13 章：深入 `session_eval` 里每一层的算子
- 第 14-16 章：深入 prefill 和 decode 两条路径
- 第 17 章：深入采样策略
- 第 18-22 章：深入各种优化技术

---

## 本章小结

- `ds4_engine` = 加载好的模型，`ds4_session` = 一次推理的状态
- 推理五步：加载模型 → 创建会话 → Prefill → Decode 循环 → 采样输出
- Prefill 批量处理输入，产出 KV 缓存和第一个 token
- Decode 每次处理 1 个 token，依赖 KV 缓存
- `session_sync` 支持增量复用：相同前缀的 prompt 不用重新算

## 动手实验

1. 在 `ds4_cli.c` 中找到 `main` 函数，追踪 `run_generation` 的调用链，在纸上画出函数调用关系
2. 运行两次相同的 prompt，第二次观察 prefill 速度是否有变化（ds4 在交互模式下可能复用 KV 缓存）
3. 打开 `ds4.c:59721`，阅读 `ds4_session_eval_internal` 函数，找到 CPU 路径和 Metal 路径分别调用了什么函数

## 下一章预告

第一部分结束了。从第 4 章开始进入第二部分"模型篇"，我们会打开 GGUF 文件，看看 86GB 的模型文件里到底装了什么。
