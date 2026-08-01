# 第 23 章 服务化部署：ds4-server

## 本章导读

前面我们用 `ds4` 命令行做单次推理。如果要给多个人用、集成到应用中，就需要 HTTP 服务。`ds4-server` 提供了 OpenAI 兼容的 API，让任何支持 OpenAI API 的客户端都能用 ds4 推理。

读完本章你会理解：
- ds4-server 的架构
- OpenAI 兼容 API 的使用
- 多用户会话管理
- 微批处理（micro-batching）

**前置知识**：第 3、17 章

---

## 23.1 为什么需要服务化

```
命令行模式 (ds4):
  - 一次启动, 一次推理, 退出
  - 每次启动要重新加载模型 (~10秒)
  - 不支持并发

服务模式 (ds4-server):
  - 启动一次, 常驻运行
  - 模型只加载一次
  - HTTP API 接收请求
  - 支持多用户并发
  - 兼容 OpenAI API 格式
```

---

## 23.2 ds4-server 架构

```
┌─────────────────────────────────────────────────┐
│  ds4-server (常驻进程)                           │
│                                                   │
│  ┌──────────────┐    ┌──────────────────────┐   │
│  │ HTTP 服务器   │    │ 引擎 (ds4_engine)     │   │
│  │ (端口 8080)   │───>│ (模型加载一次)         │   │
│  │               │    │                        │   │
│  │ /v1/chat/     │    │ ┌──────────────────┐ │   │
│  │ completions   │    │ │ Session 池        │ │   │
│  │               │    │ │ [sess1][sess2]... │ │   │
│  │ /v1/comple-   │    │ │ (多用户 KV 缓存)  │ │   │
│  │ tions         │    │ └──────────────────┘ │   │
│  └──────────────┘    └──────────────────────┘   │
└─────────────────────────────────────────────────┘
        ↑
        │ HTTP (OpenAI 兼容格式)
        │
   ┌────┴────┐
   │ 客户端   │
   │ (curl,  │
   │  Python,│
   │  前端)  │
   └─────────┘
```

### 启动服务器

```bash
./ds4-server --port 8080
```

`ds4_server.c:12885` 的 `main()` 函数处理启动参数、加载模型、启动 HTTP 服务。

---

## 23.3 OpenAI 兼容 API

ds4-server 兼容 OpenAI 的 API 格式，这意味着你可以用任何 OpenAI 客户端库直接连接。

### Chat Completions

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-chat",
    "messages": [
      {"role": "user", "content": "你好"}
    ],
    "temperature": 0.7,
    "max_tokens": 100
  }'
```

### 响应格式

```json
{
  "id": "chatcmpl-xxx",
  "choices": [{
    "message": {
      "role": "assistant",
      "content": "你好！有什么可以帮你的吗？"
    },
    "finish_reason": "stop"
  }],
  "usage": {
    "prompt_tokens": 5,
    "completion_tokens": 12,
    "total_tokens": 17
  }
}
```

### 代码调用（Python）

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="dummy")

response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[{"role": "user", "content": "解释什么是大模型推理"}],
    temperature=0.7
)
print(response.choices[0].message.content)
```

> **兼容性的价值**：你不需要改任何客户端代码，只需把 `base_url` 从 OpenAI 改成本地 ds4-server。这对企业内部部署非常有用--数据不出本地，但 API 完全兼容。

---

## 23.4 多用户会话管理

### Session 池

每个用户对话需要一个独立的 `ds4_session`（独立的 KV 缓存）。ds4-server 维护一个 session 池：

```
用户 A 发请求 -> 创建/复用 session_A
  session_A 有自己的 KV 缓存

用户 B 发请求 -> 创建/复用 session_B
  session_B 有自己的 KV 缓存

用户 A 继续对话 -> 复用 session_A
  -> 增量 sync: 只 prefill 新增的 token
  -> 快速响应
```

### KV 缓存持久化

ds4-server 用 `ds4_kvstore.c` 管理 KV 缓存的持久化：

- 活跃会话：KV 缓存在内存
- 不活跃会话：KV 缓存保存到磁盘（检查点）
- 重新激活：从磁盘加载 KV 缓存

这样在有限内存下可以服务大量用户的对话历史。

### 相关数据结构

在 `ds4.h:126-140` 中：

```c
typedef struct {
    uint8_t *ptr;
    uint64_t len;
    uint64_t cap;
} ds4_session_snapshot;

#define DS4_SESSION_PAYLOAD_MAGIC UINT32_C(0x34565344) /* "DSV4" */
#define DS4_SESSION_PAYLOAD_VERSION UINT32_C(2)
```

每个 session 的 KV 状态可以序列化为 `ds4_session_snapshot`，保存到文件。

---

## 23.5 微批处理

ds4-server 支持**微批处理**（micro-batching），把多个用户的 decode 请求合并成一批：

```
没有微批处理 (串行):
  用户 A decode 1 token -> 用户 B decode 1 token -> 用户 A decode 1 token
  每次 decode 只算 1 个 token, GPU 利用率低

有微批处理 (并行):
  [用户 A 的 token, 用户 B 的 token, 用户 C 的 token] -> 一次批量 decode
  3 个 token 同时算, GPU 利用率提高 3 倍
```

### 相关代码

在 `ds4.h:310-330` 中：

```c
int ds4_sessions_eval_batch(ds4_decode_item *items, int count,
                            char *err, size_t errlen);

int ds4_sessions_eval_batch_with_prefill(
    ds4_decode_item *items, int count,
    ds4_session *prefill_session, const ds4_tokens *prefill_prompt,
    char *err, size_t errlen);
```

`ds4_sessions_eval_batch` 接受多个 session 的 decode 请求，一次性执行。`ds4_sessions_eval_batch_with_prefill` 还能同时处理一个 prefill 请求。

### 启用批处理

```bash
./ds4-server --port 8080 --batched-sessions 4
```

`--batched-sessions 4` 表示最多同时批处理 4 个 session 的请求。

在 `ds4_engine_options` 中：

```c
bool share_session_prefill_workspace;  // 共享 prefill 工作空间
```

批处理模式下，多个 session 可以共享 prefill 的工作空间，减少内存占用。

---

## 23.6 流式输出

ds4-server 支持 SSE（Server-Sent Events）流式输出，让用户看到逐字生成：

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-chat",
    "messages": [{"role": "user", "content": "写一首诗"}],
    "stream": true
  }'
```

```
data: {"choices":[{"delta":{"content":"秋"}}]}
data: {"choices":[{"delta":{"content":"风"}}]}
data: {"choices":[{"delta":{"content":"起"}}]}
...
data: [DONE]
```

每个 token 生成后立即推送，用户体验更好（不用等全部生成完才看到结果）。

---

## 23.7 生产部署建议

```
单机部署 (Mac):
  ./ds4-server --port 8080
  -> 适合个人/小团队使用

多 GPU 部署 (CUDA):
  ./ds4-server --gpu-vram auto --port 8080 --batched-sessions 8
  -> 适合企业多用户服务

配合 Nginx 反向代理:
  Nginx -> ds4-server (8080)
  -> 负载均衡、TLS 加密、速率限制

配合 SSD 流式 (小内存):
  ./ds4-server --ssd-streaming --ssd-cache 16G --port 8080
  -> 16GB 内存也能服务
```

### 性能参考

```
8x L40S CUDA 服务器 (来自 README):
  聚合生成: 120 tok/s
  Prefill: 2000 tok/s
  多用户并发: 支持多个 session

Apple M5 Max (96GB):
  单用户 Decode: ~65 tok/s
  多用户 (batch): ~100+ tok/s 聚合
```

---

## 本章小结

- `ds4-server` 提供 OpenAI 兼容 HTTP API，无需修改客户端代码
- 模型加载一次常驻内存，支持多用户并发
- Session 池管理多用户对话，KV 缓存可持久化到磁盘
- 微批处理合并多个用户的 decode 请求，提高 GPU 利用率
- 支持 SSE 流式输出
- 配合 Nginx 可做生产级部署

## 动手实验

1. 启动 `./ds4-server --port 8080`，用 curl 发送一个 chat completion 请求
2. 用 Python OpenAI 库连接本地 ds4-server，对比和 OpenAI API 的用法差异
3. 尝试 `--batched-sessions 2`，同时发两个请求观察批处理效果

## 下一章预告

第 24 章，分布式推理--把模型拆到多台机器上跑，用流水线并行连接。
