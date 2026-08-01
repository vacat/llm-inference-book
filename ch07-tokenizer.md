# 第 7 章 分词器：从文本到 Token

## 本章导读

在推理开始之前，你输入的文本需要先变成一串整数（token id）。这个工作由**分词器（Tokenizer）**完成。这章我们拆解 ds4 中的 BPE 分词器，理解它如何把"你好世界"变成 `[57668, 10236, 10348]`。

读完本章你会理解：
- BPE（字节对编码）的合并算法
- 为什么要用"字节级"而非"字符级"
- 分词器在 GGUF 文件中如何存储
- 分词对推理的影响

**前置知识**：第 3-4 章

---

## 7.1 为什么需要分词器

模型不认识文字，只认识数字。分词器就是文字和数字之间的翻译官：

```
你输入:  "Hello, world!"
          ↓ 分词器
Token 序列: [9906, 11, 1917, 0]
          ↓ 模型推理
输出 Token: [3118, 318, 2834]
          ↓ 反分词器（detokenize）
你看到:  "How are you?"
```

分词器决定了文本被切成几段、每段多长。切得好不好直接影响：
- **推理速度**：token 越少，计算越少
- **输出质量**：切词不当可能让模型理解错误
- **上下文长度**：同样长的文本，不同分词器产生的 token 数不同

---

## 7.2 BPE 算法原理

BPE（Byte-Pair Encoding，字节对编码）是主流的分词算法。它的核心思想是：**从单字节开始，不断合并最常见的相邻字节对**。

### 训练阶段（离线）

BPE 的"词表"是预先训练好的。训练过程：

```
1. 把所有训练文本拆成单字节序列
   "low" -> [l, o, w]
   "lower" -> [l, o, w, e, r]
   "newest" -> [n, e, w, e, s, t]

2. 统计相邻字节对的出现频率，合并最高频的
   (l, o) 出现 2 次 -> 合并成 "lo"
   (lo, w) 出现 2 次 -> 合并成 "low"
   ...

3. 重复合并，直到达到目标词表大小
   每次合并产生一个新 token，赋予一个 rank（优先级）
   rank 越低 = 越早合并 = 越常见
```

最终得到一个**合并规则表**（merge table），记录"哪两个相邻片段可以合并成一个，优先级是多少"。

### 推理阶段（在线）

推理时，BPE 用训练好的合并规则来切分新文本：

```
输入: "lowest"

1. 拆成符号序列: [l, o, w, e, s, t]

2. 查所有相邻对的 rank，找 rank 最小（最优先）的合并:
   (l, o) -> rank=5  ← 最小
   (o, w) -> rank=20
   (w, e) -> rank=50
   ...

3. 合并 (l,o) -> [lo, w, e, s, t]

4. 再找最小 rank 的相邻对:
   (lo, w) -> rank=10  ← 最小
   合并 -> [low, e, s, t]

5. 继续合并，直到没有可合并的对

6. 最终: [low, est] -> 查词表得到 token id
```

> **核心规则**：每次贪心地合并 rank 最小（最常见）的相邻对，直到无法合并为止。

---

## 7.3 源码解读：bpe_emit_piece

ds4 的 BPE 实现在 `bpe_emit_piece()`（`ds4.c:35703`）。我们逐段看：

### 第 1 步：字节编码

```c
// ds4.c:35704-35705
uint64_t encoded_len = 0;
char *encoded = byte_encode(raw_piece, &encoded_len);
```

`byte_encode` 把文本转成字节序列。这是"字节级 BPE"的关键--所有输入都先变成字节，这样任意 UTF-8 文本都能处理。

> **为什么用字节级？** 如果用字符级，中文字符有几万个，词表会爆炸。用字节级，每个字符最多 4 个字节（UTF-8 编码），词表大小可控。常见的中文字/词通过 BPE 合并变成单个 token，生僻字保持多字节形式。

### 第 2 步：初始化符号序列

```c
// ds4.c:35707-35720
int n_sym = 0;
owned_str *sym = xcalloc(cap_sym, sizeof(sym[0]));

for (uint64_t off = 0; off < encoded_len;) {
    int n = utf8_len_from_first_byte((uint8_t)encoded[off]);
    // 每个 UTF-8 字符作为一个初始符号
    sym[n_sym++] = owned_copy(encoded + off, (uint64_t)n);
    off += (uint64_t)n;
}
```

把字节序列按 UTF-8 字符边界拆成初始符号序列。比如 `"low"` -> `[l, o, w]`。

### 第 3 步：贪心合并循环

```c
// ds4.c:35722-35748
for (;;) {
    int best_i = -1;
    int best_rank = INT32_MAX;

    // 扫描所有相邻对，找 rank 最小的
    for (int i = 0; i + 1 < n_sym; i++) {
        int rank = bpe_rank(vocab, &sym[i], &sym[i + 1]);
        if (rank >= 0 && rank < best_rank) {
            best_rank = rank;
            best_i = i;
        }
    }

    if (best_i < 0) break;  // 没有可合并的对，结束

    // 合并 best_i 和 best_i+1
    owned_str merged;
    merged.len = sym[best_i].len + sym[best_i + 1].len;
    merged.ptr = xmalloc(merged.len);
    memcpy(merged.ptr, sym[best_i].ptr, sym[best_i].len);
    memcpy(merged.ptr + sym[best_i].len, sym[best_i + 1].ptr, sym[best_i + 1].len);

    // 替换：merged 放到 best_i 位置，后面的前移
    sym[best_i] = merged;
    for (int j = best_i + 1; j + 1 < n_sym; j++) {
        sym[j] = sym[j + 1];
    }
    n_sym--;
}
```

这段代码精确实现了 BPE 算法：
1. 遍历所有相邻符号对
2. 查 `bpe_rank()` 得到合并优先级
3. 选 rank 最小的合并
4. 合并后更新符号序列
5. 重复直到没有可合并的

### 第 4 步：查表得到 token id

```c
// ds4.c:35750-35760
for (int i = 0; i < n_sym; i++) {
    int token = -1;
    if (table_get(&vocab->token_to_id, sym[i].ptr, sym[i].len, &token)) {
        token_vec_push(out, token);  // 找到，输出 token id
    } else {
        // 找不到完整匹配，退化为逐字节查表
        for (uint64_t j = 0; j < sym[i].len; j++) {
            if (table_get(&vocab->token_to_id, sym[i].ptr + j, 1, &token)) {
                token_vec_push(out, token);
            }
        }
    }
}
```

合并完成后，每个符号在词表中查对应的 token id。如果某个符号在词表中找不到（极少见），就退化成逐字节查表。

---

## 7.4 bpe_rank：查合并优先级

`bpe_rank()`（`ds4.c:35686`）查两个相邻符号的合并优先级：

```c
static int bpe_rank(const ds4_vocab *vocab, const owned_str *a, const owned_str *b) {
    // 拼接: a + " " + b
    uint64_t len = a->len + 1 + b->len;
    char buf[...];
    memcpy(buf, a->ptr, a->len);
    buf[a->len] = ' ';                      // 用空格分隔
    memcpy(buf + a->len + 1, b->ptr, b->len);

    // 在合并表中查找
    int rank = -1;
    table_get(&vocab->merge_rank, buf, len, &rank);
    return rank;
}
```

合并表用 `"符号A �号B"` 作为键（中间用空格分隔），存对应的 rank 值。rank 越小，优先级越高。

> **为什么用空格分隔？** 因为不用空格的话，`("ab", "c")` 和 `("a", "bc")` 拼出来都是 `"abc"`，无法区分。加空格后变成 `"ab c"` 和 `"a bc"`，可以区分。

---

## 7.5 预分词：先把文本切段

BPE 不是对整段文本一次性处理，而是先按规则切成"词"（pre-tokenization），再对每个词单独做 BPE。

`bpe_tokenize_text()`（`ds4.c:36114`）负责预分词。它用类似正则表达式的规则把文本切成片段：

```
输入: "Hello, world! 你好"

预分词:
  "Hello"     -> BPE -> [9906]
  ","         -> BPE -> [11]
  " world"    -> BPE -> [442, 1917]   ← 注意前导空格
  "!"         -> BPE -> [0]
  " 你"       -> BPE -> [57668]       ← 中文会被合并成较少 token
  "好"        -> BPE -> [10236]
```

预分词规则大致是：
- 英文按空格和标点切分
- 数字序列保持连续
- 中日韩字符逐字处理
- 空格通常和后面的词绑在一起（`" world"` 是一个预分词单元）

> **细节**：ds4 支持 DeepSeek 和 GLM 两种预分词规则。`bpe_tokenize_text()` 是 DeepSeek 路径，`bpe_tokenize_text_glm4()`（`ds4.c:35979`）是 GLM 路径。两者的 BPE 合并算法相同，但预分词的正则规则不同。

---

## 7.6 词表在 GGUF 中的存储

分词器的所有数据都存在 GGUF 文件的元数据区。`vocab_load()`（`ds4.c:36206`）负责加载：

```
GGUF 元数据中存储的分词器数据:
  - tokenizer.ggml.tokens:     所有 token 的文本（129280 个）
  - tokenizer.ggml.token_type: 每个 token 的类型（普通/控制/字节等）
  - tokenizer.ggml.merges:     BPE 合并规则表（"符号A 符号B" 列表）
  - tokenizer.ggml.eos_token_id: 结束符的 token id
  - tokenizer.ggml.bos_token_id: 起始符的 token id
```

加载后构建两个哈希表：
- `token_to_id`：文本 -> token id（用于编码）
- `merge_rank`：合并对 -> rank（用于 BPE 合并）

### ds4_vocab 结构

```c
// ds4.c:35328
struct ds4_vocab {
    // ... 词表数据
    ds4_table token_to_id;    // 文本 -> id 查找表
    ds4_table merge_rank;     // 合并对 -> rank 查找表
    // ...
};
```

---

## 7.7 分词对推理的影响

### Token 数量影响速度

同样的文本，不同分词器产生的 token 数可能差很多：

```
"人工智能是计算机科学的一个分支"
  DeepSeek 分词器: ~8 个 token
  某些英文分词器:  ~20 个 token（每个中文字 2-3 字节）
```

Token 越少，Prefill 越快，上下文利用率越高。DeepSeek V4 的分词器对中文优化得很好。

### 特殊 token

模型有一些特殊的 token：
- **BOS**（Beginning of Sequence）：序列开始标记
- **EOS**（End of Sequence）：序列结束标记，模型生成到这个 token 就停止
- **聊天模板 token**：如 `<|im_start|>`、`<|im_end|>`，标记对话角色和边界

ds4 在 `build_prompt()` 中会按聊天模板把你的输入包装成正确的 token 序列，包括添加 BOS/EOS 和角色标记。

---

## 7.8 反分词：从 token 到文字

推理输出的 token id 需要转回文字。这个过程简单得多--直接查表：

```c
// 伪代码
const char *text = vocab->tokens[token_id];
printf("%s", text);
```

但有一个细节：**流式输出**。模型一个 token 一个 token 地生成，每个 token 的文字要立即显示。有些 token 可能只输出半个 UTF-8 字符（因为字节级编码），ds4 需要缓冲不完整的字节，等凑够一个完整 UTF-8 字符再输出。

---

## 本章小结

- BPE = 字节对编码：从字节开始，贪心合并最常见（rank 最小）的相邻对
- 字节级编码让任意 UTF-8 文本都能处理，词表大小可控
- `bpe_emit_piece()` 实现 BPE 合并循环：扫描 -> 找最小 rank -> 合并 -> 重复
- `bpe_rank()` 查合并表，用 `"符号A 符号B"` 作为键
- 预分词先把文本切段，再对每段做 BPE
- 词表数据存储在 GGUF 元数据区，加载时构建两个哈希表

## 动手实验

1. 运行 `./ds4 --dump-tokens -p "你好世界"`，看看你的输入被分成了哪些 token
2. 在 `ds4.c:35703` 阅读 `bpe_emit_piece`，手推一个 3 字符串的 BPE 合并过程
3. 思考：为什么 BPE 用 rank 而不是频率来决定合并顺序？（提示：rank 是全局唯一的，频率可能有并列）

## 下一章预告

第二部分结束了。从第 8 章开始进入第三部分"算子篇"，我们逐个拆解推理的核心计算算子。第一个：Embedding（词嵌入查表）。
