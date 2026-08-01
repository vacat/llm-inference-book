# 第 26 章 性能分析与调优实战

## 本章导读

这是工程篇的最后一章，也是全书的"实战收尾"。我们整合前面学到的所有知识，用 ds4 的 profiling 工具定位瓶颈，做数据驱动的优化。

读完本章你会掌握：
- 完整的性能分析工作流
- 如何用环境变量定位瓶颈
- 如何用 ds4-bench 测量基线
- 如何用 ds4-eval 验证质量
- 常见瓶颈的优化方向

**前置知识**：全书前 25 章

---

## 26.1 性能分析工作流

```
1. 测基线     ──>  ds4-bench
2. 定瓶颈     ──>  DS4_DECODE_PROFILE_DETAIL
3. 读源码     ──>  针对最耗时的算子精读
4. 改实现     ──>  最小改动, 正确性优先
5. 验质量     ──>  ds4-eval (防止 logits 漂移)
6. 测提升     ──>  ds4-bench
7. 对比       ──>  before vs after
```

> **核心原则**（来自 `AGENT.md`）：**正确性优先于速度**。任何优化都必须用 `ds4-eval` 验证无回归。

---

## 26.2 测基线：ds4-bench

`ds4-bench`（`ds4_bench.c:554`）测量 prefill 和 decode 的吞吐。

### 基本用法

```bash
./ds4-bench -m ds4flash.gguf \
  --prompt-file speed-bench/promessi_sposi.txt \
  --ctx-start 2048 \
  --ctx-max 4096 \
  --gen-tokens 64 \
  --csv /tmp/bench.csv
```

### 参数说明

```
--prompt-file:  prefill 用的文本素材
--ctx-start:    起始上下文长度
--ctx-max:      最大上下文长度
--gen-tokens:   生成的 token 数
--csv:          输出 CSV 结果
```

### 输出解读

```
CSV 列:
  ctx_size:      上下文长度
  prefill_tps:   prefill 吞吐 (tokens/second)
  decode_tps:    decode 吞吐 (tokens/second)
  first_token_ms: 首字延迟
  ...

示例:
  ctx=2048, prefill=2100 tps, decode=68 tps
  ctx=4096, prefill=1800 tps, decode=62 tps  ← 上下文变长, decode 变慢
```

> **为什么 decode 随上下文变慢？** 上下文越长，KV 缓存越大，注意力要扫描的行越多，内存读取越多。

---

## 26.3 定瓶颈：逐算子计时

### DS4_DECODE_PROFILE_DETAIL

```bash
DS4_DECODE_PROFILE_DETAIL=1 ./ds4 -p "test" -n 10 --temp 0
```

输出每层每个阶段的耗时（第 14 章讲过）：

```
ds4: decode detail layer 0 attn hc=0.12 q=0.34 kv=0.11 rope=0.05 
  compress=0.00 indexer=0.00 attn_rows=2.31 inv_rope=0.04 out=0.28 
  post=0.03 ffn=1.67 total=4.95 ms

ds4: decode detail layer 1 attn hc=0.11 q=0.33 kv=0.10 rope=0.05 
  compress=0.00 indexer=0.00 attn_rows=2.28 inv_rope=0.04 out=0.27 
  post=0.03 ffn=1.64 total=4.85 ms
...
```

### 分析输出

```
每层的主要耗时:
  attn_rows: 注意力计算 (随上下文增长)
  ffn:       MoE 计算 (6 专家 + 1 共享)
  q:         Q 投影
  out:       输出投影

典型分布 (中等上下文):
  attn_rows: 45%  ← 上下文越长占比越大
  ffn:       33%  ← 基本固定
  q+kv+out:  15%  ← 矩阵乘法
  其他:       7%

长上下文:
  attn_rows: 70%+ ← 注意力成为绝对瓶颈
  -> 优化方向: Flash Attention, KV 压缩
```

### 其他环境变量

| 变量 | 作用 |
|------|------|
| `DS4_TOKEN_TIMING=1` | 每 token 计时（仅 argmax 路径，需 `--temp 0`） |
| `DS4_MTP_PROBE=1` | MTP 草稿命中率 |
| `DS4_MTP_TIMING=1` | MTP 推测解码计时 |
| `DS4_METAL_MATH_SAFE=1` | 关闭 fastMath 诊断数值漂移 |
| `DS4_METAL_STREAMING_SELECTED_READAHEAD_PROFILE=1` | 专家预取耗时 |

---

## 26.4 常见瓶颈与优化方向

### 瓶颈 1：Decode 太慢

```
症状: decode < 30 tok/s
可能原因:
  1. 上下文太长 -> 注意力瓶颈
     优化: 确认 KV 压缩和 SWA 生效
  2. 内存带宽不够 -> 权重读取慢
     优化: 用更激进的量化 (IQ2_XXS)
  3. SSD 流式未命中 -> 专家加载慢
     优化: 增大 --ssd-cache, 或全内存
  4. GPU 利用率低
     优化: 确认用的是 Metal/CUDA 路径而非 CPU
```

### 瓶颈 2：首字延迟高

```
症状: prefill > 2 秒
可能原因:
  1. prompt 太长
     优化: 增大 prefill_chunk, 或分段
  2. 首次加载模型 (mmap 冷启动)
     优化: --warm-weights 预热
  3. SSD 流式冷启动
     优化: --ssd-streaming-preload-experts 预加载
```

### 瓶颈 3：GPU 利用率低

```
症状: GPU 利用率 < 10%
可能原因:
  1. Decode 天然算力利用率低 (内存带宽瓶颈)
     优化: 推测解码 (MTP/DSpark)
  2. SSD/NAS I/O 瓶颈
     优化: 本地 SSD, 增大缓存
  3. CPU-GPU 同步开销
     优化: 确认图执行模式生效
```

---

## 26.5 验证质量：ds4-eval

优化后必须验证输出质量没退化。

### 使用

```bash
./ds4-eval -m ds4flash.gguf [options]
```

`ds4_eval.c:4119` 的 `main()` 函数运行质量测试，检查 logits 是否漂移。

### 验证策略

```
1. Logits 对比:
   优化前后对相同输入的 logits 做对比
   如果差异超过阈值 -> 有回归

2. 困惑度 (Perplexity):
   在标准数据集上测困惑度
   优化后困惑度不应增加

3. 输出对比:
   相同 prompt + seed, 输出应完全一致 (贪心模式)
```

---

## 26.6 优化实验示例

### 实验 1：调参对比

```bash
# 基线
./ds4-bench -m ds4flash.gguf --prompt-file speed-bench/promessi_sposi.txt \
  --ctx-start 2048 --ctx-max 4096 --gen-tokens 64 --csv /tmp/baseline.csv

# 实验 A: 开 MTP (推测解码)
./ds4-bench -m ds4flash.gguf --mtp-draft 3 \
  --prompt-file speed-bench/promessi_sposi.txt \
  --ctx-start 2048 --ctx-max 4096 --gen-tokens 64 --csv /tmp/mtp3.csv

# 实验 B: 调 power
./ds4-bench -m ds4flash.gguf --power 50 \
  --prompt-file speed-bench/promessi_sposi.txt \
  --ctx-start 2048 --ctx-max 4096 --gen-tokens 64 --csv /tmp/power50.csv

# 对比
diff /tmp/baseline.csv /tmp/mtp3.csv
```

### 实验 2：上下文长度影响

```bash
for ctx in 512 1024 2048 4096 8192; do
  ./ds4-bench -m ds4flash.gguf \
    --prompt-file speed-bench/promessi_sposi.txt \
    --ctx-start $ctx --ctx-max $ctx --gen-tokens 32 \
    --csv /tmp/ctx-${ctx}.csv
done
```

观察 decode 速度如何随上下文增长而下降。

### 实验 3：量化对比

```bash
# IQ2_XXS (2 bit)
./ds4-bench -m ds4flash.gguf ... --csv /tmp/iq2.csv

# Q4_K (4 bit, 如果有这个版本)
./ds4-bench -m deepseek-v4-flash-q4k.gguf ... --csv /tmp/q4k.csv

# 对比速度和质量
```

---

## 26.7 性能调优检查清单

```
□ 确认后端: Metal/CUDA (不是 CPU)
□ 确认量化: 用了合适的量化格式
□ 确认 KV 压缩: 长上下文有压缩
□ 确认 SWA: 滑动窗口生效
□ 确认 MTP: 推测解码开启
□ 确认 SSD 缓存大小: 够大减少未命中
□ 确认无其他进程: 实例锁正常
□ 确认预热: 首次运行后做 benchmark
□ 确认质量: ds4-eval 通过
□ 确认上下文: 没有超出 ctx_size
```

---

## 26.8 性能参考数据

```
Apple M5 Max (96GB, Metal):
  Prefill:  ~2000 tok/s
  Decode:   ~65 tok/s (单用户)
  Decode:   ~100+ tok/s (MTP 推测解码)

8x L40S (CUDA, 多卡):
  Prefill:  ~2000 tok/s
  Decode:   ~120 tok/s 聚合 (多用户微批)

2x M5 Max (张量并行, RDMA):
  Decode:   ~110 tok/s (1.7x 加速)

CPU only (参考实现, M5 Max):
  Decode:   ~5 tok/s (14x 慢于 Metal)
```

---

## 本章小结

- 性能分析七步：测基线 -> 定瓶颈 -> 读源码 -> 改实现 -> 验质量 -> 测提升 -> 对比
- `ds4-bench` 测吞吐，`DS4_DECODE_PROFILE_DETAIL` 定位逐算子瓶颈
- Decode 瓶颈通常是注意力（随上下文增长）或内存带宽
- 优化后必须用 `ds4-eval` 验证质量无回归
- 常见优化：MTP 推测解码、增大 SSD 缓存、更激进量化、张量并行
- 正确性永远优先于速度

## 动手实验

1. 运行 `ds4-bench` 获取你的机器基线数据
2. 运行 `DS4_DECODE_PROFILE_DETAIL=1 ./ds4 -p "test" -n 10`，分析瓶颈在哪
3. 尝试开关 MTP，对比 decode 速度变化
4. 用不同上下文长度做 benchmark，画出 decode 速度 vs 上下文长度的曲线

## 下一章预告

正文部分全部结束。接下来是附录：关键数据结构索引、环境变量速查、术语表、学习路线。
