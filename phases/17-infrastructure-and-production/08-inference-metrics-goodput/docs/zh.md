# Inference Metrics — TTFT、TPOT、ITL、Goodput、P99

> 四个 metrics 决定 inference deployment 是否工作正常。TTFT 是 prefill 加 queue 加 network。TPOT（等价于 ITL）是每个 token 的 memory-bound decode cost。End-to-end latency 是 TTFT 加上 TPOT 乘以 output length。Throughput 是整个 fleet 聚合的 tokens per second。但对产品最重要的是 goodput：同时满足每个 SLO 的请求比例。高 throughput、低 goodput 意味着你在处理无法及时到达用户的 tokens。2026 年 TRT-LLM 上 Llama-3.1-8B-Instruct 的参考数字：mean TTFT 162 ms，mean TPOT 7.33 ms，mean E2E 1,093 ms。永远报告 P50、P90、P99，绝不只报告 mean。还要注意 measurement trap：GenAI-Perf 在 ITL 计算中排除 TTFT，LLMPerf 会包含它；两个工具会在同一次 run 上给出不同 TPOT。

**类型：** 学习
**语言：** Python（stdlib，玩具版 percentile calculator 和 goodput reporter）
**前置要求：** 阶段 17 · 04（vLLM Serving Internals）
**时间：** ~60 分钟

## 学习目标

- 精确定义 TTFT、TPOT、ITL、E2E、throughput 和 goodput，并说出每个 metric 测量的 component。
- 解释为什么 mean 是 LLM serving 的错误统计量，以及如何读取 P50/P90/P99。
- 构造一个 SLO multi-constraint（例如 TTFT<500 ms AND TPOT<15 ms AND E2E<2 s），并用它计算 goodput。
- 说出两个在同一次 run 上 TPOT 不一致的 benchmark tools，并解释原因。

## 问题

“我们的 throughput 是 15,000 tokens per second。”所以呢？如果 40% 的请求 end-to-end 超过 2 秒，用户已经离开 session。Throughput 本身不能告诉你产品是否工作。

Inference 有多个 latency 轴，每个轴失败方式不同。Prefill 是 compute-bound，随 prompt length 扩展。Decode 是 memory-bound，随 batch size 扩展。Queuing delay 是 operational problem。Network 是 physical-distance problem。你需要为每个轴定义不同 metrics，需要 percentiles，还需要一个 composite 来回答“用户是否得到预期体验”：这就是 goodput。

## 概念

### TTFT — time to first token

`TTFT = queue_time + network_request + prefill_time`

当 prompts 很长时，prefill 主导。在 H100 上运行 Llama-3.3-70B FP8，一个 32k prompt 纯 prefill 需要约 800 ms。Queue time 是负载下的 scheduler behavior。Network request 是包括 TLS 在内的 wire time。TTFT 是用户看到任何流式返回前的等待时间。

### TPOT / ITL — inter-token latency

一个量有很多名字。`TPOT`（time per output token）、`ITL`（inter-token latency）、`decode latency per token` 都是同一个东西。它是首 token 之后，相邻 streamed tokens 之间的时间。

`TPOT = (decode_forward_time + scheduler_overhead) / tokens_produced`

在同一 Llama-3.3-70B H100 stack 加 chunked prefill 下，TPOT mean 约 7 ms。没有 chunked prefill 时，相邻 sequence 的长 prefill 期间，TPOT 可能 spike 到 50 ms。看 P99，不看 mean。

### E2E latency

`E2E = TTFT + TPOT * output_tokens + network_response`

对长输出（>500 tokens），E2E 由 TPOT 主导。对短输出加长 prompt，E2E 由 TTFT 主导。报告按 output length 条件化的 E2E。

### Throughput

`throughput = total_output_tokens / elapsed_time`

聚合 metric。告诉你 fleet efficiency。不告诉你单个 request 是否健康。

### Goodput — 你真正关心的 metric

`goodput = fraction of requests meeting (TTFT <= a) AND (TPOT <= b) AND (E2E <= c)`

SLO 是 multi-constraint。只有每个 constraint 都满足，一个 request 才是“good”。Goodput 是这个比例。高 throughput 但 goodput 只有 60% 是失败。较低 throughput 但 99% goodput 才是目标。

2026 年，goodput 是 MLPerf Inference v6.0 submissions 和 AI platform providers 内部 SLA tracking 使用的 metric。

### 为什么 mean 是错误统计量

LLM latency distributions 是 right-skewed。一个 decode batch 里有一个 long-prefill neighbor 时，可能有 500 个 tokens 以 TPOT ~7 ms 交付，另有 20 个 tokens 以 TPOT ~60 ms 交付。Mean TPOT 是 9 ms。P99 TPOT 是 65 ms。用户经常撞到 P99，这就是他们离开的原因。

永远报告三元组（P50、P90、P99）。对用户体验来说，P99 是你要优化的。

### 参考数字 — TRT-LLM 上的 Llama-3.1-8B-Instruct，2026

- mean TTFT：162 ms
- mean TPOT：7.33 ms
- mean E2E：1,093 ms
- P99 TPOT：根据 chunked-prefill configuration 在 10-25 ms 之间变化。

这些是 NVIDIA 发布的参考点。它们会随 model size（70B 会是 3-5x）、hardware（H100 vs B200 ~3x）和 load 改变。

### Measurement trap

两个 2026 年最常用 benchmark tools 在同一次 run 上会给出不同 TPOT：

- **NVIDIA GenAI-Perf**：从 ITL 计算中排除 TTFT。ITL 从 token 2 开始。
- **LLMPerf**：包含 TTFT。ITL 从 token 1 开始。

对于一个 TTFT 500 ms、100 output tokens、总 decode 700 ms 的请求，GenAI-Perf 报告 `ITL = 700/99 = 7.07 ms`，LLMPerf 报告 `ITL = 1200/100 = 12.00 ms`。工具选择会改变数字。

永远说明使用哪个工具。永远公布定义。

### 构造 SLO

2026 年，面向消费者的 70B chat model 合理 SLO：

- TTFT P99 <= 800 ms。
- TPOT P99 <= 25 ms。
- 对 <300-token outputs，E2E P99 <= 3 s。
- Goodput target >= 99%。

Enterprise SLO 会收紧 TTFT（200-400 ms）并放宽 E2E。重点是把它们写下来，测量三者，并把 goodput 作为单一 composite 跟踪。

### 如何测量

- 运行真实流量或 realistic synthetic（LLMPerf with `--mean-input-tokens 800 --stddev-input-tokens 300 --mean-output-tokens 150`）。
- Benchmark run 的目标是 2x peak concurrency。
- 运行 30-50 iterations，对合并样本取 percentiles。
- 发布时附上 tool name、tool version、model、hardware、concurrency、prompt distribution。

## 使用它

`code/main.py` 是一个玩具 goodput calculator。生成 synthetic latency distribution，应用 SLO，并计算 goodput。它还会在同一个 trace 上展示 GenAI-Perf vs LLMPerf 的 TPOT 差异。

## 交付它

本课会产出 `outputs/skill-slo-goodput-gate.md`。给定 workload 和 SLO，它会生成 CI/CD-ready benchmark recipe，用 goodput 而不是 throughput gate deploys。

## 练习

1. 运行 `code/main.py`。生成带 1% tail spike 的 distribution。当你把 P99 TPOT 从 30 ms 收紧到 15 ms 时，goodput 如何变化？
2. 一个 vendor 报价“Llama 3.3 70B H100 上 15,000 tok/s”。信任前要问哪三个问题？
3. 为什么 chunked prefill 保护 P99 TPOT，但不保护 mean TPOT？
4. 为一个 voice assistant 构造 consumer SLO（first token 是听见，不是读到）。哪个 metric 最能被用户感知？
5. 阅读 LLMPerf README 和 GenAI-Perf docs。找出另外三个工具定义不一致的 metrics。

## 关键词汇

| 术语 | 大家常说 | 实际含义 |
|------|----------|----------|
| TTFT | “time to first token” | Queue + network + prefill；长 prompt 下由 prefill 主导 |
| TPOT | “time per output token” | 首 token 后每个 token 的 memory-bound decode cost |
| ITL | “inter-token latency” | 多数工具中与 TPOT 相同（不是全部，见 GenAI-Perf） |
| E2E | “end to end” | TTFT + TPOT * output_len；再加 response-side network |
| Throughput | “tok/s” | Fleet efficiency；没有 latency percentiles 时无用 |
| Goodput | “SLO-met rate” | 同时满足每个 SLO constraint 的请求比例 |
| P99 | “tail” | 1-in-100 worst-case latency；用户体验 metric |
| SLO multi-constraint | “the joint” | 三个 latency bounds 的 AND；任何一个违反即失败 |
| GenAI-Perf vs LLMPerf | “the tool trap” | 工具对 ITL 是否包含 TTFT 意见不一致 |

## 延伸阅读

- [NVIDIA NIM — LLM Benchmarking Metrics](https://docs.nvidia.com/nim/benchmarking/llm/latest/metrics.html) — TTFT、ITL、TPOT 的 canonical definition。
- [Anyscale — LLM Serving Benchmarking Metrics](https://docs.anyscale.com/llm/serving/benchmarking/metrics) — alternative definitions 和 measurement recipe。
- [BentoML — LLM Inference Metrics](https://bentoml.com/llm/inference-optimization/llm-inference-metrics) — real deployments 上的 applied measurement。
- [LLMPerf](https://github.com/ray-project/llmperf) — Ray-based open-source benchmark。
- [GenAI-Perf](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/client/src/c++/perf_analyzer/genai-perf/README.html) — NVIDIA benchmark tool。
- [MLPerf Inference](https://mlcommons.org/benchmarks/inference-datacenter/) — 行业接受的 goodput-based benchmark。
