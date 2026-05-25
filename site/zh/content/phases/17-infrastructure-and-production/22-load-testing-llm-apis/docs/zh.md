# Load Testing LLM APIs — 为什么 k6 和 Locust 会说谎

> 传统 load testers 不是为 streaming responses、variable output lengths、token-level metrics 或 GPU saturation 设计的。两个陷阱会咬到大多数团队。GIL trap：Locust 的 token-level measurement 在 Python GIL 下运行 tokenization，高并发时会与 request generation 竞争；tokenization backlog 随后膨胀 reported inter-token latency，bottleneck 是你的 client，不是 server。Prompt-uniformity trap：循环中的 identical prompts 只测试 token distribution 上的一个点；真实流量有 variable length 和多样 prefix matches。LLMPerf 用 `--mean-input-tokens` + `--stddev-input-tokens` 修复这一点。2026 年工具映射：LLM-specialized（GenAI-Perf、LLMPerf、LLM-Locust、guidellm）用于 token-level accuracy；**k6 v2026.1.0** + **k6 Operator 1.0 GA（2025 年 9 月）**—— streaming-aware，Kubernetes-native distributed via TestRun/PrivateLoadZone CRDs，最适合 CI/CD gates；Vegeta 用于 Go constant-rate saturation；Locust 2.43.3 只有搭配 LLM-Locust extension 才适合 streaming。Load patterns：steady-state、ramp、spike（autoscaling test）、soak（memory leaks）。

**类型：** 构建
**语言：** Python（stdlib，玩具版 realistic-prompt generator + latency collector）
**前置要求：** 阶段 17 · 08（Inference Metrics），阶段 17 · 03（GPU Autoscaling）
**时间：** ~75 分钟

## 学习目标

- 解释两个让 generic load testers 对 LLM APIs 说谎的 anti-patterns（GIL trap、prompt-uniformity trap）。
- 为给定目的选择工具：LLMPerf（benchmark run）、k6 + streaming extension（CI gate）、guidellm（large-scale synthetic）、GenAI-Perf（NVIDIA reference）。
- 设计四种 load patterns（steady、ramp、spike、soak），并说出每种捕获的 failure mode。
- 用 input tokens 的 mean + stddev 构建 realistic prompt distribution，而不是固定长度。

## 问题

你用 k6 在 500 concurrent users 下测试 LLM endpoint。它撑住了。你 ship。Production 中 200 个真实 users 时服务崩了：P99 TTFT 爆炸，GPUs 打满。

发生了两件事。第一，k6 发送 500 个 identical prompts：你的 request-coalescing 和 prefix caching 让它看起来像能处理 500 concurrent decodes，实际上只处理了一个。第二，k6 不会以眼睛感知的方式跟踪 streaming responses 的 inter-token latency；它看到一个 HTTP connection，而不是以不同间隔到达的 500 个 tokens。

LLM load testing 是自己的 discipline。

## 概念

### GIL trap（Locust）

Locust 使用 Python，并在 GIL 下 client-side 运行 tokenization。高并发时 tokenizer 会排在 request generation 后面。Reported inter-token latency 包含 client-side tokenization backlog。你以为 server 慢，其实是 test harness。

修复：LLM-Locust extension 把 tokenization 移到独立 processes，或使用 compiled-language harness（k6、使用 tokenizers.rs 的 LLMPerf）。

### Prompt-uniformity trap

所有已知 load testers 都允许你配置一个 prompt。在 10,000 iterations 的 loop test 中，每次发送完全相同的 prompt。Server 每次都看到同一 prefix，prefix cache hits 接近 100%，throughput 看起来很棒。

修复：从 prompt distribution 采样。LLMPerf 使用 `--mean-input-tokens 500 --stddev-input-tokens 150`：多样长度、多样内容。

### 四种 load patterns

1. **Steady-state** — constant RPS 持续 30-60 min。捕获：baseline performance regressions。
2. **Ramp** — 15 min 内从 0 线性增加到目标 RPS。捕获：capacity breakpoint、warm-up anomalies。
3. **Spike** — 突然 3-10x RPS 持续 2 min，然后恢复。捕获：autoscaling latency、queue saturation、cold-start impact。
4. **Soak** — steady-state 持续 4-8 hours。捕获：memory leaks、connection-pool drift、observability overflow。

### 2026 tool mapping

**LLMPerf**（Anyscale）— Python，但 tokenization 背后是 Rust。Mean/stddev prompts。Streaming-aware。Performance runs 的最佳默认选择。

**NVIDIA GenAI-Perf** — NVIDIA reference。使用 Triton client；metric coverage 全面。注意它的 ITL 排除 TTFT；LLMPerf 的包含它。两个工具会对同一 server 产生不同 TPOT。

**LLM-Locust**（TrueFoundry）— 修复 GIL trap 的 Locust extension。熟悉的 Locust DSL + streaming metrics。

**guidellm** — large-scale synthetic benchmarking。

**k6 v2026.1.0** + **k6 Operator 1.0 GA（2025 年 9 月）**：
- k6 本身（Go、compiled、无 GIL）增加了 streaming-aware metrics。
- k6 Operator 使用 TestRun / PrivateLoadZone CRDs 做 Kubernetes-native distributed testing。
- 最适合 CI/CD gates 和 SLA testing。

**Vegeta** — Go，比 k6 简单。Constant-rate HTTP saturation。不是 LLM-aware，但适合 gateway / rate-limit testing。

**Locust 2.43.3 stock** — 对 LLM 有 GIL trap。只有搭配 LLM-Locust extension 才可用。

### CI 中的 SLA gate

在 PR 上运行 k6：

- 每个 baseline RPS 下 30-50 iterations。
- Gate：P50/P95 TTFT、5xx < 5%、TPOT under threshold。
- 违规时 break build。

### Realistic prompt distribution

从真实流量样本构建（如果有），或从公开 distributions 构建（例如 chat 用 ShareGPT prompts，code 用 HumanEval）。把 mean + stddev 输入给 LLMPerf。无论如何都要避免 loop-with-one-prompt。

### 你应该记住的数字

- k6 Operator 1.0 GA：2025 年 9 月。
- k6 v2026.1.0：streaming-aware metrics。
- Typical LLMPerf run：concurrency X 下 100-1000 requests。
- Typical CI gate：每个 PR 30-50 iterations。
- 四种 patterns：steady、ramp、spike、soak。

## 使用它

`code/main.py` 模拟带 realistic prompt distribution 的 load test，测量 effective TPOT，并展示 uniform-prompt trap。

## 交付它

本课会产出 `outputs/skill-load-test-plan.md`。给定 workload 和 SLA，它会选择工具并设计四种 load patterns。

## 练习

1. 运行 `code/main.py`。比较 uniform vs realistic distribution：差距在哪里？
2. 写一个 k6 script 作为 CI gate：100 concurrent 下 TTFT P95 < 800 ms，runtime 5 minutes。
3. 你的 soak test 显示 memory 每小时增长 50 MB。说出三个原因和用于区分它们的 instrumentation。
4. Spike test 从 10 RPS 到 100 RPS。如果 Karpenter + vLLM production-stack 已就位（阶段 17 · 03 + 18），expected recovery time 是多少？
5. GenAI-Perf 在同一 server 上报告 TPOT=6ms；LLMPerf 报告 TPOT=11ms。解释。

## 关键词汇

| 术语 | 大家常说 | 实际含义 |
|------|----------|----------|
| LLMPerf | “the LLM harness” | Anyscale benchmark tool，streaming-aware |
| GenAI-Perf | “NVIDIA tool” | NVIDIA reference harness |
| LLM-Locust | “Locust for LLMs” | 修复 GIL trap 的 Locust extension |
| guidellm | “synthetic benchmark” | Large-scale synthetic tool |
| k6 Operator | “K8s k6” | CRD-based distributed k6 |
| GIL trap | “Python client overhead” | Tokenization backlog 膨胀 reported latency |
| Prompt-uniformity trap | “single-prompt lie” | 循环同一 prompt 命中 cache，膨胀 throughput |
| Steady-state | “constant load” | N 分钟 flat RPS |
| Ramp | “linear up” | 在 duration 内从 0 到 target |
| Spike | “burst test” | 突然 multiplier 后恢复 |
| Soak | “long test” | 持续数小时，用于 leak detection |

## 延伸阅读

- [TianPan — Load Testing LLM Applications](https://tianpan.co/blog/2026-03-19-load-testing-llm-applications)
- [PremAI — Load Testing LLMs 2026](https://blog.premai.io/load-testing-llms-tools-metrics-realistic-traffic-simulation-2026/)
- [NVIDIA NIM — Introduction to LLM Inference Benchmarking](https://docs.nvidia.com/nim/large-language-models/1.0.0/benchmarking.html)
- [TrueFoundry — LLM-Locust](https://www.truefoundry.com/blog/llm-locust-a-tool-for-benchmarking-llm-performance)
- [LLMPerf](https://github.com/ray-project/llmperf)
- [k6 Operator](https://github.com/grafana/k6-operator)
