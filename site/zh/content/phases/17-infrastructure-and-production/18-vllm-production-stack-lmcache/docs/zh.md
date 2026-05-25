# 使用 LMCache KV Offloading 的 vLLM Production Stack

> vLLM 的 production-stack 是 reference Kubernetes deployment：router、engines 和 observability 被接在一起。LMCache 是 KV-offloading 层，会把 KV cache 从 GPU memory 中抽出，并跨 queries 和 engines 复用（先 CPU DRAM，再 disk/Ceph）。vLLM 0.11.0 KV Offloading Connector（2026 年 1 月）通过 Connector API（v0.9.0+）让这一过程 asynchronous 且 pluggable。Offload latency 不直接面向用户。即使没有 shared prefixes，LMCache 也有价值：当 GPU 用完 KV slots，preempted requests 可以从 CPU 恢复，而不是重新计算 prefill。在跨 4 台 a3-highgpu-4g 的 16x H100（80GB HBM）公开 benchmark 中，当 KV cache 超过 HBM 时，native CPU offload 和 LMCache 都显著提升 throughput；在低 KV footprint 下，所有 configs 与 baseline 接近，只带小 overhead。

**类型：** 学习
**语言：** Python（stdlib，玩具版 KV-spill simulator）
**前置要求：** 阶段 17 · 04（vLLM Serving Internals），阶段 17 · 06（SGLang/RadixAttention）
**时间：** ~60 分钟

## 学习目标

- 画出 vLLM production-stack layers：router、engines、KV offload、observability。
- 解释 KV Offloading Connector API（v0.9.0+），以及 0.11.0 asynchronous path 如何隐藏 offload latency。
- 量化 LMCache CPU-DRAM 何时有帮助（KV > HBM），何时只是增加 overhead（KV 小到能放进 HBM）。
- 在给定 deployment constraints 下，在 native vLLM CPU offload 和 LMCache connector 之间选择。

## 问题

你的 vLLM serving 显示 GPUs 在 concurrency 上升时 HBM 100%，并出现 preemption events。Requests 被 evict、requeue，然后你在一分钟内对同一个 2K-token prompt 重新 prefill 四次。GPU compute 花在冗余 prefill 上；goodput 远低于 raw throughput。

增加 GPUs 成本线性增长。增加 HBM 不可能。但 CPU DRAM 很便宜：一个 socket 有 512 GB+，latency 比 HBM 差几个数量级，但对“temporarily warm”的 KV cache 来说足够。

LMCache 把 KV cache 抽出到 CPU DRAM，让 preempted requests 快速恢复，并让 engines 之间重复 prefixes 共享 cache，不必每个 engine 都重新 prefill。

## 概念

### vLLM production-stack

`github.com/vllm-project/production-stack` 是 reference Kubernetes deployment：

- **Router** — cache-aware（阶段 17 · 11）。消费 KV events。
- **Engines** — vLLM workers。每个 GPU 一个，或每个 TP/PP group 一个。
- **KV cache offload** — LMCache deployment 或 native connector。
- **Observability** — Prometheus scrape、Grafana dashboards、OTel traces。
- **Control plane** — service discovery、config、rolling updates。

以 Helm chart + operator 形式发布。

### KV Offloading Connector API（v0.9.0+）

vLLM 0.9.0 引入 Connector API，用于 pluggable KV cache backends。Engine 把 blocks offload 到 connector；connector 存储它们（RAM、disk、object storage、LMCache）。Request 需要 block 时，connector 把它 load back。

vLLM 0.11.0（2026 年 1 月）增加 asynchronous offload path：common case 下 offload 可以在后台发生，engine 不被阻塞。End-to-end latency 和 throughput 仍取决于 workload shape、KV cache hit rate 和 system pressure；vLLM 自己的 notes 指出 custom-kernel offload 在 low hit rates 下可能降低 throughput，async scheduling 与 speculative decoding 也有已知 interaction issues。

### Native CPU offload vs LMCache

**Native vLLM CPU offload**：engine-local。把 KV blocks 存在 host RAM 中。实现快，零 network hop。不跨 engines。

**LMCache connector**：cluster-scale。把 blocks 存在共享 LMCache server（CPU DRAM + Ceph/S3 tier）。任何 engine 都可访问 blocks。已有 16x H100 benchmark 发布。

当单个 engine 有 HBM pressure 时选 native。当多个 engines 共享 prefixes（带 common system prompts 的 RAG、多 tenant 共享 templates）时选 LMCache。

### Benchmark behavior

跨 4 台 a3-highgpu-4g 的 16x H100（80 GB HBM）测试：

- 低 KV footprint（短 prompts、低 concurrency）：所有 configs 与 baseline 相同，LMCache 增加约 3-5% overhead。
- 中等 footprint：LMCache 开始在跨 engines prefix reuse 上提供帮助。
- KV 超过 HBM：native CPU offload 和 LMCache 都显著提升 throughput；LMCache 增益更大，因为 cross-engine sharing。

### LMCache 何时决定性

- Multi-tenant serving，tenants 之间共享 system prompts。
- RAG，document chunks 在 queries 之间重复。
- 同一 base 上的 fine-tuned variants（LoRA），base-model KV reuse 减少冗余工作。
- Preemption-heavy workloads：从 CPU restore 比重新 prefill 更便宜。

### 什么时候不要启用

- HBM pressure 很小：付出 overhead 但没有收益。
- Short contexts（<1K tokens）：transfer time > re-prefill。
- Single-tenant single-prompt workload：没有 reuse 可捕获。

### 与 disaggregated serving 集成

阶段 17 · 17 的 disaggregated serving + LMCache 会叠加：prefill pool 到 decode pool 的 KV transfer，如果未被使用会落进 LMCache；后续 queries 从 LMCache 拉取。阶段 17 · 11 的 cache-aware router 可以路由到 local 或 LMCache-shared cache 匹配的 engine。

### 你应该记住的数字

- vLLM 0.9.0：Connector API 发布。
- vLLM 0.11.0（2026 年 1 月）：asynchronous offload path；end-to-end latency impact 取决于 workload、KV hit rate 和 system pressure（不是绝对保证）。
- 16x H100 benchmark：KV footprint 超过 HBM 时 LMCache 有帮助。
- 小 HBM pressure：3-5% overhead，无收益。

## 使用它

`code/main.py` 模拟有无 LMCache 的 preemption-heavy workload。报告 avoided re-prefills、throughput gain 和 break-even HBM utilization。

## 交付它

本课会产出 `outputs/skill-vllm-stack-decider.md`。给定 workload shape 和 vLLM deployment，它会决定 native vs LMCache vs neither。

## 练习

1. 运行 `code/main.py`。LMCache 从哪个 HBM utilization 开始划算？
2. 一个 tenant 每小时 200 queries，共享 6K-token system prompt。计算每个 tenant 的 expected LMCache savings。
3. LMCache server 是 single point of failure。设计 HA strategy（replicas、fallback to native）。
4. LMCache 存储到 spinning disk 上的 Ceph。对于 70B FP8 上 4K-token KV（500 MB），read time 与 re-prefill 相比如何？
5. 论证 vLLM 0.11.0 asynchronous path 是否“free”：overhead 藏在哪里？

## 关键词汇

| 术语 | 大家常说 | 实际含义 |
|------|----------|----------|
| Production-stack | “the reference deployment” | vLLM 的 Kubernetes Helm chart + operator |
| Connector API | “KV backend interface” | vLLM 0.9.0+ pluggable KV store interface |
| Native CPU offload | “engine-local spill” | 在同一个 engine 的 host RAM 中存 KV |
| LMCache | “cluster KV cache” | CPU DRAM + disk 上的 cross-engine KV cache server |
| 0.11.0 async | “non-blocking offload” | Offload hidden behind engine stream |
| Preemption | “evict to make room” | HBM 满时的 KV cache shuffle |
| Prefix reuse | “same system prompt” | 多个 queries 共享开头；cache hit |
| Ceph tier | “disk tier” | Cache hierarchy 中 DRAM 下方的 durable storage |

## 延伸阅读

- [vLLM Blog — KV Offloading Connector (Jan 2026)](https://blog.vllm.ai/2026/01/08/kv-offloading-connector.html)
- [vLLM Production Stack GitHub](https://github.com/vllm-project/production-stack) — Helm chart + operator。
- [LMCache for Enterprise-Scale LLM Inference (arXiv:2510.09665)](https://arxiv.org/html/2510.09665v2)
- [LMCache GitHub](https://github.com/LMCache/LMCache) — Connector implementation。
- [vLLM 0.11.0 release notes](https://github.com/vllm-project/vllm/releases) — asynchronous path details。
