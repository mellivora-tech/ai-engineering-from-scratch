# Capstone 14 — Speculative-Decoding Inference Server

> vLLM 0.7 中的 EAGLE-3 在真实流量上带来 2.5-3x throughput。P-EAGLE（AWS 2026）把 parallel speculation 推得更远。SGLang 的 SpecForge 大规模训练 draft heads。Red Hat 的 Speculators hub 为常见 open models 发布 aligned drafts。TensorRT-LLM 让 speculative decoding 成为 NVIDIA 上的一等能力。2026 年的 production serving stack 是带 EAGLE-family drafts 的 vLLM 或 SGLang、FP8 或 INT4 quantization，以及基于 queue-wait 的 HPA。这个 capstone 要求你以 2.5x+ baseline throughput 服务两个 open models，并给出完整 tail-latency report。

**类型：** Capstone
**语言：** Python（serving）、C++ / CUDA（kernel inspection）、YAML（configs）
**前置要求：** 阶段 3（deep learning）、阶段 7（transformers）、阶段 10（LLMs from scratch）、阶段 17（infrastructure）
**覆盖阶段：** P3 · P7 · P10 · P17
**时间：** 30 小时

## 问题

Speculative decoding 在 2026 年已经商品化。EAGLE-3 draft heads 基于 target model 的 hidden states 训练，并预测未来 N 个 tokens；target model 在一次 pass 中验证。60-80% 的 acceptance rates 会转化成 2-3x end-to-end throughput。vLLM 0.7 原生集成了它。SGLang + SpecForge 提供训练 pipeline。Red Hat 的 Speculators 发布了 Llama 3.3 70B、Qwen3-Coder-30B MoE、GPT-OSS-120B 的 aligned drafts。

手艺在 serving operations，不在模型。Acceptance rate 会随 traffic distribution 漂移（ShareGPT vs code vs domain data）。rejection 下的 tail latency 比没有 speculation 更差，所以你必须报告多个 batch sizes 下的 p99，而不只是 steady-state tokens/sec。与 Anthropic / OpenAI API 比较 cost per 1M tokens，是可信度杠杆。

## 概念

Speculative decoding 有两层。**draft** model（EAGLE-3 head、ngram 或更小的 target-aligned model）每步提出 k 个 candidate tokens。**target** model 一次性验证所有 k 个；任何 accepted prefix 都会替换 greedy path。Acceptance rate 取决于 draft-target alignment 和输入分布。

EAGLE-3 在多数流量上胜过 ngram drafts。P-EAGLE 会运行 parallel speculation，构建更深的 draft trees。权衡是：rejection 时 P99 latency 更高，因为 verify pass 更大。serving config 必须报告按 batch-size 分桶的 latency，才能暴露这个问题。

Deployment 使用 Kubernetes。vLLM 0.7 每 GPU 或 tensor-parallel shard 运行一个 replica。HPA 按 queue-wait 而不是 CPU autoscale。FP8（Marlin）和 INT4（AWQ）quants 让 GPU memory 保持在 H100 / H200 envelope 内。end-to-end report 包括 throughput、acceptance rate、batch 1/8/32 下的 p50/p99，以及 $/1M tokens。

## 架构

```
request ingress
    |
    v
vLLM server (0.7) or SGLang (0.4)
    |
    +-- draft: EAGLE-3 heads | P-EAGLE parallel | ngram fallback
    +-- target: Llama 3.3 70B | Qwen3-Coder-30B | GPT-OSS-120B
    |     quantized FP8-Marlin or INT4-AWQ
    |
    v
verify pass: batch k draft tokens through target
    |
    v (accept prefix; resample for rejected suffix)
    v
token stream back to client
    |
    v
Prometheus metrics: throughput, acceptance rate, queue wait, latency p50/p99
    |
    v
HPA on queue-wait metric
```

## 技术栈

- Serving：vLLM 0.7 或 SGLang 0.4
- Speculative methods：EAGLE-3 draft heads、P-EAGLE parallel speculation、ngram fallback
- Draft training：SpecForge（SGLang）或 Red Hat Speculators
- Target models：Llama 3.3 70B、Qwen3-Coder-30B MoE、GPT-OSS-120B
- Quantization：FP8（Marlin）、INT4 AWQ
- Deployment：Kubernetes + NVIDIA device plugin；基于 queue-wait metric 的 HPA
- Eval：ShareGPT、MT-Bench-v2、GSM8K、HumanEval，用于 domain-spread acceptance measurement
- Reference：TensorRT-LLM speculative decoding，作为 vendor baseline

## 构建它

1. **Target model prep。** 选择 Llama 3.3 70B。通过 Marlin quantize 到 FP8。在 1xH100（或 2x tensor-parallel）上用 vLLM 0.7 部署。

2. **Draft source。** 从 Red Hat Speculators 拉取 aligned EAGLE-3 draft head（或用 SpecForge 训练一个）。加载到 vLLM 的 speculative-decoding config 中。

3. **Baseline numbers。** 启用 speculation 前：batch 1/8/32 下的 tokens/s、p50/p99 latency、GPU utilization。发布这些数字。

4. **Enable EAGLE-3。** 翻转 config；重新运行同一个 benchmark。报告 speedup、acceptance rate、p99 tail-latency delta。

5. **P-EAGLE。** 启用 parallel speculation；衡量 deeper draft tree vs serial EAGLE-3。报告 P-EAGLE 从有帮助变成有害的拐点。

6. **Domain traffic。** 把 ShareGPT、HumanEval 和 domain-specific traffic 跑过同一个 server。衡量每种 distribution 的 acceptance rate。识别 drafts 何时漂移。

7. **Second target model。** 在 Qwen3-Coder-30B MoE 上运行同一 pipeline。draft 更难（MoE routing noise）。报告结果。

8. **K8s HPA。** 在 K8s 下部署，并让 HPA 追踪 `queue_wait_ms`。演示 load triples 时的 scale-out。

9. **Cost comparison。** 在同一 eval 上计算 $/1M tokens，并与 Anthropic Claude Sonnet 4.7 和 OpenAI GPT-5.4 对比。发布。

## 使用它

```
$ curl https://infer.example.com/v1/chat/completions -d '{"messages":[...]}'
[serve]     vLLM 0.7, Llama 3.3 70B FP8, EAGLE-3 active
[decode]    bs=8, accepted_tokens_per_step=3.2, acceptance_rate=0.76
[latency]   first-token 42ms, full-response 980ms (620 tokens)
[cost]      $0.34 per 1M output tokens at sustained throughput
```

## 交付它

`outputs/skill-inference-server.md` 描述交付物：一个带 speculative decoding 的 measured serving stack、一份完整 benchmark report，以及一个 K8s deployment。

| 权重 | 标准 | 衡量方式 |
|:-:|---|---|
| 25 | Measured speedup vs baseline | 两个模型上 matched quality 的 2.5x+ throughput |
| 20 | Acceptance rate on realistic traffic | 按 distribution 的 acceptance-rate report |
| 20 | P99 tail-latency discipline | batch 1/8/32 下有无 speculation 的 p99 |
| 20 | Ops | K8s deploy、HPA on queue-wait、rollout smooth |
| 15 | Write-up and methodology | 清楚解释改变了什么以及为什么 |
| **100** | | |

## 练习

1. 衡量 draft 比 target 落后一个版本时的 acceptance-rate degradation（例如 Llama 3.3 -> 3.4 drift）。构建 monitoring alert。

2. 实现 ngram-fallback：如果 EAGLE-3 acceptance 低于阈值，就切换到 ngram drafts。报告 reliability improvement。

3. 运行 controlled MoE experiment：同一个 Qwen3-Coder-30B，分别注入 routing noise 和不注入。衡量 draft acceptance sensitivity。

4. 扩展到 H200（141 GB）。报告 model-size-per-replica headroom 的提升，以及是否能服务未量化的 Llama 3.3 70B。

5. 在同一 H100 硬件上基准测试 TensorRT-LLM speculative decoding。报告它在哪些地方赢过 vLLM。

## 关键词汇

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|------------------------|
| Draft model | “Speculator” | 为 target 提出 N 个 tokens 供其验证的小模型 |
| EAGLE-3 | “2026 draft architecture” | 基于 target hidden states 训练的 draft head；约 75% acceptance |
| P-EAGLE | “Parallel speculation” | 在一次 target pass 中验证的 draft branch tree |
| Acceptance rate | “Hit rate” | 无需 resampling 就被接受的 drafted tokens 比例 |
| Quantization | “FP8 / INT4” | 用更低精度 weights 在 GPU memory 中放下更多模型 |
| Queue wait | “HPA metric” | inference 开始前 request 在 pending queue 中等待的时间 |
| Speculators hub | “Aligned drafts” | Red Hat Neural Magic 为常见 open models 提供的 EAGLE drafts hub |

## 延伸阅读

- [vLLM EAGLE and P-EAGLE documentation](https://docs.vllm.ai) — reference serving stack
- [P-EAGLE (AWS 2026)](https://aws.amazon.com/blogs/machine-learning/p-eagle-faster-llm-inference-with-parallel-speculative-decoding-in-vllm/) — parallel speculative decoding paper + integration
- [SGLang SpecForge](https://github.com/sgl-project/SpecForge) — draft-head training pipeline
- [Red Hat Speculators](https://github.com/neuralmagic/speculators) — aligned draft hub
- [TensorRT-LLM speculative decoding](https://nvidia.github.io/TensorRT-LLM/) — vendor alternative
- [Fireworks.ai serving architecture](https://fireworks.ai/blog) — commercial reference
- [EAGLE-3 paper (arXiv:2503.01840)](https://arxiv.org/abs/2503.01840) — method paper
- [vLLM repository](https://github.com/vllm-project/vllm) — code and benchmarks
