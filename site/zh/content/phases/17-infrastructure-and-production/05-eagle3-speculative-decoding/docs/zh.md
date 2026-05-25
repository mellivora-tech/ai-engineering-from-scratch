# Production 中的 EAGLE-3 Speculative Decoding

> Speculative decoding 把一个快速 draft model 和 target model 配对。draft 提出 K 个 tokens；target 在一次 forward 中验证；被接受的 tokens 是免费的。到 2026 年，EAGLE-3 是 production-grade variant：它在 target model 的 hidden states 上训练 draft head，而不是在 raw tokens 上训练，把通用聊天中的 acceptance rate alpha 推到 0.6-0.8 区间。正确问题不是“draft 有多快”，而是“我的流量上 alpha 是多少？”如果 alpha 低于约 0.55，speculative decoding 在高并发下会净亏，因为每个被拒绝的 draft 都会带来第二次 target forward pass。本课教你先测 alpha，再开 flag。

**类型：** 学习
**语言：** Python（stdlib，玩具版 acceptance-rate simulator）
**前置要求：** 阶段 17 · 04（vLLM Serving Internals），阶段 10 · 18（Multi-Token Prediction）
**时间：** ~60 分钟

## 学习目标

- 说出 speculative decoding 的三代方案，并解释 EAGLE-3 相比 EAGLE-2 和 classic draft model 改变了什么。
- 定义 acceptance rate alpha，用 alpha 和 K（draft length）计算 expected speedup，并识别目标并发下的 break-even alpha。
- 解释为什么 speculative decoding 在 vLLM 2026 中是 opt-in（不是默认），以及为什么不测 alpha 就打开它是 production anti-pattern。
- 写出 measurement plan：用哪个 benchmark、哪个 prompt distribution、哪个 concurrency point、用哪个 metric 做 gate。

## 问题

Decode 是 memory-bound。在 H100 上运行 Llama 3.3 70B FP8 时，每个 decoded token 读取约 140 GB/s 的 weights，并输出一个 token。decode 期间 GPU compute 几乎空闲，bottleneck 是 HBM bandwidth，而不是 matmul throughput。

Speculative decoding 利用这个差距。用便宜的 draft model 生成 K 个 candidate tokens，然后让 target model 在单次 forward pass 中验证全部 K 个。每个 verified token 实际上是免费的（摊销到 target 本来就必须做的 batch-of-K forward 中）。

Classic draft-model approach 使用同系列的小模型（例如 Llama 3.2 1B 为 Llama 3.3 70B drafting）。它有效，但 acceptance rate 一般，因为小模型 distribution 偏离 target。EAGLE、EAGLE-2、EAGLE-3 则直接在 target model 的 internal states 上训练轻量 draft head，因此 draft distribution 更贴近 target。这就是 alpha 从 draft-model 的 0.4 提升到 EAGLE-3 的 0.6-0.8 的原因。

问题在于：EAGLE-3 在 vLLM 2026 中是 opt-in。必须显式设置 `speculative_config`。没有 flag，就没有加速。团队如果不在真实流量上测 alpha 就打开它，常常会看到 tail latency 变差，而不是变好。

## 概念

### Speculative decoding 真正买到什么

没有 spec decode 时，每个 token 的成本是一次 target forward。使用 draft length K 和 acceptance alpha 的 spec decode 时，每次 target forward 的 expected tokens 是 `1 + K * alpha`。speedup 是 `(1 + K * alpha) / (1 + epsilon)`，其中 epsilon 是 draft-plus-verify overhead。对于 K=5、alpha=0.7：`(1 + 5*0.7) / (1 + 0.1) = 4.5 / 1.1 = 4.1x`。真实世界数字集中在 2-3x，因为 production traffic 上 alpha 很少那么高，并且 high batch size 下 epsilon 会增长。

### 为什么 alpha 是唯一重要的 metric

Rejected tokens 不会消失，它们会强制 target 为第一个 rejected token 做第二次 forward。在 alpha 掉到 0.4 的 workload 上，你会支付 draft overhead、verification 和 re-roll。高并发下（比如 256 concurrent），decode batch 已经足够大，“target alone”和“target with verify”之间的 memory-bandwidth gap 变小。在大多数 2026 hardware 上，alpha 低于 0.55 时 spec decode 通常净亏。

Alpha 随 workload 变化。在 ShareGPT-style general chat 上，用 ShareGPT 训练的 EAGLE-3 达到 0.6-0.8。在 domain-specific traffic（code、medical、legal）上，用 general data 训练的 draft head 会掉到 0.4-0.6。训练 domain-specific draft head 可以恢复 alpha；相比 target finetuning，这是一个轻量、快速的训练任务。

### EAGLE generations 一览

- **Classic draft model**：同系列小模型。Alpha 0.3-0.5。基础设施简单：加载两个模型，draft 每次 target forward 前跑 K 次 forward。
- **EAGLE-1（2024）**：在 target hidden states（last layer）上训练的单个 draft head。Alpha ~0.5-0.6。在 target 之上增加少量参数开销。
- **EAGLE-2（2025）**：adaptive draft length 和 tree-based drafts（一次 target pass 验证多个 branches）。Alpha ~0.6-0.7。draft scheduler 更复杂。
- **EAGLE-3（2025-2026）**：draft head 在多个 target layers 上训练（不只是 last layer），alignment 更好。general chat 上 Alpha ~0.6-0.8。

### 2026 production recipe

1. 先 plain 交付 target model。测量目标并发下的 baseline TTFT、ITL、throughput。
2. 通过 vLLM `speculative_config` 启用 EAGLE-3 draft。重新运行 benchmark。
3. 记录 acceptance rate alpha。vLLM V1 以 `spec_decode_metrics.accepted_tokens_per_request` 报告它。除以 requested draft length 得到 alpha。
4. 如果 production traffic distribution 上 alpha < 0.55，禁用 spec decode，或训练 domain-specific EAGLE-3 draft。
5. 在 production concurrency 下重新运行。确认 P99 ITL 没有变差。

### Production 陷阱：P99 tail

Spec decode 会降低 mean ITL。如果不调参，P99 可能变差。Rejected drafts 会触发 two-pass sequence（draft + verify-fail + reroll）。在 full batch 下，这两个 passes 会串行化。看 P99 ITL，而不是 P50。

### EAGLE-3 已经部署在哪里

Google 在 2025 年把 speculative decoding 部署到 AI Overviews（同等质量，更快响应）。vLLM V1 把 `speculative_config` 作为文档化接口；V1 中的 N-gram GPU speculative decoding 是与 chunked prefill 兼容的 variant。SGLang 支持 EAGLE-3，并把它作为 prefix-heavy workloads 的推荐 draft path。

### 一行 break-even 数学

Expected speedup：`S(alpha, K) = (1 + K*alpha) / (1 + verify_overhead)`。令 `S = 1` 可解出 alpha：`alpha_breakeven = verify_overhead / K`。对于典型 verify_overhead ~0.15 和 K=5：`alpha_breakeven = 0.03`。但这是原始 decode 数学。高并发下 verify overhead 上升，decode batch 已经在 sequences 之间摊销 memory reads，所以实践中的 effective alpha_breakeven 会升到 ~0.45-0.55。

### 什么时候不要用 speculative decoding

- Batch-1 offline generation 且 latency 不重要。使用 plain target。
- 很短输出（低于 50 tokens）。Draft overhead 和 verify cost 主导。
- 没有 domain-trained draft head 的专业领域。Alpha 太低。
- vLLM v0.18.0 + draft-model spec decode + `--enable-chunked-prefill`。这个组合无法编译。文档中的例外是 V1 的 N-gram GPU spec decode。

## 使用它

`code/main.py` 在一系列 alpha values 和 draft lengths K 上模拟有无 speculative decoding 的 decode loop。它会打印 break-even alpha、measured speedup 和 tail behavior。用几个（alpha, K）组合运行它，精确观察 speculative decoding 从哪里开始不划算。

## 交付它

本课会产出 `outputs/skill-eagle3-rollout.md`。给定 target model、traffic distribution description 和 concurrency target，它会生成 staged EAGLE-3 rollout plan：benchmark baseline、enable config、measure alpha、用 alpha >= 0.55 gate，并观察 P99 ITL。

## 练习

1. 运行 `code/main.py`。K=5 时，要达到 2x speedup 需要什么 alpha？3x 呢？它对 verify_overhead 有多敏感？
2. 假设 production traffic 是 70% general chat、30% code。用 ShareGPT 训练的 EAGLE-3 在 general chat 上 alpha 0.7，在 code 上 alpha 0.4。blended alpha 是多少，spec decode 是否 net-positive？
3. 阅读 vLLM `speculative_config` 文档。说出三种 modes（draft model、EAGLE、N-gram），以及哪一种与 chunked prefill 兼容。
4. 启用 EAGLE-3 后，你看到 mean ITL 下降 25%，但 P99 ITL 上升 15%。诊断并提出缓解方案。
5. 计算 Llama 3.3 70B 的 EAGLE-3 draft head memory cost。它与把 Llama 3.2 1B 作为 classic draft 运行相比如何？

## 关键词汇

| 术语 | 大家常说 | 实际含义 |
|------|----------|----------|
| Speculative decoding | “draft plus verify” | 用便宜模型提出 K 个 tokens，在一次 target forward 中验证全部 K 个 |
| Acceptance rate alpha | “spec accept rate” | target 接受的 draft tokens 比例；唯一重要的 metric |
| Draft length K | “spec k” | 每次 target forward 前 draft 提出的 tokens 数；典型 4-8 |
| Verify overhead epsilon | “spec overhead” | verify-and-reroll 相比 plain target forward 的额外成本；随 batch 增长 |
| EAGLE-3 | “latest EAGLE” | 2025-2026 variant；在多个 target layers 上训练 draft head；general chat 上 alpha 0.6-0.8 |
| `speculative_config` | “vLLM spec config” | vLLM V1 中显式 opt-in；默认无加速 |
| N-gram spec decode | “N-gram draft” | GPU 侧使用 prompt 中 N-gram lookups 的 draft；与 chunked-prefill 兼容 |
| Break-even alpha | “no-op alpha” | spec decode 零 speedup 时的 alpha；需在 production concurrency 下观察 |
| Rejected-draft two-pass | “reroll cost” | draft reject 时两次 target forward；推高 P99 tail |

## 延伸阅读

- [vLLM — Speculative Decoding docs](https://docs.vllm.ai/en/latest/features/spec_decode/) — V1 中 `speculative_config` 和 chunked-prefill compatibility 的权威来源。
- [vLLM Speculative Config API](https://docs.vllm.ai/en/latest/api/vllm/config/speculative/) — 精确 field set。
- [EAGLE paper (arXiv:2401.15077)](https://arxiv.org/abs/2401.15077) — 原始 EAGLE draft-head formulation。
- [EAGLE-2 paper (arXiv:2406.16858)](https://arxiv.org/abs/2406.16858) — adaptive drafts 和 trees。
- [UC Berkeley EECS-2025-224](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2025/EECS-2025-224.html) — 使用 speculative decoding 的 efficient LLM system。
- [BentoML — Speculative Decoding](https://bentoml.com/llm/inference-optimization/speculative-decoding) — production rollout checklist。
