# AI Gateways — LiteLLM、Portkey、Kong AI Gateway、Bifrost

> Gateway 位于你的 apps 和 model providers 之间。核心功能包括 provider routing、fallback、retries、rate limiting、secret references、observability、guardrails。2026 年市场分化：**LiteLLM** 是 MIT OSS，100+ providers，OpenAI-compatible，但在约 2000 RPS 附近崩掉（公开 benchmarks 中 8 GB memory、cascading failures）；最适合 Python、<500 RPS、dev/prototyping。**Portkey** 定位 control plane（guardrails、PII redaction、jailbreak detection、audit trails），2026 年 3 月 Apache 2.0 open-source，20-40 ms latency overhead，production tier $49/mo。**Kong AI Gateway** 基于 Kong Gateway：Kong 自己在相同 12 CPUs 上的 benchmark 中，比 Portkey 快 228%，比 LiteLLM 快 859%；$100/model/month pricing（Plus tier 最多 5 个）；如果你已经在 Kong 上，它适合 enterprise。**Bifrost**（Maxim AI）：automatic retries with configurable backoff，OpenAI 429 时 fallback to Anthropic。**Cloudflare / Vercel AI Gateways**：managed、zero-ops、basic retry。Data residency 驱动 self-host 决策；Portkey 和 Kong 位于中间，OSS + optional managed。

**类型：** 学习
**语言：** Python（stdlib，玩具版 gateway-routing simulator）
**前置要求：** 阶段 17 · 01（Managed LLM Platforms），阶段 17 · 16（Model Routing）
**时间：** ~60 分钟

## 学习目标

- 枚举六个核心 gateway features（routing、fallback、retries、rate limits、secrets、observability、guardrails）。
- 把四个 2026 gateways（LiteLLM、Portkey、Kong AI、Bifrost）映射到 scale ceilings 和 use cases。
- 引用 Kong benchmark（相对 Portkey 228%，相对 LiteLLM 859%），并解释为什么它对 >500 RPS 重要。
- 在给定 data residency 和 ops budget 时选择 self-hosted vs managed。

## 问题

你的产品调用 OpenAI、Anthropic 和 self-hosted Llama。每个 provider 都有不同 SDK、error model、rate limit 和 auth scheme。你想要 failover（如果 OpenAI 429，就试 Anthropic）、单一 credential store、统一 observability，以及 per tenant rate limits。

在 app layer 重新发明这些，会把每个 service 耦合到每个 provider。Gateway layer 把它合并到一个 process 中，提供一个 API（通常 OpenAI-compatible），再扇出到 providers。

## 概念

### 六个核心功能

1. **Provider routing** — OpenAI、Anthropic、Gemini、self-hosted 等在一个 API 后面。
2. **Fallback** — 遇到 429、5xx 或 quality failure 时，换地方 retry。
3. **Retries** — exponential backoff，bounded attempts。
4. **Rate limits** — per-tenant、per-key、per-model。
5. **Secret references** — runtime 从 vault 拉取 credentials（绝不放 app 中）。
6. **Observability** — OTel + GenAI attributes（阶段 17 · 13）+ cost attribution。
7. **Guardrails** — PII redaction、jailbreak detection、allowed-topics filters。

### LiteLLM — MIT OSS、Python

- 100+ providers、OpenAI-compatible、router config、fallback、basic observability。
- 在 Kong benchmark 中约 2000 RPS 崩掉；8 GB memory footprint，持续负载下 cascading failures。
- Best fit：Python app、<500 RPS、dev/staging gateways、experimental routing。
- Cost：OSS $0；cloud free tier 存在。

### Portkey — control plane positioning

- 自 2026 年 3 月起 Apache 2.0 OSS。Guardrails、PII redaction、jailbreak detection、audit trails。
- 每 request 20-40 ms latency overhead。
- Production tier $49/mo，带 retention + SLA。
- Best fit：需要 bundled guardrails + observability 的 regulated industries。

### Kong AI Gateway — scale play

- 基于 Kong Gateway（成熟 API gateway product，lua+OpenResty）。
- Kong 自己在 12-CPU equivalent 上的 benchmark：比 Portkey 快 228%，比 LiteLLM 快 859%。
- Pricing：$100/model/month，Plus tier 最多 5 个。
- Best fit：已经在 Kong 上；>1000 RPS；愿意 license。

### Bifrost（Maxim AI）

- Automatic retries with configurable backoff。
- OpenAI 429 时 fallback to Anthropic 是 canonical recipe。
- 较新的 entrant；commercial。

### Cloudflare AI Gateway / Vercel AI Gateway

- Managed、zero-ops。Basic retry 和 observability。
- Best fit：部署在 Cloudflare/Vercel 上的 Edge-serving JavaScript apps。
- 相比 Kong/Portkey，在 guardrails 和 rate limits 上有限。

### Self-hosted vs managed

Data residency 是 forcing function。Healthcare 和 finance 默认 self-host（LiteLLM、Portkey OSS 或 Kong）。Consumer products 默认 managed（Cloudflare AI Gateway）或 middle-tier（Portkey managed）。Hybrid：regulated tenant self-hosted，其他 managed。

### Latency budget

- LiteLLM：typical overhead 5-15 ms。
- Portkey：20-40 ms overhead。
- Kong：3-8 ms overhead。
- Cloudflare/Vercel：1-3 ms overhead（edge advantage）。

Gateway latency 会直接加到 TTFT 上。对于 TTFT P99 < 100 ms SLA，选 Kong 或 Cloudflare。对于 P99 < 500 ms，哪个都可以。

### Rate-limit semantics 很重要

简单 token-bucket 在中等 scale 前都可用。Multi-tenant 需要 sliding-window + burst allowance + per-tenant tiering。LiteLLM 提供 token-bucket；Kong 提供 sliding-window；Portkey 提供 tiered。

### Gateway + observability + routing 可以组合

阶段 17 · 13（observability）+ 16（model routing）+ 19（gateways）在 production 中是同一层。选择一个覆盖三者的工具，或仔细 wiring：多数 2026 deployments 会组合 Helicone（observability）或 Portkey（guardrails）与 Kong（scale），分担角色。

### 你应该记住的数字

- LiteLLM：约 2000 RPS 崩掉，8 GB memory。
- Portkey：20-40 ms overhead；自 2026 年 3 月 Apache 2.0。
- Kong：比 Portkey 快 228%，比 LiteLLM 快 859%。
- Kong pricing：$100/model/month，Plus tier 最多 5 个。
- Cloudflare/Vercel：edge 上 1-3 ms overhead。

## 使用它

`code/main.py` 在 429/5xx injection 下模拟跨 3 providers 的 gateway routing with fallback。报告 latency、retry rate 和 fallback hit rate。

## 交付它

本课会产出 `outputs/skill-gateway-picker.md`。给定 scale、ops posture、compliance、latency budget，它会选择 gateway。

## 练习

1. 运行 `code/main.py`。配置 OpenAI→Anthropic→self-hosted fallback。在 5% provider error rate 下 expected hit rate 是多少？
2. 你的 SLA 是 TTFT P99 < 200 ms，baseline 为 300 ms。哪些 gateways 仍在 budget 内？
3. 一个 healthcare customer 要求 self-hosted + PII redaction + audit。选择 Portkey OSS 或 Kong。
4. 比较 LiteLLM vs Kong：团队应该在什么 RPS ceiling 迁移？
5. 为 multi-tenant SaaS 设计 rate-limit policy：free tier、trial tier、paid tier。用 token-bucket 还是 sliding-window？

## 关键词汇

| 术语 | 大家常说 | 实际含义 |
|------|----------|----------|
| Gateway | “API broker” | 位于 apps 和 providers 之间的 process |
| LiteLLM | “the MIT one” | Python OSS，100+ providers，2K RPS 崩掉 |
| Portkey | “guardrails gateway” | Control plane + observability，Apache 2.0 |
| Kong AI Gateway | “the scale one” | 基于 Kong Gateway，benchmark leader |
| Bifrost | “Maxim's gateway” | Retries + Anthropic fallback recipe |
| Cloudflare AI Gateway | “edge managed” | Edge-deployed managed gateway，zero-ops |
| PII redaction | “data scrub” | 发送给 model 前用 regex + NER mask |
| Jailbreak detection | “prompt injection guard” | 用户输入上的 classifier |
| Audit trail | “regulated log” | 每个 LLM call 的 immutable record |
| Token-bucket | “simple rate limit” | 基于 refill 的 rate limiter |
| Sliding-window | “precise rate limit” | Time-windowed rate limiter；fairness 更好 |

## 延伸阅读

- [Kong AI Gateway Benchmark](https://konghq.com/blog/engineering/ai-gateway-benchmark-kong-ai-gateway-portkey-litellm)
- [TrueFoundry — AI Gateways 2026 Comparison](https://www.truefoundry.com/blog/a-definitive-guide-to-ai-gateways-in-2026-competitive-landscape-comparison)
- [Techsy — Top LLM Gateway Tools 2026](https://techsy.io/en/blog/best-llm-gateway-tools)
- [LiteLLM GitHub](https://github.com/BerriAI/litellm)
- [Portkey GitHub](https://github.com/Portkey-AI/gateway)
- [Kong AI Gateway docs](https://docs.konghq.com/gateway/latest/ai-gateway/)
