# LLM Routing Layer：LiteLLM、OpenRouter、Portkey

> Provider lock-in 很昂贵。不同 tool-calling workloads 适合不同模型。Routing gateways 提供单一 API surface、retries、failover、cost tracking 和 guardrails。2026 年三种 archetype 占主导：LiteLLM（open-source self-hosted）、OpenRouter（managed SaaS）、Portkey（production-grade，2026 年 3 月开源）。本课命名 decision criteria，并走通一个 stdlib routing gateway。

**类型：** 学习
**语言：** Python（stdlib，routing + failover + cost tracker）
**前置要求：** 阶段 13 · 02（function calling），阶段 13 · 17（gateways）
**时间：** ~45 分钟

## 学习目标

- 区分 self-hosted、managed 和 production-grade routing options。
- 实现一个 fallback chain，按定义好的优先级顺序在 provider failures 时 retry。
- 跨 providers 跟踪 per-request cost 和 token usage。
- 针对给定生产约束，在 LiteLLM、OpenRouter 和 Portkey 之间选择。

## 问题

provider routing 很重要的场景：

1. **Cost。** Claude Sonnet 的价格是 Haiku 的 3 倍。triage task 用 Haiku 足够；synthesis task 用 Sonnet 值得。按 request route。

2. **Failover。** OpenAI 出现糟糕一小时。每个 request 都失败。你希望自动 fallback 到 Anthropic，而无需 redeploy。

3. **Latency。** live chat UI 需要很快的 time-to-first-token。batch summarizer 不需要。按 latency SLA route。

4. **Compliance。** EU users 必须留在 EU regions。按 region route。

5. **Experimentation。** 对同一个 workload A/B 两个模型。按 test bucket route。

为每个 integration 手写这些是重复劳动。routing gateway 提供一个 OpenAI-compatible API，并处理其余部分。

## 概念

### OpenAI-compatible proxy shape

所有人都说 OpenAI-shape。routing gateway 暴露 `/v1/chat/completions`，接受 OpenAI schema，并在内部 proxy 到 Anthropic / Gemini / Cohere / Ollama / anything。client 不关心。

### Model aliases

你的代码不写 `claude-3-5-sonnet-20251022`，而是写 `our_smart_model`。gateway 把 alias 映射到真实模型。当 Anthropic 发布 Claude 4 时，你在 server-side 改 alias；代码不用碰。

### Fallback chains

```
primary: openai/gpt-4o
on 5xx: anthropic/claude-3-5-sonnet
on 5xx: google/gemini-1.5-pro
on 5xx: refuse
```

gateways 在 config 中定义这个。Retries 会计入 budget，避免 fallback cascades 让成本爆炸。

### Semantic caching

完全相同或近似相同的 prompts 命中 cache，而不是 provider。在重复 agent loops 上可节省 30% 到 60%。key 基于 embedding；near-identical prompts 共享 cache slot。

### Guardrails

Gateway-level：

- **PII redaction。** 发送 prompt 前做 regex 或 ML-based pass。
- **Policy violations。** 拒绝带 prohibited content 的 prompts。
- **Output filters。** scrub completions，防止 leaks。

Portkey 和 Kong 都提供 opinionated guardrails。LiteLLM 把它们作为 optional。

### Per-key rate limits

一个 API key = 一个 team。per-key budget 防止一个 team 吃掉共享 quota。大多数 gateway 都支持。

### Self-hosted vs managed trade-offs

| Factor | LiteLLM (self-hosted) | OpenRouter (managed) | Portkey (production) |
|--------|----------------------|----------------------|----------------------|
| Code | Open source, Python | Managed SaaS | Open source (Mar 2026) + managed |
| Setup | Deploy a proxy | Sign up | Either |
| Providers | 100+ | 300+ | 100+ |
| Billing | Your own keys | OpenRouter credits | Your own keys |
| Observability | OpenTelemetry | Dashboard | Full OTel + PII redaction |
| Best for | Teams that want full control | Rapid prototyping | Production with compliance |

当你有 SRE team 并需要 data sovereignty 时，LiteLLM 胜出。想要单一 subscription 且不维护 infra 时，OpenRouter 胜出。需要开箱 guardrails 和 compliance 时，Portkey 胜出。

### Cost tracking

每个 request 携带 `provider`、`model`、`input_tokens`、`output_tokens`。乘以 per-model per-token prices（来自 gateway 维护的 pricing sheet）。按 per-user / per-team / per-project aggregation。

### MCP plus routing

gateway 可以同时 route LLM calls 和 MCP sampling requests。当 sampling request 的 modelPreferences 偏好某个模型时，gateway 会翻译到正确 backend。这就是阶段 13 · 17（MCP gateway）和本课 routing gateway 有时会合并成一个 service 的地方。

### Routing strategies

- **Static priority。** 列表第一个；出错时 fallback。
- **Load balancing。** round-robin 或 weighted。
- **Cost-aware。** 选择满足 latency / quality 的最便宜模型。
- **Latency-aware。** 选择过去 N 分钟最快的模型。
- **Task-aware。** Prompt classifier 把 coding 路由到一个模型，summarization 路由到另一个。

## 使用它

`code/main.py` 用约 150 行实现 routing gateway：接受 OpenAI-shaped requests，翻译到 per-provider stubs，运行 priority fallback chain，跟踪 per-request cost，并对输入应用 PII redaction pass。运行三个 scenario：normal request、primary-provider outage 触发 fallback、PII leakage 被 redaction 捕获。

重点看：

- `ROUTES` dict：alias -> priority-ordered list of concrete providers。
- Fallback loop 在 5xx 上 retry。
- Cost tracker 把 token usage 乘以 per-model rates。
- PII redactor 在 forwarding 前 scrub SSN-shaped patterns。

## 交付它

本课产出 `outputs/skill-routing-config-designer.md`。给定 workload profile（latency、cost、compliance），这个 skill 会选择 LiteLLM / OpenRouter / Portkey，并生成 routing config。

## 练习

1. 运行 `code/main.py`。触发 outage scenario；确认 fallback 落到第二个 provider，并且 cost 被正确归因。

2. 添加 semantic caching：prompt 的 SHA256 作为 lookup key；cache hit 立即返回。测量 repeated call 上的 cost savings。

3. 添加 prompt classifier，把 “code ...” prompts 路由到偏向 intelligence 的 alias，把 “summarize ...” prompts 路由到偏向 speed 的 alias。

4. 设计 per-team budgets：每个 team 有 monthly spend cap；cap 命中后 gateway 拒绝 request。选择 enforcement granularity（per-request 或 windowed）。

5. 并排阅读 LiteLLM、OpenRouter 和 Portkey docs。说出每个都提供、另外两个没有的一个 feature。

## 关键词

| Term | 大家常说 | 实际含义 |
|------|----------|----------|
| Routing gateway | “LLM proxy” | 位于许多 providers 前面的 one-API-surface layer |
| OpenAI-compatible | “Speaks the OpenAI schema” | 接受 `/v1/chat/completions` 形状，并翻译到任意 backend |
| Model alias | “our_smart_model” | 代码中的名称，由 gateway 映射到具体模型 |
| Fallback chain | “Retry list” | 失败时按顺序尝试的 provider list |
| Semantic caching | “Prompt-embedding cache” | key 是 prompt embedding；near-duplicates 共享 cache hit |
| Guardrails | “Input/output filters” | redact PII，reject policy violations |
| Per-key rate limit | “Team budget” | 作用于 API key 的 quota |
| Cost tracking | “Per-request spend” | 聚合 token usage x model price |
| LiteLLM | “The open proxy” | self-hostable OSS routing gateway |
| OpenRouter | “The managed SaaS” | credit-based billing 的 hosted gateway |
| Portkey | “The production option” | built-in guardrails 的 open-source + managed gateway |

## 延伸阅读

- [LiteLLM — docs](https://docs.litellm.ai/) — self-hosted routing gateway
- [OpenRouter — quickstart](https://openrouter.ai/docs/quickstart) — managed routing SaaS
- [Portkey — docs](https://portkey.ai/docs) — 带 guardrails 的 production routing
- [TrueFoundry — LiteLLM vs OpenRouter](https://www.truefoundry.com/blog/litellm-vs-openrouter) — decision guide
- [Relayplane — LLM gateway comparison 2026](https://relayplane.com/blog/llm-gateway-comparison-2026) — vendor survey
