# LLMs 的 FinOps — Unit Economics 和 Multi-Tenant Attribution

> 传统 FinOps 在 LLM spend 上会失效。成本是 token-transactions，不是 resource-uptime。Tags 映射不上：API call 是 transaction，不是 asset。工程决策（prompt design、context window、output length）就是财务决策。2026 playbook 有三个 attribution dimensions，第一天就要 instrument：per-user（`user_id`）用于 seat pricing 和 expansion，per-task（`task_id` + `route`）用于 product surface cost 和 prioritization，per-tenant（`tenant_id`）用于 unit economics 和 renewal。四个 token layers：prompt、tool、memory、response；一个 bucket 会隐藏 spend。Multi-tenant products 的 enforcement ladder：每 tenant rate limits（2-3x expected peak，清晰 429 + retry-after）；daily spend cap（1.5-3x contracted ceiling；触发 rate tightening + alert）；spend z-score > 4 时 kill switches（auto-pause + page on-call）。Attribution patterns：tag-and-aggregate、telemetry-joiner（trace-ID → billing；最高 accuracy）、sampling-and-extrapolation、model-based allocation、event-sourced、real-time streaming。Unit metric：cost per resolved query、cost per generated artifact，而不是 $/M tokens。Retroactive tagging 永远会漏；在 request creation 时 instrument。

**类型：** 学习
**语言：** Python（stdlib，带 kill switch 的玩具 cost-attribution simulator）
**前置要求：** 阶段 17 · 13（Observability），阶段 17 · 14（Caching）
**时间：** ~60 分钟

## 学习目标

- 解释为什么传统 FinOps（tags + tiers）在 LLM spend 上失效，并说出三个新的 attribution dimensions。
- 枚举四个 token layers（prompt、tool、memory、response），以及为什么 single-bucket billing 会隐藏成本。
- 为 multi-tenant product 设计 enforcement ladder（rate → spend cap → kill switch）。
- 选择 unit metric（cost per resolved query / artifact），而不是 $/M tokens。

## 问题

你的账单显示 $40,000。你不知道：
- 哪个 tenant 花了它。
- 哪个 product feature 驱动了它。
- 是否有 individual user 在滥用。
- culprit 是 prompt bloat、tool calls，还是 memory amplification。

Provider-side 的 tag-and-aggregate 适用于 cloud resources（EC2、S3），因为 tags 会传播到 line items。LLM API calls 不会自动 tag：你必须在 call site 标记 user/task/tenant 并携带下去。Retroactive attribution 永远会漏 edge cases。

## 概念

### 三个 attribution dimensions

**Per-user**（`user_id`）：谁在造成什么成本。驱动 seat pricing、expansion conversations、识别 power users。

**Per-task**（`task_id` + `route`）：哪个 product surface 花费多少。驱动 feature prioritization、kill-expensive-features decisions。

**Per-tenant**（`tenant_id`）：哪个 customer 盈利。驱动 unit economics、renewal pricing、tier thresholds。

第一天就在 call site instrument 全部三者。Retroactive 永远更差。

### 四个 token layers

| Layer | Example | Typical % of total |
|-------|---------|---------------------|
| Prompt | system + user input | 40-60% |
| Tool | tool-call results fed back | 20-40% (agent workloads) |
| Memory | prior conversation / retrieved docs | 10-30% |
| Response | model output | 10-30% |

把四者全放进一个 bucket 会让优化变盲。把它们拆到 attribution schema 中。

### Enforcement ladder

1. **Rate limit** per tenant。2-3x expected peak。返回带 `Retry-After` 的 429。Tenant 感受到 friction，但没有 surprise bill。

2. **Daily spend cap** per tenant。1.5-3x contracted ceiling。触发：收紧 rate limit + alert customer-success。

3. **Kill switch** on spend z-score > 4 relative to tenant baseline。Auto-pause tenant；page on-call；escalate to ops + CS。

### Attribution patterns

- **Tag-and-aggregate**：stamp metadata headers；后续 aggregate。简单但粗糙。
- **Telemetry joiner**：通过 trace IDs 把 traces join 到 billing。最高 accuracy。成熟团队使用。
- **Sampling + extrapolation**：sample 5-10%，乘回来。适合粗略 spend；会漏 tails。
- **Model-based allocation**：用 regression 推断 cost driver。适用于没有 tags 的 legacy data。
- **Event-sourced**：成本作为 stream 中的 events（Kafka / Kinesis）。Real-time。
- **Real-time streaming**：dashboard sub-second 更新。

### Cost per X 是 unit metric

$/M tokens 是 vendor 语言。Product metrics：

- Cost per resolved support ticket。
- Cost per generated article。
- Cost per successful agent task。
- Cost per user-session-minute。

把成本绑定到 product outcome。否则优化没有锚点。

### Cost attribution trace shape

```
trace_id: abc123
  user_id: u_42
  tenant_id: t_7
  task_id: task_classify_doc
  route: model_haiku
  layers:
    prompt_tokens: 1800
    tool_tokens: 600
    memory_tokens: 400
    response_tokens: 150
  cost_usd: 0.0135
  cached_input: true
  batch: false
```

每次 call 都 emit。存入 data lake。按维度 aggregate。阶段 17 · 13 observability stack 就是它存在的地方。

### Compounded-savings stack

Stack：cache + batch + route + gateway。四者全开：
- Cache L2（阶段 17 · 14）：input 便宜约 10x。
- Batch（阶段 17 · 15）：50% off。
- Route to cheap model（阶段 17 · 16）：60% cost reduction。
- Gateway efficiency（阶段 17 · 19）：redundancy + retries。

Best-case stacked：约 naive baseline 的 5-10%。多数团队启用了 2-3 个 levers；很少把四个全叠上。

### 你应该记住的数字

- Attribution dimensions：per-user、per-task、per-tenant。
- 四个 token layers：prompt、tool、memory、response。
- Kill switch：spend z-score > 4。
- Unit metric：cost per resolved query，不是 $/M tokens。
- Stacked optimizations：可能达到 baseline 的约 5-10%。

## 使用它

`code/main.py` 模拟一个带三层 enforcement ladder 的 multi-tenant LLM service。注入 abusive tenant，并展示 kill switch firing。

## 交付它

本课会产出 `outputs/skill-finops-plan.md`。给定 product 和 scale，它会设计 attribution schema 和 enforcement ladder。

## 练习

1. 运行 `code/main.py`。Kill switch 在什么 z-score 触发？如何选择 threshold？
2. 设计一个 per-tenant、per-task cost dashboard。最先构建哪 5 个 views？
3. 你最大的 tenant 是 unit-economics-negative。按 customer impact 从低到高提出三个 interventions。
4. 为一个 support product 计算 cost per resolved ticket：每 ticket 3M tokens、约 800 tickets/day、GPT-5 cached rate。
5. 论证 retroactive tagging 是否曾经可行。什么时候可以接受？

## 关键词汇

| 术语 | 大家常说 | 实际含义 |
|------|----------|----------|
| Per-user attribution | “user-level cost” | 每个 call 都 stamped `user_id` |
| Per-task attribution | “feature cost” | `task_id` + `route` 识别 product surface |
| Per-tenant attribution | “customer cost” | `tenant_id`；驱动 unit economics |
| Four token layers | “cost layers” | prompt + tool + memory + response |
| Rate limit | “429 guard” | 在 gateway 执行的 per-tenant ceiling |
| Daily spend cap | “daily ceiling” | Tenant-scoped budget with alert |
| Kill switch | “auto-pause” | Spend z-score > 4 触发 auto-suspension |
| Cost per resolved | “product unit metric” | 与 product outcome 绑定的成本，不是 tokens |
| Telemetry joiner | “trace-to-billing” | 最高 accuracy attribution pattern |
| Stacked optimization | “cache+batch+route+gateway” | 叠加 savings 到 baseline 的约 5-10% |

## 延伸阅读

- [FinOps Foundation — FinOps for AI Overview](https://www.finops.org/wg/finops-for-ai-overview/)
- [FinOps School — Cost per Unit 2026 Guide](https://finopsschool.com/blog/cost-per-unit/)
- [Digital Applied — LLM Agent Cost Attribution 2026](https://www.digitalapplied.com/blog/llm-agent-cost-attribution-guide-production-2026)
- [PointFive — Managed LLMs in Azure OpenAI](https://www.pointfive.co/blog/finops-for-ai-economics-of-managed-llms-in-azure-open-ai)
