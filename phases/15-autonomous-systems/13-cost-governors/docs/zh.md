# Action Budgets、Iteration Caps 与 Cost Governors

> 一个中型 e-commerce agent 的月度 LLM 成本，在团队启用“order-tracking”skill 后从 $1,200 跳到 $4,800。这不是 pricing bug。这是一个 agent 找到了新的 loop，并持续在里面花钱。Microsoft 的 Agent Governance Toolkit（2026 年 4 月 2 日）把针对这类问题的防御制度化：per-request `max_tokens`、per-task token 和 dollar budgets、per-day/month caps、iteration caps、tiered model routing、prompt caching、context windowing、昂贵 actions 上的 HITL checkpoints、budget breach 上的 kill switches。Anthropic 的 Claude Code Agent SDK 以不同名称发布了同样 primitives。Financial velocity limits——例如 10 分钟内超过 $50 就 cut access——能比 monthly caps 更快捕捉 loops。

**类型：** 学习
**语言：** Python（stdlib，layered cost-governor simulator）
**前置要求：** 阶段 15 · 10（Permission modes），阶段 15 · 12（Durable execution）
**时间：** ~60 分钟

## 问题

Autonomous agents 每一轮都在花真钱。聊天机器人的坏输出只是坏回复；agent 的坏 loop 是账单。业界记录的 failure mode 术语是“Denial of Wallet”——agent 不断 reasoning、不断 tool-calling、不断 billing，而什么都不会阻止它，因为系统没有为此设计。

修复方式不是一个数字，而是一组不同时间尺度和粒度上的限制：per-request、per-task、per-hour、per-day、per-month。设计良好的 stack 会在几分钟内捕捉 runaway loop，在几小时内捕捉 slow leak，在一天内捕捉 bad release。当 agent 是 long-horizon 且 autonomous 时，同一个 stack 也让预算仍然存在。

这是工程课：数学很简单，团队失败在纪律。下面这组 limits 要么出自 Microsoft Agent Governance Toolkit，要么出自 Anthropic Claude Code Agent SDK docs。

## 概念

### cost-governor stack

1. **每个 request 的 `max_tokens`。** 简单。防止单次调用发出无界 completion。
2. **Per-task token budget。** 整个 run 中不要超过 N tokens。达到 cap 就 hard stop。
3. **Per-task dollar budget。** 与 tokens 相同，但以货币计。Claude Code 中是 `max_budget_usd`。
4. **Per-tool call cap。** 不超过 N 次 `WebFetch` calls、N 次 `shell_exec` calls 等。
5. **Iteration cap（`max_turns`）。** agent loop 总 iterations；防止无限 reasoning loops。
6. **Per-minute / per-hour / per-day / per-month cap。** Rolling windows。在不同时间尺度捕捉 leaks。
7. **Financial velocity limit。** 例如“如果 10 分钟支出超过 $50，就 cut access”。在 monthly caps 触发前捕捉 loop-based burn。
8. **Tiered model routing。** 默认使用较小模型；只有当 classifier 判断任务需要时才升级到更大模型。
9. **Prompt caching。** System prompt 和稳定 context 存在 provider cache；重复发送的 token cost 接近零。
10. **Context windowing。** compaction / summarization 将 active context 保持在阈值以下；直接降低 token cost。
11. **昂贵 actions 上的 HITL checkpoints。** 在已知昂贵的 action（长 tool call、大下载、昂贵 model upgrade）之前，需要 human tap。
12. **Budget breach 上的 kill switch。** 当任何 cap 触发时，session abort。记录 cap；需要单独的 re-enable path。

### 为什么要 stack，而不是一个 cap

单个 monthly cap 只能在钱包已经没了之后捕捉 runaway agent。单个 per-request cap 什么 session-level 问题都捕捉不到。不同 failure modes 需要不同时间尺度：

- **Runaway loop**（agent stuck in a 5-second retry）：由 velocity limit 捕捉。
- **Slow leak**（agent 每个 task 做约 2x 预期工作）：由 daily cap 捕捉。
- **Bad release**（新版本使用 5x tokens）：由 weekly / monthly cap 捕捉。
- **Legitimate surge**（真实需求，不是 bug）：由 hour / day cap 捕捉，并有清晰 log。

### Claude Code 的 budget surface

Claude Code Agent SDK 暴露（公开 docs）：

- `max_turns` — iteration cap。
- `max_budget_usd` — dollar cap；breach 时 session abort。
- `allowed_tools` / `disallowed_tools` — tool allowlist 和 denylist。
- tool use 前的 hook points，用于 custom cost-accounting。

与 permission-mode ladder（第 10 课）结合。没有 `max_budget_usd` 的 `autoMode` session 是 ungoverned autonomy。Anthropic 明确把 Auto Mode 描述为需要 budget controls；classifier 与 cost 是正交的。

### EU AI Act、OWASP Agentic Top 10

Microsoft Agent Governance Toolkit 覆盖 OWASP Agentic Top 10 和 EU AI Act Article 14（human oversight）要求。在 EU 生产部署中，logging 和 cap enforcement 不是可选项。

### 已观察到的 $1,200 → $4,800 案例

Microsoft docs 中的真实案例：一个 e-commerce agent 在添加新 tool 后月度成本翻了三倍。该 tool 允许 agent 在每个 session 中轮询 order status。没有 loop detection。没有 per-tool cap。没有 week-over-week growth alert。修复方式是 per-tool cap 加 daily-growth alert。这是一个模板：每个新 tool surface 都是新的潜在 loop；每个新 tool 都需要自己的 cap 和 alert。

## 使用它

`code/main.py` 会模拟带有和不带 layered cost-governor stack 的 agent run。模拟 agent 在若干 turns 后漂移进 polling loop；layered stack 会在 velocity window 内捕捉它，而单个 monthly cap 可能要几天后才触发。

## 交付它

`outputs/skill-agent-budget-audit.md` 会审计拟议 agent deployment 的 cost-governor stack，并标记 missing layers。

## 练习

1. 运行 `code/main.py`。确认在 polling-loop trajectory 上，velocity limit 先于 iteration cap 触发。然后禁用 velocity limit，测量 iteration cap 捕捉前 agent “spends” 了多少。

2. 为 browser agent（第 11 课）设计 per-tool cap set。哪个 tool 需要最紧的 cap？哪个 tool 可以无界运行且没有风险？

3. 阅读 Microsoft Agent Governance Toolkit docs。列出 toolkit 命名的每种 cap type。将每种映射到一种 failure mode（runaway loop、slow leak、bad release、surge）。

4. 为一个现实的 overnight unattended run 估价（例如“triage 50 issues in a repo”）。把 `max_budget_usd` 设置为 point estimate 的 2x。解释为什么是 2x。

5. Claude Code 的 `max_budget_usd` 会在 session aggregate cost 上触发。设计一个你会外部执行的 complementary velocity limit。什么触发 cut-off，re-enable 是什么样子？

## 关键术语

| Term | What people say | What it actually means |
|---|---|---|
| Denial of Wallet | “Runaway bill” | agent loop 产生开销，且没有 cap 阻止它 |
| max_tokens | “Per-request cap” | 单个 completion size 的上限 |
| max_turns | “Iteration cap” | 一个 session 中 agent loop iterations 的上限 |
| max_budget_usd | “Dollar kill switch” | Session cost cap；breach 时 abort |
| Velocity limit | “Rate cap” | 短窗口内的支出限制（例如 $50 / 10 min） |
| Tiered routing | “Small model first” | 默认便宜模型；只有 classifier 认为需要时才升级 |
| Prompt caching | “Cached system prompt” | Provider-side cache 将重复发送 token cost 降到近零 |
| HITL checkpoint | “Human approval gate” | 昂贵 action 前需要 human tap |

## 延伸阅读

- [Anthropic Claude Code Agent SDK — agent loop and budgets](https://code.claude.com/docs/en/agent-sdk/agent-loop) — `max_turns`、`max_budget_usd`、tool allowlists。
- [Microsoft Agent Framework — human-in-the-loop and governance](https://learn.microsoft.com/en-us/agent-framework/workflows/human-in-the-loop) — cost-governor checkpoints。
- [Anthropic — Claude Managed Agents overview](https://platform.claude.com/docs/en/managed-agents/overview) — provider-side cost controls。
- [Anthropic — Prompt caching (Claude API docs)](https://platform.claude.com/docs/en/prompt-caching) — caching mechanics。
- [Anthropic — Measuring agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) — long-horizon agents 的 cost profile。
