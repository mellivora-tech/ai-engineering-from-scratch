# Human-in-the-Loop：Propose-Then-Commit

> 2026 年关于 HITL 的共识很具体。它不是“agent 提问，用户点击 Approve”。它是 propose-then-commit：拟议 action 会以 idempotency key 持久化到 durable store；连同 intent、data lineage、permissions touched、blast radius 和 rollback plan 展示给 reviewer；只有 positive acknowledgement 后才 commit；执行后 verify 以确认副作用确实发生。LangGraph 的 `interrupt()` 加 PostgreSQL checkpointing、Microsoft Agent Framework 的 `RequestInfoEvent`、Cloudflare 的 `waitForApproval()` 都实现了同一形状。典型 failure mode 是 rubber-stamp approval：“Approve?” 在没有 review 的情况下被点击。已记录的 mitigation 是带 explicit checklist 的 challenge-and-response。

**类型：** 学习
**语言：** Python（stdlib，propose-then-commit state machine with idempotency）
**前置要求：** 阶段 15 · 12（Durable execution），阶段 15 · 14（Tripwires）
**时间：** ~60 分钟

## 问题

agent 采取一个 action。用户必须决定：批准还是不批准。如果这个决定是瞬间的，它大概率不是 review。如果这个决定是结构化的，它会慢，但可信。工程问题是如何让结构化 review 成为阻力最小的路径。

2023 时代的 HITL pattern 是同步 prompt：“Agent wants to send email to X with body Y — approve?” 用户点击 Approve。每个人都觉得系统安全。实践中，这个 surface 很容易被 rubber-stamp：用户批准很快，approval 几乎不预测什么；当 agent 出错时，audit trail 显示一长串用户已经不记得的 approvals。

2026 pattern——propose-then-commit——把 HITL 移到 durable substrate 上，附带 structured metadata，并要求 positive commit。每个 managed agent SDK 都发布了某个版本：LangGraph `interrupt()`、Microsoft Agent Framework `RequestInfoEvent`、Cloudflare `waitForApproval()`。API 名称不同，形状相同。

## 概念

### propose-then-commit state machine

1. **Propose。** Agent 产生 proposed action。持久化到 durable store（PostgreSQL、Redis、Durable Object）。包括：
   - intent（agent 为什么这么做）
   - data lineage（哪个 source 导致该 proposal）
   - permissions touched（哪些 scopes / files / endpoints）
   - blast radius（最坏情况是什么）
   - rollback plan（如果 committed，如何撤销）
   - idempotency key（每个 proposal 唯一；resubmission 返回同一 record）
2. **Surface。** Reviewer 看到 proposal 及所有 metadata。reviewer 是人（不是 agent 审查自己）。
3. **Commit。** Positive acknowledgement。action 执行。
4. **Verify。** 执行后，读回 side effect 并确认。如果 verify step 失败，系统处于 known bad state，并启动 alerting。

### idempotency key

没有 idempotency key，transient failure 后的 retry 可能 double-execute 一个已批准 action。具体例子：用户批准“transfer $100 from A to B”。网络闪断。workflow retry。用户只批准了一次，但转账执行了两次。idempotency key 将 approval 绑定到一个单一、唯一的 side effect；第二次执行是 no-op。

这与 Stripe 和 AWS APIs 使用的 idempotency pattern 相同。Microsoft Agent Framework docs 明确把它复用于 agent approvals。

### Durability：为什么 approvals 能比进程活得更久

approval waiting room 是一段 agent 不拥有的 state。workflow paused（第 12 课）。approval 到来时，workflow 从那个确切点恢复。这就是为什么 LangGraph 将 `interrupt()` 与 PostgreSQL checkpointing 搭配，而不只是 in-memory state——两天后的 approval 仍能找到完整 workflow。

### Rubber-stamp approvals 与 challenge-and-response mitigation

HITL 的默认 UI（“Approve” / “Reject”按钮）会产生快速 approvals，但没有真正 review。已记录的 mitigation：challenge-and-response checklist，要求在 Approve button 启用前，对具体问题给出 positive answers。具体形态：

- “Do you understand what resource this touches? [ ]”
- “Have you verified the blast radius is acceptable? [ ]”
- “Do you have a rollback plan if this fails? [ ]”

这不是为官僚而官僚——它是 forcing function。无法勾选这些框的 reviewer 要么请求 clarification（escalation），要么 decline（safe default）。Anthropic agent-safety research 明确引用 checklist-driven HITL 作为 rubber-stamp approval patterns 的 mitigation。

### 什么算 consequential

不是每个 action 都需要 propose-then-commit。2026 guidance：

- **Consequential actions**（始终 HITL）：irreversible writes、financial transactions、outbound communication、production database changes、destructive file-system operations。
- **Reversible actions**（有时 HITL）：local files edits、staging-env changes、带明确 rollback 的 reversible writes。
- **Reads and inspections**（从不 HITL）：读取文件、列出资源、调用 read-only API。

### Post-action verification

“commit ran” 不等于 “side effect happened”。Network-partition 和 race conditions 会产生一种 workflow：它以为成功，但 backend 没有持久化。verify step 会在 commit 后重新读取 target resource 来确认。这与 database transactions 中的 `RETURNING` clauses，或 AWS `PutObject` 后的 `GetObject` 是同一模式。

### EU AI Act Article 14

Article 14 要求 EU 高风险 AI systems 具备 effective human oversight。“Effective”不是装饰。法规语言明确排除 rubber-stamp patterns。带 challenge-and-response 的 propose-then-commit，是 Microsoft Agent Governance Toolkit compliance docs 中能经受 Article 14 scrutiny 的形状。

## 使用它

`code/main.py` 用 stdlib Python 实现一个 propose-then-commit state machine。durable store 是 JSON file。idempotency key 是（thread_id, action_signature）的 hash。driver 模拟三种情况：clean approval flow、transient failure 后 retry（不得 double-execute）、rubber-stamp default 与 challenge-and-response flow 对比。

## 交付它

`outputs/skill-hitl-design.md` 会审查一个拟议 HITL workflow 是否具备 propose-then-commit 形状，并标记 missing metadata、idempotency、verification 或 challenge-and-response layers。

## 练习

1. 运行 `code/main.py`。确认 approved proposal 的 retry 使用 durable record 且不会 re-execute。然后将 idempotency key 改成包含 timestamp，并展示 retry 会 double-execute。

2. 用 `rollback` field 扩展 proposal record。模拟一个 verify step 失败的 execution。展示 rollback 自动触发。

3. 阅读 Microsoft Agent Framework 的 `RequestInfoEvent` docs。找出 API 包含而 toy engine 缺失的一个 metadata field。添加它，并解释它防护什么。

4. 为一个具体 action（例如“post to a public Twitter account”）设计 challenge-and-response checklist。reviewer 必须回答哪三个问题？为什么是这三个？

5. 选一个 synchronous “Approve?” prompt 足够的案例（不需要 durable store）。解释原因，并指出你接受的 risk class。

## 关键术语

| Term | What people say | What it actually means |
|---|---|---|
| Propose-then-commit | “Two-phase approval” | Persisted proposal + positive commit + verify |
| Idempotency key | “Retry-safe token” | 每个 proposal 唯一；第二次执行 no-ops |
| Data lineage | “它来自哪里” | 导致 proposal 的具体 source content |
| Blast radius | “最坏情况” | action 出错时的影响范围 |
| Rubber-stamp | “快速批准” | 没有 genuine review 就点击 “Approve” |
| Challenge-and-response | “Forcing checklist” | reviewer 必须明确确认具体问题 |
| RequestInfoEvent | “MS Agent Framework primitive” | 带 structured metadata 的 durable HITL request |
| `interrupt()` / `waitForApproval()` | “Framework primitives” | LangGraph / Cloudflare 中相同形状的等价物 |

## 延伸阅读

- [Microsoft Agent Framework — Human in the loop](https://learn.microsoft.com/en-us/agent-framework/workflows/human-in-the-loop) — `RequestInfoEvent`、durable approvals。
- [Cloudflare Agents — Human in the loop](https://developers.cloudflare.com/agents/concepts/human-in-the-loop/) — `waitForApproval()` 和 Durable Objects。
- [Anthropic — Measuring agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) — HITL 作为 long-horizon risk 的 mitigation。
- [EU AI Act — Article 14: Human oversight](https://artificialintelligenceact.eu/article/14/) — high-risk systems 的 regulatory baseline。
- [Anthropic — Claude's Constitution (January 2026)](https://www.anthropic.com/news/claudes-constitution) — oversight 的 constitutional framing。
