# Checkpoints 与 Rollback

> 每个 graph-state transition 都会持久化。当 worker crash 时，其 lease 过期，另一个 worker 会从最新 checkpoint 接手。Cloudflare Durable Objects 可以跨数小时或数周保存 state。Propose-then-commit（第 15 课）为每个 action 定义 rollback plan。Post-action verification 关闭 loop。EU AI Act Article 14 让 high-risk systems 必须具备 effective human oversight——实践中这意味着 checkpoints 必须 queryable，rollbacks 必须演练，audit trail 必须 survive deploy。尖锐的 failure mode：如果没有 idempotency keys 和 precondition checks，transient failure 后的 retry 可能 double-execute 一个已经 approved 的 action。Post-action verification 正是捕捉它的机制。

**类型：** 学习
**语言：** Python（stdlib，checkpoint and rollback state machine）
**前置要求：** 阶段 15 · 12（Durable execution），阶段 15 · 15（Propose-then-commit）
**时间：** ~60 分钟

## 问题

Durable execution（第 12 课）让 crashed agent 可恢复。Propose-then-commit（第 15 课）让 approved action 可审计。本课把两者连接起来：当 approved action 部分执行、crash、然后 resume 时会发生什么？rollback 何时运行，针对什么 state？

真实系统用不同方式把它接起来：

- **LangGraph** 将每个 graph-state transition checkpoint 到 PostgreSQL。worker crash 时，lease 释放，另一个 worker 从最新 checkpoint 恢复。Workflows 在 `interrupt()` 上暂停，而 `interrupt()` 本身也会持久化。
- **Cloudflare Durable Objects** 跨数小时或数周保存 per-key state。把 computation 与 approved action 的 storage colocate。
- **Microsoft Agent Framework** 在 workflow API 中暴露 `Checkpoint` primitives；replay 加 idempotency 覆盖 retries。

无论哪种情况，真正有效的组合是：idempotency key（防止 double-execute）+ precondition check（state 仍然是当初批准时的样子）+ post-action verify（side effect 确实发生）+ verify-fail 时 rollback。

## 概念

### 每个 transition 都持久化

graph-state transition 是任何将 workflow 从一个 named state 移到另一个 named state 的步骤。naive implementations 只在特定 commit points 持久化；production implementations 持久化每个 transition。成本（多几次写入）相比 reliability gain（replay 能落在任何地方，lease recovery 精确）很小。

### Lease recovery

worker crash 时，workflow 不会丢失；lease（这个 worker 正在执行此 run 的短期 claim）只是过期。另一个 worker 拿起最新 checkpoint 并 resume。lease mechanism 让 production systems 能承受 rolling deploys 而不丢 in-flight work。

### Idempotency plus preconditions

单靠 idempotency 不够。考虑：workflow 被批准“当 balance > $1000 时从 A 转 $100 到 B”。workflow committed，mid-execution crash，并 resume。如果只检查 idempotency key，execution resume 后转账只运行一次（正确）。但考虑 crash 与 resume 之间，A 的 balance 被另一个 workflow 降到 $500。idempotency check 仍然通过；precondition 不通过。没有 precondition check，我们就上线了 overdraft。

每个 consequential action 都需要二者：

- **Idempotency key**：防止 double-execute。
- **Precondition check**：确认 state 仍与批准时一致。

### Post-action verification

“tool returned 200” 不是 verification。真正的 verification 会重新读取 target state，并确认 side effect 确实发生。模式：

- Database update：`UPDATE ... RETURNING *`，然后 assert returned row matches intended state。
- Email send：submission 后检查 sent-folder 中的 message ID。
- File write：读回文件并 hash。
- API call：对 target resource 做 follow-up `GET`。

如果 verify fails，workflow 处于 known-bad state。Rollback 启动。

### Rollback plans

propose-then-commit（第 15 课）中的每个 consequential action 都携带 rollback plan。类型：

- **In-band rollback**：直接反转 side effect（`DELETE` after `INSERT`，send 后 `Send-correction-email`）。
- **Compensating transaction**：一个新 action，用于中和原 action（标准 SAGA pattern）。
- **Out-of-band rollback**：alert human、pause workflow、保留 bad state 供 investigation。

No-op rollback（“we cannot undo this”）必须在 proposal 中明确命名。没有 rollback 的 actions 在 commit time 需要更强 HITL（第 15 课 challenge-and-response）。

### EU AI Act Article 14 的 operational reading

Article 14 要求 high-risk systems 具备“effective human oversight”。在 operational terms 中，implementers 通常理解为：

- Checkpoints 可由 auditor query。
- Rollbacks 已演练（至少 end-to-end 测过一次）。
- Audit trail survives deploy（checkpoint backend 不是 ephemeral）。
- Failed verifications 会 alert，而不是 silently logged。

一个 workflow 若在 mid-commit crash、resume 后完成 side effect，却没有 verify + rollback pathway，就经不起 Article 14 test。

### 尖锐 failure mode：double-execute

这个领域最常见的 production incident：

1. Action approved，idempotency key k。
2. Commit starts，executes，returns 200。
3. Workflow 在持久化 “committed” status 前 crash。
4. Workflow resumes；看到 “approved but not committed”；重新执行。
5. Side effect 触发两次。

Mitigation：在执行前持久化 “in-flight” intent，带 idempotency key 执行，然后只有在 post-action verification 成功后标记 “committed”。如果 action 触发而 status write 失败，你知道要 verify，并在必要时 re-fire。如果 status write 成功而 action 失败，你会 verify，并通过 recovery path 精确触发一次。

## 使用它

`code/main.py` 实现一个带 checkpoint 的 workflow，包含 idempotency、preconditions、verify 和 rollback。driver 模拟四种 scenarios：clean run、crash 后 retry（idempotency 捕捉）、precondition fail（workflow 不触发并 abort）、verify fail（rollback 触发）。

## 交付它

`outputs/skill-rollback-rehearsal.md` 会为拟议 workflow 设计 rollback-rehearsal test，并审计 checkpoint backend 的 audit-trail persistence。

## 练习

1. 运行 `code/main.py`。验证四种 scenarios。对 crash-during-commit case，确认 action 在 retries 中只触发一次。

2. 修改 “mark as done first, then do it” pattern，让 status write 在 action 后触发。重新运行 crash scenario。测量有多少 duplicate actions 触发。

3. 为一个具体 production action（例如“post to a Slack channel”）设计 rollback plan。分类为 in-band、compensating 或 out-of-band。解释选择。

4. 取一个你熟悉的 workflow。识别每个 state transition。给每个标记 durability requirement（persist / do not persist）。数出当前没有持久化的数量。

5. Rehearsed-rollback test：设计一个 end-to-end test，运行真实 workflow、让它 crash，并确认 rollback path 触发。测试 assert 什么？

## 关键术语

| Term | What people say | What it actually means |
|---|---|---|
| Checkpoint | “Save point” | 每个 graph-state transition 都持久化到 durable store |
| Lease | “Worker claim” | worker 正在执行 run 的短期 claim；crash 时过期 |
| Precondition | “State gate” | 断言 state 仍与 approved action 一致 |
| Post-action verify | “Re-read check” | 确认 side effect 确实在 target system 中发生 |
| In-band rollback | “Direct undo” | 用 inverse operation 反转 side effect |
| Compensating transaction | “SAGA undo” | 中和原 action 的新 action |
| Mark-as-done-first | “Status write order” | 从 commit 返回前持久化 committed status |
| Article 14 | “EU AI Act human oversight” | Operational：queryable checkpoints、rehearsed rollbacks、auditable trail |

## 延伸阅读

- [Microsoft Agent Framework — Checkpointing and HITL](https://learn.microsoft.com/en-us/agent-framework/workflows/human-in-the-loop) — checkpoint primitives 和 lease recovery。
- [Cloudflare Agents — Human in the loop](https://developers.cloudflare.com/agents/concepts/human-in-the-loop/) — Durable Objects 作为 state substrate。
- [EU AI Act — Article 14: Human oversight](https://artificialintelligenceact.eu/article/14/) — regulatory baseline。
- [Anthropic — Measuring agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) — long-horizon workflows 的 reliability framing。
- [Anthropic — Claude Code Agent SDK: agent loop](https://code.claude.com/docs/en/agent-sdk/agent-loop) — Claude Code Routines 的 workflow shape。
