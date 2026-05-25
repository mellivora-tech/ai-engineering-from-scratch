# Production Scaling — Queues、Checkpoints、Durability

> 将 multi-agent systems 扩展到数千 concurrent runs 需要 **durable execution**。LangGraph runtime 在每个 super-step 后按 `thread_id` 写 checkpoint（默认 Postgres）；worker crashes 释放 lease，另一个 worker resume。Agents 可以无限期 sleep 等待 human input。**MegaAgent**（arXiv:2408.09955）运行 per-agent producer-consumer queue，带三种 states（Idle / Processing / Response）和两层 coordination（intra-group chat + inter-group admin chat）。对 LLM streaming 来说，**fiber/async** 胜过 thread-per-job：threads 99% 的时间在等待 tokens 时闲置，fibers 在 I/O 上 cooperatively yield。反方观点：Ashpreet Bedi 的 “Scaling Agentic Software” 主张在 load 证明必要前使用 **FastAPI + Postgres + nothing else** — 简单 architectures 比预期走得更远。本课构建 durable checkpoint log、带 state transitions 的 per-agent work queue、async-vs-thread demo，并落地务实的 “start simple” rule。

**类型：** 学习 + 构建
**语言：** Python（stdlib，`asyncio`，`sqlite3`）
**前置要求：** 阶段 16 · 09（Parallel Swarm Networks），阶段 16 · 13（Shared Memory）
**时间：** ~75 分钟

## 问题

prototype multi-agent system 在一台 laptop 上用三个 agents 和 in-memory event loop 工作正常。你搬到 production：

- Agents 有时运行数小时（long research、human-in-the-loop waits）。
- Worker processes 会 crash。restart 会丢 state。
- peak load 是 average 的 10x；你需要 horizontal scaling。
- users 按 agent-run 付费；你需要 exactly-once semantics 来 charge。

in-memory event loop 一个都不支持。你需要底层 durable execution layer。2026 年 canonical options：

1. 带 checkpoints 的 workflow engine（Temporal、LangGraph runtime）。
2. 带 state store 的 message queue（Postgres + SQS/RabbitMQ）。
3. Actor-model frameworks（MegaAgent 的 per-agent producer-consumer）。
4. Hand-rolled FastAPI + Postgres（Bedi 的论点）。

本课构建每种的 miniature。

## 概念

### Durable execution pattern

durable-execution engine 在每个 “step”（LangGraph 叫 super-step）后 persist 完整 program state。crash 时：

```
worker crashes mid-step
  -> lease timeout
  -> another worker picks up the thread_id
  -> resumes from last checkpoint
  -> no duplicate side effects
```

这要成立需要：

- **Serializable state。** 所有 agent state 都必须可 persist。带 live database connections 的 function closures 无法 survive。
- **Deterministic resume。** 给定同一 state 和同一 inputs，agent 产生同样 actions（或把 LLM calls 交给 external deterministic oracle）。
- **Idempotent side effects。** external calls（tool calls、payments）必须 idempotent，或使用 deduplication key。

LangGraph 每个 super-step 后写 checkpoint；Temporal 每个 activity 后写；Restate 使用 event-sourced journals。三者实现同一 pattern。

### LangGraph runtime

每个 agent 有 `thread_id`；state 是 typed dict；每个 super-step 向 checkpoints table 写一行。resume 时，runtime 从 last checkpoint replay，而不是从头开始。Agents 可以 `interrupt()` 等待 human input；runtime persist 并释放 worker。input 到来时，任何 worker 都可以 resume。

这是 2026 年 4 月的 reference production design。

### MegaAgent per-agent queue

arXiv:2408.09955 描述了一个 scale experiment：一个 cluster 中数千 concurrent agents。Architecture：

```
agent i:
  state ∈ {Idle, Processing, Response}
  in_queue   <- messages addressed to agent i
  out_queue  -> replies + side effects

coordinators:
  intra-group chat  (agents in the same group)
  inter-group admin chat  (high-level routing)
```

两层 coordination 让 intra-group conversation 密集发生，同时 inter-group 保持稀疏 — 这是把数千 agents 成本保持线性的 pattern。

### Async vs thread-per-job

LLM calls 是 I/O-bound。等待下一个 token 的 thread 99% 时间 idle。threads 每个约 1MB RAM；10,000 concurrent calls 时，仅 stacks 就 10GB。

Fibers（Python `asyncio`、Go goroutines、Rust `tokio`）在 I/O 上 cooperatively yield。同样 10,000 calls 可以轻松装进一个 process。在 LLM-agent scale，async 不是优化，而是 architecture。

例外：CPU-bound post-processing（embedding、tokenizer tricks）仍需要 threads 或 processes。把 I/O layer 和 CPU layer 分开。

### Bedi 的反方观点

“Scaling Agentic Software”（Ashpreet Bedi, 2026）认为，大多数 teams 在 measured load 前就 over-engineer。务实默认：

- FastAPI + Postgres。
- 每个 agent run 是一行；state 通过 optimistic concurrency 原地更新。
- background jobs 通过 `pg_notify` 或简单 Celery worker。
- retry policy 在 application code 中。

对于低于约 100 concurrent agent-runs 且 tasks 可控的 loads，这往往足够。测量到失败后再升级。

规则：只有当你遇到 simple architectures 无法解决的具体问题时，才采用 durable-execution frameworks。过早采用会把时间烧在不回本的 ceremony 上。

### Exactly-once semantics

对于 paid agent runs，你需要 “exactly-once effective”（at-least-once delivery + idempotent consumer）。engineering moves：

- **每个 run 一个 dedup key。** 每个 side-effect call 都带上它。
- **Outbox pattern。** side effects 先写 table，再由 separate process 执行。两个步骤都 idempotent。
- **Compensating transactions。** 当 side effect 成功但 tracking write 失败，安排 compensate。

这些是 database-engineering patterns，不是 LLM-specific。LLM tax 只是 LLM calls 很慢；其他都是标准 distributed systems。

### Rainbow deployment

Anthropic 的 multi-agent research system 使用 “rainbow deployments”：多个 versions 的 agent runtime 并发运行，这样 long-running agents 不必在每次 code deploy 时被 kill。新版本 canary 一小部分 traffic；旧版本在其 agents 完成后 retire。

这是 long-running stateful systems 的标准做法；2026 adaptation 是 agents 可能存活数小时，所以 deployment cycles 必须适应。

### Canonical production checklist

- Durable state（checkpoints、snapshots，或 outbox + replayable log）。
- Idempotent side effects。
- LLM calls 的 async I/O layer。
- 带 dedup 的 at-least-once delivery。
- stateful workloads 的 rainbow/canary deployment。
- Observability：per-agent traces、super-step audit、retry counter。

## 构建它

`code/main.py` 实现：

- `CheckpointStore` — SQLite-backed checkpoint log，按 thread-id keys。每个 super-step append 一行。
- `run_with_checkpoint(agent, thread_id)` — 模拟 mid-run crash；第二个 worker 从 last checkpoint resume。
- `AgentQueue` — per-agent Idle / Processing / Response state machine，带小 work queue。
- `demo_async_vs_threads()` — 用 asyncio 和 threads 运行 500 个 concurrent simulated “LLM calls”；报告 wall-clock 和近似 peak memory。

运行：

```
python3 code/main.py
```

预期输出：checkpoint resume 在 simulated crash 后成功；async version 在 < 1s 中处理 500 concurrent calls；thread version 花数秒，并且每个 concurrent unit 使用数量级更多 memory。

## 使用它

`outputs/skill-scaling-advisor.md` 根据 load、state-retention needs 和 deploy frequency，建议 durable-execution choice：FastAPI + Postgres、LangGraph runtime、Temporal 或 custom。

## 发布它

Canonical production hardening：

- **Start simple（Bedi's rule）。** FastAPI + Postgres，直到测量证明失败。
- **优化前先 instrument everything。** per-run latency histogram、per-step time、retry count、failure categorization。
- **Outbox pattern for side effects。** 尤其 payments 和 external API calls。
- **Rainbow deploys。** deploy 时永远不要 kill in-flight agent runs。
- **在遇到具体问题时采用 durable-execution engines（Temporal / LangGraph / Restate）：** hour-long human-in-the-loop waits、cross-region coordination、复杂 retry/compensation policies。
- **I/O layer 用 async。** Threads 只用于 CPU-bound post-processing。

## 练习

1. 运行 `code/main.py`。确认 checkpoint resume 工作；测量 async vs thread concurrency 差异。
2. 实现 **outbox** table：每个 tool call 先写 outbox，然后 separate goroutine/task 执行。通过运行同一个 tool call 两次验证 idempotency。
3. 模拟 **rainbow deploy**：两个 runtime versions 并发；把一半 new thread_ids route 到每个版本；确认旧版本上的 in-flight threads 不被 interrupt。
4. 阅读下面链接的 LangGraph runtime doc。识别哪些 runtime features 在 hand-rolled FastAPI + Postgres 版本中复制最耗时。这是采用的理由，还是可以 defer？
5. 阅读 MegaAgent（arXiv:2408.09955）Section 3。两层 coordination（intra-group + inter-group admin chat）是显式的。画出如何把它映射到有两个 queue families 的 message queue。

## 关键词汇

| 术语 | 人们常说 | 实际含义 |
|------|----------------|------------------------|
| Durable execution | “Persist program state” | engine 在每个 super-step 后写 state；crash recovery 是 deterministic。 |
| Super-step | “Transactional boundary” | checkpoints 之间的 work unit。LangGraph term。 |
| thread_id | “Agent run identifier” | 绑定 checkpoints 和 resume logic 的 key。 |
| Idempotency | “safe to retry” | 重复 side effect 与执行一次结果相同。 |
| Outbox pattern | “decouple side effects” | 写 intent 到 table；separate executor 执行并标记 done。 |
| At-least-once delivery | “可能 duplicate” | message queue semantics；dedup key 让 consumer effective-once。 |
| Rainbow deploy | “overlapping versions” | long-running workloads 中多个 runtime versions 并发。 |
| Async fiber | “cooperative yielding” | user-mode concurrency；对 I/O-bound loads 比 threads 便宜。 |
| Checkpoint | “state snapshot” | super-step boundary 的 serialized state；resume 的 key。 |

## 延伸阅读

- [LangChain — The runtime behind production deep agents](https://www.langchain.com/conceptual-guides/runtime-behind-production-deep-agents) — LangGraph runtime design
- [MegaAgent](https://arxiv.org/abs/2408.09955) — per-agent producer-consumer queue；thousands concurrent agents 下的 two-layer coordination
- [Matrix](https://arxiv.org/abs/2511.21686) — 使用 message queues 作为 coordination substrate 的 decentralized framework
- [Temporal docs](https://docs.temporal.io/) — durable execution 的 reference workflow engine
- [Anthropic — Multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system) — production lessons，包括 rainbow deployment
