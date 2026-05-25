# Long-Running Background Agents：Durable Execution

> 生产中的 long-horizon agents 不会运行在 `while True` 里。每次 LLM call 都变成一个带 checkpoint、retry 和 replay 的 activity。Temporal 的 OpenAI Agents SDK integration 于 2026 年 3 月 GA。Claude Code Routines（Anthropic）无需持久本地进程即可运行 scheduled Claude Code invocations。Sessions 会在 human-input 上暂停，跨 deploy 存活，并从以 `thread_id` keyed 的最新 checkpoint 恢复。新的 ergonomics 背后是一个老模式——workflow orchestration——带一个新输入：LLM calls 作为 non-deterministic activities，必须在 recovery 时 deterministic replay。

**类型：** 学习
**语言：** Python（stdlib，minimal durable-execution state machine）
**前置要求：** 阶段 15 · 10（Permission modes），阶段 15 · 01（Long-horizon agents）
**时间：** ~60 分钟

## 问题

想象一个运行四小时的 agent。它调用三个 tools，prompt 用户两次，并发起四十次 LLM calls。运行到一半，宿主机重启了。会发生什么？

- 在 naïve `while True` loop 中：一切丢失。run 从头开始。三个 tool calls（有真实副作用）再次执行。用户再次被要求批准他们已经批准过的东西。四十次 LLM calls 再次计费。
- 使用 durable execution：run 从最近的 checkpoint 恢复。已完成 activities 不会重新执行；结果从 durable log replay。用户不会重新批准已经批准过的内容。已经发生的 LLM calls 不会再次计费。

这是 workflow engines 已经上线十年的同一种模式（Temporal、Cadence、Uber's Cherami）。新的是 LLM calls 现在成为一种 activity——non-deterministic、昂贵、带副作用——而它们可以干净地适配这个模式。

本课的主线：long-horizon reliability 会衰减（METR 观察到“35-minute degradation”——success rate 大约随 horizon 二次下降）。Durable execution 让 runs 能长于 reliability profile 支持的时间；如果设计正确，这是安全失败的新方式，如果设计错误，则是不安全失败的新方式。

## 概念

### Activities、workflows 与 replay

- **Workflow**：确定性的 orchestration code。定义 activities 的顺序、branches、waits。必须 deterministic，这样才能从 event log replay，而不产生意外分歧。
- **Activity**：non-deterministic、可能失败的 work unit。LLM call、tool call、file write、HTTP request。每个 activity 都会记录 inputs，以及完成后的 outputs。
- **Event log**：durable backing store。每个 activity start、complete、fail、retry，以及每个 workflow decision 都被记录。
- **Replay**：recovery 时，workflow code 从头重新运行；已完成的每个 activity 都返回其 logged result，而不重新执行。只有尚未完成的 activities 会实际运行。

这与 React 基于 virtual DOM 重新渲染，或 Git 从 commits 重建 working tree 的形状相同。orchestrator 的 determinism 让 durability 变得便宜。

### 为什么 LLM calls 适配这个模式

LLM calls 是：
- Non-deterministic（temperature > 0；即使 temperature 0 也会随 model versions 漂移）。
- 昂贵（money 和 latency）。
- 可能失败（rate limits、timeouts）。
- 有副作用（如果它们调用 tools）。

这正是 activity profile。把每个 LLM call 包装成 activity，可以得到 exponential backoff retry、跨 restarts checkpointing，以及用于 debugging 的 replayable trace。

### 以 `thread_id` keyed 的 checkpoints

LangGraph、Microsoft Agent Framework、Cloudflare Durable Objects 和 Claude Code Routines 都收敛到相同 API 形状：一个 `thread_id`（或等价物）识别 session；每次 state transition 都持久化到 backend（PostgreSQL 默认，SQLite 用于 dev，Redis 用于 cache）；resume 读取最新 checkpoint。

backend 选择很重要：

- **PostgreSQL**：durable、queryable、survives deploys。LangGraph 的默认选择。
- **SQLite**：仅 local-dev；跨 hosts 会丢数据。
- **Redis**：快，但除非配置 AOF/snapshot，否则 ephemeral。
- **Cloudflare Durable Objects**：透明分布式；由 unique key scoped；可存活数小时到数周。

### Human-input 作为 first-class state

Propose-then-commit（第 15 课）需要一个 durable “waiting on human” state。workflow 暂停，external queue 持有 pending request，approval 从那个确切点恢复。没有 durability 时这只是 best-effort；有了它，夜间 approval 到来后，workflow 第二天早上接着执行。

### 35-minute degradation

METR 观察到，测量过的每类 agent 在连续运行超过约 35 分钟后都会出现 reliability decay。任务时长翻倍，failure rate 大约变成四倍。Durable execution 不会修复它；它让你能运行得比 reliability profile 支持的更久。安全模式是把 durability 与 re-entry 时需要 fresh HITL 的 checkpoints 结合，并与 cap total compute 的 budget kill switches（第 13 课）结合，无论 wall-clock time 多长。

### 什么时候 durable execution 是错误答案

- 几分钟以内、没有 human input 的 runs。overhead > benefit。
- 严格 read-only 的 information retrieval。
- correctness 需要 end-to-end 位于一个 context window 中的任务（某些推理任务；某些 one-shot generation）。

## 使用它

`code/main.py` 用 stdlib Python 实现一个极简 durable-execution engine。它支持：

- `@activity` decorator，将 inputs 和 outputs 记录到 JSON event log。
- 一个 workflow function 来排序 activities。
- 一个 `run_or_replay(workflow, event_log)` function，能 replay completed activities 而不重新执行。

driver 会模拟一个三 activity workflow，中途 crash，并展示（a）naive retry 会重新执行一切，而（b）replay 只运行缺失的 activity。

## 交付它

`outputs/skill-durable-execution-review.md` 会审查一个拟议 long-running agent deployment 是否具备正确 durable-execution shape：activities、determinism、checkpoint backend、human-input state 和 HITL-on-resume policy。

## 练习

1. 运行 `code/main.py`。观察 naive retry 与 replay 在 activity-execution count 上的差异。改变 crash point，并展示 replay count 如何随之变化。

2. 将 toy engine 改为显式使用 `thread_id`。模拟两个 concurrent sessions 共享 engine，并确认它们的 event logs 不会 collision。

3. 取 toy engine 中的一个 activity。在 workflow decision 内引入 non-determinism（wall-clock timestamp）。展示 replay 时的 divergence。解释真实 engines 如何处理它（side-effect registration、`Workflow.now()` APIs）。

4. 阅读 LangChain “Runtime behind production deep agents” post。列出 runtime 持久化的每个 state，并说出各自覆盖哪种 failure mode。

5. 为一个 6 小时 autonomous coding task 设计 checkpoint policy。在哪里 checkpoint？crash 后 resume 是什么样子？什么需要 fresh HITL？

## 关键术语

| Term | What people say | What it actually means |
|---|---|---|
| Workflow | “Agent 的 script” | Deterministic orchestration code；可从 event log replay |
| Activity | “一步” | Non-deterministic unit（LLM call、tool call）；前后都记录 |
| Event log | “backing store” | 每个 state transition 的 durable record |
| Replay | “Resume” | 重新运行 workflow；completed activities 返回 logged results 而不重新执行 |
| Checkpoint | “Save point” | 以 thread_id keyed 的 persisted state；resume 时 latest-wins |
| thread_id | “Session key” | scoped durable state 的 identifier |
| 35-minute degradation | “Reliability decay” | METR：success rate 大约随 horizon 二次下降 |
| Non-determinism | “Drift on replay” | Wall clock、random、LLM output；必须注册为 side effect |

## 延伸阅读

- [Anthropic — Claude Code Agent SDK: agent loop](https://code.claude.com/docs/en/agent-sdk/agent-loop) — budget、turns 和 resume semantics。
- [Microsoft — Agent Framework: human-in-the-loop and checkpointing](https://learn.microsoft.com/en-us/agent-framework/workflows/human-in-the-loop) — RequestInfoEvent shape。
- [LangChain — The Runtime Behind Production Deep Agents](https://www.langchain.com/conceptual-guides/runtime-behind-production-deep-agents) — 具体 runtime requirements。
- [OpenAI Agents SDK + Temporal integration (Trigger.dev announcement)](https://trigger.dev) — LLM calls 的 activity shape。
- [Anthropic — Measuring agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) — 35-minute degradation reference。
