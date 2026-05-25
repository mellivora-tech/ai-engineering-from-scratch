# LangGraph：Stateful Graphs 和 Durable Execution

> LangGraph 是 2026 年 low-level stateful orchestration 的参考方案。Agent 是 state machine；nodes 是 functions；edges 是 transitions；state 是 immutable 的，并在每一步后 checkpoint。任何 failure 都可以从停止处精确恢复。

**类型：** 学习 + 构建
**语言：** Python (stdlib)
**前置要求：** 阶段 14 · 01（Agent Loop），阶段 14 · 12（Workflow Patterns）
**时间：** ~75 分钟

## 学习目标

- 描述 LangGraph 的 core model：带 immutable state、function nodes、conditional edges 和 post-step checkpoints 的 state machine。
- 说出 docs 强调的四种能力：durable execution、streaming、human-in-the-loop、comprehensive memory。
- 解释 LangGraph 支持的三种 orchestration topologies：supervisor、peer-to-peer（swarm）、hierarchical（nested subgraphs）。
- 用 stdlib 实现一个 state graph，包含 immutable state、conditional edges 和 checkpoint/resume cycle。

## 问题

Agents 和 workflows 共享一个问题：当 40 步 run 在第 38 步失败时，你希望从第 38 步恢复，而不是从头开始。二等 state models 会让 operators 围着一个假设 fresh runs 的 library 拼凑 retries。

LangGraph 的设计答案：state 是一等 typed object，mutations 是显式的，每个 node 后都持久化 checkpoints。Resume 是一次 `load_state(session_id)` call。

## 概念

### Graph

一个 graph 由这些东西定义：

- **State type。** Typed dict（或 Pydantic model），每个 node 都读取和 mutate 它。
- **Nodes。** Pure functions `(state) -> state_update`。返回后 updates 会 merge 进 state。
- **Edges。** Nodes 之间的 conditional 或 direct transitions。
- **Entry and exit。** `START` 和 `END` sentinel nodes 标记边界。

示例：一个带 `classify`、`refund`、`bug`、`sales`、`done` nodes 的 agent，也就是 routing workflow as a graph。

### Durable execution

每个 node 返回后，runtime 会 serialize state，并把它写入 checkpointer（SQLite、Postgres、Redis、custom）。如果第 N 步失败，runtime 可以 `resume(session_id)`，带着精确 state 从第 N+1 步继续。

LangGraph docs 明确强调了这种能力在生产用户中的价值：Klarna、Uber、J.P. Morgan。关键主张不是 graph shape 本身，而是 graph shape 加 checkpointing 让 recovery 便宜。

### Streaming

每个 node 都可以 yield partial output。Graph 会把 per-node-delta events stream 给 caller，让 UIs 随 graph 运行更新。

### Human-in-the-loop

在 nodes 之间检查和修改 state。实现方式：在 critical node 前暂停，把 state 展示给 human，接受修改，然后 resume。Checkpointer 让这件事简单，因为 state 已经 serialized。

### Memory

Short-term（一次 run 内：state 中的 conversation history）和 long-term（跨 runs：通过 checkpointer 加独立 long-term store 持久化）。LangGraph 通过 tools 与 external memory systems（Mem0、custom）集成。

### 三种 topologies

1. **Supervisor。** Central router LLM 分发给 specialist subagents。`langgraph-supervisor` 中有 `create_supervisor()`（不过 LangChain 团队在 2026 年建议直接通过 tool calls 做这个，以获得更多 context control）。
2. **Swarm / peer-to-peer。** Agents 通过 shared tool surface 直接 hand off。没有 central router。
3. **Hierarchical。** Supervisors 管理 sub-supervisors，用 nested subgraphs 实现。

### 这个模式会在哪里出错

- **Checkpoints too small。** 只 checkpoint conversation turns，会让 tool state 和 memory writes 无法恢复。Full state 必须可 serialize。
- **Non-deterministic nodes。** Resume 假设 node inputs 会产生同样的 state update。Random seeds、wall-clock、external APIs 必须被捕获。
- **Over-use of conditional edges。** 每条 edge 都 conditional 的 graph 是无法推理的 state machine。优先使用 linear chains，偶尔分支。

## 构建它

`code/main.py` 实现了一个 stdlib stateful graph：

- `State`：typed dict，包含 `messages`、`step`、`route`、`output`、`human_approval`。
- `Node`：接收 state 并返回 update dict 的 callable。
- `StateGraph`：nodes + edges + conditional edges + run + resume。
- `SQLiteCheckpointer`（in-memory fake）：每个 node 后 serialize state；`load(session_id)` restores。
- Demo graph：classify -> branch(refund / bug / sales) -> human gate -> send。

运行它：

```
python3 code/main.py
```

Trace 会显示第一次 run 在 human gate 失败、持久化，然后 resume 生成 final output。

## 使用它

- **LangGraph**：参考实现，production-ready。使用 `create_react_agent`、`create_supervisor`，或构建自己的 graph。
- **AutoGen v0.4**（第 14 课）：适合 high-concurrency scenarios 的 actor model alternative。
- **Claude Agent SDK**（第 17 课）：带 built-in session store 的 managed harness。
- **Custom**：当你需要精确控制 state shape 或 checkpointer backend 时。

## 发布它

`outputs/skill-state-graph.md` 会在任意 target runtime 中生成 LangGraph-shaped state graph，并接好 checkpointing 和 resume。

## 练习

1. 当 classification confidence 低于 threshold 时，从 `classify` 添加一条 conditional edge 到 `end`。Human 手动设置 `route` 后 resume run。
2. 把 SQLite-like fake 换成真实 SQLite checkpointer。测量每步 serialization overhead。
3. 实现 parallel edges：两个 nodes 并发运行，用 custom reducer merge。Immutable state 在这里带来什么？
4. 阅读 `langgraph-supervisor` reference。把 toy 移植到 `create_supervisor`。比较 trace shapes。
5. 添加 streaming：每个 node 在运行时 yield partial state。打印到达的 deltas。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|----------------|------------------------|
| State graph | “Agent as state machine” | Typed state + nodes + edges + reducers |
| Checkpointer | “Persistence backend” | 每个 node 后 serialize state；启用 resume |
| Reducer | “State merger” | 把 current state 与 node update 组合的函数 |
| Conditional edge | “Branch” | 由 state 函数选择的 edge |
| Subgraph | “Nested graph” | 一个 graph 作为另一个 graph 中的 node |
| Durable execution | “Resume from failure” | 用精确 state 从最后成功 node 重启 |
| Supervisor | “Router LLM” | Specialist subagents 的 central dispatcher |
| Swarm | “P2P agents” | Agents 通过 shared tools hand off；没有 central router |

## 延伸阅读

- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview)：参考文档
- [langgraph-supervisor reference](https://reference.langchain.com/python/langgraph/supervisor/)：supervisor pattern API
- [AutoGen v0.4, Microsoft Research](https://www.microsoft.com/en-us/research/articles/autogen-v0-4-reimagining-the-foundation-of-agentic-ai-for-scale-extensibility-and-robustness/)：actor-model alternative
- [Claude Agent SDK overview](https://platform.claude.com/docs/en/agent-sdk/overview)：session store 和 subagents
