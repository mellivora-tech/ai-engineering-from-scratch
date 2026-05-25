# LangGraph：Agents 的 State Machines

> 手写的 ReAct loop 是一个 `while True`。用 LangGraph 写的 ReAct loop 是一个你可以 checkpoint、interrupt、branch 和 time-travel 的 graph。Agent 没变，变的是它外面的 harness。

**类型：** 构建
**语言：** Python
**前置要求：** 阶段 11 · 09（Function Calling），阶段 11 · 14（Model Context Protocol）
**时间：** ~75 分钟

## 问题

你发布了一个 function-calling agent。它前三个 turns 工作正常，然后出了问题：模型尝试了一个返回 500 的 tool，用户在任务中途改主意，或 agent 决定退款而没有人工签核。`while True:` loop 没有 hooks。你不能暂停它，不能 rewind 它，也不能 branch 出“如果模型选了另一个 tool 会怎样”。一旦它越过 demo 进入生产，这个 agent 就成了黑盒：要么工作了，要么没工作。

一旦看清楚，下一步很明显。Agent 已经是 state machine：system prompt + message history + pending tool calls + next action。把 state machine 显式化：用 nodes 表示“模型思考”“tool 运行”“human 批准”，用 edges 表示它们之间的 conditional transitions。Graph 显式之后，harness 免费获得四件事：checkpointing（步骤之间保存 state）、interrupts（为人工暂停）、streaming（stream tokens 和 intermediate events）、time-travel（回到 prior state 并尝试不同 branch）。

LangGraph 是发布这个抽象的库。它不是 LangChain 意义上的 agent framework（“这是 AgentExecutor，祝你好运”）。它是带 first-class state、first-class persistence 和 first-class interrupts 的 graph runtime。Agent loop 是你画出来的，而不是手写出来的。

## 概念

![LangGraph StateGraph: nodes, edges, and the checkpointer](../assets/langgraph-stategraph.svg)

`StateGraph` 有三件事。

1. **State。** 一个 typed dict（TypedDict 或 Pydantic model），在 graph 中流动。每个 node 接收完整 state 并返回 partial update，LangGraph 会用每个 field 的 *reducer* 合并它：应该累积的 lists 用 `operator.add`，默认 overwrite。
2. **Nodes。** Python functions `state -> partial_state`。每个 node 是一个离散步骤：“call the model”“run tools”“summarize”。
3. **Edges。** Nodes 之间的 transitions。Static edges 去一个地方。Conditional edges 接收 router function `state -> next_node_name`，让 graph 按 model output branch。

你 compile graph。Compile 会绑定 topology、附上 checkpointer（可选但生产中必需），并返回 runnable。用 initial state 和 `thread_id` 调用它。每个 execution step 都会持久化一个以 `(thread_id, checkpoint_id)` 为 key 的 checkpoint。

### 四种 superpowers

**Checkpointing。** 每个 node transition 都把新 state 写入 store（测试用 in-memory，生产用 Postgres/Redis/SQLite）。用相同 `thread_id` 再次调用 graph 就能 resume。Graph 从暂停处继续。

**Interrupts。** 用 `interrupt_before=["human_review"]` 标记 node，execution 会在该 node 运行前停止。State 持久化。你的 API 向用户响应“awaiting approval”。之后对同一 `thread_id` 使用 `Command(resume=...)` 继续执行。

**Streaming。** `graph.stream(state, mode="updates")` 会 yield 发生的 state deltas。`mode="messages"` stream model nodes 内部的 LLM tokens。`mode="values"` yield full snapshots。你选择 UI 要展示什么。

**Time-travel。** `graph.get_state_history(thread_id)` 返回完整 checkpoint log。把任意 prior `checkpoint_id` 传给 `graph.invoke`，就能从那里 fork。非常适合 debug（“如果模型选了 tool B 会怎样？”）和 replay production traces 的 regression tests。

### Reducers 才是关键点

每个 state field 都有 reducer。多数默认值没问题，新值 overwrite 旧值。但 message lists 需要 `operator.add`，让新 messages append 而不是 replace。Parallel edges 的 updates 会通过 reducer merge。如果两个 nodes 都更新 `messages`，而你忘了 `Annotated[list, add_messages]`，第二个会静默胜出，你丢掉半个 turn。Reducer 是库里唯一微妙的东西；把它做对，其他就能组合。

### 四个 nodes 的 ReAct graph

Production ReAct agent 是四个 nodes 和两条 edges：

1. `agent` — 用当前 message history 调用 LLM。返回 assistant message（可能包含 tool_calls）。
2. `tools` — 执行最后 assistant message 中的 tool_calls，把 tool results 作为 tool messages append。
3. 从 `agent` 出发的 conditional edge：如果最后 message 有 tool_calls，路由到 `tools`，否则到 `END`。
4. 从 `tools` 回到 `agent` 的 static edge。

就这样。你得到完整 ReAct loop（Thought -> Action -> Observation -> Thought -> ...），并带 checkpointing、interrupts 和 streaming，约 40 行代码。

### StateGraph vs Send（fanout）

`Send(node_name, state)` 让一个 node dispatch parallel subgraphs。示例：agent 决定同时 query 三个 retrievers。每个 `Send` 启动 target node 的 parallel execution；它们的 outputs 通过 state reducer merge。这就是 LangGraph 表达 orchestrator-workers pattern 的方式，不需要 threading primitives。

### Subgraphs

Compiled graph 可以作为另一个 graph 的 node。Outer graph 看到一个 node；inner graph 有自己的 state 和 checkpoints。这是 teams 构建 supervisor-worker agents 的方式：supervisor graph 将 user intent 路由到 per-domain worker subgraph。

## 构建它

### 第 1 步：state and nodes

```python
from typing import Annotated, TypedDict
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

def agent_node(state: State) -> dict:
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

def should_continue(state: State) -> str:
    last = state["messages"][-1]
    return "tools" if getattr(last, "tool_calls", None) else END

tool_node = ToolNode(tools=[search_web, read_file])

graph = StateGraph(State)
graph.add_node("agent", agent_node)
graph.add_node("tools", tool_node)
graph.set_entry_point("agent")
graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
graph.add_edge("tools", "agent")

app = graph.compile(checkpointer=MemorySaver())
```

`add_messages` 是让 message list accumulate 而不是 overwrite 的 reducer。忘掉它是最常见的 LangGraph bug。

### 第 2 步：run with a thread

```python
config = {"configurable": {"thread_id": "user-42"}}
for event in app.stream(
    {"messages": [HumanMessage("find the Anthropic headquarters address")]},
    config,
    stream_mode="updates",
):
    print(event)
```

每个 update 都是 dict `{node_name: state_delta}`。你的 frontend 可以把这些 stream 到 UI，让用户看到“agent is thinking… calling search_web… got result… answering.”

### 第 3 步：add a human-in-the-loop interrupt

标记 node，使 execution 在运行前暂停。

```python
app = graph.compile(
    checkpointer=MemorySaver(),
    interrupt_before=["tools"],  # pause before every tool call
)

state = app.invoke({"messages": [HumanMessage("delete the production database")]}, config)
# state["__interrupt__"] is set. Inspect proposed tool calls.
# If approved:
from langgraph.types import Command
app.invoke(Command(resume=True), config)
# If denied: write a rejection message and resume
app.update_state(config, {"messages": [AIMessage("Blocked by human reviewer.")]})
```

State、checkpoint 和 thread 都会跨 interrupt 持久化。执行之外没有东西留在内存中。

### 第 4 步：time-travel for debugging

```python
history = list(app.get_state_history(config))
for snapshot in history:
    print(snapshot.values["messages"][-1].content[:80], snapshot.config)

# Fork from a prior checkpoint
target = history[3].config  # three steps back
for event in app.stream(None, target, stream_mode="values"):
    pass  # replay from that point forward
```

传入 `None` 作为 input，会从给定 checkpoint replay；传入一个值，会先把它作为 update append 到该 checkpoint 的 state，再 resume。这就是在不重新跑完整 conversation 的情况下复现 bad agent run 的方式。

### 第 5 步：swap the checkpointer for production

```python
from langgraph.checkpoint.postgres import PostgresSaver

with PostgresSaver.from_conn_string("postgresql://...") as checkpointer:
    checkpointer.setup()
    app = graph.compile(checkpointer=checkpointer)
```

SQLite、Redis 和 Postgres 都已提供。`MemorySaver` 用于测试。任何需要跨 restarts 持久化的东西都应该用真实 store。

## Skill

> 你把 agents 构建成 graphs，而不是 `while True` loops。

在使用 LangGraph 前，做 60 秒设计：

1. **Name the nodes。** 每个离散 decision 或 side-effecting action 都是 node。“Agent thinks”“tool runs”“reviewer approves”“response streams”。如果你列不出来，任务还不是 agent-shaped。
2. **Declare the state。** Minimal TypedDict，并为每个 list field 设置 reducer。不要把一切塞进 `messages`；把 task-specific fields（working `plan`、`budget` counter、`retrieved_docs` list）提升到 top level。
3. **Draw the edges。** 默认 static，除非下一步依赖 model output。每个 conditional edge 都需要一个带 named branches 的 router function。
4. **Choose a checkpointer up front。** 测试用 `MemorySaver`，其他用 Postgres/Redis/SQLite。不要没有 checkpointer 就发布；没有 checkpointer 意味着没有 resume、interrupt、time-travel。
5. **Decide interrupts before tools run, not after。** Approval 放在进入 side-effecting node 的 edge 上，这样你可以在造成伤害前 cancel；validation 放在 model 输出之后的 edge 上，这样你可以便宜地拒绝 bad calls。
6. **Stream by default。** UI 用 `mode="updates"`，model nodes 内 token-level streaming 用 `mode="messages"`，eval 的 full snapshots 用 `mode="values"`。

拒绝发布没有 checkpointer 的 LangGraph agent。拒绝发布 side effect 之后才 interrupt 的 agent。拒绝发布没有把 `add_messages` 作为 reducer 的 `messages` field。

## 练习

1. **Easy.** 用 calculator tool 和 web-search tool 实现上面的 four-node ReAct graph。验证对于 two-turn conversation，`list(app.get_state_history(config))` 至少返回四个 checkpoints。
2. **Medium.** 添加一个 `planner` node，在 `agent` 前运行，并把 structured `plan: list[str]` 写入 state。让 `agent` 标记 plan steps done。如果 `plan` 在 checkpoint resume 后丢失（错误 reducer），test fail。
3. **Hard.** 构建 supervisor graph，用 `Send` 在三个 subgraphs（`researcher`、`writer`、`reviewer`）之间路由。每个 subgraph 有自己的 state 和 checkpointer。在 outer graph 上加 `interrupt_before=["writer"]`，让 human approve research brief。确认从 prior checkpoint time-travel 只 rerun forked branch。

## 关键术语

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| StateGraph | “The LangGraph graph” | Compile 前添加 nodes 和 edges 的 builder object。 |
| Reducer | “How the field merges” | 当 node 返回某 field update 时应用的 `(old, new) -> merged` 函数；默认 overwrite，`add_messages` append。 |
| Thread | “A conversation ID” | `thread_id` string，限定一个 session 的所有 checkpoints。 |
| Checkpoint | “A paused state” | Node transition 后完整 graph state 的持久化 snapshot，以 `(thread_id, checkpoint_id)` 为 key。 |
| Interrupt | “Pause for a human” | `interrupt_before` / `interrupt_after` 在 node boundary 停止 execution；用 `Command(resume=...)` 恢复。 |
| Time-travel | “Fork from a prior step” | `graph.invoke(None, config_with_old_checkpoint_id)` 从该 checkpoint 向前 replay。 |
| Send | “Parallel subgraph dispatch” | Node 可以返回的 constructor，用于启动 N 个 target node 的 parallel executions。 |
| Subgraph | “A compiled graph as a node” | 作为另一个 graph 中 node 使用的 compiled StateGraph；保留自己的 state scope。 |

## 延伸阅读

- [LangGraph documentation](https://langchain-ai.github.io/langgraph/) — StateGraph、reducers、checkpointers、interrupts 的 canonical reference。
- [LangGraph concepts: state, reducers, checkpointers](https://langchain-ai.github.io/langgraph/concepts/low_level/) — 本课使用的 mental model，来自官方。
- [LangGraph Persistence and Checkpoints](https://langchain-ai.github.io/langgraph/concepts/persistence/) — Postgres/SQLite/Redis stores、checkpoint namespaces、thread IDs 的细节。
- [LangGraph Human-in-the-loop](https://langchain-ai.github.io/langgraph/concepts/human_in_the_loop/) — `interrupt_before`、`interrupt_after`、`Command(resume=...)` 和 edit-state pattern。
- [Yao et al., "ReAct: Synergizing Reasoning and Acting in Language Models" (ICLR 2023)](https://arxiv.org/abs/2210.03629) — 每个 LangGraph agent 实现的模式；读它理解 reasoning trace rationale。
- [Anthropic — Building effective agents (Dec 2024)](https://www.anthropic.com/research/building-effective-agents) — 应该何时偏好哪些 graph shapes（chain、router、orchestrator-workers、evaluator-optimizer）。
- Phase 11 · 09（Function Calling）— 每个 LangGraph agent node 复用的 tool-call primitive。
- Phase 11 · 14（Model Context Protocol）— 可通过 MCP adapter 插入 LangGraph `ToolNode` 的 external tool discovery。
- Phase 11 · 17（Agent framework tradeoffs）— 什么时候选择 LangGraph，而不是 CrewAI、AutoGen 或 Agno。
