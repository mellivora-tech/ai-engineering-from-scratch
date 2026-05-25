# Multi-Agent Primitive Model

> 2026 年发布的每个 multi-agent framework — AutoGen、LangGraph、CrewAI、OpenAI Agents SDK、Microsoft Agent Framework — 都是四维设计空间里的一个点。四个 primitives，仅此而已：agent、handoff、shared state、orchestrator。本课从零构建它们，在一个 toy system 上运行四者，然后把每个主流 framework 映射到同一组轴上，让你用一段话读懂任何新发布。

**类型：** 学习
**语言：** Python（stdlib）
**前置要求：** 阶段 14（Agent Engineering），阶段 16 · 01（Why Multi-Agent）
**时间：** ~60 分钟

## 问题

每六个月就会有一个新的 multi-agent framework 发布。2023 年的 AutoGen。2024 年的 CrewAI。2024 年的 LangGraph 和 OpenAI Swarm。2025 年 4 月的 Google ADK。2026 年 2 月的 Microsoft Agent Framework RC。每篇 press release 都宣称自己是“正确的抽象”。

如果你逐个学习，会很快耗尽力气。APIs 看起来不同。docs 对“agent”是什么意见不一。一个 framework 把 shared memory 叫 “blackboard”，另一个叫 “message pool”，第三个叫 “StateGraph”。你开始怀疑这个领域只是反复造轮子。

不是这样。营销之下，四个 primitives 是稳定的。学一次，就能用一段话读懂每个新 framework。

## 概念

### 四个 primitives

1. **Agent** — 一个 system prompt 加一组 tools。stateless；每次 run 都从 system prompt 和当前 message history 开始。
2. **Handoff** — 从一个 agent 到另一个 agent 的 structured transfer of control。机制上，它是返回新 agent 的 tool call，或沿着条件前进的 graph edge。
3. **Shared state** — 多个 agents 可读（有时可写）的任何 data structure。Message pool、blackboard、key-value store、vector memory。
4. **Orchestrator** — 决定谁下一个发言的人或机制。选项包括：显式 graph（deterministic）、LLM speaker-selector（soft）、上一个 speaker 的 handoff call（OpenAI Swarm），或 queue 上的 scheduler（swarm architecture）。

这就是整个 design space。每个 framework 都在这些轴上选择默认值；剩下的只是 surface syntax。

### 每个 2026 framework 如何映射

| Framework | Agent | Handoff | Shared state | Orchestrator |
|-----------|-------|---------|--------------|--------------|
| OpenAI Swarm / Agents SDK | `Agent(instructions, tools)` | tool returns Agent | caller's problem | the LLM's next handoff call |
| AutoGen v0.4 / AG2 | `ConversableAgent` | speaker-selector on GroupChat | message pool | selector function (LLM or round-robin) |
| CrewAI | `Agent(role, goal, backstory)` | `Process.Sequential / Hierarchical` | Task outputs chained | manager LLM or static order |
| LangGraph | node function | graph edge + condition | `StateGraph` reducer | the graph, deterministic |
| Microsoft Agent Framework | agent + orchestration patterns | pattern-specific | thread / context | pattern-specific |
| Google ADK | agent + A2A card | A2A task | A2A artifacts | host decides |

表层差异看起来很大。底层：同样四个 knobs。

### 为什么这重要

一旦你看见 primitives，framework 比较就变成一个简短 checklist：

- orchestrator 是信任 LLM 来 route（Swarm），还是把 routing 固定在代码里（LangGraph）？
- shared state 是 full-history（GroupChat），还是 projected（StateGraph reducer）？
- agents 能修改彼此的 prompts（CrewAI manager），还是只能 hand off（Swarm）？

这三个问题回答了一个问题 80% 的 framework fit。你不再寻找“最好的 multi-agent framework”，而是围绕真正关心的轴设计。

### Stateless insight

除了 shared state，每个 primitive 都是 stateless。Agent 是 (prompt, tools) 的函数。Handoff 是 function call。Orchestrator 是 scheduler。**系统里唯一 stateful 的东西是 shared state。** 所有有趣的 bugs 都在那里：memory poisoning（第 15 课）、message ordering、versioning、write contention。

隐藏 shared state 的 frameworks（Swarm）把问题推给 caller。集中 shared state 的 frameworks（LangGraph checkpoint、AutoGen pool）让它可检查，但把 coordination cost 转移到 shared-state implementation 上。

### 单个 primitive 的解剖

#### Agent

```
Agent = (system_prompt, tools, model, optional_name)
```

没有 memory。没有 state。两个 system prompt 和 tools 相同的 agents 是可互换的。所有看起来像 per-agent state 的东西，其实都在 shared state 或 handoff protocol 中。

#### Handoff

```
Handoff = (from_agent, to_agent, reason, payload)
```

三种实现占主导：

- **Function return** — tool 返回下一个 agent。这是 OpenAI Swarm pattern。Agents 把 routing 放进 tool schemas。
- **Graph edge** — LangGraph。Edges 是 declarative。LLM 生成一个值；condition 选择下一个 node。
- **Speaker selection** — AutoGen GroupChat。selector function（有时本身也是 LLM call）读取 pool 并选择谁下一个说话。

#### Shared state

```
SharedState = { messages: [], artifacts: {}, context: {} }
```

最低限度是 message list。通常更多：structured artifacts（CrewAI Task outputs）、typed context（LangGraph reducers）、external memory（MCP、vector DB）。

两种 topology：**full pool**（每个 agent 看到每条 message）和 **projected**（agents 看到 role-scoped view）。Full pools 简单但扩展差。Projected pools 能扩展，但需要前置 schema design。

#### Orchestrator

```
Orchestrator = ({state, last_speaker}) -> next_agent
```

四种 flavor：

- **Static** — graph 在 build time 固定（LangGraph deterministic、CrewAI Sequential）。
- **LLM-selected** — LLM 读取 pool 并选择下一个 speaker（AutoGen、CrewAI Hierarchical）。
- **Handoff-driven** — 当前 agent 通过调用 handoff tool 决定（Swarm）。
- **Queue-driven** — workers 从 shared queue 拉取任务；没有显式 next-speaker（swarm architectures、Matrix）。

### Frameworks 之间真正变化的东西

一旦 primitives 固定，剩下的 design decisions 是：

- **Memory strategy** — ephemeral vs durable checkpointing（LangGraph checkpointer）。
- **Safety boundary** — 谁能批准 handoff（human-in-the-loop）。
- **Cost accounting** — per-agent token budgets。
- **Observability** — tracing handoffs，persisting state for replay。

这些都能在 primitives 之上实现。它们都不是新 primitives。

## 构建它

`code/main.py` 用约 150 行 stdlib Python 实现四个 primitives。没有真实 LLM — 每个 agent 是 scripted policy，让焦点保持在 coordination structure 上。

这个文件导出：

- `Agent` — name、system prompt、tools、policy function 的 dataclass。
- `Handoff` — 返回新 agent 的 function。
- `SharedState` — thread-safe message pool。
- `Orchestrator` — 三个变体：`StaticOrchestrator`、`HandoffOrchestrator`、`LLMSelectorOrchestrator`（simulated）。

demo 用所有三种 orchestrator types 运行同一个三 agent pipeline（research → write → review），并在最后打印 message pool。你会看到 outputs 的差异只在于 *谁选择 next*；agents 和 shared state 在各次运行中完全相同。

运行：

```
python3 code/main.py
```

预期输出：三次 orchestrator runs，每种 pattern 一次。每次都打印最终 message pool。如果 researcher 认为已经 done，handoff-driven run 会到达更少 agents — 这就是 LLM-routing tradeoff 的微缩版。

## 使用它

`outputs/skill-primitive-mapper.md` 是一个 skill：读取任意 multi-agent codebase 或 framework doc，并返回 four-primitive mapping。在深入读 docs 前，用它快速理解一个新 framework release。

## 发布它

采用新 framework 之前，为它写 primitive mapping。如果写不出来，要么 docs 不完整，要么 framework 正在发明第五个 primitive（少见 — 通常是你没见过的一种 shared-state flavor）。

把 mapping 固定进 architecture doc。新成员加入时，先发 mapping，再发 API docs。framework 版本变化时，diff mapping，而不是 changelog。

## 练习

1. 用不同 agent policies 运行 `code/main.py` 三次。观察 orchestrator choice 如何改变哪些 agents 会运行。
2. 实现第四种 orchestrator type：queue-driven，agents 从 shared state poll work。会出现什么 deadlock，你如何检测？
3. 取 LangGraph quickstart（https://docs.langchain.com/oss/python/langgraph/workflows-agents），把它改写成四个 primitives。LangGraph 的哪些 abstractions 是 1:1 映射，哪些只是 convenience wrappers？
4. 阅读 OpenAI Swarm cookbook（https://developers.openai.com/cookbook/examples/orchestrating_agents）。识别 Swarm 让四个 primitives 中哪个最 ergonomic，又把哪个推给 caller。
5. 在表里找一个完全隐藏 shared state 的 framework。解释当 agents 需要跨 handoffs 协调且不能重读 history 时会坏什么。

## 关键词汇

| 术语 | 人们常说 | 实际含义 |
|------|----------------|------------------------|
| Agent | “带 tools 的 LLM” | 一个 `(system_prompt, tools, model)` triple。Stateless。 |
| Handoff | “控制权转移” | 指名下一个 agent 和可选 payload 的 structured call。三种实现：function return、graph edge、speaker selection。 |
| Shared state | “Memory” / “context” | multi-agent system 中唯一 stateful 的部分。Message pool 或 blackboard。 |
| Orchestrator | “Coordinator” | 决定谁下一个运行的东西。Static graph、LLM selector、handoff-driven 或 queue-driven。 |
| Primitive | “Abstraction” | 每个 framework 参数化的四个轴之一。不是 framework feature。 |
| Message pool | “Shared chat history” | Full-history shared state。容易推理，扩展差。 |
| Projected state | “Scoped view” | shared state 的 role-specific view。可扩展，但需要 schema design。 |
| Speaker selection | “谁下一个说话” | 一种 orchestrator pattern：function（通常是 LLM）从 group 中选择下一个 agent。 |

## 延伸阅读

- [OpenAI cookbook: Orchestrating Agents — Routines and Handoffs](https://developers.openai.com/cookbook/examples/orchestrating_agents) — 对 handoff-driven orchestration 最清楚的阐述
- [AutoGen stable docs](https://microsoft.github.io/autogen/stable/) — GroupChat + speaker selection 是 LLM-selected orchestration 的参考
- [LangGraph workflows and agents](https://docs.langchain.com/oss/python/langgraph/workflows-agents) — graph-edge orchestration 和 reducer-based shared state
- [CrewAI introduction](https://docs.crewai.com/en/introduction) — role-goal-backstory agents、Sequential / Hierarchical processes
- [AG2 (community AutoGen continuation)](https://github.com/ag2ai/ag2) — Microsoft 将 v0.4 转入 maintenance 后仍活跃的 AutoGen v0.2 line
