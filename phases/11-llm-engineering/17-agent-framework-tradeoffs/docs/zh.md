# Agent Framework Tradeoffs：LangGraph vs CrewAI vs AutoGen vs Agno

> 每个 framework 都卖同一个 demo（research agent 生成 report），也隐藏同一个 bug（state schema 与 orchestration layer 打架）。选择 abstractions 与问题形状匹配的 framework；其他都是你会写两遍的 glue。

**类型：** 学习
**语言：** Python
**前置要求：** 阶段 11 · 09（Function Calling），阶段 11 · 16（LangGraph）
**时间：** ~45 分钟

## 问题

你有一个需要多次 LLM call 的任务。也许是 research workflow（plan、search、summarize、cite）。也许是 code-review pipeline（parse diff、critique、patch、validate）。也许是 multi-turn assistant，能 book flights、write emails、file expense reports。你选择了一个 framework。

三天后，你发现 framework 的 abstractions leak。CrewAI 给你 roles，但当 “researcher” 需要把 structured plan 交给 “writer” 时会跟你打架。AutoGen 给你 agents 之间的 chat，但没有 first-class state，所以 checkpoint 只是 conversation log 的 pickle。LangGraph 给你 state graph，但要求你在知道 agent 会做什么之前命名每个 transition。Agno 给你 single-agent primitive，但当你试图 fan out 到三个 concurrent workers 时会尖叫。

修复方式不是“选择最好的 framework”。而是把 framework 的 core abstraction 与问题形状匹配。本课画出这张地图。

## 概念

![Agent framework matrix: core abstraction vs problem shape](../assets/framework-matrix.svg)

四个 frameworks 主导 2026 年生态。它们的 core abstractions 并不相同。

| Framework | Core abstraction | Best fit | Worst fit |
|-----------|------------------|----------|-----------|
| **LangGraph** | `StateGraph` — typed state, nodes, conditional edges, checkpointer. | Workflows with explicit state and human-in-the-loop interrupts; production agents needing time-travel debugging. | Loose, role-driven brainstorming where the topology is unknown. |
| **CrewAI** | `Crew` — roles (goal, backstory), tasks, process (sequential or hierarchical). | Role-playing or persona-driven workflows with a short linear/hierarchical plan. | Anything stateful beyond the crew's turn history; complex branching. |
| **AutoGen** | `ConversableAgent` pair — two or more agents that speak in turns until an exit condition. | Multi-agent *dialogue* (teacher-student, proposer-critic, actor-reviewer) where the thinking emerges from the chat. | Deterministic workflows with a known DAG; anything needing durable state across restarts. |
| **Agno** | `Agent` — a single LLM + tools + memory, composable into teams. | Fast-to-build single agents and lightweight teams; strong multi-modality and built-in storage drivers. | Deep, explicitly-branched graphs with custom reducers. |

### “Abstraction” 到底是什么意思

Framework 的 core abstraction 是你 pitch architecture 时画在白板上的东西。

- **LangGraph** -> 你画 graph。Nodes 是 steps，edges 是 transitions，每个点的 state object 都有类型。Mental model 是 state machine。
- **CrewAI** -> 你画 org chart。每个 role 有 job description，manager 路由 tasks。Mental model 是一支小型 specialist team。
- **AutoGen** -> 你画 Slack DM。两个 agents 互相发消息；需要 moderator 时加入第三个。Mental model 是 chat。
- **Agno** -> 你画一个带 tools 的单盒子。多个盒子并排就是 team。Mental model 是“agent with batteries included”。

### State question

State 是多数 framework choices 在生产中崩溃的地方。

- **LangGraph。** Typed state（`TypedDict` 或 Pydantic model）、per-field reducers、first-class checkpointer（SQLite/Postgres/Redis）。Resume、interrupt、time-travel 都免费。（见阶段 11 · 16。）
- **CrewAI。** State 通过 `context` field 作为 strings 在 tasks 之间流动，或通过 `output_pydantic` 结构化。默认没有 durable per-crew store；如果 crew 要跨 restart 存活，你自己接。
- **AutoGen。** State 是 chat history 和用户自定义 `context`。Conversation transcripts 可以持久化；任意 workflow state 需要你写 adapters。
- **Agno。** 通过 `storage=` 附加到 `Agent` 的 built-in storage drivers（SQLite、Postgres、Mongo、Redis、DynamoDB），自动持久化 conversation sessions 和 user memories。不是 full graph checkpointer，而是 session store。

### Branching question

每个非平凡 agent 都会 branch。谁决定 branch 很重要。

- **LangGraph** — 你通过 conditional edges 决定。Routing 是带 named branches 的 Python function。Branches 是 compiled graph 中的 first-class，checkpointer 记录走了哪条 branch。
- **CrewAI** — hierarchical mode 下 manager 决定；sequential mode 下你在 build time 决定。Routing 隐含在 task list 中；manager prompt 之外没有 first-class “if”。
- **AutoGen** — agents 通过 chat 决定。Branching 从 who speaks next 中涌现。`GroupChatManager` 选择下一个 speaker；你可以手写 `speaker_selection_method`，但默认是 LLM-driven。
- **Agno** — agent 通过下一步调用哪个 tool 决定。Teams 有 coordinator/router/collaborator modes；超出这些的 branching 是开发者责任。

### Observability question

- **LangGraph** — 通过 LangSmith 或任意 OTel exporter 使用 OpenTelemetry。每个 node transition 都是 trace span；checkpoints 也作为 replayable traces。LangSmith 是 first-party；Langfuse/Phoenix 也有 adapters。
- **CrewAI** — 自 2025 年末起 first-class OpenTelemetry；集成 Langfuse、Phoenix、Opik、AgentOps。
- **AutoGen** — 通过 `autogen-core` 做 OpenTelemetry integration；AgentOps 和 Opik 有 connectors。Tracing granularity 是 per-agent-message，而不是 per-node。
- **Agno** — 内置 `monitoring=True` flag 加 OpenTelemetry exporters；与 Langfuse 的 session traces 紧密集成。

### Cost and latency

四个 frameworks 都增加 per-call overhead（framework logic、validation、serialization）。开销从低到高大致是：Agno ≈ LangGraph < CrewAI ≈ AutoGen。差异主要由 framework 多做多少额外 LLM routing 决定。CrewAI 的 hierarchical manager 会花 tokens 决定谁下一个；AutoGen 的 `GroupChatManager` 也一样。LangGraph 只在你写 `llm.invoke` 的地方花 tokens。Agno 的 single-agent path 很薄。

当 per-run cost 重要时，优先使用 explicit routing（LangGraph edges、AutoGen `speaker_selection_method`），而不是 LLM-selected routing。

### Interoperability

- **LangGraph** ↔ **LangChain** tools、retrievers、LLMs。First-class MCP adapter（tools imported as MCP servers）。
- **CrewAI** ↔ tools 继承 `BaseTool`；LangChain tools、LlamaIndex tools 和 MCP tools 都能适配。通过 `allow_delegation=True` 实现 crew-to-crew delegation。
- **AutoGen** -> `FunctionTool` 包装任意 Python callable；有 MCP adapter。与 AG2 ecosystem 的 agent-to-agent patterns 紧耦合。
- **Agno** -> `@tool` decorator 或 BaseTool subclass；MCP adapter；tools 可跨 agents 和 teams 共享。

## Skill

> 你能用一句话解释为什么某个 framework 适合某个 agent problem。

Pre-build checklist：

1. **Draw the shape。** 这是 graph（typed state、named transitions）？Role play（specialists hand off work）？Chat（agents talk until done）？还是 single agent with tools？
2. **Decide who branches。** Developer-decided branching -> LangGraph。Manager-agent-decided -> CrewAI hierarchical。Chat-emergent -> AutoGen。Tool-call-decided -> Agno。
3. **Check the state budget。** 需要 resume-from-checkpoint？Time-travel？Mid-run human interrupts？如果是，LangGraph 是默认；Agno sessions 覆盖 conversation-scoped state。
4. **Check the cost budget。** LLM-selected routing 每 turn 额外花 tokens。如果 agent 每天运行数千次，优先 explicit routing。
5. **Budget the framework overhead。** 每个 framework 都是一个额外 dependency。如果任务只是两次 LLM calls 和一个 tool，写 30 行 plain Python；没有 framework 比任何 framework 都便宜。

在你能画出 graph、org chart、chat 或 agent box 之前，拒绝拿起 framework。拒绝选择一个会迫使你与它 state model 对抗的 framework。

## Decision Matrix

| Problem shape | Preferred framework | Why |
|---------------|---------------------|-----|
| Workflow DAG with typed state, human approvals, long-running | LangGraph | First-class state, checkpointer, interrupts, time-travel. |
| Research / writing pipeline with distinct roles | CrewAI (sequential) or LangGraph subgraphs | Role-per-task is cheap to express in CrewAI; scale up with LangGraph when branching gets complex. |
| Proposer-critic or teacher-student dialogue | AutoGen | Two-agent chat is its native shape. |
| Single agent with tools, sessions, memory | Agno | Thinnest setup, built-in storage and memory. |
| Thousands of parallel fanouts with reducers | LangGraph + `Send` | The only one with a first-class parallel dispatch primitive. |
| Quick prototype, no framework commitment | Plain Python + provider SDK | No framework is the fastest framework. |

## 练习

1. **Easy.** 用同一个任务“research Anthropic's headquarters, write a 200-word brief, cite sources”分别在 LangGraph（四个 nodes：plan、search、write、cite）和 CrewAI（三个 roles：researcher、writer、editor）中实现。报告每次 run 的 token cost 和代码行数。
2. **Medium.** 在 AutoGen（researcher ↔ writer chat，editor 通过 `GroupChat` 加入）和 Agno（一个带 `search_tools`、`write_tools` 和 session store 的 single agent）中实现同一个任务。按（a）每次 run 成本，（b）crash 后 resume 能力，（c）write step 前插入 human approval 的能力，对四个实现排序。
3. **Hard.** 构建 decision-tree script `pick_framework.py`，输入一个短 problem description（JSON：`{has_typed_state, has_roles, has_dialogue, has_parallel_fanout, needs_resume}`），返回 recommendation 和一句 justification。在你自己设计的六个 cases 上验证。

## 关键术语

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Orchestration | “How the agents coordinate” | 决定哪个 node/role/agent 下一个运行的层。 |
| Durable state | “Resume after a restart” | 进程死亡后仍存在的 state，附着在 checkpoint 或 session store 上。 |
| LLM-selected routing | “Let the model decide” | Planner LLM 每 turn 选择下一步；灵活但每个 decision 都花 tokens。 |
| Explicit routing | “Developer decides” | Python function 或 static edge 选择下一步；便宜且可审计。 |
| Crew | “A CrewAI team” | Roles + tasks + process（sequential 或 hierarchical）绑定成一个 runnable。 |
| GroupChat | “AutoGen's multi-agent chat” | N 个 agents 之间由 speaker selector 管理的 conversation。 |
| Team (Agno) | “Multi-agent Agno” | 一组 agents 上的 route / coordinate / collaborate mode。 |
| StateGraph | “LangGraph's graph” | Typed-state、node、conditional-edge、checkpointer primitive。 |

## 延伸阅读

- [LangGraph documentation](https://langchain-ai.github.io/langgraph/) — StateGraph、checkpointers、interrupts、time-travel。
- [CrewAI documentation](https://docs.crewai.com/) — Crews、Flows、Agents、Tasks、Processes。
- [AutoGen documentation](https://microsoft.github.io/autogen/) — ConversableAgent、GroupChat、teams、tools。
- [Agno documentation](https://docs.agno.com/) — Agent、Team、Workflow、storage、memory。
- [Anthropic — Building effective agents (Dec 2024)](https://www.anthropic.com/research/building-effective-agents) — framework-agnostic pattern library（prompt chaining、routing、parallelization、orchestrator-workers、evaluator-optimizer）。
- [Yao et al., "ReAct: Synergizing Reasoning and Acting" (ICLR 2023)](https://arxiv.org/abs/2210.03629) — 每个 framework 都在包装的 primitive。
- [Wu et al., "AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation" (2023)](https://arxiv.org/abs/2308.08155) — AutoGen 设计论文。
- [Park et al., "Generative Agents: Interactive Simulacra of Human Behavior" (UIST 2023)](https://arxiv.org/abs/2304.03442) — CrewAI-style persona stacks 的 role-play foundation。
- Phase 11 · 16（LangGraph）— 本课对比的 framework。
- Phase 11 · 19（Reflexion）— 干净映射到 LangGraph、但很难映射到 CrewAI 的 pattern。
- Phase 11 · 22（Production observability）— 如何 instrument 你选择的 framework。
