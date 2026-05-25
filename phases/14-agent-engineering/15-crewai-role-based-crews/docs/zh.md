# CrewAI：Role-Based Crews 和 Flows

> CrewAI 是 2026 年 role-based multi-agent framework。四个 primitives：Agent、Task、Crew、Process。两个 top-level shapes：Crews（autonomous、role-based collaboration）和 Flows（event-driven、deterministic）。文档很直白：“for any production-ready application, start with a Flow.”

**类型：** 学习 + 构建
**语言：** Python (stdlib)
**前置要求：** 阶段 14 · 12（Workflow Patterns），阶段 14 · 14（Actor Model）
**时间：** ~75 分钟

## 学习目标

- 说出 CrewAI 的四个 primitives（Agent、Task、Crew、Process），以及每个负责什么。
- 区分 Sequential、Hierarchical 和 Consensual processes；为每种 workload 选择一个。
- 区分 Crews（autonomous role-based）和 Flows（event-driven deterministic），并解释 docs 的 production recommendation。
- 用 `@tool` decorator 和 `BaseTool` subclass 接工具；推理 structured outputs vs free text。
- 说出四种 CrewAI memory types，以及什么时候值得使用。
- 实现一个 stdlib three-agent crew（researcher、writer、editor），生成 brief。
- 识别三种 CrewAI failure modes：prompt-bloat、manager-LLM tax、brittle handoffs。

## 问题

采用 multi-agent frameworks 的团队会撞上同一堵墙。“Autonomous collaboration”在 demo 中听起来很棒。然后客户提交 bug，你需要 deterministic replay。或者 finance 问一次 LLM-routed crew 每次 run 要花多少钱。或者 on-call 需要知道凌晨 3 点哪个 agent 卡住了。

Free-form LLM-routed crews 都不能干净地回答这些问题。Pure DAGs 都能回答，但会失去 brainstorming agent 需要的探索形态。

CrewAI 的拆分诚实面对了这个取舍。Crews 用于 collaborative、role-based、exploratory work。Flows 用于 event-driven、code-owned、auditable production。同一个 framework，两种形状，按 surface 选择。

## 概念

### 四个 primitives

CrewAI 的 surface 很小。记住这个，其余都是 config。

- **Agent。** `role + goal + backstory + tools + (optional) llm`。Backstory 是承重结构。它塑造 tone、judgment、以及 agent 何时停止。Tools 是 agent 可以调用的函数（见下文）。
- **Task。** `description + expected_output + agent + (optional) context + (optional) output_pydantic`。可复用 work unit。`expected_output` 是 contract。`context` 列出上游 tasks，其 outputs 会传入。`output_pydantic` 强制 structured shape。
- **Crew。** Container。拥有 `agents` 列表、`tasks` 列表、`process`，以及可选的 `memory` + `verbose` + `manager_llm` settings。
- **Process。** Execution strategy。Sequential、Hierarchical、Consensual。决定 run 的形状。

Agents 不能直接互相看见。Tasks 引用 agents。Crew 编排 tasks。Process 决定谁选择下一个 task。这就是完整 mental model。

### Sequential vs Hierarchical vs Consensual

- **Sequential。** Tasks 按声明顺序运行。Task N 的 output 可以作为 `context` 提供给 task N+1。成本最低。最可预测。适合顺序固定的情况。
- **Hierarchical。** Manager Agent（单独 LLM call）在 specialists 之间 route。CrewAI 会从你的 `manager_llm` config 或默认设置生成 manager。Manager 每轮选择下一个 task，并可以 refuse 或 re-route。适合你有四个以上 specialists，且顺序真正取决于 prior output 的情况。
- **Consensual。** Beta。Agents 对下一步投票。除了研究场景，很少值得这些 round trips。

Hierarchical 会在每个 specialist call 之外额外增加 per-round LLM call（manager）。五步 run 上 token cost 可能变成三倍。只有需要 routing 时才付这个成本。

### Crews vs Flows

这是 2026 年 docs 开头就强调的 framing。

- **Crew。** LLM-driven autonomy。Framework 在 runtime 选择形状。适合：research、brainstorming、first drafts，以及 path 本身是答案一部分的地方。难 replay。难 test。prototype 便宜。
- **Flow。** 你拥有的 event-driven graph。`@start` 标记入口。`@listen(topic)` 标记某个 step 会在另一个 step 发出该 topic 时触发。每个 step 都是普通 Python（内部可以调用 Crew）。适合：production。Observable。Testable。Deterministic。

Docs 的 2026 production recommendation：从 Flow 开始。当 autonomy 值得其成本时，把 Crews 作为 Flow steps 内部的 `Crew.kickoff()` calls 折进去。Flow 给 audit trail，Crew 给 exploration。组合它们，不要二选一。

### Tool integration

给 Agent 一个 tool 有三种方式。选择能满足需求的最简单一种。

1. **`@tool` decorator。** Pure functions 变成 tools。Signature 是 schema；docstring 是 LLM 看见的 description。最适合 one-off helpers。

   ```python
   from crewai.tools import tool

   @tool("Search the web")
   def search(query: str) -> str:
       """Return top results for the query."""
       return run_search(query)
   ```

2. **`BaseTool` subclass。** Class-based tool，带 explicit args schema、async support、retries。适合 tool 有 state（client、cache）或需要 structured args 的情况。

   ```python
   from crewai.tools import BaseTool
   from pydantic import BaseModel

   class SearchArgs(BaseModel):
       query: str
       limit: int = 10

   class SearchTool(BaseTool):
       name = "web_search"
       description = "Search the web and return top results."
       args_schema = SearchArgs

       def _run(self, query: str, limit: int = 10) -> str:
           return self.client.search(query, limit=limit)
   ```

3. **Built-in toolkits。** CrewAI 提供 first-party adapters：`SerperDevTool`、`FileReadTool`、`DirectoryReadTool`、`CodeInterpreterTool`、`RagTool`、`WebsiteSearchTool`。一个 import 就能接好。

Structured outputs 使用 Pydantic。在 Task 上传入 `output_pydantic=MyModel`。CrewAI 会根据 model 校验 LLM response，并 coerce 或 retry。把它和紧凑的 `expected_output` string 搭配。Free-text outputs 适合 drafts；structured outputs 才是 downstream Flows 可消费的东西。

### Memory hooks

CrewAI 开箱提供四种 memory types。它们可以组合：一个 Crew 可以一次启用全部四种。

- **Short-term。** 单次 run 内的 conversation buffer。结束后清空。
- **Long-term。** 跨 runs 持久化。存储在 vector DB 中（默认 Chroma，可替换）。按与当前 task 的 similarity 检索。
- **Entity。** Per-entity facts。“Customer X is on the enterprise plan.” 按 entity key，而不是 similarity。跨 runs 存活。
- **Contextual。** Assembly-time retrieval。在 Agent 需要时拉取相关 memory，而不是预加载。

在 Crew 上用 `memory=True` 或按类型 config 启用。背后由你配置的 embeddings provider 支持（默认 OpenAI，可换成本地）。Memory 是 CrewAI 相比更薄 frameworks 体现价值的地方之一；纯 LangGraph 需要你自己接每一种。

### CrewAI 适合什么时候

- 三到六个 agents，具名 roles 和 collaborative workflow。Drafting、reviewing、planning、brainstorming。
- LLM 对下一步的判断本身是价值的一部分的 routing（Hierarchical）。
- 团队更愿意读 `role + goal + backstory`，而不是 graph definition 的场景。

### CrewAI 不适合什么时候

- 严格 ordering 的 deterministic DAGs。使用 LangGraph（第 13 课）。Graph shape 才是正确 abstraction；CrewAI 的 role framing 会变成摩擦。
- 亚秒级 latency budgets。Hierarchical 会增加 round trips。即使 Sequential 也会串行化包含 backstories 和 prior outputs 的 prompts。
- Single-agent loops。跳过 framework；agent loop（第 1 课）加 tool registry 更短。

第 17 课（Agent Framework Tradeoffs）会用矩阵展开这个问题。简短版本：CrewAI 位于“collaborative role-based”角落。

### Dependency shape

独立于 LangChain。Python 3.10 到 3.13。使用 `uv`。2026 年初 GitHub stars 超过 30k。AWS Bedrock integration 有文档；它们的 benchmarks 声称 QA tasks 上比 LangGraph 快 5.76 倍。把 framework-vendor numbers 当作方向性参考。

### 这个模式会在哪里出错

- **Prompt-bloat from backstories。** 每个 agent 2000 词 backstory，加上 five-agent crew，会在第一次 tool call 前烧完 context budget。Backstories 保持在 200 词以内。Agents 之间复用短语；不要重复 house style 五次。
- **Manager-LLM token tax。** Hierarchical process 会在每次 specialist call 之前增加一个 manager LLM call。五个 task 的 crew 会变成六次 LLM calls 而不是五次，而且 manager call 会携带完整 task list 和 prior outputs。除非 routing 取决于 output，否则切换到 Sequential。
- **Brittle handoffs。** Task N 的 `expected_output` 是“an outline”。Task N+1 把它作为 `context` 读取，并尝试解析三节。LLM 生成了四节。下游 Agent 现场发挥。用 Task N 上的 `output_pydantic` 修复，让 Task N+1 读取 typed object，而不是 free text。
- **Crew-as-prod。** Free-form Crew 没有 Flow wrapper 就发布到生产。Output variability 高；无法 replay；on-call 无法 diff bad run 和 good run。用 Flow 包起来。

## 构建它

`code/main.py` 实现了两种形状的 stdlib 版本，以及一个 three-agent crew。

形状：

- `Agent`、`Task` dataclasses，匹配 CrewAI surface。
- `SequentialCrew.kickoff(inputs)` 按声明顺序运行 tasks，并把 outputs 作为 `context` 穿起来。
- `HierarchicalCrew.kickoff(topic)` 添加一个 manager Agent，每轮选择下一个 specialist，在 “done” 停止。
- 带 `@start` 和 `@listen(topic)` decorators 的 `Flow`，一个 tiny event loop 和 trace。
- `tool(name)` decorator，模拟 CrewAI 的 `@tool` 形状。
- `Memory`，包含 `short_term`、`long_term`、`entity` stores；mocked similarity 使用 numpy。
- Mock LLM responses 是按 role 加 input prefix keyed 的 hardcoded strings。无网络。Deterministic。

具体 demo：researcher、writer、editor crew 为“agent engineering 2026”生成 brief。Researcher 拉取（mocked）sources。Writer 起草。Editor 收紧。同一个 crew 会通过 Flow 运行，以展示 deterministic shape。

运行它：

```bash
python3 code/main.py
```

Trace 覆盖：sequential crew 通过 `context` 穿起 outputs，hierarchical crew 带 manager picks（researcher、writer、editor，然后 “done”），flow 用 explicit topics（`researched`、`drafted`、`edited`）运行同样三步，tool calls 通过 `@tool` route，以及 long-term memory 跨两次 kickoffs 存活。

Crew trace 是流动的；manager 原则上可以重排。Flow trace 是固定的。这个选择就是本课要点。

## 使用它

- **CrewAI Flow** 用于生产。即使 Flow 只有一步，调用 `Crew.kickoff()`。Flow 给出 audit boundary。
- **CrewAI Crew (Sequential)** 用于 ordering 清晰的 collaborative work，尤其 first drafts 和 review loops。
- **CrewAI Crew (Hierarchical)** 用于 routing 取决于 output，且你有四个以上 specialists 的情况。
- **LangGraph**（第 13 课）用于 explicit state machines、durable resume、strict ordering。
- **AutoGen v0.4**（第 14 课）用于 actor-model concurrency 和 fault isolation。
- **OpenAI Agents SDK**（第 16 课）用于 OpenAI-first products，带 handoffs 和 guardrails。
- **Claude Agent SDK**（第 17 课）用于 Claude-first products，带 subagents 和 session store。

## 发布它

`outputs/skill-crew-or-flow.md` 会为一个 task 选择 Crew vs Flow，并 scaffold minimal implementation。它会 hard reject Crew-without-backstory、Flow-without-explicit-topics，以及 under three specialists 的 Hierarchical。

## 坑

- **Backstory as flavor。** 它会塑造 outputs。每个 agent 测试三种 variants；variance 是真实存在的。选一个，冻结它。
- **Skipping `expected_output`。** 没有每个 task 的 contract，下游 tasks 会接住 LLM 生成的任何东西。Crew 会跑；audit 会失败。
- **Memory always-on。** 每次 run 都写 long-term。Vector DB 增长。Retrieval 变 noisy。把 writes 限定到 fact 会持久存在的 tasks。
- **Manager prompt drift。** Hierarchical 的 manager prompt 是隐式的。如果 routing 变奇怪，打开 verbose mode dump 出来读。
- **Tool side effects in Crews。** Crew 调用 tool 的次数可能比预期更多。POST、DELETE、payment 属于 Flow step，永远不要作为 Crew tool。

## 练习

1. 把 Sequential crew 转成 Flow。数一数 variability 下降的触点。记下 readability 在哪里下降了。
2. 给 crew 添加 entity memory：关于某个 customer 的 facts 跨 kickoffs 持久化。验证 retrieval 拉到正确 entity。
3. 实现一个 Hierarchical process：manager 在 writer output 至少有三段之前，拒绝 route 给 editor。Trace 这个 retry。
4. 为（mocked）web search 接一个 `BaseTool` subclass。对比它和 `@tool` decorator 版本的 trace shape。
5. 给 editor task 添加 `output_pydantic=Brief`，其中 `Brief` 有 `title`、`summary`、`sections`。让 writer task 先输出一次 malformed JSON；验证 CrewAI 的 retry behavior 出现在 trace 中。
6. 阅读 CrewAI docs intro。把 toy 移植到真实 `crewai` API。Stdlib 版本跳过了哪些 guarantees？
7. 把 AgentOps 或 Langfuse（第 24 课）接到真实 run。Stdlib 版本漏掉了哪些 traces？

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|----------------|------------------------|
| Agent | “Persona” | Role + goal + backstory + tools |
| Task | “Unit of work” | Description + expected output + assignee + optional structured output |
| Crew | “Agent team” | Agents + Tasks + Process 的 container |
| Process | “Execution strategy” | Sequential / Hierarchical / Consensual |
| Flow | “Deterministic workflow” | Event-driven、code-owned、testable |
| Backstory | “Persona prompt” | Agent 的 tone 和 judgment shaper |
| `@tool` | “Function tool” | 把 function 转成 Agent 可调用 tool 的 decorator |
| `BaseTool` | “Class tool” | 带 args schema、retries、async support 的 class-based tool |
| Entity memory | “Per-entity facts” | 按 customer / account / issue scoped 的 memory |
| Long-term memory | “Cross-run memory” | 在 kickoffs 之间存活的 vector-backed memory |
| Contextual memory | “Just-in-time retrieval” | Agent 需要时才拉取的 memory |
| Manager LLM | “Router agent” | Hierarchical process 中选择下一个 task 的额外 LLM |
| `expected_output` | “Task contract” | 告诉 Agent（以及 audit）应该返回什么形状的 string |

## 延伸阅读

- [CrewAI docs introduction](https://docs.crewai.com/en/introduction)：概念和推荐的 production path
- [CrewAI Flows guide](https://docs.crewai.com/en/concepts/flows)：event-driven shape、`@start`、`@listen`
- [CrewAI tools reference](https://docs.crewai.com/en/concepts/tools)：`@tool`、`BaseTool`、built-in toolkits
- [CrewAI memory](https://docs.crewai.com/en/concepts/memory)：short-term、long-term、entity、contextual
- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents)：什么时候 multi-agent 有帮助，什么时候没有
- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview)：state-machine alternative
