# Orchestration Patterns：Supervisor、Swarm、Hierarchical

> 2026 年 frameworks 中反复出现四种 orchestration patterns：supervisor-worker、swarm / peer-to-peer、hierarchical、debate。Anthropic 的建议是：“It's about building the right system for your needs.” 从简单开始；只有当 single agent 加五种 workflow patterns 不够时，才增加 topology。

**类型：** 学习 + 构建
**语言：** Python (stdlib)
**前置要求：** 阶段 14 · 12（Workflow Patterns），阶段 14 · 25（Multi-Agent Debate）
**时间：** ~60 分钟

## 学习目标

- 说出四种反复出现的 orchestration patterns，以及各自适合什么时候。
- 描述 2026 年 LangChain recommendation：tool-call-based supervision vs supervisor libraries。
- 解释 Anthropic 的 “build the right system” 规则，以及它如何 gate topology choice。
- 基于 common scripted LLM 用 stdlib 实现四种模式。

## 问题

团队常常在需要之前就伸手去拿 “multi-agent”。四种模式会在 frameworks 中反复出现；一旦能命名它们，你就能选择正确那个 — 或者干脆跳过 topology。

## 概念

### Supervisor-worker

- 一个 central routing LLM dispatch 到 specialist agents。
- 决定：loop 回自己、handoff 给 specialist、terminate。
- Specialists 不互相说话；所有 routing 都经过 supervisor。

Frameworks：LangGraph `create_supervisor`、Anthropic orchestrator-workers、CrewAI Hierarchical Process。

**2026 LangChain recommendation：** 通过 direct tool calls 做 supervision，而不是 `create_supervisor`。这样 context engineering control 更细 — 你精确决定每个 specialist 看到什么。

### Swarm / peer-to-peer

- Agents 通过 shared tool surface 直接 handoff。
- 没有 central router。
- 比 supervisor 延迟更低（更少 hops）。
- 更难推理（没有 single point of control）。

Frameworks：LangGraph swarm topology、OpenAI Agents SDK handoffs（当所有 agents 都能 hand off 给所有其他 agents）。

### Hierarchical

- Supervisors 管 sub-supervisors，sub-supervisors 再管 workers。
- 在 LangGraph 中实现为 nested subgraphs；在 CrewAI 中实现为 nested crews。
- 能扩展到大型 agent populations，代价是 operational complexity。

什么时候需要它：当单个 supervisor 的 context budget 放不下所有 specialists 的 descriptions。

### Debate

- Parallel proposers + iterative cross-critique（第 25 课）。
- 严格说不算 orchestration — 更像 verification — 但在 frameworks 中会作为 topology choice 出现。

### CrewAI Crew vs Flow

CrewAI 形式化了两种 deployment modes：

- **Flow** 用于 deterministic event-driven automation（production 推荐起点）。
- **Crew** 用于 autonomous role-based collaboration。

这和上面四种 patterns 正交，但会映射到 topology：Flow 通常是 supervisor 或 hierarchical；Crew 通常是带 LLM router 的 supervisor。

### Anthropic's guidance

“Success in the LLM space isn't about building the most sophisticated system. It's about building the right system for your needs.”

Decision order：

1. Single agent + workflow patterns（第 12 课）— 从这里开始。
2. Supervisor-worker — 当你有 2-4 个 specialists。
3. Swarm — 当 latency 比 reasoning clarity 更重要。
4. Hierarchical — 只有当 supervisor context budget 失败时。
5. Debate — 当 accuracy 比 cost 更重要。

### 这个模式会在哪里出错

- **Topology-first thinking。** 在识别 multi-agent 解决什么问题之前就说 “We need multi-agent”。
- **Swarm 中 bouncing handoffs。** A -> B -> A -> B。使用 hop counters。
- **Fake hierarchy。** 三层只是因为 “enterprise”；实际只有两个 team。压平它。

## 构建它

`code/main.py` 在 stdlib 中基于 scripted LLM 实现四种 patterns：

- `Supervisor` — central router。
- `Swarm` — peer-to-peer with direct handoffs。
- `Hierarchical` — supervisors of supervisors。
- `Debate` — parallel proposers + critique。

每个 pattern 都处理同一个 three-intent task（refund / bug / sales）。Trace shapes 不同。

运行它：

```
python3 code/main.py
```

输出：per-pattern trace + op count。Supervisor 最清晰；swarm 最短；hierarchical 最深；debate 最贵。

## 使用它

- **LangGraph** 用于 supervisor 和 hierarchical（nested subgraphs）。
- **OpenAI Agents SDK** 用于 handoffs-as-tools（supervisor-shaped）。
- **CrewAI Flow** 用于 production deterministic。
- **Custom** 用于 debate，或你想要 exact control 时。

## 发布它

`outputs/skill-orchestration-picker.md` 会选择 topology 并实现它。

## 练习

1. 通过移除 router，把 supervisor-worker 转成 swarm。什么坏了？什么变好了？
2. 给 swarm 添加 hop counter：3 次 handoffs 后拒绝。能抓到 A->B->A bouncing 吗？
3. 为一个 12-specialist domain 构建两层 hierarchical system。没有 nesting 时，context budget 在哪里失败？
4. 在 production-shaped workload 上 profile 四种 patterns。哪个在 latency、cost、accuracy、debuggability 上胜出？
5. 阅读 Anthropic 的 “Building Effective Agents” 文章。把你的每个 production flow 映射到四种模式之一。有映射不干净的吗？

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|----------------|------------------------|
| Supervisor-worker | "Router + specialists" | Central LLM dispatch 到 specialists；它们不互相说话 |
| Swarm | "Peer-to-peer" | 通过 shared tools 直接 handoffs；没有 central router |
| Hierarchical | "Supervisors of supervisors" | 用于 large populations 的 nested subgraphs |
| Debate | "Proposer + critique" | Parallel proposers、cross-critique（第 25 课） |
| Tool-call-based supervision | "Supervisor without a library" | 把 supervisor 实现为 direct tool calls，以控制 context |
| Crew | "Autonomous team" | CrewAI 的 role-based collaboration mode |
| Flow | "Deterministic workflow" | CrewAI 的 event-driven production mode |

## 延伸阅读

- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) — five patterns + agent vs workflow
- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview) — supervisor、swarm、hierarchical
- [CrewAI docs](https://docs.crewai.com/en/introduction) — Crew vs Flow
- [Du et al., Society of Minds (arXiv:2305.14325)](https://arxiv.org/abs/2305.14325) — debate pattern
