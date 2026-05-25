# Case Studies 与 2026 State of the Art

> 三个 production-grade references 值得 end-to-end 学习，每个展示 multi-agent engineering 的不同切面。**Anthropic Research system**（orchestrator-worker、15x tokens、比 single-agent Opus 4 高 +90.2%、rainbow deployments）是 canonical supervisor case。**MetaGPT / ChatDev**（用于 software engineering 的 SOP-encoded role specialization；ChatDev 的 “communicative dehallucination”；MacNet extension 通过 DAGs 扩展到 >1000 agents，arXiv:2406.07155）是 canonical role-decomposition case。**OpenClaw / Moltbook**（最初是 Peter Steinberger 的 Clawdbot，2025 年 11 月；两次改名；到 2026 年 3 月 247k GitHub stars；local ReAct-loop agents；Moltbook 是 agent-only social network，launch 后几天内约 2.3M agent accounts，2026-03-10 被 Meta 收购）展示 population scale 会发生什么：emergent economic activity、prompt-injection risks、state-level regulation（中国在 2026 年 3 月限制政府电脑使用 OpenClaw）。**Framework landscape April 2026：** LangGraph 和 CrewAI 领先 production；AG2 是 community AutoGen continuation；Microsoft AutoGen 处于 maintenance mode（合并进 Microsoft Agent Framework，RC Feb 2026）；OpenAI Agents SDK 是 production Swarm successor；Google ADK（2025 年 4 月）是 A2A-native entrant。每个主流 framework 现在都支持 MCP；多数支持 A2A。本课 end-to-end 阅读每个 case，提炼共同 patterns，让你能为下一个 production system 选对 reference。

**类型：** 学习（capstone）
**语言：** —
**前置要求：** 阶段 16 全部内容（第 01-24 课）
**时间：** ~90 分钟

## 问题

Multi-agent engineering 是年轻 discipline。production references 很少，而且每个覆盖 space 的不同部分。逐个阅读很有用；把它们作为一组比较更有用。本课把三个 canonical 2026 case studies 作为 end-to-end reading list，固定 common patterns，并映射 framework landscape，让你根据 knowledge 而不是 marketing 做 framework choices。

## 概念

### Anthropic Research system

production supervisor-worker case。Claude Opus 4 负责 plan 和 synthesize；Claude Sonnet 4 subagents 并行 research。published engineering post：https://www.anthropic.com/engineering/multi-agent-research-system。

关键 measured results：

- 相比 single-agent Opus 4，在 internal research evals 上 **+90.2%** improvement。
- **BrowseComp variance 的 80%** 由 **token usage alone** 解释 — multi-agent 很大程度上因每个 subagent 有 fresh context window 而胜出。
- 相比 single-agent，每个 query **15x tokens**。
- 因为 agents long-running 且 stateful，需要 **Rainbow deployment**。

已编码的 design lessons：

1. **按 query complexity 缩放 effort。** Simple → 1 个 agent，3-10 tool calls。Medium → 3 agents。Complex research → 10+ subagents。
2. **先广后窄。** Subagents 做 wide searches；lead synthesize；follow-up subagents 做 targeted deeps。
3. **Rainbow deploys。** 保持旧 runtime versions 存活，直到 in-flight agents 完成。
4. **Verification is not optional。** 观察到如果没有 explicit verifier roles，系统会 hallucinate。

这是 production scale 下 supervisor-worker topology（阶段 16 · 05）的 reference case。

### MetaGPT / ChatDev

production SOP-role-decomposition case。覆盖 arXiv:2308.00352（MetaGPT）和 arXiv:2307.07924（ChatDev）。

MetaGPT 把 software-engineering SOPs 编码为 role prompts：Product Manager、Architect、Project Manager、Engineer、QA Engineer。论文 framing：`Code = SOP(Team)`。每个 role 都有 narrow、specialized prompt；inter-role handoffs 传递 structured artifacts（PRD docs、architecture docs、code）。

ChatDev 的贡献：**communicative dehallucination**。Agents 在回答前请求具体信息 — designer agent 会先问 programmer 目标 language，而不是猜测后 sketch UI。论文报告这种机制可测地降低 multi-agent pipelines 中的 hallucination。

MacNet（arXiv:2406.07155）通过 **DAGs** 将 ChatDev 扩展到 **>1000 agents**。每个 DAG node 是 role specialization；edges 编码 handoff contracts。能够 scale 是因为 routing 显式且 offline-computable。

Design lessons：

1. **Structure matters more than size。** 紧凑 5-role SOP team 胜过 50-agent unstructured group。
2. **Handoff contracts in writing。** roles 之间传递的 artifacts 遵循 schema。
3. **Communicative dehallucination** 是便宜且承重的 pattern。
4. **DAGs scale further than chat。** 当 flow 可知时，把它编码下来。

这是 role specialization（阶段 16 · 08）和 structured topology（阶段 16 · 15）的 reference case。

### OpenClaw / Moltbook ecosystem

production population-scale case。Timeline：

- **Nov 2025：** Clawdbot（Peter Steinberger 的 local ReAct-loop coding agent）发布。
- **Dec 2025 – Mar 2026：** 两次改名（Clawdbot → OpenClaw → continued under OpenClaw）。
- **Feb 2026：** Moltbook 作为同一 primitives 上的 agent-only social network 发布；几天内约 2.3M agent accounts。
- **Mar 2026（2026-03-10）：** Meta 收购 Moltbook。
- **Mar 2026：** 中国限制政府电脑使用 OpenClaw。
- **Mar 2026：** OpenClaw 达到 247k GitHub stars。

这就是把数百万 agents 放到 shared substrate 上的 multi-agent：

- **Emergent economic activity。** Agents 使用 token-payments 互相买卖和服务。
- **Population scale 的 prompt-injection risks。** 一个 viral agent profile 中的 malicious prompt 会在数小时内传播到数千 agent-to-agent interactions。
- **State-level regulatory response。** launch 后数周内，regulation 触达 ecosystem。

这个 case 的 design lessons 一部分是 technical，一部分是 governance：

1. **Population scale 的 multi-agent 是新 regime。** 单系统 best practices（verification、role clarity）仍适用，但不充分。
2. **Prompt injection is the new XSS。** 默认把 agent profiles 和 cross-agent messages 当作 untrusted input。
3. **Regulation 比 design cycles 更快。** 为此规划。
4. **Open-source + viral scale 会叠加。** 约 4 个月 247k stars 很不寻常；为 deploy-burst-load 设计。

见 [OpenClaw Wikipedia](https://en.wikipedia.org/wiki/OpenClaw) 以及 CNBC / Palo Alto Networks reporting 获取 ecosystem detail。technical underpinnings 方面，Clawdbot / OpenClaw repos 暴露 local ReAct loop；Moltbook public posts 展示其上的 social-graph architecture。

### Framework landscape April 2026

| Framework | Status | Best for | Notes |
|---|---|---|---|
| **LangGraph** (LangChain) | Production leader | structured graph + checkpointing + human-in-the-loop | recommended default for production |
| **CrewAI** | Production leader | role-based crews with Sequential/Hierarchical processes | strong for role decomposition |
| **AG2** | Community maintained | GroupChat + speaker selection | AutoGen v0.2 continuation |
| **Microsoft AutoGen** | Maintenance mode (Feb 2026) | — | merged into Microsoft Agent Framework RC |
| **Microsoft Agent Framework** | RC (Feb 2026) | orchestration patterns + enterprise integration | new entrant; watch |
| **OpenAI Agents SDK** | Production | Swarm successor | tool-return handoff pattern |
| **Google ADK** | Production (April 2025) | A2A-native | Google Cloud integration |
| **Anthropic Claude Agent SDK** | Production | single-agent + Research extension | see the Research system post |

每个主流 framework 现在都支持 **MCP**；多数支持 **A2A**。Protocol compatibility 不再是 differentiator。

### 三个 cases 的 common patterns

1. **Orchestrator + workers**（Anthropic explicit supervisor，MetaGPT PM-as-supervisor，OpenClaw individual agents + network effects）。
2. **Structured handoff contracts**（Anthropic subagent task descriptions，MetaGPT PRD/architecture docs，OpenClaw A2A artifacts）。
3. **Verification as first-class role**（Anthropic 的 verifier，MetaGPT 的 QA Engineer，OpenClaw 的 in-network validators）。
4. **Scaling is topology + substrate, not just more agents**（rainbow deploys，MacNet DAGs，population-scale substrates）。
5. **Cost is material and disclosed**（15x tokens，MetaGPT 的 per-role budget，Moltbook 的 per-interaction pricing）。
6. **Security posture is explicit**（Anthropic sandboxing，MetaGPT role restrictions，OpenClaw prompt-injection as known attack surface）。

### 为你的下一个项目选择 reference

- **Production research / knowledge task → Anthropic Research。** Fresh-context subagents 会赢。
- **Engineering / tool-chain workflow → MetaGPT / ChatDev。** Roles + SOPs + handoff contracts。
- **Network-effect social product → OpenClaw / Moltbook。** Substrate + emergent economy。
- **Classic enterprise automation → CrewAI or LangGraph**（production leader，stable runtime）。

### 2026 state-of-the-art summary

截至 2026 年 4 月：

- **Frameworks are converging。** MCP + A2A support 是 table stakes。Handoff semantics 是剩下的 design choice。
- **Evaluation is hardening。** SWE-bench Pro、MARBLE、STRATUS mitigation benchmarks。Pro 是当前 contamination-resistant reality check。
- **Production failure rates are measurable**（Cemri 2025 MAST；real MAS 上 41-86.7%）。field 已走出 “looks great in demo” era。
- **Cost is the central engineering constraint。** Token cost per task、wall-clock per interaction、rainbow-deploy overhead。Multi-agent 在 accuracy 上赢，在 cost 上输 — 这就是 business decision。
- **Regulation is a near-term input, not a background concern。** jurisdictions 的动作快于 individual deploy cycles。

## 使用它

`outputs/skill-case-study-mapper.md` 是一个 skill：读取 proposed multi-agent system design，并映射到最接近的 case study，暴露该 case study 已经测试过的 design decisions。

## 发布它

2026 年 production multi-agent starter rules：

- **从 case study 开始，而不是从零开始。** 在 Anthropic Research / MetaGPT / OpenClaw 中选择最接近的并改造。
- **采用 MCP + A2A。** framework portability 有价值；protocol support 是免费的。
- **用 SWE-bench Pro 或 internal Pro-equivalent 测量。** Verified 已 contaminated。
- **支付 verification tax。** independent verifier 约消耗 token budget 的 20-30%，换来可测 correctness。
- **对 long-running agents 做 rainbow deploy。** 预期 multi-hour agent runs 会成为常态。
- **阅读 WMAC 2026 和 MAST follow-ups。** 这个 discipline 变化很快。

## 练习

1. 从头到尾阅读 Anthropic Research system post。识别如果你把 Opus 4 换成较小 model（例如 Haiku 4），三个会改变的 design decisions。
2. 阅读 MetaGPT Sections 3-4（arXiv:2308.00352）。把你自己 domain（不是 software）中的一个 SOP 编码为 role prompts。这个 SOP 暗示多少 roles？
3. 阅读 ChatDev（arXiv:2307.07924）。识别 “communicative dehallucination” 的机制。把它实现进你已有的一个 multi-agent system。
4. 阅读 OpenClaw 和 Moltbook。选一个 population scale 下出现、但 5-agent system 中不会出现的具体 failure mode。你会如何工程化防护？
5. 选择你当前的 multi-agent project。三个 case studies 中哪个是最接近 reference？该 case study 中哪些 design decisions 你尚未采用？写下本季度会采用的一个。

## 关键词汇

| 术语 | 人们常说 | 实际含义 |
|------|----------------|------------------------|
| Anthropic Research | “supervisor reference” | Claude Opus 4 + Sonnet 4 subagents；15x tokens；比 single-agent +90.2%。 |
| MetaGPT | “SOP as prompts” | software engineering 的 role decomposition；`Code = SOP(Team)`。 |
| ChatDev | “Agents as roles” | Designer / programmer / reviewer / tester；communicative dehallucination。 |
| MacNet | “Scale ChatDev via DAG” | arXiv:2406.07155；通过 explicit DAG routing 扩展到 1000+ agents。 |
| OpenClaw | “Local ReAct-loop agents” | Steinberger 的项目；到 2026 年 3 月 247k stars。 |
| Moltbook | “Agent-only social network” | 2.3M agent accounts；2026 年 3 月被 Meta 收购。 |
| Rainbow deploy | “Multiple versions concurrent” | 为 in-flight long-running agents 保持旧 runtime versions 存活。 |
| Communicative dehallucination | “Ask before answering” | Agents 向 peers 请求 specifics，而不是猜测。 |
| WMAC 2026 | “AAAI workshop” | 2026 年 4 月 multi-agent coordination community focal point。 |

## 延伸阅读

- [Anthropic — How we built our multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system) — supervisor-worker production reference
- [MetaGPT — Meta Programming for Multi-Agent Collaborative Framework](https://arxiv.org/abs/2308.00352) — SOP-role decomposition
- [ChatDev — Communicative Agents for Software Development](https://arxiv.org/abs/2307.07924) — communicative dehallucination
- [MacNet — scaling role-based agents to 1000+](https://arxiv.org/abs/2406.07155) — DAG-based scale
- [OpenClaw on Wikipedia](https://en.wikipedia.org/wiki/OpenClaw) — ecosystem overview
- [WMAC 2026](https://multiagents.org/2026/) — AAAI 2026 Bridge Program Workshop on Multi-Agent Coordination
- [LangGraph docs](https://docs.langchain.com/oss/python/langgraph/workflows-agents) — production leader
- [CrewAI docs](https://docs.crewai.com/en/introduction) — role-based framework
