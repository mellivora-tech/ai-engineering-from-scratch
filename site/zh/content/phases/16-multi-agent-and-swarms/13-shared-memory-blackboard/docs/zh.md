# Shared Memory 与 Blackboard Patterns

> 2026 年的 multi-agent systems 中并存两种方法：**message pool**（每个人看到每个人的 messages，如 AutoGen GroupChat 或 MetaGPT）和 **blackboard with subscription**（agents 订阅相关 events，如 Context-Aware MCP 或 Matrix framework）。两者都是 multi-agent system 中唯一 stateful 的部分 — 也就是说，两者都是有趣 bugs 所在的位置。参考 failure mode 是 **memory poisoning**：一个 agent hallucinate 了一个“fact”，其他 agents 把它当作 verified，accuracy 以比立即 crash 更难 debug 的方式逐渐衰减。本课用 stdlib 构建两种 structures，注入一个 poisoning attack，并展示 production 中真正有效的三种 mitigations。

**类型：** 学习 + 构建
**语言：** Python（stdlib，`threading`）
**前置要求：** 阶段 16 · 04（Primitive Model），阶段 16 · 09（Parallel Swarm Networks）
**时间：** ~75 分钟

## 问题

Multi-agent systems 需要一个地方让 agents 共享 facts。一个字面选项是“把所有东西都放进 messages 传递” — 但这只是用额外 copying 重新发明 shared state。另一个是“给所有人一个 global log” — 但 global logs 会无界增长，并且容易被 poison。第三个是“为每个 agent project 一个 view” — 可扩展但 schema-heavy。

当某个 agent hallucinate 并把 hallucination 写入 shared state 时，每个 downstream agent 读取该 state 后都会把 hallucination 当成 fact。到人类注意到时，reasoning chain 已经深入五步，root cause 是第三条写入的 message。调试 multi-agent accuracy decay 比调试 crash 更难。

这就是 memory poisoning。它是 MAST taxonomy（Cemri et al., arXiv:2503.13657）中第二多被记录的 failure family，而且是 structural：任何没有 provenance 和 unwritable verifier 的 shared-memory design 迟早会表现出它。

## 概念

### 两种主要 topologies

**Full message pool。** 每个 agent 读取每条 message。AutoGen GroupChat 和 MetaGPT 使用它。简单、透明、可检查，但很难扩展到 ~10 个 agents 以上，因为每个 agent 的 context 会被其他 agents 的工作填满。

```
agent-A ──write──▶ ┌────────────────┐ ◀──read── agent-D
                   │ message pool   │
agent-B ──write──▶ │                │ ◀──read── agent-E
                   │ (global log)   │
agent-C ──write──▶ └────────────────┘ ◀──read── agent-F
```

**Blackboard with subscription。** Agents 声明自己关注的 topics；substrate 只路由相关 messages。CA-MCP（arXiv:2601.11595）和 Matrix decentralized framework（arXiv:2511.21686）使用它。能扩展更远，但需要 upfront schema design 才能让 subscriptions 有意义。

```
                   ┌─ topic: prices ──┐
agent-A ──pub────▶ │                  │ ──▶ agent-D (subscribed)
                   ├─ topic: orders ──┤
agent-B ──pub────▶ │                  │ ──▶ agent-E (subscribed)
                   ├─ topic: alerts ──┤
agent-C ──pub────▶ │                  │ ──▶ agent-F (subscribed)
                   └──────────────────┘
```

### 各自何时胜出

- **Full pool** 在 agents 很少（< 10）、heterogeneous、conversation short-horizon 时胜出。当所有人看到一切时，“谁说了什么”很容易推理。
- **Blackboard** 在 agents 很多、role homogeneous 但 instance 数量多（swarms）、conversation long-running 时胜出。Routing 节省 token cost 和 context pollution。

Production systems 经常混合：顶层（planning layer）使用小 full pool，底层（worker layer）使用 blackboards。

### Memory poisoning 场景

三个 agents 做 research task。Agent A 是 retrieval agent。Agent B 是 summarizer。Agent C 是 analyst。

1. A fetch 一个页面，并向 shared state 写入：“The study reports a 42% accuracy improvement.”
2. fetched page 实际说的是 “4.2% improvement”。A hallucinate 了 decimal。
3. B 读取 shared state，写入：“Large 42% accuracy gain reported (source: A).”
4. C 读取 shared state，写入：“Recommend adoption — 42% lift is transformative.”
5. 最终 report 引用了一个从未存在的 42% 数字。

没有 agent crash。没有 test fail。系统“工作了”。hallucination 通过 shared state 从一个 agent 的 context 跨入每个 downstream agent 的 reasoning。

### 为什么这是 structural

没有 shared state 时，agent A 的 hallucination 停留在 A 的 context。downstream agents 会 re-fetch 或 re-derive，可能捕捉错误。使用 naive shared state 时，A 的 context 变成所有人的 context，hallucination 被洗白成 fact。

问题不是 shared state 本身，而是没有 provenance 且没有 independent verifier 的 shared state。三种 mitigations 解决它：

1. **每次 write 都记录 provenance。** shared state 中每个 entry 记录谁写的、何时写的、用什么 prompt、以及（如适用）agent cite 的 source。Downstream agents 根据 provenance 带怀疑地读取。
2. **Writes versioned，并当作 append-only。** correction 是 supersede 旧 entry 的新 entry，不是 in-place update。保留 audit trail。
3. **保持至少一个无法写入 shared state 的 agent。** read-only verifier agent 抽样 entries、re-fetch sources、标记 inconsistencies。因为它不能写入 pool，所以不能通过 pool 被 poison。

### Blackboard precedent（Hayes-Roth, 1985）

blackboard pattern 比 LLM agents 早四十年。Hayes-Roth（1985，“A Blackboard Architecture for Control”）描述了 specialist Knowledge Sources：它们观察 global blackboard、贡献 partial solutions，并触发其他 sources。2026 年的 blackboard（CA-MCP、Matrix）是同样模式，只是 Knowledge Sources 换成 LLM agents，partial solutions 换成 JSON blobs。旧文献已经记录了 write contention、opportunistic control 和 consistency 的解决方案，现代系统正在重新发现。

### Projection vs full view

纯 blackboard 给每个 subscriber 相同 projection（topic-scoped）。更激进的设计是 **per-agent projection**：每个 agent 拿到为其 role 定制的 view。LangGraph 的 state reducers 是 2026 年 canonical implementation — reducer function 把 global state 折叠成 role-specific slice。

Per-agent projection 能扩展更远，但需要 schema。没有 schema，你会在每个 agent prompt 里重建 ad-hoc projection。

### Write-contention patterns

多个 agents 同时写入是 concurrency problem，不只是 LLM problem。三种 patterns 有效：

- **Sequential writer（single producer）。** 所有 writes 经过一个 coordinator agent 串行化。简单，但成为 bottleneck。
- **Optimistic concurrency with versioning。** 每个 entry 有 version；writers 在 version mismatch 时 fail 并 retry。经典数据库技术。
- **Topic partitioning。** 不同 agents 拥有不同 topics。没有 cross-topic contention。需要设计 partition boundaries。

多数 2026 frameworks 默认 sequential writer，因为 LLM calls 慢到 contention 罕见，bottleneck 不太伤。

### Unwritable verifier

最承重的 mitigation 是 read-only verifier。实现规则：

- verifier 与团队共享 state（读取 blackboard 或 pool）。
- verifier 没有 shared state write handle — 只能写 separate verification channel。
- verifier 独立 fetch writes 中 cited sources。标记 disagreement。
- verifier 的 outputs 被路由到 human 或 separate decision agent，永不 fed back into pool。

没有这种隔离，verifier 的 outputs 会变成 pool 中的新 entries，也就是说 poisoned pool 会 poison verifier，verifier 又会 poison 它自己的 verifications。

## 构建它

`code/main.py` 用 stdlib Python 实现两种 topologies，再加一个 toy poisoning attack 和三种 mitigations。

- `MessagePool` — thread-safe append-only log，可 full read-out。
- `Blackboard` — topic-keyed pub/sub，带 per-agent subscriptions。
- `ProvenanceEntry` — 每次 write 记录 (writer, timestamp, prompt_hash, source_uri)。
- `PoisoningScenario` — 运行三 agent research task，其中 agent A hallucinate 一个 decimal。打印 final report。
- `Verifier` — 一个 read-only agent，re-fetch sources 并标记 inconsistencies。在 verifier present 时运行同一 scenario。

运行：

```
python3 code/main.py
```

预期输出：
- Run 1（无 verifier）：hallucinated 42% 传播到 final report。
- Run 2（有 verifier）：verifier 标记 inconsistency，pool 被标为 “flagged”，final report 包含 retraction。

## 使用它

`outputs/skill-memory-auditor.md` 是一个 skill，用来审计任意 multi-agent system 的 shared-memory design，检查 provenance、versioning 和 verifier separation。在新 multi-agent architectures 进入 production 前运行它。

## 发布它

对任何 shared-memory design：

- 每次 write 都记录 provenance：`(writer, timestamp, prompt_hash, tool_calls_cited, source_uri)`。
- 让 log append-only。corrections 是引用 superseded entry 的新 entries。
- 部署至少一个有 independent source access 的 read-only verifier agent。
- 把 verifier output 路由到 separate channel，而不是 shared pool。
- 记录 supersession writes 的比例 — 比例上升是 hallucination patterns 的早期证据。

## 练习

1. 运行 `code/main.py`。确认 run 1 传播 hallucination，run 2 捕捉它。
2. 添加第二个 hallucination：agent B 编造 dataset size。verifier 应该在没有为它 hard-code 的情况下捕捉两者。
3. 把 full pool 切换为 topic partitions（`prices`、`summaries`、`analyses`）的 blackboard。topic partitioning 让哪些 poisoning scenarios 更难得逞？哪些没有帮助？
4. 阅读 Hayes-Roth（1985，“A Blackboard Architecture for Control”）。识别本文没讨论但 2026 systems 会受益的两个 control patterns。
5. 阅读 CA-MCP（arXiv:2601.11595）。把它的 Shared Context Store 映射到 `code/main.py` 中的 MessagePool 或 Blackboard class。CA-MCP 在上面增加了哪些 primitives？

## 关键词汇

| 术语 | 人们常说 | 实际含义 |
|------|----------------|------------------------|
| Message pool | “Shared chat history” | 每个 agent 都读取的 append-only log。完全透明，扩展差。 |
| Blackboard | “Shared workspace” | Topic-keyed pub/sub。Agents 订阅相关 topics。扩展更远。 |
| Provenance | “谁写了什么” | 每次 write 的 metadata：writer、timestamp、prompt、sources。 |
| Memory poisoning | “Hallucinations spreading” | 一个 agent 的错误进入 shared state，下游 agents 把它当成 fact。 |
| Append-only | “不做 in-place updates” | corrections 是新的 superseding entries。保留 audit trail。 |
| Unwritable verifier | “Independent auditor” | read-only agent，re-fetch sources 并标记 inconsistencies。 |
| Projection | “Scoped view” | 从 global state 计算出的 per-agent view。LangGraph reducers 是 canonical case。 |
| Knowledge Source | “Specialist agent” | Hayes-Roth 1985 对 blackboard participant 的术语。 |

## 延伸阅读

- [Cemri et al. — Why Do Multi-Agent LLM Systems Fail?](https://arxiv.org/abs/2503.13657) — MAST taxonomy；memory poisoning 是 coordination-failure sub-family
- [CA-MCP — Context-Aware Multi-Server MCP](https://arxiv.org/abs/2601.11595) — coordinated MCP servers 的 Shared Context Store
- [Matrix — decentralized multi-agent framework](https://arxiv.org/abs/2511.21686) — 没有 central orchestrator 的 message-queue-based blackboard
- [LangGraph state and reducers](https://docs.langchain.com/oss/python/langgraph/workflows-agents) — production 中的 per-agent projection pattern
- [Anthropic — How we built our multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system) — production deployment 中的 provenance 和 verification notes
