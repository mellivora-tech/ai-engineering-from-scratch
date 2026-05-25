# Generative Agents 与 Emergent Simulation

> Park et al. 2023（UIST '23, arXiv:2304.03442）在 **Smallville** 这个 sandbox 中放入 25 个 agents，使用三部分 architecture：**memory stream**（natural-language log）、**reflection**（agent 对自身 stream 生成的 higher-level syntheses）和 **plan**（day-level behavior，然后 sub-plans）。标志性结果是 Valentine's Day party emergence：一个 agent 被 seed 为“想举办 Valentine's Day party”，没有额外 scripting，邀请在 population 中传播，日期被协调，party 发生了 — 来自 24 个一开始对此毫无所知的 agents。Ablations 显示三个 components 都是 believability 所必需的。记录的 failures 是 spatial-norm errors（进入关门商店、共用 single-person bathrooms）。这是 2026 年 agent simulations 和 multi-agent social evaluation 的 reference architecture。

**类型：** 学习 + 构建
**语言：** Python（stdlib）
**前置要求：** 阶段 16 · 04（Primitive Model），阶段 16 · 13（Shared Memory）
**时间：** ~75 分钟

## 问题

大多数 multi-agent systems 是紧密 scripted teams：planner plans、coder codes、reviewer reviews。这适合 well-defined tasks。它无法捕捉 agents 有 memory、priorities 和 open world 时产生的 emergent、unscripted behavior。Research、society simulation，以及越来越多的 game AI 需要第二种系统。

Smallville architecture 是它的 benchmark。在 Park 2023 之前，最好的 agent simulations 是浅层 script-followers；之后，这个 pattern 成为 open worlds 中 generative agents 的默认做法。如果你在 2026 年构建 agent simulation，要么使用 Smallville 的三个 components，要么明确解释为什么不使用。

## 概念

### 三个 components

**Memory stream。** observations、actions、reflections 和 plans 的 append-only log。每个 entry 有 timestamp、type、description（natural language）和 derived metadata：**recency**、**importance**（agent 自评 1-10）以及 **relevance**（与当前 query 的 cosine similarity）。

```
[2026-02-14 09:12:03] observation: Isabella Rodriguez asked me if I like jazz
[2026-02-14 09:14:22] reflection:   I enjoy long conversations about music
[2026-02-14 10:05:00] plan:         Attend Isabella's Valentine's Day party tonight
```

Memory retrieval 组合三个分数：`score = w_recency * e^(-decay * age) + w_importance * importance + w_relevance * cos_sim`。Top-k entries 进入当前 prompt。

**Reflection。** 周期性地（每 N 条 memories 或重要 events 上），agent 从 recent memories 中生成 higher-order syntheses。Reflection entries 重新进入 stream，并像其他 memory 一样可 retrieval。这是 agents 建立“understandings”的方式 — architecture 中 long-term beliefs 的等价物。

**Plan。** Top-down decomposition。先是 day-level plan 的 broad strokes（“go to work, have dinner with Klaus”）。然后 hour-level plans。然后 action-level plans。Plans 可 revise：当 observation 与 plan 矛盾时，agent 重新规划受影响 segment。

### 为什么三者都重要（ablation）

Park et al. 做了 ablations，分别去掉 observation、reflection 和 plan。每次 ablation 都伤害 believability：

- 没有 **observation**，agent 错过 context，并根据 stale beliefs 行动。
- 没有 **reflection**，agent 无法形成 higher-order beliefs；interactions 停留浅层。
- 没有 **plan**，behavior 变成 reactive noise；goals 消散。

Human raters 给出的 believability scores 在三者齐全时最高；去掉任何一个都会产生可测 regression。

### Valentine's Day emergence

一个 agent，Isabella Rodriguez，被 seed goal：“wants to throw a Valentine's Day party at Hobbs Cafe on Feb 14 at 5pm。”另外 24 个 agents 没有这个 seed。经过 simulated days：

1. Isabella 的 plan 包含邀请人。
2. 每次 invitation 成为 neighbor memory stream 中的 observation。
3. 该 neighbor 的 reflection 生成 beliefs：“Isabella is throwing a party.”
4. neighbor 的 plan 加入 “attend party on Feb 14.”
5. neighbors 告诉其他 neighbors。invitation 在没有 central coordination 的情况下传播。
6. 2 月 14 日下午 5 点，几个 agents 汇聚到 Hobbs Cafe。

这是 technical sense 上的 emergence：system-level behavior（party）来自 local interactions（双边邀请 + individual planning），没有 central orchestrator。

### 已记录的 failure modes

Park et al. 明确记录：

- **Spatial norm errors。** Agents 走进关门商店。agents 尝试使用同一个 single-person bathroom。agents 在不该吃饭的房间吃饭。model 不会仅从 environment 推断 social-physical norms。
- **Memory overflow。** 深 simulation runs 让 memory-retrieval cost 增长。实用 remedy：periodic memory compaction（summarize-and-prune）和 low-importance entries 的 decay。
- **Reflection hallucination。** Reflections 会编造 memory stream 中不存在的关系。缓解：reflection prompts 包含 source memory ids，并在 retrieval time verify。

这些是 production-relevant failure modes：任何 2026 agent simulation 都继承它们。

### Three-component implementation rules

1. **Memory is append-only。** 永远不要 mutate memory entry。Corrections 是 new entries。
2. **Importance scores are cheap。** 在 write time 调 LLM 给 importance 1-10 打分。缓存分数。
3. **Retrieval is ranked, not filtered。** 按 combined score 取 top-k；不要用 hard filters（会丢 context）。
4. **Reflection runs periodically。** 当 unprocessed memories 的 importance sum 超过 threshold（例如 150）时触发。
5. **Plans are revisable。** 当新 observation 与 plan 矛盾时，只 regenerate affected segment，而不是 whole plan。

### Smallville 之后的 generative agents

2024-2026 follow-up literature 扩展了该 architecture：

- **用于 policy / market research 的 multi-agent social simulation。** Smallville-like populations 模拟用户对 features 的 behavior。比 A/B tests 更快；accuracy 有争议。
- **游戏 NPC AI。** RPGs 中的 Smallville agents 生成 emergent storylines，而不是 scripted quests。
- **Generative-agent evaluation benchmarks。** metric 不再是 task accuracy，而是长期运行中的 believability + coherence of behavior。

architecture 是 reference。extensions 会替换 components（vector store for memory、retrieval-augmented reflection、neurosymbolic plan），但保留三部分结构。

### 这对 multi-agent engineering 为什么重要

Smallville 证明了，当 components 正确时，multi-agent emergence 并不昂贵。这个 architecture 已经在 open-source models 上被复现（smaller LLMs 的 believability 会平滑下降，而不是断崖式下降）。任何需要 **emergent social behavior** 的 production system 都使用这个形状。任何需要 **tight task execution** 的系统则使用本 phase 前面讲的 supervisor / roles / primitives patterns。

## 构建它

`code/main.py` 用 stdlib Python 和 scripted agent policies（无真实 LLM）实现三个 components。demo 以 miniature 形式复现 Valentine's-party emergence：

- `MemoryStream` — 带 recency/importance/relevance retrieval 的 append-only log。
- `reflect(stream)` — 对 recent high-importance memories 做 scripted reflection。
- `plan(agent_state)` — 基于当前 beliefs 的 day-level 和 hour-level plans。
- Scenario：5 个 agents。Agent 1 从 “throw party at 5pm” 开始。经过 simulated ticks，invitation 传播，agents 汇聚。

运行：

```
python3 code/main.py
```

预期输出：tick-by-tick trace。到 final tick，5 个 agents 中至少 3 个把 party 放进 plan，并聚集到 party location。单个 seed 在没有 orchestrator 的情况下产生 coordinated arrival。

## 使用它

`outputs/skill-simulation-designer.md` 设计 generative-agent simulation：agent 数量、memory schema、reflection cadence、plan horizon 和 evaluation metric。

## 发布它

Production simulations 的规则：

- **Memory is the database。** scale 时选择真实 store（vector DB、Postgres）。in-memory stdlib 只适合 prototypes。
- **Log retrieval trace。** 对每个 action，记录驱动它的 top-k memories。这是你的 debug ability。
- **Budget per-agent tokens。** 每个 agent 的 retrieve + reflect + plan per tick 是 O(k) LLM calls。N agents × T ticks × calls-per-tick 会吞掉预算。
- **定期 compact memory。** summarize-and-prune low-importance entries。retention policy 是设计决策，不是细节。
- **显式检测 spatial / social norm violations。** architecture 不会自动学会它们。

## 练习

1. 运行 `code/main.py`。确认 3+ agents 汇聚到 party。把 agents 增加到 10 — emergence 还会发生吗？
2. 移除 reflection step。behavior 看起来怎样？映射到 Park 2023 的 ablation finding。
3. 引入一个 competing seeded goal（“Klaus wants to give a research talk at 5pm”）。agents 会分裂，还是一个 goal 占主导？决定因素是什么？
4. 添加 spatial constraints：Hobbs Cafe 最多容纳 4 个 agents。simulation 能优雅处理 overflow，还是会击中 “single-person bathroom” failure pattern？
5. 阅读 Park et al.（arXiv:2304.03442）Section 6（emergent behavior experiments）。识别一个你的 miniature 无法复现的 behavior。你需要增强 architecture 的哪个 component？

## 关键词汇

| 术语 | 人们常说 | 实际含义 |
|------|----------------|------------------------|
| Memory stream | “agent 的日记” | observations、actions、reflections、plans 的 append-only log。 |
| Recency | “memory 有多新” | 按 age 做 exponential-decay score。 |
| Importance | “agent 有多在意” | write time 自评 1-10。cached。 |
| Relevance | “与当前 query 多相关” | cosine similarity（embedding-based）。 |
| Reflection | “Higher-order belief” | 从 recent memories 生成的 synthesis，并作为新 memory 重新 ingest。 |
| Plan | “Day/hour/action decomposition” | top-down plan tree。observations 矛盾时可 revise。 |
| Smallville | “Park 2023 的 sandbox” | 25-agent simulation，产生 Valentine's Day emergence。 |
| Believability | “quality metric” | human-rater 对 behavior 是否像 plausible agent 的评分。 |

## 延伸阅读

- [Park et al. — Generative Agents: Interactive Simulacra of Human Behavior](https://arxiv.org/abs/2304.03442) — reference architecture
- [UIST '23 paper page](https://dl.acm.org/doi/10.1145/3586183.3606763) — publication venue
- [Smallville code release](https://github.com/joonspk-research/generative_agents) — reference Python implementation
- [Hayes-Roth 1985 — A Blackboard Architecture for Control](https://www.sciencedirect.com/science/article/abs/pii/0004370285900639) — structured-memory agents 的 prior art
