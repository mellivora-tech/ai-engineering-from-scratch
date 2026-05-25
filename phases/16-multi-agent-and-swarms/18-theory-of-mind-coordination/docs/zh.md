# Theory of Mind 与 Emergent Coordination

> Li et al.（arXiv:2310.10701）显示，cooperative text game 中的 LLM agents 会表现出 **emergent high-order Theory of Mind**（ToM）— 推理另一个 agent 相信第三个 agent 的 beliefs — 但会因 context management 和 hallucination 在 long-horizon planning 上失败。Riedl（arXiv:2510.05174）测量 population 中的 higher-order synergy，发现 **只有** ToM-prompt condition 会产生 identity-linked differentiation 和 goal-directed complementarity；lower-capacity LLMs 只显示 spurious emergence。也就是说，coordination emergence 是 prompt-conditional 且 model-dependent 的，不是免费的。本课实现一个 minimal ToM-aware agent，在有无 ToM prompting 的情况下运行 cooperative task，并按 Riedl 2025 protocol 测量 coordination delta。

**类型：** 学习 + 构建
**语言：** Python（stdlib）
**前置要求：** 阶段 16 · 07（Society of Mind and Debate），阶段 16 · 17（Generative Agents）
**时间：** ~75 分钟

## 问题

Multi-agent coordination 常常看起来像魔法：agents 分工、anticipate each other、避免重复。通常这种 “emergence” 是 prompt engineering 的 artifact — 有人告诉 agents “coordinate”。拿掉 prompt，coordination 也没了。

Riedl 2025 的发现更严格：在 controlled conditions 下，只有当 agents 被 prompt 去推理 **other agents' minds**（ToM）时，coordination 才会涌现。没有 ToM prompt，即使强 models 也表现出无法通过 statistical controls 的 coordination patterns。这对 production 很重要：teams 会发布 prompt-dependent 且 brittle 的 “multi-agent coordination” features。

本课把 ToM 当作一个具体能力（reasoning about beliefs about beliefs），构建 minimal ToM-aware agent，并测量真实 coordination 与 prompt dressing 的区别。

## 概念

### ToM 是什么

发展心理学：3 岁儿童认为每个人的 inner world 都与自己一样。5 岁儿童理解别人有不同 beliefs。7 岁儿童推理 beliefs about beliefs（“她认为我以为球在杯子下面”）。这就是 zeroth、first 和 second-order ToM。

对 LLM agents，ToM orders 映射为：

- **Zeroth-order：** 不建模 others。agent 只根据自己的 observations 行动。
- **First-order：** agent 有每个 other agent 的 beliefs model。“Alice believes X.”
- **Second-order：** agent 建模 recursive beliefs。“Alice believes that Bob believes X.”

Li et al. 2023 发现，LLM agents 在 cooperative games 中会涌现 first- 和 second-order ToM，但在 long horizon 和 unreliable communication 下退化。

### Sally-Anne test 简介

1985 年 false-belief test：Sally 把 marble 放在 basket A 后离开。Anne 把它移到 basket B。Sally 回来后会去哪里找？有 first-order ToM 的孩子会说 basket A（Sally 的 belief 不同于 reality）。没有的会说 basket B。

GPT-4 时代的 LLMs 在直接提问时能通过 Sally-Anne-style tests。当 narrative 很长、scene 多次变化，或 question 间接措辞时会失败。这就是 2026 production LLMs 的 ToM 实用状态。

### Riedl 的 coordination measurement

Riedl（arXiv:2510.05174）构建 population-scale test：N agents、cooperative objective、可变 prompt conditions。测量：

1. **Identity-linked differentiation。** agents 是否随时间发展出稳定 role distinctions？
2. **Goal-directed complementarity。** agents 的 actions 是否互补（不同 subtasks）而不是重复？
3. **Higher-order synergy。** group 是否完成任何 subset 都无法完成的事情的统计度量。

结果：只有 ToM prompt condition 下，三个 metrics 都产生高于 baseline 的 signal。没有 ToM prompting 时，moderate-capacity models 的 metrics 接近 chance。large models 在没有显式 ToM prompt 时也有一些 coordination，但效果小于显式 prompting。

### Coordination illusion

没有 statistical controls，demos 中的 “emergent coordination” 往往反映：

- 把 coordination 烘进去的 prompt engineering（system prompts 写着 “work together”）。
- observer bias（我们看到自己期待的 patterns）。
- 对 successful runs 的 post-hoc selection。

没有 measurable signal 却营销 “emergent coordination” 的 production systems 应被视为 marketing。先测量，再宣称。

### Minimal ToM-aware agent

结构：

```
agent state:
  own_beliefs:    {facts the agent believes}
  other_models:   {other_agent_id -> {beliefs_the_agent_attributes_to_them}}
  actions_last_N: [history of others' actions]

observation update:
  - update own_beliefs from direct observation
  - update other_models[agent_id] from their action + prior beliefs

action selection:
  - enumerate candidate actions
  - for each, predict what each other agent will do next given their modeled beliefs
  - pick action that maximizes joint outcome under those predictions
```

`other_models` attribute 是 ToM state。First-order ToM 只保留一层。Second-order 添加 `other_models[i][other_models_of_j]` — 我认为 agent i 认为 agent j 相信什么。

### 为什么 long-horizon 会伤害

Li et al. 记录：context limits 会让 agents 忘记哪个 belief 属于谁。hallucination 会向 other-agent models 加入 false beliefs。两者都会产生 “I thought he thought X” errors，并随时间复合。

论文和 2024-2026 follow-ups 中记录的缓解：

- **Prompt 中显式 ToM state。** structured format：`{agent_id: belief_list}`。强制 retrieval 保留 identity-belief binding。
- **更短 reasoning chains。** 每轮更少 ToM updates 减少 compounding hallucination。
- **External ToM store。** 在 LLM context 外维护 model；每轮只 inject 相关 parts。

### ToM 在 production 中哪里失败

- **Adversarial settings。** 拥有良好 ToM 的 agents 更容易被操纵（你可以建模它们对你的建模，然后 exploit）。
- **Heterogeneous teams。** models 不同时，适用于一个 opponent 的 ToM model 不会泛化到另一个。
- **Ground-truth-dependent tasks。** ToM 关于 beliefs；如果 correctness 取决于 facts，ToM 可能分散注意力。

### 你真正能测量的 coordination

团队 coordination 是真实而非 prompt-dressed 的三个实用 signals：

1. **Complementarity over time。** 在 multi-turn task 中，agents 的 actions 是否覆盖 disjoint sub-tasks？
2. **Anticipation。** agent A 在 T+1 的 action 是否依赖于对 B 在 T+2 行动的 prediction，且 prediction 成真？
3. **Correction。** 当 A 在 T 时误读 B 的 belief，A 是否在 T+2 前纠正？

这些都能在 logged multi-agent system 中测量。它们是 “coordination” narrative 的实质版本。

## 构建它

`code/main.py` 实现：

- `ToMAgent` — 追踪 own beliefs 和 per-other-agent belief models。
- 一个 cooperative task：三个 agents 必须从三个 boxes 收集三个 tokens；每个 box 可容纳一个 token。agents 不能通信；它们从彼此 actions 推断 intent。
- 两种配置：`zeroth_order`（无 ToM）和 `first_order`（带一层 belief model 的 ToM）。
- 在 200 randomized trials 上测量：completion rate、duplication rate（两个 agents target 同一个 box）、average turns to completion。

运行：

```
python3 code/main.py
```

预期输出：zeroth-order agents 约 35% duplication rate，10 turns 内完成约 60% trials。First-order ToM agents duplication 约 5%，completion 约 95%。delta 就是 measurable coordination effect。

## 使用它

`outputs/skill-tom-auditor.md` 是一个 skill，用来审计 multi-agent system 的 “emergent coordination” claim。检查 prompt dressing、相对 control 的 statistical significance，以及 measured complementarity。

## 发布它

Coordination claims checklist：

- **Control condition。** 去掉 coordination prompt 的系统版本。两个都测。
- **Statistical test。** system 与 control 在 metric 上的差异是否在 `p < 0.05` 显著？
- **Complementarity measure。** 随时间的 action-disjointness，而不只是 final success。
- **Failure-case log。** agents miscoordinate 时，ToM state 是什么样？
- **Model-capacity disclosure。** 如果效果在 smaller models 上消失，要说清楚。

## 练习

1. 运行 `code/main.py`。确认 first-order ToM 把 duplication rate 降低约 7x。扩展到 5 agents 和 5 boxes 时差距还在吗？
2. 实现 second-order ToM（agent A 建模 B 对 C 的想法）。它比 first-order 有提升吗？在哪些任务上？
3. 向 ToM state 注入 **hallucination**：每轮随机 flip 一个 belief。first-order performance 下降多少？
4. 阅读 Li et al.（arXiv:2310.10701）。复现 “long-horizon degradation” finding：当 turns 从 10 增加到 30，你的 first-order ToM performance 如何变化？
5. 阅读 Riedl 2025（arXiv:2510.05174）。在你的 simulation logs 上实现 higher-order synergy statistic。没有 ToM prompt condition 时效果存在吗？

## 关键词汇

| 术语 | 人们常说 | 实际含义 |
|------|----------------|------------------------|
| Theory of Mind | “理解他人的 mind” | 建模另一个 agent beliefs 的能力。按 order（0, 1, 2+）分级。 |
| Sally-Anne test | “false-belief test” | 1985 developmental psychology；LLMs 通过 plain versions，复杂版本失败。 |
| First-order ToM | “A believes X” | 建模另一个 agent 关于 facts 的 beliefs。 |
| Second-order ToM | “A believes B believes X” | 更深一层的 recursive modeling。 |
| Identity-linked differentiation | “随时间稳定 roles” | Riedl 的 metric：roles 持续，而不是随机。 |
| Goal-directed complementarity | “Disjoint actions” | agents target 不同 subtasks，而不是同一个。 |
| Higher-order synergy | “group exceeds any subset” | Riedl 对真实 coordination 的 statistical measure。 |
| Coordination illusion | “看起来 coordinated” | 没有 measurable signal 的 prompt-dressed coordination appearance。 |

## 延伸阅读

- [Li et al. — Theory of Mind for Multi-Agent Collaboration via Large Language Models](https://arxiv.org/abs/2310.10701) — cooperative games 中的 emergent ToM；long-horizon failure modes
- [Riedl — Emergent Coordination in Multi-Agent Language Models](https://arxiv.org/abs/2510.05174) — population-scale measurement；ToM prompting 是 load-bearing condition
- [Premack & Woodruff — Does the chimpanzee have a theory of mind?](https://www.cambridge.org/core/journals/behavioral-and-brain-sciences/article/does-the-chimpanzee-have-a-theory-of-mind/1E96B02CD9850E69AF20F81FA7EB3595) — ToM concept 的 1978 origin
- [Baron-Cohen, Leslie, Frith — Does the autistic child have a theory of mind?](https://www.cambridge.org/core/journals/behavioral-and-brain-sciences/article/does-the-autistic-child-have-a-theory-of-mind/) — Sally-Anne paper（1985）
