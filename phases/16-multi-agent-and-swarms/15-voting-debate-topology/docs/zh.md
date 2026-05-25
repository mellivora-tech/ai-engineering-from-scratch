# Voting、Self-Consistency 与 Debate Topology

> 最便宜的 aggregation：sample N 个 independent agents，然后 majority-vote。Wang et al. 2022 self-consistency 用一个 model 采样 N 次做了这件事。Multi-agent 用 **heterogeneous** agents 扩展它来逃离 monoculture — 不同 models、不同 prompts、不同 temperatures、不同 contexts。除了 majority vote，debate topology 也重要：MultiAgentBench（arXiv:2503.01935, ACL 2025）评估 star / chain / tree / graph coordination，发现 **graph 最适合 research**，但超过约 4 个 agents 后有 “coordination tax”。AgentVerse（ICLR 2024）记录了两种 emergent patterns — volunteer behaviors 和 conformity behaviors — conformity 既是 feature（寻找 consensus），也是 risk（groupthink，第 24 课）。本课映射 topology space，构建每种 variant，并测量 coordination tax。

**类型：** 学习 + 构建
**语言：** Python（stdlib）
**前置要求：** 阶段 16 · 07（Society of Mind and Debate），阶段 16 · 14（Consensus and BFT）
**时间：** ~75 分钟

## 问题

Debate 可以提高 accuracy（Du et al., arXiv:2305.14325）。也可以降低它。debate 是否有帮助取决于四个结构选择：

1. 谁和谁说话（topology）。
2. 多少 rounds（Du 2023：rounds 和 agents 都独立重要）。
3. agents 是否 heterogeneous（不同 base models 打破 monoculture）。
4. 是否存在 adversarial voice（steel-manning vs. straw-manning）。

把 “run 5 agents and vote” 直接加到任务上的 teams，常常比 single agent 还退步。这些失败不是随机的。它们跟 topology 和 heterogeneity 相关。本课是 topology map。

## 概念

### Self-consistency，single-model baseline

Wang et al. 2022（“Self-Consistency Improves Chain of Thought Reasoning”）以 temperature > 0 从同一个 model 采样 N 次，并对 reasoning-path answers majority-vote。GSM8K 结果：N=40 samples 比单个 greedy decode 有显著提升。Self-consistency 是 multi-agent voting 的 single-agent precursor。

限制：self-consistency 使用一个 base model。errors 天生相关。如果 model 有 systematic bias，所有 N 个 samples 都共享它。

### Multi-agent vote，heterogeneous extension

用 N 个 *不同* agents 替换 N 个 samples。不同 base models（Claude、GPT、Llama）、不同 prompts、不同 tool access。收益：uncorrelated errors。成本：不同 agents 成本不同；协调它们增加 overhead。

heterogeneous debate 的 2026 canonical name 是 **A-HMAD** — Adversarial Heterogeneous Multi-Agent Debate。并非普遍采用，但 papers 用这个术语表示“不同 models debate，从而减少 monoculture collapse 的 correlated errors”。

### 四种 topologies

```
star                chain               tree                graph

    ┌─A─┐           A─B─C─D         ┌──A──┐              A───B
    │   │                           │     │              │ × │
    B   C                           B     C              D───C
    │   │                          / \   / \
    D   E                         D   E F   G           (fully connected)
```

Star：一个 hub，其他只和 hub 说话。等价于没有 back-channel 的 supervisor-worker。
Chain：线性，每个 agent 看到前一个 output。像 pipeline。
Tree：hierarchical，hierarchical agent systems 使用（第 06 课）。
Graph：any-to-any。包括 fully-connected clique 和 arbitrary DAGs。

### Coordination tax（MultiAgentBench）

MultiAgentBench（MARBLE, ACL 2025, arXiv:2503.01935）在 research、coding、planning 等 task suite 上 benchmark star、chain、tree、graph。关键结果：

- **Graph** topology 在 research tasks 上胜出。信息 any-to-any flow；agents 能 critique 彼此。
- **Star** 在 fast-answer factual tasks 上胜出。hub 过滤并 consolidate。
- **Chain** 在 stepwise pipelines（staged refinement）上胜出。
- **Coordination tax** 在 graph topology 超过约 4 个 agents 后出现。Wall-clock 和 token cost 比 quality 增长更快。

4-agent ceiling 是 empirical，不是 fundamental。它反映 2026 LLM context capacity：每个 agent 的 context 被 peers' outputs 填满，而且一旦所有人都能看到所有人，加入 agent N+1 的 marginal value 下降。

### Multi-Agent Debate Strategies（“Should we be going MAD?”）

arXiv:2311.17371 是 2023 年对 MAD strategies 的 survey。关键发现被其他工作复现：如果使用相同 budget，结构上类似 self-consistency 的 MAD variants（independent sampling + aggregation）经常不如 self-consistency。MAD 最有帮助的情况是 agents 真正 heterogeneous，且 debate 有 adversarial structure（一个 agent 负责反对）。

### AgentVerse emergent patterns

AgentVerse（ICLR 2024, https://proceedings.iclr.cc/paper_files/paper/2024/file/578e65cdee35d00c708d4c64bce32971-Paper-Conference.pdf）记录了两种即便没有显式设计也会从 multi-agent debate 中涌现的 behaviors：

- **Volunteer。** agent 主动提供帮助（“I can take the next step”）。有用：它把工作分配给对 subtask 最有能力的 agent。
- **Conformity。** agent 调整 stance 来匹配 critic，即使 critic 是错的。这是 sycophancy 的 debate 等价物（第 14 课）。

Conformity 是为什么 debate-until-agreement 会奖励 bullies。bounded rounds 加 separate judge 能缓解。

### Heterogeneity：真正推动 accuracy 的 knob

2024-2026 practical literature 中的一个 pattern：把 N 个 agents 中的一个换成不同 base model，带来的 accuracy bump 大于 N 增加 1。直觉是 monoculture — 每个新的 independent-error source 都比一个额外 correlated sample 更有价值。

极限情况下，heterogeneity 胜过 numerosity。对于多数有清晰 ground truth 的任务，三个不同 models 胜过一个 model 的五个 copies。

### Jury methods

Sibyl framework（Minsky-LLM literature 中引用）形式化了 “jury” — 一小组 specialized agents 在每个 stage 通过 voting refine answers。不同于 plain majority vote，jury 有 roles：一个 agent cross-examines，一个提供 context，一个给 plausibility score。Jury methods 是 plain vote（便宜但易 monoculture）和 full MAD（昂贵且易 conformity）之间的中点。

### Vote-with-debate 什么时候占优

- 问题有 ground truth（fact、math、code behavior）。vote convergence 有意义。
- Agents 可以访问不同 sources 或 tools（heterogeneity 可用）。
- Rounds bounded（通常 2-3），并有 separate judge 或 verifier。
- Budget 允许 3-5 agents。graph topology 超过 5-7 后，coordination tax 占主导。

### Vote-with-debate 什么时候有害

- 问题是 opinion-shaped。Agents 收敛到最自信的答案，而不是最正确的答案。
- 所有 agents 共享 base model。Monoculture 让 consensus 失去意义。
- Rounds 无界。Conformity 每次都会赢。
- 任务简单。一个 single agent 做 N=5 self-consistency 更便宜且同样准确。

## 构建它

`code/main.py` 实现：

- `run_star(agents, hub, question)` — hub poll 每个 worker 并 aggregate。
- `run_chain(agents, question)` — sequential refinement。
- `run_tree(root, children, question)` — depth-2 aggregation 的 hierarchical。
- `run_graph(agents, question, rounds)` — all-to-all debate，bounded rounds。
- scripted heterogeneity dial：每个 agent 有一个 `error_bias`，表示其 systematic wrongness。
- measurement harness：在 N=3、5、7 下运行每种 topology，并报告 (accuracy, total_tokens, wallclock_simulated)。

运行：

```
python3 code/main.py
```

预期输出：topology × N → (accuracy, tokens, latency) 的表。在 research-style tasks 上 graph 在 N=3-5 胜出；fast-factual tasks 上 star 胜出；N=7 的 graph 展示 coordination tax（latency 比 accuracy 更快膨胀）。

## 使用它

`outputs/skill-topology-picker.md` 是一个 skill：读取 task description 并推荐 topology（star / chain / tree / graph）、N（agent 数量）、heterogeneity profile（使用哪些 base models）和 round bound。

## 发布它

对任意 ensemble：

- 从 **self-consistency at N=5** 使用一个 strong base model 开始。这是便宜 baseline。
- 如果 accuracy 重要，升级到 **heterogeneous voting at N=3**。测量 delta。
- 只有当 task 有结构（research、multi-step）且 bounded rounds 可行时，才升级到 **debate topology**。
- 始终记录 minority cluster。当 minority 持续正确时，那是 diversity signal。
- 与 accuracy 一起 benchmark wall-clock 和 tokens。“10x cost 换更高 accuracy” 是 business decision。

## 练习

1. 运行 `code/main.py`。绘制 graph topology 的 coordination-tax curve：accuracy vs N，tokens vs N。曲线在哪个 N 拐弯？
2. 实现 A-HMAD：三个 agents 带刻意不同 biases。与 Lesson 14 monoculture attack 中的 all-same-bias baseline 相比如何？
3. 给 graph topology 添加一个 “judge” role，它不投票，只给 final consensus 打分。这会改变 emergent conformity behavior 吗？
4. 阅读 AgentVerse paper（ICLR 2024）。识别你的 implementation 表现最强的 emergent behavior。能否通过 prompt change eliciting opposite behavior？
5. 阅读 MultiAgentBench（arXiv:2503.01935）Section 4（topology experiments）。使用你的 harness 在论文的一个 task 上复现 “graph-wins-research” result。

## 关键词汇

| 术语 | 人们常说 | 实际含义 |
|------|----------------|------------------------|
| Self-consistency | “Sample N times, vote” | Wang 2022。单 model，N 个 temperature>0 samples，对 reasoning paths majority vote。 |
| Heterogeneity | “Different models” | 不同 base models 或 prompt families 的 ensemble。打破 monoculture。 |
| MAD | “Multi-agent debate” | agents 多轮交换 critiques 的 generic term。见 Du 2023。 |
| A-HMAD | “Adversarial Heterogeneous MAD” | 强调 different models + adversarial structure 的 MAD variant。 |
| Topology | “谁和谁说话” | Star、chain、tree、graph。决定 information flow。 |
| Coordination tax | “Diminishing returns” | graph 中超过 ~4 agents 后，cost 增长快于 quality。 |
| Volunteer behavior | “Unprompted help” | AgentVerse emergent pattern：agent 主动提出承担一步。 |
| Conformity behavior | “Agreement under pressure” | AgentVerse emergent pattern：agent 与 critic 对齐。 |
| Jury | “Small specialized panel” | Sibyl-style ensemble，带 roles（examiner、context、scorer）。 |

## 延伸阅读

- [Wang et al. — Self-Consistency Improves Chain of Thought Reasoning](https://arxiv.org/abs/2203.11171) — single-model baseline
- [Du et al. — Improving Factuality and Reasoning via Multiagent Debate](https://arxiv.org/abs/2305.14325) — agents 和 rounds 都独立重要
- [MultiAgentBench / MARBLE](https://arxiv.org/abs/2503.01935) — topology benchmark，展示 graph 最适合 research，chain 最适合 pipelines
- [Should we be going MAD?](https://arxiv.org/abs/2311.17371) — MAD-strategy survey；发现 equal budget 下 MAD 经常输给 self-consistency
- [AgentVerse (ICLR 2024)](https://proceedings.iclr.cc/paper_files/paper/2024/file/578e65cdee35d00c708d4c64bce32971-Paper-Conference.pdf) — volunteer 和 conformity emergent patterns
- [MARBLE repo](https://github.com/ulab-uiuc/MARBLE) — reference benchmark implementation
