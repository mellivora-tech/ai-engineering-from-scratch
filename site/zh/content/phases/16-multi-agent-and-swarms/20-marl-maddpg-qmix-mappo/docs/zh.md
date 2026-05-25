# MARL — MADDPG、QMIX、MAPPO

> multi-agent coordination 的 reinforcement-learning 传承，在 2026 年仍然影响 LLM-agent systems。**MADDPG**（Lowe et al., NeurIPS 2017, arXiv:1706.02275）引入 Centralized Training, Decentralized Execution（CTDE）：每个 critic 在 training 时看到所有 agents 的 states 和 actions；test time 只运行 local actors。适用于 cooperative、competitive 和 mixed settings。**QMIX**（Rashid et al., ICML 2018, arXiv:1803.11485）是 value-decomposition，使用 monotonic mixing network；per-agent Qs 组合成 joint Q，因此 `argmax` 可以干净地分布式执行 — 在 StarCraft Multi-Agent Challenge（SMAC）上占主导。**MAPPO**（Yu et al., NeurIPS 2022, arXiv:2103.01955）是带 centralized value function 的 PPO；在 particle-world、SMAC、Google Research Football、Hanabi 上用 minimal tuning “surprisingly effective”。这些支撑了必须 decentrally act 的 agent teams 的 policy training。MAPPO 是 **2026 cooperative-MARL default baseline**。本课用一个小 grid-world toy 构建三者，在接触 LLM-agent training 前把三个 ideas 变成肌肉记忆。

**类型：** 学习
**语言：** Python（stdlib，小型 NumPy-free implementations）
**前置要求：** 阶段 09（Reinforcement Learning），阶段 16 · 09（Parallel Swarm Networks）
**时间：** ~90 分钟

## 问题

LLM-agent systems 越来越多地训练 inter-agent coordination 的 policies：何时 defer、何时 act、call 哪个 peer。告诉你如何训练这类 policies 的 literature 是 Multi-Agent Reinforcement Learning（MARL），它早于 LLM wave，并有一小组 dominant algorithms。

没有 pattern vocabulary 直接读 MARL papers 会很痛苦。Centralized training with decentralized execution（CTDE）、value decomposition 和 centralized critics 不是 buzzwords — 它们是对具体问题的具体答案：

- Independent RL（每个 agent 单独学习）从每个 agent 视角看是 non-stationary。不好。
- Centralized RL（一个 agent 控制全部）不可扩展，并且违反 execution constraints。
- CTDE 兼得二者：用 global information 训练，用 local policies 部署。

## 概念

### 论文使用的三类 environments

- **Particle World（multi-agent particle env）。** 带 cooperative/competitive tasks 的简单 2D physics。MADDPG 原始 testbed。
- **StarCraft Multi-Agent Challenge（SMAC）。** cooperative micro-management，partial observation。QMIX testbed。Discrete actions，continuous states。
- **Google Research Football、Hanabi、MPE。** MAPPO baselines。

不同 envs 有不同 action/observation types。algorithms 也据此选择。

### MADDPG（2017）— CTDE pattern

每个 agent `i` 有一个 actor `mu_i(o_i)`，把自己的 observation 映射到 action。每个 agent 还有一个 critic `Q_i(x, a_1, ..., a_n)`，training 时看到所有 observations 和所有 actions。actor 通过 policy gradient 根据 critic 的 evaluation 更新。

```
actor update:    grad_theta_i J = E[grad_theta mu_i(o_i) * grad_a_i Q_i(x, a_1..n) at a_i=mu_i(o_i)]
critic update:   TD on Q_i(x, a_1..n) given next-state joint estimate
```

为什么 CTDE：training time 我们知道每个人的 actions；用它降低每个 critic 的 variance。deploy time 每个 agent 只看到 `o_i` 并调用 `mu_i(o_i)`。

Failure mode：critics 随 N agents 增长（input 包含所有 actions）。没有 approximations 时很难超过 ~10 agents。

### QMIX（2018）— value decomposition

仅 cooperative。global reward 是 per-agent Q-values 的 monotone function：

```
Q_tot(tau, a) = f(Q_1(tau_1, a_1), ..., Q_n(tau_n, a_n)),   df/dQ_i >= 0
```

monotonicity 保证 `argmax_a Q_tot` 可由每个 agent 独立选择 `argmax_{a_i} Q_i` 计算。这正是你需要的 **decentralized execution property**。training time，mixing network 从 per-agent Qs 生成 `Q_tot`。

为什么 QMIX 在 SMAC 上胜出：cooperative StarCraft micro-management 有 homogeneous agents、local obs、global reward — 完美适合 value decomposition。

Failure mode：monotonicity constraint 很 restrictive；一些 tasks 的 reward structures 不是 monotone decomposable（一个 agent 为团队牺牲）。extensions（QTRAN、QPLEX）放松了它。

### MAPPO（2022）— 被低估的默认值

Multi-Agent PPO：带 centralized value function 的 PPO。每个 agent 有自己的 policy；所有 agents 共享（或拥有 per-agent）能看到 full state 的 value functions。Yu et al. 2022 在五个 benchmarks 上将 MAPPO 与 MADDPG、QMIX 及其 extensions 比较，发现：

- MAPPO 在 particle-world、SMAC、Google Research Football、Hanabi、MPE 上匹配或超过 off-policy MARL methods。
- 只需 minimal hyperparameter tuning。
- training 稳定；跨 seeds 可复现。

在这篇论文前，community 低估了 on-policy MARL。2026 年，MAPPO 是 cooperative MARL 的 default baseline；任何新 method 都必须打败它。

### 为什么 LLM-agent engineers 应关心

三个直接用途：

1. **Router training。** meta-agent 选择哪个 sub-agent 处理 task。这是一个 N 个 decentralized sub-agents 加一个 centralized router 的 MARL problem。MAPPO 适合。
2. **Role emergence。** 在 generative-agent simulations 中，训练 agents 随时间采用 complementary roles 是伪装的 MARL problem。QMIX-style value decomposition 通过构造强制 complementarity。
3. **Multi-agent tool use。** agents 共享 tools 并竞争 budget 时，通过 CTDE 训练能产出 respecting resource constraints 的 deployable local policies。

实用 caveat：2026 年多数 production LLM-agent systems 是 prompt policies，而不是训练 policies。MARL 在这些条件同时满足时才进场：(a) 大量 interaction data，(b) 清晰 reward signal，(c) 愿意投入 training infrastructure。

### CTDE 作为 RL 之外的 design pattern

即使不训练，CTDE 也是有用的 architecture pattern：

- 在 *design* 阶段，假设 full team visibility。
- 在 *runtime* 阶段，强制 decentralized execution：每个 agent 只看到 `o_i`。

这个 pattern 迫使你显式维护 per-agent state，并提前考虑 partial observability。许多 production multi-agent systems 默默假设 everywhere shared state — CTDE discipline 能防止这一点。

### Non-stationarity problem

当多个 agents 同时学习时，每个 agent 的 environment（包含其他 agents 的 policies）都是 non-stationary。classical single-agent RL proofs 会失效。本课的 MARL algorithms 都在处理这一点：

- MADDPG：global critic 看到所有 actions，因此 value estimate 更 stationary。
- QMIX：value decomposition 把 learning 移到 joint-Q space，其中 optimality 定义清楚。
- MAPPO：centralized value function 抑制来自其他 policy changes 的 variance。

在 LLM-agent systems 中，non-stationarity 表现为“我的 agent 上个月正常，现在 upstream 另一个 agent 变了，我的就 misbehaves”。用 CTDE 训练 MARL 是 principled fix；prompt-level fixes 更快但更不 durable。

### 本课不覆盖什么

训练真实 networks 是阶段 09 的主题。本课构建 scripted-policy versions，展示 CTDE、value-decomposition 和 centralized-value patterns，而不做 gradient updates。目标是在使用 full MARL library（PyMARL、MARLlib、RLlib multi-agent）前先内化 patterns。

## 构建它

`code/main.py` 在 tiny 2-agent cooperative grid-world 上实现三种 pattern demonstrations：

- Environment：2 agents 在 4x4 grid 上，一个 reward pellet。Reward = 1 if any agent reaches pellet；task finishes。
- `IndependentAgents` — 每个 agent 把 others 当 environment。baseline。
- `MADDPGStyle` — centralized critic 计算 joint value；actor policies 从中更新。scripted policy improvement。
- `QMIXStyle` — 带 monotone mixer 的 value decomposition。
- `MAPPOStyle` — centralized value function；policies 根据 shared baseline 更新。

四者运行同样 episodes，并报告 average steps-to-goal。CTDE variants 比 independent baseline 收敛到更短 paths。

运行：

```
python3 code/main.py
```

预期输出：independent agents 平均约 6 steps；CTDE variants 收敛到约 3.5 steps（4x4 grid 最优是 3）。即使是 scripted policies，pattern difference 也会显现。

## 使用它

`outputs/skill-marl-picker.md` 是一个 skill，用来为给定 multi-agent task 选择 MARL algorithm：cooperative vs competitive、homogeneous vs heterogeneous、action-space type、scale、reward signal。

## 发布它

MARL 在 production 中少见。当你使用它时：

- **从 MAPPO 开始。** 2022 论文确立它是 baseline；先复现它能省下追逐 fancy methods 的数周时间。
- **记录每个 agent 的 observation 和 action stream。** 没有 per-agent traces，debug MARL 几乎无望。
- **分离 training code 与 execution code。** CTDE 是一种 discipline；execution path 真的只能看到 `o_i`。
- **Reward shaping warning。** MARL 对 reward design 极其敏感。shaping 中一个 coordination bug，agents 就会学会 exploit 它。运行 adversarial tests。
- **对 LLM agents，先考虑 prompt-level policies。** 只有 interaction data + reward signal + infrastructure 都存在时，再投入 MARL training。

## 练习

1. 运行 `code/main.py`。测量 independent 与 MAPPO-style agents 的 steps-to-goal gap。在 6x6 grid 上 gap 变大还是变小？
2. 实现 competitive variant：两个 agents，一个 pellet，只有第一个到达者得 reward。哪个 pattern 干净处理 competition？历史上是 MADDPG。
3. 阅读 MADDPG（arXiv:1706.02275）Section 3。用自己的话以 pseudocode 形式实现 exact critic update rule。
4. 阅读 MAPPO（arXiv:2103.01955）。为什么作者认为 centralized value + PPO 在其 benchmarks 上击败 off-policy MARL？列出三个最强 claims。
5. 把 CTDE 作为 design pattern 应用于假设的 LLM-agent system（例如 research agent + summarizer + coder）。design time 可用但 runtime 不可用的 joint information 是什么？

## 关键词汇

| 术语 | 人们常说 | 实际含义 |
|------|----------------|------------------------|
| MARL | “Multi-Agent RL” | multi-agent systems 的 reinforcement learning。 |
| CTDE | “Centralized Training, Decentralized Execution” | 用 global info 训练；用 local policies 部署。 |
| MADDPG | “Multi-Agent DDPG” | CTDE，per-agent critic 看到所有 observations + actions。 |
| QMIX | “Value decomposition” | per-agent Qs 的 monotonic mixing。Cooperative。 |
| MAPPO | “Multi-Agent PPO” | 带 centralized value function 的 PPO。2026 default baseline。 |
| Value decomposition | “individual Qs 的和” | joint Q 表示为 per-agent Qs 的 monotone function。 |
| Non-stationarity | “moving targets” | 随着 others 学习，每个 agent 的 env 都在变。核心 MARL problem。 |
| On-policy / off-policy | “从当前 / replay 学习” | PPO 是 on-policy（MAPPO）；DDPG 和 Q-learning 是 off-policy。 |
| SMAC | “StarCraft Multi-Agent Challenge” | cooperative micromanagement benchmark；QMIX 的主场。 |

## 延伸阅读

- [Lowe et al. — Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments](https://arxiv.org/abs/1706.02275) — MADDPG；NeurIPS 2017
- [Rashid et al. — QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning](https://arxiv.org/abs/1803.11485) — QMIX；ICML 2018
- [Yu et al. — The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games](https://arxiv.org/abs/2103.01955) — MAPPO；NeurIPS 2022
- [BAIR blog post on MAPPO](https://bair.berkeley.edu/blog/2021/07/14/mappo/) — MAPPO result 的可读 framing
- [SMAC repository](https://github.com/oxwhirl/smac) — StarCraft Multi-Agent Challenge
