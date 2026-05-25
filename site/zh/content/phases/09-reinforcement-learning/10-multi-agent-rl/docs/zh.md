# Multi-Agent RL

> Single-agent RL 假设环境是 stationary 的。把两个正在学习的 agent 放进同一个世界，这个假设就坏了：每个 agent 都是另一个 agent 环境的一部分，而且二者都在变化。Multi-agent RL 是一组技巧，用来在 Markov 假设不再成立时仍让学习收敛。

**类型：** 构建
**语言：** Python
**先修：** Phase 9 · 04（Q-learning），Phase 9 · 06（REINFORCE），Phase 9 · 07（Actor-Critic）
**时间：** 约 45 分钟

## 问题

一个学习在房间中导航的机器人是 single-agent RL 问题。足球队不是。AlphaStar 对 StarCraft 对手不是。竞价 agent 组成的市场不是。两辆车协商四向停车不是。现实世界中许多 many-on-many 问题都不是。

在每个 multi-agent 设置中，从任意单个 agent 的视角看，其他 agent *就是*环境的一部分。随着它们学习并改变行为，环境变得 non-stationary。Markov property，也就是“下一个 state 只依赖当前 state 和我的 action”，被破坏了，因为下一个 state 还依赖*其他* agent 选择了什么，而它们的 policy 是移动目标。

这会破坏表格型收敛证明（Q-learning 的保证假设 stationary environment）。它也会破坏朴素 deep RL：agent 互相追逐，陷入循环，永远无法收敛到稳定 policy。你需要 multi-agent-specific 技术：centralized training / decentralized execution、counterfactual baselines、league play、self-play。

2026 年应用：机器人群、交通路由、自动驾驶车队、市场模拟器、multi-agent LLM systems（Phase 16），以及任何有多个智能玩家的游戏。

## 核心概念

![Four MARL regimes: indep, centralized critic, self-play, league](../assets/marl.svg)

**形式化：Markov Game。** MDP 的推广：states `S`、joint action `a = (a_1, …, a_n)`、transition `P(s' | s, a)`，以及每个 agent 的 reward `R_i(s, a, s')`。每个 agent `i` 都在自己的 policy `π_i` 下最大化自己的 return。如果 reward 相同，就是 **fully cooperative**。如果是 zero-sum，就是 **adversarial**。如果混合，就是 **general-sum**。

**核心挑战：**

- **Non-stationarity。** 从 agent `i` 的视角看，`P(s' | s, a_i)` 依赖正在变化的 `π_{-i}`。
- **Credit assignment。** 当 reward 共享时，是哪个 agent 造成了它？
- **Exploration coordination。** Agents 必须探索互补策略，而不是重复探索同一个 state。
- **Scalability。** Joint action space 随 `n` 指数增长。
- **Partial observability。** 每个 agent 只看到自己的 observation；global state 是隐藏的。

**四种主流范式：**

**1. Independent Q-learning / independent PPO（IQL，IPPO）。** 每个 agent 学自己的 Q 或 policy，把其他 agent 当作环境的一部分。简单，有时可行（尤其是 experience replay 充当平滑的 agent-modeling 技巧时）。理论收敛性：无。实践中：对松耦合任务可以，对紧耦合任务很差。

**2. Centralized training, decentralized execution（CTDE）。** 最常见的现代范式。每个 agent 有自己的 *policy* `π_i`，以本地 observation `o_i` 为条件，在部署时标准地 decentralized execution。*训练*期间，centralized critic `Q(s, a_1, …, a_n)` 以完整 global state 和 joint action 为条件。例如：
- **MADDPG**（Lowe et al. 2017）：每个 agent 一个 centralized critic 的 DDPG。
- **COMA**（Foerster et al. 2017）：counterfactual baseline：问“如果我采取动作 `a'`，我的 reward 会是多少？”——隔离我的贡献。
- **MAPPO** / **IPPO** with shared critic（Yu et al. 2022）：带 centralized value function 的 PPO。2026 年 cooperative MARL 主流。
- **QMIX**（Rashid et al. 2018）：value decomposition：`Q_tot(s, a) = f(Q_1(s, a_1), …, Q_n(s, a_n))`，并使用 monotonic mixing。

**3. Self-play。** 同一个 agent 的两个副本彼此对战。对手 policy 是我过去某个 snapshot 中的 policy。AlphaGo / AlphaZero / MuZero。OpenAI Five。最适合 zero-sum games；训练信号是对称的。

**4. League play。** Self-play 扩展到 general-sum / adversarial 环境：保留一群过去和当前的 policies，从 league 中采样对手并训练。加入 exploiters（专门击败当前最佳）和 main exploiters（专门击败 exploiters）。AlphaStar（StarCraft II）。当游戏存在“石头剪刀布”策略循环时，这是必要的。

**Communication。** 允许 agent 彼此发送 learned messages `m_i`。适合 cooperative settings。Foerster 等人（2016）证明可微的 inter-agent communication 可以端到端训练。今天基于 LLM 的 multi-agent systems（Phase 16）本质上是在自然语言中通信。

## 动手构建

本课使用一个 6×6 GridWorld，里面有两个 cooperative agents。它们从相对角落出发，必须到达共同目标。共享 reward：任一 agent 还在移动时每步 `-1`，两者都到达时 `+10`。见 `code/main.py`。

### 第 1 步：multi-agent env

```python
class CoopGridWorld:
    def __init__(self):
        self.size = 6
        self.goal = (5, 5)

    def reset(self):
        return ((0, 0), (5, 0))  # two agents

    def step(self, state, actions):
        a1, a2 = state
        new1 = move(a1, actions[0])
        new2 = move(a2, actions[1])
        done = (new1 == self.goal) and (new2 == self.goal)
        reward = 10.0 if done else -1.0
        return (new1, new2), reward, done
```

*Joint* action space 是 `|A|² = 16`。Global state 是两个位置。

### 第 2 步：independent Q-learning

每个 agent 运行自己的 Q-table，键是 joint state。每一步：二者都选择 ε-greedy 动作，收集 joint transition，然后各自用共享 reward 更新自己的 Q。

```python
def independent_q(env, episodes, alpha, gamma, epsilon):
    Q1, Q2 = defaultdict(default_q), defaultdict(default_q)
    for _ in range(episodes):
        s = env.reset()
        while not done:
            a1 = epsilon_greedy(Q1, s, epsilon)
            a2 = epsilon_greedy(Q2, s, epsilon)
            s_next, r, done = env.step(s, (a1, a2))
            target1 = r + gamma * max(Q1[s_next].values())
            target2 = r + gamma * max(Q2[s_next].values())
            Q1[s][a1] += alpha * (target1 - Q1[s][a1])
            Q2[s][a2] += alpha * (target2 - Q2[s][a2])
            s = s_next
```

它在这个任务上可行，因为 rewards dense 且 aligned。但在紧耦合任务上会失败（例如一个 agent 必须*等待*另一个）。

### 第 3 步：centralized Q with decomposed-value update

使用一个覆盖 joint actions 的 Q：`Q(s, a_1, a_2)`。从共享 reward 更新。执行时通过边缘化 decentralize：`π_i(s) = argmax_{a_i} max_{a_{-i}} Q(s, a_1, a_2)`。它用指数级 joint action space 换取*正确*的 global view。

### 第 4 步：simple self-play（adversarial 2-agent）

同一个 agent，两个角色。训练 agent A 对抗 agent B；每隔 `K` 个 episode，把 A 的权重复制到 B。对称训练，进展一致。微型 AlphaZero 配方。

## 常见坑

- **Non-stationary replay。** Independent agents 的 experience replay 比 single-agent 更糟，因为旧 transitions 是由现在已过时的对手生成的。修复：按 recency 重新标记或加权。
- **Credit assignment ambiguity。** 长 episode 后得到共享 reward；没有清晰方式说明哪个 agent 有贡献。修复：counterfactual baselines（COMA），或每个 agent 的 reward shaping。
- **Policy drift / chasing。** 每个 agent 的 best response 都随着另一个 agent 的更新而改变。修复：centralized critic、较慢 learning rate，或一次冻结一个。
- **通过协调 reward hacking。** Agents 找到设计者没有预料到的协同漏洞。拍卖 agent 收敛到出价为零。修复：谨慎 reward design、行为约束。
- **Exploration redundancy。** 两个 agent 探索相同 state-action 对。修复：每个 agent 的 entropy bonus，或 role-conditioning。
- **League cycles。** 纯 self-play 可能卡在 dominance cycle 中。修复：带多样对手的 league play。
- **Sample explosion。** `n` 个 agents × state space × joint actions。用函数近似；factored action spaces（每个 agent 一个 policy output head）。

## 使用它

2026 年 MARL 应用图谱：

| 领域 | 方法 | 备注 |
|------|------|------|
| Cooperative navigation / manipulation | MAPPO / QMIX | CTDE；shared critic + decentralized actors。 |
| Two-player games（chess, Go, poker） | Self-play with MCTS（AlphaZero） | Zero-sum；symmetric training。 |
| Complex multiplayer（Dota, StarCraft） | League play + imitation pretraining | OpenAI Five，AlphaStar。 |
| Autonomous-vehicle fleets | CTDE MAPPO / PPO with attention | Partial obs；variable team sizes。 |
| Auction markets | Game-theoretic equilibrium + RL | `n` → ∞ 时用 mean-field RL。 |
| LLM multi-agent systems（Phase 16） | Natural-language comm + role conditioning | RL loop 位于 agent-planning layer。 |

到 2026 年，MARL 增长最快的领域是 LLM-based：由语言模型 agent 组成的群体协商、辩论、构建软件。这里的 RL 出现在 *trajectory-level* 输出的 preference optimization 上，而不是 token-level（Phase 16 · 03）。

## 交付

保存为 `outputs/skill-marl-architect.md`：

```markdown
---
name: marl-architect
description: Pick the right multi-agent RL regime (IPPO, CTDE, self-play, league) for a given task.
version: 1.0.0
phase: 9
lesson: 10
tags: [rl, multi-agent, marl, self-play]
---

Given a task with `n` agents, output:

1. Regime classification. Cooperative / adversarial / general-sum. Justify.
2. Algorithm. IPPO / MAPPO / QMIX / self-play / league. Reason tied to coupling tightness and reward structure.
3. Information access. Centralized training (what global info goes to the critic)? Decentralized execution?
4. Credit assignment. Counterfactual baseline, value decomposition, or reward shaping.
5. Exploration plan. Per-agent entropy, population-based training, or league.

Refuse independent Q-learning on tightly-coupled cooperative tasks. Refuse to recommend self-play for general-sum with cycle risks. Flag any MARL pipeline without a fixed-opponent eval (cherry-picked self-play numbers are common).
```

## 练习

1. **简单。** 在 2-agent cooperative GridWorld 上训练 independent Q-learning。Mean return > 0 需要多少个 episode？绘制 joint learning curve。
2. **中等。** 添加一个“coordination”任务：只有两个 agent 在同一回合踏上目标时才算到达。Independent Q 还会收敛吗？哪里坏了？
3. **困难。** 为 MAPPO-style training 实现 centralized critic，并在 coordination task 上与 independent PPO 比较收敛速度。

## 关键术语

| 术语 | 人们通常怎么说 | 它实际的含义 |
|------|----------------|--------------|
| Markov game | “Multi-agent MDP” | `(S, A_1, …, A_n, P, R_1, …, R_n)`；每个 agent 有自己的 reward。 |
| CTDE | “Centralized training, decentralized execution” | 训练时 joint critic；每个 agent 的 policy 只使用 local obs。 |
| IPPO | “Independent PPO” | 每个 agent 单独运行 PPO。简单 baseline；常被低估。 |
| MAPPO | “Multi-agent PPO” | 带 centralized value function、以 global state 为条件的 PPO。 |
| QMIX | “Monotonic value decomposition” | `Q_tot = f_monotone(Q_1, …, Q_n)` 允许 decentralized argmax。 |
| COMA | “Counterfactual multi-agent” | Advantage = 我的 Q 减去对我的动作做边缘化后的 expected Q。 |
| Self-play | “Agent vs past self” | 单个 agent，两个角色；zero-sum games 的标准做法。 |
| League play | “Population training” | 缓存过去 policies，从池中采样对手；处理策略循环。 |

## 延伸阅读

- [Lowe et al. (2017). Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments (MADDPG)](https://arxiv.org/abs/1706.02275) — 带 centralized critic 的 CTDE。
- [Foerster et al. (2017). Counterfactual Multi-Agent Policy Gradients (COMA)](https://arxiv.org/abs/1705.08926) — credit assignment 的 counterfactual baselines。
- [Rashid et al. (2018). QMIX: Monotonic Value Function Factorisation](https://arxiv.org/abs/1803.11485) — 带 monotonicity 的 value decomposition。
- [Yu et al. (2022). The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games (MAPPO)](https://arxiv.org/abs/2103.01955) — PPO 在 MARL 中出人意料地强。
- [Vinyals et al. (2019). Grandmaster level in StarCraft II using multi-agent reinforcement learning (AlphaStar)](https://www.nature.com/articles/s41586-019-1724-z) — 大规模 league play。
- [Silver et al. (2017). Mastering the game of Go without human knowledge (AlphaGo Zero)](https://www.nature.com/articles/nature24270) — zero-sum game 中的纯 self-play。
- [Sutton & Barto (2018). Ch. 15 — Neuroscience & Ch. 17 — Frontiers](http://incompleteideas.net/book/RLbook2020.pdf) — 包含教科书对 multi-agent setting 和 non-stationarity 问题的简短处理；CTDE 正是为解决这个问题而设计。
- [Zhang, Yang & Başar (2021). Multi-Agent Reinforcement Learning: A Selective Overview](https://arxiv.org/abs/1911.10635) — 覆盖 cooperative、competitive 和 mixed MARL 以及收敛结果的综述。
