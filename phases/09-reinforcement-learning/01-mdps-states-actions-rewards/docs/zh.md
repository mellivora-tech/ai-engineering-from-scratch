# MDP、States、Actions 与 Rewards

> Markov Decision Process 是五样东西：states、actions、transitions、rewards、discount。RL 中的一切 — Q-learning、PPO、DPO、GRPO — 都在这个形状上优化。学会一次，后面的 reinforcement learning 基本白送。

**类型：** 学习
**语言：** Python
**前置要求：** 阶段 1 · 06（Probability & Distributions），阶段 2 · 01（ML Taxonomy）
**时间：** ~45 分钟

## 问题

你在写 chess bot。或者 inventory planner。或者 trading agent。或者训练 reasoning model 的 PPO loop。四个不同领域，一个令人惊讶的事实：四者都会塌缩到同一个数学对象。

Supervised learning 给你 `(x, y)` 对，并要求你拟合函数。Reinforcement learning 不给 labels — 只给一串 states、你采取的 actions，以及一个标量 reward。这步棋赢了吗？补货决策省钱了吗？交易盈利了吗？LLM 刚生成的 token 是否让 judge 给出更高 reward？

在形式化它之前，你无法从这条 stream 学习。“我看到了什么”、“我做了什么”、“接下来发生了什么”、“这有多好” — 每个都必须变成你可以推理的对象。这个形式化就是 Markov Decision Process。本阶段每个 RL 算法，包括末尾的 RLHF 和 GRPO loops，都在这个形状上优化。

## 概念

![Markov decision process: states, actions, transitions, rewards, discount](../assets/mdp.svg)

**五个对象。**

- **States** `S`。Agent 做决定所需的一切。GridWorld 中是格子。Chess 中是棋盘。LLM 中是 context window 加任何 memory。
- **Actions** `A`。可选项。上/下/左/右。下一步棋。发出一个 token。
- **Transitions** `P(s' | s, a)`。给定 state `s` 和 action `a`，next state 的分布。Chess 中确定，inventory 中随机，LLM decoding 中几乎确定。
- **Rewards** `R(s, a, s')`。标量信号。赢 = +1，输 = -1。收入减成本。GRPO 中的 log-likelihood ratio 项。
- **Discount** `γ ∈ [0, 1)`。未来 reward 相比现在有多重要。`γ = 0.99` 买到约 100 步 horizon；`γ = 0.9` 约 10 步。

**Markov property** `P(s_{t+1} | s_t, a_t) = P(s_{t+1} | s_0, a_0, …, s_t, a_t)`。未来只依赖当前 state。如果不是这样，state representation 不完整 — 这不是方法失败，而是 state 失败。

**Policies and returns。** Policy `π(a | s)` 把 states 映射到 action distributions。Return `G_t = r_t + γ r_{t+1} + γ² r_{t+2} + …` 是未来 rewards 的折扣和。Value `V^π(s) = E[G_t | s_t = s]` 是从 `s` 出发、遵循 policy `π` 的期望 return。Q-value `Q^π(s, a) = E[G_t | s_t = s, a_t = a]` 是从特定 action 开始的期望 return。每个 RL 算法都会估计二者之一，然后相应改善 `π`。

**Bellman equations。** 本阶段一切都会用到的 fixed-point equations：

`V^π(s) = Σ_a π(a|s) Σ_{s', r} P(s', r | s, a) [r + γ V^π(s')]`
`Q^π(s, a) = Σ_{s', r} P(s', r | s, a) [r + γ Σ_{a'} π(a'|s') Q^π(s', a')]`

它们把 expected return 拆成“这一步 reward”加“落点的 discounted value”。递归。本阶段每个算法要么迭代这个方程直到收敛（dynamic programming），要么从它采样（Monte Carlo），要么 bootstrap 一步（temporal difference）。

## 构建它

### 第 1 步：一个 tiny deterministic MDP

一个 4×4 GridWorld。Agent 从左上角开始，terminal 在右下角，每步 reward -1，actions `{up, down, left, right}`。见 `code/main.py`。

```python
GRID = 4
TERMINAL = (3, 3)
ACTIONS = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}

def step(state, action):
    if state == TERMINAL:
        return state, 0.0, True
    dr, dc = ACTIONS[action]
    r, c = state
    nr = min(max(r + dr, 0), GRID - 1)
    nc = min(max(c + dc, 0), GRID - 1)
    return (nr, nc), -1.0, (nr, nc) == TERMINAL
```

五行。这就是整个 environment。Deterministic transitions、常量 step penalty、absorbing terminal state。

### 第 2 步：roll out 一个 policy

Policy 是从 state 到 action distribution 的函数。最简单：uniform random。

```python
def uniform_policy(state):
    return {a: 0.25 for a in ACTIONS}

def rollout(policy, max_steps=200):
    s, total, steps = (0, 0), 0.0, 0
    for _ in range(max_steps):
        a = sample(policy(s))
        s, r, done = step(s, a)
        total += r
        steps += 1
        if done:
            break
    return total, steps
```

运行 random policy 1000 次。对这个 4×4 board，平均 return 约 -60 到 -80。Optimal return 是 -6（沿直线路径 down-right）。缩小这个差距就是阶段 9 的全部内容。

### 第 3 步：通过 Bellman equation 精确计算 `V^π`

对小 MDP，Bellman equation 是线性系统。枚举 states，应用 expectation，迭代直到 values 不再变化。

```python
def policy_evaluation(policy, gamma=0.99, tol=1e-6):
    V = {s: 0.0 for s in all_states()}
    while True:
        delta = 0.0
        for s in all_states():
            if s == TERMINAL:
                continue
            v = 0.0
            for a, pi_a in policy(s).items():
                s_next, r, _ = step(s, a)
                v += pi_a * (r + gamma * V[s_next])
            delta = max(delta, abs(v - V[s]))
            V[s] = v
        if delta < tol:
            return V
```

这就是 iterative policy evaluation。它是 Sutton & Barto 的第一个算法，也是后续每个 RL 方法的理论基础。

### 第 4 步：`γ` 是有物理意义的超参数

Effective horizon 大约是 `1 / (1 - γ)`。`γ = 0.9` → 10 步。`γ = 0.99` → 100 步。`γ = 0.999` → 1000 步。

太低，agent 会短视。太高，credit assignment 会很吵，因为许多早期步骤共同承担远期 reward 的责任。LLM RLHF 通常使用 `γ = 1`，因为 episodes 短且有界。Control tasks 用 `0.95–0.99`。Long-horizon strategy games 用 `0.999`。

## 陷阱

- **Non-Markovian state。** 如果你需要最近三次 observations 来做决定，那么 “state” 不只是当前 observation。修复：stack frames（Atari 上 DQN stack 4 帧）或使用 recurrent state（对 observations 做 LSTM/GRU）。
- **Sparse rewards。** 在大 state spaces 中，只有胜负奖励会让学习几乎不可能。Shape rewards（中间信号）或用 imitation bootstrap（阶段 9 · 09）。
- **Reward hacking。** 优化 proxy reward 经常产生病态行为。OpenAI 的 boat-racing agent 不去终点，而是原地转圈永远收集 powerups。始终从目标 outcome 定义 reward，而不是 proxy。
- **Discount mis-spec。** 在 infinite-horizon task 上 `γ = 1` 会让每个 value 无限大。要么用 finite horizon，要么 `γ < 1`。
- **Reward scale。** {+100, -100} 和 {+1, -1} 给出相同 optimal policies，但 gradient magnitudes 完全不同。接入 PPO/DQN 前 normalize 到类似 `[-1, 1]`。

## 使用它

2026 年 stack 会先把每个 RL pipeline 化成 MDP，再写代码：

| 场景 | State | Action | Reward | γ |
|-----------|-------|--------|--------|---|
| Control（locomotion、manipulation） | Joint angles + velocities | Continuous torques | Task-specific shaped | 0.99 |
| Games（chess、Go、poker） | Board + history | Legal move | Win=+1 / loss=-1 | 1.0（finite） |
| Inventory / pricing | Stock + demand | Order qty | Revenue - cost | 0.95 |
| RLHF for LLMs | Context tokens | Next token | Episode 末 reward-model score | 1.0（episode ~200 tokens） |
| GRPO for reasoning | Prompt + partial response | Next token | Verifier 0/1 at end | 1.0 |

写任何 training loop 之前，先写这五元组。大多数 “RL does not work” bug reports 最后都能追溯到纸面上就坏掉的 MDP formulation。

## 交付它

保存为 `outputs/skill-mdp-modeler.md`：

```markdown
---
name: mdp-modeler
description: Given a task description, produce a Markov Decision Process spec and flag formulation risks before training.
version: 1.0.0
phase: 9
lesson: 1
tags: [rl, mdp, modeling]
---

Given a task (control / game / recommendation / LLM fine-tuning), output:

1. State. Exact feature vector or tensor spec. Justify Markov property.
2. Action. Discrete set or continuous range. Dimensionality.
3. Transition. Deterministic, stochastic-with-known-model, or sample-only.
4. Reward. Function and source. Sparse vs shaped. Terminal vs per-step.
5. Discount. Value and horizon justification.

Refuse to ship any MDP where the state is non-Markovian without explicit mention of frame-stacking or recurrent state. Refuse any reward that was not defined in terms of the target outcome. Flag any `γ ≥ 1.0` on an infinite-horizon task. Flag any reward range >100x the typical step reward as a likely gradient-explosion source.
```

## 练习

1. **简单。** 在 `code/main.py` 中实现 4×4 GridWorld 和 random-policy rollout。运行 10,000 episodes。报告 return 的 mean 和 std。与 optimal return（-6）比较。
2. **中等。** 对 uniform-random policy，用 `γ ∈ {0.5, 0.9, 0.99}` 运行 `policy_evaluation`。把每个 `V` 打印成 4×4 grid。解释为什么 terminal 附近 state values 会随更大 `γ` 增长更快。
3. **困难。** 把 GridWorld 变成 stochastic：每个 action 以 `p = 0.1` 概率 slip 到相邻方向。重新评估 uniform policy。`V[start]` 变好还是变差？为什么？

## 关键词

| 术语 | 大家怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| MDP | “Reinforcement learning setup” | 满足 Markov property 的 tuple `(S, A, P, R, γ)`。 |
| State | “Agent 看到什么” | 对 chosen policy class 下的未来 dynamics 足够的统计量。 |
| Policy | “Agent 的行为” | Conditional distribution `π(a | s)` 或 deterministic map `s → a`。 |
| Return | “总 reward” | 从当前步骤开始的 discounted sum `Σ γ^t r_t`。 |
| Value | “一个 state 有多好” | 在 `π` 下从 `s` 出发的 expected return。 |
| Q-value | “一个 action 有多好” | 在 `π` 下从 `s` 出发并先采取 action `a` 的 expected return。 |
| Bellman equation | “Dynamic programming recursion” | 把 value / Q 分解成 one-step reward 加 discounted successor value 的 fixed-point。 |
| Discount `γ` | “未来 vs 现在” | 远期 reward 的几何权重；effective horizon `~1/(1-γ)`。 |

## 延伸阅读

- [Sutton & Barto (2018). Reinforcement Learning: An Introduction, 2nd ed.](http://incompleteideas.net/book/RLbook2020.pdf) — 教科书。第 3 章覆盖 MDPs 和 Bellman equations；第 1 章引出 reward hypothesis。
- [Bellman (1957). Dynamic Programming](https://press.princeton.edu/books/paperback/9780691146683/dynamic-programming) — Bellman equation 的源头。
- [OpenAI Spinning Up — Part 1: Key Concepts](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html) — 从 deep-RL 角度写的简洁 MDP primer。
- [Puterman (2005). Markov Decision Processes](https://onlinelibrary.wiley.com/doi/book/10.1002/9780470316887) — MDPs 和 exact solution methods 的 operations-research 参考。
- [Littman (1996). Algorithms for Sequential Decision Making (PhD thesis)](https://www.cs.rutgers.edu/~mlittman/papers/thesis-main.pdf) — 将 MDPs 作为 dynamic-programming specialization 的清晰推导。
