# Temporal Difference — Q-Learning 与 SARSA

> Monte Carlo 等到 episode 结束。TD 通过 bootstrap 下一个 value estimate，在每一步之后更新。Q-learning 是 off-policy 且 optimistic；SARSA 是 on-policy 且 cautious。两者都只是一行代码。两者也支撑了本阶段的每个 deep-RL 方法。

**类型：** 构建
**语言：** Python
**前置要求：** 阶段 9 · 01（MDPs），阶段 9 · 02（Dynamic Programming），阶段 9 · 03（Monte Carlo）
**时间：** ~75 分钟

## 问题

Monte Carlo 有效，但有两个昂贵要求。它需要 episodes 终止，并且只有在最终 return 到手后才更新。如果 episode 有 1,000 步，MC 要等 1,000 步才能更新任何东西。它 high-variance、low-bias，实践中慢。

Dynamic programming 的画像相反 — zero-variance bootstrapped backups — 但需要已知 model。

Temporal difference（TD）learning 折中。从单个 transition `(s, a, r, s')` 中形成 one-step target `r + γ V(s')`，并把 `V(s)` 向它轻推。不需要 model。不需要完整 episodes。RHS 上使用近似 `V` 会带来 bias，但 variance 显著低于 MC，而且从第一步开始在线更新。

这是现代 RL — DQN、A2C、PPO、SAC — 旋转的枢轴。阶段 9 剩下内容都是在你本课要写的一步 TD update 上叠加 function approximation 和技巧。

## 概念

![Q-learning vs SARSA: off-policy max vs on-policy Q(s', a')](../assets/td.svg)

**V 的 TD(0) update：**

`V(s) ← V(s) + α [r + γ V(s') - V(s)]`

括号内是 TD error `δ = r + γ V(s') - V(s)`。它是 MC 中 `G_t - V(s_t)` 的在线 analogue。收敛要求 `α` 满足 Robbins-Monro（`Σ α = ∞`，`Σ α² < ∞`），且所有 states 被无限次访问。

**Q-learning。** 用于 control 的 off-policy TD method：

`Q(s, a) ← Q(s, a) + α [r + γ max_{a'} Q(s', a') - Q(s, a)]`

`max` 假设从 `s'` 开始会遵循 *greedy* policy，不管 agent 实际采取什么 action。这种解耦让 Q-learning 在 agent 通过 ε-greedy 探索时学习 `Q*`。Mnih et al.（2015）把它转换成 Atari 上的 deep Q-learning（第 05 课）。

**SARSA。** On-policy TD method：

`Q(s, a) ← Q(s, a) + α [r + γ Q(s', a') - Q(s, a)]`

名字来自 tuple `(s, a, r, s', a')`。SARSA 使用 agent *实际* 下一步采取的 action `a'`，不是 greedy `argmax`。它收敛到当前 ε-greedy `π` 的 `Q^π`，在 `ε → 0` 的极限中变成 `Q*`。

**Cliff-walking 差异。** 在经典 cliff-walking 任务中（掉下 cliff = reward -100），Q-learning 学到沿 cliff edge 的最优路径，但探索时偶尔会吃 penalty。SARSA 学到离 cliff 一步的安全路径，因为它把 exploration noise 计入 Q-value。随着训练，当 `ε → 0` 时两者都达到最优。实践中这很重要：如果部署时仍有 exploration，SARSA 的行为更保守。

**Expected SARSA。** 用 `π` 下的期望值替换 `Q(s', a')`：

`Q(s, a) ← Q(s, a) + α [r + γ Σ_{a'} π(a'|s') Q(s', a') - Q(s, a)]`

比 SARSA variance 更低（不 sample `a'`），同样是 on-policy target。现代教材中经常作为默认。

**n-step TD 和 TD(λ)。** 通过等待 `n` 步再 bootstrap，在 TD(0) 和 MC 之间插值。`n=1` 是 TD，`n=∞` 是 MC。TD(λ) 用几何权重 `(1-λ)λ^{n-1}` 对所有 `n` 求平均。多数 deep-RL 使用 3 到 20 之间的 `n`。

## 构建它

### 第 1 步：ε-greedy policy 上的 SARSA

```python
def sarsa(env, episodes, alpha=0.1, gamma=0.99, epsilon=0.1):
    Q = defaultdict(lambda: {a: 0.0 for a in ACTIONS})

    def choose(s):
        if random() < epsilon:
            return choice(ACTIONS)
        return max(Q[s], key=Q[s].get)

    for _ in range(episodes):
        s = env.reset()
        a = choose(s)
        while True:
            s_next, r, done = env.step(s, a)
            a_next = choose(s_next) if not done else None
            target = r + (gamma * Q[s_next][a_next] if not done else 0.0)
            Q[s][a] += alpha * (target - Q[s][a])
            if done:
                break
            s, a = s_next, a_next
    return Q
```

八行。和 Q-learning 的 *唯一* 区别是 target 那一行。

### 第 2 步：Q-learning

```python
def q_learning(env, episodes, alpha=0.1, gamma=0.99, epsilon=0.1):
    Q = defaultdict(lambda: {a: 0.0 for a in ACTIONS})
    for _ in range(episodes):
        s = env.reset()
        while True:
            a = choose(s, Q, epsilon)
            s_next, r, done = env.step(s, a)
            target = r + (gamma * max(Q[s_next].values()) if not done else 0.0)
            Q[s][a] += alpha * (target - Q[s][a])
            if done:
                break
            s = s_next
    return Q
```

`max` 把 target 和 behavior 解耦。这个符号就是 on-policy 和 off-policy 的差异。

### 第 3 步：learning curves

追踪每 100 episodes 的 mean return。简单 deterministic GridWorld 中 Q-learning 收敛更快；cliff-walking 中 SARSA 更保守。在 `code/main.py` 的 4×4 GridWorld 上，使用 `α=0.1, ε=0.1` 时，两者约 2,000 episodes 后都接近最优。

### 第 4 步：和 DP truth 比较

运行 value iteration（第 02 课）得到 `Q*`。检查 `max_{s,a} |Q_learned(s,a) - Q*(s,a)|`。健康的 tabular TD agent 在 4×4 GridWorld 上训练 10,000 episodes 后会落在 `~0.5` 以内。

## 陷阱

- **Initial Q values 很重要。** Optimistic init（负 reward task 中 `Q = 0`）鼓励探索。Pessimistic init 可能让 greedy policy 永远困住。
- **α schedule。** Constant `α` 对 non-stationary problems 没问题。Decaying `α_n = 1/n` 理论上收敛，但实践中过慢 — 把 `α` 固定在 `[0.05, 0.3]` 并监控 learning curve。
- **ε schedule。** 从高开始（`ε=1.0`），decay 到 `ε=0.05`。“GLIE”（greedy in the limit with infinite exploration）是收敛条件。
- **Q-learning 中的 max bias。** 当 `Q` 有噪声时，`max` operator 会向上偏。导致 overestimation — Hasselt 的 Double Q-learning（第 05 课 DDQN 使用）通过两个 Q tables 修复。
- **Non-terminating episodes。** TD 可以在没有 terminal 的情况下学习，但你需要要么 cap steps，要么在 cap 处正确 bootstrap。标准做法：把 cap 当作 non-terminal，继续 bootstrap。
- **State hashing。** 如果 states 是 tuples/tensors，使用 hashable key（tuple，不是 list；rounded floats 的 tuple，不是 raw）。

## 使用它

2026 年 TD landscape：

| 任务 | 方法 | 原因 |
|------|--------|--------|
| Small tabular environments | Q-learning | 直接学习 optimal policy。 |
| On-policy safety-critical | SARSA / Expected SARSA | 探索期间更保守。 |
| High-dimensional state | DQN（阶段 9 · 05） | 带 replay 和 target net 的 neural-net Q-function。 |
| Continuous actions | SAC / TD3（阶段 9 · 07） | Q-network 上做 TD update；policy net 发出 actions。 |
| LLM RL（reward-model-based） | PPO / GRPO（阶段 9 · 08、12） | Actor-critic，使用 GAE 的 TD-style advantage。 |
| Offline RL | CQL / IQL（阶段 9 · 08） | 带 conservative regularization 的 Q-learning。 |

2026 论文里 90% 的 “RL” 都是 Q-learning 或 SARSA 的某种扩展。读更深之前，先把 tabular update 写到手指里。

## 交付它

保存为 `outputs/skill-td-agent.md`：

```markdown
---
name: td-agent
description: Pick between Q-learning, SARSA, Expected SARSA for a tabular or small-feature RL task.
version: 1.0.0
phase: 9
lesson: 4
tags: [rl, td-learning, q-learning, sarsa]
---

Given a tabular or small-feature environment, output:

1. Algorithm. Q-learning / SARSA / Expected SARSA / n-step variant. One-sentence reason tied to on-policy vs off-policy and variance.
2. Hyperparameters. α, γ, ε, decay schedule.
3. Initialization. Q_0 value (optimistic vs zero) and justification.
4. Convergence diagnostic. Target learning curve, `|Q - Q*|` check if DP is possible.
5. Deployment caveat. How will exploration behave at inference? Is SARSA's conservatism needed?

Refuse to apply tabular TD to state spaces > 10⁶. Refuse to ship a Q-learning agent without a max-bias caveat. Flag any agent trained with ε held at 1.0 throughout (no exploitation phase).
```

## 练习

1. **简单。** 在 4×4 GridWorld 上实现 Q-learning 和 SARSA。画 2,000 episodes 的 learning curves（每 100 episodes mean return）。谁收敛更快？
2. **中等。** 构建 cliff-walking environment（4×12，最后一行是 cliff，reward -100 并 reset 到 start）。比较 Q-learning 和 SARSA 最终 policies。截屏它们各自路径。谁离 cliff 更近？
3. **困难。** 实现 Double Q-learning。在带 noisy-reward 的 GridWorld 上（每步 reward 加 Gaussian noise σ=5），展示 Q-learning 会显著高估 `V*(0,0)`，而 Double Q-learning 不会。

## 关键词

| 术语 | 大家怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| TD error | “Update signal” | `δ = r + γ V(s') - V(s)`，bootstrapped residual。 |
| TD(0) | “One-step TD” | 每个 transition 后只使用 next state's estimate 更新。 |
| Q-learning | “Off-policy RL 101” | 对 next-state actions 做 `max` 的 TD update；无论 behavior policy 如何都学习 `Q*`。 |
| SARSA | “On-policy Q-learning” | 使用实际 next action 的 TD update；学习当前 ε-greedy π 的 `Q^π`。 |
| Expected SARSA | “Low-variance SARSA” | 用 π 下的期望替换 sampled `a'`。 |
| GLIE | “Correct exploration schedule” | Greedy in the Limit with Infinite Exploration；Q-learning 收敛所需。 |
| Bootstrapping | “在 target 中使用当前估计” | 区分 TD 和 MC 的东西。带来 bias，但大幅降低 variance。 |
| Maximization bias | “Q-learning overestimates” | 对 noisy estimates 取 `max` 会向上偏；Double Q-learning 修复。 |

## 延伸阅读

- [Watkins & Dayan (1992). Q-learning](https://link.springer.com/article/10.1007/BF00992698) — 原始论文和 convergence proof。
- [Sutton & Barto (2018). Ch. 6 — Temporal-Difference Learning](http://incompleteideas.net/book/RLbook2020.pdf) — TD(0)、SARSA、Q-learning、Expected SARSA。
- [Hasselt (2010). Double Q-learning](https://papers.nips.cc/paper_files/paper/2010/hash/091d584fced301b442654dd8c23b3fc9-Abstract.html) — maximization bias 修复。
- [Seijen, Hasselt, Whiteson, Wiering (2009). A Theoretical and Empirical Analysis of Expected SARSA](https://ieeexplore.ieee.org/document/4927542) — expected SARSA motivation。
- [Rummery & Niranjan (1994). On-line Q-learning using connectionist systems](https://www.researchgate.net/publication/2500611_On-Line_Q-Learning_Using_Connectionist_Systems) — 提出 SARSA 名称的论文（当时叫 “modified connectionist Q-learning”）。
- [Sutton & Barto (2018). Ch. 7 — n-step Bootstrapping](http://incompleteideas.net/book/RLbook2020.pdf) — 把 TD(0) 泛化到 TD(n)，这是从 Q-learning 到 eligibility traces，再到 PPO 中 GAE 的路径。
