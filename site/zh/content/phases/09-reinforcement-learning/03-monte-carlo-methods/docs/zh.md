# Monte Carlo Methods — 从完整 Episodes 中学习

> Dynamic programming 需要 model。Monte Carlo 只需要 episodes。运行 policy，观察 returns，求平均。RL 中最简单的想法 — 也是解锁后续一切的想法。

**类型：** 构建
**语言：** Python
**前置要求：** 阶段 9 · 01（MDPs），阶段 9 · 02（Dynamic Programming）
**时间：** ~75 分钟

## 问题

Dynamic programming 很优雅，但它假设你能对每个 state 和 action 查询 `P(s' | s, a)`。现实世界几乎没有这么工作。机器人无法解析计算某个 joint torque 后 camera pixels 的分布。Pricing algorithm 无法对所有可能客户反应积分。LLM 无法枚举一个 token 后所有可能 continuations。

你需要一个只要求能够从 environment *采样* 的方法。运行 policy。得到 trajectory `s_0, a_0, r_1, s_1, a_1, r_2, …, s_T`。用它估计 values。这就是 Monte Carlo。

从 DP 到 MC 的转变在哲学上很重要：我们从 *known model + exact backup* 变成 *sampled rollouts + averaged return*。Variance 增大，但适用性暴涨。这课之后的每个 RL 算法 — TD、Q-learning、REINFORCE、PPO、GRPO — 本质上都是 Monte Carlo estimator，有时再叠上 bootstrapping。

## 概念

![Monte Carlo: rollout, compute returns, average; first-visit vs every-visit](../assets/monte-carlo.svg)

**核心想法，一行：** `V^π(s) = E_π[G_t | s_t = s] ≈ (1/N) Σ_i G^{(i)}(s)`，其中 `G^{(i)}(s)` 是 policy `π` 下访问 `s` 后观察到的 returns。

**First-visit vs every-visit MC。** 给定一个 episode 多次访问 state `s`，first-visit MC 只统计第一次访问后的 return；every-visit MC 统计所有访问。二者在极限中都无偏。First-visit 更易分析（iid samples）。Every-visit 每个 episode 使用更多数据，实践中通常收敛更快。

**Incremental mean。** 不存储所有 returns，而是更新 running average：

`V_n(s) = V_{n-1}(s) + (1/n) [G_n - V_{n-1}(s)]`

重排：`V_new = V_old + α · (target - V_old)`，其中 `α = 1/n`。把 `1/n` 换成常量 step-size `α ∈ (0, 1)`，你得到一个能跟踪 `π` 变化的 non-stationary MC estimator。这个动作就是从 MC 跳到 TD，再到每个现代 RL 算法的关键。

**Exploration 现在成了问题。** DP 通过枚举触碰每个 state。MC 只看到 policy 会访问的 states。如果 `π` 是 deterministic，state space 的大片区域永远不会被 sampled，它们的 value estimates 会永远保持 0。三个修复按历史顺序是：

1. **Exploring starts。** 每个 episode 从随机 (s, a) pair 开始。保证覆盖；实践中不现实（你不能把 robot “reset” 到任意 state）。
2. **ε-greedy。** 相对当前 Q 贪心行动，但以概率 `ε` 选择随机 action。渐近上所有 state-action pairs 都会被 sampled。
3. **Off-policy MC。** 在 behavior policy `μ` 下收集数据，通过 importance sampling 学习 target policy `π`。高 variance，但它是通往 DQN 这类 replay-buffer methods 的桥梁。

**Monte Carlo Control。** Evaluate → improve → evaluate，像 policy iteration 一样，只是 evaluation 基于 sampling：

1. 运行 `π`，得到 episode。
2. 用 observed returns 更新 `Q(s, a)`。
3. 让 `π` 相对于 `Q` ε-greedy。
4. 重复。

在温和条件下（每个 pair 无限次访问，`α` 满足 Robbins-Monro），以概率 1 收敛到 `Q*` 和 `π*`。

## 构建它

### 第 1 步：rollout → (s, a, r) 列表

```python
def rollout(env, policy, max_steps=200):
    trajectory = []
    s = env.reset()
    for _ in range(max_steps):
        a = policy(s)
        s_next, r, done = env.step(s, a)
        trajectory.append((s, a, r))
        s = s_next
        if done:
            break
    return trajectory
```

没有 model，只有 `env.reset()` 和 `env.step(s, a)`。和 gym environment 接口相同，只是剥到最小。

### 第 2 步：计算 returns（反向 sweep）

```python
def returns_from(trajectory, gamma):
    returns = []
    G = 0.0
    for _, _, r in reversed(trajectory):
        G = r + gamma * G
        returns.append(G)
    return list(reversed(returns))
```

一次 pass，`O(T)`。反向 recurrence `G_t = r_{t+1} + γ G_{t+1}` 避免重复求和。

### 第 3 步：first-visit MC evaluation

```python
def mc_policy_evaluation(env, policy, episodes, gamma=0.99):
    V = defaultdict(float)
    counts = defaultdict(int)
    for _ in range(episodes):
        trajectory = rollout(env, policy)
        returns = returns_from(trajectory, gamma)
        seen = set()
        for t, ((s, _, _), G) in enumerate(zip(trajectory, returns)):
            if s in seen:
                continue
            seen.add(s)
            counts[s] += 1
            V[s] += (G - V[s]) / counts[s]
    return V
```

三行做核心工作：把 state 标为 seen、增加 count、更新 running mean。

### 第 4 步：ε-greedy MC control（on-policy）

```python
def mc_control(env, episodes, gamma=0.99, epsilon=0.1):
    Q = defaultdict(lambda: {a: 0.0 for a in ACTIONS})
    counts = defaultdict(lambda: {a: 0 for a in ACTIONS})

    def policy(s):
        if random() < epsilon:
            return choice(ACTIONS)
        return max(Q[s], key=Q[s].get)

    for _ in range(episodes):
        trajectory = rollout(env, policy)
        returns = returns_from(trajectory, gamma)
        seen = set()
        for (s, a, _), G in zip(trajectory, returns):
            if (s, a) in seen:
                continue
            seen.add((s, a))
            counts[s][a] += 1
            Q[s][a] += (G - Q[s][a]) / counts[s][a]
    return Q, policy
```

### 第 5 步：与 DP gold standard 比较

随着 episodes → ∞，你的 `V^π` MC estimate 应该与第 02 课的 DP result 一致。实践中：在 4×4 GridWorld 上跑 50,000 episodes，就能到 DP 答案 `~0.1` 内。

## 陷阱

- **Infinite episodes。** MC 要求 episodes *终止*。如果 policy 会无限循环，设置 `max_steps` cap，并把 cap 视为隐式失败。GridWorld 随机 policy 经常 timeout — 这很正常，只要你正确统计。
- **Variance。** MC 使用完整 returns。长 episodes 上 variance 很大 — 末尾一次倒霉 reward 会等量移动 `V(s_0)`。TD methods（第 04 课）通过 bootstrapping 降低它。
- **State coverage。** 新 Q 上的 greedy MC 会因为 ties 只尝试一个 action。你 *必须* explore（ε-greedy、exploring starts、UCB）。
- **Non-stationary policies。** 如果 `π` 变化（如 MC control），旧 returns 来自另一个 policy。Constant-α MC 能处理；sample-average MC 不能。
- **Off-policy importance sampling。** 权重 `π(a|s)/μ(a|s)` 会沿 trajectory 相乘。Horizon 一长 variance 爆炸。用 per-decision weighted IS 截断，或切换到 TD。

## 使用它

Monte Carlo methods 在 2026 年的角色：

| 用例 | 为什么 MC |
|----------|--------|
| Short-horizon games（blackjack、poker） | Episodes 自然终止；returns 干净。 |
| Logged policy 的 offline evaluation | 对 stored trajectories 求 average discounted returns。 |
| Monte Carlo Tree Search（AlphaZero） | 从 tree leaves 做 MC rollouts 指导 selection。 |
| LLM RL evaluation | 对给定 policy 的 sampled completions 计算 average reward。 |
| PPO 中的 baseline estimation | Advantage target `A_t = G_t - V(s_t)` 使用 MC `G_t`。 |
| Teaching RL | 最简单且真正有效的算法 — 去掉 bootstrapping 看核心。 |

现代 deep-RL algorithms（PPO、SAC）通过 `n`-step returns 或 GAE，在 pure MC（full returns）和 pure TD（one-step bootstrap）之间插值。两个端点都是同一个 estimator 的实例。

## 交付它

保存为 `outputs/skill-mc-evaluator.md`：

```markdown
---
name: mc-evaluator
description: Evaluate a policy via Monte Carlo rollouts and produce a convergence report with DP-comparison if available.
version: 1.0.0
phase: 9
lesson: 3
tags: [rl, monte-carlo, evaluation]
---

Given an environment (episodic, with reset+step API) and a policy, output:

1. Method. First-visit vs every-visit MC. Reason.
2. Episode budget. Target number, variance diagnostic, expected standard error.
3. Exploration plan. ε schedule (if needed) or exploring starts.
4. Gold-standard comparison. DP-optimal V* if tabular; otherwise a bound from a Q-learning / PPO baseline.
5. Termination check. Max-step cap, timeouts, handling of non-terminating trajectories.

Refuse to run MC on non-episodic tasks without a finite horizon cap. Refuse to report V^π estimates from fewer than 100 episodes per state for tabular tasks. Flag any policy with zero-variance actions as an exploration risk.
```

## 练习

1. **简单。** 在 4×4 GridWorld 上实现 uniform-random policy 的 first-visit MC evaluation。运行 10,000 episodes。画 `V(0,0)` 随 episode count 的曲线，并与 DP 答案比较。
2. **中等。** 用 `ε ∈ {0.01, 0.1, 0.3}` 实现 ε-greedy MC control。比较 20,000 episodes 后的 mean return。曲线长什么样？Bias-variance tradeoff 在哪里？
3. **困难。** 用 importance sampling 实现 *off-policy* MC：在 uniform-random policy `μ` 下收集数据，估计 deterministic optimal policy `π` 的 `V^π`。比较 plain IS、per-decision IS、weighted IS。哪一个 variance 最低？

## 关键词

| 术语 | 大家怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| Monte Carlo | “Random sampling” | 通过对来自分布的 iid samples 求平均来估计 expectations。 |
| Return `G_t` | “Future reward” | 从 step `t` 到 episode end 的 discounted rewards：`Σ_{k≥0} γ^k r_{t+k+1}`。 |
| First-visit MC | “每个 state 只数一次” | Episode 中只有第一次访问贡献 value estimate。 |
| Every-visit MC | “使用所有 visits” | 每次访问都贡献；略有 bias 但 sample-efficient。 |
| ε-greedy | “Exploration noise” | 以 `1-ε` 概率选 greedy action；以 `ε` 概率随机。 |
| Importance sampling | “纠正从错误分布采样” | 用 `π(a|s)/μ(a|s)` 乘积重加权 returns，从 `μ` 数据估计 `V^π`。 |
| On-policy | “从自己的数据学习” | Target policy = behavior policy。Vanilla MC、PPO、SARSA。 |
| Off-policy | “从别人的数据学习” | Target policy ≠ behavior policy。Importance-sampled MC、Q-learning、DQN。 |

## 延伸阅读

- [Sutton & Barto (2018). Ch. 5 — Monte Carlo Methods](http://incompleteideas.net/book/RLbook2020.pdf) — 标准处理。
- [Singh & Sutton (1996). Reinforcement Learning with Replacing Eligibility Traces](https://link.springer.com/article/10.1007/BF00114726) — first-visit vs every-visit analysis。
- [Precup, Sutton, Singh (2000). Eligibility Traces for Off-Policy Policy Evaluation](http://incompleteideas.net/papers/PSS-00.pdf) — off-policy MC 和 variance control。
- [Mahmood et al. (2014). Weighted Importance Sampling for Off-Policy Learning](https://arxiv.org/abs/1404.6362) — 现代低 variance IS estimators。
- [Tesauro (1995). TD-Gammon, A Self-Teaching Backgammon Program](https://dl.acm.org/doi/10.1145/203330.203343) — MC/TD self-play 收敛到超人水平的首次大规模经验展示；本阶段后半部分所有课程的概念先驱。
