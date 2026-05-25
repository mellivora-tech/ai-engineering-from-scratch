# Dynamic Programming — Policy Iteration 与 Value Iteration

> Dynamic programming 是作弊版 RL。你已经知道 transition 和 reward functions；只要迭代 Bellman equation，直到 `V` 或 `π` 不再移动。它是每个 sampling-based method 试图接近的 benchmark。

**类型：** 构建
**语言：** Python
**前置要求：** 阶段 9 · 01（MDPs）
**时间：** ~75 分钟

## 问题

你有一个已知模型的 MDP：可以对任意 state-action pair 查询 `P(s' | s, a)` 和 `R(s, a, s')`。Inventory manager 知道 demand distribution。棋盘游戏有 deterministic transitions。Gridworld 是四行 Python。你有一个 *model*。

Model-free RL（Q-learning、PPO、REINFORCE）是为没有模型的情况发明的 — 你只能从 environment 采样。但当你有模型时，有更快、更好的方法：dynamic programming。Bellman 在 1957 年设计了它们。它们仍然定义 correctness：当人们说“这个 MDP 的 optimal policy”时，他们指的是 DP 返回的 policy。

2026 年你需要它们有三个原因。第一，RL research 中每个 tabular environment（GridWorld、FrozenLake、CliffWalking）都用 DP 求解 gold-standard policy。第二，exact values 让你能 *debug* sampling methods：如果 Q-learning 对 `V*(s_0)` 的估计与 DP 答案差 30%，你的 Q-learning 有 bug。第三，现代 offline RL 和 planning methods（MCTS、AlphaZero search、阶段 9 · 10 中的 model-based RL）都会在 learned 或 given model 上迭代 Bellman backup。

## 概念

![Policy iteration and value iteration, side by side](../assets/dp.svg)

**两个算法，都在 Bellman 上做 fixed-point iteration。**

**Policy iteration。** 交替两个步骤，直到 policy 不再变化。

1. *Evaluation:* 给定 policy `π`，通过反复应用 `V(s) ← Σ_a π(a|s) Σ_{s',r} P(s',r|s,a) [r + γ V(s')]` 直到收敛，计算 `V^π`。
2. *Improvement:* 给定 `V^π`，让 `π` 相对于 `V^π` greedy：`π(s) ← argmax_a Σ_{s',r} P(s',r|s,a) [r + γ V(s')]`。

收敛有保证，因为（a）每个 improvement step 要么保持 `π` 相同，要么严格增加某些 state 的 `V^π`，（b）deterministic policies 空间有限。即使大 state spaces，也通常在 ~5–20 个 outer iterations 中收敛。

**Value iteration。** 把 evaluation 和 improvement 折叠成一次 sweep。应用 Bellman *optimality* equation：

`V(s) ← max_a Σ_{s',r} P(s',r|s,a) [r + γ V(s')]`

重复直到 `max_s |V_{new}(s) - V(s)| < ε`。最后通过 greedy action 提取 policy。每次迭代严格更快 — 没有 inner evaluation loop — 但通常需要更多 iterations 才收敛。

**Generalized policy iteration（GPI）。** 统一框架。Value function 和 policy 锁在双向 improvement loop 中；任何把二者推向 mutual consistency 的方法（async value iteration、modified policy iteration、Q-learning、actor-critic、PPO）都是 GPI 实例。

**为什么 `γ < 1` 重要。** Bellman operator 在 sup-norm 中是 `γ`-contraction：`||T V - T V'||_∞ ≤ γ ||V - V'||_∞`。Contraction 意味着唯一 fixed point 和 geometric convergence。放弃 `γ < 1` 就失去保证 — 你需要 finite horizon 或 absorbing terminal state。

## 构建它

### 第 1 步：构建 GridWorld MDP model

使用第 01 课同一个 4×4 GridWorld。我们添加 stochastic variant：agent 以 `0.1` 概率 slip 到随机垂直方向。

```python
SLIP = 0.1

def transitions(state, action):
    if state == TERMINAL:
        return [(state, 0.0, 1.0)]
    outcomes = []
    for direction, prob in action_probs(action):
        outcomes.append((apply_move(state, direction), -1.0, prob))
    return outcomes
```

`transitions(s, a)` 返回 `(s', r, p)` 列表。这就是整个 model。

### 第 2 步：policy evaluation

给定 policy `π(s) = {action: prob}`，迭代 Bellman equation 直到 `V` 不再移动：

```python
def policy_evaluation(policy, gamma=0.99, tol=1e-6):
    V = {s: 0.0 for s in states()}
    while True:
        delta = 0.0
        for s in states():
            v = sum(pi_a * sum(p * (r + gamma * V[s_prime])
                              for s_prime, r, p in transitions(s, a))
                   for a, pi_a in policy(s).items())
            delta = max(delta, abs(v - V[s]))
            V[s] = v
        if delta < tol:
            return V
```

### 第 3 步：policy improvement

用相对于 `V` 的 greedy policy 替换 `π`。如果 `π` 没有变化，返回 — 我们已经在 optimum。

```python
def policy_improvement(V, gamma=0.99):
    new_policy = {}
    for s in states():
        best_a = max(
            ACTIONS,
            key=lambda a: sum(p * (r + gamma * V[s_prime])
                              for s_prime, r, p in transitions(s, a)),
        )
        new_policy[s] = best_a
    return new_policy
```

### 第 4 步：把它们接起来

```python
def policy_iteration(gamma=0.99):
    policy = {s: "up" for s in states()}   # arbitrary start
    for _ in range(100):
        V = policy_evaluation(lambda s: {policy[s]: 1.0}, gamma)
        new_policy = policy_improvement(V, gamma)
        if new_policy == policy:
            return V, policy
        policy = new_policy
```

4×4 上典型收敛：4–6 个 outer iterations。输出 `V*(0,0) ≈ -6` 和严格减少步数的 policy。

### 第 5 步：value iteration（one-loop 版本）

```python
def value_iteration(gamma=0.99, tol=1e-6):
    V = {s: 0.0 for s in states()}
    while True:
        delta = 0.0
        for s in states():
            v = max(sum(p * (r + gamma * V[s_prime])
                       for s_prime, r, p in transitions(s, a))
                   for a in ACTIONS)
            delta = max(delta, abs(v - V[s]))
            V[s] = v
        if delta < tol:
            break
    policy = policy_improvement(V, gamma)
    return V, policy
```

相同 fixed point，更少代码。

## 陷阱

- **忘记处理 terminals。** 如果对 absorbing state 应用 Bellman，它仍然会选一个“best action”，但其实什么都不变。用 `if s == terminal: V[s] = 0` guard。
- **Sup-norm vs L2 convergence。** 使用 `max |V_new - V|`，不是平均值。理论保证在 sup-norm 上。
- **In-place vs synchronous updates。** In-place 更新 `V[s]`（Gauss-Seidel）比单独 `V_new` dict（Jacobi）收敛更快。生产代码用 in-place。
- **Policy ties。** 如果两个 actions 有相同 Q-value，`argmax` 可能每次以不同方式 break ties，导致 “policy stable” check 振荡。使用稳定 tie-break（固定顺序中的第一个 action）。
- **State-space explosion。** DP 每次 sweep 是 `O(|S| · |A|)`。可用到约 10⁷ states。再往上需要 function approximation（阶段 9 · 05 起）。

## 使用它

2026 年，DP 是 correctness baseline，也是 planners 的 inner loop：

| 用例 | 方法 |
|----------|--------|
| 精确求解小 tabular MDP | Value iteration（更简单）或 policy iteration（更少 outer steps） |
| 验证 Q-learning / PPO 实现 | 在 toy environment 上与 DP-optimal V* 比较 |
| Model-based RL（阶段 9 · 10） | 在 learned transition model 上做 Bellman backup |
| AlphaZero / MuZero 中的 planning | Monte Carlo Tree Search = async Bellman backup |
| Offline RL（CQL、IQL） | Conservative Q-iteration — 带 OOD actions penalty 的 DP |

每当有人说 “the optimal value function”，意思就是 “DP fixed point”。在论文里看到 `V*` 或 `Q*` 时，脑中就浮现这个 loop。

## 交付它

保存为 `outputs/skill-dp-solver.md`：

```markdown
---
name: dp-solver
description: Solve a small tabular MDP exactly via policy iteration or value iteration. Report convergence behavior.
version: 1.0.0
phase: 9
lesson: 2
tags: [rl, dynamic-programming, bellman]
---

Given an MDP with a known model, output:

1. Choice. Policy iteration vs value iteration. Reason tied to |S|, |A|, γ.
2. Initialization. V_0, starting policy. Convergence sensitivity.
3. Stopping. Sup-norm tolerance ε. Expected number of sweeps.
4. Verification. V*(s_0) computed exactly. Greedy policy extracted.
5. Use. How this baseline will be used to debug/evaluate sampling-based methods.

Refuse to run DP on state spaces > 10⁷. Refuse to claim convergence without a sup-norm check. Flag any γ ≥ 1 on an infinite-horizon task as a guarantee violation.
```

## 练习

1. **简单。** 在 4×4 GridWorld 上用 `γ ∈ {0.9, 0.99}` 运行 value iteration。多少 sweeps 后 `max |ΔV| < 1e-6`？把 `V*` 打印成 4×4 grid。
2. **中等。** 在 *stochastic* GridWorld（slip probability `0.1`）上比较 policy iteration 和 value iteration。统计：sweeps、wall-clock time、final `V*(0,0)`。哪个在 iterations 上更快？哪个在 wall-clock 上更快？
3. **困难。** 构建 modified policy iteration：evaluation step 中只运行 `k` sweeps，而不是到收敛。对 `k ∈ {1, 2, 5, 10, 50}`，画出 `V*(0,0)` error vs `k`。曲线告诉你 evaluation/improvement tradeoff 什么？

## 关键词

| 术语 | 大家怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| Policy iteration | “DP algorithm” | 交替 evaluation（`V^π`）和 improvement（相对 `V^π` 的 greedy `π`），直到 policy 不变。 |
| Value iteration | “Faster DP” | 一次 sweep 中应用 Bellman optimality backup；几何收敛到 `V*`。 |
| Bellman operator | “The recursion” | `(T V)(s) = max_a Σ P (r + γ V(s'))`；sup-norm 中的 `γ`-contraction。 |
| Contraction | “为什么 DP 收敛” | 任何满足 `||T x - T y|| ≤ γ ||x - y||` 的 operator 都有唯一 fixed point。 |
| GPI | “一切都是 DP” | Generalized Policy Iteration：任何推动 `V` 和 `π` 到 mutual consistency 的方法。 |
| Synchronous update | “Jacobi-style” | 一个 sweep 中始终使用旧 `V`；容易分析但更慢。 |
| In-place update | “Gauss-Seidel-style” | 使用正在更新中的 `V`；实践中收敛更快。 |

## 延伸阅读

- [Sutton & Barto (2018). Ch. 4 — Dynamic Programming](http://incompleteideas.net/book/RLbook2020.pdf) — policy iteration 和 value iteration 的标准呈现。
- [Bertsekas (2019). Reinforcement Learning and Optimal Control](http://www.athenasc.com/rlbook.html) — contraction-mapping 论证的严格处理。
- [Puterman (2005). Markov Decision Processes](https://onlinelibrary.wiley.com/doi/book/10.1002/9780470316887) — modified policy iteration 及其 convergence analysis。
- [Howard (1960). Dynamic Programming and Markov Processes](https://mitpress.mit.edu/9780262582300/dynamic-programming-and-markov-processes/) — 原始 policy iteration 论文。
- [Bertsekas & Tsitsiklis (1996). Neuro-Dynamic Programming](http://www.athenasc.com/ndpbook.html) — 从 DP 到 approximate-DP / deep RL 的桥梁，后续每课都会用到。
