# Policy Gradient — 从零实现 REINFORCE

> 停止估计 value。直接参数化 policy，计算 expected return 的梯度，然后向上走一步。Williams（1992）用一个定理写清了它。这就是 PPO、GRPO 以及所有 LLM RL 循环存在的原因。

**类型：** 构建
**语言：** Python
**先修：** Phase 3 · 03（Backpropagation），Phase 9 · 03（Monte Carlo），Phase 9 · 04（TD Learning）
**时间：** 约 75 分钟

## 问题

Q-learning 和 DQN 参数化的是 *value* 函数。你通过 `argmax Q` 选择动作。这对离散动作和离散状态没问题。但当动作连续时（要对 10 维 torque 做哪个 `argmax`？），或者你想要一个随机 policy 时，它就会失效（`argmax` 按构造就是确定性的）。

Policy gradients 改为参数化 *policy* 本身。`π_θ(a | s)` 是一个输出动作分布的神经网络。行动时从中采样。计算 expected return 关于 `θ` 的梯度。向上走一步。没有 `argmax`。没有 Bellman recursion。只有对 `J(θ) = E_{π_θ}[G]` 做 gradient ascent。

REINFORCE 定理（Williams 1992）告诉我们这个梯度是可计算的：`∇J(θ) = E_π[ G · ∇_θ log π_θ(a | s) ]`。运行一个 episode。计算 return。每一步都乘上 `∇ log π_θ(a | s)`。求平均。做梯度上升。完成。

2026 年的每个 LLM-RL 算法，包括 PPO、DPO、GRPO，都是 REINFORCE 的改良版。把它练到手上，是理解本阶段后续内容，以及 Phase 10 · 07（RLHF implementation）和 Phase 10 · 08（DPO）的前提。

## 核心概念

![Policy gradient: softmax policy, log-π gradient, return-weighted update](../assets/policy-gradient.svg)

**Policy gradient theorem。** 对任意由 `θ` 参数化的 policy `π_θ`：

`∇J(θ) = E_{τ ~ π_θ}[ Σ_{t=0}^{T} G_t · ∇_θ log π_θ(a_t | s_t) ]`

其中 `G_t = Σ_{k=t}^{T} γ^{k-t} r_{k+1}` 是从第 `t` 步开始的 discounted return。期望取在从 `π_θ` 采样的完整 trajectory `τ` 上。

**证明很短。** 在期望下对 `J(θ) = Σ_τ P(τ; θ) G(τ)` 求导。使用 `∇P(τ; θ) = P(τ; θ) ∇ log P(τ; θ)`（log-derivative trick）。分解 `log P(τ; θ) = Σ log π_θ(a_t | s_t) + environment terms that do not depend on θ`。环境项消失。两行代数就得到定理。

**方差降低技巧。** 原始 REINFORCE 的方差很吓人：return 有噪声，`∇ log π` 有噪声，它们的乘积更有噪声。两个标准修复：

1. **Baseline subtraction。** 对任意不依赖 `a_t` 的 baseline `b(s_t)`，把 `G_t` 替换为 `G_t - b(s_t)`。这是无偏的，因为 `E[b(s_t) · ∇ log π(a_t | s_t)] = 0`。典型选择：`b(s_t) = V̂(s_t)`，由 critic 学到 → actor-critic（第 07 课）。
2. **Reward-to-go。** 把 `Σ_t G_t · ∇ log π_θ(a_t | s_t)` 替换为 `Σ_t G_t^{from t} · ∇ log π_θ(a_t | s_t)`。对某个动作而言只有未来 return 重要，过去奖励只贡献零均值噪声。

结合起来得到：

`∇J ≈ (1/N) Σ_{i=1}^{N} Σ_{t=0}^{T_i} [ G_t^{(i)} - V̂(s_t^{(i)}) ] · ∇_θ log π_θ(a_t^{(i)} | s_t^{(i)})`

这就是带 baseline 的 REINFORCE，也是 A2C（第 07 课）和 PPO（第 08 课）的直接祖先。

**Softmax policy 参数化。** 对离散动作，标准选择是：

`π_θ(a | s) = exp(f_θ(s, a)) / Σ_{a'} exp(f_θ(s, a'))`

其中 `f_θ` 是任意每个动作输出一个 score 的神经网络。梯度形式很干净：

`∇_θ log π_θ(a | s) = ∇_θ f_θ(s, a) - Σ_{a'} π_θ(a' | s) ∇_θ f_θ(s, a')`

也就是，被采取动作的 score 减去它在 policy 下的期望值。

**连续动作的 Gaussian policy。** `π_θ(a | s) = N(μ_θ(s), σ_θ(s))`。`∇ log N(a; μ, σ)` 有闭式形式。这就是 Phase 9 · 07 中 SAC 需要的全部。

## 动手构建

### 第 1 步：softmax policy network

```python
def policy_logits(theta, state_features):
    return [dot(theta[a], state_features) for a in range(N_ACTIONS)]

def softmax(logits):
    m = max(logits)
    exps = [exp(l - m) for l in logits]
    Z = sum(exps)
    return [e / Z for e in exps]
```

在表格环境中使用线性 policy（每个动作一个权重向量）。在 Atari 中，换成 CNN，并保留 softmax head。

### 第 2 步：采样与 log-probability

```python
def sample_action(probs, rng):
    x = rng.random()
    cum = 0
    for a, p in enumerate(probs):
        cum += p
        if x <= cum:
            return a
    return len(probs) - 1

def log_prob(probs, a):
    return log(probs[a] + 1e-12)
```

### 第 3 步：rollout 并捕获 log-probs

```python
def rollout(theta, env, rng, gamma):
    trajectory = []
    s = env.reset()
    while not done:
        logits = policy_logits(theta, s)
        probs = softmax(logits)
        a = sample_action(probs, rng)
        s_next, r, done = env.step(s, a)
        trajectory.append((s, a, r, probs))
        s = s_next
    return trajectory
```

### 第 4 步：REINFORCE update

```python
def reinforce_step(theta, trajectory, gamma, lr, baseline=0.0):
    returns = compute_returns(trajectory, gamma)
    for (s, a, _, probs), G in zip(trajectory, returns):
        advantage = G - baseline
        grad_log_pi_a = [-p for p in probs]
        grad_log_pi_a[a] += 1.0
        for i in range(N_ACTIONS):
            for j in range(len(s)):
                theta[i][j] += lr * advantage * grad_log_pi_a[i] * s[j]
```

梯度 `∇ log π(a|s) = e_a - π(·|s)`（`a` 的 onehot 减去概率）是 softmax policy gradient 的核心。把它练成肌肉记忆。

### 第 5 步：baselines

用最近 episode 的 `G` 运行均值做 baseline，就足以让 4×4 GridWorld 跑起来；大约 500 个 episode 收敛。把 baseline 升级为学到的 `V̂(s)`，你就得到了 actor-critic。

## 常见坑

- **梯度爆炸。** Returns 可能非常大。在乘以 `∇ log π` 前，始终把 batch 内的 `G` 归一化到约 `~N(0, 1)`。
- **Entropy collapse。** Policy 太早收敛到近乎确定性的动作，停止探索，然后卡住。修复：向目标加入 entropy bonus `β · H(π(·|s))`。
- **高方差。** 原始 REINFORCE 需要成千上万个 episode。Critic baseline（第 07 课）或 TRPO/PPO 的 trust region（第 08 课）是标准修复。
- **样本效率低。** On-policy 意味着每个 transition 更新一次后就丢弃。通过 importance sampling 做 off-policy correction 可以重新使用数据，但代价是方差（PPO 的 ratio 就是 clipped IS weight）。
- **非平稳梯度。** 100 个 episode 前的同一个梯度用的是旧 `π`。因此 on-policy 方法每几个 rollout 就更新一次。
- **Credit assignment。** 没有 reward-to-go 时，过去奖励会贡献噪声。始终使用 reward-to-go。

## 使用它

到 2026 年，REINFORCE 很少被直接运行，但它的梯度公式无处不在：

| 使用场景 | 派生方法 |
|----------|----------|
| 连续控制 | PPO / SAC with Gaussian policy |
| LLM RLHF | 带 KL penalty 的 PPO，运行在 token-level policy 上 |
| LLM reasoning（DeepSeek） | GRPO：带 group-relative baseline、无 critic 的 REINFORCE |
| Multi-agent | Centralized-critic REINFORCE（MADDPG，COMA） |
| 离散动作机器人 | A2C，A3C，PPO |
| 仅有偏好数据的设置 | DPO：把 REINFORCE 改写成 preference-likelihood loss，无采样 |

当你在 2026 年的训练脚本里看到 `loss = -advantage * log_prob`，那就是带 baseline 的 REINFORCE。整篇论文（DPO、GRPO、RLOO）都只是叠在这一行之上的方差降低技巧。

## 交付

保存为 `outputs/skill-policy-gradient-trainer.md`：

```markdown
---
name: policy-gradient-trainer
description: Produce a REINFORCE / actor-critic / PPO training config for a given task and diagnose variance issues.
version: 1.0.0
phase: 9
lesson: 6
tags: [rl, policy-gradient, reinforce]
---

Given an environment (discrete / continuous actions, horizon, reward stats), output:

1. Policy head. Softmax (discrete) or Gaussian (continuous) with parameter counts.
2. Baseline. None (vanilla), running mean, learned `V̂(s)`, or A2C critic.
3. Variance controls. Reward-to-go on by default, return normalization, gradient clip value.
4. Entropy bonus. Coefficient β and decay schedule.
5. Batch size. Episodes per update; on-policy data freshness contract.

Refuse REINFORCE-no-baseline on horizons > 500 steps. Refuse continuous-action control with a softmax head. Flag any run with `β = 0` and observed policy entropy < 0.1 as entropy-collapsed.
```

## 练习

1. **简单。** 在 4×4 GridWorld 上用线性 softmax policy 实现 REINFORCE。不使用 baseline 训练 1,000 个 episode。绘制学习曲线；测量方差（returns 的 std）。
2. **中等。** 加入 running-mean baseline。再次训练。把样本效率和方差与 vanilla run 比较。baseline 能把收敛步数减少多少？
3. **困难。** 加入 entropy bonus `β · H(π)`。扫描 `β ∈ {0, 0.01, 0.1, 1.0}`。绘制最终 return 和 policy entropy。这个任务上的甜点在哪里？

## 关键术语

| 术语 | 人们通常怎么说 | 它实际的含义 |
|------|----------------|--------------|
| Policy gradient | “直接训练 policy” | `∇J(θ) = E[G · ∇ log π_θ(a|s)]`；由 log-derivative trick 推导而来。 |
| REINFORCE | “最早的 PG 算法” | Williams（1992）；Monte Carlo returns 乘以 log-policy gradient。 |
| Log-derivative trick | “Score function estimator” | `∇P(τ;θ) = P(τ;θ) · ∇ log P(τ;θ)`；让期望的梯度可处理。 |
| Baseline | “方差降低” | 从 `G` 中减去任意 `b(s)`；无偏，因为 `E[b · ∇ log π] = 0`。 |
| Reward-to-go | “只有未来 return 算数” | 用 `G_t^{from t}` 替代完整的 `G_0`；正确且方差更低。 |
| Entropy bonus | “鼓励探索” | `+β · H(π(·|s))` 项防止 policy collapse。 |
| On-policy | “在刚看到的数据上训练” | 梯度期望相对于当前 policy，不能直接复用旧数据。 |
| Advantage | “比平均好多少” | `A(s, a) = G(s, a) - V(s)`；带 baseline 的 REINFORCE 要乘上的有符号量。 |

## 延伸阅读

- [Williams (1992). Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning](https://link.springer.com/article/10.1007/BF00992696) — REINFORCE 原始论文。
- [Sutton et al. (2000). Policy Gradient Methods for Reinforcement Learning with Function Approximation](https://papers.nips.cc/paper_files/paper/1999/hash/464d828b85b0bed98e80ade0a5c43b0f-Abstract.html) — 带函数近似的现代 policy-gradient theorem。
- [Sutton & Barto (2018). Ch. 13 — Policy Gradient Methods](http://incompleteideas.net/book/RLbook2020.pdf) — 教科书讲解。
- [OpenAI Spinning Up — VPG / REINFORCE](https://spinningup.openai.com/en/latest/algorithms/vpg.html) — 清晰的教学式讲解，含 PyTorch 代码。
- [Peters & Schaal (2008). Reinforcement Learning of Motor Skills with Policy Gradients](https://homes.cs.washington.edu/~todorov/courses/amath579/reading/PolicyGradient.pdf) — 方差降低与 natural-gradient 视角，把 REINFORCE 连接到 trust-region 家族（TRPO、PPO）。
