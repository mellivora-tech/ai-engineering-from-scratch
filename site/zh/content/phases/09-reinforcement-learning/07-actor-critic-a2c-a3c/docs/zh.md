# Actor-Critic — A2C 和 A3C

> REINFORCE 噪声很大。加入一个学习 `V̂(s)` 的 critic，从 return 中减去它，就得到一个期望相同但方差低得多的 advantage。这就是 actor-critic。A2C 同步运行它；A3C 在线程间异步运行它。二者都是理解每个现代 deep-RL 方法的心智模型。

**类型：** 构建
**语言：** Python
**先修：** Phase 9 · 04（TD Learning），Phase 9 · 06（REINFORCE）
**时间：** 约 75 分钟

## 问题

Vanilla REINFORCE 能工作，但方差很糟。Monte Carlo returns `G_t` 在 episode 之间可能相差 10 倍以上。把这种噪声乘以 `∇ log π` 再求平均，会产生一个梯度估计器：它需要成千上万个 episode，才能把 policy 推动到 DQN 用少得多的更新就能到达的位置。

方差来自使用原始 returns。如果你减去一个 baseline `b(s_t)`，也就是任意 state 的函数（包括学到的 value），期望不变而方差下降。最可行的 baseline 是 `V̂(s_t)`。此时乘以 `∇ log π` 的量就是 *advantage*：

`A(s, a) = G - V̂(s)`

如果某个动作带来高于平均的 return，它就是好动作；低于平均就是坏动作。带 learned critic 的 REINFORCE 就是 *actor-critic*。Critic 给 actor 提供低方差老师。这是 2015 年之后每个 deep-policy 方法的基础（A2C、A3C、PPO、SAC、IMPALA）。

## 核心概念

![Actor-critic: policy net plus value net, TD residual as advantage](../assets/actor-critic.svg)

**两个网络，一个共享 loss：**

- **Actor** `π_θ(a | s)`：policy。用于采样行动。通过 policy gradient 训练。
- **Critic** `V_φ(s)`：估计从某个 state 出发的 expected return。通过最小化 `(V_φ(s) - target)²` 训练。

**Advantage。** 两种标准形式：

- *MC advantage:* `A_t = G_t - V_φ(s_t)`。无偏，方差更高。
- *TD advantage:* `A_t = r_{t+1} + γ V_φ(s_{t+1}) - V_φ(s_t)`。有偏（使用 `V_φ`），但方差低得多。也叫 *TD residual* `δ_t`。

**n-step advantage。** 在二者之间插值：

`A_t^{(n)} = r_{t+1} + γ r_{t+2} + … + γ^{n-1} r_{t+n} + γ^n V_φ(s_{t+n}) - V_φ(s_t)`

`n = 1` 是纯 TD。`n = ∞` 是 MC。大多数实现对 Atari 用 `n = 5`，对 MuJoCo 上的 PPO 用 `n = 2048`。

**Generalized Advantage Estimation（GAE）。** Schulman 等人（2016）提出对所有 n-step advantages 做指数加权平均：

`A_t^{GAE} = Σ_{l=0}^{∞} (γλ)^l δ_{t+l}`

其中 `λ ∈ [0, 1]`。`λ = 0` 是 TD（低方差，高偏差）。`λ = 1` 是 MC（高方差，无偏）。`λ = 0.95` 是 2026 年默认值：调它，直到 bias/variance 旋钮落在你想要的位置。

**A2C：synchronous advantage actor-critic。** 在 `N` 个并行环境中收集 `T` 步。为每一步计算 advantage。在合并 batch 上更新 actor 和 critic。重复。这是 A3C 更简单、更易扩展的兄弟。

**A3C：asynchronous advantage actor-critic。** Mnih 等人（2016）。启动 `N` 个 worker thread，每个运行一个 env。每个 worker 在自己的 rollout 上本地计算梯度，然后异步应用到共享参数服务器。不需要 replay buffer：worker 通过运行不同 trajectory 来去相关。A3C 证明了你可以在 CPU 上规模化训练。到 2026 年，基于 GPU 的 A2C（batched parallel envs）占主导，因为 GPU 喜欢大 batch。

**组合 loss。**

`L(θ, φ) = -E[ A_t · log π_θ(a_t | s_t) ]  +  c_v · E[(V_φ(s_t) - G_t)²]  -  c_e · E[H(π_θ(·|s_t))]`

三个项：policy-gradient loss、value regression、entropy bonus。`c_v ~ 0.5`、`c_e ~ 0.01` 是典型起点。

## 动手构建

### 第 1 步：critic

线性 critic `V_φ(s) = w · features(s)` 用 MSE 更新：

```python
def critic_update(w, x, target, lr):
    v_hat = dot(w, x)
    err = target - v_hat
    for j in range(len(w)):
        w[j] += lr * err * x[j]
    return v_hat
```

在表格环境中，critic 几百个 episode 就能收敛。在 Atari 中，把线性 critic 换成共享 CNN trunk + value head。

### 第 2 步：n-step advantage

给定长度为 `T` 的 rollout 和 bootstrapped final `V(s_T)`：

```python
def compute_advantages(rewards, values, gamma=0.99, lam=0.95, last_value=0.0):
    advantages = [0.0] * len(rewards)
    gae = 0.0
    for t in reversed(range(len(rewards))):
        next_v = values[t + 1] if t + 1 < len(values) else last_value
        delta = rewards[t] + gamma * next_v - values[t]
        gae = delta + gamma * lam * gae
        advantages[t] = gae
    returns = [a + v for a, v in zip(advantages, values)]
    return advantages, returns
```

`returns` 是 critic target。`advantages` 是要乘以 `∇ log π` 的量。

### 第 3 步：组合 update

```python
for step_i, (x, a, _r, probs) in enumerate(traj):
    adv = advantages[step_i]
    target_v = returns[step_i]

    # critic
    critic_update(w, x, target_v, lr_v)

    # actor
    for i in range(N_ACTIONS):
        grad_logpi = (1.0 if i == a else 0.0) - probs[i]
        for j in range(N_FEAT):
            theta[i][j] += lr_a * adv * grad_logpi * x[j]
```

On-policy，每个 rollout 更新一次，actor 和 critic 使用分开的 learning rate。

### 第 4 步：并行化（A3C vs A2C）

- **A3C：** 启动 `N` 个线程。每个线程运行自己的 env 和 forward pass。周期性把梯度更新推送到共享 master。master 上不加锁：竞态没关系，只是增加噪声。
- **A2C：** 在单进程中运行 `N` 个 env instance，把 observation 堆成 `[N, obs_dim]` batch，batched forward pass，batched backward pass。GPU 利用率更高、确定性更强，也更容易推理。2026 年默认这样做。

我们的玩具代码为了清晰是单线程的；改写成 batched A2C 只需要三行 numpy。

## 常见坑

- **Actor gradient 前的 critic bias。** 如果 critic 是随机的，它的 baseline 没有信息量，你就是在纯噪声上训练。先 warm up critic 几百步，再打开 policy gradient，或者使用较慢的 actor learning rate。
- **Advantage normalization。** 每个 batch 内把 advantage 归一化到零均值/单位标准差。几乎零成本，却能极大稳定训练。
- **Shared trunk。** 对图像输入，actor 和 critic 使用共享 feature extractor。头部分开。共享特征能同时从两个 loss 中受益。
- **On-policy contract。** A2C 对数据只复用一次。更多次会让梯度有偏（importance-sampling correction 正是 PPO 加入的东西）。
- **Entropy collapse。** 没有 `c_e > 0`，policy 会在几百次更新内变得近乎确定性并停止探索。
- **Reward scale。** Advantage 量级依赖奖励尺度。对奖励归一化（例如除以 running std），让不同任务上的梯度量级保持一致。

## 使用它

A2C/A3C 在 2026 年很少是最终选择，但它们是后续所有方法精炼的架构：

| 方法 | 与 A2C 的关系 |
|------|---------------|
| PPO | A2C + clipped importance ratio，用于 multi-epoch updates |
| IMPALA | A3C + V-trace off-policy correction |
| SAC（Phase 9 · 07） | 带 soft-value critic 的 off-policy A2C（下一课） |
| GRPO（Phase 9 · 12） | 没有 critic 的 A2C：group-relative advantage |
| DPO | A2C 折叠成 preference-ranking loss，无采样 |
| AlphaStar / OpenAI Five | A2C + league training + imitation pre-training |

如果你在 2026 年的论文里看到 “advantage”，就想到 actor-critic。

## 交付

保存为 `outputs/skill-actor-critic-trainer.md`：

```markdown
---
name: actor-critic-trainer
description: Produce an A2C / A3C / GAE configuration for a given environment, with advantage estimation and loss weights specified.
version: 1.0.0
phase: 9
lesson: 7
tags: [rl, actor-critic, gae]
---

Given an environment and compute budget, output:

1. Parallelism. A2C (GPU batched) vs A3C (CPU async) and the number of workers.
2. Rollout length T. Steps per env per update.
3. Advantage estimator. n-step or GAE(λ); specify λ.
4. Loss weights. `c_v` (value), `c_e` (entropy), gradient clip.
5. Learning rates. Actor and critic (separate if using).

Refuse single-worker A2C on environments with horizon > 1000 (too on-policy, too slow). Refuse to ship without advantage normalization. Flag any run with `c_e = 0` and observed entropy < 0.1 as entropy-collapsed.
```

## 练习

1. **简单。** 在 4×4 GridWorld 上用 MC advantage（`G_t - V(s_t)`）训练 actor-critic。和第 06 课中带 running-mean-baseline 的 REINFORCE 比较样本效率。
2. **中等。** 切换到 TD-residual advantage（`r + γ V(s') - V(s)`）。测量 advantage batch 的方差。下降了多少？
3. **困难。** 实现 GAE(λ)。扫描 `λ ∈ {0, 0.5, 0.9, 0.95, 1.0}`。绘制最终 return 与样本效率。这个任务上的 bias/variance 甜点在哪里？

## 关键术语

| 术语 | 人们通常怎么说 | 它实际的含义 |
|------|----------------|--------------|
| Actor | “policy net” | `π_θ(a|s)`，由 policy gradient 更新。 |
| Critic | “value net” | `V_φ(s)`，通过对 returns / TD targets 做 MSE regression 更新。 |
| Advantage | “比平均好多少” | `A(s, a) = Q(s, a) - V(s)` 或其估计量。`∇ log π` 的乘数。 |
| TD residual | “δ” | `δ_t = r + γ V(s') - V(s)`；一步 advantage estimate。 |
| GAE | “插值旋钮” | n-step advantages 的指数加权和，由 `λ` 参数化。 |
| A2C | “同步 actor-critic” | 跨 env 批处理；每个 rollout 做一个梯度步骤。 |
| A3C | “异步 actor-critic” | Worker 线程把梯度推送到共享参数服务器。原始论文方法；2026 年较少见。 |
| Bootstrap | “在 horizon 使用 V” | 截断 rollout，加入 `γ^n V(s_{t+n})` 来闭合求和。 |

## 延伸阅读

- [Mnih et al. (2016). Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1602.01783) — A3C，原始异步 actor-critic 论文。
- [Schulman et al. (2016). High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438) — GAE。
- [Sutton & Barto (2018). Ch. 13 — Actor-Critic Methods](http://incompleteideas.net/book/RLbook2020.pdf) — 基础内容；当 critic 是神经网络时，建议和第 9 章函数近似一起读。
- [Espeholt et al. (2018). IMPALA](https://arxiv.org/abs/1802.01561) — 带 V-trace off-policy correction 的可扩展 distributed actor-critic。
- [OpenAI Baselines / Stable-Baselines3](https://stable-baselines3.readthedocs.io/) — 值得阅读的生产级 A2C/PPO 实现。
- [Konda & Tsitsiklis (2000). Actor-Critic Algorithms](https://papers.nips.cc/paper/1786-actor-critic-algorithms) — 两时间尺度 actor-critic 分解的基础收敛性结果。
