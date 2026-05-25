# Deep Q-Networks (DQN)

> 2013 年：Mnih 用原始像素训练了一个 Q-learning 网络，在 7 个 Atari 游戏上击败了所有经典 RL agent。2015 年：扩展到 49 个游戏，发表于 Nature，点燃了 deep-RL 时代。DQN 就是 Q-learning 加上三个让函数近似稳定下来的技巧。

**类型：** 构建
**语言：** Python
**先修：** Phase 3 · 03（Backpropagation），Phase 9 · 04（Q-learning，SARSA）
**时间：** 约 75 分钟

## 问题

表格型 Q-learning 需要为每个 `(state, action)` 对保存一个独立的 Q-value。棋盘大约有 10⁴³ 个状态。一个 Atari 帧是 210×160×3 = 100,800 个特征。表格型 RL 在几千个状态时就开始吃不消，更不用说数十亿个状态。

事后看来，修复方式很明显：把 Q-table 换成神经网络 `Q(s, a; θ)`。但这种“事后明显”花了几十年才变得可行。朴素地把函数近似接到 Q-learning 上，会因为“deadly triad”而发散：函数近似 + bootstrapping + off-policy learning。Mnih 等人（2013、2015）找到了三个能稳定学习的工程技巧：

1. **Experience replay** 打散 transition 之间的相关性。
2. **Target network** 冻结 bootstrap target。
3. **Reward clipping** 归一化梯度量级。

Atari 上的 DQN 是第一次用单一架构和单一超参数集合，从原始像素解决几十个控制问题。此后一切“deep-RL”工作，包括 DDQN、Rainbow、Dueling、Distributional、R2D2、Agent57，都是叠在这组三技巧底座之上。

## 核心概念

![DQN training loop: env, replay buffer, online net, target net, Bellman TD loss](../assets/dqn.svg)

**目标函数。** DQN 在神经 Q 函数上最小化一步 TD loss：

`L(θ) = E_{(s,a,r,s')~D} [ (r + γ max_{a'} Q(s', a'; θ^-) - Q(s, a; θ))² ]`

`θ` = online network，每一步通过梯度下降更新。`θ^-` = target network，周期性从 `θ` 复制而来（大约每 10,000 步）。`D` = 存放过去 transition 的 replay buffer。

**三个技巧，按重要性排序：**

**Experience replay。** 一个容量约为 `~10⁶` 的 transition 环形缓冲区。每个训练步骤都从中均匀随机采样一个 minibatch。这会打破时间相关性（连续帧几乎相同），让网络可以多次学习稀有的高奖励 transition，并去相关连续的梯度更新。没有它，带神经网络的 on-policy TD 在 Atari 上会发散。

**Target network。** 在 Bellman 方程两边都使用同一个网络 `Q(·; θ)`，会让 target 在每次更新后移动，也就是“追着自己的尾巴跑”。修复方式：保留第二个网络 `Q(·; θ^-)`，其权重冻结。每隔 `C` 步复制一次 `θ → θ^-`。这样回归 target 会在数千个梯度步骤内保持稳定。Soft update `θ^- ← τ θ + (1-τ) θ^-`（用于 DDPG、SAC）是更平滑的变体。

**Reward clipping。** Atari 的奖励量级从 1 到 1000+ 不等。裁剪到 `{-1, 0, +1}` 可以阻止某个游戏独占梯度。当奖励幅度本身重要时这会出错；但在 Atari 中通常只有符号重要，所以可以接受。

**Double DQN。** Hasselt（2016）修复了 maximization bias：用 online net *选择* 动作，用 target net *评估* 这个动作。

`target = r + γ Q(s', argmax_{a'} Q(s', a'; θ); θ^-)`

这是即插即用的替换，通常稳定更好。默认就用它。

**其他改进（Rainbow，2017）：** prioritized replay（更多采样高 TD-error transition）、dueling architecture（拆分 `V(s)` 和 advantage head）、noisy networks（可学习探索）、n-step returns、distributional Q（C51/QR-DQN）、multi-step bootstrapping。每个技巧带来几个百分点的收益；收益大致可叠加。

## 动手构建

这里的代码只用标准库，不用 numpy：我们在一个很小的连续 GridWorld 上手写单隐藏层 MLP，所以每个训练步骤都在微秒级完成。算法形状和大规模 Atari DQN 完全一致。

### 第 1 步：replay buffer

```python
class ReplayBuffer:
    def __init__(self, capacity):
        self.buf = []
        self.capacity = capacity
    def push(self, s, a, r, s_next, done):
        if len(self.buf) == self.capacity:
            self.buf.pop(0)
        self.buf.append((s, a, r, s_next, done))
    def sample(self, batch, rng):
        return rng.sample(self.buf, batch)
```

Atari 通常用约 50,000 的容量；我们的玩具环境 5,000 就够。

### 第 2 步：一个很小的 Q-network（手写 MLP）

```python
class QNet:
    def __init__(self, n_in, n_hidden, n_actions, rng):
        self.W1 = [[rng.gauss(0, 0.3) for _ in range(n_in)] for _ in range(n_hidden)]
        self.b1 = [0.0] * n_hidden
        self.W2 = [[rng.gauss(0, 0.3) for _ in range(n_hidden)] for _ in range(n_actions)]
        self.b2 = [0.0] * n_actions
    def forward(self, x):
        h = [max(0.0, sum(w * xi for w, xi in zip(row, x)) + b) for row, b in zip(self.W1, self.b1)]
        q = [sum(w * hi for w, hi in zip(row, h)) + b for row, b in zip(self.W2, self.b2)]
        return q, h
```

Forward pass：linear → ReLU → linear。这就是整个网络。

### 第 3 步：DQN update

```python
def train_step(online, target, batch, gamma, lr):
    grads = zeros_like(online)
    for s, a, r, s_next, done in batch:
        q, h = online.forward(s)
        if done:
            y = r
        else:
            q_next, _ = target.forward(s_next)
            y = r + gamma * max(q_next)
        td_error = q[a] - y
        accumulate_grads(grads, online, s, h, a, td_error)
    apply_sgd(online, grads, lr / len(batch))
```

形状和第 04 课的 Q-learning 一样，只有两点不同：（a）我们通过可微的 `Q(·; θ)` 反向传播，而不是索引表；（b）target 使用 `Q(·; θ^-)`。

### 第 4 步：外层循环

每个 episode 中，根据 `Q(·; θ)` 做 ε-greedy 行动，把 transition 推入 buffer，采样 minibatch，执行梯度步骤，并周期性同步 `θ^- ← θ`。模式如下：

```python
for episode in range(N):
    s = env.reset()
    while not done:
        a = epsilon_greedy(online, s, epsilon)
        s_next, r, done = env.step(s, a)
        buffer.push(s, a, r, s_next, done)
        if len(buffer) >= batch:
            train_step(online, target, buffer.sample(batch), gamma, lr)
        if steps % sync_every == 0:
            target = copy(online)
        s = s_next
```

在我们这个 16 维 one-hot 状态的小 GridWorld 中，agent 大约 500 个 episode 就能学到接近最优的 policy。在 Atari 上，需要把它扩展到 200M frames，并加入 CNN feature extractor。

## 常见坑

- **Deadly triad。** 函数近似 + off-policy + bootstrapping 会发散。DQN 用 target net + replay 缓解它；不要移除任何一个。
- **探索。** ε 必须衰减，通常在训练前约 10% 的步数里从 1.0 衰减到 0.01。早期探索不足会让 Q-net 收敛到局部盆地。
- **高估。** 对有噪声的 Q 取 `max` 会向上偏。生产环境始终用 Double DQN。
- **奖励尺度。** 裁剪或归一化奖励；梯度量级与奖励量级成正比。
- **Replay buffer 冷启动。** 不要在 buffer 只有少量 transition 时训练。基于约 20 个样本的早期梯度会过拟合。
- **Target sync 频率。** 太频繁 ≈ 没有 target net；太不频繁 ≈ target 过旧。Atari DQN 使用 10,000 个环境步。经验法则：每约训练 horizon 的 1/100 同步一次。
- **Observation preprocessing。** Atari DQN 堆叠 4 帧，使 state 满足 Markov 性。任何包含速度信息的环境都需要 frame-stacking 或 recurrent state。

## 使用它

到 2026 年，DQN 很少再是 state-of-the-art，但仍是参考级 off-policy 算法：

| 任务 | 首选方法 | 为什么不用 DQN？ |
|------|----------|------------------|
| 离散动作 Atari-like | Rainbow DQN or Muesli | 同一框架，更多技巧。 |
| 连续控制 | SAC / TD3（Phase 9 · 07） | DQN 没有 policy network。 |
| On-policy / 高吞吐 | PPO（Phase 9 · 08） | 没有 replay buffer；更容易扩展。 |
| Offline RL | CQL / IQL / Decision Transformer | 保守 Q target，避免 bootstrapping 爆炸。 |
| 大离散动作空间（推荐系统） | DQN with action embedding, or IMPALA | 可以；细节装饰很重要。 |
| LLM RL | PPO / GRPO | 序列级，不是 step-level；loss 不同。 |

这些经验仍会迁移。Replay 和 target network 出现在 SAC、TD3、DDPG、SAC-X、AlphaZero 的 self-play buffer，以及每一种 offline RL 方法中。Reward clipping 在 PPO 的 advantage normalization 中继续存在。这个架构就是蓝图。

## 交付

保存为 `outputs/skill-dqn-trainer.md`：

```markdown
---
name: dqn-trainer
description: Produce a DQN training config (buffer, target sync, ε schedule, reward clipping) for a discrete-action RL task.
version: 1.0.0
phase: 9
lesson: 5
tags: [rl, dqn, deep-rl]
---

Given a discrete-action environment (observation shape, action count, horizon, reward scale), output:

1. Network. Architecture (MLP / CNN / Transformer), feature dim, depth.
2. Replay buffer. Capacity, minibatch size, warmup size.
3. Target network. Sync strategy (hard every C steps or soft τ).
4. Exploration. ε start / end / schedule length.
5. Loss. Huber vs MSE, gradient clip value, reward clipping rule.
6. Double DQN. On by default unless explicit reason to disable.

Refuse to ship a DQN with no target network, no replay buffer, or ε held at 1. Refuse continuous-action tasks (route to SAC / TD3). Flag any reward range > 10× per-step mean as needing clipping or scale normalization.
```

## 练习

1. **简单。** 运行 `code/main.py`。绘制每个 episode 的 return 曲线。运行均值超过 -10 需要多少个 episode？
2. **中等。** 禁用 target network（Bellman target 两边都用 online net）。测量训练不稳定性：return 会振荡还是发散？
3. **困难。** 加入 Double DQN：用 online net 选择 `argmax a'`，target net 评估它。在带噪声奖励的 GridWorld 上，训练 1,000 个 episode 后，比较有无 Double DQN 时 `Q(s_0, best_a)` 相对真实 `V*(s_0)` 的偏差。

## 关键术语

| 术语 | 人们通常怎么说 | 它实际的含义 |
|------|----------------|--------------|
| DQN | “Deep Q-learning” | 带神经 Q 函数、replay buffer 和 target network 的 Q-learning。 |
| Experience replay | “打乱的 transitions” | 每个梯度步骤均匀采样的环形缓冲区；去相关数据。 |
| Target network | “冻结的 bootstrap” | Q 的周期性副本，用在 Bellman target 中；稳定训练。 |
| Deadly triad | “RL 为什么发散” | 函数近似 + bootstrapping + off-policy = 没有收敛保证。 |
| Double DQN | “maximization bias 的修复” | Online net 选择动作，target net 评估动作。 |
| Dueling DQN | “V 和 A head” | 分解 Q = V + A - mean(A)；输出相同，梯度流更好。 |
| Rainbow | “所有技巧” | DDQN + PER + dueling + n-step + noisy + distributional 合在一起。 |
| PER | “Prioritized Replay” | 按 TD-error 大小成比例采样 transition。 |

## 延伸阅读

- [Mnih et al. (2013). Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602) — 引爆 deep RL 的 2013 年 NeurIPS workshop 论文。
- [Mnih et al. (2015). Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236) — Nature 论文，49-game DQN。
- [Hasselt, Guez, Silver (2016). Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461) — DDQN。
- [Wang et al. (2016). Dueling Network Architectures](https://arxiv.org/abs/1511.06581) — dueling DQN。
- [Hessel et al. (2018). Rainbow: Combining Improvements in Deep RL](https://arxiv.org/abs/1710.02298) — stacked-tricks 论文。
- [OpenAI Spinning Up — DQN](https://spinningup.openai.com/en/latest/algorithms/dqn.html) — 清晰的现代讲解。
- [Sutton & Barto (2018). Ch. 9 — On-policy Prediction with Approximation](http://incompleteideas.net/book/RLbook2020.pdf) — 教科书中对“deadly triad”（函数近似 + bootstrapping + off-policy）的处理；DQN 的 target network 和 replay buffer 正是为驯服它而设计。
- [CleanRL DQN implementation](https://docs.cleanrl.dev/rl-algorithms/dqn/) — ablation study 中常用的单文件 DQN 参考实现；适合和本课的 from-scratch 版本对照阅读。
