# Proximal Policy Optimization (PPO)

> A2C 对每个 rollout 只更新一次就丢弃。PPO 用 clipped importance ratio 包住 policy gradient，让你可以在同一批数据上做 10+ 个 epoch，而不会让 policy 爆炸。Schulman 等人（2017）。到 2026 年，它仍是默认的 policy-gradient 算法。

**类型：** 构建
**语言：** Python
**先修：** Phase 9 · 06（REINFORCE），Phase 9 · 07（Actor-Critic）
**时间：** 约 75 分钟

## 问题

A2C（第 07 课）是 on-policy：梯度 `E_{π_θ}[A · ∇ log π_θ]` 要求数据采样自*当前* `π_θ`。更新一次后，`π_θ` 就变了；刚用过的数据现在已经 off-policy。复用它会让梯度有偏。

Rollout 很昂贵。在 Atari 上，跨 8 个 env × 128 步的一次 rollout = 1024 个 transition，还要十几秒的环境时间。一次梯度步骤后就把它丢掉很浪费。

Trust Region Policy Optimization（TRPO，Schulman 2015）是第一个修复：约束每次更新，使旧 policy 和新 policy 之间的 KL divergence 低于 `δ`。理论很干净，但每次更新都需要 conjugate-gradient 求解。2026 年没人再跑 TRPO。

PPO（Schulman et al. 2017）用一个简单的 clipped objective 替代硬 trust-region 约束。只多一行代码。每个 rollout 十个 epoch。不需要 conjugate gradients。理论保证足够好。九年后，它仍然是从 MuJoCo 到 RLHF 的默认 policy-gradient 算法。

## 核心概念

![PPO clipped surrogate objective: ratio clipping at 1 ± ε](../assets/ppo.svg)

**Importance ratio。**

`r_t(θ) = π_θ(a_t | s_t) / π_{θ_old}(a_t | s_t)`

这是新 policy 相对于采集数据的旧 policy 的 likelihood ratio。`r_t = 1` 表示没有变化。`r_t = 2` 表示新 policy 采取 `a_t` 的概率是旧 policy 的两倍。

**Clipped surrogate。**

`L^{CLIP}(θ) = E_t [ min( r_t(θ) A_t, clip(r_t(θ), 1-ε, 1+ε) A_t ) ]`

两个项：

- 如果 advantage `A_t > 0`，并且 ratio 试图增长到 `1 + ε` 之外，clip 会把梯度压平：不要把好动作的概率推到比旧概率高 `+ε` 以上。
- 如果 advantage `A_t < 0`，并且 ratio 试图越过 `1 - ε`（也就是相对于被裁剪后的降低，我们会让坏动作更可能发生），clip 会限制梯度：不要把坏动作推到低于 `-ε`。

`min` 处理另一个方向：如果 ratio 已经朝着*有利*方向移动，你仍会得到梯度（不会在会伤害你的那一侧裁剪）。

典型 `ε = 0.2`。把 objective 画成 `r_t` 的函数：它是一个分段线性函数，在“好的一侧”有平屋顶，在“坏的一侧”有平地板。

**完整 PPO loss。**

`L(θ, φ) = L^{CLIP}(θ) - c_v · (V_φ(s_t) - V_t^{target})² + c_e · H(π_θ(·|s_t))`

和 A2C 一样是 actor-critic 结构。三个系数通常是 `c_v = 0.5`、`c_e = 0.01`、`ε = 0.2`。

**训练循环。**

1. 在 `N` 个并行 env 中，每个收集 `T` 步，总共 `N × T` 个 transition。
2. 计算 advantages（GAE），并把它们冻结为常量。
3. 把当前 `π_θ` 快照成 `π_{θ_old}` 并冻结。
4. 对 `K` 个 epoch，对每个 `(s, a, A, V_target, log π_old(a|s))` 的 minibatch：
   - 计算 `r_t(θ) = exp(log π_θ(a|s) - log π_old(a|s))`。
   - 应用 `L^{CLIP}` + value loss + entropy。
   - 梯度步骤。
5. 丢弃 rollout。回到第 1 步。

`K = 10` 和 64 的 minibatch 是标准超参数组合。PPO 很稳健：具体数字在 ±50% 内通常影响不大。

**KL-penalty 变体。** 原论文提出了另一个版本：使用自适应 KL penalty：`L = L^{PG} - β · KL(π_θ || π_old)`，并根据观测到的 KL 调整 `β`。Clipping 版本成为主流；KL 版本在 RLHF 中保留下来（因为那里你本来就总想约束到 reference policy 的 KL）。

## 动手构建

### 第 1 步：rollout 时捕获 `log π_old(a | s)`

```python
for step in range(T):
    probs = softmax(logits(theta, state_features(s)))
    a = sample(probs, rng)
    s_next, r, done = env.step(s, a)
    buffer.append({
        "s": s, "a": a, "r": r, "done": done,
        "v_old": value(w, state_features(s)),
        "log_pi_old": log(probs[a] + 1e-12),
    })
    s = s_next
```

快照只在 rollout 时采集一次。它在更新 epoch 中不会改变。

### 第 2 步：计算 GAE advantages（第 07 课）

和 A2C 一样。跨 batch 归一化。

### 第 3 步：clipped surrogate update

```python
for _ in range(K_EPOCHS):
    for mb in minibatches(buffer, size=64):
        for rec in mb:
            x = state_features(rec["s"])
            probs = softmax(logits(theta, x))
            logp = log(probs[rec["a"]] + 1e-12)
            ratio = exp(logp - rec["log_pi_old"])
            adv = rec["advantage"]
            surrogate = min(
                ratio * adv,
                clamp(ratio, 1 - EPS, 1 + EPS) * adv,
            )
            # backprop -surrogate, add value loss, subtract entropy
            grad_logpi = onehot(rec["a"]) - probs
            if (adv > 0 and ratio >= 1 + EPS) or (adv < 0 and ratio <= 1 - EPS):
                pg_grad = 0.0  # clipped
            else:
                pg_grad = ratio * adv
            for i in range(N_ACTIONS):
                for j in range(N_FEAT):
                    theta[i][j] += LR * pg_grad * grad_logpi[i] * x[j]
```

“clipped → zero gradient” 模式是 PPO 的核心。如果新 policy 已经在有利方向上漂移太远，更新就会停止。

### 第 4 步：value 和 entropy

像 A2C 一样，对 critic target 加标准 MSE，对 actor 加 entropy bonus。

### 第 5 步：diagnostics

每次更新都要观察三件事：

- **Mean KL** `E[log π_old - log π_θ]`。应保持在 `[0, 0.02]`。如果冲过 `0.1`，降低 `K_EPOCHS` 或 `LR`。
- **Clip fraction**：ratio 落在 `[1-ε, 1+ε]` 之外的样本比例。应约为 `~0.1-0.3`。如果接近 `~0`，说明 clip 从未触发 → 提高 `LR` 或 `K_EPOCHS`。如果 `~0.5+`，说明你在过拟合 rollout → 降低它们。
- **Explained variance** `1 - Var(V_target - V_pred) / Var(V_target)`。Critic 质量指标。随着 critic 学习，应逐渐接近 1。

## 常见坑

- **Clip coefficient 调错。** `ε = 0.2` 是事实标准。降到 `0.1` 会让更新太胆小；`0.3+` 会引入不稳定。
- **Epoch 太多。** `K > 20` 经常不稳定，因为 policy 会远离 `π_old`。限制 epoch，尤其是大网络。
- **没有 reward normalization。** 大奖励尺度会吃掉 clip range。计算 advantages 前先归一化奖励（running std）。
- **忘记 advantage normalization。** 每个 batch 做零均值/单位标准差归一化是标准操作。跳过它会毁掉大多数 benchmark 上的 PPO。
- **Learning rate 没有 decay。** PPO 受益于线性 LR decay 到零。常数 LR 往往更差。
- **Importance ratio 数学错误。** 为了数值稳定，始终用 `exp(log_new - log_old)`，而不是 `new / old`。
- **梯度符号错误。** 最大化 surrogate = *最小化* `-L^{CLIP}`。符号翻转是最常见的 PPO bug。

## 使用它

PPO 是 2026 年很多领域默认的 RL 算法：

| 使用场景 | PPO 变体 |
|----------|----------|
| MuJoCo / robotics control | PPO with Gaussian policy, GAE(0.95) |
| Atari / 离散游戏 | PPO with categorical policy, rolling 128-step rollouts |
| LLM 的 RLHF | 带 reference model KL penalty 的 PPO，reward 来自 response 末尾的 RM |
| 大规模游戏 agent | IMPALA + PPO（AlphaStar，OpenAI Five） |
| Reasoning LLMs | GRPO（第 12 课）：无 critic 的 PPO 变体 |
| 仅偏好数据 | DPO：PPO+KL 的闭式折叠，无 online sampling |

PPO 的 *loss 形状*，也就是 clipped surrogate + value + entropy，是 DPO、GRPO 以及几乎每个 RLHF pipeline 的脚手架。

## 交付

保存为 `outputs/skill-ppo-trainer.md`：

```markdown
---
name: ppo-trainer
description: Produce a PPO training config and a diagnostic plan for a given environment.
version: 1.0.0
phase: 9
lesson: 8
tags: [rl, ppo, policy-gradient]
---

Given an environment and training budget, output:

1. Rollout size. `N` envs × `T` steps.
2. Update schedule. `K` epochs, minibatch size, LR schedule.
3. Surrogate params. `ε` (clip), `c_v`, `c_e`, advantage normalization on.
4. Advantage. GAE(`λ`) with explicit `γ` and `λ`.
5. Diagnostics plan. KL, clip fraction, explained variance thresholds with alerts.

Refuse `K > 30` or `ε > 0.3` (unsafe trust region). Refuse any PPO run without advantage normalization or KL/clip monitoring. Flag clip fraction sustained above 0.4 as drift.
```

## 练习

1. **简单。** 在 4×4 GridWorld 上运行 PPO，使用 `ε=0.2, K=4`。在 env steps 匹配的情况下，与 A2C（每个 rollout 一个 epoch）比较样本效率。
2. **中等。** 扫描 `K ∈ {1, 4, 10, 30}`。绘制 return vs env steps，并跟踪每次更新的 mean KL。这个任务上 `K` 到多少时 KL 会爆炸？
3. **困难。** 用 adaptive KL penalty 替换 clipped surrogate（如果 `KL > 2·target`，`β` 加倍；如果 `KL < target/2`，`β` 减半）。比较最终 return、稳定性和不依赖 clip 的程度。

## 关键术语

| 术语 | 人们通常怎么说 | 它实际的含义 |
|------|----------------|--------------|
| Importance ratio | “r_t(θ)” | `π_θ(a|s) / π_old(a|s)`；相对于采集数据的 policy 的偏离程度。 |
| Clipped surrogate | “PPO 的主要技巧” | `min(r·A, clip(r, 1-ε, 1+ε)·A)`；在有利侧越过 clip 后梯度变平。 |
| Trust region | “TRPO / PPO 的意图” | 限制每次更新的 KL，以保证单调改进。 |
| KL penalty | “Soft trust region” | 另一种 PPO：`L - β · KL(π_θ || π_old)`。自适应 `β`。 |
| Clip fraction | “clipping 触发频率” | 诊断指标，应为 0.1-0.3；超出说明调参不当。 |
| Multi-epoch training | “数据复用” | 每个 rollout 做 K 个 epoch；用方差成本换样本效率。 |
| On-policy-ish | “基本 on-policy” | PPO 名义上是 on-policy，但 K>1 个 epoch 会安全地使用略微 off-policy 的数据。 |
| PPO-KL | “另一种 PPO” | KL-penalty 变体；用于 RLHF，因为到 reference 的 KL 本来就是约束。 |

## 延伸阅读

- [Schulman et al. (2017). Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) — PPO 论文。
- [Schulman et al. (2015). Trust Region Policy Optimization](https://arxiv.org/abs/1502.05477) — TRPO，PPO 的前身。
- [Andrychowicz et al. (2021). What Matters In On-Policy RL? A Large-Scale Empirical Study](https://arxiv.org/abs/2006.05990) — 对每个 PPO 超参数做 ablation。
- [Ouyang et al. (2022). Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) — InstructGPT；RLHF 中的 PPO 配方。
- [OpenAI Spinning Up — PPO](https://spinningup.openai.com/en/latest/algorithms/ppo.html) — 带 PyTorch 的清晰现代讲解。
- [CleanRL PPO implementation](https://github.com/vwxyzjn/cleanrl) — 很多论文使用的单文件 PPO 参考实现。
- [Hugging Face TRL — PPOTrainer](https://huggingface.co/docs/trl/main/en/ppo_trainer) — 语言模型 PPO 的生产配方；建议和第 09 课（RLHF）一起读。
- [Engstrom et al. (2020). Implementation Matters in Deep Policy Gradients](https://arxiv.org/abs/2005.12729) — “37 个代码级优化”论文；哪些 PPO 技巧是真正承重的，哪些只是传说。
