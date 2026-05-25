# Flow Matching 与 Rectified Flows

> Diffusion models 需要 20-50 个 sampling steps，因为它们从噪声到数据走的是弯曲路径。Flow matching（Lipman et al., 2023）和 rectified flow（Liu et al., 2022）训练的是直线路径。路径越直，步骤越少，推理越快。Stable Diffusion 3、Flux.1 和 AudioCraft 2 都在 2024 年切换到了 flow matching。

**类型：** 构建
**语言：** Python
**前置要求：** 阶段 8 · 06（DDPM），阶段 1 · Calculus
**时间：** ~45 分钟

## 问题

DDPM 的 reverse process 是从 `N(0, I)` 回到数据分布的 1000-step stochastic walk。DDIM 把它压到 20-50 个 deterministic steps。你想要更少步骤 — 最好一步。阻碍是解 reverse process 的 ODE 很 stiff；路径是弯的。

如果你能训练模型，让从噪声到数据的路径是一条 *直线*，从 `t=1` 到 `t=0` 做单个 Euler step 就能工作。Flow matching 直接构建这个目标：定义从 `x_1 ∼ N(0, I)` 到 `x_0 ∼ data` 的直线插值，训练 vector field `v_θ(x, t)` 去匹配其时间导数，推理时积分。

Rectified flow（Liu 2022）更进一步：通过 reflow 过程迭代拉直路径，产生越来越接近线性的 ODE。两次 reflow iteration 后，一个 2-step sampler 就能匹配 50-step DDPM 的质量。

## 概念

![Flow matching: straight-line interpolation between noise and data](../assets/flow-matching.svg)

### Straight-line flow

定义：

```
x_t = t · x_1 + (1 - t) · x_0,   t ∈ [0, 1]
```

其中 `x_0 ~ data`，`x_1 ~ N(0, I)`。沿这条直线的时间导数是常量：

```
dx_t / dt = x_1 - x_0
```

定义 neural vector field `v_θ(x_t, t)`，并训练它匹配这个导数：

```
L = E_{x_0, x_1, t} || v_θ(x_t, t) - (x_1 - x_0) ||²
```

这就是 **conditional flow matching** loss（Lipman 2023）。训练是 simulation-free：你从不 unroll ODE。只采样 `(x_0, x_1, t)` 并回归。

### Sampling

推理时，沿时间 *反向* 积分 learned vector field：

```
x_{t-Δt} = x_t - Δt · v_θ(x_t, t)
```

从 `x_1 ~ N(0, I)` 开始，用 Euler step 走到 `t=0`。

### Rectified flow（Liu 2022）

Straight-line flow 有效，但 learned paths *实际上并不直* — 它们会弯曲，因为许多 `x_0` 可能映射到同一个 `x_1`。Rectified flow 的 reflow step：

1. 用随机 pairings 训练 flow model v_1。
2. 通过从 `x_1` 积分到落点 `x_0`，采样 N 对 `(x_1, x_0)`。
3. 在这些 paired examples 上训练 v_2。因为 pairs 现在是 “ODE-matched”，它们之间的 straight-line interpolant 真正更平。
4. 重复。

实践中，2 次 reflow iterations 就能接近线性，实现 2-4 step inference。SDXL-Turbo、SD3-Turbo、LCM 都是从 flow-matching models distilled 而来。

### 为什么它在 2024 年赢下图像

三个原因：

1. **Simulation-free training** — 训练时没有 ODE unrolling，实现极简单。
2. **更好的 loss geometry** — 直线路径有一致 signal-to-noise，而 DDPM ε-loss 在 schedule 两端 SNR 很差。
3. **更快推理** — 4-8 steps 达到 SDXL-Turbo 质量；consistency distillation 后 1 step。

## Flow matching vs DDPM — 精确连接

带 Gaussian-conditional path 的 flow matching 就是使用特定 noise schedule 的 diffusion。选择 `x_t = α(t) x_0 + σ(t) x_1` schedule，flow matching 会恢复 Stratonovich-reformulated diffusion，其中 `v = α'·x_0 - σ'·x_1`。对 Gaussian paths，两者代数等价。

Flow matching 增加的是：目标的 *清晰性*（plain velocity）、更干净的 loss，以及实验 non-Gaussian interpolants 的许可。

## 构建它

`code/main.py` 在双峰 Gaussian mixture 上实现 1-D flow matching。Vector field `v_θ(x, t)` 是 tiny MLP，用 straight-line target 训练。推理时用 1、2、4、20 个 Euler steps 积分并比较样本质量。

### 第 1 步：training loss

```python
def train_step(x0, net, rng, lr):
    x1 = rng.gauss(0, 1)
    t = rng.random()
    x_t = t * x1 + (1 - t) * x0
    target = x1 - x0
    pred = net_forward(x_t, t)
    loss = (pred - target) ** 2
    # backprop + update
```

### 第 2 步：multi-step inference

```python
def sample(net, num_steps):
    x = rng.gauss(0, 1)
    for i in range(num_steps):
        t = 1.0 - i / num_steps
        dt = 1.0 / num_steps
        x -= dt * net_forward(x, t)
    return x
```

### 第 3 步：比较 step counts

预期 4-step sampler 已经匹配 20-step 质量 — 这对延迟意义很大。

## 陷阱

- **Time parameterization。** Flow matching 使用 `t ∈ [0, 1]`，`t=0` 是数据，`t=1` 是噪声。DDPM 使用 `t ∈ [0, T]`，`t=0` 是数据，`t=T` 是噪声。同一方向，不同尺度。论文经常写错。
- **Schedule choice。** Rectified flow 的直线是 “the” flow-matching schedule，但你可以使用 cosine 或 logit-normal t-sampling（SD3 使用）来更好覆盖 scale。
- **Reflow cost。** 为 reflow 生成 paired dataset，是对每个 sample 做一次完整 inference pass。只有真的需要 1-2 step inference 时才做。
- **Classifier-free guidance 仍然适用。** 只要把 ε 换成 v 做线性组合：`v_cfg = (1+w) v_cond - w v_uncond`。

## 使用它

| 用例 | 2026 stack |
|----------|-----------|
| Text-to-image，最佳质量 | Flow matching：SD3、Flux.1-dev |
| Text-to-image，1-4 steps | Distilled flow matching：Flux.1-schnell、SD3-Turbo、SDXL-Turbo |
| Real-time inference | 从 flow-matched base 做 consistency distillation（LCM、PCM） |
| Audio generation | Flow matching：Stable Audio 2.5、AudioCraft 2 |
| Video generation | Flow matching 混合 diffusion（Sora、Veo、Stable Video） |
| Science / physics（particle trajectories、molecules） | Flow matching + equivariant vector field |

每当 2025-2026 年论文说 “faster than diffusion”，它几乎总是 flow matching + distillation。

## 交付它

保存 `outputs/skill-fm-tuner.md`。Skill 接收 diffusion-style model spec，并把它转换成 flow-matching training config：schedule choice、time sampling distribution（uniform / logit-normal）、optimizer、reflow plan、target step count、eval protocol。

## 练习

1. **简单。** 运行 `code/main.py`，比较 1-step vs 20-step 相对真实数据分布的 MSE。
2. **中等。** 从 uniform `t` sampling 切换到 logit-normal（把采样集中在 mid-t）。模型质量会提升吗？
3. **困难。** 实现一次 reflow iteration：通过积分第一个模型生成 paired (x_0, x_1)，在 pairs 上训练第二个模型，并比较 1-step sample quality。

## 关键词

| 术语 | 大家怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| Flow matching | “Straight-line diffusion” | 训练 `v_θ(x, t)`，使其沿 interpolant 匹配 `x_1 - x_0`。 |
| Rectified flow | “Reflow” | 拉直 learned flows 的迭代过程。 |
| Velocity field | “v_θ” | 模型输出 — `x_t` 应该移动的方向。 |
| Straight-line interpolant | “The path” | `x_t = (1-t)·x_0 + t·x_1`；target derivative 平凡。 |
| Euler sampler | “1st order ODE solver” | 最简单 integrator；路径直时效果很好。 |
| Logit-normal t | “SD3 sampling” | 把 `t` sampling 集中到 gradient 最强的 mid-values。 |
| Consistency distillation | “1-step sampler” | 训练 student，把任意 `x_t` 直接映射到 `x_0`。 |
| CFG with velocity | “v-CFG” | `v_cfg = (1+w) v_cond - w v_uncond`；同一个技巧，新变量。 |

## 生产备注：Flux.1-schnell 是最快的 flow matching

Flow matching 的生产胜利是 Flux.1-schnell — flow-matched DiT 被 distilled 到 1-4 inference steps，同时保持 Flux-dev 级质量。Niels 的 “Run Flux on an 8GB machine” notebook 是参考部署配方：T5 + CLIP encode，quantized MMDiT denoise（schnell 4 steps，而 dev 50 steps），VAE decode。成本账：

| 变体 | Steps | L4 上 1024² latency | Total FLOPs (relative) |
|---------|-------|------------------------|------------------------|
| Flux.1-dev (raw) | 50 | ~15 s | 1.0× |
| Flux.1-schnell | 4 | ~1.2 s | 0.08×（快 12×） |
| SDXL-base | 30 | ~4 s | 0.25× |
| SDXL-Lightning 2-step | 2 | ~0.3 s | 0.03× |

生产规则：**flow-matched base + distillation = 2026 年 fast text-to-image 默认。** 每个主要 vendor 都提供这个组合：SD3-Turbo（SD3 + flow + distillation）、Flux-schnell（Flux-dev + rectified-flow straightening）、CogView-4-Flash。纯 diffusion bases 只存在于 legacy checkpoints。

## 延伸阅读

- [Liu, Gong, Liu (2022). Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow](https://arxiv.org/abs/2209.03003) — rectified flow。
- [Lipman et al. (2023). Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747) — flow matching。
- [Esser et al. (2024). Scaling Rectified Flow Transformers for High-Resolution Image Synthesis](https://arxiv.org/abs/2403.03206) — SD3，大规模 rectified flow。
- [Albergo, Vanden-Eijnden (2023). Stochastic Interpolants](https://arxiv.org/abs/2303.08797) — 覆盖 FM + diffusion 的通用框架。
- [Song et al. (2023). Consistency Models](https://arxiv.org/abs/2303.01469) — diffusion / flow 的 1-step distillation。
- [Sauer et al. (2023). Adversarial Diffusion Distillation (SDXL-Turbo)](https://arxiv.org/abs/2311.17042) — turbo 变体。
- [Black Forest Labs (2024). Flux.1 models](https://blackforestlabs.ai/announcing-black-forest-labs/) — 生产中的 flow matching。
