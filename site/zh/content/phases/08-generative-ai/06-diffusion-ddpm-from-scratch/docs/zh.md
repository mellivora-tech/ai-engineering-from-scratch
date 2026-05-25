# Diffusion Models — 从零实现 DDPM

> Ho、Jain、Abbeel（2020）给了领域一个戒不掉的配方。用一千个小步骤用噪声摧毁数据。训练一个神经网络预测噪声。推理时反转这个过程。今天每个主流图像、视频、3D 和音乐模型都运行在这个循环上，可能再叠加 flow matching 或 consistency 技巧。

**类型：** 构建
**语言：** Python
**前置要求：** 阶段 3 · 02（Backprop），阶段 8 · 02（VAE）
**时间：** ~75 分钟

## 问题

你想要一个 `p_data(x)` 的 sampler。GANs 玩 minimax game，经常发散。VAEs 用 Gaussian decoder 产生模糊样本。你真正想要的是一个训练目标，满足：（a）单一稳定 loss（没有 saddle point，没有 minimax），（b）`log p(x)` 的 lower bound（所以你有 likelihoods），（c）样本匹配 SOTA 质量。

Sohl-Dickstein et al.（2015）有理论答案：定义一个 Markov chain `q(x_t | x_{t-1})`，逐渐加入 Gaussian noise，并训练一个 reverse chain `p_θ(x_{t-1} | x_t)` 来 denoise。Ho、Jain、Abbeel（2020）展示这个 loss 可以简化成一行 — 预测噪声 — 并整理了数学。2020 年它还只是好奇点。2021 年它产生 state-of-the-art samples。2022 年它成为 Stable Diffusion。2026 年它是底座。

## 概念

![DDPM: forward noise, reverse denoise](../assets/ddpm.svg)

**Forward process `q`。** 在 `T` 个小步骤中加入 Gaussian noise。Closed form — 数学可处理的原因 — 是 cumulative step 也仍然是 Gaussian：

```
q(x_t | x_0) = N( sqrt(α̅_t) · x_0,  (1 - α̅_t) · I )
```

其中 `α̅_t = ∏_{s=1..t} (1 - β_s)`，`β_t` 来自某个 schedule。把 `β_t` 在 T=1000 steps 上从 1e-4 线性取到 0.02，`x_T` 近似为 `N(0, I)`。

**Reverse process `p_θ`。** 学习一个神经网络 `ε_θ(x_t, t)`，预测加入的噪声。给定 `x_t`，按如下方式 denoise：

```
x_{t-1} = (1 / sqrt(α_t)) · ( x_t - (β_t / sqrt(1 - α̅_t)) · ε_θ(x_t, t) )  +  σ_t · z
```

其中 `σ_t` 要么是 `sqrt(β_t)`，要么是 learned variance。表达式很丑，但只是代数 — 基于 posterior `q(x_{t-1} | x_t, x_0)` 解出 `x_{t-1}`，再用噪声预测估计替换 `x_0`。

**Training loss。**

```
L_simple = E_{x_0, t, ε} [ || ε - ε_θ( sqrt(α̅_t) · x_0 + sqrt(1 - α̅_t) · ε,  t ) ||² ]
```

从数据采样 `x_0`，随机选择 `t`，采样 `ε ~ N(0, I)`，用 closed form 一次性算出 noisy `x_t`，然后回归噪声。一个 loss，没有 minimax，没有 KL，没有 reparameterization tricks。

**Sampling。** 从 `x_T ~ N(0, I)` 开始。从 `t = T` 到 `1` 迭代 reverse step。完成。

## 为什么有效

三个直觉：

1. **Denoising 容易；generation 难。** 在 `t=T`，数据是纯噪声 — net 要解的是平凡问题。在 `t=0`，net 只需要清理少量像素。中间 `t` 很难，但 net 从每个噪声级别通过同一组权重获得许多梯度。

2. **伪装成别的 score matching。** Vincent（2011）证明，预测噪声等价于估计 `∇_x log q(x_t | x_0)`，也就是 *score*。Reverse SDE 使用这个 score 沿密度梯度上行 — 一次朝高概率区域移动的 guided random walk。

3. **ELBO 简化成 MSE。** 完整 variational lower bound 每个 timestep 有一个 KL 项。用 DDPM 的 parameterization，这些 KL 项会简化成带特定系数的 noise prediction MSE；Ho 去掉了系数（称为 “simple” loss），质量反而 *提升*。

## 构建它

`code/main.py` 实现一个 1-D DDPM。数据是双峰 mixture。“Net” 是 tiny MLP，接收 `(x_t, t)` 并输出 predicted noise。训练是一行 loss。Sampling 迭代 reverse chain。

### 第 1 步：forward schedule（closed form）

```python
betas = [1e-4 + (0.02 - 1e-4) * t / (T - 1) for t in range(T)]
alphas = [1 - b for b in betas]
alpha_bars = []
cum = 1.0
for a in alphas:
    cum *= a
    alpha_bars.append(cum)
```

### 第 2 步：一次性采样 `x_t`

```python
def forward_sample(x0, t, alpha_bars, rng):
    a_bar = alpha_bars[t]
    eps = rng.gauss(0, 1)
    x_t = math.sqrt(a_bar) * x0 + math.sqrt(1 - a_bar) * eps
    return x_t, eps
```

### 第 3 步：一个 training step

```python
def train_step(x0, model, alpha_bars, rng):
    t = rng.randrange(T)
    x_t, eps = forward_sample(x0, t, alpha_bars, rng)
    eps_hat = model_forward(model, x_t, t)
    loss = (eps - eps_hat) ** 2
    return loss, gradient_step(model, ...)
```

### 第 4 步：reverse sampling

```python
def sample(model, alpha_bars, T, rng):
    x = rng.gauss(0, 1)
    for t in range(T - 1, -1, -1):
        eps_hat = model_forward(model, x, t)
        beta_t = 1 - alphas[t]
        x = (x - beta_t / math.sqrt(1 - alpha_bars[t]) * eps_hat) / math.sqrt(alphas[t])
        if t > 0:
            x += math.sqrt(beta_t) * rng.gauss(0, 1)
    return x
```

对于 40 timesteps 和 24-unit MLP 的 1-D 问题，它大约 200 epochs 学会双峰 mixture。

## Time conditioning

Net 需要知道自己正在 denoise 哪个 timestep。两个标准选项：

- **Sinusoidal embedding。** 类似 Transformer positional encoding。`embed(t) = [sin(t/ω_0), cos(t/ω_0), sin(t/ω_1), ...]`。通过 MLP 后 broadcast 到 net。
- **Film / group-norm conditioning。** 在每个 block 上把 embedding 投影为 per-channel scale/bias（FiLM）。

我们的 toy code 使用 sinusoidal → concat。生产 U-Nets 使用 FiLM。

## 陷阱

- **Schedule 非常重要。** Linear `β` 是 DDPM 默认值，但 cosine schedule（Nichol & Dhariwal, 2021）在同等 compute 下给出更好 FID。如果质量平台化，切换 schedules。
- **Timestep embedding 很脆弱。** 对 toy 1-D 直接传 raw `t` float 可行，但图像会失败；总是使用 proper embedding。
- **V-prediction vs ε-prediction。** 在非常小或非常大的 t 这种窄 regime 中，`ε` 的 signal-to-noise 很差。V-prediction（`v = α·ε - σ·x`）更稳定；SDXL、SD3、Flux 都使用它。
- **Classifier-free guidance。** 推理时同时计算 conditional 和 unconditional `ε`，然后 `ε_cfg = (1 + w) · ε_cond - w · ε_uncond`，其中 `w ≈ 3-7`。第 08 课覆盖。
- **1000 steps 很多。** 生产使用 DDIM（20-50 steps）、DPM-Solver（10-20 steps）或 distillation（1-4 steps）。见第 12 课。

## 使用它

| 角色 | 2026 年典型 stack |
|------|-----------------------|
| Image pixel-space diffusion（小型、玩具） | DDPM + U-Net |
| Image latent diffusion | VAE encoder + U-Net 或 DiT（第 07 课） |
| Video latent diffusion | Spatiotemporal DiT（Sora、Veo、WAN） |
| Audio latent diffusion | Encodec + diffusion transformer |
| Science（molecules、proteins、physics） | Equivariant diffusion（EDM、RFdiffusion、AlphaFold3） |

Diffusion 是通用生成 backbone。Flow matching（第 13 课）是 2024–2026 的竞争者，通常在同质量下赢得推理速度。

## 交付它

保存 `outputs/skill-diffusion-trainer.md`。Skill 接收 dataset + compute budget，并输出：schedule（linear/cosine/sigmoid）、prediction target（ε/v/x）、steps 数、guidance scale、sampler family 和 eval protocol。

## 练习

1. **简单。** 把 `code/main.py` 中 T 从 40 改到 10。样本质量（输出 visual histogram）如何下降？T 到多少时双峰结构崩塌？
2. **中等。** 从 ε-prediction 切到 v-prediction。重新推导 reverse step。比较最终 sample quality。
3. **困难。** 加入 classifier-free guidance。以 class label `c ∈ {0, 1}` 为 condition，训练时有 10% 时间 drop 掉它，采样时使用 `ε = (1+w)·ε_cond - w·ε_uncond`。在 `w = 0, 1, 3, 7` 下测量 conditional-mode-hit rate。

## 关键词

| 术语 | 大家怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| Forward process | “加噪” | 固定 Markov chain `q(x_t | x_{t-1})`，摧毁数据。 |
| Reverse process | “Denoising” | 学习到的 chain `p_θ(x_{t-1} | x_t)`，重建数据。 |
| β schedule | “噪声阶梯” | 每步 variance；linear、cosine 或 sigmoid。 |
| α̅ | “Alpha bar” | 累积乘积 `∏(1 - β)`；给出从 `x_0` 到 `x_t` 的 closed-form。 |
| Simple loss | “噪声 MSE” | `||ε - ε_θ(x_t, t)||²`；所有 variational 推导都塌缩到这里。 |
| ε-prediction | “预测噪声” | 输出是被加入的噪声；标准 DDPM。 |
| V-prediction | “预测 velocity” | 输出是 `α·ε - σ·x`；跨 t 的 conditioning 更好。 |
| DDPM | “那篇论文” | Ho et al. 2020；linear β、1000 steps、U-Net。 |
| DDIM | “Deterministic sampler” | Non-Markov sampler，20-50 steps，同一训练目标。 |
| Classifier-free guidance | “CFG” | 混合 conditional 和 unconditional noise predictions 来放大 conditioning。 |

## 生产备注：diffusion inference 是 step-count 问题

DDPM 论文运行 T=1000 reverse steps。没人把这个用于生产。每个真实推理栈都会选择三种策略之一 — 每种都能清楚映射到“延迟来自哪里”的生产框架：

1. **更快 sampler，同一模型。** DDIM（20-50 steps）、DPM-Solver++（10-20）、UniPC（8-16）。替换 reverse loop；训练好的 `ε_θ` 权重不变。延迟降低 20-50×。
2. **Distillation。** 训练 student 用更少 steps 匹配 teacher：Progressive Distillation（2 → 1）、Consistency Models（任意 → 1-4）、LCM、SDXL-Turbo、SD3-Turbo。延迟再降 5-10×，需要重新训练。
3. **Caching and compilation。** `torch.compile(unet, mode="reduce-overhead")`、TensorRT-LLM diffusion backends、`xformers`/SDPA attention、bf16 weights。每步延迟约降 2×。可与（1）和（2）叠加。

对生产 diffusion server 来说，预算讨论和 LLM 生产文献相同：latency 是 `num_steps × step_cost + VAE_decode`，throughput 是 `batch_size × (num_steps × step_cost)^-1`。TTFT 很小（一步）；TPOT-equivalent 是完整 response time，因为从用户视角，image generation 是 “all-at-once”。

## 延伸阅读

- [Sohl-Dickstein et al. (2015). Deep Unsupervised Learning using Nonequilibrium Thermodynamics](https://arxiv.org/abs/1503.03585) — diffusion 论文，超前于时代。
- [Ho, Jain, Abbeel (2020). Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) — DDPM。
- [Song, Meng, Ermon (2021). Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502) — DDIM，更少 steps。
- [Nichol & Dhariwal (2021). Improved DDPM](https://arxiv.org/abs/2102.09672) — cosine schedule、learned variance。
- [Dhariwal & Nichol (2021). Diffusion Models Beat GANs on Image Synthesis](https://arxiv.org/abs/2105.05233) — classifier guidance。
- [Ho & Salimans (2022). Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598) — CFG。
- [Karras et al. (2022). Elucidating the Design Space of Diffusion-Based Generative Models (EDM)](https://arxiv.org/abs/2206.00364) — unified notation，最干净 recipe。
