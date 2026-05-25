# Autoencoders 与 Variational Autoencoders (VAE)

> 普通 autoencoder 先压缩再重建。它会记忆。它不会生成。加一个技巧 — 强迫 code 看起来像 Gaussian — 你就得到一个 sampler。这个单一技巧，也就是 `z = μ + σ·ε` 的 reparameterization，解释了为什么你在 2026 年使用的每个 latent-diffusion 和 flow-matching 图像模型，输入处都有一个 VAE。

**类型：** 构建
**语言：** Python
**前置要求：** 阶段 3 · 02（Backprop），阶段 3 · 07（CNNs），阶段 8 · 01（Taxonomy）
**时间：** ~75 分钟

## 问题

把 784 像素的 MNIST 数字压缩成 16 个数的 code，然后重建。普通 autoencoder 会在 reconstruction MSE 上拿高分，但 code space 会是一团疙瘩。从 code space 随机选一个点并 decode，你得到的是噪声。它没有 sampler。它只是穿着生成模型外衣的压缩模型。

你真正想要的是：（a）code space 是一个干净、平滑、可采样的分布 — 比如 isotropic Gaussian `N(0, I)`，（b）decode 任意 sample 都能产生 plausible digit，（c）encoder 和 decoder 仍然压缩得好。三个目标，一个架构，一个 loss。

Kingma 2013 年的 VAE 通过让 encoder 输出一个 *分布* `q(z|x) = N(μ(x), σ(x)²)` 来解决这件事，用 KL penalty 把该分布拉向 prior `N(0, I)`，然后在 decode 前从 `q(z|x)` 采样 `z`。推理时丢掉 encoder，采样 `z ~ N(0, I)`，decode。KL penalty 正是强迫 code space 有结构的东西。

2026 年 VAEs 很少独立上线 — raw image quality 上已经被 diffusion 超过 — 但它们是每个 latent-diffusion model（SD 1/2/XL/3、Flux、AudioCraft）的默认 encoder。学会 VAE，就学会了你使用的每条 image pipeline 的隐形第一层。

## 概念

![Autoencoder vs VAE: the reparameterization trick](../assets/vae.svg)

**Autoencoder。** `z = encoder(x)`，`x̂ = decoder(z)`，loss = `||x - x̂||²`。Code space 无结构。

**VAE encoder。** 输出两个向量：`μ(x)` 和 `log σ²(x)`。它们定义 `q(z|x) = N(μ, diag(σ²))`。

**Reparameterization trick。** 从 `q(z|x)` 采样不可微。把 sample 改写成 `z = μ + σ·ε`，其中 `ε ~ N(0, I)`。现在 `z` 是 `(μ, σ)` 加上非参数噪声的确定函数 — gradients 可以流过 `μ` 和 `σ`。

**Loss。** Evidence Lower BOund（ELBO），两个项：

```
loss = reconstruction + β · KL[q(z|x) || N(0, I)]
     = ||x - x̂||²  + β · Σ_i ( σ_i² + μ_i² - log σ_i² - 1 ) / 2
```

Reconstruction 把 `x̂` 推向 `x`。KL 把 `q(z|x)` 推向 prior。两者 trade off。小 β（<1）= 样本更锐利，code space 较不像 Gaussian。大 β（>1）= code space 更干净，样本更模糊。β-VAE（Higgins 2017）让这个旋钮出名，并开启 disentanglement 研究。

**Sampling。** 推理时：抽 `z ~ N(0, I)`，送入 decoder。一次 forward pass — 不像 diffusion 需要 iterative sampling。

## 构建它

`code/main.py` 实现了一个不依赖 numpy 或 torch 的 tiny VAE。输入是从 8-D 二组分 Gaussian mixture 抽取的 8 维 synthetic data。Encoder 和 decoder 是单 hidden-layer MLPs。我们实现 tanh activation、forward pass、loss 和手写 backward pass。不是生产代码 — 是教学。

### 第 1 步：encoder forward

```python
def encode(x, enc):
    h = tanh(add(matmul(enc["W1"], x), enc["b1"]))
    mu = add(matmul(enc["W_mu"], h), enc["b_mu"])
    log_sigma2 = add(matmul(enc["W_sig"], h), enc["b_sig"])
    return mu, log_sigma2
```

输出 `log σ²` 而不是 `σ`，这样网络输出不受约束（对 σ 做 softplus 是陷阱 — σ ≈ 0 时 gradients 会死）。

### 第 2 步：reparameterize 并 decode

```python
def reparameterize(mu, log_sigma2, rng):
    eps = [rng.gauss(0, 1) for _ in mu]
    sigma = [math.exp(0.5 * lv) for lv in log_sigma2]
    return [m + s * e for m, s, e in zip(mu, sigma, eps)]

def decode(z, dec):
    h = tanh(add(matmul(dec["W1"], z), dec["b1"]))
    return add(matmul(dec["W_out"], h), dec["b_out"])
```

### 第 3 步：ELBO

```python
def elbo(x, x_hat, mu, log_sigma2, beta=1.0):
    recon = sum((a - b) ** 2 for a, b in zip(x, x_hat))
    kl = 0.5 * sum(math.exp(lv) + m * m - lv - 1 for m, lv in zip(mu, log_sigma2))
    return recon + beta * kl, recon, kl
```

因为两个分布都是 Gaussian，所以 KL 有精确 closed-form。不要数值积分。2026 年仍有人上线 monte-carlo KL estimates — 它慢 3x，还没有理由。

### 第 4 步：generate

```python
def sample(dec, z_dim, rng):
    z = [rng.gauss(0, 1) for _ in range(z_dim)]
    return decode(z, dec)
```

这就是生成模型。五行。

## 陷阱

- **Posterior collapse。** KL 项过于强烈地把 `q(z|x) → N(0, I)`，导致 `z` 不携带 `x` 的信息。修复：β-annealing（从 β=0 开始，逐渐升到 1）、free bits，或跳过 inactive dimensions 上的 KL。
- **样本模糊。** Gaussian decoder likelihood 意味着 MSE reconstruction，而 MSE 对 L2 的 Bayes-optimal 是均值 — 一组 plausible digits 的均值就是模糊数字。修复：discrete decoder（VQ-VAE、NVAE），或只把 VAE 用作 encoder，并在 latents 上叠 diffusion（Stable Diffusion 就是这么做的）。
- **β 太大、太早。** 见 posterior collapse。从 β≈0.01 开始并逐步 ramp。
- **Latent dim 太小。** MNIST 用 16-D，ImageNet 256² 用 256-D，ImageNet 1024² 用 2048-D。Stable Diffusion 的 VAE 把 512×512×3 压缩成 64×64×4（空间面积下采样 32x，channels 也 32x）。

## 使用它

2026 年 VAE stack：

| 场景 | 选择 |
|-----------|------|
| Image-latent encoder for diffusion | Stable Diffusion VAE（`sd-vae-ft-ema`）或 Flux VAE |
| Audio-latent encoder | Encodec（Meta）、SoundStream 或 DAC（Descript） |
| Video latents | Sora 的 spatiotemporal patches、Latte VAE、WAN VAE |
| Disentangled representation learning | β-VAE、FactorVAE、TCVAE |
| Discrete latents（用于 transformer modelling） | VQ-VAE、RVQ（ResidualVQ） |
| Continuous latents for generation | Plain VAE，然后在该 latent space 中 condition 一个 flow/diffusion model |

Latent-diffusion model 就是一个 VAE，中间住着一个 diffusion model。VAE 做粗压缩，diffusion model 做重活。视频（VAE + video-diffusion DiT）和音频（Encodec + MusicGen transformer）也是同样模式。

## 交付它

保存 `outputs/skill-vae-trainer.md`。

Skill 接收：dataset profile + latent-dim target + downstream use（reconstruction、sampling 或 latent-diffusion input），并输出：architecture choice（plain/β/VQ/RVQ）、β schedule、latent dim、decoder likelihood（Gaussian vs categorical）和 evaluation plan（recon MSE、KL per dim、`q(z|x)` 与 `N(0, I)` 之间的 Fréchet distance）。

## 练习

1. **简单。** 把 `code/main.py` 中的 `β` 改成 `0.01`、`0.1`、`1.0`、`5.0`。记录最终 reconstruction MSE 和 KL。哪个 β 对你的 synthetic data 是 Pareto-best？
2. **中等。** 用 Bernoulli likelihood（cross-entropy loss）替换 Gaussian decoder likelihood。在同一 synthetic data 的 binarized 版本上比较 sample quality。
3. **困难。** 把 `code/main.py` 扩展成 mini VQ-VAE：用 K=32 entries 的 codebook 中最近邻替换 continuous `z`。比较 reconstruction MSE，并报告使用了多少 codebook entries（codebook collapse 很真实）。

## 关键词

| 术语 | 大家怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| Autoencoder | Encode-decode network | `x → z → x̂`，学习 MSE。不是生成模型。 |
| VAE | 带 sampler 的 AE | Encoder 输出分布，KL penalty 塑造 code space。 |
| ELBO | Evidence lower bound | `log p(x) ≥ recon - KL[q(z|x) \|\| p(z)]`；当 `q = p(z|x)` 时 tight。 |
| Reparameterization | `z = μ + σ·ε` | 把 stochastic node 改写为 deterministic + pure noise。允许 backprop 穿过 sampling。 |
| Prior | `p(z)` | latent 的目标分布，通常是 `N(0, I)`。 |
| Posterior collapse | “KL term wins” | Encoder 忽略 `x`，输出 prior；decoder 必须 hallucinate。 |
| β-VAE | 可调 KL 权重 | `loss = recon + β·KL`。β 越高越 disentangled，但越模糊。 |
| VQ-VAE | Discrete latent | 用最近 codebook vector 替换 continuous `z`；支持 transformer modelling。 |

## 生产备注：VAE 是 diffusion server 中最热的路径

在 Stable Diffusion / Flux / SD3 pipeline 中，VAE 每次请求会被调用两次 — img2img / inpainting 时 encode 一次，decode 一次。在 1024² 下，decoder pass 往往是整个 pipeline 中最大的 activation-memory peak，因为它把 `128×128×16` latents 上采样回 `1024×1024×3`。两个实际后果：

- **Slice 或 tile decode。** `diffusers` 暴露 `pipe.vae.enable_slicing()` 和 `pipe.vae.enable_tiling()`。Tiling 用小的 seam artifact 换 `O(tile²)` 内存，而不是 `O(H·W)`。对 consumer GPUs 上的 1024²+ 必不可少。
- **bf16 decoder，最终 resize 用 fp32 numerics。** SD 1.x VAE 发布时是 fp32，在 1024²+ cast 到 fp16 会 *悄悄产生 NaNs*。SDXL 提供 `madebyollin/sdxl-vae-fp16-fix` — 总是优先使用 fp16-fix 变体，或使用 bf16。

## 延伸阅读

- [Kingma & Welling (2013). Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) — VAE 论文。
- [Higgins et al. (2017). β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework](https://openreview.net/forum?id=Sy2fzU9gl) — disentangled β-VAE。
- [van den Oord et al. (2017). Neural Discrete Representation Learning](https://arxiv.org/abs/1711.00937) — VQ-VAE。
- [Vahdat & Kautz (2021). NVAE: A Deep Hierarchical Variational Autoencoder](https://arxiv.org/abs/2007.03898) — state-of-the-art image VAE。
- [Rombach et al. (2022). High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752) — Stable Diffusion；VAE 作为 encoder。
- [Défossez et al. (2022). High Fidelity Neural Audio Compression](https://arxiv.org/abs/2210.13438) — Encodec，音频 VAE 标准。
