# 生成模型 — 分类与历史

> 每个图像模型、文本模型、视频模型和 3D 模型都能放进五个桶之一。选错桶，你会和数学缠斗几周。选对桶，过去十二年的领域进展会在脑子里清楚地堆起来。

**类型：** 学习
**语言：** Python
**前置要求：** 阶段 2（ML Fundamentals），阶段 3（Deep Learning Core），阶段 7 · 14（Transformers）
**时间：** ~45 分钟

## 问题

生成模型只做一件事：给定从某个未知分布 `p_data(x)` 中抽出的训练样本，输出看起来像来自同一分布的新样本。人脸、句子、MIDI 文件、蛋白结构 — 眯起眼看都是同一个问题。

麻烦在于 `p_data` 位于数百万维空间中（512x512 RGB 图像约 786k 维），样本落在这个空间内部一张很薄的 manifold 上，而你可能只有 10M 个例子。暴力求密度没有希望。每个生成模型都是一种妥协：把一个困难问题换成稍微不那么困难的问题。

过去十二年存活下来的有五个家族。知道每个家族做了什么妥协，你就能理解它为什么在某些任务上赢、在另一些任务上崩。

## 概念

![Five families of generative models — taxonomy by what they model](../assets/taxonomy.svg)

**1. Explicit density, tractable。** 把 `log p(x)` 写成一个真的能求值的和。Autoregressive models（PixelCNN、WaveNet、GPT）把 `p(x)` 分解为 `p(x) = ∏ p(x_i | x_<i)`。Normalizing flows（RealNVP、Glow）把 `p(x)` 构造成简单 base distribution 的可逆变换。优点：精确 likelihood，训练 loss 干净。缺点：autoregressive inference 是顺序的（长序列慢），flows 需要可逆架构（架构受限）。

**2. Explicit density, approximate。** 从下方 bound `log p(x)`（ELBO）并优化这个 bound。VAEs（Kingma 2013）使用带 variational posterior 的 encoder-decoder。Diffusion models（DDPM，Ho 2020）训练 denoiser，隐式优化加权 ELBO。Diffusion 是 2026 年图像、视频和 3D 的主导 backbone。

**3. Implicit density。** 完全跳过密度；学习一个 generator `G(z)` 产生样本，以及一个 discriminator `D(x)` 判断真假。GANs（Goodfellow 2014）。推理快（一次 forward pass），但训练出了名不稳定。即使在 2026 年，StyleGAN 1/2/3 仍然是固定领域 photorealism（人脸、卧室）的 state of the art。

**4. Score-based / continuous-time。** 直接学习 log-density 的梯度 `∇_x log p(x)`（score）。Song & Ermon（2019）展示了 score matching 如何把 diffusion 泛化为 SDE。Flow matching（Lipman 2023）是 2024–2026 的热门方向：无需模拟训练、更直路径、采样比 DDPM 快 4–10x。Stable Diffusion 3、Flux、AudioCraft 2 都使用 flow matching。

**5. Token-based autoregressive over discrete codes。** 用 VQ-VAE 或 residual quantizer 把高维数据压缩成一段较短离散 tokens，再用 Transformer 建模 token 序列。Parti、MuseNet、AudioLM、VALL-E、Sora 的 patch tokenizer 都用这个。这是 bucket 1 加上一个 learned tokenizer。

## 简史

| 年份 | 模型 | 为什么重要 |
|------|-------|-----------------|
| 2013 | VAE (Kingma) | 第一个有可用训练 loss 的深度生成模型。 |
| 2014 | GAN (Goodfellow) | Implicit density，无 likelihood — 样本锐利得惊人。 |
| 2015 | DRAW, PixelCNN | 顺序图像生成。 |
| 2017 | Glow, RealNVP | 可逆 flows；深层下的精确 likelihood。 |
| 2017 | Progressive GAN | 第一批 megapixel faces。 |
| 2019 | StyleGAN / StyleGAN2 | 对人脸这个领域，photorealistic faces 仍然很难打败。 |
| 2020 | DDPM (Ho) | Diffusion 变得实用。 |
| 2021 | CLIP, DALL-E 1, VQGAN | Text-to-image 进入主流。 |
| 2022 | Imagen, Stable Diffusion 1, DALL-E 2 | Latent diffusion + text conditioning = 商品化。 |
| 2022 | ControlNet, LoRA | 对 pretrained diffusion 的精细控制。 |
| 2023 | SDXL, Midjourney v5, Flow matching | Scale + 更好的训练动态。 |
| 2024 | Sora, Stable Diffusion 3, Flux.1 | Video diffusion；flow matching 胜出。 |
| 2025 | Veo 2, Kling 1.5, Runway Gen-3, Nano Banana | 生产级视频。 |
| 2026 | Consistency + Rectified Flow | 从 diffusion backbones 做 one-step sampling。 |

## 五问题 triage

当一篇新的生成模型论文出现时，读 method section 之前先回答这五个问题。

1. **建模的是什么？** Pixels、latents、discrete tokens、3D Gaussians、meshes、waveforms？
2. **密度是 explicit 还是 implicit？** 他们有没有写下 `log p(x)`？
3. **Sampling 是 one-shot 还是 iterative？** Iterative 意味着推理更慢；one-shot 通常意味着 adversarial 或 distilled。
4. **Conditioning 是什么：unconditional、class、text、image、pose？** 这决定 loss 和 architecture scaffolding。
5. **Evaluation：FID、CLIP score、IS、human preference、task accuracy？** 每种都有已知 failure modes（见第 14 课）。

这个阶段的每一课你都会重新回答这五个问题。到最后，它会变成反射。

## 构建它

本课代码是一个轻量可视化：用三种玩具方法（kernel density、discrete histogram、nearest-sample “GAN-ish” generator）从样本拟合一个 1-D mixture-of-Gaussians，让你在一屏输出里看清 explicit vs implicit density 的区别。

运行 `code/main.py`。它从双峰 Gaussian mixture 抽 2000 个样本，然后打印：

```
explicit density (histogram): p(x in [-0.5, 0.5]) ≈ 0.38
approximate density (KDE):     p(x in [-0.5, 0.5]) ≈ 0.41
implicit (nearest-sample gen): 20 new samples printed, no p(x)
```

注意：前两个允许你问“这个点有多可能？”第三个不能。这就是 *explicit vs implicit* 区别，它会影响后面每一课。

## 使用它

2026 年，哪个家族适合哪个任务？

| 任务 | 最佳家族 | 原因 |
|------|-------------|-----|
| Photoreal faces, narrow domain | StyleGAN 2/3 | 仍然最锐利，推理最快。 |
| General text-to-image | Latent diffusion + flow matching | SD3, Flux.1, DALL-E 3。 |
| Fast text-to-image | Rectified flow + distillation | SDXL-Turbo, SD3-Turbo, LCM。 |
| Text-to-video | Diffusion Transformer + flow matching | Sora, Veo 2, Kling。 |
| Speech + music | Token-based AR（AudioLM、VALL-E、MusicGen）或 flow matching（AudioCraft 2） | 离散 tokens 可低成本 scale。 |
| 3D scenes | Gaussian Splatting fit, diffusion prior | 3D-GS 用于 reconstruction，diffusion 用于 novel-view。 |
| Density estimation（不采样） | Flows | 唯一有精确 `log p(x)` 的家族。 |
| Simulation / physics | Flow matching, score SDE | 直线路径、平滑 vector fields。 |

## 交付它

保存为 `outputs/skill-model-chooser.md`。

这个 skill 接收任务描述并输出：（1）该用哪个家族，（2）三个 open options 和三个 hosted options 的排序列表，（3）你应该关注的可能 failure mode，（4）compute/time budget。

## 练习

1. **简单。** 对以下五个产品，识别其 family 和 backbone：ChatGPT image、Midjourney v7、Sora、Runway Gen-3、ElevenLabs。证据应来自公开技术报告。
2. **中等。** 你明天要读的一篇论文声称 sampling 比 diffusion 快 100x。写下三个问题，用来检查这个 speedup 是否在 conditioning 和高分辨率下仍然成立。
3. **困难。** 选择一个你关心的领域（例如 protein structure、CAD、molecules、trajectories）。对该领域当前 SOTA 模型回答五问题 triage，并草拟一个更好模型会改变什么。

## 关键词

| 术语 | 大家怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| Generative model | “它会造新东西” | 学习 `p_data(x)` 的 sampler，可选地暴露 `log p(x)`。 |
| Explicit density | “你可以求值” | 模型提供 closed-form 或 tractable 的 `log p(x)`。 |
| Implicit density | “GAN-style” | 只有 sampler — 无法评估给定点的 `p(x)`。 |
| ELBO | “Evidence lower bound” | `log p(x)` 的 tractable lower bound；VAEs 和 diffusion 优化它。 |
| Score | “log-density 的梯度” | `∇_x log p(x)`；diffusion 和 SDE models 学习这个场。 |
| Manifold hypothesis | “数据活在一张曲面上” | 高维数据集中在低维 manifold 上；这也是 dimensionality reduction 有效的原因。 |
| Autoregressive | “预测下一个片段” | 把 joint 分解为 conditionals 的乘积。 |
| Latent | “压缩 code” | Decoder 可从中重建输入的低维表示。 |

## 生产备注：五个家族，五种推理形状

每个家族都对应不同的 inference-server 成本曲线。production-inference 文献把 LLM inference 分成 prefill + decode；同样的分解也适用于这里：

- **Autoregressive（bucket 1 和 5）。** 顺序 decode 主导延迟；KV-cache、continuous batching 和 speculative decoding 都直接适用。
- **VAE / diffusion / flow-matching（bucket 2 和 4）。** 没有 LLM 意义上的 decode。成本 = `num_steps × step_cost`，而 `step_cost` 是在完整 latent resolution 上的一次 transformer 或 U-Net forward。生产旋钮是 step count（DDIM / DPM-Solver / distillation）、batch size 和 precision（bf16 / fp8 / int4）。
- **GAN（bucket 3）。** 一次 forward pass。没有 schedule，没有 KV-cache。TTFT ≈ 总延迟。这就是 StyleGAN 在窄领域 UX 中仍然胜出的原因。

当你在论文摘要里看到 “faster than diffusion”，把它翻译成“更少 steps × 相同 step cost”或“相同步数 × 更便宜 step cost”。其他都是营销。

## 延伸阅读

- [Goodfellow et al. (2014). Generative Adversarial Nets](https://arxiv.org/abs/1406.2661) — GAN 论文。
- [Kingma & Welling (2013). Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) — VAE 论文。
- [Ho, Jain, Abbeel (2020). Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) — DDPM 论文。
- [Song et al. (2021). Score-Based Generative Modeling through SDEs](https://arxiv.org/abs/2011.13456) — diffusion as an SDE。
- [Lipman et al. (2023). Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747) — flow matching 论文。
- [Esser et al. (2024). Scaling Rectified Flow Transformers for High-Resolution Image Synthesis](https://arxiv.org/abs/2403.03206) — Stable Diffusion 3。
