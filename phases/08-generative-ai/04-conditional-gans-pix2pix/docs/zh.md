# Conditional GANs 与 Pix2Pix

> 2014–2017 年第一个大解锁，是控制 GAN 生成什么。接上一个 label、一张图像，或一个句子。Pix2Pix 做了图像版本，在窄 image-to-image 任务上，它到现在仍然击败每个通用 text-to-image 模型。

**类型：** 构建
**语言：** Python
**前置要求：** 阶段 8 · 03（GANs），阶段 4 · 06（U-Net），阶段 3 · 07（CNNs）
**时间：** ~75 分钟

## 问题

Unconditional GAN 会采样任意人脸。演示有用，生产无用。你想要的是：*把 sketch 映射成 photo*，*把 map 映射成 aerial photo*，*把白天场景映射成夜晚*，*给灰度图上色*。这些任务中，你都有输入图像 `x`，并且必须输出与其有语义对应的 `y`。每个 `x` 对应许多 plausible `y`。Mean-squared error 会把它们压扁成糊状。Adversarial loss 不会，因为“看起来真实”是锐利的。

Conditional GAN（Mirza & Osindero, 2014）把 condition `c` 作为输入加到 `G` 和 `D`。Pix2Pix（Isola et al., 2017）将它专门化：condition 是完整输入图像，generator 是 U-Net，discriminator 是 *patch-based* classifier（PatchGAN），loss 是 adversarial + L1。即使在 2026 年，这个配方仍然在窄 image-to-image 领域胜过 from-scratch text-to-image models，因为它在 *paired data* 上训练 — 你拥有正好需要的信号。

## 概念

![Pix2Pix: U-Net generator, PatchGAN discriminator](../assets/pix2pix.svg)

**Conditional G。** `G(x, z) → y`。在 Pix2Pix 中，`z` 是 G 内部的 dropout（没有 input noise — Isola 发现显式 noise 会被忽略）。

**Conditional D。** `D(x, y) → [0, 1]`。输入是 *pair*（condition, output）。这是关键差异：D 必须判断 `y` 是否与 `x` 一致，而不只是 `y` 看起来是否真实。

**U-Net generator。** 带 bottleneck 跨层 skip connections 的 encoder-decoder。对输入和输出共享低层结构（edges、silhouette）的任务很关键。没有 skips，高频细节会消失。

**PatchGAN discriminator。** D 不输出单个 real/fake score，而是输出一个 `N×N` grid，每个 cell 判断约 70×70 像素的 receptive field。再取平均。这是一个 Markov random field 假设：真实感是局部的。训练快得多、参数更少、输出更锐利。

**Loss。**

```
loss_G = -log D(x, G(x)) + λ · ||y - G(x)||_1
loss_D = -log D(x, y) - log (1 - D(x, G(x)))
```

L1 项稳定训练，并把 G 推向已知 target。L1 比 L2 产生更锐利的边缘（medians，而不是 means）。`λ = 100` 是 Pix2Pix 默认值。

## CycleGAN — 当你没有 pairs

Pix2Pix 需要 paired `(x, y)` 数据。CycleGAN（Zhu et al., 2017）以额外 loss 为代价去掉了这个要求：*cycle consistency* loss。两个 generators：`G: X → Y` 和 `F: Y → X`。训练它们使 `F(G(x)) ≈ x` 且 `G(F(y)) ≈ y`。这让你能在没有 paired examples 的情况下，把马翻译成斑马，把夏天翻译成冬天。

2026 年，unpaired image-to-image 大多通过 diffusion（ControlNet、IP-Adapter）完成，而不是 CycleGAN，但 cycle-consistency 思想仍存在于几乎每篇 unpaired domain adaptation 论文中。

## 构建它

`code/main.py` 在 1-D 数据上实现一个 tiny conditional GAN。Condition `c` 是 class label（0 或 1）。任务：为给定 class 产生来自 conditional distribution 的样本。

### 第 1 步：把 condition 追加到 G 和 D 输入

```python
def G(z, c, params):
    return mlp(concat([z, one_hot(c)]), params)

def D(x, c, params):
    return mlp(concat([x, one_hot(c)]), params)
```

One-hot encoding 是最简单方式。更大模型会使用 learned embeddings、FiLM modulation 或 cross-attention。

### 第 2 步：conditional 训练

```python
for step in range(steps):
    x, c = sample_real_conditional()
    noise = sample_noise()
    update_D(x_real=x, x_fake=G(noise, c), c=c)
    update_G(noise, c)
```

Generator 必须匹配 *给定 condition* 的真实分布，而不是 marginal。

### 第 3 步：验证 per-class output

```python
for c in [0, 1]:
    samples = [G(noise, c) for noise in batch]
    mean_c = mean(samples)
    assert_near(mean_c, real_mean_for_class_c)
```

## 陷阱

- **Condition 被忽略。** G 学会 marginalize，D 从不惩罚，因为 condition signal 太弱。修复：更强地 condition D（早期 layer，而不只是后期），使用 projection discriminator（Miyato & Koyama 2018）。
- **L1 权重太低。** G 漂向任意真实感输出，而不是忠实输出。Pix2Pix-style tasks 从 λ≈100 开始。
- **L1 权重太高。** G 产生模糊输出，因为 L1 仍然是 L_p norm。训练稳定后 anneal down。
- **Ground-truth leakage in D。** 把 `(x, y)` 拼接为 D 输入，而不是只给 `y`。没有这个，D 无法检查一致性。
- **Mode collapse per class。** 每个 class 都可能独立 collapse。运行 class-conditional diversity checks。

## 使用它

2026 年 image-to-image 任务状态：

| 任务 | 最佳方法 |
|------|---------------|
| Sketch → photo，同领域，paired data | Pix2Pix / Pix2PixHD（仍然快，仍然锐利） |
| Sketch → photo，unpaired | 带 Scribble conditioning model 的 ControlNet |
| Semantic seg → photo | SPADE / GauGAN2 或 SD + ControlNet-Seg |
| Style transfer | 带 IP-Adapter 或 LoRA 的 diffusion；GAN methods 是 legacy |
| Depth → photo | Stable Diffusion 上的 ControlNet-Depth |
| Super-resolution | Real-ESRGAN（GAN）、ESRGAN-Plus 或 SD-Upscale（diffusion） |
| Colorization | ColTran、diffusion-based colorizers 或 Pix2Pix-color |
| Daytime → nighttime、seasons、weather | CycleGAN 或 ControlNet-based |

当（a）你有数千个 paired examples，（b）任务窄且可重复，（c）你需要快速推理时，Pix2Pix 仍然是正确工具。通用 open-domain 任务上，diffusion 胜出。

## 交付它

保存 `outputs/skill-img2img-chooser.md`。Skill 接收任务描述、data availability（paired vs unpaired、N samples）和 latency/quality budget，然后输出：approach（Pix2Pix、CycleGAN、ControlNet variant、SDXL + IP-Adapter）、training data requirements、inference cost 和 eval protocol（LPIPS、FID、task-specific）。

## 练习

1. **简单。** 修改 `code/main.py`，添加第三个 class。确认 G 仍然把每个 class 的 noise 映射到正确 mode。
2. **中等。** 在 1-D 设置中用 perceptual-style loss 替换 L1（例如用一个小 frozen D 作为 feature extractor）。它会改变 conditional distribution 的锐利度吗？
3. **困难。** 在 1-D 设置中草拟一个 CycleGAN：两个分布、两个 generators、cycle loss。展示它如何在没有 paired data 的情况下学习映射。

## 关键词

| 术语 | 大家怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| Conditional GAN | “带 labels 的 GAN” | G(z, c)、D(x, c)。两个网络都看到 condition。 |
| Pix2Pix | “Image-to-image GAN” | Paired cGAN，使用 U-Net G 和 PatchGAN D + L1 loss。 |
| U-Net | “带 skips 的 encoder-decoder” | 对称 conv network；skips 保留 high-freq。 |
| PatchGAN | “Local-realism classifier” | D 输出 per-patch score，而不是 global score。 |
| CycleGAN | “Unpaired image translation” | 两个 G + cycle-consistency loss；不需要 paired data。 |
| SPADE | “GauGAN” | 用 semantic map 对中间 activations 做 normalization。segmentation-to-image。 |
| FiLM | “Feature-wise linear modulation” | 来自 condition 的 per-feature affine transform；便宜的 conditioning。 |

## 生产备注：Pix2Pix 作为 latency-bound baseline

当你有 paired data 和窄任务（sketch → render、semantic map → photo、day → night）时，Pix2Pix 的 one-shot inference 在 latency 上比 diffusion 快一个数量级。生产比较通常是：

| Path | Steps | Typical latency at 512² on a single L4 |
|------|-------|----------------------------------------|
| Pix2Pix (U-Net forward) | 1 | ~30 ms |
| SD-Inpaint or SD-Img2Img | 20 | ~1.2 s |
| SDXL-Turbo Img2Img | 1-4 | ~0.15-0.35 s |
| ControlNet + SDXL base | 20-30 | ~3-5 s |

Pix2Pix 在 static batches 中赢得 throughput（每个请求 FLOPs 相同）。Diffusion 在质量和泛化上赢。现代玩法通常是为窄任务上线 Pix2Pix-style distilled model，并为长尾输入准备 diffusion fallback。

## 延伸阅读

- [Mirza & Osindero (2014). Conditional Generative Adversarial Nets](https://arxiv.org/abs/1411.1784) — cGAN 论文。
- [Isola et al. (2017). Image-to-Image Translation with Conditional Adversarial Networks](https://arxiv.org/abs/1611.07004) — Pix2Pix。
- [Zhu et al. (2017). Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/abs/1703.10593) — CycleGAN。
- [Wang et al. (2018). High-Resolution Image Synthesis with Conditional GANs](https://arxiv.org/abs/1711.11585) — Pix2PixHD。
- [Park et al. (2019). Semantic Image Synthesis with Spatially-Adaptive Normalization](https://arxiv.org/abs/1903.07291) — SPADE / GauGAN。
- [Miyato & Koyama (2018). cGANs with Projection Discriminator](https://arxiv.org/abs/1802.05637) — projection D。
