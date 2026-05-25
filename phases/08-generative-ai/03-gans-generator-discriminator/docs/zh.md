# GANs — Generator vs Discriminator

> Goodfellow 在 2014 年的技巧是完全跳过 density。两个网络。一个造假。一个抓假。它们互相对抗，直到假样本无法和真样本区分。它不该有效。它也经常无效。但一旦有效，在窄领域里样本仍然是文献中最锐利的。

**类型：** 构建
**语言：** Python
**前置要求：** 阶段 3 · 02（Backprop），阶段 3 · 08（Optimizers），阶段 8 · 02（VAE）
**时间：** ~75 分钟

## 问题

VAEs 会产生模糊样本，因为它们的 MSE decoder loss 对 *mean* image 是 Bayes-optimal — 许多 plausible digits 的均值就是一个模糊数字。你想要的是奖励 *plausibility* 的 loss，而不是到某个 target 的 pixel-wise 距离。Plausibility 没有 closed-form。你必须学习它。

Goodfellow 的想法：训练一个 classifier `D(x)` 区分真图像和假图像。训练一个 generator `G(z)` 欺骗 `D`。`G` 的 loss signal 是 `D` 当前认为“看起来真实”的东西。这个 signal 会随着 `G` 改善而更新，追逐一个移动目标。如果两个网络都收敛，`G` 就在从未写下 `log p(x)` 的情况下学到了数据分布。

这就是 adversarial training。数学上是 minimax game：

```
min_G max_D  E_real[log D(x)] + E_fake[log(1 - D(G(z)))]
```

2026 年 GANs 已不再是 SOTA generator（diffusion 和 flow matching 吃掉了王冠）。但 StyleGAN 2/3 仍然是上线过的最锐利人脸模型，GAN discriminators 被用作 diffusion training 中的 *perceptual losses*，adversarial training 也驱动了让你能上线实时 diffusion 的快速 1-step distillations（SDXL-Turbo、SD3-Turbo、LCM）。

## 概念

![GAN training: generator and discriminator in minimax](../assets/gan.svg)

**Generator `G(z)`。** 把 noise vector `z ~ N(0, I)` 映射到样本 `x̂`。形状像 decoder 的网络（dense 或 transposed conv）。

**Discriminator `D(x)`。** 把样本映射到标量概率（或 score）。真 → 1，假 → 0。

**Loss。** 两个交替更新：

- **训练 `D`：** `loss_D = -[ log D(x) + log(1 - D(G(z))) ]`。真实=1、伪造=0 的 binary cross-entropy。
- **训练 `G`：** `loss_G = -log D(G(z))`。这是 Goodfellow 使用的 *non-saturating* 形式（原始 `log(1 - D(G(z)))` 会在 `D` 很确信时饱和并杀死梯度）。

**Training loop。** 一步 `D`，一步 `G`。重复。

**为什么有效。** 如果 `G` 完全匹配 `p_data`，那么 `D` 最多只能随机猜测，到处输出 0.5；`G` 不再得到梯度。达到均衡。

**为什么会坏。** Mode collapse（`G` 找到一个 `D` 分不出的 mode 然后永远生产它）、vanishing gradient（`D` 学太快，`log D` 饱和）、训练不稳定（learning rates、batch sizes，什么都可能）。

## 让 GANs 有效的变体

| 年份 | 创新 | 修复 |
|------|------------|-----|
| 2015 | DCGAN | Conv/deconv、batch norm、LeakyReLU — 第一个稳定架构。 |
| 2017 | WGAN, WGAN-GP | 用 Wasserstein distance + gradient penalty 替换 BCE。修复 vanishing gradient。 |
| 2017 | Spectral normalization | 对 discriminator 做 Lipschitz bound。2026 年 discriminators 仍在用。 |
| 2018 | Progressive GAN | 先训练低分辨率，再加层。第一批 megapixel 结果。 |
| 2019 | StyleGAN / StyleGAN2 | Mapping network + adaptive instance norm。固定领域 photorealism 的 SOTA。 |
| 2021 | StyleGAN3 | Alias-free、translation-equivariant — 2026 年仍是人脸黄金标准。 |
| 2022 | StyleGAN-XL | Conditional、class-aware、更大规模。 |
| 2024 | R3GAN | 用更强 regularization 重新包装；无需技巧可在 1024² 上工作。 |

## 构建它

`code/main.py` 在 1-D 数据上训练一个 tiny GAN：两个 Gaussians 的 mixture。Generator 和 discriminator 都是 single-hidden-layer MLPs。我们手写 forward、backward 和 minimax loop。目标是亲眼看到两个关键 failure modes（mode collapse + vanishing gradient）发生。

### 第 1 步：non-saturating loss

Vanilla Goodfellow loss `log(1 - D(G(z)))` 会在 D 高置信地把 G 的假样本判为假时趋近 0。此时 G 的梯度基本为零 — G 无法改善。Non-saturating 形式 `-log D(G(z))` 拥有相反渐近行为：D 越确信，它越爆炸，给 G 强信号。

```python
def g_loss(d_fake):
    # maximize log D(G(z))  <=>  minimize -log D(G(z))
    return -sum(math.log(max(p, 1e-8)) for p in d_fake) / len(d_fake)
```

### 第 2 步：每一步 generator 对应一步 discriminator

```python
for step in range(steps):
    # train D
    real_batch = sample_real(batch_size)
    fake_batch = [G(z) for z in sample_noise(batch_size)]
    update_D(real_batch, fake_batch)

    # train G
    fake_batch = [G(z) for z in sample_noise(batch_size)]  # fresh fakes
    update_G(fake_batch)
```

G 要用 fresh fakes，否则 gradients 过期。

### 第 3 步：观察 mode collapse

```python
if step % 200 == 0:
    samples = [G(z) for z in sample_noise(500)]
    mode_a = sum(1 for s in samples if s < 0)
    mode_b = 500 - mode_a
    if min(mode_a, mode_b) < 50:
        print("  [!] mode collapse: one mode is starved")
```

经典症状：两个真实 modes 中有一个不再被生成。Discriminator 停止修正它，因为它从未作为 fake 出现。

## 陷阱

- **Discriminator 太强。** 把 D 的 learning rate 降低 2-5x，或给输入加 instance/layer noise。如果 D 准确率达到 >95%，G 已经死了。
- **Generator 记住一个 mode。** 给 D 输入加噪声，用 minibatch-discriminator layer，或切换到 WGAN-GP。
- **Batch norm 泄漏统计量。** Real batch + fake batch 经过同一个 BN layer 会混合统计量。改用 instance norm 或 spectral norm。
- **Inception-score gaming。** FID 和 IS 在低样本数下噪声大。评估时使用 ≥10k samples。
- **One-shot sampling 对 conditional tasks 是谎言。** 你仍然需要 CFG scales、truncation tricks 和 re-sampling 才能得到可用输出。

## 使用它

2026 年 GAN stack：

| 场景 | 选择 |
|-----------|------|
| Photoreal human faces, fixed pose | StyleGAN3（最锐利，最小） |
| Anime / stylized faces | StyleGAN-XL 或 Stable Diffusion LoRA |
| Image-to-image translation | Pix2Pix / CycleGAN（阶段 8 · 04）或 ControlNet（阶段 8 · 08） |
| Fast 1-step text-to-image | Diffusion 的 adversarial distillation（SDXL-Turbo、SD3-Turbo） |
| Diffusion trainer 内部的 perceptual loss | 图像 crops 上的小 GAN discriminator |
| 任何 multi-modal、open-ended 任务 | 不要 — 用 diffusion 或 flow matching |

GANs 锐利但窄。一旦你的 domain 打开 — 照片、任意文本 prompts、视频 — 就切换到 diffusion。Adversarial 技巧作为组件（perceptual losses、distillation）继续存在，而不是独立 generator。

## 交付它

保存 `outputs/skill-gan-debugger.md`。Skill 接收一次失败 GAN run（loss curves、sample grid、dataset size），并输出按可能性排序的原因、一行修复和 rerun protocol。

## 练习

1. **简单。** 用默认设置运行 `code/main.py`。然后设置 `D_LR = 5 * G_LR` 并重新运行。G 的 loss 多快塌缩成常数？
2. **中等。** 用 WGAN loss 替换 Goodfellow BCE loss：`loss_D = E[D(fake)] - E[D(real)]`，`loss_G = -E[D(fake)]`，并把 D 权重 clip 到 `[-0.01, 0.01]`。训练更稳定吗？比较 wall-clock convergence。
3. **困难。** 把 1-D 示例扩展到 2-D 数据（环上的 8 个 Gaussians mixture）。追踪 generator 在 steps 1k、5k、10k 捕捉了多少个 modes。实现 minibatch discrimination 并重新测量。

## 关键词

| 术语 | 大家怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| Generator | “G” | Noise-to-sample network，`G: z → x̂`。 |
| Discriminator | “D” | Classifier `D: x → [0, 1]`，判断 real vs fake。 |
| Minimax | “The game” | 一个 joint objective 的 `min_G max_D`。 |
| Non-saturating loss | “The fix” | 对 G 使用 `-log D(G(z))`，而不是 `log(1 - D(G(z)))`。 |
| Mode collapse | “G 记住了一个东西” | 尽管数据多样，generator 只产生少数不同输出。 |
| WGAN | “Wasserstein” | 用 Earth-Mover distance + gradient penalty 替换 BCE；梯度更平滑。 |
| Spectral norm | “Lipschitz trick” | 约束 D 的权重范数以限制斜率；稳定训练。 |
| StyleGAN | “真正好用那个” | Mapping network + AdaIN；人脸领域最佳，2026 年仍然如此。 |

## 生产备注：one-shot inference 是 GAN 持久优势

GANs 在 open-domain generation 的 sample quality 上不再赢，但在 inference cost 上仍然赢。用 production-inference 文献的词汇，一个 GAN 有：

- **没有 prefill，没有 decode stages。** 单次 `G(z)` forward pass。TTFT ≈ 总延迟。
- **没有 KV-cache 压力。** 唯一状态是权重。Batch size 受 activation memory 限制，不受 cache 限制。
- **Trivial continuous batching。** 因为每个请求都是相同固定 FLOPs，在服务器目标 occupancy 上做 static batch 通常最优。不需要 in-flight scheduler。

这就是为什么 GAN distillation（SDXL-Turbo、SD3-Turbo、ADD、LCM）是 2026 年 fast text-to-image 的主导技术：它把 20-50-step diffusion pipeline 折叠成 1-4 次 GAN-style forward passes，同时保持 diffusion base 的分布。Adversarial loss 作为训练时旋钮继续存在，用来把慢 generator 变成快 generator。

## 延伸阅读

- [Goodfellow et al. (2014). Generative Adversarial Nets](https://arxiv.org/abs/1406.2661) — 原始 GAN 论文。
- [Radford et al. (2015). Unsupervised Representation Learning with DCGAN](https://arxiv.org/abs/1511.06434) — 第一个稳定架构。
- [Arjovsky, Chintala, Bottou (2017). Wasserstein GAN](https://arxiv.org/abs/1701.07875) — WGAN。
- [Miyato et al. (2018). Spectral Normalization for GANs](https://arxiv.org/abs/1802.05957) — SN。
- [Karras et al. (2020). Analyzing and Improving the Image Quality of StyleGAN](https://arxiv.org/abs/1912.04958) — StyleGAN2。
- [Karras et al. (2021). Alias-Free Generative Adversarial Networks](https://arxiv.org/abs/2106.12423) — StyleGAN3。
- [Sauer et al. (2023). Adversarial Diffusion Distillation](https://arxiv.org/abs/2311.17042) — SDXL-Turbo。
