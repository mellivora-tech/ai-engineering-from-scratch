# StyleGAN

> 大多数 generators 会把 `z` 同时搅进每一层。StyleGAN 把它拆开：先把 `z` 映射到中间变量 `w`，再通过 AdaIN 在每个 resolution level *注入* `w`。这个单一变化解缠了 latent space，并让 photorealistic faces 连续七年成为已解决问题。

**类型：** 构建
**语言：** Python
**前置要求：** 阶段 8 · 03（GANs），阶段 4 · 08（Normalization），阶段 3 · 07（CNNs）
**时间：** ~45 分钟

## 问题

DCGAN 通过一堆 transposed convolutions 把 `z` 映射成图像。问题是：`z` 控制一切 — 姿态、光照、身份、背景 — 全部纠缠在一起。沿 `z` 的一个轴移动，四者都会变化。你无法要求模型“同一个人，不同姿态”，因为表示并没有那样 factorize。

Karras et al.（2019，NVIDIA）提出：停止把 `z` 直接喂入 conv layers。用一个常量 `4×4×512` tensor 作为网络输入。学习一个 8-layer MLP，把 `z ∈ Z → w ∈ W`。通过 *adaptive instance normalization*（AdaIN）在每个 resolution 注入 `w`：normalize 每个 conv feature map，然后用 `w` 的 affine projections 做 scale 和 shift。再加入 per-layer noise 处理随机细节（皮肤毛孔、发丝）。

结果：`W` 大致拥有“高级风格”（姿态、身份）和“精细风格”（光照、颜色）的正交轴。你可以用图像 A 的 `w` 控制低分辨率 levels，用图像 B 的 `w` 控制高分辨率 levels，从而在两张图之间交换 styles。这解锁了 editing、cross-domain stylization 和整个 “StyleGAN-inversion” 研究线。

## 概念

![StyleGAN: mapping network + AdaIN + per-layer noise](../assets/stylegan.svg)

**Mapping network。** `f: Z → W`，一个 8-layer MLP。`Z = N(0, I)^512`。`W` 不被强制为 Gaussian — 它会学习适配数据的形状。

**Synthesis network。** 从学习得到的常量 `4×4×512` 开始。每个 resolution block：`upsample → conv → AdaIN(w_i) → noise → conv → AdaIN(w_i) → noise`。Resolution 翻倍：4、8、16、32、64、128、256、512、1024。

**AdaIN。**

```
AdaIN(x, y) = y_scale · (x - mean(x)) / std(x) + y_bias
```

其中 `y_scale` 和 `y_bias` 来自 `w` 的 affine projections。先按 feature map normalize，再重新 style。“Style” 在这里指 feature map 的一阶和二阶统计量。

**Per-layer noise。** 给每个 feature map 加单通道 Gaussian noise，并用学习到的 per-channel 因子缩放。控制 stochastic detail，而不影响全局结构。

**Truncation trick。** 推理时采样 `z`，计算 `w = mapping(z)`，然后 `w' = ŵ + ψ·(w - ŵ)`，其中 `ŵ` 是大量 samples 上 `w` 的均值。`ψ < 1` 用多样性换质量。几乎每个 StyleGAN demo 都用 `ψ ≈ 0.7`。

## StyleGAN 1 → 2 → 3

| 版本 | 年份 | 创新 |
|---------|------|------------|
| StyleGAN | 2019 | Mapping network + AdaIN + noise + progressive growing。 |
| StyleGAN2 | 2020 | Weight demodulation 替代 AdaIN（修复 droplet artifacts）；skip/residual architecture；path-length regularization。 |
| StyleGAN3 | 2021 | Alias-free convolution + equivariant kernels；消除 texture sticking to pixel grid。 |
| StyleGAN-XL | 2022 | Class-conditional、1024²、ImageNet。 |
| R3GAN | 2024 | 用更强 reg 重新包装；以 20x 更少参数在 FFHQ-1024 上缩小与 diffusion 的差距。 |

2026 年 StyleGAN3 仍然是默认选择，适用于（a）窄领域高 FPS photorealism，（b）few-shot domain adaptation（用 100 张新数据训练，freeze mapping），（c）inversion-based editing（找到重建真实照片的 `w`，再编辑该 `w`）。对 open-domain text-to-image，它不是工具 — diffusion 才是。

## 构建它

`code/main.py` 实现了一个 1-D 的 toy “style-GAN lite”：mapping MLP、一个接收 learned constant vector 并用 `w` 派生 scale/bias 调制的 synthesis function，以及 per-layer noise。它展示了通过 affine-modulation 注入 `w`，可以匹配或超过把 `z` 拼接进 generator 输入的做法。

### 第 1 步：mapping network

```python
def mapping(z, M):
    h = z
    for i in range(num_layers):
        h = leaky_relu(add(matmul(M[f"W{i}"], h), M[f"b{i}"]))
    return h
```

### 第 2 步：adaptive instance normalization

```python
def adain(x, w_scale, w_bias):
    mu = mean(x)
    sd = std(x)
    x_norm = [(xi - mu) / (sd + 1e-8) for xi in x]
    return [w_scale * xi + w_bias for xi in x_norm]
```

Per-feature-map scale 和 bias 通过 linear projection 从 `w` 得到。

### 第 3 步：per-layer noise

```python
def add_noise(x, sigma, rng):
    return [xi + sigma * rng.gauss(0, 1) for xi in x]
```

Sigma per-channel 可学习。

## 陷阱

- **Droplet artifacts。** StyleGAN 1 会在 feature maps 中产生 blobby droplet，因为 AdaIN 把 mean 清零。StyleGAN 2 的 weight demodulation 通过缩放 convolution weights 修复它。
- **Texture sticking。** StyleGAN 1 和 2 的纹理跟随 pixel coordinates，而不是 object coordinates（插值时明显）。StyleGAN 3 的 alias-free convolutions 用 windowed sinc filters 修复。
- **Mode coverage。** Truncation `ψ < 0.7` 看起来干净，但只从窄 cone 中采样；如果需要多样性，用 `ψ = 1.0`。
- **Inversion 有损。** 把真实照片 invert 到 `W` 通常通过 optimization 或 encoder（e4e、ReStyle、HyperStyle）完成。迭代多了结果会漂移。

## 使用它

| 用例 | 方法 |
|----------|----------|
| Photoreal human faces（anime、product、narrow） | StyleGAN3 FFHQ / custom fine-tune |
| 从照片做 face editing | e4e inversion + StyleSpace / InterFaceGAN directions |
| Face swap / reenactment | StyleGAN + encoder + blending |
| Avatar pipelines | StyleGAN3 w/ ADA 做 low-data fine-tune |
| 从少量图像做 domain adaptation | Freeze mapping network，fine-tune synthesis |
| Multi-modal 或 text-conditioned generation | 不要 — 用 diffusion |

对于答案是“人的脸部照片”的 product-grade demos，StyleGAN 在 inference cost（单次 forward pass，4090 上 <10ms）和同等质量门槛下的 sharpness 上都胜过 diffusion。

## 交付它

保存 `outputs/skill-stylegan-inversion.md`。Skill 接收一张真实照片并输出：inversion method（e4e / ReStyle / HyperStyle）、expected latent loss、editing budget（你可以在 `W` 中移动多远而不出 artifacts）以及 known-good editing directions（age、expression、pose）列表。

## 练习

1. **简单。** 运行 `code/main.py`，分别设置 `adain_on=True` 和 `adain_on=False`。比较固定 latent 与 perturbed latent 的输出 spread。
2. **中等。** 实现 mixing regularization：对一个训练 batch，计算 `w_a`、`w_b`，并在 synthesis 前半部分使用 `w_a`、后半部分使用 `w_b`。Decoder 会学到 disentangled styles 吗？
3. **困难。** 取 pretrained StyleGAN3 FFHQ model（ffhq-1024.pkl）。通过在 labelled samples 上训练 SVM，找到控制 “smile” 的 `w` direction；报告在 identity drift 前能推多远。

## 关键词

| 术语 | 大家怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| Mapping network | “那个 MLP” | `f: Z → W`，8 层，把 latent geometry 与 data statistics 解耦。 |
| W space | “Style space” | Mapping network 的输出；大致 disentangled。 |
| AdaIN | “Adaptive instance norm” | Normalize feature map，然后用 `w` projection 做 scale + shift。 |
| Truncation trick | “Psi” | `w = mean + ψ·(w - mean)`，ψ<1 用多样性换质量。 |
| Path-length regularization | “PL reg” | 惩罚图像相对 `w` 单位变化的过大变化；让 `W` 更平滑。 |
| Weight demodulation | “StyleGAN2 fix” | Normalize conv weights 而不是 activations；消除 droplet artifacts。 |
| Alias-free | “StyleGAN3 技巧” | Windowed sinc filters；消除 texture sticking to pixel grid。 |
| Inversion | “为真实图像找到 w” | Optimize 或 encode `x → w`，使 `G(w) ≈ x`。 |

## 生产备注：为什么 StyleGAN 在 2026 年仍会上线

StyleGAN3 在 4090 上生成一张 1024² FFHQ face 不到 10 ms — `num_steps = 1`，没有 VAE decode，没有 cross-attention pass。生产上，这是任何 image generator 的延迟下限。同分辨率 50-step SDXL + VAE-decode pipeline 约 3 秒。这是 **300× 差距**，对窄领域产品（avatar services、ID document pipelines、stock face generation）来说，它在 TCO 上胜出。

两个运维后果：

- **没有 scheduler，没有 batcher。** 在目标 occupancy 上用 static batch 最优。Continuous batching（LLMs 和 diffusion 必需）没有收益，因为每个请求 FLOPs 相同。
- **Truncation `ψ` 是 safety knob。** `ψ < 0.7` 会从 mapping network range 的狭窄 cone 中采样。这是 serving layer 控制 sample variance 的唯一杠杆。高峰负载时降低 `ψ`，给 premium users 提高。

## 延伸阅读

- [Karras et al. (2019). A Style-Based Generator Architecture for GANs](https://arxiv.org/abs/1812.04948) — StyleGAN。
- [Karras et al. (2020). Analyzing and Improving the Image Quality of StyleGAN](https://arxiv.org/abs/1912.04958) — StyleGAN2。
- [Karras et al. (2021). Alias-Free Generative Adversarial Networks](https://arxiv.org/abs/2106.12423) — StyleGAN3。
- [Tov et al. (2021). Designing an Encoder for StyleGAN Image Manipulation](https://arxiv.org/abs/2102.02766) — e4e inversion。
- [Sauer et al. (2022). StyleGAN-XL: Scaling StyleGAN to Large Diverse Datasets](https://arxiv.org/abs/2202.00273) — StyleGAN-XL。
- [Huang et al. (2024). R3GAN: The GAN is dead; long live the GAN!](https://arxiv.org/abs/2501.05441) — 现代 minimal GAN recipe。
