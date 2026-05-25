# Visual Autoregressive Modeling (VAR)：Next-Scale Prediction

> Diffusion models 在时间上 iterative sample（denoising steps）。VAR 在尺度上 iterative sample — 它先预测 1x1 token，再预测 2x2，再预测 4x4，一直到最终分辨率，每个 scale 都 condition on 前一个。2024 年论文显示，VAR 在图像生成上匹配 GPT-style scaling laws，并在同等 compute budget 下击败 DiT。本课构建核心机制。

**类型：** 构建
**语言：** Python（with PyTorch）
**前置要求：** 阶段 7 第 03 课（Multi-Head Attention），阶段 8 第 06 课（DDPM）
**时间：** ~90 分钟

## 问题

Autoregressive generation 主导语言建模，因为它可预测地 scale：更多 compute、更多参数、更低 perplexity、更好输出。2024 年之前，图像生成主要有两种 AR 尝试：PixelRNN/PixelCNN（pixel-by-pixel）和 DALL-E 1 / Parti / MuseGAN（在 VQ-VAE codes 上 token-by-token）。

两者都遇到 generation-order 问题。Pixels 和 tokens 排列在 2D grid 中，但 AR model 必须按 1D raster order 访问它们。早期 corner pixel 并不知道图像最终会变成什么。Generation quality 比 GPT-on-text 的 scale 更差，也从未在 matched compute 下达到 diffusion-model quality。

VAR 通过改变被生成的对象来修复 generation-order 问题。它不是在空间上逐个预测 image tokens，而是在逐渐提高分辨率的过程中预测整张图像。Step 1：预测 1x1 token（整张图的 “summary”）。Step 2：预测 2x2 token grid（更粗特征）。Step 3：预测 4x4 grid。Step K：预测最终 `(H/8)x(W/8)` grid。

每个 scale attend 到所有 previous scales（按 “scale order” causally），并在自己的 scale 内并行。Order 问题消失：scale k 上的整张图在一次 transformer pass 中生成。

## 概念

### VQ-VAE Multi-Scale Tokenizer

VAR 需要一个 **multi-scale discrete tokenizer**。对于图像 x，它产生一串逐渐更高分辨率的 token grids：

```
x -> encoder -> latent f
f -> tokenize at 1x1: token grid z_1 of shape (1, 1)
f -> tokenize at 2x2: token grid z_2 of shape (2, 2)
...
f -> tokenize at (H/p)x(W/p): token grid z_K of shape (H/p, W/p)
```

每个 z_k 使用同一个 codebook（典型大小 4096-16384）。每个 scale 的 tokenization 不是独立的 — 它被训练成让各 scale residual 求和后重建 f：

```
f ≈ upsample(embed(z_1), target_size) + ... + upsample(embed(z_K), target_size)
```

这是 **residual VQ** 变体。Scale k 捕捉 scales 1..k-1 遗漏的内容。Decoder 接收所有 scale embeddings 的和并产生图像。

Multi-scale VQ tokenizer 只训练一次（像 VQGAN），然后冻结。所有生成工作都由上层 autoregressive model 完成。

### Next-Scale Prediction

生成模型是一个 transformer，它看到所有 previous scales 的 tokens，并预测 next scale 的 tokens。

Input sequence structure：
```
[START, z_1 tokens, z_2 tokens, z_3 tokens, ..., z_K tokens]
```

Position embeddings 同时编码 scale index 和该 scale 内的 spatial position。Attention 在 scale order 上 causal：scale k、position (i, j) 的 token 可以 attend 到 scales 1..k 的所有 tokens，以及 scale k 内在所用 intra-scale order 中更早的 tokens（VAR 使用固定 positional attention，没有 intra-scale causality — 一个 scale 内所有 positions 并行预测）。

Training loss：在每个 scale k，用所有 prior-scale tokens 预测 tokens z_k。离散 VQ codes 上的 cross-entropy loss。结构与 GPT 相同，只是“sequence”现在具有 scale structure。

### Generation

推理时：
```
generate z_1 = sample from p(z_1)                    # 1 token
generate z_2 = sample from p(z_2 | z_1)              # 4 tokens in parallel
generate z_3 = sample from p(z_3 | z_1, z_2)         # 16 tokens in parallel
...
decode: f = sum of embed-and-upsample scales 1..K
image = VAE_decoder(f)
```

当 K = 10 scales 时，generation 是 10 次 transformer forward passes。每次 pass 并行生成整个 scale — scale 内没有 per-token autoregression。对 256x256 图像，大约是 10 passes，而 DiT 是 28-50。

### 为什么 Next-Scale 胜过 Next-Token

三个结构性优势：
1. **Coarse-to-fine 符合自然图像统计。** 人类视觉感知和图像数据集都体现 scale-dependent regularities：低频结构稳定且可预测；高频细节 condition on 低频内容。Next-scale prediction 利用这一点。
2. **Scale 内并行生成。** 不同于 GPT-style token AR，VAR 一步产生一个 scale 的所有 tokens。有效 generation length 是 log-scale，而不是 linear。
3. **没有 generation order bias。** Scale k 的 tokens 看到完整 scale k-1；不存在 “left-of” 或 “above” bias 迫使早期 tokens 在获得后期 context 之前先做承诺。

### Scaling Law

Tian et al. 证明，VAR 在 ImageNet 上的 FID 遵循 power-law scaling curve — 就像 GPT 的 perplexity。参数或 compute 翻倍，会可靠地减少 error。这是第一个像语言模型一样干净展示这种 scaling behavior 的 image-generative model。结果是 VAR-scale 预测可以从 compute 中预测，而不是对每个架构做经验猜测。

### 与 Diffusion 的关系

VAR 和 diffusion 共享同一个 data-compression 故事：二者都把生成问题拆成一串更容易的子问题。

- Diffusion：逐渐加入噪声，学习撤销一步。
- VAR：逐渐增加分辨率，学习预测下一个 scale。

它们是穿过同一问题的不同轴。两者都产生 tractable conditional distributions。经验上，VAR 推理更快（passes 更少，scale 内全并行），并在 class-conditional ImageNet 上匹配或击败 DiT。Text-conditional VAR（VARclip、HART）是活跃研究方向。

## 构建它

在 `code/main.py` 中你会：
1. 在 synthetic “image” data（2D Gaussian rings）上构建 tiny **multi-scale VQ tokenizer**。
2. 训练 **VAR-style transformer** 来 next-scale-predict tokens。
3. 通过调用 transformer 4 次（4 scales）采样并 decode。
4. 验证 scale-ordered training 让 generation 在 scale 内并行。

这是 toy implementation。重点是亲眼看到 scale-structured attention mask 和 parallel-within-scale generation 实际工作。

## 交付它

本课产出 `outputs/skill-var-tokenizer-designer.md` — 一个用于设计 multi-scale tokenizer 的 skill：scales 数、scale ratios、codebook size、residual sharing、decoder architecture。

## 练习

1. **Scale count ablation。** 用 4、6、8、10 scales 训练 VAR。测量 reconstruction quality vs autoregressive passes 数。Scales 越多 = residuals 越细 = 质量更好但 passes 更多。

2. **Codebook size。** 用 codebook sizes 512、4096、16384 训练 tokenizers。更大 codebooks 给更好 reconstruction，但 prediction 更难。找到拐点。

3. **Parallel-within-scale check。** 对训练好的 VAR，显式测量 attention pattern。Scale k 内，模型是否 attend 到 cross-scale positions 但不 attend intra-scale？验证 mask implementation。

4. **VAR vs DiT scaling。** 对同一个 ImageNet class-conditional 任务，在 matched param budgets（例如 33M、130M、458M）下训练 VAR 和 DiT。画 FID vs compute。VAR 应该在每个 size 上领先 DiT — 在小规模复现论文结果。

5. **Text conditioning。** 扩展 VAR，让它通过 adaLN 接收 text embedding（CLIP pooled）作为额外 conditioning input。这是 HART 配方。它在 text-aligned sampling 上能让 FID 改善多少？

## 关键词

| 术语 | 大家怎么说 | 实际含义 |
|------|----------------|----------------------|
| VAR | “Visual AutoRegressive” | 通过 VQ token grids 金字塔上的 next-scale prediction 进行图像生成 |
| Next-scale prediction | “先粗后细预测” | 模型按逐渐升高的 resolution scales 预测 tokens，并 condition on 所有 previous scales |
| Multi-scale VQ tokenizer | “Residual VQ” | 产生 K 个递增分辨率 token grids 的 VQ-VAE，decoder 汇总所有 scales |
| Scale k | “Pyramid level k” | K 个 resolution levels 之一，从 k=1 的 1x1 到 k=K 的 (H/p)x(W/p) |
| Parallel-within-scale | “每 scale 一次 forward” | Scale k 的所有 tokens 在一次 transformer pass 中预测，而不是 autoregressively |
| Causal-across-scales | “Scale-ordered attention” | Scale k 的 token 可以 attend 到 scales 1..k，但不能 attend 到 k+1..K |
| Residual VQ | “Additive tokenization” | 每个 scale 的 tokens 编码 lower scales 留下的 residual；decoder 汇总所有 scale embeddings |
| VAR scaling law | “Image GPT scaling” | FID 像语言模型 perplexity 一样，按 compute 遵循可预测 power law |
| HART | “Hybrid VAR + text” | Text-conditional VAR 变体，把 MaskGIT-style iterative decoding 与 VAR 的 scale structure 结合 |
| Scale position embedding | “(scale, row, col) triple” | Positional encoding 同时携带 scale index 和 scale 内 spatial coordinates |

## 延伸阅读

- [Tian et al., 2024 — "Visual Autoregressive Modeling: Scalable Image Generation via Next-Scale Prediction"](https://arxiv.org/abs/2404.02905) — VAR 论文，标准参考
- [Peebles and Xie, 2022 — "Scalable Diffusion Models with Transformers"](https://arxiv.org/abs/2212.09748) — DiT，diffusion 对比 baseline
- [Esser et al., 2021 — "Taming Transformers for High-Resolution Image Synthesis"](https://arxiv.org/abs/2012.09841) — VQGAN，VAR multi-scale tokenizer 扩展的 tokenizer family
- [van den Oord et al., 2017 — "Neural Discrete Representation Learning"](https://arxiv.org/abs/1711.00937) — VQ-VAE，离散图像 tokenization 基础
- [Tang et al., 2024 — "HART: Efficient Visual Generation with Hybrid Autoregressive Transformer"](https://arxiv.org/abs/2410.10812) — text-conditional VAR
