# Transfusion：一个 Transformer 中的 Autoregressive Text + Diffusion Image

> Chameleon 和 Emu3 把赌注全押在 discrete tokens 上。它们能工作，但 quantization bottleneck 很明显，image quality 会在 continuous-space diffusion models 之下平台化。Transfusion（Meta，Zhou 等人，2024 年 8 月）反向下注：保留 continuous image，完全丢掉 VQ-VAE，并用两个 loss 训练一个 transformer。Text tokens 用 next-token-prediction。Image patches 用 flow-matching / diffusion loss。两个 objectives 优化同一组权重。Stable Diffusion 3 底层架构（MMDiT）是它的近亲。本课阅读 Transfusion thesis，构建一个 toy two-loss trainer，并追踪让一个 transformer 同时做两件事的 attention mask。

**类型：** 构建
**语言：** Python（stdlib，MNIST-scale toy 上的 two-loss trainer）
**前置要求：** 阶段 12 · 11（Chameleon），阶段 8（Generative AI）
**时间：** ~180 分钟

## 学习目标

- 接线一个 transformer，让它在同一个 backbone 上运行两个 loss（text tokens 上的 NTP、image patches 上的 diffusion MSE）。
- 解释为什么 image patches 内部 bidirectional attention 加 text tokens 上 causal attention 是正确的 mask choice。
- 在 compute、quality 和 code complexity 上比较 Transfusion-style（continuous images, diffusion loss）与 Chameleon-style（discrete images, NTP）。
- 说出 MMDiT 的贡献：每个 block 有 modality-specific weights，residual stream 上 joint attention。

## 问题

Discrete vs continuous image tokens 的争论比 LLM 更早。Continuous representations（raw pixels、VAE latents）保留细节。Discrete tokens（VQ indices）适合 transformer 的原生 vocabulary，但在 quantization step 丢失细节。

Chameleon / Emu3 走 discrete：一个 loss，一个 architecture，但 image fidelity 被 tokenizer quality 限制。

Diffusion models 走 continuous：卓越 image quality，但与 LLM 是 separate model，需要复杂 noise-schedule engineering，也没有与 text generation 的干净整合。

Transfusion 问：能不能两者兼得？保留 continuous images，仍然训练一个模型，用两个 loss 缝进一个 gradient step。

## 概念

### Two-loss architecture

单个 decoder-only transformer 处理一个包含以下内容的序列：

- Text tokens（离散，来自 BPE vocab）。
- Image patches（连续，16x16 pixel blocks 通过 linear embedding 投影到 hidden dim，与 ViT encoder 的输入相同）。
- 标记 continuous patches 所在位置的 `<image>` 和 `</image>` tags。

Forward pass 只运行一次。Loss 按 token 选择两个 head 之一：

- 对 text tokens：在 vocab-logits head 上做标准 cross-entropy。
- 对 image patches：在 continuous patches 上做 diffusion loss，即预测加入每个 patch 的 noise。

梯度流过共享 transformer body。两个 loss 同时改进共享权重。

### Attention mask：causal text + bidirectional image

Text tokens 必须是 causal 的：不能让 text token attend 到未来 text，否则 teacher forcing 会被破坏。但 image patches 表示同一个 snapshot，它们应当在同一个 image block 内互相 bidirectionally attend。

Mask：

```
M[i, j] = 1 if:
  (i is text and j is text and j <= i)   # causal for text
  OR (i is image and j is image and same_image_block(i, j))   # bidirectional within image
  OR (i is text and j is image and j < i_image_end)   # text attends to previous images
  OR (i is image and j is text and j < i_image_start)   # image attends to preceding text
```

训练与推理时实现为 block-triangular mask。

### Transformer 内部的 diffusion loss

Diffusion loss 是标准的：给 image patch 加噪声，要求模型预测噪声（或等价地预测 clean patch）。Transfusion 版本使用 flow matching，即从 noisy 到 clean 预测 velocity field。

训练时：
1. 对每个 image patch x0，采样随机 timestep t。
2. 采样 noise ε，计算 xt = (1-t) * x0 + t * ε（flow matching 的线性插值）。
3. Transformer 预测 v_theta(xt, t)；loss = MSE(v_theta(xt, t), ε - x0)。
4. 与同一序列中的 text NTP losses 一起 backprop。

推理时，generation 是：
- Text tokens：标准 autoregressive sampling。
- Image patches：以先前 text tokens 为条件运行 diffusion sampling loop（通常 10-30 步）。

### MMDiT：Stable Diffusion 3 的变体

Stable Diffusion 3（Esser 等人，2024 年 3 月）大约与 Transfusion 同期发布了 MMDiT（Multimodal Diffusion Transformer）。两者是 sibling architecture。

MMDiT 的关键差异：

- 每个 block 有 modality-specific weights。每个 transformer block 为 text tokens 与 image patches 分别有 Q、K、V 和 MLP 权重。Attention 是 joint（cross-modality）；其他部分 modality-specific。
- Rectified flow training。一种特定 flow-matching 变体，相比 DDPM 采样清晰、数学更简单。
- Scale。MMDiT 是 SD3 的 backbone（2B 和 8B 参数变体）。Transfusion 论文 scale 到 7B。

两者收敛到同一个核心想法：一个 transformer 对 text 做 NTP，对 continuous image representations 做 diffusion。

### 为什么它超过 Chameleon-style

Continuous-diffusion 与 discrete-NTP 在 image generation 上的质量差距是可测的。Transfusion 论文报告：

- 在 7B params 下，在 FID 上比同尺寸 Chameleon-style model 好 3-5 分。
- 不需要 tokenizer training，image encoder 更简单（Linear projection to hidden，与 ViT input layer 相同）。
- Inference 可并行 denoise image patches，不像 autoregressive image tokens。

缺点：Transfusion 是 dual-loss model，训练动态更麻烦。Loss weights 需要调参。NTP 与 diffusion 的 schedule mismatch 可能导致一个 head 占主导。

### 下游分支

Janus-Pro（第 12.15 课）通过 decoupling 用于 understanding 和 generation 的 vision encoder 改进 Transfusion 的思路：一个用 SigLIP，另一个用 VQ，同时共享 transformer body。Show-o（第 12.14 课）把 diffusion 换成 discrete-diffusion（masked prediction）。Unified-generation family 在 Transfusion 后快速分叉。

会发图的 2026 production VLMs，比如 Gemini 3 Pro、GPT-5、Claude Opus 4.7 的 image generation path，几乎肯定使用这个 family 的某个后代。细节是 proprietary。

## 使用它

`code/main.py` 在一个 tiny MNIST-like problem 上构建 toy Transfusion：

- Text captions 是描述数字（0-9）的短整数序列。
- Images 是 4x4 byte grids。
- 一对 shared-weight linear projections 充当 transformer stand-in；text 上做 NTP loss，noisy patches 上做 MSE loss。
- Training loop 交替两个 loss，attention mask 显式构建。
- Generation 在一个 forward pass 中产生 text caption 和 4x4 image。

Transformer 是 toy。真正的 artifacts 是 two-loss plumbing、attention mask construction 和 inference loop。

## 交付它

本课产出 `outputs/skill-two-loss-trainer-designer.md`。给定一个新的 multimodal training task（text + image、text + audio、text + video），它会设计 two-loss schedule（loss weights、mask shape、shared vs modality-specific blocks），并标记实现风险。

## 练习

1. 一个 Transfusion-style model 训练时有 70% text tokens 和 30% image patches。Image diffusion loss 的 magnitude 约为 text NTP loss 的 10 倍。什么 loss weights 能平衡它们？

2. 为序列 `[T, T, <image>, P, P, P, P, </image>, T]` 实现 block-triangular mask。把每个 entry 标为 0 或 1。

3. MMDiT 有 modality-specific QKV weights。相比 Transfusion 的 fully-shared transformer，这会增加多少 parameter count overhead？在 7B params 下值得吗？

4. Generation：给定 text prompt，模型先运行 NTP 50 个 token，然后遇到 `<image>`，再对 256 个 patches 运行 20 步 denoise diffusion。一共多少 forward passes？

5. 阅读 SD3 论文第 3 节。描述 rectified flow，以及为什么它比 DDPM 用更少 inference steps 收敛。

## 关键术语

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| Two-loss training | “NTP + diffusion” | 单个 transformer 在同一个 gradient step 中同时优化 text tokens 上的 cross-entropy 和 continuous image patches 上的 MSE |
| Flow matching | “Rectified flow” | 预测从 noise 到 clean data 的 velocity field 的 diffusion 变体；数学比 DDPM 更简单 |
| MMDiT | “Multimodal DiT” | Stable Diffusion 3 架构：joint attention，modality-specific MLPs and norms |
| Block-triangular mask | “Causal text + bidirectional image” | Attention mask，对 text tokens causal，对 image regions bidirectional |
| Continuous image representation | “No VQ” | Image patches 是 real-valued vectors，而不是 integer codebook indices |
| Velocity prediction | “v-parameterization” | Network output 是 noise 与 data 之间的 velocity field，而不是 noise 本身 |

## 延伸阅读

- [Zhou et al. — Transfusion (arXiv:2408.11039)](https://arxiv.org/abs/2408.11039)
- [Esser et al. — Stable Diffusion 3 / MMDiT (arXiv:2403.03206)](https://arxiv.org/abs/2403.03206)
- [Peebles & Xie — DiT (arXiv:2212.09748)](https://arxiv.org/abs/2212.09748)
- [Zhao et al. — MonoFormer (arXiv:2409.16280)](https://arxiv.org/abs/2409.16280)
- [Xie et al. — Show-o (arXiv:2408.12528)](https://arxiv.org/abs/2408.12528)
