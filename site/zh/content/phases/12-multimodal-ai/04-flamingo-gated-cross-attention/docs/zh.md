# Flamingo 与 Few-Shot VLM 的 Gated Cross-Attention

> DeepMind 的 Flamingo（2022）先于其他人做了两件事。它证明了单个模型可以处理任意 interleaved 的图像、视频和文本序列。它还证明了 VLM 可以做 in-context learning：给一个带三个示例（image, caption）pair 的 few-shot prompt，模型无需任何梯度步骤就能为新图像生成 caption。机制是 gated cross-attention layers，插入 frozen LLM 的既有层之间，并带有一个从零开始的 learned tanh gate，使 LLM 的文本能力在初始化时被保留。本课讲解 Flamingo 的 Perceiver resampler 和 gated cross-attention 架构，它是 Gemini interleaved inputs 与 Idefics2 visual tokens 的祖先。

**类型：** 学习
**语言：** Python（stdlib，gated cross-attention + Perceiver resampler demo）
**前置要求：** 阶段 12 · 03（BLIP-2 Q-Former）
**时间：** ~120 分钟

## 学习目标

- 解释 gated cross-attention 如何通过 tanh(gate) = 0，在初始化时保留 frozen LLM 的文本能力。
- 走通 Perceiver resampler：N 个 image patches -> K 个固定 “latent” queries，通过 cross-attention 实现。
- 描述 Flamingo 如何用尊重图像位置的 causal masking 处理 interleaved image-text sequences。
- 复现一个 few-shot multimodal prompt 结构（3 个 image-caption 示例，然后一个 query image）。

## 问题

BLIP-2 把 32 个 visual token 喂入 frozen LLM 的 input layer。它适合每个 prompt 一张图。但如果你想喂入很多张图，并与文本交错，比如“这是 image A，给它写 caption；这是 image B，给它写 caption；现在这是 image C，给它写 caption”？LLM 的 self-attention 需要在单一 stream 中处理 image token 和 text token，而且哪些位置能 attend 到哪些图像会变得很麻烦。

Flamingo 的答案是：完全不要改变 LLM 的 input stream。在既有 LLM block 之间插入额外 cross-attention layers。Text token 仍然像往常一样流经 LLM 的 causal self-attention。在每隔几个 LLM block 之间，text token 也通过一个新的 gated layer 对 image features 做 cross-attention。Gate（初始化为零）意味着第 0 步时这些新层是 no-op，模型行为与 pretrained LLM 完全相同。随着训练推进，gate 打开，视觉信息开始流入。

Flamingo 回答的第二个问题是：如何处理每个 prompt 中数量可变的图像（0、1 或很多）？答案是 Perceiver resampler。它是一个小型 cross-attention 模块，接收任意数量的 patches，并产生固定数量的 visual latent tokens。不管 prompt 中有多少图像，LLM cross-attention layer 看到的 shape 都相同。

## 概念

### Frozen LLM

Flamingo 从一个 frozen Chinchilla 70B LLM 开始。全部 70B 权重不动。既有 text self-attention 和 FFN 正常工作。

### Perceiver resampler

对于 prompt 中的每张图，ViT 会产生 N 个 patch token。Perceiver resampler 有 K 个固定 learnable latents（Flamingo 使用 K=64）。每个 resampler block 包含两个子步骤：

1. Cross-attention：K 个 latents attend 到 N 个 patch token（Q 来自 latents，K/V 来自 patches）。
2. Latents 内部的 self-attention + FFN。

经过 6 个 resampler block 后，输出是 K=64 个 dim 1024 的 visual token，与 ViT 产生了多少 patch 无关。224x224 图像（196 patches）和 480x480 图像（900 patches）都会输出 64 个 resampler token。

对于视频，resampler 会按时间应用：每帧 patches 产生 64 个 latents，temporal positional encoding 让模型区分 t=0 与 t=N。完整视频变成 T * 64 个 visual token。

### Gated cross-attention

在 frozen LLM 每 M 层之间（Flamingo 使用 M=4）插入一个新的 gated cross-attention block：

```
x_after_llm_block = llm_block(x_before)
cross = cross_attn(x_after, resampler_output)
gated = tanh(alpha) * cross + x_after
x_before_next_block = gated
```

- `alpha` 是一个初始化为零的可学习标量。
- `tanh(0) = 0`，所以初始化时 gated branch 贡献为零。
- 随着 `alpha` 远离零，cross-attention 贡献会平滑增长。
- Residual connection 意味着即使 gate 完全打开，也不会覆盖 LLM 的文本 representation，只是在其上添加视觉信息。

这是 Flamingo 中最重要的设计选择：visual conditioning 是 additive、gated，并且初始化时为零。第 0 步的 Flamingo 在 text-only inputs 上就是一个完美的 Chinchilla 70B。

### Interleaved inputs 的 masked cross-attention

在像 “<image A> caption A <image B> caption B <image C> ?” 这样的 prompt 中，每个 text token 应该只能看到序列中位于它之前的图像。Cross-attention mask 强制：位置 `t` 的 text token 只能 attend 到 image index `i < i_t` 的 image resampler tokens，其中 `i_t` 是位置 `t` 前最近的图像。“只看最近的前置图像”或“看所有前置图像”都是有效选择；Flamingo 选择了前者。

### In-context few-shot learning

Flamingo prompt 看起来像：

```
<image1> A photo of a cat. <image2> A photo of a dog. <image3> A photo of a
```

模型看到 completion pattern 后会输出 “bird”（或 image3 中展示的任何东西）。没有梯度步骤。Frozen LLM 的 in-context learning 能力通过 gated cross-attention 延续了下来，这正是这篇论文的 punchline，也是它重要的原因。

### 训练数据

Flamingo 在三个数据集上训练：

1. MultiModal MassiveWeb（M3W）：43M web pages，包含 interleaved images and text，并重建 reading order。
2. Image-Text Pairs（ALIGN + LTIP）：4.4B pairs。
3. Video-Text Pairs（VTP）：27M 短视频 clips。

OBELICS（2023）是 interleaved web corpus 的开放复现，Idefics、Idefics2 和大多数开放 “Flamingo-like” 模型都在其上训练。

### OpenFlamingo 与 Otter

OpenFlamingo（2023）是开放复现。架构相同（Perceiver resampler + gated cross-attention on frozen LLaMA or MPT）。Checkpoint 有 3B、4B、9B。由于 base LLM 更小、数据更少，质量落后 Flamingo。

Otter（2023）基于 OpenFlamingo，并在 MIMIC-IT（一个 multimodal instructions 数据集）上做 instruction tuning，表明 gated cross-attention 也适合 instruction following。

### 后代

- Idefics / Idefics2 / Idefics3：Hugging Face 的 gated cross-attention lineage，逐步简化（Idefics2 放弃 resampler，改用 direct patch tokens with adaptive pooling）。
- Flamingo-to-Chameleon transition：到 2024 年许多团队转向 early-fusion（第 12.11 课）；但当需要冻结 backbone 时，Flamingo-style gated cross-attention 仍然留在生产中。
- Gemini 的 interleaved input：概念上继承了 Flamingo 的 interleaved-format 灵活性，尽管具体机制是 proprietary。

### 与 BLIP-2 对比

| | BLIP-2 | Flamingo |
|---|---|---|
| Visual bridge | Q-Former once at input | Gated cross-attention at every M layers |
| Visual tokens | 32 per image | 64 per image per cross-attn layer |
| Frozen LLM | Yes | Yes |
| Few-shot in-context | Weak | Strong — the paper's centerpiece |
| Interleaved inputs | No native support | Yes, the design target |
| Training data | 130M pairs | 1.3B pairs + 43M interleaved pages |
| Parameter count | 188M trained | ~10B trained (cross-attn layers) |
| Compute | Days on 8 A100s | Weeks on thousands of TPUv4 |

预算内做 single-image VQA，选 BLIP-2。需要 interleaved、few-shot 或 multi-image reasoning，选 Flamingo/Idefics2。

## 使用它

`code/main.py` 演示：

1. 在 36 个 fake patch token 上运行 Perceiver resampler，使用 8 个 learnable latents（纯 Python cross-attention）。
2. 一个 gated cross-attention 步骤：`alpha = 0` -> output 等于 input（LLM unchanged），然后 `alpha = 2.0` -> 混入视觉贡献。
3. 一个 interleaved-mask builder，能为 “(image 1) (text 1) (image 2) (text 2)” sequence 生成 2D attention mask。

## 交付它

本课产出 `outputs/skill-gated-bridge-diagnostic.md`。给定一个 open VLM config（resampler Y/N、cross-attn frequency、gate scheme），它会识别 Flamingo lineage 元素并解释 freezing strategy。它适合用于 debug 为什么 fine-tune 降低了文本性能（答案通常是：gate 开得太快太大）。

## 练习

1. 计算 Flamingo-9B 的 visual parameter count：9B LLM + 1.4B gated cross-attention layers + 64M resampler。被训练的参数占总参数的比例是多少？

2. 用 PyTorch 实现 gated residual `y = tanh(alpha) * cross + x`。实验展示当 `alpha=0` 时，初始化处 `y==x` 完全成立。

3. 阅读 OpenFlamingo 第 3.2 节（arXiv:2308.01390），了解它们如何在每个 prompt 图像数量不同的 batch 中处理 multiple images。描述 padding strategy。

4. 为什么 Flamingo 的 cross-attention mask 让 text token 只 attend 到*最近的*前置图像，而不是所有前置图像？阅读 Flamingo 论文第 2.4 节并解释 tradeoff。

5. In-context few-shot：为一个新的 Flamingo variant 构造 4 个“image -> color of main object”示例 prompt。描述当示例数量从 0 增加到 8 时，预期 accuracy pattern 如何变化。

## 关键术语

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Perceiver resampler | “Fixed-latent cross-attention” | 从可变数量 input patches 产生 K 个固定 token 的模块 |
| Gated cross-attention | “Tanh-gated bridge” | Residual layer `y = tanh(alpha)*cross + x`，alpha 可学习，初始化为 0 |
| Interleaved input | “Mixed sequence” | 图像和文本按 reading order 自由混合的 prompt format |
| Frozen LLM | “No LLM gradients” | Text LLM 权重不更新；只训练 resampler + cross-attn layers |
| Few-shot | “In-context examples” | 在 prompt 中给几个（image, answer）pair；模型无需 finetuning 即可泛化 |
| OBELICS | “Interleaved web corpus” | 包含图像和按 reading order 排列文本的 141M web pages 开放数据集 |
| Chinchilla | “70B frozen base” | Flamingo 使用的 frozen text LLM，来自 DeepMind 的 Chinchilla 论文 |
| Gate schedule | “How alpha moves” | 训练期间 cross-attention gate 打开的速度 |
| Cross-attn frequency | “Every M layers” | 插入 gated cross-attention block 的频率；Flamingo 使用 M=4 |
| OpenFlamingo | “Open reproduction” | MosaicML/LAION 的 3-9B 开放 checkpoint；架构与 Flamingo 相同 |

## 延伸阅读

- [Alayrac et al. — Flamingo (arXiv:2204.14198)](https://arxiv.org/abs/2204.14198) — 原始论文。
- [Awadalla et al. — OpenFlamingo (arXiv:2308.01390)](https://arxiv.org/abs/2308.01390) — 开放复现。
- [Laurençon et al. — OBELICS (arXiv:2306.16527)](https://arxiv.org/abs/2306.16527) — interleaved web corpus。
- [Jaegle et al. — Perceiver IO (arXiv:2107.14795)](https://arxiv.org/abs/2107.14795) — 通用 Perceiver 架构。
- [Li et al. — Otter (arXiv:2305.03726)](https://arxiv.org/abs/2305.03726) — instruction-tuned Flamingo 后代。
- [Laurençon et al. — Idefics2 (arXiv:2405.02246)](https://arxiv.org/abs/2405.02246) — Flamingo approach 的现代简化版。
