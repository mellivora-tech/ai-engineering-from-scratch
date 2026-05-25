# Vision Transformers 与 Patch-Token 原语

> 在任何多模态之前，图像都必须先变成 transformer 能吃进去的一串 token。2020 年的 ViT 论文用 16x16 像素 patch、线性投影和位置嵌入回答了这个问题。五年后，所有 2026 年 frontier model（Claude Opus 4.7 的 2576px native、Gemini 3.1 Pro、Qwen3.5-Omni）仍然从这里开始，只是编码器从 ViT 变成 DINOv2、SigLIP 2，加入了 register token，位置方案变成 2D-RoPE，但这个原语保留下来了。本课端到端读懂 patch-token pipeline，并用 stdlib Python 构建它，让阶段 12 后续内容对“visual tokens”有一个具体的心智模型。

**类型：** 学习
**语言：** Python（stdlib，patch tokenizer + geometry calculator）
**前置要求：** 阶段 7（Transformers），阶段 4（Computer Vision）
**时间：** ~120 分钟

## 学习目标

- 将一个 HxWx3 图像转换成带有正确位置编码的 patch token 序列。
- 针对给定的（patch size、resolution、hidden dim、depth）ViT，计算序列长度、参数量和 FLOPs。
- 说出让 ViT 从 2020 年研究原型走向 2026 年生产系统的三个升级：self-supervised pretraining（DINO / MAE）、register tokens、native-resolution packing。
- 为下游任务在 CLS pooling、mean pooling 和 register tokens 之间做选择。

## 问题

Transformer 处理的是向量序列。文本天然就是序列（byte 或 token）。图像则是带有三个颜色通道的 2D 像素网格，不是序列。如果你把每个像素都展平，一个 224x224 RGB 图像会变成 150,528 个 token，而这么长的 self-attention 根本不可行（复杂度随序列长度平方增长）。

2020 年之前的做法是在前面接一个 CNN 特征提取器：ResNet 产生一个由 2048 维向量组成的 7x7 feature map，再把这 49 个 token 喂给 transformer。这能工作，但继承了 CNN 的偏置（translation equivariance、本地感受野），也失去了 transformer 对规模的胃口。

Dosovitskiy 等人（2020）问了一个很直接的问题：如果跳过 CNN 会怎样？把图像切成固定大小的 patch（比如 16x16 像素），把每个 patch 线性投影为一个向量，加入位置嵌入，然后把序列喂给普通 transformer。当时这很离经叛道，等于是没有卷积的视觉模型。只要数据足够多（JFT-300M，然后是 LAION），它就在 ImageNet 上超过了 ResNet，并继续提升。

到 2026 年，ViT 原语已经是无可争议的基础。每个 open-weights VLM 的 vision tower 都是某个后代（DINOv2、SigLIP 2、CLIP、EVA、InternViT）。问题不再是“要不要用 patch？”，而是“用多大的 patch size、什么 resolution schedule、什么 pretraining objective、什么 positional encoding”。

## 概念

### Patch 作为 token

给定形状为 `(H, W, 3)` 的图像 `x` 和 patch size `P`，你把图像切成 `(H/P) x (W/P)` 的不重叠 patch 网格。每个 patch 是一个 `P x P x 3` 的像素立方体。把每个立方体展平成一个 `3 P^2` 向量。应用形状为 `(3 P^2, D)` 的共享线性投影 `W_E`，把每个 patch 映射到模型 hidden dimension `D`。

对于 ViT-B/16 这个经典配置：
- Resolution 224，patch size 16 -> grid 14x14 -> 196 个 patch token。
- 每个 patch 是 `16 x 16 x 3 = 768` 个像素值，投影到 `D = 768`。
- 加入一个可学习的 `[CLS]` token -> 序列长度 197。

Patch projection 在数学上等价于一个 kernel size 为 `P`、stride 为 `P`、输出通道数为 `D` 的 2D convolution。生产代码实际上就是这样实现的，即 `nn.Conv2d(3, D, kernel_size=P, stride=P)`。“线性投影”是概念表述；kernel 表述更高效。

### 位置嵌入

Patch 本身没有内在顺序，transformer 看到的是一个集合。早期 ViT 加入可学习的 1D positional embedding（每个位置一个 768 维向量，共 197 个）。它能工作，但会把模型绑到训练分辨率上：推理时如果改变 grid，就必须插值 position table。

现代视觉 backbone 使用 2D-RoPE（Qwen2-VL 的 M-RoPE、SigLIP 2 的默认方案）或 factorized 2D positions。2D-RoPE 会根据 patch 的（row, column）索引旋转 query 和 key 向量，因此模型可以从旋转角度推断相对 2D 位置。没有 position table。模型在推理时可以处理任意 grid size。

### CLS token、pooled output 与 register tokens

图像级 representation 是什么？三种选择共存：

1. `[CLS]` token。把一个可学习向量 prepend 到 patch 序列前。经过所有 transformer block 后，CLS token 的 hidden state 就是图像 representation。继承自 BERT。原始 ViT、CLIP 使用它。
2. Mean pool。对 patch token 的输出 hidden state 求平均。SigLIP、DINOv2 和大多数现代 VLM 使用它。
3. Register tokens。Darcet 等人（2023）观察到，没有显式 sink token 训练的 ViT 会产生高范数的“artifact”patch，并劫持 self-attention。加入 4-16 个可学习 register token 可以吸收这部分负载，并提升 dense-prediction 质量（segmentation、depth）。DINOv2 和 SigLIP 2 都带有 register。

这个选择会影响下游任务。CLS 适合分类。对于把 patch token 喂给 LLM 的 VLM，你完全跳过 pooling，每个 patch 都变成 LLM 输入 token。Register 会在交接前丢弃（它们是脚手架，不是内容）。

### 预训练：supervised、contrastive、masked、self-distilled

2020 年的 ViT 使用 JFT-300M 的 supervised classification 预训练。它很快被这些方法取代：

- CLIP（2021）：在 400M 图文对上做 contrastive image-text。第 12.02 课。
- MAE（2021，He 等人）：mask 75% 的 patch，重建像素。Self-supervised，适用于纯图像。
- DINO（2021）/ DINOv2（2023）：student-teacher self-distillation，无标签、无 caption。2023 年的 DINOv2 ViT-g/14 是最强的纯视觉 backbone，也是“dense features”用例的默认选择。
- SigLIP / SigLIP 2（2023，2025）：使用 sigmoid loss 的 CLIP，并用 NaFlex 支持 native aspect ratio。它是 2026 年 open VLM（Qwen、Idefics2、LLaVA-OneVision）中的主流 vision tower。

你选择的预训练方式决定了 backbone 擅长什么：CLIP/SigLIP 擅长与文本做语义匹配，DINOv2 擅长 dense visual features，MAE 适合作为下游 finetuning 的起点。

### Scaling laws

ViT scaling（Zhai 等人，2022）证明了 ViT 的质量会随 model size、data size 和 compute 遵循可预测规律。在固定 compute 下：
- 更大的模型 + 更多数据 -> 更好的质量。
- Patch size 是控制 sequence length 与 fidelity 的杠杆。Patch 14（DINOv2/SigLIP SO400m 常用）比 patch 16 每张图产生更多 token；对 OCR 和 dense tasks 更好，但更慢。
- Resolution 是另一个大杠杆。从 224 到 384 到 512 几乎总是有帮助，但 FLOPs 成本按平方增长。

ViT-g/14（1B params、patch 14、resolution 224 -> 256 token）和 SigLIP SO400m/14（400M params、patch 14）是 2026 年 open VLM 的两个主力 encoder。

### ViT 的参数量

完整计算在 `code/main.py` 中。对于 224 分辨率下的 ViT-B/16：

```
patch_embed = 3 * 16 * 16 * 768 + 768  =  591k
cls + pos    = 768 + 197 * 768          =  152k
block        = 4 * 768^2 (QKVO) + 2 * 4 * 768^2 (MLP) + 2 * 2*768 (LN)
             = 12 * 768^2 + 3k          =  7.1M
12 blocks    = 85M
final LN    = 1.5k
total       ≈ 86M
```

在加载 checkpoint 之前，先用这种方式粗估每个 ViT。Backbone size 会决定任何下游 VLM 的 VRAM 下限。

### 2026 年生产配置

2026 年大多数 open VLM 搭载的 encoder 是 native resolution（NaFlex）下的 SigLIP 2 SO400m/14。它有：
- 400M 参数。
- Patch size 14，默认 resolution 384 -> 每张图 729 个 patch token。
- 图像级任务使用 mean pool；做 VQA 时全部 729 个 patch 都流入 LLM。
- 4 个 register token，在交给 LLM 前丢弃。
- 使用 2D-RoPE，并为 native aspect ratio 做 image-level scaling。

这个配置里的每个决定都可以追溯到一篇你能读懂的论文。

## 使用它

`code/main.py` 是一个 patch tokenizer 和 geometry calculator。它接收（image H、W、patch P、hidden D、depth L）并报告：

- Patching 后的 grid shape 和 sequence length。
- 针对合成 8x8 像素 toy image 的 token 序列（逐步走过 flatten + project 路径）。
- 按 patch embed、position embed、transformer blocks 和 head 拆分的参数量。
- 目标分辨率下每次 forward pass 的 FLOPs。
- ViT-B/16 @ 224、ViT-L/14 @ 336、DINOv2 ViT-g/14 @ 224、SigLIP SO400m/14 @ 384 的对比表。

运行它。把参数量和公开数字对上。修改 patch size 和 resolution，体会 token count 的成本。

## 交付它

本课产出 `outputs/skill-patch-geometry-reader.md`。给定一个 ViT config（patch size、resolution、hidden dim、depth），它会输出 token-count、parameter-count 和 VRAM estimate，并附上理由。每次为 VLM 选择 vision backbone 时都使用这个 skill，它能避免“token 爆炸，把我的 LLM context 填满了”的意外。

## 练习

1. 计算 Qwen2.5-VL 在 native 1280x720 输入、patch size 14 下的 patch-token sequence length。它和只用 CLS 的 representation 相比如何？

2. 一个 1080p frame（1920x1080）在 patch 14 下会产生多少 token？一个 30 FPS、5 分钟视频总共有多少 visual token？哪种方式最省成本：pooling、frame sampling，还是 token merging？

3. 用纯 Python 实现对 patch token 的 mean pooling。验证对 DINOv2 输出的 196 个 token 做 mean-pool，是否与模型 `forward` 返回 pooled embedding 时匹配。

4. 阅读 "Vision Transformers Need Registers"（arXiv:2309.16588）第 3 节。用两句话描述 register 吸收了什么 artifact，以及为什么它对下游 dense prediction 重要。

5. 修改 `code/main.py` 以支持 patch-n'-pack：给定一组不同分辨率图像，生成一个 packed sequence 和 block-diagonal attention mask。等你学到第 12.06 课时再对照验证。

## 关键术语

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Patch | “16x16 pixel square” | 输入图像中固定大小、互不重叠的区域；会变成一个 token |
| Patch embedding | “Linear projection” | 共享的可学习矩阵（或 stride=P 的 Conv2d），把展平的 patch 像素映射为 D 维向量 |
| CLS token | “Class token” | prepend 的可学习向量，其最终 hidden state 表示整张图；在 2026 年是可选项 |
| Register token | “Sink token” | 额外的可学习 token，用来吸收 ViT 在预训练期间形成的高范数 attention artifacts |
| Position embedding | “Positional info” | 让序列具备顺序意识的 per-position 向量或旋转；2D-RoPE 是现代默认方案 |
| Grid | “Patch grid” | 给定 resolution 和 patch size 下的 `(H/P) x (W/P)` patch 二维数组 |
| NaFlex | “Native flexible resolution” | SigLIP 2 特性：单个模型无需重新训练即可服务多种 aspect ratio 和 resolution |
| Backbone | “Vision tower” | 预训练图像编码器；它的 patch-token 输出会在 VLM 中喂给 LLM |
| Pooling | “Image-level summary” | 把 patch token 变成一个向量的策略：CLS、mean、attention pool 或 register-based |
| Patch 14 vs 16 | “Finer vs coarser grid” | Patch 14 每张图产生更多 token，OCR fidelity 更好但更慢；patch 16 是经典默认值 |

## 延伸阅读

- [Dosovitskiy et al. — An Image is Worth 16x16 Words (arXiv:2010.11929)](https://arxiv.org/abs/2010.11929) — 原始 ViT。
- [He et al. — Masked Autoencoders Are Scalable Vision Learners (arXiv:2111.06377)](https://arxiv.org/abs/2111.06377) — MAE，self-supervised pretraining。
- [Oquab et al. — DINOv2 (arXiv:2304.07193)](https://arxiv.org/abs/2304.07193) — 大规模 self-distillation，无标签。
- [Darcet et al. — Vision Transformers Need Registers (arXiv:2309.16588)](https://arxiv.org/abs/2309.16588) — register token 与 artifact 分析。
- [Tschannen et al. — SigLIP 2 (arXiv:2502.14786)](https://arxiv.org/abs/2502.14786) — 2026 年默认 vision tower。
- [Zhai et al. — Scaling Vision Transformers (arXiv:2106.04560)](https://arxiv.org/abs/2106.04560) — 经验 scaling laws。
