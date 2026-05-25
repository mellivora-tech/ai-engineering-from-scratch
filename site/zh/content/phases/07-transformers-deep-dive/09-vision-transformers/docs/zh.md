# Vision Transformers (ViT)

> 一张图像是一格格 patches。一个句子是一格格 tokens。同一个 transformer 可以吃下两者。

**类型：** 构建
**语言：** Python
**前置要求：** 阶段 7 · 05（Full Transformer），阶段 4 · 03（CNNs），阶段 4 · 14（Vision Transformers intro）
**时间：** ~45 分钟

## 问题

2020 年之前，computer vision 意味着 convolutions。ImageNet、COCO 和 detection benchmarks 上的每个 SOTA 都使用 CNN backbone。Transformers 是给语言用的。

Dosovitskiy et al.（2020）— “An Image is Worth 16x16 Words” — 展示了你可以完全丢掉 convolutions。把图像切成固定大小的 patches，把每个 patch 线性投影成 embedding，把序列送入 vanilla transformer encoder。在足够规模下（ImageNet-21k pretraining 或更大），ViT 可以匹配或超过基于 ResNet 的模型。

ViT 开启了 2026 年更广泛的模式：一个架构，多种模态。Whisper tokenizes audio。ViT tokenizes images。机器人使用 action tokens。视频使用 pixel tokens。Transformer 不在乎 — 给它一个序列，它就会学习。

到 2026 年，ViT 及其后代（DeiT、Swin、DINOv2、ViT-22B、SAM 3）占据了大多数视觉任务。CNNs 仍然在边缘设备和 latency-sensitive 任务上胜出。其他地方的 stack 里几乎都有一个 ViT。

## 概念

![Image → patches → tokens → transformer](../assets/vit.svg)

### 第 1 步 — patchify

把一个 `H × W × C` 图像切成一个 `N × (P·P·C)` 的扁平 patches 序列。典型设置：`224 × 224` 图像，`16 × 16` patches → 196 个 patches，每个有 768 个值。

```
image (224, 224, 3) → 14 × 14 grid of 16x16x3 patches → 196 vectors of length 768
```

Patch size 是杠杆。更小 patches = 更多 tokens、更好分辨率、二次 attention 成本。更大 patches = 更粗、更便宜。

### 第 2 步 — linear embedding

一个学习得到的矩阵把每个扁平 patch 投影到 `d_model`。这等价于 kernel size 为 `P`、stride 为 `P` 的 convolution。在 PyTorch 中这字面上就是 `nn.Conv2d(C, d_model, kernel_size=P, stride=P)` — 两行实现。

### 第 3 步 — prepend `[CLS]` token，添加 positional embeddings

- 前置一个可学习的 `[CLS]` token。它最终的 hidden state 是用于分类的图像表示。
- 添加 learnable positional embeddings（ViT-original）或 sinusoidal 2D（后续变体）。
- 2024+ 中 RoPE 被扩展到 2D 位置，有时不需要显式 embeddings。

### 第 4 步 — standard transformer encoder

堆叠 L 个 `LayerNorm → Self-Attention → + → LayerNorm → MLP → +` blocks。和 BERT 完全相同。没有视觉专用层。这是那篇论文在教学上的关键点。

### 第 5 步 — head

分类时：取 `[CLS]` hidden state → linear → softmax。对于 DINOv2 或 SAM，丢掉 `[CLS]`，直接使用 patch embeddings。

### 重要变体

| 模型 | 年份 | 变化 |
|-------|------|--------|
| ViT | 2020 | 原始版本。固定 patch size，完整 global attention。 |
| DeiT | 2021 | Distillation；只用 ImageNet-1k 也能训练。 |
| Swin | 2021 | 带 shifted windows 的层级结构。固定 sub-quadratic 成本。 |
| DINOv2 | 2023 | Self-supervised（无标签）。最好的通用视觉特征。 |
| ViT-22B | 2023 | 22B 参数；scaling laws 适用。 |
| SigLIP | 2023 | ViT + language pair，sigmoid contrastive loss。 |
| SAM 3 | 2025 | Segment anything；ViT-Large + promptable mask decoder。 |

### 为什么花了一段时间

ViT 需要 *大量* 数据才能匹配 CNN，因为它没有 CNN 的 inductive biases（translation invariance、locality）。没有 >100M 标注图像或强 self-supervised pretraining 时，同等 compute 下 CNN 仍然胜出。DeiT 在 2021 年用 distillation 技巧修复了这个问题；DINOv2 在 2023 年用 self-supervision 永久修复了它。

## 构建它

见 `code/main.py`。纯 stdlib patchify + linear embedding + sanity checks。不训练 — 任何现实规模的 ViT 都需要 PyTorch 和数小时 GPU 时间。

### 第 1 步：fake image

把 24 × 24 RGB 图像表示为 `(R, G, B)` tuples 的行列表。我们使用 6×6 patches → 16 个 patches，每个 108-d embedding vector。

### 第 2 步：patchify

```python
def patchify(image, P):
    H = len(image)
    W = len(image[0])
    patches = []
    for i in range(0, H, P):
        for j in range(0, W, P):
            patch = []
            for di in range(P):
                for dj in range(P):
                    patch.extend(image[i + di][j + dj])
            patches.append(patch)
    return patches
```

Raster order：按 grid 的 row-major 顺序。每个 ViT 都使用这个顺序。

### 第 3 步：linear embed

把每个扁平 patch 乘以随机 `(patch_flat_size, d_model)` 矩阵。前置 `[CLS]` 后，验证输出形状是 `(N_patches + 1, d_model)`。

### 第 4 步：为现实 ViT 统计参数

打印 ViT-Base 的参数量：12 层、12 heads、d=768、patch=16。和 ResNet-50（~25M）比较。ViT-Base 约 86M。ViT-Large 约 307M。ViT-Huge 约 632M。

## 使用它

```python
from transformers import ViTImageProcessor, ViTModel
import torch
from PIL import Image

processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")

img = Image.open("cat.jpg")
inputs = processor(img, return_tensors="pt")
out = model(**inputs).last_hidden_state   # (1, 197, 768): [CLS] + 196 patches
cls_emb = out[:, 0]                       # image representation
```

**DINOv2 embeddings 是 2026 年图像特征默认选择。** 冻结 backbone，训练一个 tiny head。适用于分类、检索、检测、captioning。Meta 的 DINOv2 checkpoints 在每个非文本视觉任务上都超过 CLIP。

**选择 patch size。** 小模型使用 16×16（ViT-B/16）。Dense prediction（segmentation）使用 8×8 或 14×14（SAM、DINOv2）。超大模型使用 14×14。

## 交付它

见 `outputs/skill-vit-configurator.md`。这个 skill 会根据 dataset size、resolution 和 compute budget，为新的视觉任务选择 ViT 变体和 patch size。

## 练习

1. **简单。** 运行 `code/main.py`。验证 patches 数量等于 `(H/P) * (W/P)`，扁平 patch 维度等于 `P*P*C`。
2. **中等。** 实现 2D sinusoidal positional embeddings — 为每个 patch 的 `row` 和 `col` 分别生成 sinusoidal codes，再拼接。把它们送入 tiny PyTorch ViT，并在 CIFAR-10 上比较 learnable positional embeddings 的准确率。
3. **困难。** 构建一个 3 层 ViT（PyTorch），用 4×4 patches 在 1,000 张 MNIST 图像上训练。测量测试准确率。现在在同样 1,000 张图像上加入 DINOv2 pretraining（简化版：训练 encoder 从 masked patches 预测 patch embeddings）。准确率会提升吗？

## 关键词

| 术语 | 大家怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| Patch | “vision-transformer token” | 图像中 `P × P × C` 区域的像素值扁平向量。 |
| Patchify | “切块 + flatten” | 把图像切成不重叠 patches，并把每个 patch flatten 成向量。 |
| `[CLS]` token | “图像摘要” | 前置的可学习 token；它的最终 embedding 是图像表示。 |
| Inductive bias | “模型假设什么” | ViT 比 CNN 先验更少；需要更多数据弥补差距。 |
| DINOv2 | “Self-supervised ViT” | 使用图像增强 + momentum teacher，无标签训练。2026 年最好的通用图像特征。 |
| SigLIP | “CLIP 的继任者” | ViT + text encoder，用 sigmoid contrastive loss 训练；同等 compute 下优于 CLIP。 |
| Swin | “Windowed ViT” | 带 local attention + shifted windows 的层级 ViT；sub-quadratic。 |
| Register tokens | “2023 技巧” | 少量额外可学习 tokens，用来吸收 attention sinks；改善 DINOv2 features。 |

## 延伸阅读

- [Dosovitskiy et al. (2020). An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929) — ViT 论文。
- [Touvron et al. (2021). Training data-efficient image transformers & distillation through attention](https://arxiv.org/abs/2012.12877) — DeiT。
- [Liu et al. (2021). Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/abs/2103.14030) — Swin。
- [Oquab et al. (2023). DINOv2: Learning Robust Visual Features without Supervision](https://arxiv.org/abs/2304.07193) — DINOv2。
- [Darcet et al. (2023). Vision Transformers Need Registers](https://arxiv.org/abs/2309.16588) — DINOv2 的 register-token 修复。
