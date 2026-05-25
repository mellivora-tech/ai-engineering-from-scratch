# CLIP 与 Contrastive Vision-Language Pretraining

> OpenAI 的 CLIP（2021）证明了一个足以驱动接下来五年的大想法：只用噪声很大的 web 图像-caption 对和 contrastive loss，把 image encoder 与 text encoder 对齐到同一个向量空间。没有 supervised labels。400M 对数据。得到的 embedding space 可以做 zero-shot classification、image-text retrieval，并作为 vision tower 接入每个 2026 年 VLM。SigLIP 2（2025）用 sigmoid 替换 softmax，以更低成本超过 CLIP。本课从 InfoNCE 到 sigmoid pairwise loss 走一遍数学，并用 stdlib Python 构建训练步骤。

**类型：** 构建
**语言：** Python（stdlib，InfoNCE + sigmoid loss 实现）
**前置要求：** 阶段 12 · 01（ViT patches），阶段 7（Transformers）
**时间：** ~180 分钟

## 学习目标

- 从 mutual information 推导 InfoNCE loss，并实现一个数值稳定的向量化版本。
- 解释为什么 sigmoid pairwise loss（SigLIP）可以扩展到 batch 32768+，而不需要 softmax 要求的 all-gather 开销。
- 通过构造 text template（`a photo of a {class}`）并对 cosine similarity 取 argmax，运行 zero-shot ImageNet classification。
- 说出 CLIP / SigLIP pretraining 给你的四个杠杆：batch size、temperature、prompt template、data quality。

## 问题

CLIP 之前的视觉模型主要是 supervised。收集带标签数据集（ImageNet：1.2M 图像、1000 类），训练 CNN，然后上线。标签昂贵，标签会偏向标注者能达成共识的内容，而且不经过 finetuning 就很难迁移到新任务。

图像-caption web 上有十亿级松散标注的数据对，几乎是免费的。一张 golden retriever 的照片配上 alt text “my dog Max in the park”，就携带了监督信号：文本描述了图像。问题是：你能不能把它变成有用的训练？

CLIP 的答案是：把 image-caption 对当成匹配任务。给定一个包含 N 张图像和 N 条 caption 的 batch，学习把每张图像与自己的 caption 匹配起来，并与 N-1 个干扰项区分开。监督信号是“这两个东西属于一起；另外 N-1 个不是”。没有 class label。没有人工标注。只有一个 contrastive loss。

得到的 embedding space 能做的远不止训练目标本身。ImageNet zero-shot 能工作，是因为 “a photo of a cat” 的 embedding 会靠近那些从未被显式标成 cat 的猫图像。这就是催生每个 2026 年 VLM 的赌注。

## 概念

### Dual encoder

CLIP 有两个 tower：

- Image encoder `f`：ViT 或 ResNet，每张图输出一个 D 维向量。
- Text encoder `g`：小型 transformer，每条 caption 输出一个 D 维向量。

两个 tower 都把输出归一化为单位长度。因为两边都是 unit-norm，所以 similarity 是 `cos(f(x), g(y)) = f(x)^T g(y)`。

对于一个包含 N 个（image, caption）对的 batch，构造形状为 `(N, N)` 的 similarity matrix `S`：

```
S[i, j] = cos(f(x_i), g(y_j)) / tau
```

其中 `tau` 是可学习 temperature（CLIP 初始化为 0.07；在 log-space 中学习）。

### InfoNCE loss

CLIP 对行和列使用对称 cross-entropy：

```
loss_i2t = CE(S, labels=identity)     # each image's positive is its own caption
loss_t2i = CE(S^T, labels=identity)   # each caption's positive is its own image
loss = (loss_i2t + loss_t2i) / 2
```

这就是 InfoNCE。CE 中的 softmax 会强迫每张图像与自己的 caption 的匹配程度高于 batch 中所有其他 caption。“negatives”就是其他 batch items。更大的 batch = 更多 negatives = 更强的信号。CLIP 使用 batch 32k 训练；规模很重要。

### Temperature

`tau` 控制 softmax 的尖锐程度。低 tau -> 分布更尖锐，有 hard negative mining 的效果。高 tau -> 更柔和，所有样本都有贡献。CLIP 学习 log(1/tau)，并裁剪它以防止 collapse。SigLIP 2 固定初始 tau，改用一个 learned bias。

### 为什么 sigmoid 更易扩展（SigLIP）

Softmax 需要同步整个 similarity matrix。在分布式训练中，你必须把每个 embedding all-gather 到每个 replica，然后做 softmax。这在通信上会随 world size 近似二次增长。

SigLIP 用 element-wise sigmoid 替换 softmax：对于每个 `(i, j)` pair，loss 是一个“这是不是匹配 pair？”的 binary classification；对角线是正类，其他都是负类。Loss 是：

```
L = -1/N sum over (i, j) [ y_ij log sigmoid(S[i,j]) + (1-y_ij) log sigmoid(-S[i,j]) ]
```

如果 `i == j`，则 `y_ij = 1`，否则为 0。每个 pair 的 loss 相互独立。不需要 all-gather。每块 GPU 计算本地 block 并求和。SigLIP 2 能低成本扩展到 batch 32k-512k，而 CLIP 需要成比例增加通信。

### Zero-shot classification

给定 N 个 class name，为每个 class 构造 text template：

```
"a photo of a {class}"
```

用 text encoder 嵌入每个 template。用 image encoder 嵌入图像。Cosine similarity 的 argmax 就是预测 class。目标类别上不需要训练。

Prompt template 很重要。CLIP 原论文对每个 class 使用了 80 个 template（plain、artistic、photo、painting 等），并平均 embedding。ImageNet 提升 3 个点。现代用法通常选择一两个 template。

### Linear probes 与 finetuning

Zero-shot 是 baseline。Linear probe（在 frozen CLIP features 上为目标类别训练一个线性层）在 in-domain task 上超过 zero-shot。Full finetuning 在 in-domain 上超过 linear probe，但可能伤害 zero-shot transfer。三种模式对应三种 trade-off。

### SigLIP 2：NaFlex 与 dense features

SigLIP 2（2025）加入了：
- NaFlex：单个模型处理可变 aspect ratio 和 resolution。
- 更好的 dense features，用于 segmentation 和 depth estimation，目标是作为 VLM 中 frozen backbone。
- Multilingual：训练覆盖 100+ 种语言，而 CLIP 仅英语。
- 1B 参数规模，而 CLIP 最高约 400M。

在 2026 年 open VLM 中，SigLIP 2 SO400m/14 是默认 vision tower。CLIP 仍然是纯 image-text retrieval 的默认选择，特别是当 LAION-2B 训练分布与你的 query pattern 匹配时。

### ALIGN、BASIC、OpenCLIP、EVA-CLIP

ALIGN（Google，2021）：与 CLIP 同样的思路，1.8B 对规模，90% 噪声。证明了噪声数据也能扩展。OpenCLIP（LAION）：在 LAION-400M / 2B 上对 CLIP 的开放复现，多种规模，是常用 open checkpoint。EVA-CLIP：从 masked image modeling 初始化；作为 VLM backbone 很强。BASIC：Google 的 CLIP+ALIGN 混合体。它们都是同一个家族，只是数据和调参不同。

### Zero-shot ceiling

CLIP 类模型在 ImageNet zero-shot 上大约封顶在 76%（CLIP-G、OpenCLIP-G）。继续提升需要更大数据（SigLIP 2 达到 80%+）或架构变化（supervised heads、更多参数）。这个 benchmark 正在饱和；真正的价值是 downstream VLM 会消费的 embedding space。

## 使用它

`code/main.py` 实现了：

1. 一个 toy dual encoder（hash-based image features、text char features），让你不使用 numpy 也能看到 InfoNCE 的形状。
2. 纯 Python InfoNCE loss（通过 log-sum-exp 保证数值稳定）。
3. 用于对比的 sigmoid pairwise loss。
4. 一个 zero-shot classification routine：计算与一组 text prompt 的 cosine similarity，并取 argmax 作为预测。

运行它并观察 loss curve。绝对数值是 toy，但形状与真实 CLIP trainer 的输出一致。

## 交付它

本课产出 `outputs/skill-clip-zero-shot.md`。给定一组图像（通过 path）和一组目标类别，它会用 CLIP template 构造 text prompt，用指定 checkpoint（例如 `openai/clip-vit-large-patch14`）嵌入两侧，并返回 top-1 / top-5 预测及 similarity scores。该 skill 会拒绝对 prompt list 之外的类别做断言。

## 练习

1. 手工为一个包含 4 个 pair 的 batch 实现 InfoNCE。构造 4x4 similarity matrix，运行 softmax，取出对角线，计算 cross-entropy。用这个手算结果验证你的 Python 实现。

2. SigLIP 除 temperature 外还使用 bias 参数 `b`：`S'[i,j] = S[i,j]/tau + b`。当 batch 存在严重 class imbalance（每行 negatives 远多于 positives）时，`b` 起什么作用？阅读 SigLIP 第 3 节（arXiv:2303.15343）。

3. 构建一个 cats vs dogs 的 zero-shot classifier。尝试两个 prompt template：`a photo of a {class}` 和 `a picture of a {class}`。在 100 张测试图上测量 accuracy。Template ensemble 是否优于单个 template？

4. 计算 512-GPU、batch 32k 训练时 softmax InfoNCE 与 sigmoid pairwise 的通信成本。哪一个是 O(N)，哪一个是 O(N^2)？引用 SigLIP 第 4 节。

5. 阅读 OpenCLIP scaling-laws 论文（arXiv:2212.07143，Cherti 等人）。从图中复现他们关于 data scaling 的结论：在固定 model size 下，ImageNet zero-shot accuracy 与 training data size 之间的 log-linear relationship 是什么？

## 关键术语

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| InfoNCE | “Contrastive loss” | 针对 batch similarity matrix 的 cross-entropy；每个 item 的 positive 是配对 item，negatives 是其他所有 item |
| Sigmoid loss | “SigLIP loss” | Per-pair binary cross-entropy；没有 softmax、没有 all-gather，分布式训练成本低 |
| Temperature | “tau” | 在 softmax/sigmoid 前缩放 logits 的标量；控制分布尖锐程度 |
| Zero-shot | “no-finetune classification” | 用 text prompt 构造 class embedding，并用 cosine similarity 分类；目标类别上不训练 |
| Prompt template | “a photo of a ...” | 包在 class name 外面的文本脚手架；会影响 1-5 个点的 zero-shot accuracy |
| Dual encoder | “Two-tower” | 一个 image encoder + 一个 text encoder，输出到共享 D 维空间 |
| Hard negative | “Tough distractor” | 与 positive 足够相似、迫使模型努力区分的 negative |
| Linear probe | “Frozen + one layer” | 只在 frozen features 上训练线性分类器；用于衡量 feature quality |
| NaFlex | “Native flexible resolution” | SigLIP 2 无需 resize 即可接受任意 aspect ratio 和 resolution 图像的能力 |
| Temperature scaling | “log-parametrized tau” | CLIP 参数化 `log(1/tau)` 以改善梯度；并裁剪它防止 tau collapse 到近零 |

## 延伸阅读

- [Radford et al. — Learning Transferable Visual Models From Natural Language Supervision (arXiv:2103.00020)](https://arxiv.org/abs/2103.00020) — CLIP 论文。
- [Zhai et al. — Sigmoid Loss for Language Image Pre-Training (arXiv:2303.15343)](https://arxiv.org/abs/2303.15343) — SigLIP。
- [Tschannen et al. — SigLIP 2 (arXiv:2502.14786)](https://arxiv.org/abs/2502.14786) — multilingual + NaFlex。
- [Jia et al. — ALIGN (arXiv:2102.05918)](https://arxiv.org/abs/2102.05918) — 使用噪声 web data 扩展。
- [Cherti et al. — Reproducible scaling laws for contrastive language-image learning (arXiv:2212.07143)](https://arxiv.org/abs/2212.07143) — OpenCLIP scaling laws。
