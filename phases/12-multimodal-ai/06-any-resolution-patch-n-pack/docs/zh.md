# Any-Resolution Vision：Patch-n'-Pack 与 NaFlex

> 真实图像不是 224x224 的正方形。收据是 9:16，图表是 16:9，医学扫描可能是 4096x4096，手机截图是 9:19.5。2024 年前 VLM 的答案是把一切 resize 到固定正方形，这会丢掉让 OCR、document understanding 和 high-resolution scene parsing 真正有效的信号。NaViT（Google，2023）表明，可以用 block-diagonal masking 把 variable-resolution patches 打包进单个 transformer batch。Qwen2-VL 的 M-RoPE（2024）完全去掉了 absolute positional tables。LLaVA-NeXT 的 AnyRes 把高分辨率图像切成 base + sub-images。SigLIP 2 的 NaFlex 变体（2025）现在是 open VLM 的默认 encoder，当它们希望一个 checkpoint 服务所有 aspect ratio 时。本课端到端实现 patch-n'-pack。

**类型：** 构建
**语言：** Python（stdlib，patch packer + block-diagonal mask）
**前置要求：** 阶段 12 · 01（ViT patches），阶段 12 · 05（LLaVA）
**时间：** ~120 分钟

## 学习目标

- 把一个 variable-resolution image batch 的 patches 打包进一个序列，并构建 block-diagonal attention mask。
- 针对给定任务，在 AnyRes tiling（LLaVA-NeXT）、NaFlex（SigLIP 2）和 M-RoPE（Qwen2-VL）之间选择。
- 在不 resize 的情况下计算 OCR、charts 和 photography 的 token budgets。
- 说出 square-resize 的三个 failure modes：文字被挤压、内容被裁剪、padding 浪费 token。

## 问题

Transformer 期待的是序列。Batch 是一组长度相同的序列。如果你的图像都是 224x224，每次得到 196 个 patch token，不需要 padding，任务完成。训练用 224，推理用 224，再也不用考虑分辨率。

现实并不配合。文档是 portrait（8.5x11 英寸，约 2:3）。图表截图是 landscape（16:9）。收据又高又窄（1:3）。医学影像是 2048x2048 或更大。移动设备截图是 1170x2532（0.46:1）。

2024 年前有三种选择，以及它们为什么失败：

1. Resize 到固定正方形（224x224 或 336x336）。挤压会扭曲文本和人脸。Downscale 会破坏图表标签和 OCR 内容。直到 LLaVA-1.5 前，这都是标准做法。
2. Crop 到固定 aspect ratio。你会丢掉图像的大部分内容，而且选择 crop location 本身就是一个视觉问题。
3. Pad 到最长边。解决扭曲，但对 portrait images 会浪费 50%+ token 在 padding 上。所有 pad token 都要承担二次 attention 成本。

2024-2025 年的答案是：让 transformer 在图像 native resolution 下吃 patch，并想办法把 heterogeneous batch 打包成一个序列，同时不浪费 compute。

## 概念

### NaViT 与 patch-n'-pack

NaViT（Dehghani 等人，2023）证明了这套方法可以规模化。思路很机械：

1. 对 batch 中每张图，按所选 patch size（比如 14）计算其 native patch grid。
2. 把每张图的 patches 展平成自己的 variable-length sequence。
3. 把所有图像的 patches 拼成 batch 的一个长序列。
4. 构建 block-diagonal attention mask，让 image A 的 patches 只在 image A 内部 attend。
5. 携带 per-patch position information（2D RoPE 或 fractional position embeddings）。

三张图像的 batch：336x336（576 tokens）、224x224（256 tokens）、448x336（768 tokens），会变成一个 1600-token 序列，配一个 1600x1600 block-diagonal mask。没有 padding。没有浪费 compute。Transformer 可以处理任意 aspect ratio。

NaViT 还在训练中引入 fractional patch dropping：在整个 batch 中随机 drop 50% 的 patches。这既正则化又加速训练。SigLIP 2 继承了它。

### AnyRes（LLaVA-NeXT）

LLaVA-NeXT 的 AnyRes 是更务实的替代方案。给定一张高分辨率图像和固定 encoder（CLIP 或 SigLIP at 336），把图像 tile：

1. 从预定义集合中选择最适合图像 aspect ratio 的 grid layout，例如（1x1）、（1x2）、（2x1）、（1x3）、（3x1）、（2x2）等。
2. 把完整图像切成这个 grid；每个 tile 都变成 336x336 crop。
3. 额外生成一个 thumbnail：整张图 resize 到 336x336，作为 global-context token。
4. 每个 tile 都通过 frozen 336-encoder 编码。拼接 tile tokens + thumbnail tokens。

对于 672x672 图像，2x2 grid 加 thumbnail：4 * 576 + 576 = 2880 visual tokens。昂贵但有效，LLM 同时看到局部细节和全局上下文。

当你的 encoder 是 frozen 且只支持一个 resolution 时，AnyRes 是首选路径。它会让大图的 token count 爆炸（1344x1344 图像在 4x4 grid 下是 9216 + 576 ≈ 9800 tokens，会填满大部分 8k LLM context）。

### M-RoPE（Qwen2-VL）

Qwen2-VL 引入了 Multimodal Rotary Position Embedding。不同于 NaViT 的 fractional positions 或 AnyRes 的 tile-and-thumbnail，每个 patch 携带 3D position（temporal、height、width）。Query/key rotations 处理任意 H、W 和 temporal length。

M-RoPE 让模型无需重新训练就能 native dynamic resolution。推理时喂入任意 HxW 图像，patch embedder 产生 H/14 x W/14 tokens，每个 token 获得自己的（t=0, r=row, c=col）position，RoPE 用正确频率旋转 attention，完成。Qwen2.5-VL 和 Qwen3-VL 继续使用它。InternVL3 的 V2PE 是同一个思路，只是按 modality 使用 variable encoding。

不同于 AnyRes，M-RoPE 在 native resolution 下是 O(H x W / P^2) tokens，没有乘法级 tile overhead。不同于 NaViT，它仍然期待单张图一次 forward。跨分辨率 batching 仍然需要在上层使用 patch-n'-pack。

### NaFlex（SigLIP 2）

NaFlex 是 SigLIP 2 checkpoint 的 native-flex mode。一个模型在推理时服务多种 sequence length（256、729、1024 tokens）。内部训练使用 NaViT-style patch-n'-pack，并为每个 patch 使用 absolute fractional positions。卖点是：一个 checkpoint，根据任务在推理时选择 token budget。

语义任务（classification、retrieval）用 256 tokens。OCR 或 chart understanding 用 1024 tokens。无需重新训练。

### Packing mask

Block-diagonal mask 是大多数实现容易踩坑的地方。对于一个总长度为 `N_total` 的 packed sequence，覆盖图像 `i=0..B-1`，每张图长度为 `n_i`，形状为 `(N_total, N_total)` 的 mask `M` 在两个索引落在同一个图像 block 中时为 1，否则为 0。你可以用 cumulative length list 构建它：

```
offsets = [0, n_0, n_0+n_1, ..., N_total]
M[i, j] = 1 iff there exists b where offsets[b] <= i < offsets[b+1] and offsets[b] <= j < offsets[b+1]
```

在 PyTorch 中可以用 `torch.block_diag` 或显式 gather 一行完成。FlashAttention 的 variable-length path（`cu_seqlens`）完全跳过 dense mask，直接用 cumulative-length tensor 在各自序列内部 attend，典型 batch 下比 dense mask 快约 10 倍。

### Token budgets

按任务选择策略：

- OCR / documents：1024-4096 tokens。SigLIP 2 NaFlex at 1024，或 AnyRes 3x3 + thumbnail。
- Charts and UI：在 384-448 native 下用 729-1024 tokens。Qwen2.5-VL dynamic resolution 配 max pixels cap。
- Natural photos：256-576 tokens 足够。下游 LLM 已经能看到足够信息。把 token 花在 content density 高的地方。
- Video：空间 pooling 后每帧 64-128 tokens，2-8 FPS。第 12.17 课讲这个。

2026 年生产规则：为每个任务选择 per-task max-pixels cap，以 native aspect ratio 编码到该 cap，打包 batch，跳过 padding。Qwen2.5-VL 暴露的 `min_pixels` 和 `max_pixels` 正是这个旋钮。

## 使用它

`code/main.py` 为一组 heterogeneous image batch 实现 patch-n'-pack，使用整数像素坐标。它会：

- 接收一组（H, W）image sizes。
- 计算每张图在 patch size 14 下的 patch sequence length。
- 把它们打包成总长度为 `sum(n_i)` 的一个 sequence。
- 构建 block-diagonal attention mask（为了清晰使用 dense）。
- 比较 packed cost 与 square-resize、AnyRes tiling。
- 为混合 batch（receipt、chart、screenshot、photo）打印 token budget table。

运行它。输出的数字就是每个 2026 年 open VLM 都使用 patch-n'-pack 的原因。

## 交付它

本课产出 `outputs/skill-resolution-budget-planner.md`。给定一个 mixed-aspect-ratio workload（OCR、charts、photos、video frames）和 total-token budget，它会选择正确策略（NaFlex、AnyRes、M-RoPE 或 fixed-square），并输出 per-request configuration。把它用于 VLM 产品 sizing，可以避免悄悄把 latency budget 杀死的 10x token blowup。

## 练习

1. 一张收据是 600x1500（1:2.5）。在 patch size 14 下有多少 native-resolution tokens？Square-resize 到 336 后有多少？实践中哪一个损失更多 OCR accuracy？

2. 为四张图长度分别为 256、576、729、1024 的 batch 构建 block-diagonal mask。验证 attention matrix 是 2585x2585，并且恰好有 `256^2 + 576^2 + 729^2 + 1024^2` 个非零 entries。

3. 对一个 1792x896 图像、patch 14，对比：（a）square-resize 到 336 再编码，（b）AnyRes 2x1 + thumbnail，（c）M-RoPE native。哪种 token 最少？哪种保留最多细节？

4. 实现 fractional patch dropping：给定 packed sequence，随机均匀 drop 50% token，并相应更新 block-diagonal mask。测量 mask sparsity 变化。

5. 阅读 Qwen2-VL 论文（arXiv:2409.12191）第 3.2 节。用两句话描述 `min_pixels` 和 `max_pixels` 控制什么，以及为什么两个 bounds 都重要。

## 关键术语

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| Patch-n'-pack | “NaViT-style packing” | 把不同图像的 variable-length patch sequences 拼接进一个 batch dimension |
| Block-diagonal mask | “Packing mask” | Attention mask，限制每张图的 patches 只 attend 到自身，而不是 pack 中邻居 |
| AnyRes | “LLaVA-NeXT tiling” | 把高分辨率图像切成固定大小 tile grid，加一个 global thumbnail；每个 tile 用固定 encoder 编码 |
| NaFlex | “SigLIP 2 native-flex” | 单个 SigLIP 2 checkpoint，在推理时服务 256/729/1024-token budgets，无需重新训练 |
| M-RoPE | “Multimodal RoPE” | 3D rotary position encoding（time、row、column），无需 position tables 即可处理任意 H、W、T |
| cu_seqlens | “FlashAttention packing” | FlashAttention varlen path 使用的 cumulative-length tensor，用来替代 dense block-diagonal mask |
| min_pixels / max_pixels | “Resolution bounds” | Qwen2.5-VL 的 per-request 旋钮，用于限制极小或极大输入的 token count |
| Visual token budget | “How many tokens per image” | 每张图输出的 patch token 粗略数量；决定 LLM prompt budget 和 attention cost |

## 延伸阅读

- [Dehghani et al. — Patch n' Pack: NaViT (arXiv:2307.06304)](https://arxiv.org/abs/2307.06304)
- [Wang et al. — Qwen2-VL (arXiv:2409.12191)](https://arxiv.org/abs/2409.12191)
- [Laurençon et al. — What matters when building vision-language models? (Idefics2, arXiv:2405.02246)](https://arxiv.org/abs/2405.02246)
- [Tschannen et al. — SigLIP 2 (arXiv:2502.14786)](https://arxiv.org/abs/2502.14786)
- [Qwen Team — Qwen2.5-VL Technical Report (arXiv:2502.13923)](https://arxiv.org/abs/2502.13923)
