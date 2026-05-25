# LLaVA-OneVision：Single-Image、Multi-Image、Video 一个模型

> LLaVA-OneVision（Li 等人，2024 年 8 月）之前，open-VLM 世界有几条分开的 lineage：single image 用 LLaVA-1.5，multi-image 用 Mantis 和 VILA，video 用 Video-LLaVA 和 Video-LLaMA。每条都赢自己的 benchmark，却在其他场景失败。LLaVA-OneVision 认为，一个 curriculum 可以训练同一个模型主导三个场景，而且 emergent task-transfer effects（single-image 技能迁移到 video，multi-image reasoning 迁移到 single-image）强过各个专才之和。Recipe deceptively simple：一个在各场景中保持常量的 visual-token budget，加上从 single-image 到 OneVision（multi-image）再到 video 的显式 curriculum。本课阅读 budget、curriculum 和 emergent behaviors。

**类型：** 构建
**语言：** Python（stdlib，token budget solver + curriculum planner）
**前置要求：** 阶段 12 · 05（LLaVA），阶段 12 · 06（any-resolution）
**时间：** ~180 分钟

## 学习目标

- 设计一个在 single-image、multi-image 和 video inputs 间保持常量的 visual-token budget。
- 安排一个 training curriculum，让技能从 single-image 迁移到 video，同时避免 catastrophic forgetting。
- 解释当 curriculum 做对时，为什么单个模型在相同 parameter count 下超过专门模型。
- 说出 LLaVA-OneVision 报告的三个 emergent capabilities：multi-camera reasoning、set-of-mark prompting、iPhone-screenshot agent。

## 问题

Image、multi-image 和 video 对模型施加的压力不同。

Single-image 想要高分辨率 token（AnyRes，约 2880 visual tokens），以捕捉 OCR 和细节。每个 sample 预算：一张图，2880 tokens。

Multi-image 想要几张中等分辨率图（每张约 576 tokens），让跨图 reasoning 能放进 context。每个 sample 预算：4-8 张图，每张 576，总计 2300-4600 tokens。

Video 想要很多低分辨率帧（pooling 后每帧约 196 tokens），捕捉 temporal dynamics。每个 sample 预算：8-32 frames，每帧 196，总计 1600-6200 tokens。

如果训练分开的模型，你只选一种 budget。如果训练一个模型，就需要 budget 在不同场景间合理伸缩，又不炸 context。

OneVision 之前，默认答案是“训练一个场景，忽略其他场景”。Video-LLaVA 在 image model 上用额外训练阶段改装 video。LLaVA-NeXT 通过 tiling 增加 multi-image 支持。没有一个能干净处理三者。

## 概念

### OneVision token budget

LLaVA-OneVision 选择每个 sample 约 3000-4000 tokens 的统一 visual-token budget，并按场景分配：

- Single image：AnyRes-9（3x3 tiles + thumbnail），每个 tile at 384，729 patches，aggressive bilinear pooling 2x2 -> 每 tile 182。总计：9 * 182 + 182 = 1820 tokens。或 AnyRes-4，每 tile 729 -> 2916 + 729。
- Multi-image：每张图中等分辨率（384，不 tiling），729 tokens，不 pooling。预算 6 张图 -> 4374 tokens。
- Video：32 frames at 384 resolution，aggressive 3x3 bilinear pool -> 每帧 81 tokens。总计：32 * 81 = 2592 tokens。

这种分配让总 token 大致保持常量。LLM 永远不会看到一个炸掉 context 的 batch。Encoder 在不同场景中产生不同 geometry，但 LLM 消费的是同一预算。

### 三阶段 curriculum

LLaVA-OneVision 分三阶段训练：

1. Single-image SFT（stage SI）。全部数据都是 single-image-plus-text。用高分辨率 AnyRes input 训练。这教会 perception、OCR 和 fine-grained understanding。使用 LLaVA-NeXT 数据加 OneVision-specific single-image data。
2. OneVision SFT（stage OV）。混合 single-image + multi-image + video（均匀采样 frames）。在统一 token budget 上训练。这教模型处理 heterogeneous batch shapes。不重置权重，继续 stage SI。
3. Task transfer（stage TT）。继续使用目标 task mix，通常根据产品更偏 multi-image 或 video。可选 deployment fine-tune。

关键点：curriculum order 很重要。即使数据相同，video-first 或 multi-image-first 也会产生比 single-image-first 更差的图像表现。论文明确 ablate 了这一点。

### 为什么 curriculum 有效

Single-image training 建立 perceptual base。Patch tokens 携带细粒度视觉特征，LLM 学会把它们与文本整合。Multi-image 和 video 引入结构挑战（哪张图是哪张、先发生了什么），但如果没有强 perception base，很难学好。

如果从头把所有场景混在一起训练，模型会 underfit perception（每个 batch 中 single-image data 有限），并 overfit structure（multi-image / video data 很多）。结果是：模型遵循跨图 reasoning patterns，但视觉很浅。

Curriculum order 让你从 stage SI 得到 perception strength，再从 stage OV 得到 compositional/temporal reasoning，并且不丢失任一方。

### 跨场景 emergent skills

LLaVA-OneVision 论文报告了三个 emergent capabilities：

1. Multi-camera reasoning。分别在 multi-image + video 上训练；推理时要求它推理 multi-camera driving scene。模型能正确整合视角，尽管训练中从未见过完全相同格式。
2. Set-of-mark prompting。用户在图像中用编号 mark 标注对象；模型推理“mark 3 相对 mark 7 在做什么”。既没有在 marks 上训练，也没有在 annotation 上训练；能力来自 spatial grounding + multi-image reference 的组合。
3. iPhone-screenshot agent。用户提供 iPhone 屏幕截图并要求规划下一次点击。模型在 UI screenshots、用户 workflow video、multi-image before/after pairs 上训练。泛化到了 agent 用例。

这些不是训练任务，而是从 curriculum 的组合结构中涌现出来的。

### Visual-token pooling

Token budget 需要 pooling。OneVision 在 2D patch grid 上使用 bilinear interpolation：24x24 = 576 patches 变成 12x12 = 144（2x factor）或 8x8 = 64（3x factor）。Pooling 在 patch-grid space 中完成，而不是 token space，以保留 locality。

每个场景的 pooling factor 本身就是 hyperparameter。更少 pooling = 更多 tokens = 更丰富 representation。更多 pooling = 更少 tokens = 容纳更多 frames / images。

### LLaVA-OneVision-1.5

2025 年 follow-up（LLaVA-OneVision-1.5，arXiv 2509.23661）在 training data、model weights 和 code 上“fully open”。在一些 benchmark 上缩小与 proprietary model 的差距，并使 recipe 民主化。同样 curriculum，更多数据，更好 base LLM。没有架构变化。

### 与 Qwen2.5-VL 对比

Qwen2.5-VL（第 12.09 课）选择不同。它使用 M-RoPE 和 dynamic FPS，而不是固定 pooling。它的 budget 随输入变化：1 分钟视频比 5 秒视频使用更多 token。LLaVA-OneVision 固定 budget，并调整 pooling。两者都有效；一个用可配置性换可预测性，另一个相反。

## 使用它

`code/main.py` 是 OneVision-style VLM 的 curriculum 与 budget planner。给定每个 sample 的 token budget 和目标场景 mix（例如 40% single-image、30% multi-image、30% video），它会：

- 为每个场景分配 resolution、pooling factor 和 frames。
- 检查每个场景都能放进共享 budget。
- 报告 expected token count、LLM FLOPs，以及哪些场景 under-tokenized。
- 打印分阶段 training schedule。

用它规划 OneVision fine-tune，或 sanity-check VLM deployment 的 per-request cost。

## 交付它

本课产出 `outputs/skill-onevision-budget-planner.md`。给定 target task distribution 和 per-sample budget，它会输出 AnyRes factor、per-frame pooling、video frame count 和 curriculum stage weights。训练或 fine-tune unified-scenario VLM 时都使用它。

## 练习

1. 你的产品支持 80% single-image、10% multi-image（2-4 images）、10% video（8-16 frames）。设计 token budget。由于不做 heavy multi-image 而省下的额外 budget，你会放在哪里？

2. 阅读 LLaVA-OneVision 第 4.3 节（emergent capabilities）。提出一个 curriculum 很可能解锁但论文没有报告的第四种 emergent skill。

3. 调换 curriculum order：先 train multi-image，再 single-image，再 video。预测哪些 benchmarks 会退化，以及为什么。

4. 论文报告 video benchmarks 只用每个 sample 8 帧训练。这能泛化到 inference 时的 30 秒视频吗？最先坏掉的是 token budget 还是 temporal reasoning？

5. 把 24x24 patches bilinear pooling 到 12x12，是每个维度 4x reduction。用 stdlib Python 实现 pooling，并验证每个 2x2 block 的 mean 与 bilinear output 匹配。

## 关键术语

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| OneVision scenario | “Single-image, multi-image, or video” | Unified VLM 处理的三种 input shape 之一；budget 在它们之间保持常量 |
| Token budget | “How many tokens per sample” | 每个 training / inference sample 中 LLM 看到的 visual tokens 总数，通常 3000-4000 |
| Curriculum | “Training order” | 为 emergent transfer 选择的阶段顺序（single-image -> multi-image -> video） |
| Bilinear pooling | “Token shrink” | 对 patch grid（2D）应用 bilinear interpolation，以在保留 locality 的同时减少 token count |
| Emergent skill | “Not trained, still works” | 由于 curriculum composition，在没有匹配训练数据时于 inference 出现的能力 |
| AnyRes-k | “k-tile setup” | k 个固定 resolution sub-tiles 加一个 thumbnail，常见 k ∈ {4, 9} |
| Task transfer | “Cross-scenario generalization” | 通过共享 backbone，把 single-image 学到的技能应用到 video（反之亦然） |

## 延伸阅读

- [Li et al. — LLaVA-OneVision (arXiv:2408.03326)](https://arxiv.org/abs/2408.03326)
- [LLaVA-OneVision-1.5: Fully Open Framework (arXiv:2509.23661)](https://arxiv.org/abs/2509.23661)
- [Lin et al. — Video-LLaVA (arXiv:2311.10122)](https://arxiv.org/abs/2311.10122)
- [Lin et al. — VILA (arXiv:2312.07533)](https://arxiv.org/abs/2312.07533)
- [Wang et al. — Qwen2-VL (arXiv:2409.12191)](https://arxiv.org/abs/2409.12191)
