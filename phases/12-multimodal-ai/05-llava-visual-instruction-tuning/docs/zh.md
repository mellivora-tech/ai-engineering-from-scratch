# LLaVA 与 Visual Instruction Tuning

> LLaVA（2023 年 4 月）是地球上被复制最多的多模态架构。它用 2-layer MLP 替换了 BLIP-2 的 Q-Former，用朴素 token concatenation 替换了 Flamingo 的 gated cross-attention，并在 158k 条 visual-instruction turns 上训练，这些数据由 GPT-4 根据 text-only captions 生成。2023 到 2026 年之间，任何构建 VLM 的实践者都做过某种 LLaVA 变体。LLaVA-1.5 加入 AnyRes。LLaVA-NeXT 提升分辨率。LLaVA-OneVision 用一个 recipe 统一 single image、multi-image 和 video。本课阅读这个 recipe，实现 projector，并解释为什么“更简单赢了”。

**类型：** 构建
**语言：** Python（stdlib，projector + instruction-template builder）
**前置要求：** 阶段 12 · 02（CLIP），阶段 11（LLM Engineering — instruction tuning）
**时间：** ~180 分钟

## 学习目标

- 构建一个 2-layer MLP projector，把 ViT patch embeddings（dim 1024）映射到 LLM embedding dim（dim 4096）。
- 走通 LLaVA 两阶段 recipe：（1）在 558k caption pairs 上做 projector alignment，（2）在 158k GPT-4-generated turns 上做 visual instruction tuning。
- 构造 LLaVA-format prompt，包含 image token placeholder、system prompt 和 user/assistant turns。
- 解释为什么尽管 Q-Former 有 token-budget 优势，社区仍然从 Q-Former 转向 MLP。

## 问题

BLIP-2 的 Q-Former（第 12.03 课）把图像压缩成 32 个 token。干净、高效、benchmark 表现好。但它有两个问题。

第一，Q-Former 是可训练的，但它的 loss 不是最终任务。Stage 1 训练 ITC+ITM+ITG。Stage 2 训练 LM loss。Queries 学到的是某种中间 representation，LLM 还必须解码它。Bottleneck 中会丢失信息。

第二，Q-Former 有 188M 参数，在 LLaVA 的 2023 年规模下，你必须让它与你的目标 LLM 共同设计。换 LLM，就要重训 Q-Former。换 vision encoder，也要重训。每一种组合都是一个独立 R&D 项目。

LLaVA 的答案简单到有点尴尬：拿 ViT 的 576 个 patch token，让每个 token 通过一个 2-layer MLP（`1024 → 4096 → 4096`），然后把全部 576 个 token 倒进 LLM input sequence。没有 bottleneck。没有在奇怪 objectives 上做 stage 1 pretraining。只在直接 LM loss 上训练 MLP。

数据从哪里来？LLaVA 的第二个洞见是：用 GPT-4（text-only）生成 instruction data。把图像的 COCO caption 和 bounding-box data 喂给 GPT-4，让它生成 conversations、descriptions 和 complex reasoning questions。免费得到 158k instruction-response turns。没有人工标注。

结果是：一个在 8 张 A100 上训练一天就能跑出来的 VLM，在 MMMU 上超过 Flamingo，并发布了社区可以扩展的 open checkpoint。到 2023 年末，它已经催生 50+ forks。

## 概念

### 架构

LLaVA-1.5 at 13B：
- Vision encoder：CLIP ViT-L/14 @ 336（stage 1 frozen，stage 2 可选 unfreeze）。
- Projector：带 GELU activation 的 2-layer MLP，`1024 → 4096 → 4096`。
- LLM：Vicuna-13B（后来是 Llama-3.1-8B）。

图像 + 文本 prompt 的 forward pass：

```
img -> ViT -> 576 patches of dim 1024
patches -> MLP -> 576 tokens of dim 4096
prompt: system + "<image>" placeholder + user question
replace <image> token with the 576 projected tokens
feed the full sequence to the LLM
decode response
```

图像占用 LLM context 中的 576 个 token。在 2048 context 中，文本还剩 1472 个 token。在 32k context 中，这只是一个零头。

### Stage 1：projector alignment

冻结 ViT。冻结 LLM。只训练 2-layer MLP。数据集：558k image-caption pairs（LAION-CC-SBU）。Loss：在 projected image tokens 条件下，对 caption 做 language modeling。

在 batch 128 下单 epoch 几小时完成。Projector 学会把 ViT-space 映射到 LLM-space。没有 task-specific supervision。

### Stage 2：visual instruction tuning

Projector 保持可训练。Unfreeze LLM（通常 full，有时 LoRA）。在 158k visual-instruction turns 上训练。

Instruction data 是关键。Liu 等人这样生成：
1. 取一张 COCO 图像。
2. 提取文本描述（5 条人工 caption + bounding-box list）。
3. 用三个 prompt template 发给 GPT-4：
   - Conversation: "Generate a back-and-forth dialogue between a user and assistant about this image."
   - Detailed description: "Give a rich, detailed description of the image."
   - Complex reasoning: "Ask a question that requires reasoning about the image, then answer it."
4. 把 GPT-4 输出解析成（instruction, response）pair。

这些步骤完全不直接接触图像，只看文本描述。GPT-4 会 hallucinate 一些合理的图像内容。有噪声，但它奏效了：158k turns 足以解锁对话能力。

### 为什么社区复制了它

- 没有 stage-1-specific losses 需要调。全程 LM loss。
- Projector 训练以小时计，而不是天。
- LLM 可以替换（LLaVA-Llama2、LLaVA-Mistral、LLaVA-Llama3），只需用最少 retrain 重新训练 projector。
- Visual-instruction data pipeline 使用 GPT-4，便宜，并且可以为新 domain 重新生成。

### LLaVA-1.5 与 LLaVA-NeXT

LLaVA-1.5（2023 年 10 月）加入：
- Academic-task data（VQA、OKVQA、RefCOCO）混入 instruction tuning。
- 更好的 system prompt。
- 2048 -> 32k context。

LLaVA-NeXT（2024 年 1 月）加入：
- AnyRes：把高分辨率图像切成 2x2 或 1x3 的 336x336 crops，再加一个全局低分辨率 thumbnail。每个 crop 变成 576 个 token；每张图总计约 2880 个 visual token。OCR 和 chart tasks 大幅提升。
- 使用 ShareGPT4V（高质量 GPT-4V captions）构建更好的 instruction data mixture。
- 更强 base LLM（Mistral-7B、Yi-34B）。

### LLaVA-OneVision

第 12.08 课会深入讲 OneVision。简短版：仍然使用同一个 projector，但通过覆盖 single-image、multi-image 和 video 的 curriculum，用共享 visual-token budget 训练一个模型。

### 与 Q-Former 对比

| | Q-Former (BLIP-2) | MLP (LLaVA) |
|---|---|---|
| Visual tokens per image | 32 | 576 (base) or 2880 (AnyRes) |
| Trainable params | 188M + LM | 40M + LM |
| Stage 1 loss | ITC+ITM+ITG | LM only |
| LLM drop-in | Requires retrain | Swap with minimal retrain |
| Multi-image | Awkward | Natural (concat) |
| Video | Awkward | Natural (per-frame concat) |
| Token budget | Small | Large |

MLP 赢在简单和 token flexibility。Q-Former 赢在 token budget。到 2023 年末，token budget 已经不再是主要瓶颈（LLM contexts 增长到 32k-128k+），简单性占了上风。

### Prompt format

```
A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: <image> Describe this image in detail. ASSISTANT: The image shows ...
```

`<image>` 是 placeholder token。在 tokenization 之前，它会被替换为 576 个 visual token（AnyRes 下是 2880）。Tokenizer 看到的序列比它训练时稍长，但 stage 1 已经教会 LLM 处理这种新输入。

### 参数经济学

LLaVA-1.5-7B 拆分：
- CLIP ViT-L/14 @ 336：303M（stage 1 frozen，stage 2 常常 unfrozen）。
- Projector（2x linear）：约 22M trainable。
- Llama-7B：7B。
- 总计：7.3B params。Stage 2 中 trainable：完整 7B + 22M projector。

Stage 2 训练成本：约 20 小时，8xA100。这是关键数字：一天，一个节点，可复现。这就是 LLaVA 扩散开的原因。

## 使用它

`code/main.py` 实现：

1. 纯 Python 2-layer MLP projector（toy scale：dim 16 -> 32 -> 32）。
2. Prompt-building pipeline：system prompt + `<image>` 替换为 N 个 projected tokens + user turn + assistant generation placeholder。
3. 一个 visualizer，展示 576-token visual block 在 LLM context 中的样子（占 2k / 32k / 128k context 的百分比）。

## 交付它

本课产出 `outputs/skill-llava-vibes-eval.md`。给定一个 LLaVA-family checkpoint，它会运行 10-prompt vibes-eval suite（3 个 captioning、3 个 VQA、2 个 reasoning、2 个 refusal），并报告人类可读的 scorecard。这不是 benchmark，而是 smoke test，用来确认 projector 与 LLM 连接良好。

## 练习

1. 计算 `1024 → 4096 → 4096` 的 2-layer MLP projector 的 trainable-parameter count。带 GELU 和 bias 时，它占 LLaVA-13B 的比例是多少？

2. 为一个 “refusal” case 构造 LLaVA prompt：图像中包含私人个体。写出期望 assistant response。为什么 LLaVA 应该 zero-shot 拒绝？需要什么训练数据来强化这种拒绝？

3. 阅读 LLaVA-NeXT blog 的 AnyRes 部分。计算 1344x672 图像在 AnyRes 下的 visual token count。与 336x336 下的 base 576 tokens 对比。

4. LLaVA stage-1 projector 使用 captions 上的 LM loss 训练。如果跳过 stage 1，直接进入 stage 2（visual instruction tuning），会发生什么？引用 Prismatic VLMs ablation（arXiv:2402.07865）回答。

5. LLaVA-Instruct-150k 使用 GPT-4 和 COCO captions 生成 instructions。对于新 domain（medical X-rays、satellite imagery），描述生成 domain instructions 的四步数据 pipeline。每一步可能出什么问题？

## 关键术语

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Projector | “MLP bridge” | 带 GELU 的 2-layer MLP，把 ViT dim 映射到 LLM dim |
| Image token | “<image> placeholder” | Prompt marker，在 inference 前被 N 个 projected visual tokens 替换 |
| Visual instruction tuning | “LLaVA stage 2” | 在 GPT-4-generated（image, instruction, response）triplets 上训练 |
| Stage 1 alignment | “Projector pretraining” | 冻结 ViT 和 LLM，在 captions 上用 LM loss 训练 projector |
| AnyRes | “Multi-crop tiling” | 把高分辨率图像切成 tile grid，并拼接每个 tile 的 visual tokens |
| LLaVA-Instruct | “GPT-4-generated” | 从 COCO captions + GPT-4 合成的 158k instruction-response pairs |
| Vision encoder freeze | “Backbone locked” | CLIP 权重在 stage 1 不更新，有时 stage 2 也不更新 |
| ShareGPT4V | “Better captions” | GPT-4V 生成的 1M dense captions，用于更高质量 alignment |
| VQA | “Visual question answering” | 回答关于图像的 free-form question 的任务 |
| Prismatic VLMs | “Design-space paper” | Karamcheti 2024 ablation，系统测试 projector 和 data 选择 |

## 延伸阅读

- [Liu et al. — Visual Instruction Tuning (arXiv:2304.08485)](https://arxiv.org/abs/2304.08485) — LLaVA 论文。
- [Liu et al. — Improved Baselines with Visual Instruction Tuning (arXiv:2310.03744)](https://arxiv.org/abs/2310.03744) — LLaVA-1.5。
- [Chen et al. — ShareGPT4V (arXiv:2311.12793)](https://arxiv.org/abs/2311.12793) — dense captions dataset。
- [Karamcheti et al. — Prismatic VLMs (arXiv:2402.07865)](https://arxiv.org/abs/2402.07865) — design-space ablations。
- [Li et al. — LLaVA-OneVision (arXiv:2408.03326)](https://arxiv.org/abs/2408.03326) — unified single-image、multi-image、video。
