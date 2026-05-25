# 从 CLIP 到 BLIP-2：作为 Modality Bridge 的 Q-Former

> CLIP 能对齐图像和文本，但不能生成 caption、回答问题或进行对话。BLIP-2（Salesforce，2023）用一个很小的可训练 bridge 解决了这个问题：32 个可学习 query vector 通过 cross-attention 读取 frozen ViT 的特征，然后直接插入 frozen LLM 的输入流。188M 参数的 bridge 把一个 11B LLM 连接到 ViT-g/14。直到 2026 年，每个 adapter-based VLM（MiniGPT-4、InstructBLIP、LLaVA 的近亲）都是它的后代。本课阅读 Q-Former 架构，解释它的两阶段训练，并构建一个 toy 版本，把 visual token 喂给 frozen text decoder。

**类型：** 构建
**语言：** Python（stdlib，cross-attention + learnable-query demo）
**前置要求：** 阶段 12 · 02（CLIP），阶段 7（Transformers）
**时间：** ~180 分钟

## 学习目标

- 解释为什么在 frozen vision encoder 和 frozen LLM 之间放一个可训练 bottleneck，在成本和稳定性上优于 end-to-end finetuning。
- 实现一个 cross-attention block，其中固定数量的 learnable queries attend 到外部 image features。
- 走通 BLIP-2 的两阶段预训练：representation（ITC + ITM + ITG），然后 generative（对 frozen decoder 使用 LM loss）。
- 比较 Q-Former 与 LLaVA 使用的更简单 MLP projector，并说明什么时候各自更优。

## 问题

你有一个 frozen ViT，每张图产生 256 个 dim 1408 的 patch token。你还有一个 frozen 7B LLM，它期待 dim 4096 的 token embedding。最直接的 bridge 是从 1408 到 4096 的线性层；它能工作，但把全部 256 个 patch token 喂进 LLM context，会让每张图多消耗 256 个 token。对于 32 张图的 batch，visual modality alone 就会消耗 8192 个 token。

BLIP-2 的问题是：能不能把 256-token 图像 representation 压缩成少得多的 token（比如 32 个），同时保留足够信息，让 LLM 能 caption、回答问题、推理图像？并且能不能在不动 frozen backbones 的情况下训练这个 bridge，让训练成本只来自 bridge 参数？

答案是 Q-Former。32 个可学习 “query” vector 对 ViT 的 patch token 做 cross-attention，产生 32-token visual summary 供 LLM 消费。总共 188M 参数。在碰到 LLM 之前，先用 contrastive、matching 和 generative objectives 训练。

## 概念

### Learnable queries

Q-Former 的核心技巧是：不要让 LLM 的 text token attend 到 image patches，而是引入一组新的 32 个 learnable query vector `Q`，让它们 attend 到 image patches。Queries 是模型参数，在训练期间学习，同一组 32 个 query 用于每张图。

经过 cross-attention 后，每个 query 都携带图像的压缩摘要，例如“描述主物体”“描述背景”“数物体”等。Queries 不会真的按语义标签专业化；它们学习的是能让下游 loss 下降的编码。

### 架构

Q-Former 是一个小型 transformer（12 层，约 100M params），有两条路径：

1. Query path：32 个 query vector 经过 self-attention（彼此之间）、再对 frozen ViT 的 patch token 做 cross-attention，然后经过 FFN。
2. Text path：一个 BERT-like text encoder，与 query path 共享 self-attention 和 FFN 权重。Text path 禁用 cross-attention。

训练时两条路径都会运行。Queries 和 text 通过共享 self-attention 交互，这意味着在需要的任务（ITM、ITG）中，queries 可以以文本为条件。推理时做 VLM handoff，只让 queries 流过，得到 32 个 visual token。

### 两阶段训练

BLIP-2 分两阶段预训练：

Stage 1：representation learning（没有 LLM）。三个 loss：
- ITC（image-text contrastive）：CLIP-style contrastive，作用在 pooled query tokens 与 text CLS token 之间。
- ITM（image-text matching）：binary classifier，判断这个 image-text pair 是否匹配；使用 hard-negative-mined 样本。
- ITG（image-grounded text generation）：在以 queries 为条件的文本上做 causal LM head。迫使 queries 编码可由文本生成的内容。

只训练 Q-Former。ViT frozen。不涉及 LLM。

Stage 2：generative learning。接入一个 frozen LLM（OPT-2.7B 或 Flan-T5-XL 等）。通过一个小型线性层把 32 个 query output 投影到 LLM embedding dim。把它们 prepend 到 text prompt。只训练 linear projection 和 Q-Former，对拼接的 prompt + image + caption sequence 使用 LM loss。

Stage 2 之后，Q-Former + projection 就是完整 visual adapter。推理时：image -> ViT -> Q-Former -> linear proj -> prepend 到 text -> frozen LLM 输出结果。

### 参数经济学

BLIP-2 使用 ViT-g/14（1.1B，frozen）+ OPT-6.7B（6.7B，frozen）+ Q-Former（188M，trained）= 总共 8B，训练 188M。Q-Former 只占整套 stack 参数的约 2.4%。训练成本也反映了这一点：在少量 A100 上训练数天，而不是 end-to-end 训练数周。

质量：BLIP-2 在 zero-shot VQA 上匹配或超过 Flamingo-80B，同时小 50 倍。这个 bridge 是有效的。

### InstructBLIP 与 instruction-aware Q-Former

InstructBLIP（2023）给 Q-Former 加了一个额外输入：instruction text 本身。在 cross-attention 时，queries 现在同时能访问 image patches 和 instruction。Queries 可以按 instruction 专门化（“count the cars”“describe the mood”），而不是学习一个固定 summary。在 held-out task benchmark 上有收益。

### MiniGPT-4 与 projector-only 方案

MiniGPT-4 保留了 Q-Former，但只训练输出 linear projection，其他全部 frozen。便宜，但代价是质量，因为 queries 是 BLIP-2 的，不是你自己的。适合快速迭代，但不是最佳架构。

### 为什么 LLaVA 选择了更简单的做法

LLaVA（2023，第 12.05 课）用一个普通 2-layer MLP 替换 Q-Former，把每个 ViT patch token 投影到 LLM space。对于 24x24 grid，每张图 576 个 token，全部喂给 LLM。压缩更差，但让 LLM 可以 attend 到原始 patch。当时这很有争议；到 2023 年末它成为主流，因为 visual instruction data（LLaVA-Instruct-150k）证明 MLP 能被训练到保留足够信号。Tradeoff 是：LLaVA 的 context 填得更快，但可以自然扩展到 multi-image 和 video。

到 2026 年，领域分裂为两派：当 token budget 重要时（long video、多图）Q-Former 仍然存在；当 priority 是 raw quality per token 时，MLP projector 占主导。

### Gated cross-attention：Flamingo 这个祖先

Flamingo（第 12.04 课）早于 BLIP-2，使用了相同的 cross-attention 思路，但它把 cross-attention 放在每个 frozen LLM layer，而不是单个 bridge。BLIP-2 证明了只压缩到 input layer 也可以工作。Gemini 和 Idefics 结合了两者：interleaved input tokens 加上可选的 gated cross-attention，用于 in-context few-shot。

### 2026 年后代

- Q-Former：BLIP-2、InstructBLIP、MiniGPT-4，以及大多数出于 token budget 考虑的视频语言模型。
- Perceiver resampler：Flamingo 的变体（第 12.04 课）；Idefics family、Eagle、OmniMAE。
- MLP projector：LLaVA、LLaVA-NeXT、LLaVA-OneVision、Cambrian-1。
- Attention pool：VILA、PaliGemma。

四种都合理。决定性问题是你受限于 token budget，还是受限于 quality-per-token。

## 使用它

`code/main.py` 构建了一个 stdlib Q-Former 风格的 cross-attention：

1. 模拟 256 个 image patch token（dim 128）。
2. 实例化 32 个 learnable query（dim 128）。
3. 运行 scaled-dot-product cross-attention（Q 来自 queries，K/V 来自 patches）。
4. 通过线性层投影到 LLM-dim（512）。
5. 输出 32 个 LLM-ready visual token。

所有数学都用纯 Python（对向量做嵌套循环）。它是 toy，但 shape 正确。程序会打印 attention-weight matrix，让你看到每个 query 从哪些 patch 中取信息。

## 交付它

本课产出 `outputs/skill-modality-bridge-picker.md`。给定一个目标 VLM 配置（vision encoder token count、LLM context budget、deployment constraints、quality target），它会推荐 Q-Former、MLP 或 Perceiver resampler，并给出简短理由与每种 bridge 的参数量估算。

## 练习

1. 用 PyTorch 实现 cross-attention block。验证在 32 个 queries 和 256 个 keys/values 下，attention-weight matrix 是 32 x 256，并且 softmax 后每行求和为 1。

2. 在 BLIP-2 stage 1 中，Q-Former 同时运行三个 loss：ITC、ITM、ITG。为每个 loss 写出 pseudo-code forward signature。哪一个需要启用 text encoder path？

3. 比较参数量：Q-Former（12 层，768 hidden）vs 2-layer MLP projector（1408 -> 4096，两层）。在多大 LLM scale 下，188M Q-Former 的成本能通过训练效率收回？

4. 阅读 BLIP-2 论文（arXiv:2301.12597）第 3.2 节，了解 Q-Former 如何初始化。解释为什么从 BERT-base 初始化（而不是随机初始化）能加速收敛。

5. 对一个 10 分钟视频以 1 FPS 采样到 60 帧，计算每帧 token 成本：（Q-Former -> 32 tokens/frame）vs（MLP projector -> 576 tokens/frame）。哪一个能放进 128k-token LLM context window？

## 关键术语

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Q-Former | “Querying transformer” | 带有 32 个 learnable query vector 的小型 transformer，对 frozen ViT features 做 cross-attention |
| Learnable queries | “Soft prompt for vision” | 一组固定参数，作为 cross-attention 的 query 侧；每个模型学习一次，并在所有输入间共享 |
| Cross-attention | “Q from here, K/V from there” | Query、key、value 来自不同来源的 attention；queries 通过它从 ViT patches 中取信息 |
| ITC | “Image-text contrastive” | 应用于 Q-Former pooled queries 与 text CLS 之间的 CLIP-style loss |
| ITM | “Image-text matching” | 对 hard-negative-mined pairs 做 binary classifier；迫使 queries 区分细粒度 mismatch |
| ITG | “Image-grounded text generation” | 以 queries 为条件生成文本的 causal LM loss；迫使 queries 编码可被文本解码的内容 |
| Two-stage pretraining | “Representation then generative” | Stage 1 单独训练 Q-Former（ITC/ITM/ITG）；Stage 2 接入 frozen LLM，只训练 projection + Q-Former |
| Frozen backbone | “Do not finetune” | Vision encoder 与 LLM 权重固定；只训练 bridge |
| Projection head | “Linear to LLM dim” | 把 Q-Former output 映射到 LLM embedding dimension 的最终线性层 |
| Perceiver resampler | “Flamingo's version” | 类似的 learnable-query cross-attention；Flamingo 在每层使用它，而不是作为单个 bridge |

## 延伸阅读

- [Li et al. — BLIP-2 (arXiv:2301.12597)](https://arxiv.org/abs/2301.12597) — 核心论文。
- [Li et al. — BLIP (arXiv:2201.12086)](https://arxiv.org/abs/2201.12086) — 带有 ITC/ITM/ITG 三件套的前身。
- [Li et al. — ALBEF (arXiv:2107.07651)](https://arxiv.org/abs/2107.07651) — “align before fuse”，stage 1 training 的概念祖先。
- [Dai et al. — InstructBLIP (arXiv:2305.06500)](https://arxiv.org/abs/2305.06500) — instruction-aware Q-Former。
- [Zhu et al. — MiniGPT-4 (arXiv:2304.10592)](https://arxiv.org/abs/2304.10592) — projector-only 方案。
- [Jaegle et al. — Perceiver IO (arXiv:2107.14795)](https://arxiv.org/abs/2107.14795) — learnable-query cross-attention 的通用架构。
