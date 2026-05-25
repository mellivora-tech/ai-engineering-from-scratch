# Show-o 与 Discrete-Diffusion Unified Models

> Transfusion 混合 continuous 和 discrete representations。Show-o（Xie 等人，2024 年 8 月）反向选择：text tokens 使用 causal next-token prediction，image tokens 使用 MaskGIT 风格的 masked discrete diffusion。二者位于一个带 hybrid attention mask 的 transformer 中。结果是在一个 backbone、每个 modality 一个 tokenizer、一种 loss formulation（next-token 扩展到 masked prediction）上统一 VQA、text-to-image、inpainting 和 mixed-modality generation。本课讲解 Show-o 设计，即为什么 masked discrete diffusion 是一种 parallel、few-step image generator，并与 Transfusion 和 Emu3 对比。

**类型：** 学习
**语言：** Python（stdlib，masked-discrete-diffusion sampler）
**前置要求：** 阶段 12 · 13（Transfusion）
**时间：** ~120 分钟

## 学习目标

- 解释 masked discrete diffusion：uniformly mask tokens，然后要求 transformer 恢复它们的 schedule。
- 在速度和质量上比较 parallel image decoding（Show-o、MaskGIT）与 autoregressive image decoding（Chameleon、Emu3）。
- 说出 Show-o 在一个 checkpoint 中处理的三个任务：T2I、VQA、image inpainting。
- 选择 masking schedule（cosine、linear、truncated），并推理它对 sample quality 的影响。

## 问题

Transfusion 的 two-loss training 有效，但动态更麻烦，因为 continuous diffusion loss 与 discrete NTP loss 位于不同 numerical scale。平衡 loss weights 是一次 hyperparameter search。架构有效但复杂。

Show-o 的答案是：像 Chameleon 一样让两个 modality 都保持 discrete，但用 masked discrete diffusion 并行生成图像，而不是顺序生成。训练目标变成 single masked-token-prediction，自然推广 next-token-prediction。

## 概念

### Masked discrete diffusion（MaskGIT）

原始 Chang 等人（2022）的 MaskGIT 技巧很优雅。从完全 masked image 开始（每个 token 都是特殊 `<MASK>` id）。每一步并行预测所有 masked tokens，然后保留 top-K 最自信预测，重新 mask 其余部分。约 8-16 次迭代后，所有 token 都被填满。每步 unmask 多少 token 的 schedule 需要调；cosine schedules 表现很好。

训练很简单：从 [0, 1] 均匀采样 masking ratio，把它应用到图像 VQ tokens，训练 transformer 恢复 masked ones。正是 BERT 为文本做的事，扩展到 image generation。

### Show-o：一个 transformer，hybrid mask

Show-o 把 MaskGIT 放进 causal-language-model transformer。Attention mask 是：

- Text tokens：causal（标准 LLM）。
- Image tokens：在 image block 内 full bidirectional（这样 masked tokens 在预测时能看到每个其他 image token）。
- Text-to-image：text attend 到 prior images，image attend 到 prior text。

训练在三者间切换：
1. Text sequences 上的标准 NTP。
2. T2I samples：text -> image，带 masked image tokens，使用 masked-token-prediction loss。
3. VQA samples：image -> text，带 masked text tokens（实际就是 NTP）。

Unified loss 是 `<MASK>` tokens 上的 cross-entropy，覆盖 text NTP（只有最后一个 token 被“masked”）和 image masked-diffusion（随机子集被 masked）。

### Parallel sampling

Show-o 约 16 步生成图像，而不是约 1000 步（按 token autoregressive）或约 20 步（diffusion）。每一步并行预测所有 masked tokens；提交 top-K confident；重复。

对比：
- Chameleon / Emu3（tokens 上 autoregressive）：每张图 N_tokens 次 forward passes，通常 1024-4096。
- Transfusion（continuous diffusion）：约 20 步，每步一个完整 transformer pass。
- Show-o（masked discrete diffusion）：约 16 步，每步一个完整 transformer pass。

Show-o 比同规模 Chameleon 更快，step count 大致匹配 Transfusion，但每步成本更低（discrete vocab logits vs continuous MSE loss）。

### 一个 checkpoint 中的任务

Show-o 在 inference 时支持四个任务，由 prompt format 选择：

- Text generation：标准 autoregressive text output。
- VQA：image in，text out。
- T2I：text in，通过 masked discrete diffusion 输出 image。
- Inpainting：带有一些 masked tokens 的 image，填补它们。

Inpainting 能力来自 masked-prediction training，几乎是免费得到的。Mask VQ-token grid 的一个区域，输入剩余部分加 text prompt，预测 masked tokens。

### Masking schedule

每步 unmask 多少 token 的 schedule 会塑造质量。Show-o 推荐 cosine：

```
mask_ratio(t) = cos(pi * t / (2 * T))   # t = 0..T
```

Step 0 时所有 token 被 mask（ratio 1.0）。Step T 时没有 token 被 mask。Cosine 把质量集中在 prediction 最有信息量的中间 ratio 区间。Linear schedules 也可用，但更早平台化。

### Show-o2

Show-o2（2025 follow-up，arXiv 2506.15564）扩展了 Show-o：更大的 LLM base、更好的 tokenizer、改进 mask schedule。同样的架构模式。

### Show-o 的位置

在 2026 年 taxonomy 中：

- Discrete tokens + NTP：Chameleon、Emu3。简单但 inference 慢。
- Discrete tokens + masked diffusion：Show-o、MaskGIT、LlamaGen、Muse。并行 sampling，但仍受 tokenizer 有损限制。
- Continuous + diffusion：Transfusion、MMDiT、DiT。最高质量，训练更复杂。
- Continuous + flow matching in a VLM：JanusFlow、InternVL-U。最新。

按任务选择：当你想在一个 open model 中同时要 T2I + inpainting + VQA，并且希望速度合理，选 Show-o；当质量最重要且能承受 two-loss plumbing，选 Transfusion。

## 使用它

`code/main.py` 模拟 Show-o sampling：

- 一个 16 个 VQ tokens 的 toy grid。
- 一个 mock “transformer”，基于 prompt 和当前 unmasked tokens 预测 logits。
- 使用 cosine schedule，做 8 步 parallel masked sampling。
- 打印中间状态（mask pattern evolution）和最终 tokens。

运行它，看 mask 一步步溶解。

## 交付它

本课产出 `outputs/skill-unified-gen-model-picker.md`。给定一个既需要 understanding（VQA、captioning）又需要 generation（T2I、inpainting），并且要求 open-weights 的产品，它会在 Show-o family、Transfusion/MMDiT family、Emu3 / Chameleon family 中选择，并给出具体 trade-offs。

## 练习

1. Masked discrete diffusion 约 16 步采样。为什么不是 1 步？如果在 step 0 一次性 unmask 一切，会坏在哪里？

2. Inpainting 对 masked diffusion 是免费的。提出一个真实或假设的产品用例，其中 Show-o 的 inpainting 优于 specialist model。

3. Cosine schedule vs linear schedule：追踪 T=8 时每步 unmasked tokens 数量。哪一个更均衡？

4. 一个 512x512 Show-o 图像是 1024 tokens。Vocab K=16384 时，模型输出 1024 * log2(16384) = 14,336 bits（约 1.75 KiB）数据。Stable Diffusion 输出 512*512*24 bits = 6,291,456 bits（约 768 KiB）raw pixels。Compression ratio 是多少，它买来了什么质量？

5. 阅读 LlamaGen（arXiv:2406.06525）。LlamaGen 的 class-conditional autoregressive image model 与 Show-o 的 masked approach 有什么不同？

## 关键术语

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| Masked discrete diffusion | “MaskGIT-style” | 训练预测 masked tokens；推理时迭代 unmask 最自信预测 |
| Cosine schedule | “Unmask schedule” | 推理步骤中 mask ratio 的衰减；让 confidence growth 集中在中间阶段 |
| Parallel decoding | “All tokens at once” | 每一步一次 forward pass 预测完整 masked sequence，然后提交 top-K |
| Hybrid attention | “Causal + bidirectional” | 对 text tokens causal、对 image blocks bidirectional 的 mask |
| Inpainting | “Fill-in generation” | 以部分 tokens 被 mask 的图像为条件，预测缺失部分；训练目标免费带来 |
| Commitment rate | “Top-K per step” | 每次迭代宣告“完成”的 token 数量；控制 inference vs quality trade-off |

## 延伸阅读

- [Xie et al. — Show-o (arXiv:2408.12528)](https://arxiv.org/abs/2408.12528)
- [Show-o2 (arXiv:2506.15564)](https://arxiv.org/abs/2506.15564)
- [Chang et al. — MaskGIT (arXiv:2202.04200)](https://arxiv.org/abs/2202.04200)
- [Sun et al. — LlamaGen (arXiv:2406.06525)](https://arxiv.org/abs/2406.06525)
- [Chang et al. — Muse (arXiv:2301.00704)](https://arxiv.org/abs/2301.00704)
