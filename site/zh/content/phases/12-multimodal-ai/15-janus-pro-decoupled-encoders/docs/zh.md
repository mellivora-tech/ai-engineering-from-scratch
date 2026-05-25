# Janus-Pro：Unified Multimodal Models 的 Decoupled Encoders

> Unified multimodal models 有一个无法回避的张力。Understanding 需要 semantic features，即 SigLIP 或 DINOv2 输出的、富含 concept-level information 的向量。Generation 需要 reconstruction-friendly codes，即能组合回清晰 pixels 的 VQ tokens。这两个目标在一个 encoder 中不兼容。Janus（DeepSeek，2024 年 10 月）和 Janus-Pro（DeepSeek，2025 年 1 月）认为修复方式是停止强行合并：decouple 两个 encoders。在任务之间共享 transformer body，但 understanding 通过 SigLIP 路由，generation 通过 VQ tokenizer 路由。7B 的 Janus-Pro 在 GenEval 上超过 DALL-E 3，同时在 MMMU 上匹配 LLaVA。本课阅读为什么两个 encoders 能解决一个 encoder 失败的问题。

**类型：** 构建
**语言：** Python（stdlib，dual-encoder routing + shared-body signal）
**前置要求：** 阶段 12 · 13（Transfusion），阶段 12 · 14（Show-o）
**时间：** ~120 分钟

## 学习目标

- 解释为什么单个 shared encoder 会牺牲 understanding 或 generation 质量。
- 描述 Janus-Pro 的 routing：input side understanding 使用 SigLIP features，generation 的 input 与 output 都使用 VQ tokens。
- 追踪让 Janus-Pro 成功而 Janus 没有成功的 data-mix scaling。
- 比较 decoupled（Janus-Pro）、coupled-continuous（Transfusion）和 coupled-discrete（Show-o）architectures。

## 问题

Unified models 在 understanding 和 generation 间共享 transformer body。此前尝试（Chameleon、Show-o、Transfusion）都用一个 visual tokenizer 处理两个方向。这个 tokenizer 是折中：

- 为 reconstruction（generation）优化：VQ-VAE 捕捉细粒度 pixel detail，但 tokens 的 semantic coherence 弱。
- 为 semantics（understanding）优化：SigLIP embeddings 让 “cat” 图像靠近 “cat” tokens，但无法良好重建。

Show-o 和 Transfusion 为此在某一个方向付出明显质量税。Janus-Pro 问：任务需求不同，为什么还要求一个 tokenizer？

## 概念

### Decoupled visual encoding

Janus-Pro 架构分离两个 encoders：

- Understanding path。Input image -> SigLIP-SO400m -> 2-layer MLP -> transformer body。
- Generation path。Input image（如果以现有图像为条件）-> VQ tokenizer -> token IDs -> transformer body。
- Output generation。Transformer 预测 image tokens -> VQ decoder -> pixels。

Transformer body 是共享的。Body 上游和下游的一切都是 task-specific。

输入通过 prompt format disambiguate：`<understand>` tag 走 SigLIP；`<generate>` 走 VQ。也可以从任务隐式路由。

### 为什么有效

Understanding loss 获得 SigLIP features，这些 features 已经通过 CLIP-style pretraining 被调成 semantic similarity。模型的 perception benchmarks 超过 Show-o / Transfusion，因为 input features 更适合任务。

Generation loss 获得 VQ tokens，这些 tokens 由 tokenizer 为 reconstruction 调过。Image quality 超过 Show-o，因为 VQ codes 能干净组合回 pixels。

Shared transformer body 看到两种 input distributions（SigLIP 和 VQ），并学会同时处理。Claim 是：足够数据 + 足够参数，body 能吸收这个切换。

### Data scaling：Janus vs Janus-Pro

Janus（原始版，arXiv 2410.13848）引入了 decoupling，但规模小（1.3B params，有限数据）。Janus-Pro（arXiv 2501.17811）扩展：

- 7B params（vs 1.3B）。
- Stage 1（alignment）使用 90M image-text pairs，从 72M 增加。
- Stage 2（unified）使用 72M，从 26M 增加。
- Stage 3 增加 200k image-gen instruction samples。

结果：Janus-Pro-7B 在 MMMU 上匹配 LLaVA（60.3 vs ~58），在 GenEval 上超过 DALL-E 3（0.80 vs 0.67）。一个 open model，在 unified spectrum 两侧都有竞争力。

### JanusFlow：rectified flow 变体

JanusFlow（arXiv 2411.07975）把 VQ generation path 换成 rectified-flow generation path（continuous）。拆分变成 SigLIP-for-understanding + rectified-flow-for-generation。Quality ceilings 进一步提高。架构仍然是 decoupled-encoders-shared-body。

### Shared body 的工作

Transformer body 处理统一序列，但输入分布有两种。它的工作是：

- Understanding：消费 SigLIP features + text tokens -> autoregressively 输出文本。
- Generation：消费 text tokens +（可选 image VQ tokens）-> autoregressively 输出 image VQ tokens。

Body 的每个 block 没有 modality-specific weights。它就是你期待在 Qwen 或 Llama 中看到的 text-style transformer，再加上两个 input adapters。

有趣的是，这意味着 Janus-Pro 的 body 可以从 pretrained LLM 初始化。Janus-Pro 确实从 DeepSeek-MoE-7B 初始化。这个选择很重要：LLM 贡献了 pure-from-scratch unified models 难以达到的 reasoning ability。

### 与 InternVL-U 对比

InternVL-U（第 12.10 课）是 2026 年 follow-up。它结合：

- Native multimodal pretraining（InternVL3 backbone）。
- Decoupled-encoder routing（SigLIP in，VQ + diffusion heads out）。
- Unified understanding + generation + editing。

InternVL-U 把 Janus-Pro 的架构选择纳入更大的框架。Decoupled-encoder idea 现在是 scale 级 unified models 的默认方式。

### 限制

Decoupled encoders 增加架构复杂度。要训练两个 tokenizers，维护两条输入路径，两组 fail modes。对于不需要 generation 的产品，Janus-Pro 过度设计了，选择 LLaVA-family understanding model。

对于不需要 understanding 的产品，Janus-Pro 也过度了，选择 Stable Diffusion 3 / Flux model。

对于两者都需要的产品，Janus-Pro 现在是参考 open architecture。

## 使用它

`code/main.py` 模拟 Janus-Pro routing：

- 两个 mock encoders：SigLIP-like（产生 256-dim semantic vectors）和 VQ-like（产生 integer codes）。
- 一个 prompt router，根据 task tag 选择 encoder。
- 一个 shared body（stand-in），无论哪个 encoder 产生 token sequence，都处理它。
- 从 stage 1（alignment）到 stage 3（instruction tune）的 weighted-sample schedule 切换。

打印 3 个示例的 routed paths：image QA、T2I、image editing。

## 交付它

本课产出 `outputs/skill-decoupled-encoder-picker.md`。给定一个希望在 frontier-ish quality 下统一 generation + understanding 的产品，它会选择 Janus-Pro、JanusFlow 或 InternVL-U，并给出具体 data-scale recommendation。

## 练习

1. Janus-Pro-7B 在 GenEval 上超过 DALL-E 3。解释为什么 7B open model 可以在 generation 上匹配 frontier proprietary model，但不能在 understanding 上匹配。

2. 实现一个 router function：给定 prompt text，分类为 `understand` 或 `generate`。如何处理像 “describe and then sketch” 这种 ambiguous prompts？

3. JanusFlow 用 rectified flow 替换 VQ path。Transformer body 现在输出什么，loss 有什么变化？

4. 提出 Janus-Pro 架构可以用一个额外 decoupled encoder 处理的第四种任务。例如：image segmentation（DINO-style）、depth（MiDaS-style）。

5. 阅读 Janus-Pro 第 4.2 节关于 data scaling 的内容。哪个 data stage 对相较 Janus 的 T2I quality gain 贡献最大？

## 关键术语

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| Decoupled encoding | “Two visual encoders” | 每个方向使用单独 tokenizer 或 encoder：understanding 用 semantic，generation 用 reconstruction |
| Shared body | “One transformer” | 单个 transformer 处理任一 encoder 的输出；没有 modality-specific weights |
| SigLIP for understanding | “Semantic features” | CLIP-family vision tower，提供丰富概念特征但重建能力差 |
| VQ for generation | “Reconstruction codes” | Vector-quantized tokens，可干净解码回 pixels |
| JanusFlow | “Rectified-flow variant” | 用 continuous flow-matching generation head 替代 VQ 的 Janus-Pro |
| Routing tag | “Task tag” | 选择 input encoder 的 prompt marker（`<understand>` / `<generate>`） |

## 延伸阅读

- [Wu et al. — Janus (arXiv:2410.13848)](https://arxiv.org/abs/2410.13848)
- [Chen et al. — Janus-Pro (arXiv:2501.17811)](https://arxiv.org/abs/2501.17811)
- [Ma et al. — JanusFlow (arXiv:2411.07975)](https://arxiv.org/abs/2411.07975)
- [InternVL-U (arXiv:2603.09877)](https://arxiv.org/abs/2603.09877)
- [Dong et al. — DreamLLM (arXiv:2309.11499)](https://arxiv.org/abs/2309.11499)
