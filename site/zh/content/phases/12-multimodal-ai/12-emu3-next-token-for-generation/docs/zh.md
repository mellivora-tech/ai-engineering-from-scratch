# Emu3：用于图像和视频生成的 Next-Token Prediction

> BAAI 的 Emu3（Wang 等人，2024 年 9 月）是本该终结 diffusion-versus-autoregressive 争论的 2024 年结果。一个 Llama-style decoder-only transformer，只用 next-token-prediction objective，在 text + VQ image tokens + 3D VQ video tokens 的统一 vocabulary 上训练，就能在 image generation 上超过 SDXL，在 perception 上超过 LLaVA-1.6。没有 CLIP loss。没有 diffusion schedule。Inference 时使用 classifier-free guidance 提升质量，但核心训练目标是带 teacher forcing 的 next-token prediction。发表于 Nature。本课阅读 Emu3 thesis，即为什么更好的 tokenizer 加规模就足够，并与 diffusion 方法对比。

**类型：** 学习
**语言：** Python（stdlib，3D video tokenizer math + autoregressive sampler skeleton）
**前置要求：** 阶段 12 · 11（Chameleon）
**时间：** ~120 分钟

## 学习目标

- 解释为什么 Emu3 的 single-loss next-token objective 能工作，尽管长期假设认为图像质量需要 diffusion。
- 描述 3D video tokenizer：spatiotemporal VQ codebook 是什么样，为什么 patch 跨越时间。
- 在 training compute、inference cost、quality ceiling 上比较 Emu3 与 Stable Diffusion XL。
- 说出同一个 Emu3 model 扮演的三个角色：Emu3-Gen（image gen）、Emu3-Chat（perception）、Emu3-Stage2（video gen）。

## 问题

到 2024 年为止，传统观点是：image generation 需要 diffusion。理由是：离散 image tokens 会丢失太多信息，无法重建细节；autoregressive sampling 会在数千个 token 上累积错误。Stable Diffusion、DALL-E 3、Imagen、Midjourney 都使用某种 diffusion。Chameleon（第 12.11 课）在小规模上部分反驳了这一点，但质量没有匹配 SDXL。

Emu3 正面挑战这个论点。它的 claim 是：更好的 visual tokenizer + 足够规模 + next-token loss = 在同一个也能做 perception 的模型中超过 diffusion 的 image generation。

发表时这个赌注很有争议。两年后，open-source unified-generation family（Emu3、Show-o、Janus-Pro、Transfusion）成为研究默认路径；production frontier models 看起来也使用某种变体。

## 概念

### Emu3 tokenizer

关键成分是 visual tokenizer。Emu3 训练了一个 custom IBQ-class tokenizer（Inverse Bottleneck Quantizer，SBER-MoVQGAN family），每个 token 做 8x8 resolution-reduction。512x512 图像变成 64x64 = 4096 tokens，codebook size 32768。

这比 Chameleon 的每张 512x512 图 1024 tokens、K=8192 更大，但每个 token 更便宜（更小 codebook lookups、更简单 codec）。关键指标是 reconstruction PSNR 30.5 dB，与 Stable Diffusion 的 continuous latent space 32 dB 具有竞争力。

对于视频：3D VQ tokenizer 把 spatiotemporal patch（4x4x4 pixels）编码为一个整数。4 秒、8 FPS 的 clip 有 32 帧；在 256x256 下，4x spatial 和 4x temporal reduction，token count 为 (256/4) * (256/4) * (32/4) = 64 * 64 * 8 = 32,768 tokens。

Tokenizer quality 是 ceiling。Emu3 的贡献一部分就是“我们训练了一个非常好的 tokenizer”。

### Single-loss training

Emu3 使用一个 objective：在 text tokens、2D image tokens、3D video tokens 的共享 vocabulary 上做 next-token prediction。训练时用 modality-specific factors 乘以权重，以平衡贡献，但 loss function 是同一个。

训练混合：
- Image gen：`<text caption> <image> image_tokens </image>`
- Image perception：`<image> image_tokens </image> <question> text_tokens`
- Video gen：`<text caption> <video> video_tokens </video>`
- Video perception：类似。
- Text only：标准 NTP。

模型从数据分布中学会何时发出 image tokens、何时发出 text tokens。Generation 从模型在 `<image>` tag 后预测 image tokens 中涌现。

### Classifier-free guidance 与 temperature

Autoregressive image generation 在 inference 时使用 classifier-free guidance（CFG）会好很多。Emu3 使用它：生成两次，一次使用完整 caption，一次使用 empty caption，然后用 guidance weight 混合 logits（常见 3.0-7.0）。这是 diffusion 使用的同一个 CFG 技巧，被借到 autoregressive setting。

Temperature 很重要：太高会有 artifacts；太低会 mode collapse。Emu3 推荐 perception 用 1.0，image generation 用 0.8。

### 三个角色，一个模型

Emu3 作为三个功能上不同的 API 发布，但底层是同一套权重：

- Emu3-Gen。Image generation。输入文本，输出 image tokens。
- Emu3-Chat。VQA 和 captioning。输入 image（tokens），输出文本。
- Emu3-Stage2。Video generation 和 video VQA。输入文本或视频，输出文本或视频。

没有 task-specific heads。只是 prompt templates 不同。同一个 checkpoint。

### Benchmarks

来自 Emu3 论文（2024 年 9 月）：

- Image generation：在 MJHQ-30K FID 上超过 SDXL（5.4 vs 5.6），GenEval overall（0.54 vs 0.55，统计上打平），Deep-Eval composite 接近。
- Image perception：在 VQAv2 上超过 LLaVA-1.6（75.1 vs 72.4），在 MMMU 上大致匹配。
- Video generation：4 秒 clip 质量在 FVD 上与 Sora-era 公开 benchmarked models 有竞争力。

数字并不总是全赢，Emu3 会在这里换一分、那里换一分，但“next-token prediction is all you need” 在多 modality 上是可辩护的。

### Compute cost

Emu3 是在约 300B multimodal tokens 上训练的 7B-parameter model。GPU-hours 大致与 Llama-2-7B pretraining 相当（A100-class silicon 上 2k-4k GPU-years）。Stable Diffusion 3 这样的 diffusion models 训练预算类似，但需要 separate text encoders 和更复杂 pipelines。

推理时，Emu3 每张图比 SDXL 慢：4096 image tokens，30 tok/s 时约 2 分钟生成一张 512x512 图，而 SDXL 是 2-5 秒。Speculative decoding 和 KV-cache optimization 可以缩小差距，但无法消除。Autoregressive image gen compute-heavy，这是持续存在的 trade-off。

### 为什么它重要

Emu3 的深层贡献是概念性的。如果 next-token prediction 能 scale 到在 image generation 上匹配 diffusion，那么 unified-model path（一种 loss、一个 backbone、任意 modality）就是可行的。未来模型不需要 separate text encoders、separate diffusion schedulers、separate VAEs。一个 transformer、每个 modality 一个 tokenizer、scale。

Show-o、Janus-Pro 和 InternVL-U 都在这个 thesis 上构建或挑战它。到 2025 年，中国实验室（BAAI、DeepSeek）在这个方向上比美国实验室发表得更积极。

## 使用它

`code/main.py` 构建两个 toy pieces：

- 2D vs 3D VQ tokenizer count calculator：给定（resolution、patch、clip_length、FPS），计算 image vs video 的 token counts。
- 一个带 classifier-free guidance 和 temperature 的 autoregressive image-token sampler。

CFG 实现匹配 Emu3 recipe：用 guidance weight 混合 conditional 和 unconditional logits。

## 交付它

本课产出 `outputs/skill-token-gen-cost-analyzer.md`。给定 generation product spec（image 或 video、target resolution、quality tier、latency budget），它计算 token counts、inference cost，并在 Emu3-family 与 diffusion 之间选择。

## 练习

1. Emu3 在 8x8 reduction 下，每张 512x512 图产生 4096 tokens。计算 1024x1024 和 2048x2048 的等价 token 数。Inference latency 会怎样？

2. 阅读 Emu3 第 3.3 节关于 video tokenizer 的内容。描述 3D VQ patch shape，以及为什么是 4x4x4 而不是 8x8x1。

3. Classifier-free guidance weight 5.0 vs 3.0：视觉效果是什么？在 `code/main.py` 中追踪数学。

4. 计算 Emu3-7B 在 300B tokens 上的 training FLOPs，并与 Stable Diffusion 3 对比。哪个训练更贵？

5. Emu3 在 FID 上超过 SDXL，但在 VQAv2 上没有超过 specialized VLMs。解释为什么 unified-loss approach 在不同 benchmark 上相对 specialists 展现不同强项。

## 关键术语

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| Next-token prediction | “NTP” | 标准 autoregressive loss：给定 token[0..i] 预测 token[i+1]；所有 modality tokenize 后都可用 |
| IBQ tokenizer | “Inverse bottleneck quantizer” | 一类 VQ-VAE，使用更大 codebook（32768+），reconstruction 优于 Chameleon |
| 3D VQ | “Spatiotemporal quantizer” | 由（time, row, col）索引的 codebook；一个 token 覆盖 4x4x4 pixel cube |
| Classifier-free guidance | “CFG” | 用权重 gamma 混合 conditional 与 unconditional logits；推理时提升 image quality |
| Unified vocabulary | “Shared tokens” | Text + image + video 都来自同一个 integer space；模型预测下一个出现的 modality |
| MJHQ-30K | “Image gen benchmark” | 含 30k prompts 的 Midjourney-quality benchmark；Emu3 在这里报告 FID |

## 延伸阅读

- [Wang et al. — Emu3: Next-Token Prediction is All You Need (arXiv:2409.18869)](https://arxiv.org/abs/2409.18869)
- [Sun et al. — Emu: Generative Pretraining in Multimodality (arXiv:2307.05222)](https://arxiv.org/abs/2307.05222)
- [Liu et al. — LWM (arXiv:2402.08268)](https://arxiv.org/abs/2402.08268)
- [Yu et al. — MAGVIT-v2 (arXiv:2310.05737)](https://arxiv.org/abs/2310.05737)
- [Tian et al. — VAR (arXiv:2404.02905)](https://arxiv.org/abs/2404.02905)
