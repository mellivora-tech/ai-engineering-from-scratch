# Watermarking：SynthID、Stable Signature、C2PA

> 三种技术构成 2026 AI-generated-content provenance。SynthID（Google DeepMind）：image watermarking 于 2023 年 8 月推出，text+video 于 2024 年 5 月推出（Gemini + Veo），text 于 2024 年 10 月通过 Responsible GenAI Toolkit 开源，2025 年 11 月与 Gemini 3 Pro 一起推出 unified multi-media detector。Text watermarking 以不可察觉的方式调整 next-token sampling probability；image/video watermark 能承受 compression、cropping、filter、frame-rate change。Stable Signature（Fernandez et al., ICCV 2023, arXiv:2303.15435）：fine-tune latent diffusion decoder，使每个 output 都包含固定 message；只剩 10% 内容的 cropped generated image 仍能以 >90% 检出率、FPR<1e-6 被检测。后续 “Stable Signature is Unstable”（arXiv:2405.07145，2024 年 5 月）：fine-tuning 会移除 watermark，同时保留质量。C2PA：cryptographically signed、tamper-evident metadata standard（C2PA 2.2 Explainer 2025）。Watermarking 与 C2PA 互补：metadata 可以被剥离但携带更丰富 provenance；watermark 能穿过 transcoding 但携带更少信息。

**类型：** 构建
**语言：** Python（stdlib，token-watermark embed + detect）
**前置要求：** 阶段 10 · 04（sampling），阶段 01 · 09（information theory）
**时间：** ~75 分钟

## 学习目标

- 描述 token-level watermarking（SynthID-text style）以及它可被检测的机制。
- 描述 Stable Signature 和 2024 年破坏它的 removal attack。
- 说明 C2PA 的角色，以及为什么它与 watermarking 互补。
- 描述关键限制：model-specific signal、paraphrase 下的 robustness，以及 meaning-preserving attack（arXiv:2508.20228）。

## 问题

2023-2024 年，deepfake 和 AI-generated content 大规模进入政治与消费语境。Watermarking 是被提出的技术 provenance signal：在创建时标记 generation，之后再检测。2025 年证据显示：没有 watermark 是无条件 robust 的，但与 C2PA metadata 分层使用时，组合能提供可用的 provenance story。

## 概念

### Text watermarking（SynthID-text style）

Kirchenbauer et al. 2023 机制，由 Google 生产化：

1. 在每个 decoding step，hash 前 K 个 token，产生一个把 vocabulary 分成 “green” 和 “red” set 的 pseudorandom partition。
2. 通过给 green logits 加 δ，把 sampling 偏向 green set。
3. Generation 中 green token 数量会高于随机产生的数量。

Detection：重新 hash 每个 prefix，统计 generation 中的 green token，计算 z-score。Watermarked text 的 z-score >0；human text 约为 0。

性质：
- 对 reader 不可察觉（δ 足够小，quality loss 较小）。
- 有 vocabulary partition function access 时可检测。
- 对 paraphrase 不 robust，重写文本会破坏 signal。

SynthID-text 于 2024 年 10 月通过 Google Responsible GenAI Toolkit 开源。

### Stable Signature（image）

Fernandez et al. ICCV 2023。Fine-tune latent diffusion decoder，让每张 generated image 都在 latent representation 中嵌入固定 binary message。Detection 由 neural decoder 从 latent 中解码。Cropped（到 10% 内容）image 检出率 >90%，FPR<1e-6。

2024 年 5 月 “Stable Signature is Unstable”（arXiv:2405.07145）：fine-tuning decoder 会移除 watermark，同时保留 image quality。Adversarial post-generation fine-tuning 便宜；watermark 的 adversarial robustness 有限。

### SynthID unified detector（2025 年 11 月）

与 Gemini 3 Pro 同时推出：一个 multi-media detector，在一个 API 中读取 text、image、audio、video 的 SynthID signal。统一了 Google provenance stack。

### C2PA

Coalition for Content Provenance and Authenticity。Cryptographically signed tamper-evident metadata standard。C2PA 2.2 Explainer（2025）。C2PA manifest 记录 provenance claim（谁创建、何时、做了什么 transformation），并由 creator key 签名。

与 watermarking 互补：
- Metadata 可以被剥离；watermark 不容易被剥离。
- Metadata 丰富（完整 provenance chain）；watermark 携带 bits。
- C2PA 依赖 platform adoption；watermark 自动嵌入。

Google 在 Search、Ads 和 “About this image” 中同时集成两者。

### 限制

- **Model-specific。** SynthID watermark 来自启用 SynthID 的模型。来自没有 SynthID 的模型的 generation 没有 watermark，因此“没有 SynthID signal”不是 authenticity proof。
- **Paraphrase。** Text watermark 无法承受 meaning-preserving paraphrase。
- **Transformation attacks。** arXiv:2508.20228（2025）展示了会破坏 text watermark 与许多 image watermark 的 meaning-preserving attack。
- **Fine-tune removal。** 按 “Stable Signature is Unstable”，post-generation fine-tuning 会移除 embedded watermark。

### EU AI Act Article 50

AI-generated content labeling 的 Transparency Code（2025 年 12 月第一稿，2026 年 3 月第二稿，按 [European Commission status page](https://digital-strategy.ec.europa.eu/en/policies/code-practice-ai-generated-content) 预计 2026 年 6 月最终稿）。截至 2026 年 4 月，该 Code 仍是草案，timeline 可能变化。这是要求技术层的监管层。Deepfake 必须标注。

### 它在阶段 18 中的位置

第 22-23 课讨论模型 emit 的东西（private data、provenance signal）。第 27 课覆盖 training-data governance。第 24 课是要求这些技术措施的 regulatory framework。

## 使用它

`code/main.py` 构建一个玩具 text watermark。Token 是整数 0..N-1；watermarked sampling 会偏向 hash-defined green set。Detector 计算 green-token z-score。你可以在 1000-token generation 上观察 detection，看到 paraphrase 破坏 signal，并测量 human text 上的 false-positive rate。

## 交付它

本课会生成 `outputs/skill-provenance-audit.md`。给定一个带 provenance claim 的 content deployment，它会 audit：watermark mechanism（如果有）、C2PA signing chain（如果有）、各自的 adversarial robustness，以及 per-modality coverage。

## 练习

1. 运行 `code/main.py`。报告 watermarked 1000-token generation 与 human-authored text 的 z-score。识别 95% confidence threshold 下的 false-positive rate。

2. 实现一个 paraphrase attack，用 synonym 替换 30% token。重新测量 z-score。

3. 阅读 Kirchenbauer et al. 2023 第 6 节关于 robustness 的内容。为什么 text watermark 在 paraphrase 下失败，而 image watermark 能承受 cropping？

4. 设计一个使用 SynthID-text + C2PA metadata 的 deployment。描述 consumer 看到的 provenance chain。指出每个组件的一个 failure mode。

5. 2024 “Stable Signature is Unstable” 结果表明 fine-tuning 会移除 image watermark。设计一个限制该 attack 的 deployment control，例如要求 fine-tuned checkpoint 的 signed releases。

## 关键词汇

| 术语 | 人们常说 | 实际含义 |
|------|-----------------|------------------------|
| SynthID | “Google's watermark” | Cross-modal provenance signal；text、image、audio、video |
| Token watermark | “Kirchenbauer-style” | 可用 green-token z-score 检测的 biased-sampling text watermark |
| Stable Signature | “image watermark” | fine-tuned-decoder watermark；ICCV 2023 |
| C2PA | “metadata standard” | cryptographically signed tamper-evident provenance metadata |
| Paraphrase robustness | “rewording 会不会打破它” | Text watermark 性质；目前有限 |
| Fine-tune removal | “adversarial unwatermark” | 通过 decoder fine-tuning 移除 image watermark 的 attack |
| Cross-modal detector | “unified SynthID” | 2025 年 11 月跨 modality unified API |

## 延伸阅读

- [Kirchenbauer et al. — A Watermark for Large Language Models (ICML 2023, arXiv:2301.10226)](https://arxiv.org/abs/2301.10226) — token-watermark mechanism
- [Fernandez et al. — Stable Signature (ICCV 2023, arXiv:2303.15435)](https://arxiv.org/abs/2303.15435) — image watermark paper
- ["Stable Signature is Unstable" (arXiv:2405.07145)](https://arxiv.org/abs/2405.07145) — removal attack
- [Google DeepMind — SynthID](https://deepmind.google/models/synthid/) — cross-modal watermark
- [C2PA 2.2 Explainer (2025)](https://c2pa.org/specifications/specifications/2.2/explainer/Explainer.html) — metadata standard
