# MIO 与 Any-to-Any Streaming Multimodal Models

> GPT-4o 提供了大多数 open models 无法复制的产品：一个能听见语音、看见视频，并实时说话回应的 agent。到 2024 年末，开放生态的答案是 MIO（Wang 等人，2024 年 9 月）。MIO tokenize text、image、speech 和 music，在 interleaved sequences 上训练一个 causal transformer，并实现任意 modality 到任意 modality 的生成。AnyGPT（Zhan 等人，2024 年 2 月）是 proof of concept；MIO 是 scale-up；Unified-IO 2（Allen AI，2023 年 12 月）是带 vision + action grounding 的近亲。本课阅读 any-to-any pattern：四个 tokenizers，一个 transformer，streaming-friendly decode。

**类型：** 学习
**语言：** Python（stdlib，four-modality token allocator + streaming decode loop）
**前置要求：** 阶段 12 · 11（Chameleon），阶段 6（Speech and Audio）
**时间：** ~120 分钟

## 学习目标

- 设计一个共享 vocabulary，让 text、image、speech 和 music tokens 不发生 ID collision。
- 在 compression + reconstruction trade-offs 上比较 SEED-Tokenizer（images）与 SpeechTokenizer residual-VQ（speech）。
- 解释构建 any-to-any generation 的四阶段 curriculum。
- 说出三个开放 any-to-any recipes 及其主要 trade-offs：MIO、AnyGPT、Unified-IO 2。

## 问题

统一 multimodal model 很容易宣称，但规模化构建很难。直到 2024 年，多数 “any-to-any” 系统都是 pipeline：vision model -> text representation -> speech model -> audio。每一跳都会丢信息、增加 latency，并让训练复杂化。GPT-4o 的 demo video 展示了 subsecond response 的 single-model 替代方案；开放系统落后了数月。

工程挑战：

- 每个 modality 都必须有 tokenizer，压缩要足够接近 lossless 才能重建，并且 token rate 要是 transformer 能消费的。
- 单一 vocabulary 必须为 text（32k+）、image（16k+）、speech（4k+）、music（8k+）分配空间。至少四万多个 entries。
- Training data 必须覆盖每个 input-output pair（text->image、image->speech、speech->image 等），否则模型必须自己组合。
- Inference 必须足够快地 stream output tokens，才能达到 conversational latency（<500ms time-to-first-audio-byte）。

## 概念

### 四种 modality 的四个 tokenizers

MIO 的 tokenizer stack：

- Text：标准 BPE，vocab ~32000。
- Image：SEED-Tokenizer（2023）— 带离散 codebook 的 quantized VAE，4096 entries，每张图 32x32 tokens。
- Speech：SpeechTokenizer residual-VQ（2023）— 把 16kHz waveform 编码进 8 个 hierarchical codebooks；第一层是粗 content，后续层加入 prosody 和 speaker identity。
- Music：类似 residual-VQ（Meta 的 MusicGen / Encodec family），4-8 个 codebooks。

每种 modality 都产生整数 tokens。这些 tokens 在共享 vocabulary 中获得不重叠的 ID ranges：

```
text:   0..31999
image:  32000..36095  (4096 image tokens)
speech: 36096..40191  (4096 speech base tokens, plus residual layers)
music:  40192..48383  (8192 music tokens)
sep:    48384..48390  (<image>, <speech>, <music>, </...>, etc.)
```

总计约 48k vocabulary。Input embedding 和 output projection 覆盖全部。

### Streaming decode

Speech generation 使用 residual-VQ。Transformer 预测 base（layer 0）speech tokens；parallel-decoded residual quantizer 预测后续 layers。每个 layer 0 token 大约对应 16kHz audio 的 50ms。

Streaming pattern：

1. 用户对麦克风说话；real-time audio tokenizer 每 50ms 输出 speech tokens。
2. MIO 边到达边消费 tokens（prompt prefill + incremental forward）。
3. Output tokens 作为生成结果 stream out；parallel speech decoder 以约 50-150ms latency 把它们转换为 audio samples。
4. Time-to-first-audio-byte：MIO 论文约 300-500ms，接近 GPT-4o 的约 250ms。

Mini-Omni（arXiv:2408.16725）、GLM-4-Voice（arXiv:2412.02612）和 Moshi（arXiv:2410.00037）是互补的 streaming speech-LLM designs。Moshi 尤其能在单 GPU 上达到 160ms round-trip。

### 四阶段 curriculum

MIO 的训练 curriculum：

1. Stage 1 — alignment。大规模 modality-pair corpora：text-image、text-speech、text-music。每个 pair 使用自己的 token vocabulary segment。训练 shared vocabulary。
2. Stage 2 — interleaved。Multi-modality interleaved documents（带图像 + 视频的博客、带 transcripts 的 podcasts 等）。训练 cross-modality context。
3. Stage 3 — speech-enhanced。额外 audio data，用于提升 speech quality，同时不丢失 text capability。
4. Stage 4 — SFT。跨 modality instruction tuning：VQA、captioning、narration、speech-to-speech dialogue。

缺少某个阶段会削弱特定能力：跳过 stage 2，模型失去 cross-modality context；跳过 stage 3，speech 会很差。

### Chain-of-visual-thought

MIO 引入 chain-of-visual-thought：模型把 intermediate image tokens 作为 reasoning step 发出。对于“猫是不是在爬树？”，模型：

1. 发出 `<image>` tokens，渲染场景（来自 input image 或 sketch）。
2. 发出文本分析 sketch。
3. 发出最终答案。

这个渲染出来的 intermediate image 作为 scratchpad。Spatial-reasoning tasks 上 benchmark 会提升。这个想法类似文本 reasoning 中的 chain-of-thought。

### Any-to-any 竞争者

- AnyGPT（arXiv:2402.12226）：4 种 modalities（text、image、speech、music），设计类似。
- Unified-IO 2（arXiv:2312.17172）：增加 vision action outputs、depth、normals。任务更多样，规模更小。
- NExT-GPT（arXiv:2309.05519）：LLM + modality-specific diffusion decoders。不是 single-model approach。
- CoDi（arXiv:2305.11846）：composable diffusion；通过共享 latent 实现 any-to-any。

MIO 最接近 pure-token any-to-any。AnyGPT 是它的概念祖先。

### Latency budget

对于 conversational product，每个组件的 latency 都重要：

- Mic to audio tokens：约 50ms。
- Prefill（audio tokens + history）：8B model 上约 100ms。
- First output token：约 50ms。
- Parallel residual-VQ + speech decoder：约 100-150ms。

总 time-to-first-audio-byte：最低约 300ms。GPT-4o 声称约 250ms。Moshi 声称 160ms。MIO/AnyGPT 在公开 benchmark 中约 400-600ms。

### 为什么 any-to-any 仍然难

即使到 2026 年，open any-to-any models 仍在两个轴上落后 closed models：

- Speech quality。Residual-VQ tokenizer 有损；conversational speech 相比 ElevenLabs-class voices 听起来更机械。
- Cross-modality reasoning。要求模型“唱出你看到的东西”仍然比纯视觉任务更常失败。

这些仍是开放研究问题。Qwen3-Omni（第 12.20 课）是 2025 年最先进的开放尝试。

## 使用它

`code/main.py`：

- 定义 four-modality vocabulary allocation 并打印。
- 将一组 multimodal inputs（text、image、audio-clip、music）通过 tokenizer router 路由。
- 模拟 text-to-speech response 的 streaming decode，并统计 latency。
- 给定 encoder、prefill 和 decoder latencies，计算 expected time-to-first-audio-byte。

## 交付它

本课产出 `outputs/skill-any-to-any-pipeline-auditor.md`。给定 conversational product spec（modalities in、modalities out、latency target），它会审计 MIO-family design choices，并计算 latency budget。

## 练习

1. 你的产品接受 speech input 并返回 speech output。端到端 latency budget target 是什么？列出花费时间的组件。

2. SpeechTokenizer residual-VQ 使用 8 个 codebooks。解释为什么 parallel-decoding residual levels 是必要的（相对于 sequential），它带来什么 latency savings。

3. 你的 vocabulary 有 32k text + 4k image + 4k speech。再加 8k music 和约 10 个 separators。在 hidden dim 4096 下，embedding-matrix parameter cost 是多少？

4. Chain-of-visual-thought 会输出 intermediate image。什么问题受益？哪些问题会被额外 tokens 伤害？

5. 阅读 Moshi（arXiv:2410.00037）。描述它的 “inner monologue” 技巧，并与 MIO 的 chain-of-visual-thought 对比。

## 关键术语

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| Any-to-any | “Multimodal in/out” | 单个模型接受并输出 text、image、speech、music 的任意方向 |
| Residual-VQ | “Speech tokenizer stack” | Multi-codebook tokenization，每一层加入信息；base layer 是 content，后续层是 prosody |
| SEED-Tokenizer | “Image codes” | MIO 使用的 4096-entry codebook 离散 image tokenizer |
| Chain-of-visual-thought | “Visual scratchpad” | 模型在最终答案前生成 intermediate image 作为 reasoning step |
| Time-to-first-audio-byte | “TTFAB” | 从用户语音到首个 audio output 的 latency；<500ms 才有 conversational feel |
| Four-stage curriculum | “Training recipe” | Alignment -> interleaved -> speech-enhanced -> SFT，按这个顺序 |

## 延伸阅读

- [Wang et al. — MIO (arXiv:2409.17692)](https://arxiv.org/abs/2409.17692)
- [Zhan et al. — AnyGPT (arXiv:2402.12226)](https://arxiv.org/abs/2402.12226)
- [Lu et al. — Unified-IO 2 (arXiv:2312.17172)](https://arxiv.org/abs/2312.17172)
- [Wu et al. — NExT-GPT (arXiv:2309.05519)](https://arxiv.org/abs/2309.05519)
- [Tang et al. — CoDi (arXiv:2305.11846)](https://arxiv.org/abs/2305.11846)
