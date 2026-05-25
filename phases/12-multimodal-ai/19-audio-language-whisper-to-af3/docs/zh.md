# Audio-Language Models：从 Whisper 到 Audio Flamingo 3

> Whisper（Radford 等人，2022 年 12 月）解决了 speech recognition：680k 小时 weakly-supervised multilingual speech、简单 encoder-decoder transformer，以及让后续每个 ASR release 都引用它的 benchmark。但 recognition 不是 reasoning。问“这段录音里有哪些乐器”“说话者表达了什么情绪”“第 3 分钟发生了什么”，需要 audio understanding，而不是 transcription。Qwen-Audio、SALMONN、LTU 和 NVIDIA 的 Audio Flamingo 3（AF3，2025 年 7 月）逐步搭建了这套 stack：保留 Whisper-class encoders，接上 Q-formers，在 audio-text instruction data 上训练，加入 chain-of-thought reasoning。本课走完整条弧线。

**类型：** 构建
**语言：** Python（stdlib，log-Mel spectrogram + audio Q-former skeleton）
**前置要求：** 阶段 6（Speech and Audio），阶段 12 · 03（Q-Former）
**时间：** ~180 分钟

## 学习目标

- 从 waveform 计算 log-Mel spectrogram：windowing、FFT、filter banks、log transform。
- 比较 encoder options：Whisper encoder、BEATs、AF-Whisper hybrid。什么时候各自胜出。
- 构建 audio Q-former：N 个 learnable queries 对 spectrogram patches 做 cross-attend。
- 解释 cascaded（Whisper-then-LLM）vs end-to-end audio-LLM training：为什么 end-to-end 在 reasoning 上更可扩展。

## 问题

Speech recognition 已由 Whisper 解决。Audio 的 OCR 已经商品化。但“商品化”停在 transcription。如果模型不能对它听到的内容推理，比如 timing、speakers、emotion、music structure、environmental sounds，那么 transcription alone 无法驱动产品功能。

三条显然路线：

1. Cascade：Whisper transcribes，LLM 对 transcript 推理。适合 pure-speech scenarios。对 music、environmental audio、multi-speaker overlap、emotion 失败。

2. End-to-end audio-LLM：audio encoder 直接把 audio tokens 喂进 LLM，跳过 transcription。保留 acoustic information（emotion、speaker、environment）。需要新训练数据。

3. Hybrid：audio encoder + text decoder，既能 transcribe 又能 reason。Qwen-Audio 和 Audio Flamingo 选择这条。

## 概念

### Log-Mel spectrogram：输入特征

每个 audio encoder 都从同一种 feature 开始：log-Mel spectrogram。

1. Resample 到 16 kHz。
2. 用 25ms windows、10ms hop 做 short-time Fourier transform。
3. 取 FFT result 的 magnitude。
4. 应用 Mel filter banks（通常 80 个 filters，0-8000 Hz log-spaced），warp 到 perceptual frequency。
5. Log compress（log(1 + x)）以处理 dynamic range。

结果是形状为（T, 80）的 2D array，其中 T 是 time frames 数。30 秒 clip、100 Hz frame rate 下是（3000, 80）。

### Whisper encoder

Whisper encoder 是一个 12-layer ViT-style transformer，把 log-Mel spectrogram 当作 time frames 序列处理。输出：每个 time frame 一个 hidden-state vector。

对于 ASR，Whisper decoder 是一个 cross-attention transformer，以 encoder output 为条件生成 text tokens。标准 encoder-decoder。

对于 ALMs（audio-LLMs），你希望把 encoder output 作为另一个 LLM 的输入。模式是：Whisper encoder frozen，Q-former trainable，LLM frozen 或 tuned。

### BEATs 与 audio-specific encoders

Whisper 在 speech-dominant data 上训练。它在 music 和 environmental audio 上较弱。

BEATs（Chen 等人，2022）是在 AudioSet 上训练的 self-supervised transformer。同参数量下，它比 Whisper 更好地捕捉 music 和 environmental sounds。

AF-Whisper（Audio Flamingo 3 的 hybrid）：把 Whisper + BEATs features concat 作为 audio input。Whisper 携带 linguistic signal，BEATs 携带 acoustic signal。

### Audio Q-former

模式与 BLIP-2 的 visual Q-former 相同。固定数量 learnable queries（通常 32 或 64）对 audio encoder 的 output frames 做 cross-attend。Queries 变成 LLM 消费的 audio tokens。

Training alignment stage：只训练 Q-former，在 audio-text pairs（AudioCaps、Clotho）上用 contrastive + captioning losses。Instruction stage：end-to-end，unfreeze LLM，在 instruction data 上训练。

### 这条弧线：SALMONN、Qwen-Audio、AF3

SALMONN（Tang 等人，2023）：Whisper + BEATs + Q-former + LLaMA。第一个有认真 reasoning ability 的 open audio-LLM。MMAU benchmark composite 约 0.55。

Qwen-Audio（Chu 等人，2023）：类似架构，在更丰富数据集上训练，并针对 multi-turn dialogue 调优。MMAU 约 0.60。

LTU — Listen, Think, Understand（Gong 等人，2023）：显式 reasoning data，关注 audio clips 上的 chain-of-thought。更小但更聚焦。

Audio Flamingo 3（Goel 等人，2025 年 7 月）：当前 open SOTA。8B LLM backbone（Qwen2 7B）、Whisper-large encoder concat BEATs、64-query Q-former，在 1M+ audio-text instruction pairs 上训练。MMAU 0.72，在部分子任务上匹配 proprietary frontier。

AF3 还引入 audio 的 on-demand chain-of-thought：模型可选择在最终答案前输出 thinking tokens（“let me identify the instruments first: ...”）。启用 thinking 时，复杂 reasoning tasks accuracy 提升 3-5 分。

### Cascaded vs end-to-end

Cascaded pipeline：

1. Whisper 把 audio 转写成文本。
2. LLM 对文本推理。

对“summarize this podcast” 完美有效。对以下情况失败：
- “What's the mood of this song?” — mood 在声音里，不在文字里。
- “Who is speaking, Alice or Bob?” — 需要 speaker identification。
- “At what second does the explosion happen?” — text 中丢失 temporal grounding。
- “Is this real or generated audio?” — deepfake detection 需要 acoustic features。

End-to-end 保留 acoustic signal。Qwen-Audio 和 AF3 原生处理 music、environment 和 emotion。

### 2026 production recipe

对于新的 audio-understanding product：

- 如果目标是 transcription、没有 music、没有 emotion inference，使用 cascaded。
- 如果涉及 music、emotion、multi-speaker 或 complex audio reasoning，使用 AF3 / Qwen-Audio-family。

Cascaded 更便宜、更简单。End-to-end 更强。

### MMAU：audio reasoning benchmark

MMAU（Massive Multimodal Audio Understanding）是 2024-2025 年 audio reasoning benchmark：

- 10,000 audio-text QA pairs，覆盖 speech、music、environmental sounds。
- 覆盖 classification、temporal reasoning、causal reasoning、open-ended QA。
- 测试 cascaded pipelines 系统性错过的能力。

Open SOTA（AF3）为 0.72；proprietary frontier 约 0.78（Gemini 2.5 Pro、Claude Opus 4.7）。差距小于 VideoMME 的 open-vs-closed delta，说明 audio-LLMs 正在成熟。

## 使用它

`code/main.py`：

- 用 stdlib 实现 log-Mel spectrogram computation：windowing、naive DFT、Mel filter-bank。
- Audio Q-former skeleton：给定 encoder output frames，计算 Q、K、V、attention，并输出 N tokens。
- 在 toy task 上比较 cascaded-vs-end-to-end。

## 交付它

本课产出 `outputs/skill-audio-llm-pipeline-picker.md`。给定 audio task（transcription、music tagging、emotion inference、multi-speaker diarization、environment classification），它会在 cascaded、end-to-end AF3 或 hybrid 中选择。

## 练习

1. 对 16kHz、25ms window、10ms hop、80 Mel bins 的 30 秒 clip，计算 log-Mel spectrogram dimension。48kHz 下会如何变化？

2. 为什么 Whisper 在 music 上表现较弱？BEATs 捕捉了 Whisper 没捕捉到的哪些 audio features？

3. Audio Q-former 使用 64 queries vs 32：什么 task complexity 下 64 值得？32 为哪些任务省 compute？

4. 阅读 AF3 第 4 节关于 on-demand thinking 的内容。提出三个 chain-of-thought 最有帮助的 audio tasks。

5. 使用 AF3 输出实现一个 minimal diarization pipeline。你如何标记 speaker changes？

## 关键术语

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| Log-Mel spectrogram | “Mel features” | Mel filter banks 后的 log-magnitude values 组成的 2D（time, frequency）数组 |
| Audio Q-former | “Audio Perceiver” | 从 audio encoder output 到固定长度 queries 的 cross-attention bottleneck，喂给 LLM |
| Cascaded | “ASR-then-LLM” | Whisper 转写，text LLM 推理的 pipeline；会丢失 acoustic information |
| End-to-end | “Audio-LLM” | Audio features 通过 Q-former 直接进入 LLM；保留 acoustic signal |
| BEATs | “Audio AudioSet encoder” | 在 AudioSet 上训练的 SSL transformer；擅长 music + environmental sounds |
| MMAU | “Audio reasoning bench” | 覆盖 speech、music、environment 的 10k QA pairs；2024 eval standard |
| On-demand thinking | “Audio CoT” | 模型可选择在最终答案前输出 reasoning tokens，提升 accuracy 3-5 分 |

## 延伸阅读

- [Radford et al. — Whisper (arXiv:2212.04356)](https://arxiv.org/abs/2212.04356)
- [Chu et al. — Qwen-Audio (arXiv:2311.07919)](https://arxiv.org/abs/2311.07919)
- [Goel et al. — Audio Flamingo 3 (arXiv:2507.08128)](https://arxiv.org/abs/2507.08128)
- [Tang et al. — SALMONN (arXiv:2310.13289)](https://arxiv.org/abs/2310.13289)
- [Gong et al. — LTU (arXiv:2305.10790)](https://arxiv.org/abs/2305.10790)
