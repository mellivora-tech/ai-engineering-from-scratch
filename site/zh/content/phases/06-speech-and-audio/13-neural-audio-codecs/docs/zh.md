# Neural Audio Codecs：EnCodec、SNAC、Mimi、DAC 与 Semantic-Acoustic Split

> 2026 年 audio generation 几乎全是 tokens。EnCodec、SNAC、Mimi、DAC 把连续 waveforms 变成 transformer 可预测的离散序列。Semantic-vs-acoustic token split：第一 codebook 语义，其余声学，是 Transformer 以来音频架构最重要的转变。

**类型：** 学习
**语言：** Python
**前置要求：** 阶段 6 · 02（Spectrograms），阶段 10 · 11（Quantization），阶段 5 · 19（Subword Tokenization）
**时间：** ~60 分钟

## 问题

Language models 在离散 tokens 上工作。Audio 是连续的。如果你想要一个 speech / music 的 LLM-style model：MusicGen、Moshi、Sesame CSM、VibeVoice、Orpheus，你首先需要 **neural audio codec**：一个把 audio 离散化成小 vocabulary tokens 的 learned encoder，以及一个重建 waveform 的 matching decoder。

出现了两类：

1. **Reconstruction-first codecs**：EnCodec、DAC。优化 perceptual audio quality。Tokens 是 “acoustic”，捕捉 speaker identity、timbre、background noise 等所有内容。
2. **Semantic-first codecs**：Mimi（Kyutai）、SpeechTokenizer。强制第一 codebook 编码 linguistic / phonetic content（通常从 WavLM distill）。后续 codebooks 是 acoustic detail。

2024-2026 的洞见：**用纯 reconstruction codec 从 text 生成会得到模糊 speech。** Codec-token LLM 必须在同一 codebook 中同时学 language structure 和 acoustic structure，扩展性不好。把两者分离：semantic codebook 0，acoustic codebooks 1-N，正是 Moshi 和 Sesame CSM 能工作的原因。

## 概念

![Four codec landscape: EnCodec, DAC, SNAC (multi-scale), Mimi (semantic+acoustic)](../assets/codec-comparison.svg)

### 核心技巧：Residual Vector Quantization（RVQ）

不是用一个巨大 codebook（高质量需要数百万 codes），现代 audio codecs 都用 **RVQ**：一串小 codebooks。第一个 codebook quantize encoder output；第二个 quantize residual；依此类推。每个 codebook 有 1024 codes。8 个 codebooks 的有效 vocabulary 是 1024^8 = 10^24。

推理时，decoder 对每帧选中的所有 codes 求和来重建。

### 2026 年重要的四个 codecs

**EnCodec（Meta, 2022）。** Baseline。Waveform 上的 encoder-decoder，RVQ bottleneck。24 kHz，可用 32 codebooks，默认 4 codebooks @ 1.5 kbps。架构使用 `1D conv + transformer + 1D conv`。MusicGen 使用。

**DAC（Descript, 2023）。** RVQ + L2-normalized codebooks、periodic activation functions、改进 losses。开放 codec 中 reconstruction fidelity 最高，12 codebooks 时 speech 有时难以和原始区分。44.1 kHz full-band。

**SNAC（Hubert Siuzdak, 2024）。** Multi-scale RVQ：coarse codebooks 在低帧率上工作，fine codebooks 在高帧率上工作。相当于层级化建模 audio：约 12 Hz 的 coarse “sketch” 加 50 Hz detail。Orpheus-3B 使用它，因为 hierarchical structure 很适合 LM-based generation。

**Mimi（Kyutai, 2024）。** 2026 年 game-changer。12.5 Hz frame rate（极低），8 codebooks @ 4.4 kbps。Codebook 0 **从 WavLM distill**：训练来预测 WavLM 的 speech-content features。Codebooks 1-7 是 acoustic residuals。这个 split 驱动 Moshi（第 15 课）和 Sesame CSM。

### Frame rates 对 language modeling 很重要

更低 frame rate = 更短 sequence = 更快 LM。

| Codec | Frame rate | 1 s = N frames | 适合 |
|-------|-----------|----------------|---------|
| EnCodec-24k | 75 Hz | 75 | music, general audio |
| DAC-44.1k | 86 Hz | 86 | high-fidelity music |
| SNAC-24k (coarse) | ~12 Hz | 12 | AR-LM efficient |
| Mimi | 12.5 Hz | 12.5 | streaming speech |

12.5 Hz 下，10 秒 utterance 只有 125 codec frames，transformer 很容易预测。

### Semantic vs acoustic tokens

```
frame_t → [semantic_token_t, acoustic_token_0_t, acoustic_token_1_t, ..., acoustic_token_6_t]
```

- **Semantic token（Mimi 的 codebook 0）。** 编码说了什么：phonemes、words、content。通过辅助 prediction loss 从 WavLM distill。
- **Acoustic tokens（codebooks 1-7）。** 编码 timbre、speaker identity、prosody、background noise、fine detail。

AR LM 先预测 semantic token（基于 text 条件），再预测 acoustic tokens（基于 semantic + speaker reference 条件）。这个 factorization 是现代 TTS 能 zero-shot clone voices 的原因：semantic model 管 content；acoustic model 管 timbre。

### 2026 reconstruction quality（bits per sec，bitrate 越低越好）

| Codec | Bitrate | PESQ | ViSQOL |
|-------|---------|------|--------|
| Opus-20kbps | 20 kbps | 4.0 | 4.3 |
| EnCodec-6kbps | 6 kbps | 3.2 | 3.8 |
| DAC-6kbps | 6 kbps | 3.5 | 4.0 |
| SNAC-3kbps | 3 kbps | 3.3 | 3.8 |
| Mimi-4.4kbps | 4.4 kbps | 3.1 | 3.7 |

传统 codecs 如 Opus 在每 bit perceptual quality 上仍然胜出。Neural codecs 的优势是 **discrete tokens**（Opus 不产生）和 **generative-model quality**（LM 能用这些 tokens 做什么）。

## 构建它

### 第 1 步：用 EnCodec encode

```python
from encodec import EncodecModel
import torch

model = EncodecModel.encodec_model_24khz()
model.set_target_bandwidth(6.0)  # kbps

wav = torch.randn(1, 1, 24000)
with torch.no_grad():
    encoded = model.encode(wav)
codes, scale = encoded[0]
# codes: (1, n_codebooks, n_frames), dtype=int64
```

6 kbps 下 `n_codebooks=8`。每个 code 是 0-1023（10-bit）。

### 第 2 步：decode 并测 reconstruction

```python
with torch.no_grad():
    wav_recon = model.decode([(codes, scale)])

from torchaudio.functional import compute_deltas
import torch.nn.functional as F

mse = F.mse_loss(wav_recon[:, :, :wav.shape[-1]], wav).item()
```

### 第 3 步：semantic-acoustic split（Mimi-style）

```python
from moshi.models import loaders
mimi = loaders.get_mimi()

with torch.no_grad():
    codes = mimi.encode(wav)  # shape (1, 8, frames@12.5Hz)

semantic = codes[:, 0]
acoustic = codes[:, 1:]
```

Semantic codebook 0 是 WavLM-aligned。你可以训练 text-to-semantic transformer，vocabulary 远小于 direct-to-audio。然后 separate acoustic-to-waveform decoder 根据 speaker reference 条件化。

### 第 4 步：为什么 codec tokens 上的 AR LM 有效

对 10 秒 speech clip，Mimi 12.5 Hz × 8 codebooks：

```
N_tokens = 10 * 12.5 * 8 = 1000 tokens
```

1000 tokens 对 transformer 是很小的 context。一个 256M-parameter transformer 在现代 GPU 上能用毫秒级生成 10 秒 speech。

## 使用它

Problem → codec：

| Task | Codec |
|------|-------|
| General music generation | EnCodec-24k |
| Highest-fidelity reconstruction | DAC-44.1k |
| AR LM over speech（TTS） | SNAC 或 Mimi |
| Streaming full-duplex speech | Mimi（12.5 Hz） |
| Sound-effect library with text | EnCodec + T5 condition |
| Fine-grained audio editing | DAC + inpainting |

经验法则：**如果在构建 generative model，从 Mimi 或 SNAC 开始。如果在构建 compression pipeline，用 Opus。**

## 坑

- **Too many codebooks。** 加 codebooks 会线性提高 fidelity，也会线性增加 LM sequence length。停在 8-12。
- **Frame-rate mismatch。** 在 12.5 Hz Mimi 上训练 LM，再 fine-tune 到 50 Hz EnCodec，会静默失败。
- **假设所有 codebooks 等价。** 在 Mimi 中，codebook 0 携带 content；丢掉它会毁掉 intelligibility。丢掉 codebook 7 几乎听不出。
- **只看 reconstruction quality。** 一个 codec reconstruction 很好，但如果 semantic structure 差，可能对 LM-based generation 没用。

## 交付它

保存为 `outputs/skill-codec-picker.md`。为给定 generative 或 compression task 选择 codec。

## 练习

1. **简单。** 运行 `code/main.py`。它实现 toy scalar + residual quantizer，并测量随着 codebooks 增加 reconstruction error 如何变化。
2. **中等。** 安装 `encodec`，在 held-out speech clip 上比较 1、4、8、32 codebooks。画 PESQ 或 MSE vs bitrate。
3. **困难。** 加载 Mimi。Encode 一个 clip。把 codebook 0 替换成随机整数并 decode；再同样替换 codebook 7。比较两个 corruption：codebook 0 应该毁掉 intelligibility，codebook 7 几乎不变。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|-----------------|-----------------------|
| RVQ | Residual quantization | 小 codebooks 级联；每个量化前一残差。 |
| Frame rate | Codec speed | 每秒 token-frames 数。越低 LM 越快。 |
| Semantic codebook | Codebook 0（Mimi） | 从 SSL features distill 的 codebook；编码 content。 |
| Acoustic codebooks | 其余全部 | Timbre、prosody、noise、fine detail。 |
| PESQ / ViSQOL | Perceptual quality | 与 MOS 相关的 objective metrics。 |
| EnCodec | Meta codec | RVQ baseline；MusicGen 使用。 |
| Mimi | Kyutai codec | 12.5 Hz frame rate；semantic-acoustic split；驱动 Moshi。 |

## 延伸阅读

- [Défossez et al. (2023). EnCodec](https://arxiv.org/abs/2210.13438) — RVQ baseline。
- [Kumar et al. (2023). Descript Audio Codec (DAC)](https://arxiv.org/abs/2306.06546) — 最高保真开放 codec。
- [Siuzdak (2024). SNAC](https://arxiv.org/abs/2410.14411) — multi-scale RVQ。
- [Kyutai (2024). Mimi codec](https://kyutai.org/codec-explainer) — semantic-acoustic split、WavLM distillation。
- [Borsos et al. (2023). AudioLM](https://arxiv.org/abs/2209.03143) — two-stage semantic/acoustic paradigm。
- [Zeghidour et al. (2021). SoundStream](https://arxiv.org/abs/2107.03312) — 原始 streamable RVQ codec。
