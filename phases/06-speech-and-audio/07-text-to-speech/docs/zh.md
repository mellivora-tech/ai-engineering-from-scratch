# Text-to-Speech（TTS）：从 Tacotron 到 F5 和 Kokoro

> ASR 把 speech 反转成 text；TTS 把 text 反转成 speech。2026 年 stack 有三部分：text → tokens，tokens → mel，mel → waveform。每部分都有能在笔记本上跑的默认模型。

**类型：** 构建
**语言：** Python
**前置要求：** 阶段 6 · 02（Spectrograms & Mel），阶段 5 · 09（Seq2Seq），阶段 7 · 05（Full Transformer）
**时间：** ~75 分钟

## 问题

你有一个字符串：“Please remind me to water the plants at 6 pm.” 你需要一个 3 秒 audio clip：听起来自然，有正确 prosody（停顿、重音），把 “plants” 的元音发对，并且在 live voice assistant 中能在 CPU 上 300 ms 内运行。你还需要换 voices，处理 code-switched input（“remind me at 6 pm, daijoubu?”），并且别在名字上出丑。

现代 TTS pipelines 如下：

1. **Text frontend。** Normalize text（dates、numbers、emails），转成 phonemes 或 subword tokens，预测 prosody features。
2. **Acoustic model。** Text → mel spectrogram。Tacotron 2（2017）、FastSpeech 2（2020）、VITS（2021）、F5-TTS（2024）、Kokoro（2024）。
3. **Vocoder。** Mel → waveform。WaveNet（2016）、WaveRNN、HiFi-GAN（2020）、BigVGAN（2022）、2024+ 的 neural codec vocoders。

2026 年，acoustic + vocoder split 在 end-to-end diffusion 和 flow-matching models 中变得模糊。但三部分 mental model 对调试仍然成立。

## 概念

![Tacotron, FastSpeech, VITS, F5/Kokoro side-by-side](../assets/tts.svg)

**Tacotron 2（2017）。** Seq2seq：char-embedding → BiLSTM encoder → location-sensitive attention → autoregressive LSTM decoder 输出 mel frames。慢（AR），长文本不稳。仍作为 baseline 被引用。

**FastSpeech 2（2020）。** Non-autoregressive。Duration predictor 输出每个 phoneme 占多少 mel frames。1-pass，比 Tacotron 快 10 倍。自然度略损（monotonic alignment），但到处交付。

**VITS（2021）。** 用 variational inference 端到端联合训练 encoder + flow-based duration + HiFi-GAN vocoder。高质量、单模型。2022-2024 主导开源 TTS。变体：YourTTS（multi-speaker zero-shot）、XTTS v2（2024，Coqui）。

**F5-TTS（2024）。** Flow matching 上的 diffusion transformer。自然 prosody，5 秒 reference audio 就能 zero-shot voice cloning。2026 open-source TTS leaderboards 顶部。335M params。

**Kokoro（2024）。** 小（82M）、CPU 可跑、实时英文 TTS 里同类最佳。Closed-vocabulary English-only，apache-2.0。

**OpenAI TTS-1-HD、ElevenLabs v2.5、Google Chirp-3。** Commercial state of the art。ElevenLabs v2.5 的 emotion tags（“[whispered]”、“[laughing]”）和 character voices 主导 2026 audiobook production。

### Vocoder evolution

| Era | Vocoder | Latency | Quality |
|-----|---------|---------|---------|
| 2016 | WaveNet | offline only | SOTA at release |
| 2018 | WaveRNN | ~realtime | good |
| 2020 | HiFi-GAN | 100× realtime | near-human |
| 2022 | BigVGAN | 50× realtime | generalizes across speakers/langs |
| 2024 | SNAC, DAC (neural codecs) | integrated with AR models | discrete tokens, bit-efficient |

到 2026 年，多数 “TTS” models 是 text-to-waveform end-to-end；mel spectrogram 是内部表示。

### Evaluation

- **MOS（Mean Opinion Score）。** 1-5 分，crowd-sourced。仍是 gold standard；非常慢。
- **CMOS（Comparative MOS）。** A-vs-B preference。同样 annotation 数下 confidence intervals 更紧。
- **UTMOS、DNSMOS。** Reference-free neural MOS predictors。用于 leaderboards。
- **CER（Character Error Rate）via ASR。** 把 TTS output 送进 Whisper，对 input text 算 CER。Intelligibility proxy。
- **SECS（Speaker Embedding Cosine Similarity）。** Voice-cloning quality。

2026 年 LibriTTS test-clean 上的数字：

| Model | UTMOS | CER (via Whisper) | Size |
|-------|-------|-------------------|------|
| Ground truth | 4.08 | 1.2% | — |
| F5-TTS | 3.95 | 2.1% | 335M |
| XTTS v2 | 3.81 | 3.5% | 470M |
| VITS | 3.62 | 3.1% | 25M |
| Kokoro v0.19 | 3.87 | 1.8% | 82M |
| Parler-TTS Large | 3.76 | 2.8% | 2.3B |

## 构建它

### 第 1 步：phonemize input

```python
from phonemizer import phonemize
ph = phonemize("Hello world", language="en-us", backend="espeak")
# 'həloʊ wɜːld'
```

Phonemes 是通用桥梁。低于 VITS-level 质量时，避免把 raw text 直接喂给模型。

### 第 2 步：运行 Kokoro（2026 CPU default）

```python
from kokoro import KPipeline
tts = KPipeline(lang_code="a")  # "a" = American English
audio, sr = tts("Please remind me to water the plants at 6 pm.", voice="af_bella")
# audio: float32 tensor, sr=24000
```

离线运行，单文件，82M params。

### 第 3 步：用 F5-TTS 做 voice cloning

```python
from f5_tts.api import F5TTS
tts = F5TTS()
wav = tts.infer(
    ref_file="my_voice_5s.wav",
    ref_text="The quick brown fox jumps over the lazy dog.",
    gen_text="Please remind me to water the plants.",
)
```

传入 5 秒 reference clip + 其 transcript；F5 会 clone prosody 和 timbre。

### 第 4 步：从零的 HiFi-GAN vocoder

太大，放不进 tutorial script，但形状如下：

```python
class HiFiGAN(nn.Module):
    def __init__(self, mel_channels=80, upsample_rates=[8, 8, 2, 2]):
        super().__init__()
        # 4 upsample blocks, total 256x to go from mel-rate to audio-rate
        ...
    def forward(self, mel):
        return self.blocks(mel)  # -> waveform
```

训练：adversarial（短窗口 discriminator）+ mel-spectrogram reconstruction loss + feature-matching loss。已经 commoditized：使用 `hifi-gan` repo 或 nvidia-NeMo 的 pretrained checkpoints。

### 第 5 步：完整 pipeline（pseudocode）

```python
text = "Please remind me at 6 pm."
phones = phonemize(text)
mel = acoustic_model(phones, speaker=alice)      # [T, 80]
wav = vocoder(mel)                                # [T * 256]
soundfile.write("out.wav", wav, 24000)
```

## 使用它

2026 年技术栈：

| 场景 | 选择 |
|-----------|------|
| Real-time English voice assistant | Kokoro（CPU）或 XTTS v2（GPU） |
| 5 s reference voice cloning | F5-TTS |
| Commercial character voices | ElevenLabs v2.5 |
| Audiobook narration | ElevenLabs v2.5 或 XTTS v2 + fine-tune |
| Low-resource language | 在 5-20 h target-lang data 上训练 VITS |
| Expressive / emotion tags | ElevenLabs v2.5 或 StyleTTS 2 fine-tune |

截至 2026，开源 leader：**F5-TTS 代表质量，Kokoro 代表效率**。除非你是历史学家，不要再用 Tacotron。

## 坑

- **没有 text normalizer。** “Dr. Smith” 读作 “Doctor” 还是 “Drive”？“2026” 读作 “twenty twenty six” 还是 “two zero two six”？先 normalize，再 phonemizer。
- **OOV proper nouns。** “Ghumare” → “ghyu-mair”？为 unknown tokens 交付 fallback grapheme-to-phoneme model。
- **Clipping。** Vocoder output 很少 clip，但 inference 时 mel scaling mismatch 可能超过 ±1.0。始终 `np.clip(wav, -1, 1)`。
- **Sample-rate mismatch。** Kokoro 输出 24 kHz；下游 pipeline 期望 16 kHz。要 resample，否则 aliasing。

## 交付它

保存为 `outputs/skill-tts-designer.md`。为给定 voice、latency 和 language target 设计 TTS pipeline。

## 练习

1. **简单。** 运行 `code/main.py`。从 toy vocab 构建 phoneme dictionary，估计每个 phoneme duration，并打印一个假的 “mel” schedule。
2. **中等。** 安装 Kokoro，用 voice `af_bella` 和 `am_adam` 合成同一句话。比较 audio durations 和主观质量。
3. **困难。** 录一段 5 秒自己的 reference clip。用 F5-TTS clone。报告 reference 和 cloned output 之间的 SECS。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|-----------------|-----------------------|
| Phoneme | 声音单位 | 抽象 sound class；英语中 39 个（ARPABet）。 |
| Duration predictor | 每个 phoneme 持续多久 | Non-AR model output；每个 phoneme 的 integer frames。 |
| Vocoder | Mel → waveform | 把 mel-spec 映射到 raw samples 的 neural net。 |
| HiFi-GAN | 标准 vocoder | GAN-based；2020-2024 主导。 |
| MOS | 主观质量 | 人类评分的 1-5 mean opinion score。 |
| SECS | Voice-clone metric | Target 和 output speaker embeddings 的 cosine similarity。 |
| F5-TTS | 2024 open-source SOTA | Flow-matching diffusion；zero-shot cloning。 |
| Kokoro | CPU English leader | 82M-param model，Apache 2.0。 |

## 延伸阅读

- [Shen et al. (2017). Tacotron 2](https://arxiv.org/abs/1712.05884) — seq2seq baseline。
- [Kim, Kong, Son (2021). VITS](https://arxiv.org/abs/2106.06103) — end-to-end flow-based。
- [Chen et al. (2024). F5-TTS](https://arxiv.org/abs/2410.06885) — 当前 open-source SOTA。
- [Kong, Kim, Bae (2020). HiFi-GAN](https://arxiv.org/abs/2010.05646) — 2026 年仍在交付的 vocoder。
- [Kokoro-82M on HuggingFace](https://huggingface.co/hexgrad/Kokoro-82M) — 2024 CPU-friendly English TTS。
