# Audio Transformers — Whisper Architecture

> 音频是频率随时间变化的图像。Whisper 是一个吃 mel spectrograms 并说出文本的 ViT。

**类型：** 学习
**语言：** Python
**前置要求：** 阶段 7 · 05（Full Transformer），阶段 7 · 08（Encoder-Decoder），阶段 7 · 09（ViT）
**时间：** ~45 分钟

## 问题

Whisper（OpenAI, Radford et al. 2022）之前，state-of-the-art automatic speech recognition（ASR）意味着 wav2vec 2.0 和 HuBERT — self-supervised feature extractors 加 fine-tuned head。质量高，但数据管线昂贵，对 domain 脆弱。多语言语音识别需要为每个语言族准备单独模型。

Whisper 做了三个下注：

1. **在一切上训练。** 680,000 小时从互联网抓取的 weakly-labeled audio，覆盖 97 种语言。没有干净学术语料。没有 phoneme labels。
2. **Multi-task single model。** 一个 decoder 通过 task tokens 联合训练 transcription、translation、voice activity detection、language ID 和 timestamping。
3. **标准 encoder-decoder transformer。** Encoder 消费 log-mel spectrograms。Decoder autoregressively 生成 text tokens。没有 vocoder，没有 CTC，没有 HMM。

结果：Whisper large-v3 对口音、噪声和没有干净标注数据的语言都很鲁棒。它是 2026 年每个开源语音助手和大多数商业语音助手的默认 speech front-end。

## 概念

![Whisper pipeline: audio → mel → encoder → decoder → text](../assets/whisper.svg)

### 第 1 步 — resample + window

16 kHz 音频。Clip/pad 到 30 秒。计算 log-mel spectrogram：80 mel bins，10 ms stride → 约 3,000 frames × 80 features。这就是 Whisper 看到的“input image”。

### 第 2 步 — convolutional stem

两个 Conv1D 层，kernel 3、stride 2，把 3,000 frames 减少到 1,500。用很少参数把序列长度减半。

### 第 3 步 — encoder

一个 24 层（large 版本）transformer encoder，处理 1,500 timesteps。Sinusoidal positional encoding、self-attention、GELU FFN。产生 1,500 × 1,280 hidden states。

### 第 4 步 — decoder

一个 24 层 transformer decoder。它 autoregressively 从 BPE vocabulary 生成 tokens；这个 vocabulary 是 GPT-2 vocabulary 的超集，并加了一些音频特定 special tokens。

### 第 5 步 — task tokens

Decoder prompt 以控制 tokens 开头，告诉模型要做什么：

```
<|startoftranscript|>  <|en|>  <|transcribe|>  <|0.00|>
```

或：

```
<|startoftranscript|>  <|fr|>  <|translate|>   <|0.00|>
```

模型是在这个约定上训练的。你通过 prefix 控制任务。这是 2026 年 instruction-tuning 的语音等价物。

### 第 6 步 — output

Beam search（width 5）加 log-prob threshold。当没有 `<|notimestamps|>` token 时，timestamps 会按每 0.02 秒音频预测一次。

### Whisper sizes

| 模型 | Params | Layers | d_model | Heads | VRAM (fp16) |
|-------|--------|--------|---------|-------|-------------|
| Tiny | 39M | 4 | 384 | 6 | ~1 GB |
| Base | 74M | 6 | 512 | 8 | ~1 GB |
| Small | 244M | 12 | 768 | 12 | ~2 GB |
| Medium | 769M | 24 | 1024 | 16 | ~5 GB |
| Large | 1550M | 32 | 1280 | 20 | ~10 GB |
| Large-v3 | 1550M | 32 | 1280 | 20 | ~10 GB |
| Large-v3-turbo | 809M | 32 | 1280 | 20 | ~6 GB（4-layer decoder） |

Large-v3-turbo（2024）把 decoder 从 32 层砍到 4 层。解码快 8×，WER 回退 <1 点。这个解码速度解锁，正是 Whisper-turbo 成为 2026 年实时 voice agents 默认选项的原因。

### Whisper 不做什么

- 没有 diarization（谁在说话）。需要和 pyannote 搭配。
- 原生没有 real-time streaming — 30 秒 window 是固定的。现代 wrappers（`faster-whisper`、`WhisperX`）通过 VAD + overlap 加上 streaming。
- 没有外部 chunking 时，30 秒之外没有 long-form context。实践中仍然有效，因为人类语音转写很少需要长距离 context。

### 2026 年格局

| 任务 | 模型 | 备注 |
|------|-------|-------|
| English ASR | Whisper-turbo, Moonshine | Moonshine 在 edge 上快 4× |
| Multilingual ASR | Whisper-large-v3 | 97 种语言 |
| Streaming ASR | faster-whisper + VAD | 150 ms latency targets 可达 |
| TTS | Piper, XTTS-v2, Kokoro | Encoder-decoder pattern，但形状像 Whisper |
| Audio + language | AudioLM, SeamlessM4T | Text tokens + audio tokens 在一个 transformer 里 |

## 构建它

见 `code/main.py`。我们不训练 Whisper — 我们构建 log-mel spectrogram pipeline + task-token prompt formatter。这些才是生产中你真正会碰的部分。

### 第 1 步：合成音频

生成一个 16 kHz 采样、440 Hz 的 1 秒正弦波。16,000 个 samples。

### 第 2 步：log-mel spectrogram（简化版）

完整 mel spectrogram 需要 FFT。我们做一个简化的 framing + per-frame energy 版本，不依赖 `librosa` 也能展示管线：

```python
def frame_signal(x, frame_size=400, hop=160):
    frames = []
    for start in range(0, len(x) - frame_size + 1, hop):
        frames.append(x[start:start + frame_size])
    return frames
```

Frame = 25 ms，hop = 10 ms。匹配 Whisper 的 windowing。Per-frame energy 在教学中代替 mel bins。

### 第 3 步：pad 到 30 s

Whisper 总是处理 30 秒 chunks。把 spectrogram pad（或 clip）到 3,000 frames。

### 第 4 步：构建 prompt tokens

```python
def whisper_prompt(lang="en", task="transcribe", timestamps=True):
    tokens = ["<|startoftranscript|>", f"<|{lang}|>", f"<|{task}|>"]
    if not timestamps:
        tokens.append("<|notimestamps|>")
    return tokens
```

这就是完整的 task-control surface。4-token prefix。

## 使用它

```python
import whisper
model = whisper.load_model("large-v3-turbo")
result = model.transcribe("meeting.wav", language="en", task="transcribe")
print(result["text"])
print(result["segments"][0]["start"], result["segments"][0]["end"])
```

更快且兼容 OpenAI：

```python
from faster_whisper import WhisperModel
model = WhisperModel("large-v3-turbo", compute_type="int8_float16")
segments, info = model.transcribe("meeting.wav", vad_filter=True)
for s in segments:
    print(f"{s.start:.2f} - {s.end:.2f}: {s.text}")
```

**2026 年什么时候选 Whisper：**

- 用一个模型做 multilingual ASR。
- 鲁棒转写有噪声、多样化的音频。
- 研究 / 原型 ASR — 最快起点。

**什么时候选别的：**

- Edge 上 ultra-low latency streaming — 同等质量下 Moonshine 胜过 Whisper。
- 需要 <200 ms 的实时 conversational AI — 专用 streaming ASR。
- Speaker diarization — Whisper 不做这个；接上 pyannote。

## 交付它

见 `outputs/skill-asr-configurator.md`。这个 skill 会为新的 speech application 选择 ASR model、decoding parameters 和 preprocessing pipeline。

## 练习

1. **简单。** 运行 `code/main.py`。确认 16 kHz、10 ms hop 的 1 秒信号 frame count 约为 100。30 秒约为 3,000 frames。
2. **中等。** 用 `numpy.fft` 构建完整 log-mel spectrogram。验证 80 mel bins 在数值误差内匹配 `librosa.feature.melspectrogram(n_mels=80)`。
3. **困难。** 实现 streaming inference：把音频切成带 2 s overlap 的 10 s windows，对每个 chunk 运行 Whisper，合并 transcripts。在一个 5 分钟 podcast sample 上测量 word-error rate，并和 single-pass 比较。

## 关键词

| 术语 | 大家怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| Mel spectrogram | “音频图像” | 2D 表示：一个轴是 frequency bins，另一个轴是 time frames；每个 cell 是 log-scaled energy。 |
| Log-mel | “Whisper 看到的东西” | 经过 log 的 mel spectrogram；近似人类对响度的感知。 |
| Frame | “一个时间切片” | 25 ms 的 samples window；以 10 ms stride 重叠。 |
| Task token | “语音 prompt prefix” | Decoder prompt 中的 `<|transcribe|>` / `<|translate|>` 这类 special tokens。 |
| Voice activity detection (VAD) | “找到语音” | 在 ASR 前移除静音的 gate；极大降低成本。 |
| CTC | “Connectionist Temporal Classification” | 用于 alignment-free training 的经典 ASR loss；Whisper 不使用它。 |
| Whisper-turbo | “小 decoder，完整 encoder” | large-v3 encoder + 4-layer decoder；解码快 8×。 |
| Faster-whisper | “生产 wrapper” | CTranslate2 reimplementation；int8 quantization；比 OpenAI reference 快 4×。 |

## 延伸阅读

- [Radford et al. (2022). Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356) — Whisper 论文。
- [OpenAI Whisper repo](https://github.com/openai/whisper) — 参考代码 + 模型权重。阅读 `whisper/model.py`，可以在约 400 行中从头到尾看到 Conv1D stem + encoder + decoder。
- [OpenAI Whisper — `whisper/decoding.py`](https://github.com/openai/whisper/blob/main/whisper/decoding.py) — 第 5–6 步中的 beam-search + task-token 逻辑就在这里；500 行，完全可读。
- [Baevski et al. (2020). wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations](https://arxiv.org/abs/2006.11477) — 前身；在某些场景仍是 SOTA features。
- [SYSTRAN/faster-whisper](https://github.com/SYSTRAN/faster-whisper) — 生产 wrapper，比 reference 快 4×。
- [Jia et al. (2024). Moonshine: Speech Recognition for Live Transcription and Voice Commands](https://arxiv.org/abs/2410.15608) — 2024 年 edge-friendly ASR，形状像 Whisper 但更小。
- [HuggingFace blog — "Fine-Tune Whisper For Multilingual ASR with 🤗 Transformers"](https://huggingface.co/blog/fine-tune-whisper) — 标准 fine-tuning recipe，包括 mel spectrogram preprocessor 和 token-timestamp 处理。
- [HuggingFace `modeling_whisper.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/whisper/modeling_whisper.py) — 完整实现（encoder、decoder、cross-attention、generation），对应本课架构图。
