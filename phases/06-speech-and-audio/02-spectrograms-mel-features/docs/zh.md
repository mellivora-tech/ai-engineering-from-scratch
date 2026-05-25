# Spectrograms、Mel Scale 与 Audio Features

> Neural nets 不擅长直接消耗 raw waveforms。它们消耗 spectrograms。Mel spectrograms 更好。2026 年每个 ASR、TTS 和 audio classifier 的成败，都取决于这个 preprocessing 选择。

**类型：** 构建
**语言：** Python
**前置要求：** 阶段 6 · 01（Audio Fundamentals）
**时间：** ~45 分钟

## 问题

拿一个 10 秒、16 kHz 的 clip。它是 160,000 个 floats，全在 `[-1, 1]` 中，和 label “dog barking” 或 “the word cat” 几乎没有直接相关。Raw waveform 包含信息，但形式很难让模型抽取。相同 phoneme 间隔 100 ms 被说出来时，raw samples 完全不同。

Spectrogram 修复了这个问题。它压缩人类感知不关心的时间细节（microsecond jitter），保留感知会关注的结构（哪些频率有能量，以及这些能量在 ~10-25 ms windows 中如何变化）。

Mel spectrograms 进一步推进。人类对 pitch 的感知近似 logarithmic：100 Hz vs 200 Hz 与 1000 Hz vs 2000 Hz 听起来“距离相同”。Mel scale 会 warp frequency axis 以匹配这种感知。2010 到 2026 年，mel-scaled spectrogram 是 speech ML 最重要的单一 feature。

## 概念

![Waveform to STFT to mel spectrogram to MFCC ladder](../assets/mel-features.svg)

**STFT（Short-Time Fourier Transform）。** 把 waveform 切成重叠 frames（典型：25 ms window、10 ms hop = 16 kHz 下 400 samples / 160 samples）。每个 frame 乘 window function（默认 Hann；Hamming tradeoff 略不同）。对每个 frame 做 FFT。把 magnitude spectra 堆成 `(n_frames, n_freq_bins)` 矩阵。这就是 spectrogram。

**Log-magnitude。** Raw magnitudes 跨 5-6 个数量级。取 `log(|X| + 1e-6)` 或 `20 * log10(|X|)` 来压缩 dynamic range。每个生产 pipeline 都用 log-magnitude，不用 raw magnitude。

**Mel scale。** Hz 频率 `f` 映射到 mel `m`：`m = 2595 * log10(1 + f / 700)`。该映射在 1 kHz 以下近似 linear，在以上近似 logarithmic。覆盖 0-8 kHz 的 80 mel bins 是 ASR 标准输入。

**Mel filterbank。** 一组在 mel scale 上等距的 triangular filters。每个 filter 是相邻 FFT bins 的加权和。把 STFT magnitude 乘 filterbank matrix，一次 matmul 就得到 mel spectrogram。

**Log-mel spectrogram。** `log(mel_spec + 1e-10)`。Whisper 的输入。Parakeet 的输入。SeamlessM4T 的输入。2026 年通用 audio frontend。

**MFCCs。** 对 log-mel spectrogram 应用 DCT（type II），保留前 13 个 coefficients。它会 decorrelate features 并进一步压缩。直到约 2015 年，MFCC 都是主导 feature；之后 raw log-mels 上的 CNNs/Transformers 赶上。Speaker recognition（x-vectors、ECAPA）中仍然使用。

**Resolution trade。** 更大的 FFT = 更好的 frequency resolution，但更差的 time resolution。25 ms / 10 ms 是 audio-ML 默认；music 用 50 ms / 12.5 ms；transient detection（drum hits、plosives）用 5 ms / 2 ms。

## 构建它

### 第 1 步：frame waveform

```python
def frame(signal, frame_len, hop):
    n = 1 + (len(signal) - frame_len) // hop
    return [signal[i * hop : i * hop + frame_len] for i in range(n)]
```

10 秒、16 kHz clip 在 `frame_len=400, hop=160` 下产生 998 frames。

### 第 2 步：Hann window

```python
import math

def hann(N):
    return [0.5 * (1 - math.cos(2 * math.pi * n / (N - 1))) for n in range(N)]
```

FFT 前逐元素相乘。它能移除非零 endpoints 截断导致的 spectral leakage。

### 第 3 步：STFT magnitude

```python
def stft_magnitude(signal, frame_len=400, hop=160):
    win = hann(frame_len)
    frames = frame(signal, frame_len, hop)
    return [magnitudes(dft([w * s for w, s in zip(win, f)])) for f in frames]
```

生产用 `torch.stft` 或 `librosa.stft`（FFT-backed、vectorized）。这里的 loop 是教学用；`code/main.py` 中只跑短 clips。

### 第 4 步：mel filterbank

```python
def hz_to_mel(f):
    return 2595.0 * math.log10(1.0 + f / 700.0)

def mel_to_hz(m):
    return 700.0 * (10 ** (m / 2595.0) - 1)

def mel_filterbank(n_mels, n_fft, sr, fmin=0, fmax=None):
    fmax = fmax or sr / 2
    mels = [hz_to_mel(fmin) + (hz_to_mel(fmax) - hz_to_mel(fmin)) * i / (n_mels + 1)
            for i in range(n_mels + 2)]
    hzs = [mel_to_hz(m) for m in mels]
    bins = [int(h * n_fft / sr) for h in hzs]
    fb = [[0.0] * (n_fft // 2 + 1) for _ in range(n_mels)]
    for m in range(n_mels):
        for k in range(bins[m], bins[m + 1]):
            fb[m][k] = (k - bins[m]) / max(1, bins[m + 1] - bins[m])
        for k in range(bins[m + 1], bins[m + 2]):
            fb[m][k] = (bins[m + 2] - k) / max(1, bins[m + 2] - bins[m + 1])
    return fb
```

`n_fft=400` 时，覆盖 0-8 kHz 的 80 mels 会给出 `(80, 201)` matrix。把 `(n_frames, 201)` STFT magnitude 乘其转置，得到 `(n_frames, 80)` mel spectrogram。

### 第 5 步：log-mel

```python
def log_mel(mel_spec, eps=1e-10):
    return [[math.log(max(v, eps)) for v in frame] for frame in mel_spec]
```

常见替代：`librosa.power_to_db`（reference-normalized dB）、`10 * log10(power + eps)`。Whisper 使用更复杂的 clip + normalize routine（见 Whisper 的 `log_mel_spectrogram`）。

### 第 6 步：MFCCs

```python
def dct_ii(x, n_coeffs):
    N = len(x)
    return [
        sum(x[n] * math.cos(math.pi * k * (2 * n + 1) / (2 * N)) for n in range(N))
        for k in range(n_coeffs)
    ]
```

对每个 log-mel frame 应用 DCT，保留前 13 个 coefficients。这就是 MFCC matrix。第一个 coefficient 通常丢弃（它编码 overall energy）。

## 使用它

2026 年技术栈：

| 任务 | Features |
|------|----------|
| ASR（Whisper、Parakeet、SeamlessM4T） | 80 log-mels，10 ms hop，25 ms window |
| TTS acoustic model（VITS、F5-TTS、Kokoro） | 80 mels，5-12 ms hop 用于精细 temporal control |
| Audio classification（AST、PANNs、BEATs） | 128 log-mels，10 ms hop |
| Speaker embedding（ECAPA-TDNN、WavLM） | 80 log-mels 或 raw-waveform SSL |
| Music（MusicGen、Stable Audio 2） | EnCodec discrete tokens（不是 mels） |
| Keyword spotting | 小设备用 40 MFCCs |

经验法则：**如果不是做 music，从 80 log-mels 开始。** 任何偏离都需要证据。

## 2026 年仍会交付的坑

- **Mel count mismatch。** 训练用 80 mels，推理用 128 mels。静默失败。训练和推理都记录 feature shape。
- **上游 sample-rate mismatch。** 22.05 kHz 上算出的 mels 与 16 kHz 不同。先修 SR，再 featurization。
- **dB vs log。** Whisper 期望 log-mel，不是 dB-mel。一些 HF pipelines 会 autodetect；你的 custom code 不会。
- **Normalization drift。** 训练时 per-utterance normalization，推理时 global normalization。会让 WER 翻倍的生产 bug。
- **Padding leakage。** Clip 尾部 zero-padding 会在 trailing frames 产生 flat spectrum。Symmetric pad 或 replicate。

## 交付它

保存为 `outputs/skill-feature-extractor.md`。这个 skill 会为给定模型目标选择 feature type、mel count、frame/hop 和 normalization。

## 练习

1. **简单。** 运行 `code/main.py`。它合成一个 chirp（频率从 200 → 4000 Hz 扫过），并打印每帧的 argmax mel bin。可选画图，确认它匹配 sweep。
2. **中等。** 用 `n_mels` ∈ `{40, 80, 128}` 和 `frame_len` ∈ `{200, 400, 800}` 重新运行。测量 time axis 上 sharp-peak bandwidth。哪种组合最能分辨 chirp？
3. **困难。** 实现 `power_to_db`，并比较 tiny CNN classifier 在 AudioMNIST 上使用 (a) raw log-mel、(b) dB-mel with `ref=max`、(c) MFCC-13 + delta + delta-delta 的 ASR accuracy。报告 top-1 accuracy。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|-----------------|-----------------------|
| Frame | 一个切片 | 送入一次 FFT 的 25 ms waveform chunk。 |
| Hop | 步长 | 相邻 frames 间 samples；10 ms 是 ASR 默认。 |
| Window | Hann/Hamming 那个 | 逐点 multiplier，把 frame 边缘 taper 到零。 |
| STFT | Spectrogram generator | Framed + windowed FFT；产生 time × frequency matrix。 |
| Mel | Warped frequency | 对数感知尺度；`m = 2595·log10(1 + f/700)`。 |
| Filterbank | 那个矩阵 | 三角 filters，将 STFT 投影到 mel bins。 |
| Log-mel | Whisper 的输入 | `log(mel_spec + eps)`；2026 年标准化。 |
| MFCC | Old-school feature | Log-mel 的 DCT；13 coeffs，decorrelated。 |

## 延伸阅读

- [Davis, Mermelstein (1980). Comparison of parametric representations for monosyllabic word recognition](https://ieeexplore.ieee.org/document/1163420) — MFCC 论文。
- [Stevens, Volkmann, Newman (1937). A Scale for the Measurement of the Psychological Magnitude Pitch](https://pubs.aip.org/asa/jasa/article-abstract/8/3/185/735757/) — 原始 mel scale。
- [OpenAI — Whisper source, log_mel_spectrogram](https://github.com/openai/whisper/blob/main/whisper/audio.py) — 阅读参考实现。
- [librosa feature extraction docs](https://librosa.org/doc/main/feature.html) — `mfcc`、`melspectrogram`、hop/window 的参考。
- [NVIDIA NeMo — audio preprocessing](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/asr/asr_all.html#featurizers) — Parakeet + Canary 的 production-scale pipeline。
