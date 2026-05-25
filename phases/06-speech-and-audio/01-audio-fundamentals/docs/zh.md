# 音频基础：Waveforms、Sampling、Fourier Transform

> Waveforms 是原始信号。Spectrograms 是表示。Mel features 是适合 ML 的形式。每个现代 ASR 和 TTS pipeline 都会走这条梯子，而第一阶是理解 sampling 和 Fourier。

**类型：** 学习
**语言：** Python
**前置要求：** 阶段 1 · 06（向量与矩阵），阶段 1 · 14（概率分布）
**时间：** ~45 分钟

## 问题

麦克风产生 pressure-vs-time signal。你的 neural net 消耗 tensors。两者之间有一套 conventions，一旦违反，就会产生静默 bug：模型训练看起来正常，但 WER 翻倍；TTS 交付后带 hiss；voice cloning system 记住了麦克风，而不是 speaker。

语音系统中的每个 bug 都能追溯到三个问题之一：

1. 数据录制时的 sample rate 是多少，模型期望什么？
2. Signal 是否 aliased？
3. 你处理的是 raw samples，还是 frequency representation？

这些做对，Phase 6 后面的内容就可控。做错，即使 Whisper-Large-v4 也会输出垃圾。

## 概念

![Waveform, sampling, DFT, and frequency bins visualized](../assets/audio-fundamentals.svg)

**Waveform。** `[-1.0, 1.0]` 中的一维 float array。按 sample number 索引。转换成秒：除以 sample rate：`t = n / sr`。16 kHz 下 10 秒 clip 是 160,000 个 floats。

**Sampling rate（sr）。** 每秒多少 samples。2026 年常见 rates：

| Rate | Use |
|------|-----|
| 8 kHz | Telephony，legacy VOIP。Nyquist at 4 kHz 会杀掉辅音。ASR 避免使用。 |
| 16 kHz | ASR 标准。Whisper、Parakeet、SeamlessM4T v2 都消耗 16 kHz。 |
| 22.05 kHz | 老模型的 TTS vocoder training。 |
| 24 kHz | 现代 TTS（Kokoro、F5-TTS、xTTS v2）。 |
| 44.1 kHz | CD audio，music。 |
| 48 kHz | Film、pro audio、high-fidelity TTS（VALL-E 2、NaturalSpeech 3）。 |

**Nyquist-Shannon。** `sr` 的 sample rate 可以无歧义表示到 `sr/2` 的频率。`sr/2` 边界是 *Nyquist frequency*。Nyquist 以上的能量会 *alias*，折叠到低频并污染信号。Downsampling 前始终 low-pass filter。

**Bit depth。** 16-bit PCM（signed int16，范围 ±32,767）是通用交换格式。Music 用 24-bit，内部 DSP 用 32-bit float。`soundfile` 等库读取 int16，但暴露 `[-1, 1]` 中的 float32 arrays。

**Fourier Transform。** 任何有限信号都是不同频率 sinusoids 的和。Discrete Fourier Transform（DFT）对 `N` 个 samples 计算 `N` 个 complex coefficients：每个 frequency bin 一个。`bin k` 映射到频率 `k · sr / N` Hz。Magnitude 是该频率的 amplitude，angle 是 phase。

**FFT。** Fast Fourier Transform：当 `N` 是 2 的幂时，用 `O(N log N)` 算法计算 DFT。每个音频库底层都用 FFT。16 kHz 下 1024-sample FFT 给出 512 个可用 frequency bins，覆盖 0-8 kHz，分辨率 15.6 Hz。

**Framing + window。** 我们不会对整个 clip 做 FFT。我们把它切成重叠 *frames*（通常 25 ms window、10 ms hop），给每个 frame 乘 window function（Hann、Hamming）来消除边界不连续，再对每个 frame 做 FFT。这就是 Short-Time Fourier Transform（STFT）。第 02 课从这里继续。

## 构建它

### 第 1 步：读取 clip 并画 waveform

`code/main.py` 只使用 stdlib `wave` module，保持 demo 无依赖。生产中你会使用 `soundfile` 或 `torchaudio.load`（两者都返回 `(waveform, sr)` tuple）：

```python
import soundfile as sf
waveform, sr = sf.read("clip.wav", dtype="float32")  # shape (T,), sr=int
```

### 第 2 步：从第一性原理合成 sine wave

```python
import math

def sine(freq_hz, sr, seconds, amp=0.5):
    n = int(sr * seconds)
    return [amp * math.sin(2 * math.pi * freq_hz * i / sr) for i in range(n)]
```

16 kHz 下 1 秒的 440 Hz sine（concert A）是 16,000 个 floats。用 16-bit PCM encoding 写入 `wave.open(..., "wb")`。

### 第 3 步：手算 DFT

```python
def dft(x):
    N = len(x)
    out = []
    for k in range(N):
        re = sum(x[n] * math.cos(-2 * math.pi * k * n / N) for n in range(N))
        im = sum(x[n] * math.sin(-2 * math.pi * k * n / N) for n in range(N))
        out.append((re, im))
    return out
```

`O(N²)`。`N=256` 时可用于确认正确性；真实音频不可用。真实代码调用 `numpy.fft.rfft` 或 `torch.fft.rfft`。

### 第 4 步：找 dominant frequency

Magnitude peak index `k_star` 映射到频率 `k_star * sr / N`。在 440 Hz sine 上运行，应在 bin `440 * N / sr` 看到峰值。

### 第 5 步：演示 aliasing

在 10 kHz 下采样 7 kHz sine（Nyquist = 5 kHz）。7 kHz tone 超过 Nyquist，会折叠到 `10 − 7 = 3 kHz`。FFT peak 出现在 3 kHz。这是经典 aliasing demo，也是每个 DAC/ADC 都带 brick-wall low-pass filter 的原因。

## 使用它

2026 年你实际会交付的 stack：

| 任务 | Library | 原因 |
|------|---------|-----|
| Read/write WAV/FLAC/OGG | `soundfile`（libsndfile wrapper） | 最快、稳定、返回 float32。 |
| Resample | `torchaudio.transforms.Resample` 或 `librosa.resample` | 内置正确 anti-aliasing。 |
| STFT / Mel | `torchaudio` 或 `librosa` | GPU-friendly；PyTorch 生态。 |
| Real-time streaming | `sounddevice` 或 `pyaudio` | Cross-platform PortAudio bindings。 |
| Inspect a file | `ffprobe` 或 `soxi` | CLI、快速、报告 sr/channels/codec。 |

决策规则：**先匹配 sample rate，再匹配任何别的东西**。Whisper 期望 16 kHz mono float32。给它 44.1 kHz stereo，你会得到看起来像模型 bug 的垃圾。

## 交付它

保存为 `outputs/skill-audio-loader.md`。这个 skill 帮你检查音频输入是否匹配下游模型期望，并在不匹配时正确 resample。

## 练习

1. **简单。** 在 16 kHz 下合成 1 秒 220 Hz + 440 Hz + 880 Hz 的混合。运行 DFT。确认三个 peaks 在预期 bins。
2. **中等。** 录一段 48 kHz、3 秒的自己声音 WAV。用 `torchaudio.transforms.Resample`（带 anti-aliasing）下采样到 16 kHz，再用 naive decimation（每三个 sample 取一个）下采样到 16 kHz。对两者做 FFT。Aliasing 出现在哪里？
3. **困难。** 只用 `math` 和第 3 步 DFT 从零构建 STFT。Frame size 400，hop 160，Hann window。用 `matplotlib.pyplot.imshow` 画 magnitudes。这就是第 02 课的 spectrogram。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|-----------------|-----------------------|
| Sample rate | 每秒多少 samples | ADC 测量信号的频率，单位 Hz。 |
| Nyquist | 可表示最大频率 | `sr/2`；高于它的能量会 alias 回低频。 |
| Bit depth | 每个 sample 的分辨率 | `int16` = 65,536 levels；`float32` = `[-1, 1]` 中 24-bit precision。 |
| DFT | 序列的 Fourier transform | `N` samples → `N` 个 complex frequency coefficients。 |
| FFT | 快速 DFT | `O(N log N)` 算法，要求 `N` 为 2 的幂。 |
| Bin | 频率列 | `k · sr / N` Hz；resolution = `sr / N`。 |
| STFT | Spectrogram 的底层 | 随时间做 framed + windowed FFT。 |
| Aliasing | 奇怪的频率幽灵 | Nyquist 以上能量镜像到较低 bins。 |

## 延伸阅读

- [Shannon (1949). Communication in the Presence of Noise](https://people.math.harvard.edu/~ctm/home/text/others/shannon/entropy/entropy.pdf) — sampling theorem 背后的论文。
- [Smith — The Scientist and Engineer's Guide to Digital Signal Processing](https://www.dspguide.com/ch8.htm) — 免费的经典 DSP 教材。
- [librosa docs — audio primer](https://librosa.org/doc/latest/tutorial.html) — 带代码的实用 walkthrough。
- [Heinrich Kuttruff — Room Acoustics (6th ed.)](https://www.routledge.com/Room-Acoustics/Kuttruff/p/book/9781482260434) — 真实音频为什么不是干净 sinusoid 的参考。
- [Steve Eddins — FFT Interpretation notebook](https://blogs.mathworks.com/steve/2020/03/30/fft-spectrum-and-spectral-densities/) — 10 分钟理清 frequency bin intuition。
