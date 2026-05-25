# Audio Generation

> 音频是 16-48 kHz 的 1-D 信号。一个 5 秒 clip 是 80-240k samples。没有 transformer 会直接 attend 到这个序列。2026 年每个生产 audio model 的解决方案相同：neural codec（Encodec、SoundStream、DAC）把音频压缩成 50-75 Hz 的离散 tokens，然后由 transformer 或 diffusion model 生成 tokens。

**类型：** 构建
**语言：** Python
**前置要求：** 阶段 6 · 02（Audio Features），阶段 6 · 04（ASR），阶段 8 · 06（DDPM）
**时间：** ~45 分钟

## 问题

三个 audio generation 任务：

1. **Text-to-speech。** 给定文本，产生语音。干净语音是 narrow-band，且有强 phonetic structure — transformer-over-tokens 已经很好解决。VALL-E（Microsoft）、NaturalSpeech 3、ElevenLabs、OpenAI TTS。
2. **Music generation。** 给定 prompt（text、melody、chord progression、genre），产生音乐。分布宽得多。MusicGen（Meta）、Stable Audio 2.5、Suno v4、Udio、Riffusion。
3. **Audio effects / sound design。** 给定 prompt，产生 ambient sound 或 Foley。AudioGen、AudioLDM 2、Stable Audio Open。

三者都运行在同一底座上：neural audio codec + token-AR 或 diffusion generator。

## 概念

![Audio generation: codec tokens + transformer or diffusion](../assets/audio-generation.svg)

### Neural audio codecs

Encodec（Meta, 2022）、SoundStream（Google, 2021）、Descript Audio Codec（DAC, 2023）。Convolutional encoder 把 waveform 压缩成 per-timestep vector；residual vector quantization（RVQ）把每个 vector 转成 K 个 codebook indices 的级联。Decoder 反向恢复。24 kHz audio 在 2 kbps 下，使用 8 个 RVQ codebooks、75 Hz = 600 tokens/sec。

```
waveform (16000 samples/sec)
    └─ encoder conv ─┐
                     ├─ RVQ layer 1 → indices at 75 Hz
                     ├─ RVQ layer 2 → indices at 75 Hz
                     ├─ ...
                     └─ RVQ layer 8
```

### 上层两种生成范式

**Token-autoregressive。** 把 RVQ tokens flatten 成序列，运行 decoder-only transformer。MusicGen 使用 “delayed parallel”，通过 per-stream offsets 并行发出 K 个 codebook streams。VALL-E 从 text prompt + 3-second voice sample 生成 speech tokens。

**Latent diffusion。** 把 codec tokens 打包成 continuous latents，或用 categorical diffusion 建模。Stable Audio 2.5 在 continuous audio latents 上使用 flow matching。AudioLDM 2 使用 text-to-mel-to-audio diffusion。

2024–2026 趋势：flow matching 在音乐上胜出（推理更快、样本更干净），而 token-AR 仍主导语音，因为它天然 causal 且适合 streaming。

## 生产格局

| 系统 | 任务 | Backbone | Latency |
|--------|------|----------|---------|
| ElevenLabs V3 | TTS | Token-AR + neural vocoder | ~300ms first token |
| OpenAI GPT-4o audio | Full-duplex speech | End-to-end multimodal AR | ~200ms |
| NaturalSpeech 3 | TTS | Latent flow matching | Non-streaming |
| Stable Audio 2.5 | Music / SFX | DiT + flow matching on audio latents | ~10s for 1-minute clip |
| Suno v4 | Full songs | Undisclosed; token-AR suspected | ~30s per song |
| Udio v1.5 | Full songs | Undisclosed | ~30s per song |
| MusicGen 3.3B | Music | Token-AR on Encodec 32kHz | Real-time |
| AudioCraft 2 | Music + SFX | Flow matching | ~5s for 5s clip |
| Riffusion v2 | Music | Spectrogram diffusion | ~10s |

## 构建它

`code/main.py` 模拟核心思想：在合成 “audio token” 序列上训练一个 tiny next-token transformer，这些序列来自两种不同 “styles”（style A 交替低高 tokens，style B 单调 ramp）。Condition on style 并采样。

### 第 1 步：synthetic audio tokens

```python
def make_tokens(style, length, vocab_size, rng):
    if style == 0:  # "speech-like": alternating
        return [i % vocab_size for i in range(length)]
    # "music-like": ramp
    return [(i * 3) % vocab_size for i in range(length)]
```

### 第 2 步：训练 tiny token predictor

一个按 style conditioned 的 bigram-style predictor。重点是模式：codec tokens → cross-entropy training → autoregressive sampling。

### 第 3 步：conditional sample

给定 style token 和起始 token，从预测分布中采样下一个 token。继续 20-40 tokens。

## 陷阱

- **Codec quality 限制输出质量。** 如果 codec 无法忠实表示某种声音，再强 generator 也无济于事。DAC 是当前 open best。
- **RVQ error accumulation。** 每个 RVQ layer 建模上一层 residual。Layer 1 的错误会传播。高层用 temperature 0 采样有帮助。
- **Musical structure。** 30 秒 tokens 在 75 Hz 下是 20k+ tokens。Transformer 很难。MusicGen 使用 sliding window + prompt continuation；Stable Audio 使用更短 clips + crossfading。
- **Artifacts at boundaries。** 生成 clips 之间 crossfade 需要小心 overlap-add。
- **Clean-data appetite。** Music generators 需要数万小时 licensed music。Suno / Udio RIAA lawsuit（2024）让这个问题浮上水面。
- **Voice cloning ethics。** 3 秒 sample + text prompt 就足以让 VALL-E / XTTS / ElevenLabs clone 一个声音。每个生产模型都需要 abuse detection + opt-out lists。

## 使用它

| 任务 | 2026 stack |
|------|------------|
| Commercial TTS | ElevenLabs、OpenAI TTS 或 Azure Neural |
| Voice cloning（consent-verified） | XTTS v2（open）或 ElevenLabs Pro |
| Background music, fast | Stable Audio 2.5 API、Suno 或 Udio |
| Music with lyrics | Suno v4 或 Udio v1.5 |
| Sound effects / Foley | AudioCraft 2、ElevenLabs SFX 或 Stable Audio Open |
| Real-time voice agent | GPT-4o realtime 或 Gemini Live |
| Open-weights music research | MusicGen 3.3B、Stable Audio Open 1.0、AudioLDM 2 |
| Dubbing / translation | HeyGen、ElevenLabs Dubbing |

## 交付它

保存 `outputs/skill-audio-brief.md`。Skill 接收 audio brief（task、duration、style、voice、license），并输出：model + hosting、prompt format（genre tags、style descriptors、structural markers）、codec + generator + vocoder chain、seed protocol 和 eval plan（MOS / CLAP score / CER for TTS / user A/B）。

## 练习

1. **简单。** 运行 `code/main.py` 并显式设置 style。验证生成序列匹配该 style 的 pattern。
2. **中等。** 添加 delayed parallel decoding：模拟 2 条 token streams，它们必须保持 1 step offset。训练 joint predictor。
3. **困难。** 使用 HuggingFace transformers 本地运行 MusicGen-small。用三个不同 prompts 生成 10 秒 clips；做 style adherence 的 A/B。

## 关键词

| 术语 | 大家怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| Codec | “Neural compression” | 音频 encoder / decoder；典型输出是 50-75 Hz tokens。 |
| RVQ | “Residual VQ” | K 个 quantizers 级联；每个建模上一层 residual。 |
| Token | “一个 codec symbol” | Codebook 中的离散 index；典型 1024 或 2048。 |
| Delayed parallel | “Offset codebooks” | 用错开 offsets 发出 K 个 token streams，缩短序列长度。 |
| Flow matching | “2024 年 audio 胜利” | Diffusion 的更直路径替代；采样更快。 |
| Voice prompt | “3 秒 sample” | 引导 cloned voice 的 speaker embedding 或 token prefix。 |
| Mel spectrogram | “视觉表示” | Log-magnitude perceptual spectrogram；很多 TTS 系统使用。 |
| Vocoder | “Mel to wave” | 把 mel spectrograms 转回音频的神经组件。 |

## 生产备注：audio 是 streaming 问题

Audio 是用户期望 *边生成边到达*，而不是一次性到达的输出模态。生产上这意味着 TPOT（Time Per Output Token）重要，因为用户的聆听速度就是目标吞吐量 — 不是阅读速度。对 16kHz audio、约 75 tokens/second（Encodec）tokenization 来说，服务器必须为每个用户生成 ≥75 tokens/sec 才能保持播放顺滑。

两个架构后果：

- **Flow-matching audio models 不能轻松 streaming。** Stable Audio 2.5 和 AudioCraft 2 会一次渲染固定 clip length。要 streaming，你需要 chunk clip 并 overlap boundaries — 类似 sliding-window diffusion — 相比 codec AR model 增加 100-300ms latency overhead。

如果产品是 “live voice chat” 或 “real-time music continuation”，选择 codec AR path。如果产品是 “submit 后渲染一个 30 秒 clip”，flow-matching 在质量和总延迟上胜出。

## 延伸阅读

- [Défossez et al. (2022). Encodec: High Fidelity Neural Audio Compression](https://arxiv.org/abs/2210.13438) — codec 标准。
- [Zeghidour et al. (2021). SoundStream](https://arxiv.org/abs/2107.03312) — 第一个广泛使用的 neural audio codec。
- [Kumar et al. (2023). High-Fidelity Audio Compression with Improved RVQGAN (DAC)](https://arxiv.org/abs/2306.06546) — DAC。
- [Wang et al. (2023). Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers (VALL-E)](https://arxiv.org/abs/2301.02111) — VALL-E。
- [Copet et al. (2023). Simple and Controllable Music Generation (MusicGen)](https://arxiv.org/abs/2306.05284) — MusicGen。
- [Liu et al. (2023). AudioLDM 2: Learning Holistic Audio Generation with Self-supervised Pretraining](https://arxiv.org/abs/2308.05734) — AudioLDM 2。
- [Stability AI (2024). Stable Audio 2.5](https://stability.ai/news/introducing-stable-audio-2-5) — 使用 flow matching 的 2025 text-to-music。
