# Music Generation：MusicGen、Stable Audio、Suno 与 Licensing Earthquake

> 2026 年 music generation：Suno v5 和 Udio v4 主导商业；MusicGen、Stable Audio Open、ACE-Step 领先开源。技术问题基本解决。法律问题（Warner Music $500M settlement、UMG settlement）在 2025-2026 重塑了领域。

**类型：** 构建
**语言：** Python
**前置要求：** 阶段 6 · 02（Spectrograms），阶段 4 · 10（Diffusion Models）
**时间：** ~75 分钟

## 问题

Text → 30 秒到 4 分钟的 music clip，包含 lyrics、vocals 和 structure。三个子问题：

1. **Instrumental generation。** 文本如 “lo-fi hip-hop drums with warm keys” → audio。MusicGen、Stable Audio、AudioLDM。
2. **Song generation（with vocals + lyrics）。** “Country song about rainy Texas nights” → full song。Suno、Udio、YuE、ACE-Step。
3. **Conditional / controllable。** 扩展已有 clip，重新生成 bridge，替换 genre，stem-separate，或 inpaint。Udio 的 inpainting + stem separation 是 2026 年要对标的功能。

## 概念

![Music generation: token-LM vs diffusion, the 2026 model map](../assets/music-generation.svg)

### Neural-codec tokens 上的 Token LM

Meta 的 **MusicGen**（2023，MIT）及许多衍生模型：基于 text/melody embeddings 条件，autoregressively 预测 EnCodec tokens（32 kHz，4 codebooks），再用 EnCodec decode。300M - 3.3B params。强 baseline；超过 30 秒会挣扎。

**ACE-Step**（开源，4B XL，2026 年 4 月发布）把这条路线扩展到 lyric-conditioned full-song generation。它是开源社区最接近 Suno 的东西。

### Mels 或 latents 上的 Diffusion

**Stable Audio（2023）** 和 **Stable Audio Open（2024）**：compressed audio 上的 latent diffusion。擅长 loops、sound design、ambient textures。不擅长结构化 full songs。

**AudioLDM / AudioLDM2**：T2I-style latent diffusion 扩展到 text-to-audio，覆盖 music、sound effects、speech。

### Hybrid（production）：Suno、Udio、Lyria

闭源权重。很可能是 AR codec LM + diffusion-based vocoder，并配 specialized voice / drum / melody heads。Suno v5（2026）是 ELO 1293 quality leader。Udio v4 加入 inpainting + stem separation（bass、drums、vocals 可分别下载）。

### Evaluation

- **FAD（Fréchet Audio Distance）。** 用 VGGish 或 PANNs features，测 generated vs real audio distribution 的 embedding-level distance。越低越好。MusicGen small 在 MusicCaps 上 FAD 4.5；SOTA 约 3.0。
- **Musicality（subjective）。** Human preference。Suno v5 ELO 1293 领先。
- **Text-audio alignment。** Prompt 和 output 的 CLAP score。
- **Musicality artifacts。** Off-beat transitions、vocal-phrase drift、超过 30 s 后结构丢失。

## 2026 model map

| Model | Params | Length | Vocals | License |
|-------|--------|--------|--------|---------|
| MusicGen-large | 3.3B | 30 s | no | MIT |
| Stable Audio Open | 1.2B | 47 s | no | Stability non-commercial |
| ACE-Step XL (Apr 2026) | 4B | &gt; 2 min | yes | Apache-2.0 |
| YuE | 7B | &gt; 2 min | yes, multilingual | Apache-2.0 |
| Suno v5 (closed) | ? | 4 min | yes, ELO 1293 | commercial |
| Udio v4 (closed) | ? | 4 min | yes + stems | commercial |
| Google Lyria 3 (closed) | ? | real-time | yes | commercial |
| MiniMax Music 2.5 | ? | 4 min | yes | commercial API |

## 法律格局（2025-2026）

- **Warner Music vs Suno settlement。** $500M。WMG 现在监督 Suno 上的 AI-likeness、music rights 和 user-generated tracks。Udio 上也有类似 UMG settlement。
- **EU AI Act** + **California SB 942**：AI-generated music 必须披露。
- **Riffusion / MusicGen** 在 MIT 下没有合规包袱，但也没有商业 vocals。

安全交付模式：

1. 只生成 instrumental（MusicGen、Stable Audio Open、MIT/CC0 outputs）。
2. 使用 commercial APIs（Suno、Udio、ElevenLabs Music），按 generation 获得 license。
3. 在 owned 或 licensed catalog 上训练（大多数企业最终会走这里）。
4. 给 generations 加 watermarks + metadata。

## 构建它

### 第 1 步：用 MusicGen 生成

```python
from audiocraft.models import MusicGen
import torchaudio

model = MusicGen.get_pretrained("facebook/musicgen-small")
model.set_generation_params(duration=10)
wav = model.generate(["upbeat synthwave with driving drums, 128 BPM"])
torchaudio.save("out.wav", wav[0].cpu(), 32000)
```

三种大小：`small`（300M，快）、`medium`（1.5B）、`large`（3.3B）。Small 足以判断 idea 是否成立。

### 第 2 步：melody conditioning

```python
melody, sr = torchaudio.load("humming.wav")
wav = model.generate_with_chroma(
    ["jazz piano cover"],
    melody.squeeze(),
    sr,
)
```

MusicGen-melody 接受 chromagram，保留 tune，同时替换 timbre。适合“把这个旋律变成 string quartet”。

### 第 3 步：FAD evaluation

```python
from frechet_audio_distance import FrechetAudioDistance
fad = FrechetAudioDistance()

fad.get_fad_score("generated_folder/", "reference_folder/")
```

计算 VGGish-embedding distance。适合 genre-level regression tests；不能替代 human listeners。

### 第 4 步：加入 LLM-music workflow

结合第 7-8 课思路：

```python
prompt = "Write a 30-second jazz loop. Describe the drums, bass, and piano voicing."
description = llm.complete(prompt)
music = musicgen.generate([description], duration=30)
```

## 使用它

| Goal | Stack |
|------|-------|
| Instrumental sound design | Stable Audio Open |
| Game / adaptive music | Google Lyria RealTime（closed） |
| Full songs with vocals（commercial） | Suno v5 或 Udio v4 with explicit license |
| Full songs with vocals（open） | ACE-Step XL 或 YuE |
| Short ad jingle | MusicGen melody-conditioned on a hummed reference |
| Music-video background | MusicGen + Stable Video Diffusion |

## 2026 年仍会交付的坑

- **Copyright-laundering prompts。** “Song in the style of Taylor Swift”——商业 Suno/Udio 现在会过滤，开源模型不会。加自己的 filter list。
- **超过 30 s 的 repetition / drift。** AR models 会 loop。Crossfade 多次 generations，或用 ACE-Step 保持结构 coherence。
- **Tempo drift。** 模型会偏离 BPM。Prompt 中加 BPM tags，并用 librosa 的 `beat_track` 后过滤。
- **Vocal intelligibility。** Suno 很强；开源模型的歌词常糊。如果 lyrics 重要，用商业 API 或 fine-tune。
- **Mono output。** 开源模型生成 mono 或 fake-stereo。用合适 stereo reconstruction 升级（ezst、Cartesia's stereo diffusion）。

## 交付它

保存为 `outputs/skill-music-designer.md`。为 music-gen deployment 选择 model、license strategy、length / structure plan 和 disclosure metadata。

## 练习

1. **简单。** 运行 `code/main.py`。它用 ASCII symbols 生成“generative” chord progression + drum pattern，像 music-gen cartoon。想听可以用任意 MIDI renderer 播放。
2. **中等。** 安装 `audiocraft`，用 MusicGen-small 针对 4 个 genre prompts 生成 10 秒 clips，测量相对 reference genre set 的 FAD。
3. **困难。** 使用 ACE-Step（或 MusicGen-melody）用不同 timbre prompts 生成同一 tune 的三个 variations。计算与 prompt 的 CLAP similarity 验证 alignment。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|-----------------|-----------------------|
| FAD | Audio FID | Real vs generated embedding distributions 的 Fréchet distance。 |
| Chromagram | Melody as pitches | 每帧 12-dim vector；melody conditioning 的输入。 |
| Stems | Instrument tracks | 分离的 bass / drums / vocals / melody WAV。 |
| Inpainting | Regen a section | Mask 一个 time window；只重生成那一段。 |
| CLAP | Text-audio CLIP | Contrastive audio-text embedding；评估 text-audio alignment。 |
| EnCodec | Music codec | Meta neural codec，MusicGen 使用；32 kHz，4 codebooks。 |

## 延伸阅读

- [Copet et al. (2023). MusicGen](https://arxiv.org/abs/2306.05284) — 开源 autoregressive benchmark。
- [Evans et al. (2024). Stable Audio Open](https://arxiv.org/abs/2407.14358) — sound-design 默认。
- [ACE-Step](https://github.com/ace-step/ACE-Step) — 2026 年 4 月开源 4B full-song generator。
- [Suno v5 platform docs](https://suno.com) — 商业质量 leader。
- [AudioLDM2](https://arxiv.org/abs/2308.05734) — music + sound effects 的 latent diffusion。
- [WMG-Suno settlement coverage](https://www.musicbusinessworldwide.com/suno-warner-music-settlement/) — 2025 年 11 月 precedent。
