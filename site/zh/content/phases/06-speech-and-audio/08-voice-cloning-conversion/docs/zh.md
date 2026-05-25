# Voice Cloning 与 Voice Conversion

> Voice cloning 用别人的声音读你的文本。Voice conversion 把你的声音改写成别人的声音，同时保留你说了什么。两者都依赖同一个 primitive：把 speaker identity 和 content 分离。

**类型：** 构建
**语言：** Python
**前置要求：** 阶段 6 · 06（Speaker Recognition），阶段 6 · 07（TTS）
**时间：** ~75 分钟

## 问题

到 2026 年，5 秒 audio clip 就足以用消费级 GPU 生成任何人的高质量 voice clone。ElevenLabs、F5-TTS、OpenVoice v2、VoiceBox 都交付了 zero-shot 或 few-shot cloning。这项技术既是福音（accessibility TTS、dubbing、assistive voices），也是武器（诈骗电话、政治 deepfakes、IP theft）。

两个密切相关的任务：

- **Voice cloning（TTS-side）：** text + 5-second reference voice → 该声音的 audio。
- **Voice conversion（speech-side）：** source audio（A 说 X）+ B 的 reference voice → B 说 X 的 audio。

两者都把 waveform 分解为（content, speaker, prosody），再把一个 source 的 content 和另一个 source 的 speaker 重组。

2026 年你交付时的关键约束：**watermarking 和 consent gates 在 EU（AI Act，2026 年 8 月 enforceable）以及 California（AB 2905，2025 生效）已成为法律要求**。你的 pipeline 必须发出 inaudible watermark，并拒绝 non-consensual clones。

## 概念

![Voice cloning vs conversion: factorize, swap speaker, recombine](../assets/voice-cloning.svg)

**Zero-shot cloning。** 给模型传入 5 秒 clip，模型已经在数千 speakers 上训练过。Speaker encoder 把 clip 映射到 speaker embedding；TTS decoder 基于该 embedding 和 text 条件生成。

使用者：F5-TTS（2024）、YourTTS（2022）、XTTS v2（2024）、OpenVoice v2（2024）。

**Few-shot fine-tuning。** 录制目标 voice 5-30 分钟。对 base model 做 LoRA fine-tune 1 小时。质量从“还行”跳到“难以区分”。Coqui 和 ElevenLabs 都支持这种模式；社区也在 F5-TTS 上使用。

**Voice conversion（VC）。** 两大类：

- **Recognition-synthesis。** 运行 ASR-like model 提取 content representation（例如 soft phoneme posteriors、PPGs），再用 target speaker embedding resynthesize。对语言和口音鲁棒。KNN-VC（2023）、Diff-HierVC（2023）使用。
- **Disentanglement。** 训练 autoencoder，在 bottleneck latent space 中分离 content、speaker、prosody。推理时替换 speaker embedding。质量较低但更快。AutoVC（2019）、VITS-VC variants 使用。

**Neural codec-based cloning（2024+）。** VALL-E、VALL-E 2、NaturalSpeech 3、VoiceBox：把 audio 当成 SoundStream / EnCodec 产生的 discrete tokens，训练大 autoregressive 或 flow-matching model 预测 codec tokens。短 prompts 上质量接近 ElevenLabs。

### 伦理不是外挂

**Watermarking。** PerTh（Perth）和 SilentCipher（2024）在音频中不可感知地嵌入约 16-32 bit ID。能经受 re-encoding、streaming 和常见 edits。生产可用开源。

**Consent gates。** 每个 cloned output 必须关联可验证 consent record。“I, Rohit, on 2026-04-22, authorize this voice for X purpose.” 存入 tamper-evident log。

**Detection。** AASIST、RawNet2、Wav2Vec2-AASIST 可作为 detectors。ASVspoof 2025 challenge 公布了 against ElevenLabs、VALL-E 2、Bark outputs 的 state-of-the-art detectors EER：0.8-2.3%。

### 数字（2026）

| Model | Zero-shot? | SECS (target sim) | WER (intel.) | Params |
|-------|-----------|--------------------|--------------|--------|
| F5-TTS | Yes | 0.72 | 2.1% | 335M |
| XTTS v2 | Yes | 0.65 | 3.5% | 470M |
| OpenVoice v2 | Yes | 0.70 | 2.8% | 220M |
| VALL-E 2 | Yes | 0.77 | 2.4% | 370M |
| VoiceBox | Yes | 0.78 | 2.1% | 330M |

SECS > 0.70 对多数听众通常已经与目标难以区分。

## 构建它

### 第 1 步：用 recognition-synthesis 分解（main.py 中 code-only demo）

```python
def clone_pipeline(ref_audio, text, target_embedder, tts_model):
    speaker_emb = target_embedder.encode(ref_audio)
    mel = tts_model(text, speaker=speaker_emb)
    return vocoder(mel)
```

概念简单；实现体量在 `tts_model` 和 speaker encoder 中。

### 第 2 步：用 F5-TTS 做 zero-shot clone

```python
from f5_tts.api import F5TTS
tts = F5TTS()
wav = tts.infer(
    ref_file="rohit_5s.wav",
    ref_text="The quick brown fox jumps over the lazy dog.",
    gen_text="Please add milk and bread to my list.",
)
```

Reference transcript 必须和 audio 完全匹配；不匹配会破坏 alignment。

### 第 3 步：用 KNN-VC 做 voice conversion

```python
import torch
from knnvc import KNNVC  # 2023 model, https://github.com/bshall/knn-vc
vc = KNNVC.load("wavlm-base-plus")
out_wav = vc.convert(source="my_voice.wav", target_pool=["alice_1.wav", "alice_2.wav"])
```

KNN-VC 用 WavLM 为 source 和 target pool 提取 per-frame embeddings，再用 target pool 中最近邻替换每个 source frame。Non-parametric，有一分钟 target speech 就能工作。

### 第 4 步：嵌入 watermark

```python
from silentcipher import SilentCipher
sc = SilentCipher(model="2024-06-01")
payload = b"consent_id:abc123;ts:1745353200"
watermarked = sc.embed(wav, sr=24000, message=payload)
detected = sc.detect(watermarked, sr=24000)   # returns payload bytes
```

约 32 bits payload，可在 MP3 re-encode 和轻微 noise 后检测。

### 第 5 步：consent gate

```python
def cloned_inference(text, ref_audio, consent_record):
    assert verify_signature(consent_record), "Signed consent required"
    assert consent_record["speaker_id"] == hash_speaker(ref_audio)
    wav = tts.infer(ref_file=ref_audio, gen_text=text)
    wav = watermark(wav, payload=consent_record["id"])
    return wav
```

## 使用它

2026 年技术栈：

| 场景 | 选择 |
|-----------|------|
| 5-sec zero-shot clone，open-source | F5-TTS 或 OpenVoice v2 |
| Commercial production cloning | ElevenLabs Instant Voice Clone v2.5 |
| Voice conversion（rewriting） | KNN-VC 或 Diff-HierVC |
| Many-speaker fine-tune | StyleTTS 2 + speaker adapter |
| Cross-lingual cloning | XTTS v2 或 VALL-E X |
| Deepfake detection | Wav2Vec2-AASIST |

## 坑

- **Misaligned reference transcript。** F5-TTS 等模型要求 reference text 精确匹配 reference audio，包括标点。
- **Reverberant reference。** Echo 会毁掉 clone。使用 dry、close-mic 录音。
- **Emotional mismatch。** “cheerful” reference 会让所有 clones 都 cheerful。让 reference emotion 匹配目标用途。
- **Language leakage。** 克隆英语 speaker 再让模型说法语，常会带口音；用 cross-lingual models（XTTS、VALL-E X）。
- **No watermark。** 2026 年 8 月起在 EU 不可合法交付。

## 交付它

保存为 `outputs/skill-voice-cloner.md`。设计带 consent gate + watermark + quality target 的 cloning 或 conversion pipeline。

## 练习

1. **简单。** 运行 `code/main.py`。通过计算两个 “speakers” 在 swap 前后的 cosine，演示 speaker-embedding swap。
2. **中等。** 用 OpenVoice v2 clone 你自己的声音。测量 reference 和 clone 的 SECS。用 Whisper 测 CER。
3. **困难。** 对 20 个 clones 应用 SilentCipher watermark，让它们经过 128 kbps MP3 encode+decode 后检测 payload。报告 bit-accuracy。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|-----------------|-----------------------|
| Zero-shot clone | 5 秒就够 | Pretrained model + speaker embedding；无需训练。 |
| PPG | Phonetic posteriorgram | ASR per-frame posteriors，用作 language-agnostic content rep。 |
| KNN-VC | Nearest-neighbor conversion | 用最近的 target-pool frame 替换每个 source frame。 |
| Neural codec TTS | VALL-E style | EnCodec/SoundStream tokens 上的 AR model。 |
| Watermark | Inaudible signature | 嵌入 audio 的 bits，能经受 re-encode。 |
| SECS | Cloning fidelity | Target 和 clone speaker embeddings 的 cosine。 |
| AASIST | Deepfake detector | Anti-spoof model；检测 synthesized speech。 |

## 延伸阅读

- [Chen et al. (2024). F5-TTS](https://arxiv.org/abs/2410.06885) — open-source SOTA zero-shot cloning。
- [Baevski et al. / Microsoft (2023). VALL-E](https://arxiv.org/abs/2301.02111) and [VALL-E 2 (2024)](https://arxiv.org/abs/2406.05370) — neural-codec TTS。
- [Qian et al. (2019). AutoVC](https://arxiv.org/abs/1905.05879) — disentanglement-based voice conversion。
- [Baas, Waubert de Puiseau, Kamper (2023). KNN-VC](https://arxiv.org/abs/2305.18975) — retrieval-based VC。
- [SilentCipher (2024) — Audio Watermarking](https://github.com/sony/silentcipher) — production-ready 32-bit audio watermark。
- [ASVspoof 2025 results](https://www.asvspoof.org/) — detector vs synthesizer arms race，2026 更新。
