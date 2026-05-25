# Voice Anti-Spoofing 与 Audio Watermarking：ASVspoof 5、AudioSeal、WaveVerify

> Voice cloning 交付速度快于防御。2026 年生产 voice systems 需要两样东西：一个 detector（AASIST、RawNet2）区分 real vs fake speech，以及一个能经受 compression 和 editing 的 watermark（AudioSeal）。两者都交付，否则不要交付 voice cloning。

**类型：** 构建
**语言：** Python
**前置要求：** 阶段 6 · 06（Speaker Recognition），阶段 6 · 08（Voice Cloning）
**时间：** ~75 分钟

## 问题

三个相关防御：

1. **Anti-spoofing / deepfake detection。** 给定 audio clip，它是 synthetic 还是真实？ASVspoof benchmarks（ASVspoof 2019 → 2021 → 5）是 gold standard。
2. **Audio watermarking。** 在 generated audio 中嵌入不可感知信号，后续 detector 能提取。AudioSeal（Meta）和 WavMark 是开源选项。
3. **Authenticated provenance。** 对 audio files + metadata 做 cryptographic signing。C2PA / Content Authenticity Initiative。

Detection 处理不合作的 adversaries。Watermarking 处理 compliance：AI-generated audio 应可识别。2026 年两者都必需。

## 概念

![Anti-spoofing vs watermarking vs provenance — three defense layers](../assets/spoofing-watermark.svg)

### ASVspoof 5：2024-2025 benchmark

与前几届相比最大变化：

- **Crowdsourced data**（不是 studio clean）——更接近真实条件。
- **~2000 speakers**（之前约 100）。
- **32 attack algorithms。** TTS + voice conversion + adversarial perturbation。
- **两个 tracks。** Countermeasure（CM）standalone detection；Spoofing-robust ASV（SASV）用于 biometric systems。

ASVspoof 5 上 SOTA：约 7.23% EER。旧 ASVspoof 2019 LA 上：0.42% EER。真实部署中，in-the-wild clips 上预期 5-10% EER。

### AASIST 和 RawNet2：检测模型家族

**AASIST**（2021，持续更新至 2026）。Spectral features 上的 graph-attention。ASVspoof 5 countermeasure task 当前 SOTA。

**RawNet2。** Raw waveform 上的 convolutional front-end + TDNN backbone。更简单的 baseline；fine-tuning 后仍有竞争力。

**NeXt-TDNN + SSL features。** 2025 variant：ECAPA-style + WavLM features + focal loss。在 ASVspoof 2019 LA 上达到 0.42% EER。

### AudioSeal：2024 watermark 默认

Meta 的 **AudioSeal**（2024 年 1 月，v0.2 2024 年 12 月）。关键设计：

- **Localized。** 在 16 kHz sample resolution（1/16000 s）per-frame 检测 watermark。
- **Generator + detector jointly trained。** Generator 学会嵌入 inaudible signal；detector 学会在 augmentations 下找到它。
- **Robust。** 经受 MP3 / AAC compression、EQ、speed-shift ±10%、noise mix +10 dB SNR。
- **Fast。** Detector 485× realtime；比 WavMark 快 1000×。
- **Capacity。** 每个 utterance 可嵌入 16-bit payload（model ID、generation timestamp、user ID）。

### WavMark

AudioSeal 前的开源 baseline。Invertible neural network，32 bits/sec。问题：

- Synchronization brute-force 很慢。
- 可被 Gaussian noise 或 MP3 compression 移除。
- 不适合 real-time。

### WaveVerify（2025 年 7 月）

解决 AudioSeal 弱点，尤其是 temporal manipulations（reversal、speed）。使用 FiLM-based generator + Mixture-of-Experts detector。在标准 attacks 上与 AudioSeal 竞争；能处理 temporal edits。

### Adversaries 利用的缺口

AudioMarkBench 指出：“under pitch shift, all watermarks show Bit Recovery Accuracy below 0.6, indicating near-complete removal.” **Pitch-shift 是通用攻击。** 2026 年没有 watermark 能完全抵抗 aggressive pitch modification。这就是为什么 watermarking 之外还需要 detection（AASIST）。

### C2PA / Content Authenticity Initiative

不是 ML 技术，而是 manifest format。Audio files 携带 cryptographically signed metadata，记录 creation tool、author、date。Audobox / Seamless 使用它。适合 provenance；如果坏人 re-encode 并 strip metadata，它无能为力。

## 构建它

### 第 1 步：简单 spectral-feature detector（toy）

```python
def spectral_rolloff(spec, percentile=0.85):
    cum = 0
    total = sum(spec)
    if total == 0:
        return 0
    threshold = total * percentile
    for k, v in enumerate(spec):
        cum += v
        if cum >= threshold:
            return k
    return len(spec) - 1

def is_suspicious(audio):
    spec = magnitude_spectrum(audio)
    rolloff = spectral_rolloff(spec)
    return rolloff / len(spec) > 0.92
```

Synthetic speech 经常有异常平坦的 high-frequency energy。生产 detectors 用 AASIST，而不是这个。但直觉成立。

### 第 2 步：AudioSeal embed + detect

```python
from audioseal import AudioSeal
import torch

generator = AudioSeal.load_generator("audioseal_wm_16bits")
detector = AudioSeal.load_detector("audioseal_detector_16bits")

audio = load_wav("generated.wav", sr=16000)[None, None, :]
payload = torch.tensor([[1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0]])
watermark = generator.get_watermark(audio, sample_rate=16000, message=payload)
watermarked = audio + watermark

result, decoded_payload = detector.detect_watermark(watermarked, sample_rate=16000)
# result: float in [0, 1] — probability of watermark presence
# decoded_payload: 16 bits; match against embedded payload
```

### 第 3 步：evaluation：EER

```python
def eer(real_scores, fake_scores):
    thresholds = sorted(set(real_scores + fake_scores))
    best = (1.0, 0.0)
    for t in thresholds:
        far = sum(1 for s in fake_scores if s >= t) / len(fake_scores)
        frr = sum(1 for s in real_scores if s < t) / len(real_scores)
        if abs(far - frr) < best[0]:
            best = (abs(far - frr), (far + frr) / 2)
    return best[1]
```

### 第 4 步：production integration

```python
def safe_tts(text, voice, clone_reference=None):
    if clone_reference is not None:
        verify_consent(user_id, clone_reference)
    audio = tts_model.synthesize(text, voice)
    audio_with_wm = audioseal_embed(audio, payload=build_payload(user_id, model_id))
    manifest = c2pa_sign(audio_with_wm, user_id, timestamp=now())
    return audio_with_wm, manifest
```

每次 generation 都交付：（1）watermark，（2）signed manifest，（3）符合 retention-policy 的 audit log。

## 使用它

| 用例 | 防御 |
|----------|---------|
| Shipping TTS / voice cloning | 每个 output 都 AudioSeal embed（不可协商） |
| Biometric voice unlock | AASIST + ECAPA ensemble；liveness challenge |
| Call-center fraud detection | 对 20% incoming calls 抽样跑 AASIST |
| Podcast authenticity | 上传时 C2PA signing；若 AI-generated，加 AudioSeal |
| Research / training detectors | ASVspoof 5 train/dev/eval sets |

## 坑

- **Watermark without detector ever running。** 没意义。在 CI 中交付 detector。
- **Detection without calibration。** AASIST 在 ASVspoof LA 上训练会 overfit；真实世界 accuracy 会掉。在你的 domain 上 calibrate。
- **Pitch-shift gap。** Aggressive pitch shift 会移除大多数 watermarks。准备 detection fallback。
- **Metadata strip-and-rehost。** C2PA 可被 re-encoding 轻易绕过。始终同时使用 cryptographic + perceptual（watermark）防御。
- **Liveness as detection。** 要求用户说随机短语。它能防 replay attacks，但不能防 real-time cloning。

## 交付它

保存为 `outputs/skill-spoof-defender.md`。为 voice-gen deployment 选择 detection model、watermark、provenance manifest 和 operational playbook。

## 练习

1. **简单。** 运行 `code/main.py`。在 synthetic audio 上做 toy detector + toy watermark embed/detect。
2. **中等。** 安装 `audioseal`，在 TTS output 中嵌入 16-bit payload，再 decode。用 noise 腐蚀 audio，并测 Bit Recovery Accuracy。
3. **困难。** 在 ASVspoof 2019 LA 上 fine-tune RawNet2 或 AASIST。测量 EER。再在 held-out F5-TTS-generated clips 上测试，观察 OOD detection 如何退化。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|-----------------|-----------------------|
| ASVspoof | benchmark | 双年挑战；2024 = ASVspoof 5。 |
| CM (countermeasure) | Detector | Classifier：real speech vs synthetic / converted。 |
| SASV | Speaker verif + CM | 集成 biometric + spoof detection。 |
| AudioSeal | Meta watermark | Localized，16-bit payload，比 WavMark 快 485×。 |
| Bit Recovery Accuracy | Watermark survival | 攻击后恢复 payload bits 的比例。 |
| C2PA | Provenance manifest | 关于 creation / authorship 的 cryptographic metadata。 |
| AASIST | Detector family | Graph-attention-based anti-spoofing SOTA。 |

## 延伸阅读

- [Todisco et al. (2024). ASVspoof 5](https://dl.acm.org/doi/10.1016/j.csl.2025.101825) — 当前 benchmark。
- [Defossez et al. (2024). AudioSeal](https://arxiv.org/abs/2401.17264) — watermark 默认。
- [Chen et al. (2025). WaveVerify](https://arxiv.org/abs/2507.21150) — temporal attacks 的 MoE detector。
- [Jung et al. (2022). AASIST](https://arxiv.org/abs/2110.01200) — SOTA detection backbone。
- [AudioMarkBench (2024)](https://proceedings.neurips.cc/paper_files/paper/2024/file/5d9b7775296a641a1913ab6b4425d5e8-Paper-Datasets_and_Benchmarks_Track.pdf) — robustness evaluation。
- [C2PA specification](https://c2pa.org/specifications/specifications/) — provenance manifest format。
