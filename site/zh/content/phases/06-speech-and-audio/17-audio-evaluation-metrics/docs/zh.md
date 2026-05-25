# Audio Evaluation：WER、MOS、UTMOS、MMAU、FAD 与 Open Leaderboards

> 不能测量，就不能交付。本课命名 2026 年每类音频任务的 metrics：ASR（WER、CER、RTFx）、TTS（MOS、UTMOS、SECS、WER-on-ASR-round-trip）、audio-language（MMAU、LongAudioBench）、music（FAD、CLAP）、speaker（EER）。以及你该在哪些 leaderboards 上比较。

**类型：** 学习
**语言：** Python
**前置要求：** 阶段 6 · 04、06、07、09、10；阶段 2 · 09（Model Evaluation）
**时间：** ~60 分钟

## 问题

每个 audio task 都有多个 metrics，每个测量不同轴。用错 metric，就会交付一个 dashboard 上很好、生产中很糟的模型。2026 年 canonical list：

| Task | Primary | Secondary |
|------|---------|-----------|
| ASR | WER | CER · RTFx · first-token latency |
| TTS | MOS / UTMOS | SECS · WER-on-ASR-round-trip · CER · TTFA |
| Voice cloning | SECS (ECAPA cosine) | MOS · CER |
| Speaker verification | EER | minDCF · FAR / FRR at operating point |
| Diarization | DER | JER · speaker confusion |
| Audio classification | top-1 · mAP | macro F1 · per-class recall |
| Music generation | FAD | CLAP · listening panel MOS |
| Audio language model | MMAU-Pro | LongAudioBench · AudioCaps FENSE |
| Streaming S2S | latency P50/P95 | WER · MOS |

## 概念

![Audio evaluation matrix — metrics vs tasks vs 2026 leaderboards](../assets/eval-landscape.svg)

### ASR metrics

**WER（Word Error Rate）。** `(S + D + I) / N`。Scoring 前 lowercase、strip punctuation、normalize numbers。用 `jiwer` 或 OpenAI 的 `whisper_normalizer`。< 5% = human-parity read speech。

**CER（Character Error Rate）。** 同一公式，character-level。用于普通话、粤语等 word segmentation 模糊的 tone languages。

**RTFx（inverse real-time factor）。** 每 wall-clock second 可处理多少 audio seconds。越高越好。Parakeet-TDT 达到 3380×。Whisper-large-v3 约 30×。

**First-token latency。** 从 audio input 到第一个 transcript token 的 wall-clock。Streaming 中关键。Deepgram Nova-3：约 150 ms。

### TTS metrics

**MOS（Mean Opinion Score）。** 1-5 human rating。Gold standard 但慢。每个模型收集 100+ samples，每个 sample 20+ listeners。

**UTMOS（2022-2026）。** Learned MOS predictor。在标准 benchmarks 上与 human MOS 相关约 0.9。F5-TTS：UTMOS 3.95；ground truth：4.08。

**SECS（Speaker Encoder Cosine Similarity）。** 用于 voice cloning。Reference 和 cloned output 的 ECAPA embedding cosine。> 0.75 = recognizable clone。

**WER-on-ASR-round-trip。** 把 TTS output 跑过 Whisper，对 input text 计算 WER。能抓 intelligibility regressions。2026 SOTA：< 2% CER。

**TTFA（time-to-first-audio）。** Wall-clock latency。Kokoro-82M：约 100 ms；F5-TTS：约 1 s。

### Voice-cloning-specific

**SECS + MOS + CER** 三件套。SECS 高但 MOS 低，说明 timbre 对但不自然；反过来说明声音自然但 speaker 错。

### Speaker verification

**EER（Equal Error Rate）。** False Accept Rate 等于 False Reject Rate 的 threshold。ECAPA on VoxCeleb1-O：0.87%。

**minDCF（min Detection Cost）。** 选定 operating point（常为 FAR=0.01）下的 weighted cost。比 EER 更贴近生产。

### Diarization

**DER（Diarization Error Rate）。** `(FA + Miss + Confusion) / total_speaker_time`。Missed speech + false-alarm speech + speaker-confusion，各自作为 fraction。AMI meetings：DER ~10-20% 很现实。pyannote 3.1 + Precision-2 commercial 在录音良好时 <10% DER。

**JER（Jaccard Error Rate）。** DER 替代指标，对 short-segment bias 更鲁棒。

### Audio classification

Multi-label：**mAP（mean Average Precision）**，跨所有 classes。AudioSet：BEATs-iter3 0.548 mAP。

Multi-class exclusive：**top-1、top-5 accuracy**。Speech Commands v2：Audio-MAE 99.0% top-1。

Imbalanced：**macro F1** + **per-class recall**。报告 per-class；aggregate accuracy 会隐藏哪些 classes 失败。

### Music generation

**FAD（Fréchet Audio Distance）。** Real vs generated audio 的 VGGish-embedding distributions 距离。MusicGen-small on MusicCaps：4.5。MusicLM：4.0。越低越好。

**CLAP Score。** 用 CLAP embeddings 评估 text-audio alignment。> 0.3 = reasonable alignment。

**Listening panel MOS。** 对 consumer-grade music 仍是最终判断。Suno v5 在 TTS Arena（paired human preferences）上 ELO 1293。

### Audio-language benchmarks

**MMAU（Massive Multi-Audio Understanding）。** 10k audio-QA pairs。

**MMAU-Pro。** 1800 个 hard items，四类：speech / sound / music / multi-audio。4-way random chance 25%。Gemini 2.5 Pro overall ~60%；multi-audio 上所有模型约 ~22%。

**LongAudioBench。** 带 semantic queries 的多分钟 clips。Audio Flamingo Next 击败 Gemini 2.5 Pro。

**AudioCaps / Clotho。** Captioning benchmarks。SPICE、CIDEr、FENSE metrics。

### Streaming speech-to-speech

**Latency P50 / P95 / P99。** 从 end-of-user-speech 到第一个可听 response 的 wall-clock。Moshi：200 ms；GPT-4o Realtime：300 ms。

**WER / MOS** on the output。

**Barge-in responsiveness。** 从用户打断到 assistant mute 的时间。目标 < 150 ms。

### 2026 leaderboards

| Leaderboard | Tracks | URL |
|------------|--------|-----|
| Open ASR Leaderboard (HF) | English + multilingual + long-form | `huggingface.co/spaces/hf-audio/open_asr_leaderboard` |
| TTS Arena (HF) | English TTS | `huggingface.co/spaces/TTS-AGI/TTS-Arena` |
| Artificial Analysis Speech | TTS + STT, ELO from paired votes | `artificialanalysis.ai/speech` |
| MMAU-Pro | LALM reasoning | `mmaubenchmark.github.io` |
| SpeakerBench / VoxSRC | Speaker recognition | `voxsrc.github.io` |
| MMAU music subset | Music LALM | (within MMAU) |
| HEAR benchmark | Self-supervised audio | `hearbenchmark.com` |

## 构建它

### 第 1 步：带 normalization 的 WER

```python
from jiwer import wer, Compose, ToLowerCase, RemovePunctuation, Strip

transform = Compose([ToLowerCase(), RemovePunctuation(), Strip()])
score = wer(
    truth="Please turn on the lights.",
    hypothesis="please turn on the light",
    truth_transform=transform,
    hypothesis_transform=transform,
)
# ~0.17
```

### 第 2 步：TTS round-trip WER

```python
def ttr_wer(tts_model, asr_model, texts):
    errors = []
    for txt in texts:
        audio = tts_model.synthesize(txt)
        recog = asr_model.transcribe(audio)
        errors.append(wer(truth=txt, hypothesis=recog))
    return sum(errors) / len(errors)
```

### 第 3 步：voice cloning 的 SECS

```python
from speechbrain.inference.speaker import EncoderClassifier
sv = EncoderClassifier.from_hparams("speechbrain/spkrec-ecapa-voxceleb")

emb_ref = sv.encode_batch(load_wav("reference.wav"))
emb_clone = sv.encode_batch(load_wav("cloned.wav"))
secs = torch.nn.functional.cosine_similarity(emb_ref, emb_clone, dim=-1).item()
```

### 第 4 步：music generation 的 FAD

```python
from frechet_audio_distance import FrechetAudioDistance
fad = FrechetAudioDistance()
score = fad.get_fad_score("generated_folder/", "reference_folder/")
```

### 第 5 步：speaker verification 的 EER（同第 6 课）

```python
def eer(same_scores, diff_scores):
    thresholds = sorted(set(same_scores + diff_scores))
    best = (1.0, 0.0)
    for t in thresholds:
        far = sum(1 for s in diff_scores if s >= t) / len(diff_scores)
        frr = sum(1 for s in same_scores if s < t) / len(same_scores)
        if abs(far - frr) < best[0]:
            best = (abs(far - frr), (far + frr) / 2)
    return best[1]
```

## 使用它

每次部署都配一个 fixed eval harness，并在每次 model update 上运行。三条基本规则：

1. **Normalize before scoring。** Lowercase、punctuation-strip、number-expand。报告 normalization rule。
2. **报告 distributions，不只是 averages。** Latency 的 P50/P95/P99。Classification 的 per-class recall。MMAU 的 per-category。
3. **运行一个 canonical public benchmark。** 即使生产数据不同，在 Open ASR / TTS Arena / MMAU 上报告，能让 reviewers apples-to-apples 比较。

## 坑

- **UTMOS extrapolation。** 在 VCTK-style clean speech 上训练；对 noisy / cloned / emotional audio 评分差。
- **MOS panel bias。** 20 个 Amazon Mechanical Turk workers ≠ 20 个目标用户。高 stakes 时付费做 domain panel。
- **FAD depends on reference set。** 不同模型必须用同一 reference distribution 比较。
- **Aggregate WER。** Overall 5% WER 可能隐藏 accented speech 上 30% WER。按 demographic slice 报告。
- **Public benchmark saturation。** 多数 frontier models 在标准 benchmarks 上接近 ceiling。构建反映真实 traffic 的 in-house held-out set。

## 交付它

保存为 `outputs/skill-audio-evaluator.md`。为任意 audio model release 选择 metrics、benchmarks 和 reporting format。

## 练习

1. **简单。** 运行 `code/main.py`。在 toy inputs 上计算 WER / CER / EER / SECS / FAD-ish / MMAU-ish。
2. **中等。** 构建 TTS round-trip WER harness。把 Kokoro 或 F5-TTS output 跑过 Whisper。对 50 个 prompts 计算 WER。标记 WER > 10% 的 prompts。
3. **困难。** 在 MMAU-Pro speech + multi-audio subsets（各 50 items）上评估你第 10 课选择的 LALM。报告 per-category accuracy 并与公开数字比较。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|-----------------|-----------------------|
| WER | ASR score | Normalization 后 word level 的 `(S+D+I)/N`。 |
| CER | Character WER | 用于 tone languages 或 char-level systems。 |
| MOS | Human opinion | 1-5 rating；20+ listeners × 100 samples。 |
| UTMOS | ML MOS predictor | Learned model；与 human MOS 相关约 0.9。 |
| SECS | Voice-clone similarity | Reference 和 clone 的 ECAPA cosine。 |
| EER | Speaker verif score | FAR = FRR 的 threshold。 |
| DER | Diarization score | (FA + Miss + Confusion) / total。 |
| FAD | Music-gen quality | VGGish embeddings 上的 Fréchet distance。 |
| RTFx | Throughput | 每 wall-clock second 处理的 audio seconds。 |

## 延伸阅读

- [jiwer](https://github.com/jitsi/jiwer) — 带 normalization utilities 的 WER/CER library。
- [UTMOS (Saeki et al. 2022)](https://arxiv.org/abs/2204.02152) — learned MOS predictor。
- [Fréchet Audio Distance (Kilgour et al. 2019)](https://arxiv.org/abs/1812.08466) — music-gen 标准。
- [Open ASR Leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard) — 2026 live rankings。
- [TTS Arena](https://huggingface.co/spaces/TTS-AGI/TTS-Arena) — human-vote TTS leaderboard。
- [MMAU-Pro benchmark](https://mmaubenchmark.github.io/) — LALM reasoning leaderboard。
- [HEAR benchmark](https://hearbenchmark.com/) — audio SSL benchmarks。
