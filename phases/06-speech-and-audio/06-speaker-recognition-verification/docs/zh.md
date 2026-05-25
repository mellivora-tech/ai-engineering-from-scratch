# Speaker Recognition 与 Verification

> ASR 问“他们说了什么？”Speaker recognition 问“是谁说的？”数学看起来一样：embeddings + cosine。但每个生产决策都系在一个 EER 数字上。

**类型：** 构建
**语言：** Python
**前置要求：** 阶段 6 · 02（Spectrograms & Mel），阶段 5 · 22（Embedding Models）
**时间：** ~45 分钟

## 问题

用户说了一句 passphrase。你想知道：这是否是他们声称的那个人（*verification*, 1:1），或它是否是 enrollment bank 中的第一个人（*identification*, 1:N）？或者都不是：这是不是 unknown speaker（*open-set*）？

2018 前：GMM-UBM + i-vectors。EER 还可以，但对 channel shift（phone vs laptop）和情绪很脆。2018-2022：x-vectors（用 angular margin 训练的 TDNN backbone）。2022+：ECAPA-TDNN 和 WavLM-large embeddings。到 2026 年，这个领域由三个模型和一个 metric 主导。

Metric 是 **EER**：Equal Error Rate。设置 decision threshold，使 False Accept Rate = False Reject Rate。交点就是 EER。每篇论文、每个 leaderboard、每次采购评审都用它。

## 概念

![Enrollment + verification pipeline with embedding + cosine + EER](../assets/speaker-verification.svg)

**Pipeline。** Enrollment：录制目标 speaker 5-30 秒；计算固定维度 embedding（ECAPA-TDNN 192-d，WavLM-large 256-d）。Verification：获得 test utterance embedding；计算 cosine similarity；和 threshold 比较。

**ECAPA-TDNN（2020，2026 仍主导）。** Emphasized Channel Attention, Propagation and Aggregation - Time-Delay Neural Network。1D conv blocks + squeeze-excitation + multi-head attention pooling，后接 linear layer 到 192-d。在 VoxCeleb 1+2（2,700 speakers，1.1M utterances）上用 Additive Angular Margin loss（AAM-softmax）训练。

**WavLM-SV（2022+）。** 用 AAM loss fine-tune 预训练 WavLM-large SSL backbone。质量更高但更慢：300+ MB vs 15 MB。

**x-vector（baseline）。** TDNN + statistics pooling。经典；在 CPU / edge 上仍然有用。

**AAM-softmax。** 标准 softmax 加上 angular space 的 margin `m`：正确类使用 `cos(θ + m)`。强制 inter-class angular separation。典型 `m=0.2`，scale `s=30`。

### Scoring

- **Cosine** between enrollment and test embeddings。Threshold-based decision。
- **PLDA（Probabilistic LDA）。** 把 embeddings 投影到 latent space，在那里 same-speaker vs different-speaker 有闭式 likelihood ratio。叠在 cosine 上可降低 10-20% EER。2020 前标准；现在只在 closed-set setups 中使用。
- **Score normalization。** `S-norm` 或 `AS-norm`：用 imposter cohort 的均值和 std 对每个 score 归一化。Cross-domain eval 中必需。

### 2026 年你应该知道的数字

| Model | VoxCeleb1-O EER | Params | Throughput (A100) |
|-------|-----------------|--------|-------------------|
| x-vector (classic) | 3.10% | 5 M | 400× RT |
| ECAPA-TDNN | 0.87% | 15 M | 200× RT |
| WavLM-SV large | 0.42% | 316 M | 20× RT |
| Pyannote 3.1 segmentation + embedding | 0.65% | 6 M | 100× RT |
| ReDimNet (2024) | 0.39% | 24 M | 100× RT |

### Diarization

“谁在什么时候说话”。Pipeline：VAD → segment → embed each segment → cluster（agglomerative 或 spectral）→ smooth boundaries。现代 stack：`pyannote.audio` 3.1，它把 speaker segmentation + embedding + clustering 封装成一次调用。2026 年 AMI 上 SOTA DER 约 15%（2022 年是 23%）。

## 构建它

### 第 1 步：用 MFCC statistics 做 toy embedding

```python
def embed_mfcc_stats(signal, sr):
    frames = featurize_mfcc(signal, sr, n_mfcc=13)
    mean = [sum(f[i] for f in frames) / len(frames) for i in range(13)]
    std = [
        math.sqrt(sum((f[i] - mean[i]) ** 2 for f in frames) / len(frames))
        for i in range(13)
    ]
    return mean + std  # 26-d
```

离 SOTA 很远，只用于教学。`code/main.py` 在 synthetic speaker data 上用它做 proof-of-concept。

### 第 2 步：cosine similarity + threshold

```python
def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0

def verify(enroll, test, threshold=0.75):
    return cosine(enroll, test) >= threshold
```

### 第 3 步：从 similarity pairs 计算 EER

```python
def eer(same_scores, diff_scores):
    thresholds = sorted(set(same_scores + diff_scores))
    best = (1.0, 1.0, 0.0)  # (fa, fr, threshold)
    for t in thresholds:
        fr = sum(1 for s in same_scores if s < t) / len(same_scores)
        fa = sum(1 for s in diff_scores if s >= t) / len(diff_scores)
        if abs(fa - fr) < abs(best[0] - best[1]):
            best = (fa, fr, t)
    return (best[0] + best[1]) / 2, best[2]
```

返回（eer, threshold_at_eer）。两者都报告。

### 第 4 步：用 SpeechBrain 生产化

```python
from speechbrain.pretrained import EncoderClassifier

clf = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")

# enroll: average the embeddings of 3-5 clean samples
enroll = torch.stack([clf.encode_batch(load(x)) for x in enrollment_clips]).mean(0)
# verify
score = clf.similarity(enroll, clf.encode_batch(load("test.wav"))).item()
verdict = score > 0.25   # ECAPA typical threshold; tune on your data
```

### 第 5 步：用 pyannote 做 diarization

```python
from pyannote.audio import Pipeline

pipe = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
diarization = pipe("meeting.wav", num_speakers=None)
for turn, _, speaker in diarization.itertracks(yield_label=True):
    print(f"{turn.start:.1f}–{turn.end:.1f}  {speaker}")
```

## 使用它

2026 年技术栈：

| 场景 | 选择 |
|-----------|------|
| Closed-set 1:1 verification，edge | ECAPA-TDNN + cosine threshold |
| Open-set verification，cloud | WavLM-SV + AS-norm |
| Diarization（meetings、podcasts） | `pyannote/speaker-diarization-3.1` |
| Anti-spoofing（replay / deepfake detection） | AASIST 或 RawNet2 |
| Tiny embedded（KWS + enrollment） | Titanet-Small（NeMo） |

## 坑

- **Channel mismatch。** VoxCeleb（web video）训练的模型 ≠ phone-call audio。始终在目标 channel 上评估。
- **Short utterances。** Test audio 低于 3 秒时 EER 急剧退化。
- **Enrollment with noise。** 一个 noisy enrollment 会污染 anchor。使用 ≥3 个 clean samples 并平均。
- **Fixed threshold across conditions。** 始终在目标领域的 held-out dev set 上调 threshold。
- **Cosine on non-normalized embeddings。** 先 L2-normalize；否则 magnitude 会主导。

## 交付它

保存为 `outputs/skill-speaker-verifier.md`。选择 model、enrollment protocol、threshold-tuning plan 和 fraud safeguards。

## 练习

1. **简单。** 运行 `code/main.py`。构建 synthetic “speakers”（不同 tone profiles），enroll，并在 100-pair trial list 上计算 EER。
2. **中等。** 在 30 条 VoxCeleb1 utterances（5 speakers × 每人 6 条）上使用 SpeechBrain ECAPA。比较 cosine vs PLDA 的 EER。
3. **困难。** 用 `pyannote.audio` 构建完整 enroll → diarize → verify pipeline。在 AMI dev set 上评估 DER。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|-----------------|-----------------------|
| EER | headline metric | False Accept = False Reject 的 threshold。 |
| Verification | 1:1 | “这是 Alice 吗？” |
| Identification | 1:N | “谁在说话？” |
| Open-set | 可能 unknown | Test set 可包含未 enrolled speakers。 |
| Enrollment | 注册 | 计算 speaker reference embedding。 |
| AAM-softmax | loss | 带 additive angular margin 的 softmax；强制 cluster separation。 |
| PLDA | Classic scoring | Embeddings 上的 Probabilistic LDA likelihood-ratio scoring。 |
| DER | Diarization metric | Diarization Error Rate：miss + false alarm + confusion。 |

## 延伸阅读

- [Snyder et al. (2018). X-Vectors: Robust DNN Embeddings for Speaker Recognition](https://www.danielpovey.com/files/2018_icassp_xvectors.pdf) — 经典 deep-embedding 论文。
- [Desplanques et al. (2020). ECAPA-TDNN](https://arxiv.org/abs/2005.07143) — 2020-2026 主导架构。
- [Chen et al. (2022). WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing](https://arxiv.org/abs/2110.13900) — 用于 SV 和 diarization 的 SSL backbone。
- [Bredin et al. (2023). pyannote.audio 3.1](https://github.com/pyannote/pyannote-audio) — production diarization + embedding stack。
- [VoxCeleb leaderboard (updated 2026)](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/) — 当前模型 EER 排名。
