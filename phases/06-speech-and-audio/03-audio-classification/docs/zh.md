# Audio Classification：从 MFCC 上的 k-NN 到 AST 和 BEATs

> 从“dog barking vs siren”到“这是哪种语言”，都是 audio classification。Features 是 mels。架构每十年换一次。评估始终是 AUC、F1 和 per-class recall。

**类型：** 构建
**语言：** Python
**前置要求：** 阶段 6 · 02（Spectrograms & Mel），阶段 3 · 06（CNNs），阶段 5 · 08（CNNs & RNNs for Text）
**时间：** ~75 分钟

## 问题

你拿到一个 10 秒 clip。你想知道：“它是什么？”城市声音（siren、drill、dog）、speech command（yes/no/stop）、language ID（en/es/ar）、speaker emotion（angry/neutral），或 environmental sound（indoor/outdoor、babble）。这些都是 *audio classification*。2026 年 baseline architecture 已经成熟：log-mel → CNN 或 Transformer → softmax。

核心难点不是网络，而是数据。Audio datasets 有严重 class imbalance、强 domain shift（clean vs noisy）和 label noise（谁决定“urban babble”还是“restaurant noise”？）。80% 的问题是 curation、augmentation 和 evaluation，不是把 CNN 换成 Transformer。

## 概念

![Audio classification ladder: k-NN on MFCCs to AST to BEATs](../assets/audio-classification.svg)

**MFCC 上的 k-NN（1990s baseline）。** 对每个 clip flatten MFCCs，计算与 labeled bank 的 cosine similarity，返回 top K majority vote。在干净小数据集（Speech Commands、ESC-50）上出人意料地强。不需要 GPU。

**Log-mels 上的 2D CNN（2015-2019）。** 把 `(T, n_mels)` log-mel 当成图像。应用 ResNet-18 或 VGG-style。对 time axis 做 global mean pool。Softmax over classes。2026 年多数 kaggle competitions 仍用它做 baseline。

**Audio Spectrogram Transformer，AST（2021-2024）。** 把 log-mel patchify（例如 16×16 patches），加 position embeddings，送入 ViT。监督学习上 AudioSet SOTA（mAP 0.485）。

**BEATs 和 WavLM-base（2024-2026）。** 在数百万小时上 self-supervised pretraining。Fine-tune 到你的任务，只需原先监督数据的 1-10%。2026 年这是 non-speech audio 的默认起点。BEATs-iter3 在 AudioSet 上比 AST 高 1-2 mAP，用 1/4 compute。

**Whisper-encoder 作为 frozen backbone（2024）。** 取 Whisper encoder，去掉 decoder，接 linear classifier。在 language ID 和简单 event classification 上几乎 SOTA，而且无需 audio augmentation。这是“免费午餐” baseline。

### Class imbalance 才是真挑战

ESC-50：50 类，每类 40 clips，balanced、简单。UrbanSound8K：10 类，10:1 imbalance。AudioSet：632 类，有 100,000:1 long tail。有效技术：

- 训练时 balanced sampling（评估时不要）。
- Mixup：线性插值两个 clips（以及 labels）作为 augmentation。
- SpecAugment：随机 mask time 和 frequency bands。简单但关键。

### Evaluation

- Multiclass exclusive（Speech Commands）：top-1 accuracy、top-5 accuracy。
- Multiclass multi-label（AudioSet、UrbanSound-style）：mean average precision（mAP）。
- 严重 imbalanced：per-class recall + macro F1。

你应该知道的 2026 数字：

| Benchmark | Baseline | SOTA 2026 | Source |
|-----------|----------|-----------|--------|
| ESC-50 | 82% (AST) | 97.0% (BEATs-iter3) | BEATs paper (2024) |
| AudioSet mAP | 0.485 (AST) | 0.548 (BEATs-iter3) | HEAR leaderboard 2026 |
| Speech Commands v2 | 98% (CNN) | 99.0% (Audio-MAE) | HEAR v2 results |

## 构建它

### 第 1 步：featurize

```python
def featurize_mfcc(signal, sr, n_mfcc=13, n_mels=40, frame_len=400, hop=160):
    mag = stft_magnitude(signal, frame_len, hop)
    fb = mel_filterbank(n_mels, frame_len, sr)
    mels = apply_filterbank(mag, fb)
    log = log_transform(mels)
    return [dct_ii(frame, n_mfcc) for frame in log]
```

### 第 2 步：fixed-length summary

```python
def summarize(mfcc_frames):
    n = len(mfcc_frames[0])
    mean = [sum(f[i] for f in mfcc_frames) / len(mfcc_frames) for i in range(n)]
    var = [
        sum((f[i] - mean[i]) ** 2 for f in mfcc_frames) / len(mfcc_frames) for i in range(n)
    ]
    return mean + var
```

简单但强：time 维上的 mean + variance 给 13-coef MFCC 生成 26-dim fixed embedding。瞬间运行。2017 年它还在 ESC-50 上击败过一些当时的 SOTA NN baselines。

### 第 3 步：k-NN

```python
def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1e-12
    nb = math.sqrt(sum(x * x for x in b)) or 1e-12
    return dot / (na * nb)

def knn_classify(q, bank, labels, k=5):
    sims = sorted(range(len(bank)), key=lambda i: -cosine(q, bank[i]))[:k]
    votes = Counter(labels[i] for i in sims)
    return votes.most_common(1)[0][0]
```

### 第 4 步：升级到 log-mels 上的 CNN

PyTorch 中：

```python
import torch.nn as nn

class AudioCNN(nn.Module):
    def __init__(self, n_mels=80, n_classes=50):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(128, n_classes)

    def forward(self, x):  # x: (B, 1, T, n_mels)
        return self.head(self.body(x).flatten(1))
```

3M 参数。单张 RTX 4090 上用 ~10 分钟训练 ESC-50。80%+ accuracy。

### 第 5 步：2026 默认：fine-tune BEATs

```python
from transformers import ASTFeatureExtractor, ASTForAudioClassification

ext = ASTFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
model = ASTForAudioClassification.from_pretrained(
    "MIT/ast-finetuned-audioset-10-10-0.4593",
    num_labels=50,
    ignore_mismatched_sizes=True,
)

inputs = ext(audio, sampling_rate=16000, return_tensors="pt")
logits = model(**inputs).logits
```

BEATs 用 `beats` library 中的 `microsoft/BEATs-base`；transformers API 形状相同。

## 使用它

2026 年技术栈：

| 场景 | 从这里开始 |
|-----------|-----------|
| Tiny dataset（<1000 clips） | MFCC means 上的 k-NN（你的 baseline）+ audio augmentation |
| Medium dataset（1K-100K） | BEATs 或 AST fine-tune |
| Large dataset（>100K） | 从零训练或 fine-tune Whisper-encoder |
| Real-time, edge | 40-MFCC CNN，quantized to int8（KWS-style） |
| Multi-label（AudioSet） | BEATs-iter3 + BCE loss + mixup + SpecAugment |
| Language ID | MMS-LID，SpeechBrain VoxLingua107 baseline |

决策规则：**从 frozen backbone 开始，不要从新模型开始**。Fine-tuning BEATs head 能在几小时内得到 95% SOTA，而不是几周。

## 交付它

保存为 `outputs/skill-classifier-designer.md`。为给定 audio classification task 选择 architecture、augmentations、class-balance strategy 和 eval metric。

## 练习

1. **简单。** 运行 `code/main.py`。它在 4 类 synthetic dataset（不同 pitches 的 pure tones）上训练 k-NN MFCC baseline。报告 confusion matrix。
2. **中等。** 把 `summarize` 替换为 [mean, var, skew, kurtosis]。在同一 synthetic dataset 上，4-moment pooling 是否胜过 mean+var？
3. **困难。** 用 `torchaudio` 在 ESC-50 fold 1 上训练 2D CNN。报告 5-fold cross-validation accuracy。加入 SpecAugment（time mask = 20，freq mask = 10）并报告 delta。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|-----------------|-----------------------|
| AudioSet | 音频界 ImageNet | Google 的 2M-clip、632-class weakly-labeled YouTube dataset。 |
| ESC-50 | 小型 classification benchmark | 50 类 × 40 clips environmental sounds。 |
| AST | Audio Spectrogram Transformer | Log-mel patches 上的 ViT；2021 SOTA。 |
| BEATs | Self-supervised audio | Microsoft model，iter3 截至 2026 领先 AudioSet。 |
| Mixup | Pair augmentation | `x = λ·x1 + (1-λ)·x2; y = λ·y1 + (1-λ)·y2`。 |
| SpecAugment | Mask-based augmentation | 把 spectrogram 的随机 time/frequency bands 置零。 |
| mAP | 主要 multi-label metric | 跨 classes 和 thresholds 的 mean average precision。 |

## 延伸阅读

- [Gong, Chung, Glass (2021). AST: Audio Spectrogram Transformer](https://arxiv.org/abs/2104.01778) — 2021-2024 记录架构。
- [Chen et al. (2022, rev. 2024). BEATs: Audio Pre-Training with Acoustic Tokenizers](https://arxiv.org/abs/2212.09058) — 2024+ 默认。
- [Park et al. (2019). SpecAugment](https://arxiv.org/abs/1904.08779) — 主导 audio augmentation。
- [Piczak (2015). ESC-50 dataset](https://github.com/karolpiczak/ESC-50) — 持续存在的 50-class benchmark。
- [Gemmeke et al. (2017). AudioSet](https://research.google.com/audioset/) — 632-class YouTube taxonomy；仍是 gold standard。
