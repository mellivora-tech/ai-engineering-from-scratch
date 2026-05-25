# OCR 与文档理解

> OCR 是一个三阶段流水线：检测文本框、识别字符，然后排版。每个现代 OCR 系统都会重排这些阶段，或把它们合并。

**类型：** 学习 + 使用
**语言：** Python
**前置要求：** 阶段 4 第 06 课（Detection），阶段 7 第 02 课（Self-Attention）
**时间：** ~45 分钟

## 学习目标

- 追踪经典 OCR 流水线（detect -> recognise -> layout）和现代端到端替代方案（Donut、Qwen-VL-OCR）
- 为 sequence-to-sequence OCR 训练实现 CTC（Connectionist Temporal Classification）loss
- 使用 PaddleOCR 或 EasyOCR，在不训练的情况下做生产级文档解析
- 区分 OCR、layout parsing 和 document understanding，并为每个任务选择正确工具

## 问题

充满文字的图片无处不在：收据、发票、证件、扫描书籍、表单、白板、招牌、截图。从中提取结构化数据，而不只是字符，还包括“这是总金额”，是最高价值的应用视觉问题之一。

这个领域分成三层技能：

1. **真正的 OCR**：把像素变成文本。
2. **Layout parsing**：把 OCR 输出分组成区域（title、body、table、header）。
3. **Document understanding**：从 layout 中抽取结构化字段（`invoice_total = $42.50`）。

每一层都有经典方法和现代方法，而“我想从图片里拿到文字”和“我需要这张收据的总金额”之间的距离，比多数团队意识到的更大。

## 概念

### 经典流水线

```mermaid
flowchart LR
    IMG["Image"] --> DET["Text detection<br/>(DB, EAST, CRAFT)"]
    DET --> BOX["Word/line<br/>bounding boxes"]
    BOX --> CROP["Crop each region"]
    CROP --> REC["Recognition<br/>(CRNN + CTC)"]
    REC --> TXT["Text strings"]
    TXT --> LAY["Layout<br/>ordering"]
    LAY --> OUT["Reading-order text"]

    style DET fill:#dbeafe,stroke:#2563eb
    style REC fill:#fef3c7,stroke:#d97706
    style OUT fill:#dcfce7,stroke:#16a34a
```

- **Text detection** 产生每行或每个词的 quadrilaterals。
- **Recognition** 把每个区域裁剪到固定高度，运行 CNN + BiLSTM + CTC，输出字符序列。
- **Layout** 重建阅读顺序（拉丁文字是从上到下、从左到右；阿拉伯语、日语不同）。

### 一段话理解 CTC

OCR recognition 从固定长度 feature map 产生变长序列。CTC（Graves et al., 2006）让你不用字符级 alignment 也能训练它。模型在每个时间步输出一个 over（vocab + blank）的分布；CTC loss 会对所有 alignment 求边际化，这些 alignment 在合并重复字符并移除 blanks 后都还原成目标文本。

```
raw output: "h h h _ _ e e l l _ l l o _ _"
after merge repeats and remove blanks: "hello"
```

CTC 是 CRNN 在 2015 年有效的原因，也仍然训练着 2026 年大多数生产 OCR 模型。

### 现代端到端模型

- **Donut**（Kim et al., 2022）：ViT encoder + text decoder；读取一张图片并直接输出 JSON。没有 text detector，没有 layout module。
- **TrOCR**：用于 line-level OCR 的 ViT + transformer decoder。
- **Qwen-VL-OCR / InternVL**：为 OCR 任务 fine-tuned 的完整 vision-language models；2026 年在复杂文档上准确率最好。
- **PaddleOCR**：成熟生产包中的经典 DB + CRNN 流水线；仍然是开源主力。

端到端模型需要更多数据和算力，但会跳过多阶段流水线的错误累积。

### Layout parsing

对结构化文档，运行一个 layout detector（LayoutLMv3、DocLayNet）来标注每个区域：Title、Paragraph、Figure、Table、Footnote。阅读顺序就变成了“按 layout 顺序遍历区域并拼接”。

对表单，使用 **Key-Value extraction** 模型（视觉丰富文档用 Donut，普通扫描件用 LayoutLMv3）。它们接收 image + detected text + positions，并预测结构化 key-value pairs。

### 评估指标

- **Character Error Rate（CER）**：Levenshtein distance / reference 长度。越低越好。生产目标：干净扫描件上 < 2%。
- **Word Error Rate（WER）**：词级别的同样指标。
- **结构化字段 F1**：用于 key-value 任务；衡量 `{invoice_total: 42.50}` 是否正确出现。
- **JSON edit distance**：用于端到端文档解析；Donut 论文引入了 normalised tree edit distance。

## 构建它

### 第 1 步：CTC loss + greedy decoder

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


def ctc_loss(log_probs, targets, input_lengths, target_lengths, blank=0):
    """
    log_probs:      (T, N, C) log-softmax over vocab including blank at index 0
    targets:        (N, S) int targets (no blanks)
    input_lengths:  (N,) per-sample time steps used
    target_lengths: (N,) per-sample target length
    """
    return F.ctc_loss(log_probs, targets, input_lengths, target_lengths,
                      blank=blank, reduction="mean", zero_infinity=True)


def greedy_ctc_decode(log_probs, blank=0):
    """
    log_probs: (T, N, C) log-softmax
    returns: list of index sequences (blanks removed, repeats merged)
    """
    preds = log_probs.argmax(dim=-1).transpose(0, 1).cpu().tolist()
    out = []
    for seq in preds:
        decoded = []
        prev = None
        for idx in seq:
            if idx != prev and idx != blank:
                decoded.append(idx)
            prev = idx
        out.append(decoded)
    return out
```

可用时，`F.ctc_loss` 会使用高效的 CuDNN 实现。Greedy decoder 比 beam search 简单，并且通常只差 1% CER 以内。

### 第 2 步：Tiny CRNN recogniser

用于 line OCR 的最小 CNN + BiLSTM。

```python
class TinyCRNN(nn.Module):
    def __init__(self, vocab_size=40, hidden=128, feat=32):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, feat, 3, 1, 1), nn.BatchNorm2d(feat), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(feat, feat * 2, 3, 1, 1), nn.BatchNorm2d(feat * 2), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(feat * 2, feat * 4, 3, 1, 1), nn.BatchNorm2d(feat * 4), nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),
            nn.Conv2d(feat * 4, feat * 4, 3, 1, 1), nn.BatchNorm2d(feat * 4), nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),
        )
        self.rnn = nn.LSTM(feat * 4, hidden, bidirectional=True, batch_first=True)
        self.head = nn.Linear(hidden * 2, vocab_size)

    def forward(self, x):
        # x: (N, 1, H, W)
        f = self.cnn(x)                # (N, C, H', W')
        f = f.mean(dim=2).transpose(1, 2)  # (N, W', C)
        h, _ = self.rnn(f)
        return F.log_softmax(self.head(h).transpose(0, 1), dim=-1)  # (W', N, vocab)
```

固定高度输入（CNN max-pools 高度到 1）。宽度是 CTC 的时间维度。

### 第 3 步：Synthetic OCR

生成白底黑字的数字字符串，用于端到端 smoke test。

```python
import numpy as np

def synthetic_line(text, height=32, char_width=16):
    W = char_width * len(text)
    img = np.ones((height, W), dtype=np.float32)
    for i, c in enumerate(text):
        x = i * char_width
        shade = 0.0 if c.isalnum() else 0.5
        img[6:height - 6, x + 2:x + char_width - 2] = shade
    return img


def build_batch(strings, vocab):
    H = 32
    W = 16 * max(len(s) for s in strings)
    imgs = np.ones((len(strings), 1, H, W), dtype=np.float32)
    target_lengths = []
    targets = []
    for i, s in enumerate(strings):
        imgs[i, 0, :, :16 * len(s)] = synthetic_line(s)
        ids = [vocab.index(c) for c in s]
        targets.extend(ids)
        target_lengths.append(len(ids))
    return torch.from_numpy(imgs), torch.tensor(targets), torch.tensor(target_lengths)


vocab = ["_"] + list("0123456789abcdefghijklmnopqrstuvwxyz")
imgs, targets, lengths = build_batch(["hello", "world"], vocab)
print(f"images: {imgs.shape}   targets: {targets.shape}   lengths: {lengths.tolist()}")
```

真实 OCR 数据集会加入字体、噪声、旋转、模糊和颜色。上面的 pipeline 是一样的。

### 第 4 步：Training sketch

```python
model = TinyCRNN(vocab_size=len(vocab))
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

for step in range(200):
    strings = ["abc" + str(step % 10)] * 4 + ["xyz" + str((step + 1) % 10)] * 4
    imgs, targets, target_lens = build_batch(strings, vocab)
    log_probs = model(imgs)  # (W', 8, vocab)
    input_lens = torch.full((8,), log_probs.size(0), dtype=torch.long)
    loss = ctc_loss(log_probs, targets, input_lens, target_lens, blank=0)
    opt.zero_grad(); loss.backward(); opt.step()
```

在这个平凡的 synthetic data 上，loss 应该会在 200 steps 内从 ~3 降到 ~0.2。

## 使用它

三条生产路径：

- **PaddleOCR**：成熟、快速、多语言。单行用法：`paddleocr.PaddleOCR(lang="en").ocr(image_path)`。
- **EasyOCR**：Python-native、多语言、PyTorch backbone。
- **Tesseract**：经典工具；在模型吃力的旧扫描文档上仍然有用。

端到端文档解析使用 Donut 或 VLM：

```python
from transformers import DonutProcessor, VisionEncoderDecoderModel

processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")
model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")
```

对于收据、发票和结构可重复的表单，fine-tune Donut。对于任意文档或带推理的 OCR，像 Qwen-VL-OCR 这样的 VLM 是当前默认选择。

## 交付它

本课产出：

- `outputs/prompt-ocr-stack-picker.md`：一个 prompt，会根据文档类型、语言和结构选择 Tesseract / PaddleOCR / Donut / VLM-OCR。
- `outputs/skill-ctc-decoder.md`：一个 skill，会从零写出 greedy 和 beam-search CTC decoders，并包含 length normalisation。

## 练习

1. **（简单）** 在 5 位随机数字字符串上训练 TinyCRNN 500 steps。报告 held-out set 上的 CER。
2. **（中等）** 用 beam search（beam_width=5）替换 greedy decoding。报告 CER 变化。在哪些输入上 beam search 更好？
3. **（困难）** 对 20 张收据使用 PaddleOCR，抽取 line items，并针对手工标注的 `{item_name, price}` pairs 计算 F1。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|----------------|----------------------|
| OCR | “从像素取文本” | 把图像区域转成字符序列 |
| CTC | “无需对齐的 loss” | 不需要 per-timestep labels 就能训练序列模型的 loss；对 alignments 求边际化 |
| CRNN | “经典 OCR 模型” | Conv feature extractor + BiLSTM + CTC；2015 年 baseline 仍在生产使用 |
| Donut | “端到端 OCR” | ViT encoder + text decoder；直接从图片输出 JSON |
| Layout parsing | “找区域” | 检测并标注文档中的 Title/Table/Figure/Paragraph 区域 |
| Reading order | “文本序列” | 把识别区域排序成句子；拉丁文字简单，混合 layout 不简单 |
| CER / WER | “错误率” | 字符或词粒度的 Levenshtein distance / reference length |
| VLM-OCR | “会读的 LLM” | 针对 OCR 任务训练或提示的 vision-language model；复杂文档上的当前 SOTA |

## 延伸阅读

- [CRNN (Shi et al., 2015)](https://arxiv.org/abs/1507.05717) — 原始 CNN+RNN+CTC 架构
- [CTC (Graves et al., 2006)](https://www.cs.toronto.edu/~graves/icml_2006.pdf) — 原始 CTC 论文；算法思想非常密集
- [Donut (Kim et al., 2022)](https://arxiv.org/abs/2111.15664) — OCR-free document understanding transformer
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) — 开源生产 OCR stack
