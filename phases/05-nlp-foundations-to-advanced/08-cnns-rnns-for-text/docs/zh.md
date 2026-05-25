# 用于文本的 CNNs 与 RNNs

> Convolutions 学 n-grams。Recurrences 负责记忆。二者都被 attention 超越了。二者在受限硬件上仍然重要。

**类型：** 构建
**语言：** Python
**前置要求：** 阶段 3 第 11 课（PyTorch Intro）、阶段 5 第 03 课（Word Embeddings）、阶段 4 第 02 课（Convolutions from Scratch）
**时间：** ~75 分钟

## 问题

TF-IDF 和 Word2Vec 产出的是忽略词序的扁平向量。建立在它们之上的分类器分不清 `dog bites man` 和 `man bites dog`。词序有时就是信号。

Transformer 出现前，有两类架构填补了这个空白。

**文本卷积网络（TextCNN）。** 在 word embeddings 序列上应用 1D convolutions。宽度为 3 的 filter 是一个可学习 trigram detector：它跨过三个词并输出分数。堆叠不同宽度（2、3、4、5）来检测多尺度模式。Max-pool 到固定大小表示。扁平、并行、快速。

**循环网络（RNN、LSTM、GRU）。** 一次处理一个 token，维护携带前文信息的 hidden state。顺序、带记忆、支持灵活输入长度。2014 到 2017 年主导 sequence modeling，直到 attention 出现。

本课会构建二者，然后点名那个促成 attention 的失败。

## 概念

**TextCNN**（Kim, 2014）。Tokens 被嵌入。宽度为 `k` 的 1D convolution 把 filter 滑过连续的 `k`-grams embeddings，产出 feature map。在 feature map 上做 global max-pooling，取最强 activation。拼接几个 filter widths 的 max-pooled outputs。送入 classifier head。

它为什么有效？Filter 就是可学习的 n-gram。Max-pooling 是 position-invariant 的，所以 "not good" 在评论开头或中间都会激活同一特征。三个 filter widths、每个 100 个 filters，给你 300 个学出来的 n-gram detectors。训练是并行的，没有 sequential dependency。

**RNN。** 在每个时间步 `t`，hidden state `h_t = f(W * x_t + U * h_{t-1} + b)`。在时间上共享 `W`、`U`、`b`。时间 `T` 的 hidden state 是整个 prefix 的摘要。分类时，在 `h_1 ... h_T` 上做 pooling（max、mean 或 last）。

普通 RNN 会遭遇 vanishing gradients。**LSTM** 加入 gates，决定忘掉什么、存储什么、输出什么，从而稳定长序列中的梯度。**GRU** 把 LSTM 简化为两个 gates；参数更少，效果相近。

**Bidirectional RNNs** 同时跑一个正向 RNN 和一个反向 RNN，并拼接 hidden states。每个 token 的表示都能看到左右两侧上下文。对 tagging tasks 很关键。

## 构建它

### 第 1 步：PyTorch 中的 TextCNN

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, n_classes, filter_widths=(2, 3, 4), n_filters=64, dropout=0.3):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, n_filters, kernel_size=k)
            for k in filter_widths
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(n_filters * len(filter_widths), n_classes)

    def forward(self, token_ids):
        x = self.embed(token_ids).transpose(1, 2)
        pooled = []
        for conv in self.convs:
            c = F.relu(conv(x))
            p = F.max_pool1d(c, c.size(2)).squeeze(2)
            pooled.append(p)
        h = torch.cat(pooled, dim=1)
        return self.fc(self.dropout(h))
```

`transpose(1, 2)` 会把 `[batch, seq_len, embed_dim]` 变成 `[batch, embed_dim, seq_len]`，因为 `nn.Conv1d` 把中间轴当成 channels。无论输入长度如何，pooled output 都是固定大小。

### 第 2 步：LSTM classifier

```python
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_classes, bidirectional=True, dropout=0.3):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=bidirectional)
        factor = 2 if bidirectional else 1
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * factor, n_classes)

    def forward(self, token_ids):
        x = self.embed(token_ids)
        out, _ = self.lstm(x)
        pooled = out.max(dim=1).values
        return self.fc(self.dropout(pooled))
```

对序列做 max-pool，而不是 last-state pool。分类时，max-pooling 通常比取最后 hidden state 更好，因为长序列末尾的信息往往会主导 last state。

### 第 3 步：vanishing gradient demo（直觉）

没有 gates 的普通 RNN 无法学习长距离依赖。考虑一个玩具任务：预测 token `A` 是否在序列中出现过。如果 `A` 在位置 1，而序列长度是 100，loss 的梯度必须穿过 recurrent weight 的 99 次乘法才能回到那里。如果权重小于 1，梯度消失。如果大于 1，梯度爆炸。

```python
def vanishing_gradient_sim(seq_len, recurrent_weight=0.9):
    import math
    return math.pow(recurrent_weight, seq_len)


# At weight=0.9 over 100 steps:
#   0.9 ^ 100 ≈ 2.7e-5
# The gradient from step 100 to step 1 is effectively zero.
```

LSTM 用 **cell state** 修复这个问题：它以加法交互穿过网络（forget gate 会乘法缩放它，但梯度仍然能沿着 “highway” 流动）。GRU 用更少参数做类似事情。二者都能让 100+ 步序列稳定训练。

### 第 4 步：为什么这仍然不够

即使用了 LSTM，仍然有三个问题。

1. **Sequential bottleneck。** 在长度为 1000 的序列上训练 RNN，需要 1000 个串行 forward/backward steps。无法沿时间并行。
2. **Encoder-decoder 设置中的固定大小 context vector。** Decoder 只能看到 encoder 的最终 hidden state，这是整个输入的压缩。长输入会丢细节。第 09 课直接讲这个。
3. **远距离依赖准确率天花板。** LSTM 优于普通 RNN，但仍然很难在 200+ 步之间传播特定信息。

Attention 解决了这三个问题。Transformers 完全移除了 recurrence。第 10 课就是转折点。

## 使用它

PyTorch 的 `nn.LSTM`、`nn.GRU` 和 `nn.Conv1d` 已经可以用于生产。训练代码是标准写法。

Hugging Face 提供 pretrained embeddings，你可以把它们接成输入层：

```python
from transformers import AutoModel

encoder = AutoModel.from_pretrained("bert-base-uncased")
for param in encoder.parameters():
    param.requires_grad = False


class BertCNN(nn.Module):
    def __init__(self, n_classes, filter_widths=(2, 3, 4), n_filters=64):
        super().__init__()
        self.encoder = encoder
        self.convs = nn.ModuleList([nn.Conv1d(768, n_filters, kernel_size=k) for k in filter_widths])
        self.fc = nn.Linear(n_filters * len(filter_widths), n_classes)

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            out = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        x = out.transpose(1, 2)
        pooled = [F.max_pool1d(F.relu(conv(x)), kernel_size=conv(x).size(2)).squeeze(2) for conv in self.convs]
        return self.fc(torch.cat(pooled, dim=1))
```

适用约束 checklist：

- **Edge / on-device inference。** 带 GloVe embeddings 的 TextCNN 比 transformer 小 10-100 倍。如果你的部署目标是手机，这就是该用的栈。
- **Streaming / online classification。** RNN 一次处理一个 token；transformers 需要完整序列。实时输入文本时，LSTM 仍然赢。
- **Tiny models for baselines。** 在新任务上快速迭代。TextCNN 可以在 CPU 上 5 分钟训练完。
- **有限数据下的 sequence labeling。** BiLSTM-CRF（第 06 课）对于 1k-10k 标注句子的 NER，仍然是生产级架构。

其他情况都交给 transformer。

## 交付它

保存为 `outputs/prompt-text-encoder-picker.md`：

```markdown
---
name: text-encoder-picker
description: 根据约束集合选择文本 encoder 架构。
phase: 5
lesson: 08
---

给定约束（任务、数据量、延迟预算、部署目标、计算预算），输出：

1. Encoder architecture：TextCNN、BiLSTM、BiLSTM-CRF、transformer fine-tune，或 "use a pretrained transformer as a frozen encoder + small head"。
2. Embedding input：random init、GloVe / fastText frozen，或 contextualized transformer embeddings。
3. 5 行 training recipe：optimizer、learning rate、batch size、epochs、regularization。
4. 一个监控信号。对 RNN/CNN models：没有 attention mechanism 意味着它们会漏掉 long-range deps；检查 per-length accuracy。对 transformers：LR 过高会导致 fine-tuning collapse；检查 train loss。

当数据少于约 500 个标注样本，且没有证明 TextCNN / BiLSTM baseline 已经 plateau 时，拒绝推荐 fine-tuning transformer。指出 edge deployment 需要 architecture-before-everything。
```

## 练习

1. **简单。** 在一个 3 类玩具数据集（你自己发明数据）上训练 TextCNN。验证 filter widths（2、3、4）在平均 F1 上优于单一 width（3）。
2. **中等。** 为 LSTM classifier 实现 max-pool、mean-pool 和 last-state pooling。在小数据集上比较；记录哪个 pooling 胜出，并假设原因。
3. **困难。** 构建 BiLSTM-CRF NER tagger（结合第 06 课和本课）。在 CoNLL-2003 上训练。和第 06 课的 CRF-alone baseline 以及 BERT fine-tune 对比。报告训练时间、内存和 F1。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|-----------------|-----------------------|
| TextCNN | 文本 CNN | 在 word embeddings 上叠 1D convolutions，并做 global max-pool。Kim (2014)。 |
| RNN | Recurrent net | 每个时间步更新 hidden state：`h_t = f(W x_t + U h_{t-1})`。 |
| LSTM | Gated RNN | 加入 input / forget / output gates 和 cell state。能在长序列上稳定训练。 |
| GRU | 更简单的 LSTM | 两个 gates 而不是三个。准确率相近，参数更少。 |
| Bidirectional | 双向 | 正向 + 反向 RNN 拼接。每个 token 都能看到上下文两侧。 |
| Vanishing gradient | 训练信号消失 | 普通 RNN 中反复乘以小于 1 的权重，使早期 step 的梯度几乎为零。 |

## 延伸阅读

- [Kim, Y. (2014). Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882) — TextCNN 论文。八页，可读性很好。
- [Hochreiter, S. and Schmidhuber, J. (1997). Long Short-Term Memory](https://www.bioinf.jku.at/publications/older/2604.pdf) — LSTM 论文。意外地清晰。
- [Olah, C. (2015). Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) — 让所有人都看懂 LSTM 的图解。
