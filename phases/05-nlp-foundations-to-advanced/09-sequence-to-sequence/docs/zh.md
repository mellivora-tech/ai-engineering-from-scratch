# Sequence-to-Sequence Models

> 两个 RNN 假装自己是翻译器。它们撞上的瓶颈，就是 attention 存在的理由。

**类型：** 构建
**语言：** Python
**前置要求：** 阶段 5 第 08 课（CNNs + RNNs for Text）、阶段 3 第 11 课（PyTorch Intro）
**时间：** ~75 分钟

## 问题

分类把可变长度序列映射到单个标签。翻译把可变长度序列映射到另一个可变长度序列。输入和输出位于不同词表中，可能属于不同语言，而且不保证长度相等。

Seq2seq 架构（Sutskever, Vinyals, Le, 2014）用一个刻意简单的配方攻克了这个问题。两个 RNN。一个读取源句子并产出固定大小的 context vector。另一个读取这个向量，并逐 token 生成目标句子。和你在第 08 课写过的代码一样，只是以不同方式粘在一起。

它值得学习有两个原因。第一，context-vector bottleneck 是 NLP 中最适合教学的失败。它解释了 attention 和 transformers 擅长的一切。第二，训练配方（teacher forcing、scheduled sampling、推理时 beam search）仍然适用于每个现代生成系统，包括 LLM。

## 概念

**Encoder。** 读取源句子的 RNN。它的最终 hidden state 是 **context vector**，也就是整个输入的固定大小摘要。理论上，除了源句子本身以外什么都不丢。

**Decoder。** 另一个用 context vector 初始化的 RNN。在每一步，它接收前一个生成 token 作为输入，并产出目标词表上的分布。Sample 或 argmax 选择下一个 token。再把它喂回去。重复，直到产生 `<EOS>` token 或达到最大长度。

**训练：** 每个 decoder step 上的 cross-entropy loss，对序列求和。对两个网络做标准 backprop through time。

**Teacher forcing。** 训练期间，decoder 在 step `t` 的输入是位置 `t-1` 的 *ground-truth* token，而不是 decoder 自己上一步的预测。这会稳定训练；没有它，早期错误会级联，模型永远学不好。推理时你必须使用模型自己的预测，所以总有 train/inference distribution gap。这个 gap 叫 **exposure bias**。

**瓶颈。** Encoder 学到的关于源句子的全部信息，都必须挤进那一个 context vector。长句会丢细节。稀有词会被模糊。重排序（chat noir vs. black cat）必须被记住，而不是计算出来。

Attention（第 10 课）通过让 decoder 查看 *每个* encoder hidden state，而不只是最后一个，修复了这个问题。这就是全部卖点。

## 构建它

### 第 1 步：encoder

```python
import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, src_vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embed = nn.Embedding(src_vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)

    def forward(self, src):
        e = self.embed(src)
        outputs, hidden = self.gru(e)
        return outputs, hidden
```

`outputs` 的 shape 是 `[batch, seq_len, hidden_dim]`，每个输入位置一个 hidden state。`hidden` 的 shape 是 `[1, batch, hidden_dim]`，最终 step。第 08 课说“分类时在 outputs 上做 pooling”。这里我们把最后 hidden state 当作 context vector，并忽略每步 outputs。

### 第 2 步：decoder

```python
class Decoder(nn.Module):
    def __init__(self, tgt_vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embed = nn.Embedding(tgt_vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, tgt_vocab_size)

    def forward(self, token, hidden):
        e = self.embed(token)
        out, hidden = self.gru(e, hidden)
        logits = self.fc(out)
        return logits, hidden
```

Decoder 每次调用一步。输入：一批单 token 和当前 hidden state。输出：下一个 token 的 vocabulary logits，以及更新后的 hidden state。

### 第 3 步：带 teacher forcing 的训练循环

```python
def train_batch(encoder, decoder, src, tgt, bos_id, optimizer, teacher_forcing_ratio=0.9):
    optimizer.zero_grad()
    _, hidden = encoder(src)
    batch_size, tgt_len = tgt.shape
    input_token = torch.full((batch_size, 1), bos_id, dtype=torch.long)
    loss = 0.0
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)

    for t in range(tgt_len):
        logits, hidden = decoder(input_token, hidden)
        step_loss = loss_fn(logits.squeeze(1), tgt[:, t])
        loss += step_loss
        use_teacher = torch.rand(1).item() < teacher_forcing_ratio
        if use_teacher:
            input_token = tgt[:, t].unsqueeze(1)
        else:
            input_token = logits.argmax(dim=-1)

    loss.backward()
    optimizer.step()
    return loss.item() / tgt_len
```

两个值得点名的旋钮。`ignore_index=0` 跳过 padding tokens 上的 loss。`teacher_forcing_ratio` 是每步使用真实 token 而不是模型预测的概率。从 1.0（完全 teacher forcing）开始，训练过程中退火到约 0.5，以缩小 exposure-bias gap。

### 第 4 步：推理循环（greedy）

```python
@torch.no_grad()
def greedy_decode(encoder, decoder, src, bos_id, eos_id, max_len=50):
    _, hidden = encoder(src)
    batch_size = src.shape[0]
    input_token = torch.full((batch_size, 1), bos_id, dtype=torch.long)
    output_ids = []
    for _ in range(max_len):
        logits, hidden = decoder(input_token, hidden)
        next_token = logits.argmax(dim=-1)
        output_ids.append(next_token)
        input_token = next_token
        if (next_token == eos_id).all():
            break
    return torch.cat(output_ids, dim=1)
```

Greedy decoding 每一步都选最高概率 token。它可能走偏：一旦承诺了一个 token，就不能收回。**Beam search** 会保留 top-`k` 个部分序列，在最后选择得分最高的完整序列。Beam width 3-5 是标准选择。

### 第 5 步：演示瓶颈

在玩具 copy task 上训练模型：source `[a, b, c, d, e]`，target `[a, b, c, d, e]`。增加序列长度。观察准确率。

```
seq_len=5   copy accuracy: 98%
seq_len=10  copy accuracy: 91%
seq_len=20  copy accuracy: 62%
seq_len=40  copy accuracy: 23%
```

单个 GRU hidden state 无法无损记住 40-token 输入。信息在每个 encoder step 都存在，但 decoder 只看最后一个 state。Attention 直接修复这个问题。

## 使用它

PyTorch 有 `nn.Transformer` 和基于 `nn.LSTM` 的 seq2seq 模板。Hugging Face 的 `transformers` 库提供完整 encoder-decoder models（BART、T5、mBART、NLLB），它们在数十亿 token 上训练过。

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tok = AutoTokenizer.from_pretrained("facebook/bart-base")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-base")

src = tok("Translate this to French: Hello, how are you?", return_tensors="pt")
out = model.generate(**src, max_new_tokens=50, num_beams=4)
print(tok.decode(out[0], skip_special_tokens=True))
```

现代 encoder-decoders 已经从 RNN 换成 transformers。高层形状（encoder、decoder、逐 token 生成）和 2014 年 seq2seq 论文完全一样。每个 block 内部机制不同。

### 什么时候仍然使用 RNN-based seq2seq

对新项目来说，几乎永远不要。特定例外：

- 流式翻译：一次消费一个输入 token，并保持有界内存。
- 设备端文本生成：transformer 内存成本太高。
- 教学。理解 encoder-decoder bottleneck，是理解 transformers 为什么胜出的最快路径。

### Exposure bias 及其缓解

- **Scheduled sampling。** 训练期间退火 teacher forcing ratio，让模型学会从自己的错误中恢复。
- **Minimum risk training。** 在句子级 BLEU score 上训练，而不是 token-level cross-entropy。更接近你真正想要的目标。
- **Reinforcement learning fine-tuning。** 用指标奖励序列生成器。现代 LLM RLHF 中也会用。

这三者仍然适用于 transformer-based generation。

## 交付它

保存为 `outputs/prompt-seq2seq-design.md`：

```markdown
---
name: seq2seq-design
description: 为给定任务设计 sequence-to-sequence pipeline。
phase: 5
lesson: 09
---

给定任务（translation、summarization、paraphrase、question rewrite），输出：

1. Architecture。Pretrained transformer encoder-decoder（BART、T5、mBART、NLLB）是默认选择。RNN-based seq2seq 只用于特定约束。
2. 起始 checkpoint。写出名称（`facebook/bart-base`、`google/flan-t5-base`、`facebook/nllb-200-distilled-600M`）。让 checkpoint 匹配任务和语言覆盖。
3. Decoding strategy。确定性输出用 greedy，追求质量用 beam search（width 4-5），追求多样性用带 temperature 的 sampling。用一句话说明理由。
4. 发布前要验证的一个失败模式。Exposure bias 会在较长输出上表现为 generation drift；抽样 20 个位于 90th-percentile length 的输出并人工检查。

标注平行样本少于一百万时，拒绝推荐从零训练 seq2seq。指出任何面向用户内容使用 greedy decoding 的 pipeline 都很脆弱（greedy 会重复和循环）。
```

## 练习

1. **简单。** 实现玩具 copy task。训练一个 GRU seq2seq，其中 input-output pairs 的 target 等于 source。衡量长度 5、10、20 上的 accuracy。复现瓶颈。
2. **中等。** 添加 beam search decoding，beam width 3。在小型平行语料上对比 greedy，衡量 BLEU。记录 beam search 赢在哪里（通常是最后几个 token），以及哪里没差别。
3. **困难。** 在 10k 对 paraphrase 数据集上 fine-tune `facebook/bart-base`。把 fine-tuned model 的 beam-4 输出和 base model 在 held-out inputs 上的输出对比。报告 BLEU，并挑 10 个定性例子。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|-----------------|-----------------------|
| Encoder | 输入 RNN | 读取源序列。产出每步 hidden states 和最终 context vector。 |
| Decoder | 输出 RNN | 从 context vector 初始化。一次生成一个 target token。 |
| Context vector | 摘要 | Encoder 最终 hidden state。固定大小。Attention 解决的瓶颈。 |
| Teacher forcing | 使用真实 token | 训练时喂入 ground-truth previous token。稳定学习。 |
| Exposure bias | 训练/测试 gap | 模型训练时一直看真实 token，从未练习从自己的错误中恢复。 |
| Beam search | 更好的 decoding | 每一步保留 top-k 部分序列，而不是贪心承诺。 |

## 延伸阅读

- [Sutskever, Vinyals, Le (2014). Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215) — 原始 seq2seq 论文。四页。
- [Cho et al. (2014). Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation](https://arxiv.org/abs/1406.1078) — 引入 GRU 和 encoder-decoder framing。
- [Bahdanau, Cho, Bengio (2014). Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473) — attention 论文。本课之后立刻读它。
- [PyTorch NLP from Scratch tutorial](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html) — 可构建的 seq2seq + attention 代码。
