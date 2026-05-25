# Attention Mechanism：突破点

> Decoder 不再眯着眼看压缩摘要，而是开始看整个源序列。之后的一切都是 attention 加工程。

**类型：** 构建
**语言：** Python
**前置要求：** 阶段 5 第 09 课（Sequence-to-Sequence Models）
**时间：** ~45 分钟

## 问题

第 09 课以一个可测量失败收尾。在玩具 copy task 上训练的 GRU encoder-decoder，从长度 5 的 89% 准确率掉到长度 80 的接近随机。原因是结构性的，不是训练 bug：encoder 采集到的每一 bit 信息都必须塞进一个固定大小 hidden state，而 decoder 永远看不到别的东西。

Bahdanau、Cho 和 Bengio 在 2014 年发表了一个三行修复。不要只把最终 encoder state 给 decoder，而是保留每个 encoder state。在每个 decoder step，计算 encoder states 的加权平均，其中权重表示“decoder 现在需要看 encoder 位置 `i` 的程度”。这个加权平均就是 context，而且每个 decoder step 都会变化。

这就是整个想法。Transformers 扩展了它。Self-attention 把它应用到单个序列。Multi-head attention 并行运行它。但 2014 版已经打破瓶颈；一旦你理解它，转向 transformers 就是工程问题，不是概念问题。

## 概念

![Bahdanau attention: decoder queries all encoder states](../assets/attention.svg)

在每个 decoder step `t`：

1. 使用前一个 decoder hidden state `s_{t-1}` 作为 **query**。
2. 把它和每个 encoder hidden state `h_1, ..., h_T` 打分。每个 encoder 位置一个标量。
3. 对分数做 softmax，得到和为 1 的 attention weights `α_{t,1}, ..., α_{t,T}`。
4. Context vector `c_t = Σ α_{t,i} * h_i`。Encoder states 的加权平均。
5. Decoder 接收 `c_t` 加上前一个 output token，产出下一个 token。

加权平均才是重点。当 decoder 需要把 "Je" 翻译成 "I" 时，它会给 "Je" 上方的 encoder state 高权重，其他位置低权重。当它需要 "not" 时，会给 "pas" 高权重。Context vector 每一步都会重塑。

## Shapes（每个人都会踩的坑）

这是每个 attention 实现第一次都会出错的地方。慢慢读。

| 东西 | Shape | 备注 |
|-------|-------|-------|
| Encoder hidden states `H` | `(T_enc, d_h)` | 如果是 BiLSTM，`d_h = 2 * d_hidden` |
| Decoder hidden state `s_{t-1}` | `(d_s,)` | 一个向量 |
| Attention score `e_{t,i}` | scalar | 每个 encoder 位置一个 |
| Attention weight `α_{t,i}` | scalar | 在所有 `i` 上 softmax 之后 |
| Context vector `c_t` | `(d_h,)` | 和 encoder state 同 shape |

**Bahdanau（additive）score。** `e_{t,i} = v_α^T * tanh(W_a * s_{t-1} + U_a * h_i)`。

- `s_{t-1}` 的 shape 是 `(d_s,)`，`h_i` 的 shape 是 `(d_h,)`。
- `W_a` 的 shape 是 `(d_attn, d_s)`。`U_a` 的 shape 是 `(d_attn, d_h)`。
- 二者在 tanh 内相加后 shape 是 `(d_attn,)`。
- `v_α` 的 shape 是 `(d_attn,)`。和 `v_α` 做内积后坍缩成标量。**这就是 `v_α` 的作用。** 它不是魔法。它是把 attention-dim vector 投影成 scalar score 的投影。

**Luong（multiplicative）score。** 三种变体：

- `dot`：`e_{t,i} = s_t^T * h_i`。要求 `d_s == d_h`。硬约束。如果 encoder 是 bidirectional，就跳过。
- `general`：`e_{t,i} = s_t^T * W * h_i`，其中 `W` shape 为 `(d_s, d_h)`。移除了等维约束。
- `concat`：本质上是 Bahdanau 形式。现在很少用，因为前两个更便宜。

**一个值得点名的 Bahdanau / Luong gotcha。** Bahdanau 使用 `s_{t-1}`（生成当前词 *之前* 的 decoder state）。Luong 使用 `s_t`（生成当前词 *之后* 的 state）。混用会产生微妙错误的梯度，极难调试。选一篇论文，坚持它的约定。

## 构建它

### 第 1 步：additive（Bahdanau）attention

```python
import numpy as np


def additive_attention(decoder_state, encoder_states, W_a, U_a, v_a):
    projected_dec = W_a @ decoder_state
    projected_enc = encoder_states @ U_a.T
    combined = np.tanh(projected_enc + projected_dec)
    scores = combined @ v_a
    weights = softmax(scores)
    context = weights @ encoder_states
    return context, weights


def softmax(x):
    x = x - np.max(x)
    e = np.exp(x)
    return e / e.sum()
```

把你的 shapes 和上表逐一对照。`encoder_states` 的 shape 是 `(T_enc, d_h)`。`projected_enc` 的 shape 是 `(T_enc, d_attn)`。`projected_dec` 的 shape 是 `(d_attn,)`，并会 broadcast。`combined` 的 shape 是 `(T_enc, d_attn)`。`scores` 的 shape 是 `(T_enc,)`。`weights` 的 shape 是 `(T_enc,)`。`context` 的 shape 是 `(d_h,)`。可以交付。

### 第 2 步：Luong dot 与 general

```python
def dot_attention(decoder_state, encoder_states):
    scores = encoder_states @ decoder_state
    weights = softmax(scores)
    return weights @ encoder_states, weights


def general_attention(decoder_state, encoder_states, W):
    projected = W.T @ decoder_state
    scores = encoder_states @ projected
    weights = softmax(scores)
    return weights @ encoder_states, weights
```

每个都是三行。这就是 Luong 论文能落地的原因。多数任务上准确率相同，代码少很多。

### 第 3 步：一个数值例子

给定三个 encoder states（大致代表 "cat"、"sat"、"mat"）和一个最接近第一个的 decoder state，attention distribution 会集中在位置 0。如果 decoder state 移动到更接近最后一个，attention 会移到位置 2。Context vector 会跟着变化。

```python
H = np.array([
    [1.0, 0.0, 0.2],
    [0.5, 0.5, 0.1],
    [0.1, 0.9, 0.3],
])

s_close_to_cat = np.array([0.9, 0.1, 0.2])
ctx, w = dot_attention(s_close_to_cat, H)
print("weights:", w.round(3))
```

```
weights: [0.464 0.305 0.231]
```

第一行胜出。然后把 decoder state 移到更接近第三个 encoder state，再观察 weights 转移。就是这样。Attention 是显式 alignment。

### 第 4 步：为什么它是通向 transformers 的桥

把上面的语言翻译成 Q/K/V：

- **Query** = decoder state `s_{t-1}`
- **Key** = encoder states（用于打分的东西）
- **Value** = encoder states（用于加权求和的东西）

在经典 attention 中，keys 和 values 是同一个东西。Self-attention 会把它们分开：你可以让一个序列 query 自己，并为 K 和 V 使用不同的学习投影。Multi-head attention 用不同学习投影并行跑它。Transformers 会把整个阶段堆叠很多次，并丢掉 RNN。

数学一样。Shapes 一样。从 Bahdanau attention 到 scaled dot-product attention 的教学跳跃主要是记号变化。

## 使用它

PyTorch 和 TensorFlow 直接提供 attention。

```python
import torch
import torch.nn as nn

mha = nn.MultiheadAttention(embed_dim=128, num_heads=8, batch_first=True)
query = torch.randn(2, 5, 128)
key = torch.randn(2, 10, 128)
value = torch.randn(2, 10, 128)

output, weights = mha(query, key, value)
print(output.shape, weights.shape)
```

```
torch.Size([2, 5, 128]) torch.Size([2, 5, 10])
```

这就是一个 transformer attention layer。Query batch 有 5 个位置，key/value batch 有 10 个位置，每个 128 维，8 个 heads。`output` 是新的、带 context 的 queries。`weights` 是你可以可视化的 5x10 alignment matrix。

### 经典 attention 仍然重要的地方

- 教学。Single-head、single-layer、RNN-based 版本让每个概念都可见。
- Transformer 放不下的设备端 sequence tasks。
- 2014-2017 年的任何论文。不知道 Bahdanau 约定就会读错。
- MT 中的细粒度 alignment analysis。Raw attention weights 即使在 transformer models 上也是解释工具；读它们需要知道它们是什么。

### attention-weight-as-explanation 陷阱

Attention weights 看起来可解释。它们是跨位置求和为一的权重；你可以画出来；高值意味着“看了这里”。审稿人喜欢它们。

它们并没有看起来那么可解释。Jain 和 Wallace（2019）展示过，在某些任务中，attention distributions 可以被置换或替换成任意 alternative，而模型预测不变。没有 ablation 或 counterfactual check 时，永远不要把 attention weights 当作推理证据来报告。

## 交付它

保存为 `outputs/prompt-attention-shapes.md`：

```markdown
---
name: attention-shapes
description: 调试 attention 实现中的 shape bugs。
phase: 5
lesson: 10
---

给定一个坏掉的 attention 实现，你识别 shape mismatch。输出：

1. 哪个矩阵 shape 错了。写出 tensor 名称。
2. 它应该是什么 shape，从 (d_s, d_h, d_attn, T_enc, T_dec, batch_size) 推导出来。
3. 一行修复。Transpose、reshape 或 project。
4. 捕获回归的测试。通常是：assert `output.shape == (batch, T_dec, d_h)`、`weights.shape == (batch, T_dec, T_enc)`，并且 `weights.sum(dim=-1) close to 1`。

拒绝推荐会静默 broadcast 的修复。Broadcast 隐藏的 bug 会在之后表现为静默准确率下降，这是最糟糕的 attention bug。

对于 Bahdanau 混淆，坚持 decoder input 是 `s_{t-1}`（pre-step state）。对于 Luong，是 `s_t`（post-step state）。对于 dot-product，把 query 和 key 的维度不匹配标记为第一次实现时最常见错误。
```

## 练习

1. **简单。** 实现 `softmax` masking，让 encoder 中的 padding tokens 得到零 attention weight。在可变长度序列 batch 上测试。
2. **中等。** 给 Luong `general` 形式添加 multi-head attention。把 `d_h` 拆成 `n_heads` 组，每个 head 跑 attention，再拼接。验证 single-head 情况与你之前的实现一致。
3. **困难。** 在第 09 课的玩具 copy task 上训练带 Bahdanau attention 的 GRU encoder-decoder。画出 accuracy vs sequence length。和没有 attention 的 baseline 对比。你应该会看到长度越长差距越大，确认 attention 抬起了瓶颈。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|-----------------|-----------------------|
| Attention | 看东西 | 对 value sequence 做 weighted average，权重由 query-key similarity 计算。 |
| Query, Key, Value | QKV | 三个投影：Q 提问，K 用来匹配，V 用来返回。 |
| Additive attention | Bahdanau | Feed-forward score：`v^T tanh(W q + U k)`。 |
| Multiplicative attention | Luong dot / general | Score 是 `q^T k` 或 `q^T W k`。更便宜，多数任务准确率相同。 |
| Alignment matrix | 好看的图 | Attention weights 构成的 `(T_dec, T_enc)` 网格。读它可以看模型 attend 到哪里。 |

## 延伸阅读

- [Bahdanau, Cho, Bengio (2014). Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473) — 这篇论文。
- [Luong, Pham, Manning (2015). Effective Approaches to Attention-based Neural Machine Translation](https://arxiv.org/abs/1508.04025) — 三种 score 变体及其比较。
- [Jain and Wallace (2019). Attention is not Explanation](https://arxiv.org/abs/1902.10186) — 关于可解释性的警告。
- [Dive into Deep Learning — Bahdanau Attention](https://d2l.ai/chapter_attention-mechanisms-and-transformers/bahdanau-attention.html) — 使用 PyTorch 的可运行 walkthrough。
