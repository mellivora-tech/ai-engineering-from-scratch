# 从零实现 Self-Attention

> Attention 是一张查询表：每个词都会问“谁对我重要？” — 然后学会答案。

**类型：** 构建
**语言：** Python
**前置要求：** 阶段 3（深度学习核心），阶段 5 第 10 课（Sequence-to-Sequence）
**时间：** ~90 分钟

## 学习目标

- 只用 NumPy 从零实现 scaled dot-product self-attention，包括 query/key/value 投影和 softmax 加权求和
- 构建一个 multi-head attention 层：拆分 heads、并行计算 attention、再拼接结果
- 追踪 attention 矩阵如何捕捉 token 关系，并解释为什么除以 sqrt(d_k) 可以避免 softmax 饱和
- 应用 causal mask，把双向 attention 转换成自回归（decoder-style）attention

## 问题

RNN 一次处理一个 token。当你走到第 50 个 token 时，第 1 个 token 的信息已经被挤过 50 次压缩。长距离依赖被压进固定大小的 hidden state — 这是一个瓶颈，再多 LSTM 门控也无法完全解决。

2014 年 Bahdanau attention 论文展示了修复方式：让 decoder 回看每个 encoder 位置，并决定当前步骤需要哪些位置。但它仍然是接在 RNN 上的组件。2017 年 “Attention Is All You Need” 论文问了一个更尖锐的问题：如果 attention 是 *唯一* 机制呢？没有 recurrence。没有 convolution。只有 attention。

Self-attention 让序列中的每个位置在单个并行步骤中 attend 到每个其他位置。这就是 transformer 快、可扩展，并最终占据主导的原因。

## 概念

### 数据库查询类比

把 attention 想成一次软数据库查询：

```
Traditional database:
  Query: "capital of France"  -->  exact match  -->  "Paris"

Attention:
  Query: "capital of France"  -->  similarity to ALL keys  -->  weighted blend of ALL values
```

每个 token 会生成三个向量：
- **Query (Q)**： “我在寻找什么？”
- **Key (K)**： “我包含什么？”
- **Value (V)**： “如果我被选中，我提供什么信息？”

一个 query 和所有 keys 的点积会产生 attention scores。高分意味着“这个 key 匹配我的 query”。这些分数会给 values 加权。输出是 values 的加权和。

### Q、K、V 计算

每个 token embedding 会通过三个学习得到的权重矩阵进行投影：

```
Input embeddings (sequence of n tokens, each d-dimensional):

  X = [x1, x2, x3, ..., xn]       shape: (n, d)

Three weight matrices:

  Wq  shape: (d, dk)
  Wk  shape: (d, dk)
  Wv  shape: (d, dv)

Projections:

  Q = X @ Wq    shape: (n, dk)      each token's query
  K = X @ Wk    shape: (n, dk)      each token's key
  V = X @ Wv    shape: (n, dv)      each token's value
```

从一个 token 的角度看：

```
             Wq
  x_i ------[*]------> q_i    "What am I looking for?"
       |
       |     Wk
       +----[*]------> k_i    "What do I contain?"
       |
       |     Wv
       +----[*]------> v_i    "What do I offer?"
```

### Attention 矩阵

一旦你有了所有 token 的 Q、K、V，attention scores 就会形成一个矩阵：

```
Scores = Q @ K^T    shape: (n, n)

              k1    k2    k3    k4    k5
        +-----+-----+-----+-----+-----+
   q1   | 2.1 | 0.3 | 0.1 | 0.8 | 0.2 |   <- how much q1 attends to each key
        +-----+-----+-----+-----+-----+
   q2   | 0.4 | 1.9 | 0.7 | 0.1 | 0.3 |
        +-----+-----+-----+-----+-----+
   q3   | 0.2 | 0.6 | 2.3 | 0.5 | 0.1 |
        +-----+-----+-----+-----+-----+
   q4   | 0.9 | 0.1 | 0.4 | 1.7 | 0.6 |
        +-----+-----+-----+-----+-----+
   q5   | 0.1 | 0.3 | 0.2 | 0.5 | 2.0 |
        +-----+-----+-----+-----+-----+

Each row: one token's attention over the entire sequence
```

### 为什么要缩放？

点积会随着维度 dk 增大而变大。如果 dk = 64，点积可能落在几十这个量级，把 softmax 推到梯度消失的区域。修复方式：除以 sqrt(dk)。

```
Scaled scores = (Q @ K^T) / sqrt(dk)
```

这样可以让数值留在 softmax 能产生有效梯度的范围内。

### Softmax 把分数变成权重

Softmax 会把原始分数转换为每一行上的概率分布：

```
Raw scores for q1:   [2.1, 0.3, 0.1, 0.8, 0.2]
                            |
                         softmax
                            |
Attention weights:   [0.52, 0.09, 0.07, 0.14, 0.08]   (sums to ~1.0)
```

现在每个 token 都有一组权重，表示它应该对其他每个 token 关注多少。

### Values 的加权和

每个 token 的最终输出是所有 value 向量的加权和：

```
output_i = sum( attention_weight[i][j] * v_j  for all j )

For token 1:
  output_1 = 0.52 * v1 + 0.09 * v2 + 0.07 * v3 + 0.14 * v4 + 0.08 * v5
```

### 完整流水线

```
                    +-------+
  X (input)  ----->|  @ Wq  |-----> Q
                    +-------+
                    +-------+
  X (input)  ----->|  @ Wk  |-----> K
                    +-------+                     +----------+
                    +-------+                     |          |
  X (input)  ----->|  @ Wv  |-----> V ---------->| weighted |----> output
                    +-------+          ^          |   sum    |
                                       |          +----------+
                              +--------+--------+
                              |    softmax      |
                              +---------+-------+
                                        ^
                              +---------+-------+
                              | Q @ K^T / sqrt  |
                              +-----------------+
```

一行公式：

```
Attention(Q, K, V) = softmax( Q @ K^T / sqrt(dk) ) @ V
```

## 构建它

### 第 1 步：从零实现 Softmax

Softmax 会把原始 logits 转换为概率。为了数值稳定性，先减去最大值。

```python
import numpy as np

def softmax(x):
    shifted = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(shifted)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

logits = np.array([2.0, 1.0, 0.1])
print(f"logits:  {logits}")
print(f"softmax: {softmax(logits)}")
print(f"sum:     {softmax(logits).sum():.4f}")
```

### 第 2 步：Scaled dot-product attention

这是核心函数。它接收 Q、K、V 矩阵，并返回 attention 输出和权重矩阵。

```python
def scaled_dot_product_attention(Q, K, V):
    dk = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(dk)
    weights = softmax(scores)
    output = weights @ V
    return output, weights
```

### 第 3 步：带学习投影的 Self-attention 类

完整的 self-attention 模块，包含 Wq、Wk、Wv 权重矩阵，并用类似 Xavier 的缩放初始化。

```python
class SelfAttention:
    def __init__(self, d_model, dk, dv, seed=42):
        rng = np.random.default_rng(seed)
        scale = np.sqrt(2.0 / (d_model + dk))
        self.Wq = rng.normal(0, scale, (d_model, dk))
        self.Wk = rng.normal(0, scale, (d_model, dk))
        scale_v = np.sqrt(2.0 / (d_model + dv))
        self.Wv = rng.normal(0, scale_v, (d_model, dv))
        self.dk = dk

    def forward(self, X):
        Q = X @ self.Wq
        K = X @ self.Wk
        V = X @ self.Wv
        output, weights = scaled_dot_product_attention(Q, K, V)
        return output, weights
```

### 第 4 步：在一句话上运行

为一句话创建假的 embeddings，并观察 attention weights。

```python
sentence = ["The", "cat", "sat", "on", "the", "mat"]
n_tokens = len(sentence)
d_model = 8
dk = 4
dv = 4

rng = np.random.default_rng(42)
X = rng.normal(0, 1, (n_tokens, d_model))

attn = SelfAttention(d_model, dk, dv, seed=42)
output, weights = attn.forward(X)

print("Attention weights (each row: where that token looks):\n")
print(f"{'':>6}", end="")
for token in sentence:
    print(f"{token:>6}", end="")
print()

for i, token in enumerate(sentence):
    print(f"{token:>6}", end="")
    for j in range(n_tokens):
        w = weights[i][j]
        print(f"{w:6.3f}", end="")
    print()
```

### 第 5 步：用 ASCII heatmap 可视化 attention

把 attention weights 映射成字符，快速获得视觉感受。

```python
def ascii_heatmap(weights, tokens, chars=" ░▒▓█"):
    n = len(tokens)
    print(f"\n{'':>6}", end="")
    for t in tokens:
        print(f"{t:>6}", end="")
    print()

    for i in range(n):
        print(f"{tokens[i]:>6}", end="")
        for j in range(n):
            level = int(weights[i][j] * (len(chars) - 1) / weights.max())
            level = min(level, len(chars) - 1)
            print(f"{'  ' + chars[level] + '   '}", end="")
        print()

ascii_heatmap(weights, sentence)
```

## 使用它

PyTorch 的 `nn.MultiheadAttention` 做的正是我们刚构建的事，另外还加上 multi-head 拆分和输出投影：

```python
import torch
import torch.nn as nn

d_model = 8
n_heads = 2
seq_len = 6

mha = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, batch_first=True)

X_torch = torch.randn(1, seq_len, d_model)

output, attn_weights = mha(X_torch, X_torch, X_torch)

print(f"Input shape:            {X_torch.shape}")
print(f"Output shape:           {output.shape}")
print(f"Attention weight shape: {attn_weights.shape}")
print(f"\nAttn weights (averaged over heads):")
print(attn_weights[0].detach().numpy().round(3))
```

关键差异是：multi-head attention 会并行运行多个 attention 函数，每个函数都有自己的 Q、K、V 投影，大小为 dk = d_model / n_heads，然后把结果拼接起来。这让模型可以同时关注不同类型的关系。

## 交付它

本课会产出：
- `outputs/prompt-attention-explainer.md` - 一个用数据库查询类比解释 attention 的 prompt

## 练习

1. 修改 `scaled_dot_product_attention`，让它接受一个可选 mask 矩阵，在 softmax 之前把某些位置设为负无穷（这就是 causal/decoder masking 的工作方式）
2. 从零实现 multi-head attention：把 Q、K、V 拆成 `n_heads` 份，在每份上运行 attention，拼接，再通过最终权重矩阵 Wo 投影
3. 取两句长度相同的不同句子，把它们送入同一个 SelfAttention 实例，并比较 attention patterns。什么变了？什么保持不变？

## 关键词

| 术语 | 大家怎么说 | 实际含义 |
|------|----------------|----------------------|
| Query (Q) | “问题向量” | 输入的一个学习投影，表示这个 token 正在寻找什么信息 |
| Key (K) | “标签向量” | 表示这个 token 包含什么信息的学习投影，用来和 query 匹配 |
| Value (V) | “内容向量” | 携带真正会被聚合的信息，聚合权重由 attention scores 决定 |
| Scaled dot-product attention | “attention 公式” | softmax(QK^T / sqrt(dk)) @ V - 缩放避免高维下 softmax 饱和 |
| Self-attention | “token 看自己和别人” | Q、K、V 都来自同一序列的 attention，让每个位置 attend 到每个其他位置 |
| Attention weights | “关注多少” | 对位置的概率分布，由 scaled dot products 经过 softmax 得到 |
| Multi-head attention | “并行 attention” | 用不同投影运行多个 attention 函数，再拼接结果得到更丰富的表示 |

## 延伸阅读

- [Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762) - 原始 transformer 论文
- [The Illustrated Transformer (Jay Alammar)](https://jalammar.github.io/illustrated-transformer/) - 全架构最好的可视化讲解
- [The Annotated Transformer (Harvard NLP)](https://nlp.seas.harvard.edu/annotated-transformer/) - 逐行 PyTorch 实现和解释
