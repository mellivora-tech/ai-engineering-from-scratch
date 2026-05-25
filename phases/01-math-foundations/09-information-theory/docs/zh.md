# 信息论

> 信息论衡量惊讶程度。Loss functions 建立在它之上。

**类型：** 学习
**语言：** Python
**前置要求：** 阶段 1，第 06 课（概率）
**时间：** ~60 分钟

## 学习目标

- 从零计算 entropy、cross-entropy 和 KL divergence，并解释它们的关系
- 推导为什么最小化 cross-entropy loss 等价于最大化 log-likelihood
- 计算 features 与 target 之间的 mutual information，用于排序 feature importance
- 解释 perplexity 如何表示语言模型正在从多少有效词汇中做选择

## 问题

你在训练每个分类模型时都会调用 `CrossEntropyLoss()`。你在每篇语言模型论文中都会看到 "perplexity"。你会在 VAE、distillation 和 RLHF 中读到 KL divergence。这些不是互不相干的概念。它们都是同一个思想，只是戴着不同帽子。

信息论给你一套语言，用来推理不确定性、压缩和预测。Claude Shannon 在 1948 年发明它，是为了解决通信问题。结果发现，训练神经网络也是一个通信问题：模型试图通过由学到的权重组成的 noisy channel，传递正确标签。

本课会从零构建每个公式，让你看到它们从哪里来，以及为什么有效。

## 概念

### 信息量（惊讶）

当不太可能发生的事情发生时，它携带更多信息。一枚硬币正面？不惊讶。中彩票？非常惊讶。

概率为 p 的事件的信息量是：

```
I(x) = -log(p(x))
```

使用以 2 为底的 log，单位是 bits。使用自然 log，单位是 nats。同一个思想，不同单位。

```
Event              Probability    Surprise (bits)
Fair coin heads    0.5            1.0
Rolling a 6        0.167          2.58
1-in-1000 event    0.001          9.97
Certain event      1.0            0.0
```

确定会发生的事件携带零信息。你本来就知道它会发生。

### Entropy（平均惊讶）

Entropy 是一个分布所有可能结果的期望惊讶程度。

```
H(P) = -sum( p(x) * log(p(x)) )  for all x
```

公平硬币对二元变量有最大 entropy：1 bit。有偏硬币（99% 正面）entropy 很低：0.08 bits。你几乎已经知道会发生什么，所以每次抛掷几乎不告诉你新信息。

```
Fair coin:    H = -(0.5 * log2(0.5) + 0.5 * log2(0.5)) = 1.0 bit
Biased coin:  H = -(0.99 * log2(0.99) + 0.01 * log2(0.01)) = 0.08 bits
```

Entropy 衡量一个分布中不可约的不确定性。你无法压缩到低于它。

### Cross-Entropy（你每天使用的 Loss Function）

Cross-entropy 衡量：当你使用分布 Q 去编码实际上来自分布 P 的事件时，平均惊讶程度是多少。

```
H(P, Q) = -sum( p(x) * log(q(x)) )  for all x
```

P 是真实分布（标签）。Q 是模型预测。如果 Q 与 P 完全匹配，cross-entropy 等于 entropy。任何不匹配都会让它变大。

在分类中，P 是 one-hot vector（真实类别概率为 1，其他为 0）。这会把 cross-entropy 简化成：

```
H(P, Q) = -log(q(true_class))
```

这就是分类中完整的 cross-entropy loss 公式。最大化正确类别的预测概率。

### KL Divergence（分布之间的距离）

KL divergence 衡量使用 Q 代替 P 会带来多少额外惊讶。

```
D_KL(P || Q) = sum( p(x) * log(p(x) / q(x)) )  for all x
             = H(P, Q) - H(P)
```

Cross-entropy 是 entropy 加 KL divergence。由于真实分布的 entropy 在训练期间是常数，最小化 cross-entropy 等价于最小化 KL divergence。你在把模型分布推向真实分布。

KL divergence 不对称：D_KL(P || Q) != D_KL(Q || P)。它不是真正的距离度量。

### Mutual Information

Mutual information 衡量知道一个变量能告诉你关于另一个变量多少信息。

```
I(X; Y) = H(X) - H(X|Y)
        = H(X) + H(Y) - H(X, Y)
```

如果 X 和 Y 独立，mutual information 为零。知道一个不会告诉你另一个任何信息。如果它们完全相关，mutual information 等于任一变量的 entropy。

在 feature selection 中，feature 与 target 之间的 mutual information 高，意味着这个 feature 有用。Mutual information 低，意味着它是噪声。

### Conditional Entropy

H(Y|X) 衡量观察 X 之后，关于 Y 还剩多少不确定性。

```
H(Y|X) = H(X,Y) - H(X)
```

两个极端：
- 如果 X 完全决定 Y，则 H(Y|X) = 0。知道 X 会消除关于 Y 的全部不确定性。例子：X = 摄氏温度，Y = 华氏温度。
- 如果 X 完全不告诉你关于 Y 的信息，则 H(Y|X) = H(Y)。知道 X 不会减少你的不确定性。例子：X = 抛硬币结果，Y = 明天天气。

Conditional entropy 总是非负，并且不超过 H(Y)：

```
0 <= H(Y|X) <= H(Y)
```

在机器学习中，conditional entropy 出现在 decision trees 中。每次 split 时，算法选择让 H(Y|X) 最小的 feature X：也就是最能减少关于 label Y 不确定性的 feature。

### Joint Entropy

H(X,Y) 是 X 和 Y 联合分布的 entropy。

```
H(X,Y) = -sum sum p(x,y) * log(p(x,y))   for all x, y
```

关键性质：

```
H(X,Y) <= H(X) + H(Y)
```

当 X 和 Y 独立时等号成立。如果它们共享信息，joint entropy 会小于单独 entropy 之和。“缺失”的 entropy 正好是 mutual information。

```mermaid
graph TD
    subgraph "Information Venn Diagram"
        direction LR
        HX["H(X)"]
        HY["H(Y)"]
        MI["I(X;Y)<br/>Mutual<br/>Information"]
        HXgY["H(X|Y)<br/>= H(X) - I(X;Y)"]
        HYgX["H(Y|X)<br/>= H(Y) - I(X;Y)"]
        HXY["H(X,Y) = H(X) + H(Y) - I(X;Y)"]
    end

    HXgY --- MI
    MI --- HYgX
    HX -.- HXgY
    HX -.- MI
    HY -.- MI
    HY -.- HYgX
    HXY -.- HXgY
    HXY -.- MI
    HXY -.- HYgX
```

关系：
- H(X,Y) = H(X) + H(Y|X) = H(Y) + H(X|Y)
- I(X;Y) = H(X) - H(X|Y) = H(Y) - H(Y|X)
- H(X,Y) = H(X) + H(Y) - I(X;Y)

### Mutual Information（深入）

Mutual information I(X;Y) 量化知道一个变量能减少多少关于另一个变量的不确定性。

```
I(X;Y) = H(X) - H(X|Y)
       = H(Y) - H(Y|X)
       = H(X) + H(Y) - H(X,Y)
       = sum sum p(x,y) * log(p(x,y) / (p(x) * p(y)))
```

性质：
- I(X;Y) >= 0 永远成立。观察某件事不会让你丢失信息。
- I(X;Y) = 0 当且仅当 X 和 Y 独立。
- I(X;Y) = I(Y;X)。它是对称的，不像 KL divergence。
- I(X;X) = H(X)。一个变量与自身共享全部信息。

**用于 feature selection 的 mutual information。** 在 ML 中，你想要对 target 有信息量的 features。Mutual information 给了你一种有原则的 feature 排序方式：

1. 对每个 feature X_i，计算 I(X_i; Y)，其中 Y 是 target variable。
2. 按 MI score 排序 features。
3. 保留前 k 个 features。

这适用于 feature 与 target 之间的任何关系：线性、非线性、单调或非单调。Correlation 只能捕捉线性关系。MI 能捕捉一切统计依赖。

| 方法 | 检测什么 | 计算成本 | 支持 categorical？ |
|--------|---------|-------------------|---------------------|
| Pearson correlation | 线性关系 | O(n) | 否 |
| Spearman correlation | 单调关系 | O(n log n) | 否 |
| Mutual information | 任意统计依赖 | 带 binning 时 O(n log n) | 是 |

### Label Smoothing 和 Cross-Entropy

标准分类使用 hard targets：[0, 0, 1, 0]。真实类别得到概率 1，其他都是 0。Label smoothing 会把它们替换成 soft targets：

```
soft_target = (1 - epsilon) * hard_target + epsilon / num_classes
```

当 epsilon = 0.1 且有 4 个类别时：
- Hard target:  [0, 0, 1, 0]
- Soft target:  [0.025, 0.025, 0.925, 0.025]

从信息论角度看，label smoothing 增加了 target distribution 的 entropy。Hard one-hot targets 的 entropy 为 0：没有不确定性。Soft targets 有正 entropy。

为什么有帮助：
- 防止模型把 logits 推到极端值（要在 cross-entropy 下完美匹配 one-hot target，需要无穷大的 logits）
- 充当 regularization：模型不能 100% 自信
- 改善 calibration：预测概率更好地反映真实不确定性
- 减少训练行为和推理行为之间的差距

带 label smoothing 的 cross-entropy loss 变成：

```
L = (1 - epsilon) * CE(hard_target, prediction) + epsilon * H_uniform(prediction)
```

第二项会惩罚远离 uniform 的预测：这是对 confidence 的直接正则化。

### 为什么 Cross-Entropy 是分类 Loss

三个视角，同一个结论。

**信息论视角。** Cross-entropy 衡量如果用模型分布而不是真实分布，会浪费多少 bits。最小化它，会让模型成为现实的最高效编码器。

**Maximum likelihood 视角。** 对 N 个训练样本，其真实类别为 y_i：

```
Likelihood     = product( q(y_i) )
Log-likelihood = sum( log(q(y_i)) )
Negative log-likelihood = -sum( log(q(y_i)) )
```

最后一行就是 cross-entropy loss。最小化 cross-entropy = 最大化训练数据在模型下的 likelihood。

**Gradient 视角。** Cross-entropy 对 logits 的 gradient 简单地是（predicted - true）。干净、稳定、计算快速。这就是它与 softmax 完美配合的原因。

### Bits vs Nats

唯一差别是 log 的底数。

```
log base 2   -> bits      (information theory tradition)
log base e   -> nats      (machine learning convention)
log base 10  -> hartleys  (rarely used)
```

1 nat = 1/ln(2) bits = 1.4427 bits。PyTorch 和 TensorFlow 默认使用自然 log（nats）。

### Perplexity

Perplexity 是 cross-entropy 的指数。它告诉你模型平均相当于在多少个同等可能选择之间犹豫。

```
Perplexity = 2^H(P,Q)   (if using bits)
Perplexity = e^H(P,Q)   (if using nats)
```

Perplexity 为 50 的语言模型，平均来说就像是在从 50 个可能的 next tokens 中均匀选择。越低越好。

GPT-2 在常见 benchmark 上达到过约 30 的 perplexity。现代模型在代表性良好的领域中已经能达到个位数。

## 构建它

### 第 1 步：信息量和 entropy

```python
import math

def information_content(p, base=2):
    if p <= 0 or p > 1:
        return float('inf') if p <= 0 else 0.0
    return -math.log(p) / math.log(base)

def entropy(probs, base=2):
    return sum(
        p * information_content(p, base)
        for p in probs if p > 0
    )

fair_coin = [0.5, 0.5]
biased_coin = [0.99, 0.01]
fair_die = [1/6] * 6

print(f"Fair coin entropy:   {entropy(fair_coin):.4f} bits")
print(f"Biased coin entropy: {entropy(biased_coin):.4f} bits")
print(f"Fair die entropy:    {entropy(fair_die):.4f} bits")
```

### 第 2 步：Cross-entropy 和 KL divergence

```python
def cross_entropy(p, q, base=2):
    total = 0.0
    for pi, qi in zip(p, q):
        if pi > 0:
            if qi <= 0:
                return float('inf')
            total += pi * (-math.log(qi) / math.log(base))
    return total

def kl_divergence(p, q, base=2):
    return cross_entropy(p, q, base) - entropy(p, base)

true_dist = [0.7, 0.2, 0.1]
good_model = [0.6, 0.25, 0.15]
bad_model = [0.1, 0.1, 0.8]

print(f"Entropy of true dist:     {entropy(true_dist):.4f} bits")
print(f"CE (good model):          {cross_entropy(true_dist, good_model):.4f} bits")
print(f"CE (bad model):           {cross_entropy(true_dist, bad_model):.4f} bits")
print(f"KL divergence (good):     {kl_divergence(true_dist, good_model):.4f} bits")
print(f"KL divergence (bad):      {kl_divergence(true_dist, bad_model):.4f} bits")
```

### 第 3 步：Cross-entropy 作为分类 loss

```python
def softmax(logits):
    max_logit = max(logits)
    exps = [math.exp(z - max_logit) for z in logits]
    total = sum(exps)
    return [e / total for e in exps]

def cross_entropy_loss(true_class, logits):
    probs = softmax(logits)
    return -math.log(probs[true_class])

logits = [2.0, 1.0, 0.1]
true_class = 0

probs = softmax(logits)
loss = cross_entropy_loss(true_class, logits)

print(f"Logits:      {logits}")
print(f"Softmax:     {[f'{p:.4f}' for p in probs]}")
print(f"True class:  {true_class}")
print(f"Loss:        {loss:.4f} nats")
print(f"Perplexity:  {math.exp(loss):.2f}")
```

### 第 4 步：Cross-entropy 等于 negative log-likelihood

```python
import random

random.seed(42)

n_samples = 1000
n_classes = 3
true_labels = [random.randint(0, n_classes - 1) for _ in range(n_samples)]
model_logits = [[random.gauss(0, 1) for _ in range(n_classes)] for _ in range(n_samples)]

ce_loss = sum(
    cross_entropy_loss(label, logits)
    for label, logits in zip(true_labels, model_logits)
) / n_samples

nll = -sum(
    math.log(softmax(logits)[label])
    for label, logits in zip(true_labels, model_logits)
) / n_samples

print(f"Cross-entropy loss:      {ce_loss:.6f}")
print(f"Negative log-likelihood: {nll:.6f}")
print(f"Difference:              {abs(ce_loss - nll):.2e}")
```

### 第 5 步：Mutual information

```python
def mutual_information(joint_probs, base=2):
    rows = len(joint_probs)
    cols = len(joint_probs[0])

    margin_x = [sum(joint_probs[i][j] for j in range(cols)) for i in range(rows)]
    margin_y = [sum(joint_probs[i][j] for i in range(rows)) for j in range(cols)]

    mi = 0.0
    for i in range(rows):
        for j in range(cols):
            pxy = joint_probs[i][j]
            if pxy > 0:
                mi += pxy * math.log(pxy / (margin_x[i] * margin_y[j])) / math.log(base)
    return mi

independent = [[0.25, 0.25], [0.25, 0.25]]
dependent = [[0.45, 0.05], [0.05, 0.45]]

print(f"MI (independent): {mutual_information(independent):.4f} bits")
print(f"MI (dependent):   {mutual_information(dependent):.4f} bits")
```

## 使用它

使用 NumPy 实现同样的概念，这是实践中你会使用的方式：

```python
import numpy as np

def np_entropy(p):
    p = np.asarray(p, dtype=float)
    mask = p > 0
    result = np.zeros_like(p)
    result[mask] = p[mask] * np.log(p[mask])
    return -result.sum()

def np_cross_entropy(p, q):
    p, q = np.asarray(p, dtype=float), np.asarray(q, dtype=float)
    mask = p > 0
    return -(p[mask] * np.log(q[mask])).sum()

def np_kl_divergence(p, q):
    return np_cross_entropy(p, q) - np_entropy(p)

true = np.array([0.7, 0.2, 0.1])
pred = np.array([0.6, 0.25, 0.15])
print(f"Entropy:    {np_entropy(true):.4f} nats")
print(f"Cross-ent:  {np_cross_entropy(true, pred):.4f} nats")
print(f"KL div:     {np_kl_divergence(true, pred):.4f} nats")
```

你已经从零构建了 `torch.nn.CrossEntropyLoss()` 内部在做的事情。现在你知道为什么训练时 loss 会下降：模型预测分布正在接近真实分布，度量单位是浪费掉的信息 nats。

## 练习

1. 假设英文字母均匀分布（26 个字母），计算其 entropy。然后使用真实字母频率估计它。哪个更高，为什么？

2. 某模型对真实类别为 1 的样本输出 logits [5.0, 2.0, 0.5]。手算 cross-entropy loss，然后用你的 `cross_entropy_loss` 函数验证。什么 logits 会给出零 loss？

3. 证明 KL divergence 不对称。选择两个分布 P 和 Q，计算 D_KL(P || Q) 和 D_KL(Q || P)。解释它们为什么不同。

4. 构建一个函数，为一段 token predictions 计算 perplexity。给定一组 (true_token_index, predicted_logits) 对，返回这个序列的 perplexity。

## 关键术语

| 术语 | 人们常说 | 它实际意味着什么 |
|------|----------------|----------------------|
| 信息量 | “惊讶” | 编码某个事件所需的 bits（或 nats）数：-log(p) |
| Entropy | “随机性” | 分布所有结果的平均惊讶程度。衡量不可约不确定性。 |
| Cross-entropy | “Loss function” | 使用模型分布 Q 编码真实分布 P 中事件时的平均惊讶程度。 |
| KL divergence | “分布之间的距离” | 使用 Q 而不是 P 所浪费的额外 bits。等于 cross-entropy 减 entropy。不对称。 |
| Mutual information | “X 和 Y 有多相关” | 通过知道 Y，关于 X 的不确定性减少了多少。为零表示独立。 |
| Softmax | “把 logits 变成概率” | 指数化并归一化。把任意实数向量映射成有效概率分布。 |
| Perplexity | “模型有多困惑” | Cross-entropy 的指数。模型每一步有效选择的 vocabulary size。 |
| Bits | “Shannon 的单位” | 以 2 为底 log 衡量的信息。一个 bit 解决一次公平抛硬币。 |
| Nats | “ML 的单位” | 用自然 log 衡量的信息。PyTorch 和 TensorFlow 默认使用。 |
| Negative log-likelihood | “NLL loss” | 对 one-hot labels 来说，与 cross-entropy loss 相同。最小化它会最大化正确预测的概率。 |

## 延伸阅读

- [Shannon 1948: A Mathematical Theory of Communication](https://people.math.harvard.edu/~ctm/home/text/others/shannon/entropy/entropy.pdf) - 原始论文，今天仍然可读
- [Visual Information Theory (Chris Olah)](https://colah.github.io/posts/2015-09-Visual-Information/) - 对 entropy 和 KL divergence 最好的视觉解释
- [PyTorch CrossEntropyLoss docs](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html) - 框架如何实现你刚构建的内容
