# 概率与分布

> 概率是 AI 表达不确定性的语言。

**类型：** 学习
**语言：** Python
**前置要求：** 阶段 1，第 01-04 课
**时间：** ~75 分钟

## 学习目标

- 从零为 Bernoulli、categorical、Poisson、uniform 和 normal 分布实现 PMF 与 PDF
- 计算期望值、方差，并用 Central Limit Theorem 解释为什么 Gaussian 无处不在
- 使用数值稳定技巧（减去最大 logit）构建 softmax 和 log-softmax 函数
- 从 logits 计算 cross-entropy loss，并将它连接到 negative log-likelihood

## 问题

一个分类器输出 `[0.03, 0.91, 0.06]`。一个语言模型从 50,000 个候选词中选择下一个词。一个 diffusion model 通过从学到的分布中采样来生成图像。这些都是概率在发挥作用。

模型做出的每个预测都是一个概率分布。每个 loss function 都在衡量预测分布与真实分布之间的距离。每个训练步骤都在调整参数，让一个分布更像另一个分布。没有概率，你读不懂任何 ML 论文，调不动任何模型，也无法理解为什么训练 loss 会变成 NaN。

## 概念

### 事件、样本空间和概率

样本空间 S 是所有可能结果的集合。事件是样本空间的子集。概率把事件映射到 0 到 1 之间的数字。

```
Coin flip:
  S = {H, T}
  P(H) = 0.5,  P(T) = 0.5

Single die roll:
  S = {1, 2, 3, 4, 5, 6}
  P(even) = P({2, 4, 6}) = 3/6 = 0.5
```

三个公理定义了全部概率论：
1. 对任意事件 A，P(A) >= 0
2. P(S) = 1（总会发生某件事）
3. 当 A 和 B 不可能同时发生时，P(A or B) = P(A) + P(B)

其他所有内容（Bayes' theorem、期望、分布）都从这三条规则推出。

### 条件概率和独立性

P(A|B) 是在 B 已经发生的条件下 A 发生的概率。

```
P(A|B) = P(A and B) / P(B)

Example: deck of cards
  P(King | Face card) = P(King and Face card) / P(Face card)
                      = (4/52) / (12/52)
                      = 4/12 = 1/3
```

如果知道一个事件不会告诉你关于另一个事件的任何信息，这两个事件就是独立的：

```
Independent:   P(A|B) = P(A)
Equivalent to: P(A and B) = P(A) * P(B)
```

抛硬币是独立的。不放回抽牌不是。

### Probability Mass Functions vs Probability Density Functions

离散随机变量有 probability mass function（PMF）。每个结果都有一个可以直接读出的具体概率。

```
PMF: P(X = k)

Fair die:
  P(X = 1) = 1/6
  P(X = 2) = 1/6
  ...
  P(X = 6) = 1/6

  Sum of all probabilities = 1
```

连续随机变量有 probability density function（PDF）。单点处的密度不是概率。概率来自对某个区间上的密度积分。

```
PDF: f(x)

P(a <= X <= b) = integral of f(x) from a to b

f(x) can be greater than 1 (density, not probability)
integral from -inf to +inf of f(x) dx = 1
```

这个区别在 ML 中很重要。分类输出是 PMF（离散选择）。VAE 的 latent space 使用 PDF（连续）。

### 常见分布

**Bernoulli：** 一次试验，两个结果。建模二分类。

```
P(X = 1) = p
P(X = 0) = 1 - p
Mean = p,  Variance = p(1-p)
```

**Categorical：** 一次试验，k 个结果。建模多分类（softmax 输出）。

```
P(X = i) = p_i,  where sum of p_i = 1
Example: P(cat) = 0.7,  P(dog) = 0.2,  P(bird) = 0.1
```

**Uniform：** 所有结果等可能。用于随机初始化。

```
Discrete: P(X = k) = 1/n for k in {1, ..., n}
Continuous: f(x) = 1/(b-a) for x in [a, b]
```

**Normal（Gaussian）：** 钟形曲线。由均值（mu）和方差（sigma^2）参数化。

```
f(x) = (1 / sqrt(2*pi*sigma^2)) * exp(-(x - mu)^2 / (2*sigma^2))

Standard normal: mu = 0, sigma = 1
  68% of data within 1 sigma
  95% within 2 sigma
  99.7% within 3 sigma
```

**Poisson：** 固定区间内稀有事件的计数。建模事件发生率。

```
P(X = k) = (lambda^k * e^(-lambda)) / k!
Mean = lambda,  Variance = lambda
```

### 期望值和方差

期望值是按概率加权的平均结果。

```
Discrete:   E[X] = sum of x_i * P(X = x_i)
Continuous: E[X] = integral of x * f(x) dx
```

方差衡量围绕均值的离散程度。

```
Var(X) = E[(X - E[X])^2] = E[X^2] - (E[X])^2
Standard deviation = sqrt(Var(X))
```

在 ML 中，期望值会以 loss function 的形式出现（数据分布上的平均 loss）。方差告诉你模型稳定性。Gradient 的高方差意味着训练噪声大。

### 联合分布和边际分布

联合分布 P(X, Y) 同时描述两个随机变量。

联合 PMF 例子（X = 天气，Y = 雨伞）：

| | Y=0（无伞） | Y=1（有伞） | 边际 P(X) |
|---|---|---|---|
| X=0（晴） | 0.40 | 0.10 | P(X=0) = 0.50 |
| X=1（雨） | 0.05 | 0.45 | P(X=1) = 0.50 |
| **边际 P(Y)** | P(Y=0) = 0.45 | P(Y=1) = 0.55 | 1.00 |

边际分布会把另一个变量求和消掉：

```
P(X = x) = sum over all y of P(X = x, Y = y)
```

上表的行总和与列总和就是边际分布。

### 为什么正态分布到处出现

Central Limit Theorem：许多独立随机变量的和（或平均值）会收敛到正态分布，无论原始分布是什么。

```
Roll 1 die:  uniform distribution (flat)
Average of 2 dice:  triangular (peaked)
Average of 30 dice: nearly perfect bell curve

This works for ANY starting distribution.
```

这就是为什么：
- 测量误差近似正态（很多微小独立来源）
- 神经网络权重初始化使用正态分布
- SGD 中的 gradient noise 近似正态（许多样本 gradients 的和）
- 在给定均值和方差时，正态分布是 maximum entropy distribution

### Log Probabilities

原始概率会造成数值问题。许多小概率相乘，很快就会下溢为零。

```
P(sentence) = P(word1) * P(word2) * ... * P(word_n)
            = 0.01 * 0.003 * 0.02 * ...
            -> 0.0 (underflow after ~30 terms)
```

Log probabilities 可以解决这个问题。乘法变成加法。

```
log P(sentence) = log P(word1) + log P(word2) + ... + log P(word_n)
                = -4.6 + -5.8 + -3.9 + ...
                -> finite number (no underflow)
```

规则：
- log(a * b) = log(a) + log(b)
- log probabilities 总是 <= 0（因为 0 < P <= 1）
- 越负 = 越不可能
- Cross-entropy loss 是正确类别的负 log probability

### Softmax 作为概率分布

神经网络输出原始分数（logits）。Softmax 把它们转换成有效的概率分布。

```
softmax(z_i) = exp(z_i) / sum(exp(z_j) for all j)

Properties:
  - All outputs are in (0, 1)
  - All outputs sum to 1
  - Preserves relative ordering of inputs
  - exp() amplifies differences between logits
```

Softmax 技巧：指数化之前减去最大 logit，防止溢出。

```
z = [100, 101, 102]
exp(102) = overflow

z_shifted = z - max(z) = [-2, -1, 0]
exp(0) = 1  (safe)

Same result, no overflow.
```

Log-softmax 把 softmax 和 log 结合在一起，以获得数值稳定性。PyTorch 在 cross-entropy loss 内部使用它。

### 采样

采样意味着从分布中抽取随机值。在 ML 中：
- Dropout 随机采样要置零的 neurons
- Data augmentation 采样随机变换
- 语言模型从预测分布中采样下一个 token
- Diffusion models 采样噪声并逐步去噪

从任意分布采样需要 inverse transform sampling、rejection sampling 或 reparameterization trick（VAE 中使用）等技术。

## 构建它

### 第 1 步：概率基础

```python
import math
import random

def factorial(n):
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result

def combinations(n, k):
    return factorial(n) // (factorial(k) * factorial(n - k))

def conditional_probability(p_a_and_b, p_b):
    return p_a_and_b / p_b

p_king_given_face = conditional_probability(4/52, 12/52)
print(f"P(King | Face card) = {p_king_given_face:.4f}")
```

### 第 2 步：从零实现 PMF 和 PDF

```python
def bernoulli_pmf(k, p):
    return p if k == 1 else (1 - p)

def categorical_pmf(k, probs):
    return probs[k]

def poisson_pmf(k, lam):
    return (lam ** k) * math.exp(-lam) / factorial(k)

def uniform_pdf(x, a, b):
    if a <= x <= b:
        return 1.0 / (b - a)
    return 0.0

def normal_pdf(x, mu, sigma):
    coeff = 1.0 / (sigma * math.sqrt(2 * math.pi))
    exponent = -0.5 * ((x - mu) / sigma) ** 2
    return coeff * math.exp(exponent)
```

### 第 3 步：期望值和方差

```python
def expected_value(values, probabilities):
    return sum(v * p for v, p in zip(values, probabilities))

def variance(values, probabilities):
    mu = expected_value(values, probabilities)
    return sum(p * (v - mu) ** 2 for v, p in zip(values, probabilities))

die_values = [1, 2, 3, 4, 5, 6]
die_probs = [1/6] * 6
mu = expected_value(die_values, die_probs)
var = variance(die_values, die_probs)
print(f"Die: E[X] = {mu:.4f}, Var(X) = {var:.4f}, SD = {var**0.5:.4f}")
```

### 第 4 步：从分布中采样

```python
def sample_bernoulli(p, n=1):
    return [1 if random.random() < p else 0 for _ in range(n)]

def sample_categorical(probs, n=1):
    cumulative = []
    total = 0
    for p in probs:
        total += p
        cumulative.append(total)
    samples = []
    for _ in range(n):
        r = random.random()
        for i, c in enumerate(cumulative):
            if r <= c:
                samples.append(i)
                break
    return samples

def sample_normal_box_muller(mu, sigma, n=1):
    samples = []
    for _ in range(n):
        u1 = random.random()
        u2 = random.random()
        z = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
        samples.append(mu + sigma * z)
    return samples
```

### 第 5 步：Softmax 和 log probabilities

```python
def softmax(logits):
    max_logit = max(logits)
    shifted = [z - max_logit for z in logits]
    exps = [math.exp(z) for z in shifted]
    total = sum(exps)
    return [e / total for e in exps]

def log_softmax(logits):
    max_logit = max(logits)
    shifted = [z - max_logit for z in logits]
    log_sum_exp = max_logit + math.log(sum(math.exp(z) for z in shifted))
    return [z - log_sum_exp for z in logits]

def cross_entropy_loss(logits, target_index):
    log_probs = log_softmax(logits)
    return -log_probs[target_index]
```

### 第 6 步：Central Limit Theorem 演示

```python
def demonstrate_clt(dist_fn, n_samples, n_averages):
    averages = []
    for _ in range(n_averages):
        samples = [dist_fn() for _ in range(n_samples)]
        averages.append(sum(samples) / len(samples))
    return averages
```

### 第 7 步：可视化

```python
import matplotlib.pyplot as plt

xs = [mu + sigma * (i - 500) / 100 for i in range(1001)]
ys = [normal_pdf(x, mu, sigma) for x, mu, sigma in ...]
plt.plot(xs, ys)
```

包含全部可视化的完整实现位于 `code/probability.py`。

## 使用它

使用 NumPy 和 SciPy，上面所有内容都是一行代码：

```python
import numpy as np
from scipy import stats

normal = stats.norm(loc=0, scale=1)
samples = normal.rvs(size=10000)
print(f"Mean: {np.mean(samples):.4f}, Std: {np.std(samples):.4f}")
print(f"P(X < 1.96) = {normal.cdf(1.96):.4f}")

logits = np.array([2.0, 1.0, 0.1])
from scipy.special import softmax, log_softmax
probs = softmax(logits)
log_probs = log_softmax(logits)
print(f"Softmax: {probs}")
print(f"Log-softmax: {log_probs}")
```

你已经从零构建了这些东西。现在你知道库调用内部在做什么。

## 练习

1. 为指数分布实现 inverse transform sampling。采样 10,000 个值，并把 histogram 与真实 PDF 对比来验证。

2. 为两个有偏骰子构建 joint distribution table。计算 marginal distributions，并检查骰子是否独立。

3. 一个 5 类分类器在正确类别为索引 3 时输出 logits `[2.0, 0.5, -1.0, 3.0, 0.1]`。计算 cross-entropy loss。然后用 PyTorch 的 `nn.CrossEntropyLoss` 验证答案。

4. 写一个函数，接收 log probabilities 列表，并返回最可能序列、总 log probability，以及等价的原始概率。用一个 50 词句子测试，其中每个词的概率都是 0.01。

## 关键术语

| 术语 | 人们常说 | 它实际意味着什么 |
|------|----------------|----------------------|
| 样本空间 | “所有可能性” | 实验所有可能结果组成的集合 S |
| PMF | “概率函数” | 给出每个离散结果精确概率的函数，总和为 1 |
| PDF | “概率曲线” | 连续变量的密度函数。对区间积分才能得到概率 |
| 条件概率 | “给定某事的概率” | P(A\|B) = P(A and B) / P(B)。Bayesian thinking 和 Bayes' theorem 的基础 |
| 独立性 | “互不影响” | P(A and B) = P(A) * P(B)。知道一个事件不会告诉你另一个事件的信息 |
| 期望值 | “平均值” | 所有结果按概率加权求和。Loss function 是一个期望值 |
| 方差 | “有多分散” | 相对均值的平方偏差的期望。高方差 = 噪声大、不稳定的估计 |
| 正态分布 | “钟形曲线” | f(x) = (1/sqrt(2*pi*sigma^2)) * exp(-(x-mu)^2/(2*sigma^2))。由于 CLT，到处都会出现 |
| Central Limit Theorem | “平均值会变正态” | 许多独立样本的均值会收敛到正态分布，与来源分布无关 |
| 联合分布 | “两个变量放在一起” | P(X, Y) 描述 X 和 Y 每种结果组合的概率 |
| 边际分布 | “把另一个变量求和消掉” | P(X) = sum_y P(X, Y)。从联合分布恢复一个变量的分布 |
| Log probability | “在 log 空间工作” | log P(x)。把乘积变成求和，防止长序列中的数值下溢 |
| Softmax | “把分数变成概率” | softmax(z_i) = exp(z_i) / sum(exp(z_j))。把实数 logits 映射成有效概率分布 |
| Cross-entropy | “Loss function” | -sum(p_true * log(p_predicted))。衡量两个分布有多不同。越低越好 |
| Logits | “原始模型输出” | Softmax 之前的未归一化分数。名字来自 logistic function |
| Sampling | “抽随机值” | 按概率分布生成值。模型生成输出的方式 |

## 延伸阅读

- [3Blue1Brown: But what is the Central Limit Theorem?](https://www.youtube.com/watch?v=zeJD6dqJ5lo) - 为什么平均值会变成正态分布的视觉证明
- [Stanford CS229 Probability Review](https://cs229.stanford.edu/section/cs229-prob.pdf) - 覆盖本课和更多内容的简明参考
- [The Log-Sum-Exp Trick](https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/) - 为什么数值稳定性重要，以及如何实现它
