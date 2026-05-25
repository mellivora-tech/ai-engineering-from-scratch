# Transformers 之前的文本生成：N-gram 语言模型

> 如果一个词让模型感到“意外”，模型就不好。Perplexity 把这种意外变成数字。Smoothing 让它保持有限。

**类型：** 构建
**语言：** Python
**前置要求：** 阶段 5 · 01（文本处理），阶段 2 · 14（朴素贝叶斯）
**时间：** ~45 分钟

## 问题

在 transformers、RNN、word embeddings 之前，language model 通过数数来预测下一个词：某个词跟在前面 `n-1` 个词之后出现过多少次。数到 `"the cat"` → `"sat"` 47 次，`"the cat"` → `"jumped"` 12 次，`"the cat"` → `"refrigerator"` 0 次。归一化后得到一个概率分布。

这就是 n-gram language model。从 1980 年到 2015 年，它驱动了几乎所有语音识别器、拼写检查器和基于短语的机器翻译系统。今天当你需要便宜的端侧 language modeling 时，它仍然有用。

真正有意思的问题是：没见过的 n-gram 怎么办？一个原始的计数模型会给任何没见过的序列分配 0 概率，这很灾难，因为句子很长，几乎每个长句都会包含至少一个没见过的序列。五十年的 smoothing 研究修好了这个问题。Kneser-Ney smoothing 是其中的结果，现代 deep learning 也继承了这种重视实证的传统。

## 概念

![N-gram model: count, smooth, generate](../assets/ngram.svg)

**N-gram 概率：** `P(w_i | w_{i-n+1}, ..., w_{i-1})`。固定 `n`（trigram 通常是 3，4-gram 是 4）。从计数计算：

```text
P(w | context) = count(context, w) / count(context)
```

**零计数问题。** 训练中没见过的任何 n-gram 都会得到 0 概率。2007 年一项在 Brown corpus 上的研究发现，即使是 4-gram 模型，held-out 集合里也有 30% 的 4-gram 在训练中没出现过。不做 smoothing，你就无法在任何真实文本上评估。

**Smoothing 方法，按复杂度递增：**

1. **Laplace（add-one）。** 给每个计数加 1。简单，但对稀有事件很糟。
2. **Good-Turing。** 根据 frequency-of-frequencies，把高频事件的一部分概率质量重新分配给没见过的事件。
3. **Interpolation。** 用可调权重组合 n-gram、(n-1)-gram 等估计。
4. **Backoff。** 如果 n-gram 计数为 0，就退回到 (n-1)-gram。Katz backoff 会做归一化。
5. **Absolute discounting。** 从所有计数里减去固定折扣 `D`，再把释放的概率质量分配给没见过的事件。
6. **Kneser-Ney。** Absolute discounting 加上一个聪明的低阶模型：使用 continuation probability（一个词出现在多少种上下文里），而不是原始频率。

Kneser-Ney 的洞见很深。“San Francisco” 是常见 bigram。Unigram `"Francisco"` 主要出现在 `"San"` 之后。朴素的 absolute discounting 会给 `"Francisco"` 很高的 unigram 概率（因为计数高）。Kneser-Ney 注意到 `"Francisco"` 只出现在一个上下文里，于是相应降低它的 continuation probability。结果：一个新的、以 `"Francisco"` 结尾的 bigram 会得到合适的低概率。

**评估：perplexity。** 在 held-out 测试集上，每个词平均负对数似然的指数。越低越好。Perplexity 为 100 意味着模型的困惑程度相当于在 100 个词中均匀随机选择。

```text
perplexity = exp(- (1/N) * Σ log P(w_i | context_i))
```

## 构建它

### 第 1 步：trigram 计数

```python
from collections import Counter, defaultdict


def train_ngram(corpus_tokens, n=3):
    ngrams = Counter()
    contexts = Counter()
    for sentence in corpus_tokens:
        padded = ["<s>"] * (n - 1) + sentence + ["</s>"]
        for i in range(len(padded) - n + 1):
            ctx = tuple(padded[i:i + n - 1])
            word = padded[i + n - 1]
            ngrams[ctx + (word,)] += 1
            contexts[ctx] += 1
    return ngrams, contexts


def raw_probability(ngrams, contexts, context, word):
    ctx = tuple(context)
    if contexts.get(ctx, 0) == 0:
        return 0.0
    return ngrams.get(ctx + (word,), 0) / contexts[ctx]
```

输入是已经分词的句子列表。输出是 n-gram 计数和 context 计数。`<s>` 和 `</s>` 是句子边界。

### 第 2 步：Laplace smoothing

```python
def laplace_probability(ngrams, contexts, vocab_size, context, word):
    ctx = tuple(context)
    numerator = ngrams.get(ctx + (word,), 0) + 1
    denominator = contexts.get(ctx, 0) + vocab_size
    return numerator / denominator
```

给每个计数加 1。它能平滑，但会把过多概率质量分给没见过的事件，也会伤害已经见过的稀有事件。

### 第 3 步：Kneser-Ney（bigram，interpolated）

```python
def kneser_ney_bigram_model(corpus_tokens, discount=0.75):
    unigrams = Counter()
    bigrams = Counter()
    unigram_contexts = defaultdict(set)

    for sentence in corpus_tokens:
        padded = ["<s>"] + sentence + ["</s>"]
        for i, w in enumerate(padded):
            unigrams[w] += 1
            if i > 0:
                prev = padded[i - 1]
                bigrams[(prev, w)] += 1
                unigram_contexts[w].add(prev)

    total_unique_bigrams = sum(len(ctx_set) for ctx_set in unigram_contexts.values())
    continuation_prob = {
        w: len(ctx_set) / total_unique_bigrams for w, ctx_set in unigram_contexts.items()
    }

    context_totals = Counter()
    for (prev, w), count in bigrams.items():
        context_totals[prev] += count

    unique_follow = defaultdict(set)
    for (prev, w) in bigrams:
        unique_follow[prev].add(w)

    def prob(prev, w):
        count = bigrams.get((prev, w), 0)
        denom = context_totals.get(prev, 0)
        if denom == 0:
            return continuation_prob.get(w, 1e-9)
        first_term = max(count - discount, 0) / denom
        lambda_prev = discount * len(unique_follow[prev]) / denom
        return first_term + lambda_prev * continuation_prob.get(w, 1e-9)

    return prob
```

这里有三个活动部件。`continuation_prob` 捕捉“这个词出现在多少种不同上下文里？”（Kneser-Ney 的创新）。`lambda_prev` 是折扣释放出来的概率质量，用来给 backoff 加权。最终概率是折扣后的主项，加上加权的 continuation 项。

### 第 4 步：用 sampling 生成文本

```python
import random


def generate(prob_fn, vocab, prefix, max_len=30, seed=0):
    rng = random.Random(seed)
    tokens = list(prefix)
    for _ in range(max_len):
        candidates = [(w, prob_fn(tokens[-1], w)) for w in vocab]
        total = sum(p for _, p in candidates)
        r = rng.random() * total
        acc = 0.0
        for w, p in candidates:
            acc += p
            if r <= acc:
                tokens.append(w)
                break
        if tokens[-1] == "</s>":
            break
    return tokens
```

按概率比例 sampling。每个 seed 都会给出不同输出。如果想要 beam-search 风格的输出，可以在每一步选择 argmax（greedy），再加一个小的随机性旋钮（temperature）。

### 第 5 步：perplexity

```python
import math


def perplexity(prob_fn, sentences):
    total_log_prob = 0.0
    total_tokens = 0
    for sentence in sentences:
        padded = ["<s>"] + sentence + ["</s>"]
        for i in range(1, len(padded)):
            p = prob_fn(padded[i - 1], padded[i])
            total_log_prob += math.log(max(p, 1e-12))
            total_tokens += 1
    return math.exp(-total_log_prob / total_tokens)
```

越低越好。在 Brown corpus 上，一个调得不错的 4-gram KN 模型 perplexity 大约 140。Transformer LM 在同一测试集上能到 15-30。差距大约 10 倍。这就是领域继续向前走的原因。

## 使用它

- **经典 NLP 教学。** 这是理解 smoothing、MLE 和 perplexity 最清楚的入口。
- **KenLM。** 生产级 n-gram 库。在语音和 MT 系统里作为 rescorer 使用，尤其是在低延迟重要时。
- **端侧 autocomplete。** 键盘里的 trigram 模型。现在仍然存在。
- **Baselines。** 在宣布你的 neural LM 很好之前，先算一个 n-gram LM perplexity。如果 transformer 没有大幅击败 KN，某处一定有问题。

## 交付它

保存为 `outputs/prompt-lm-baseline.md`：

```markdown
---
name: lm-baseline
description: 在训练 neural LM 之前，构建可复现的 n-gram language model baseline。
phase: 5
lesson: 16
---

给定一个 corpus 和目标用途（next-word prediction、rescoring、perplexity baseline），输出：

1. N-gram 阶数。通用英语用 trigram，corpus 很大用 4-gram，speech rescoring 用 5-gram。
2. Smoothing。默认使用 Modified Kneser-Ney；Laplace 只用于教学。
3. Library。生产用 `kenlm`，教学用 `nltk.lm`，只有学习时才自己实现。
4. Evaluation。在 train/test tokenization 一致的 held-out 集上计算 perplexity。

拒绝报告在不同 tokenization 下比较的 perplexity。Perplexity 数字只有在完全相同 tokenization 下才可比。标记 test set 的 OOV rate；除非训练时保留特殊 <UNK> token，否则 KN 对 OOV 处理很差。
```

## 练习

1. **简单。** 在 1,000 句 Shakespeare corpus 上训练 trigram LM。生成 20 个句子。它们会局部像样，但整体不连贯。这是经典 demo。
2. **中等。** 在 held-out Shakespeare split 上为你的 KN 模型实现 perplexity。和 Laplace 比较。你应该看到 KN 的 perplexity 低 30-50%。
3. **困难。** 构建一个 trigram 拼写纠错器：给定拼错词和上下文，生成候选修正，并按 LM 下的上下文概率排序。在 Birkbeck spelling corpus（公开）上评估。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|-----------------|-----------------------|
| N-gram | 词序列 | `n` 个连续 token 的序列。 |
| Smoothing | 避免零概率 | 重新分配概率质量，让没见过的事件也有非零概率。 |
| Perplexity | LM 质量指标 | held-out 数据上的 `exp(-average log-prob)`。越低越好。 |
| Backoff | 退回短上下文 | 如果 trigram 计数为零，就用 bigram。Katz backoff 将其形式化。 |
| Kneser-Ney | n-gram 最好的 smoothing | Absolute discounting + 低阶模型的 continuation probability。 |
| Continuation probability | KN 专用 | `P(w)` 按 `w` 出现的上下文数量加权，而不是按原始计数。 |

## 延伸阅读

- [Jurafsky and Martin — Speech and Language Processing, Chapter 3 (2026 draft)](https://web.stanford.edu/~jurafsky/slp3/3.pdf) — n-gram LM 和 smoothing 的经典处理。
- [Chen and Goodman (1998). An Empirical Study of Smoothing Techniques for Language Modeling](https://dash.harvard.edu/handle/1/25104739) — 确立 Kneser-Ney 为最佳 n-gram smoother 的论文。
- [Kneser and Ney (1995). Improved Backing-off for M-gram Language Modeling](https://ieeexplore.ieee.org/document/479394) — 原始 KN 论文。
- [KenLM](https://kheafield.com/code/kenlm/) — 快速生产级 n-gram LM，2026 年仍用于延迟敏感应用。
