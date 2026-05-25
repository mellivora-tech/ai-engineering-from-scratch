# GloVe、FastText 与 Subword Embeddings

> Word2Vec 为每个词训练一个 embedding。GloVe 分解共现矩阵。FastText 嵌入词的组成部分。BPE 则架起了通往 transformers 的桥。

**类型：** 构建
**语言：** Python
**前置要求：** 阶段 5 第 03 课（Word2Vec from Scratch）
**时间：** ~45 分钟

## 问题

Word2Vec 留下了两个开放问题。

第一，有一条并行研究路线会直接分解共现矩阵（LSA、HAL），而不是做在线 skip-gram 更新。Word2Vec 的迭代方法本质上更好吗？还是差异只是两种方法处理计数方式带来的假象？**GloVe** 回答了这个问题：用精心选择的 loss 做矩阵分解，可以匹配甚至超过 Word2Vec，而且训练成本更低。

第二，二者都没有很好地处理从未见过的词。`Zoomer-approved`、`dogecoin`、上周刚造出来的任何专有名词、罕见词根的每一种屈折形式。**FastText** 通过嵌入 character n-grams 修复了这个问题：一个词是其组成部分之和，包括 morphemes，所以即使是 out-of-vocabulary 词也能得到合理向量。

第三，transformers 出现后，问题又发生了变化。词级词表最多扩到大约一百万个条目；真实语言比这开放得多。**Byte-pair encoding（BPE）** 及其亲戚通过学习频繁 subword 单元的词表解决了这个问题，可以覆盖一切。每个现代 LLM 的每个现代 tokenizer 都是 subword tokenizer。

本课会走过这三者，然后解释什么时候该选哪个。

## 概念

**GloVe（Global Vectors）。** 构建 word-word 共现矩阵 `X`，其中 `X[i][j]` 表示词 `j` 出现在词 `i` 上下文中的次数。训练向量，使 `v_i · v_j + b_i + b_j ≈ log(X[i][j])`。对 loss 加权，避免高频词对主导训练。完成。

**FastText。** 一个词是其 character n-grams 加上词本身的总和。`where` 会变成 `<wh, whe, her, ere, re>, <where>`。词向量是这些组件向量之和。训练方式和 Word2Vec 一样。收益：未见过的词（`whereupon`）可以由已知 n-grams 组合出来。

**BPE（Byte-Pair Encoding）。** 从单个 byte（或字符）组成的词表开始。统计语料中每个相邻 pair。把最频繁的 pair 合并成新 token。重复 `k` 次。结果是一个包含 `k + 256` 个 token 的词表，其中频繁序列（`ing`、`tion`、`the`）是单个 token，稀有词会被拆成熟悉片段。任何句子都能 tokenize 成某种东西。

## 构建它

### GloVe：分解共现矩阵

```python
import numpy as np
from collections import Counter


def build_cooccurrence(docs, window=5):
    pair_counts = Counter()
    vocab = {}
    for doc in docs:
        for token in doc:
            if token not in vocab:
                vocab[token] = len(vocab)
    for doc in docs:
        indexed = [vocab[t] for t in doc]
        for i, center in enumerate(indexed):
            for j in range(max(0, i - window), min(len(indexed), i + window + 1)):
                if i != j:
                    distance = abs(i - j)
                    pair_counts[(center, indexed[j])] += 1.0 / distance
    return vocab, pair_counts


def glove_train(vocab, pair_counts, dim=16, epochs=100, lr=0.05, x_max=100, alpha=0.75, seed=0):
    n = len(vocab)
    rng = np.random.default_rng(seed)
    W = rng.normal(0, 0.1, size=(n, dim))
    W_tilde = rng.normal(0, 0.1, size=(n, dim))
    b = np.zeros(n)
    b_tilde = np.zeros(n)

    for epoch in range(epochs):
        for (i, j), x_ij in pair_counts.items():
            weight = (x_ij / x_max) ** alpha if x_ij < x_max else 1.0
            diff = W[i] @ W_tilde[j] + b[i] + b_tilde[j] - np.log(x_ij)
            coef = weight * diff

            grad_W_i = coef * W_tilde[j]
            grad_W_tilde_j = coef * W[i]
            W[i] -= lr * grad_W_i
            W_tilde[j] -= lr * grad_W_tilde_j
            b[i] -= lr * coef
            b_tilde[j] -= lr * coef

    return W + W_tilde
```

这里有两个值得点名的活动部件。加权函数 `f(x) = (x/x_max)^alpha` 会降低非常高频 pair（比如 `(the, and)`）的权重，避免它们主导 loss。最终 embedding 是 `W`（center）和 `W_tilde`（context）两个 table 的和。把两者相加是论文里发布过的技巧，通常比只用其中一个更好。

### FastText：subword-aware embeddings

```python
def char_ngrams(word, n_min=3, n_max=6):
    wrapped = f"<{word}>"
    grams = {wrapped}
    for n in range(n_min, n_max + 1):
        for i in range(len(wrapped) - n + 1):
            grams.add(wrapped[i:i + n])
    return grams
```

```python
>>> char_ngrams("where")
{'<where>', '<wh', 'whe', 'her', 'ere', 're>', '<whe', 'wher', 'here', 'ere>', '<wher', 'where', 'here>'}
```

每个词由它的 n-grams 集合表示（通常是 3 到 6 个字符）。词 embedding 是其 n-gram embeddings 的和。做 skip-gram 训练时，把它接到 Word2Vec 原本使用单个向量的位置即可。

```python
def fasttext_vector(word, ngram_table):
    grams = char_ngrams(word)
    vecs = [ngram_table[g] for g in grams if g in ngram_table]
    if not vecs:
        return None
    return np.sum(vecs, axis=0)
```

对一个未见过的词，只要它的一部分 n-grams 已知，你仍然可以得到向量。`whereupon` 和 `where` 共享 `<wh`、`her`、`ere`、`<where`，所以二者会落得很近。

### BPE：学习出来的 subword 词表

```python
def learn_bpe(corpus, k_merges):
    vocab = Counter()
    for word, freq in corpus.items():
        tokens = tuple(word) + ("</w>",)
        vocab[tokens] = freq

    merges = []
    for _ in range(k_merges):
        pair_freq = Counter()
        for tokens, freq in vocab.items():
            for a, b in zip(tokens, tokens[1:]):
                pair_freq[(a, b)] += freq
        if not pair_freq:
            break
        best = pair_freq.most_common(1)[0][0]
        merges.append(best)

        new_vocab = Counter()
        for tokens, freq in vocab.items():
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i + 1 < len(tokens) and (tokens[i], tokens[i + 1]) == best:
                    new_tokens.append(tokens[i] + tokens[i + 1])
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            new_vocab[tuple(new_tokens)] = freq
        vocab = new_vocab
    return merges


def apply_bpe(word, merges):
    tokens = list(word) + ["</w>"]
    for a, b in merges:
        new_tokens = []
        i = 0
        while i < len(tokens):
            if i + 1 < len(tokens) and tokens[i] == a and tokens[i + 1] == b:
                new_tokens.append(a + b)
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        tokens = new_tokens
    return tokens
```

```python
>>> corpus = Counter({"low": 5, "lower": 2, "newest": 6, "widest": 3})
>>> merges = learn_bpe(corpus, k_merges=10)
>>> apply_bpe("lowest", merges)
['low', 'est</w>']
```

第一轮会合并最常见的相邻 pair。迭代足够多次后，频繁子串（`low`、`est`、`tion`）会成为单个 token，稀有词也能被干净拆开。

真正的 GPT / BERT / T5 tokenizer 会学习 30k-100k 个 merges。结果是：任何文本都能 tokenize 成长度有界的已知 ID 序列，永远没有 OOV。

## 使用它

实践中，你很少自己训练这些东西。你会加载预训练 checkpoint。

```python
import fasttext.util
fasttext.util.download_model("en", if_exists="ignore")
ft = fasttext.load_model("cc.en.300.bin")
print(ft.get_word_vector("whereupon").shape)
print(ft.get_word_vector("zoomerapproved").shape)
```

Transformer 时代的 BPE 风格 subword tokenization：

```python
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("gpt2")
print(tok.tokenize("unbelievably tokenized"))
```

```
['un', 'bel', 'iev', 'ably', 'Ġtoken', 'ized']
```

`Ġ` 前缀标记词边界（GPT-2 约定）。每个现代 tokenizer 都是 BPE 变体、WordPiece（BERT）或 SentencePiece（T5、LLaMA）。

### 什么时候选哪个

| 场景 | 选择 |
|-----------|------|
| 通用预训练 word vectors，不需要 OOV 容忍 | GloVe 300d |
| 通用预训练 word vectors，必须处理拼写错误 / 新词 / 形态丰富语言 | FastText |
| 任何送进 transformer 的东西（训练或推理） | 模型自带的 tokenizer。永远不要替换。 |
| 从零训练自己的 language model | 先在语料上训练 BPE 或 SentencePiece tokenizer |
| 用线性模型做生产文本分类 | 仍然是 TF-IDF。见第 02 课。 |

## 交付它

保存为 `outputs/skill-embeddings-picker.md`：

```markdown
---
name: tokenizer-picker
description: 为新的 language model 或 text pipeline 选择 tokenization 方法。
version: 1.0.0
phase: 5
lesson: 04
tags: [nlp, tokenization, embeddings]
---

给定任务和数据集描述后，输出：

1. Tokenization 策略（word-level、BPE、WordPiece、SentencePiece、byte-level）。用一句话说明原因。
2. 目标词表大小（例如英文单语 LM 用 32k，多语言用 64k-100k）。
3. 带精确训练命令的库调用。写出库名。引用参数。
4. 一个可复现性陷阱。Tokenizer-model mismatch 是最常见的静默生产 bug；指出哪一对必须一起使用。

如果用户是在 fine-tuning 预训练 LLM，且可以接受原 tokenizer，拒绝推荐训练自定义 tokenizer。拒绝为任何面向生产推理的模型推荐 word-level tokenization。把非英语 / 多文字体系语料标记为需要带 byte fallback 的 SentencePiece。
```

## 练习

1. **简单。** 运行 `char_ngrams("playing")` 和 `char_ngrams("played")`。计算两个 n-gram 集合的 Jaccard overlap。你应该会看到大量共享片段（`pla`、`lay`、`play`），这就是为什么 FastText 能在形态变体之间很好地迁移。
2. **中等。** 扩展 `learn_bpe`，追踪词表增长。画出 tokens-per-corpus-character 随 merge 数变化的曲线。你应该会看到一开始压缩很快，随后在每 token 约 2-3 个字符附近渐近。
3. **困难。** 在莎士比亚全集上训练 1k-merge BPE。比较常见词和稀有专有名词的 tokenization。衡量前后的平均每词 token 数。写下让你惊讶的发现。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|-----------------|-----------------------|
| Co-occurrence matrix | 词-词频率表 | `X[i][j]` = 词 `j` 出现在词 `i` 周围窗口中的次数。 |
| Subword | 词的一部分 | Character n-gram（FastText）或学习出来的 token（BPE/WordPiece/SentencePiece）。 |
| BPE | Byte-pair encoding | 迭代合并最高频相邻 pair，直到词表达到目标大小。 |
| OOV | Out of vocabulary | 模型从没见过的词。Word2Vec/GloVe 会失败。FastText 和 BPE 能处理。 |
| Byte-level BPE | 在原始 bytes 上做 BPE | GPT-2 的方案。词表从 256 个 byte 开始，所以永远不会 OOV。 |

## 延伸阅读

- [Pennington, Socher, Manning (2014). GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/pubs/glove.pdf) — GloVe 论文，七页，至今仍是 loss 最好的推导。
- [Bojanowski et al. (2017). Enriching Word Vectors with Subword Information](https://arxiv.org/abs/1607.04606) — FastText。
- [Sennrich, Haddow, Birch (2016). Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909) — 把 BPE 引入现代 NLP 的论文。
- [Hugging Face tokenizer summary](https://huggingface.co/docs/transformers/tokenizer_summary) — BPE、WordPiece、SentencePiece 在实践中到底有什么不同。
