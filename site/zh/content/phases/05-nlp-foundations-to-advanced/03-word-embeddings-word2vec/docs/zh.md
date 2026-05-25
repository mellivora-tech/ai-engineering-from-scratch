# Word Embeddings：从零实现 Word2Vec

> 看一个词常和谁在一起，就能知道它是谁。把这个想法交给浅层网络训练，几何结构就会浮现。

**类型：** 构建
**语言：** Python
**前置要求：** 阶段 5 第 02 课（BoW + TF-IDF）、阶段 3 第 03 课（Backpropagation from Scratch）
**时间：** ~75 分钟

## 问题

TF-IDF 知道 `dog` 和 `puppy` 是不同的词。它不知道二者含义几乎相同。一个只在 `dog` 上训练过的分类器，不能泛化到关于 `puppy` 的评论。你可以列同义词来掩盖这个问题，但它会在稀有术语、领域黑话，以及所有你没预料到的语言上失败。

你想要一种表示：`dog` 和 `puppy` 在空间中靠得很近；`king - man + woman` 会落在 `queen` 附近；在 `dog` 上训练的模型能免费把部分信号迁移到 `puppy`。

Word2Vec 给了我们这个空间。两层神经网络、万亿 token 训练、2013 年发表。架构简单到几乎令人不好意思。结果却重塑了 NLP 十年。

## 概念

**分布式假说**（Firth, 1957）：“You shall know a word by the company it keeps.” 如果两个词出现在相似上下文中，它们大概含义相近。

Word2Vec 有两种形式，都利用了这个想法。

- **Skip-gram。** 给定中心词，预测周围词。窗口大小为 2 时，`cat -> (the, sat, on)`。
- **CBOW（continuous bag of words）。** 给定周围词，预测中心词。`(the, sat, on) -> cat`。

Skip-gram 训练更慢，但对稀有词更好。它成了默认选择。

这个网络有一个隐藏层，没有非线性。输入是词表上的 one-hot vector。输出是词表上的 softmax。训练结束后，丢掉输出层。隐藏层权重就是 embeddings。

```
one-hot(center) ── W ──▶ hidden (d-dim) ── W' ──▶ softmax(vocab)
                          ^
                          this is the embedding
```

技巧在于：在 10 万词上做 softmax 代价太高。Word2Vec 使用 **negative sampling** 把它变成二分类任务。预测“这个 context word 是否出现在这个 center word 附近，是还是否”。每个训练对只采样少量负例（没有共现的词），而不是在整个词表上计算 softmax。

## 构建它

### 第 1 步：从语料生成训练对

```python
def skipgram_pairs(docs, window=2):
    pairs = []
    for doc in docs:
        for i, center in enumerate(doc):
            for j in range(max(0, i - window), min(len(doc), i + window + 1)):
                if i == j:
                    continue
                pairs.append((center, doc[j]))
    return pairs
```

```python
>>> skipgram_pairs([["the", "cat", "sat", "on", "mat"]], window=2)
[('the', 'cat'), ('the', 'sat'),
 ('cat', 'the'), ('cat', 'sat'), ('cat', 'on'),
 ('sat', 'the'), ('sat', 'cat'), ('sat', 'on'), ('sat', 'mat'),
 ...]
```

窗口内每个 `(center, context)` 对都是一个正训练样本。

### 第 2 步：embedding tables

两个矩阵。`W` 是中心词 embedding table（你最终保留的那个）。`W'` 是上下文词 table（通常丢弃，有时和 `W` 平均）。

```python
import numpy as np


def init_embeddings(vocab_size, dim, seed=0):
    rng = np.random.default_rng(seed)
    W = rng.normal(0, 0.1, size=(vocab_size, dim))
    W_prime = rng.normal(0, 0.1, size=(vocab_size, dim))
    return W, W_prime
```

小随机初始化。词表 10k、维度 100 是现实规模；教学时 50 个词、16 维就足够看见几何效果。

### 第 3 步：negative sampling objective

对每个正例 `(center, context)`，从词表随机采样 `k` 个词作为负例。训练模型，让正例的点积 `W[center] · W'[context]` 变高，让负例的点积变低。

```python
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


def train_pair(W, W_prime, center_idx, context_idx, negative_indices, lr):
    v_c = W[center_idx]
    u_pos = W_prime[context_idx]
    u_negs = W_prime[negative_indices]

    pos_score = sigmoid(v_c @ u_pos)
    neg_scores = sigmoid(u_negs @ v_c)

    grad_center = (pos_score - 1) * u_pos
    for i, u in enumerate(u_negs):
        grad_center += neg_scores[i] * u

    W[context_idx] = W[context_idx]
    W_prime[context_idx] -= lr * (pos_score - 1) * v_c
    for i, neg_idx in enumerate(negative_indices):
        W_prime[neg_idx] -= lr * neg_scores[i] * v_c
    W[center_idx] -= lr * grad_center
```

核心公式：正例上做 logistic loss（希望 sigmoid 接近 1），负例上做 logistic loss（希望 sigmoid 接近 0）。梯度会流向两个 table。完整推导在原论文里；如果你想真正记住，拿纸笔走一遍。

### 第 4 步：在玩具语料上训练

```python
def train(docs, dim=16, window=2, k_neg=5, epochs=100, lr=0.05, seed=0):
    vocab = build_vocab(docs)
    vocab_size = len(vocab)
    rng = np.random.default_rng(seed)
    W, W_prime = init_embeddings(vocab_size, dim, seed=seed)
    pairs = skipgram_pairs(docs, window=window)

    for epoch in range(epochs):
        rng.shuffle(pairs)
        for center, context in pairs:
            c_idx = vocab[center]
            ctx_idx = vocab[context]
            negs = rng.integers(0, vocab_size, size=k_neg)
            negs = [n for n in negs if n != ctx_idx and n != c_idx]
            train_pair(W, W_prime, c_idx, ctx_idx, negs, lr)
    return vocab, W
```

在大语料上跑足够多 epoch 后，共享上下文的词会得到相似的中心词 embeddings。在玩具语料上，你只能隐约看到这个效果。在数十亿 token 上，它会非常明显。

### 第 5 步：analogy trick

```python
def nearest(vocab, W, target_vec, topk=5, exclude=None):
    exclude = exclude or set()
    inv_vocab = {i: w for w, i in vocab.items()}
    norms = np.linalg.norm(W, axis=1, keepdims=True) + 1e-9
    W_norm = W / norms
    target = target_vec / (np.linalg.norm(target_vec) + 1e-9)
    sims = W_norm @ target
    order = np.argsort(-sims)
    out = []
    for i in order:
        if i in exclude:
            continue
        out.append((inv_vocab[i], float(sims[i])))
        if len(out) == topk:
            break
    return out


def analogy(vocab, W, a, b, c, topk=5):
    v = W[vocab[b]] - W[vocab[a]] + W[vocab[c]]
    return nearest(vocab, W, v, topk=topk, exclude={vocab[a], vocab[b], vocab[c]})
```

在预训练的 300d Google News 向量上：

```python
>>> analogy(vocab, W, "man", "king", "woman")
[('queen', 0.71), ('monarch', 0.62), ('princess', 0.59), ...]
```

`king - man + woman = queen`。不是因为模型知道王室是什么。而是因为向量 `(king - man)` 捕获了类似“royal”的方向，把它加到 `woman` 上，就落到了 royal-female 区域附近。

## 使用它

从零写 Word2Vec 是为了教学。生产 NLP 使用 `gensim`。

```python
from gensim.models import Word2Vec

sentences = [
    ["the", "cat", "sat", "on", "the", "mat"],
    ["the", "dog", "ran", "across", "the", "room"],
]

model = Word2Vec(
    sentences,
    vector_size=100,
    window=5,
    min_count=1,
    sg=1,
    negative=5,
    workers=4,
    epochs=30,
)

print(model.wv["cat"])
print(model.wv.most_similar("cat", topn=3))
```

真实工作中，你几乎从不自己训练 Word2Vec。你会下载预训练向量。

- **GloVe** — Stanford 的共现矩阵分解方法。50d、100d、200d、300d checkpoint。通用覆盖很好。第 04 课专门讲 GloVe。
- **fastText** — Facebook 对 Word2Vec 的扩展，会嵌入 character n-grams。通过组合 subwords 处理 out-of-vocabulary 词。第 04 课。
- **Google News 上的预训练 Word2Vec** — 300d、300 万词词表，2013 年发布。今天仍然每天有人下载。

### 2026 年 Word2Vec 仍然赢的场景

- 轻量级领域检索。在笔记本上用一小时训练医学摘要，就能得到通用模型捕获不到的专用向量。
- analogy 风格的特征工程。`gender_vector = mean(man - woman pairs)`。从其他词中减掉它，得到性别中性轴。公平性研究里仍然会用。
- 可解释性。100d 小到可以用 PCA 或 t-SNE 画出来，并且真的看见 cluster 形成。
- 任何必须在无 GPU 设备端运行推理的地方。Word2Vec lookup 就是取一行。

### Word2Vec 失败的地方

一词多义墙。`bank` 只有一个向量。`river bank` 和 `financial bank` 共用它。`table`（电子表格 vs 家具）也共用它。下游分类器无法只从这个向量区分语义。

Contextual embeddings（ELMo、BERT，以及之后的每个 transformer）通过根据周围上下文为每次词出现生成不同向量，解决了这个问题。这就是从 Word2Vec 到 BERT 的跃迁：从 static 到 contextual。阶段 7 会覆盖 transformer 这半边。

另一个失败是 out-of-vocabulary 问题。如果训练数据里没见过 `Zoomer-approved`，Word2Vec 就没有办法。没有 fallback。fastText 用 subword composition 修复了这个问题（第 04 课）。

## 交付它

保存为 `outputs/skill-embedding-probe.md`：

```markdown
---
name: embedding-probe
description: 检查 word2vec 模型。运行 analogies、查找近邻、诊断质量。
version: 1.0.0
phase: 5
lesson: 03
tags: [nlp, embeddings, debugging]
---

你负责探测训练好的 word embeddings，验证它们是否正常工作。给定一个 `gensim.models.KeyedVectors` 对象和词表，你运行：

1. 三个经典 analogy 测试。`king : man :: queen : woman`。`paris : france :: tokyo : japan`。`walking : walked :: swimming : ?`。报告 top-1 结果及其 cosine。
2. 对用户提供的领域词运行五个 nearest-neighbor 测试。打印 top-5 邻居和 cosines。
3. 一个对称性检查。`similarity(a, b) == similarity(b, a)`，允许 float 精度误差。
4. 一个退化检查。如果任何 embedding 的 norm 低于 0.01 或高于 100，模型存在训练 bug。标记它。

拒绝只根据 analogy accuracy 声称模型良好。Analogy benchmark 很容易被刷分，也不能迁移到下游任务。建议同时做 intrinsic evaluation 和 downstream evaluation。
```

## 练习

1. **简单。** 在一个小语料（20 个关于猫和狗的句子）上运行训练循环。200 个 epoch 后，验证 `nearest(vocab, W, W[vocab["cat"]])` 的前 3 个结果中包含 `dog`。如果没有，增加 epoch 或词表。
2. **中等。** 添加高频词 subsampling。频率高于 `10^-5` 的词，会按其频率比例从训练对中丢弃。衡量它对稀有词相似度的影响。
3. **困难。** 在 20 Newsgroups 语料上训练模型。计算两个 bias axes：`he - she` 和 `doctor - nurse`。把职业词投影到两个轴上。报告哪些职业有最大的 bias gap。这是公平性研究者会使用的 probe。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|-----------------|-----------------------|
| Word embedding | 把词表示成向量 | 从上下文中学习到的 dense、低维（通常 100-300）表示。 |
| Skip-gram | Word2Vec 技巧 | 从中心词预测上下文词。比 CBOW 慢，但更适合稀有词。 |
| Negative sampling | 训练捷径 | 用针对 `k` 个随机词的二分类替代全词表 softmax。 |
| Static embedding | 每个词一个向量 | 不管上下文如何，同一个词总是同一向量。在一词多义上失败。 |
| Contextual embedding | 上下文敏感向量 | 基于周围词，为每次出现生成不同向量。Transformer 产出的就是这个。 |
| OOV | Out of vocabulary | 训练中没见过的词。Word2Vec 无法为这些词生成向量。 |

## 延伸阅读

- [Mikolov et al. (2013). Distributed Representations of Words and Phrases and their Compositionality](https://arxiv.org/abs/1310.4546) — negative-sampling 论文。短且易读。
- [Rong, X. (2014). word2vec Parameter Learning Explained](https://arxiv.org/abs/1411.2738) — 如果原论文数学太密，这是最清晰的梯度推导。
- [gensim Word2Vec tutorial](https://radimrehurek.com/gensim/models/word2vec.html) — 真正有效的生产训练设置。
