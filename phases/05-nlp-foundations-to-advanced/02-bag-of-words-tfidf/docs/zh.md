# Bag of Words、TF-IDF 与文本表示

> 先计数，再思考。到了 2026 年，TF-IDF 在定义清晰的任务上仍然能打败 embeddings。

**类型：** 构建
**语言：** Python
**前置要求：** 阶段 5 第 01 课（Text Processing）、阶段 2 第 02 课（Linear Regression from Scratch）
**时间：** ~75 分钟

## 问题

模型需要数字。你手里是字符串。

每条 NLP pipeline 都必须回答同一个问题：怎样把可变长度的 token 流转换成分类器能消费的固定大小向量。这个领域最先落地的答案，就是最笨但有效的办法：数单词，做向量。

这个向量承载过的生产 NLP，比任何 embedding model 都多。垃圾邮件过滤、主题分类、日志异常检测、搜索排序（在 BM25 之前）、第一波情感分析、学术 NLP benchmark 的第一个十年。到了 2026 年，实践者在窄分类任务上仍然会先拿它开路。它快、可解释，并且在只关心词是否出现的任务上，经常和 400M 参数的 embedding model 没什么差别。

本课会从零构建 bag of words，再构建 TF-IDF。然后展示 scikit-learn 如何用三行代码做同样的事。最后指出那个会让你转向 embeddings 的失败模式。

## 概念

**Bag of Words（BoW）** 丢掉顺序。对每篇文档，统计词表中每个词出现了多少次。向量长度就是词表大小。位置 `i` 是第 `i` 个词的计数。

**TF-IDF** 会重新加权 BoW。一个出现在每篇文档里的词没有信息量，所以把它压低。一个在全语料里稀有、但在某篇文档中频繁出现的词是信号，所以把它抬高。

```
TF-IDF(w, d) = TF(w, d) * IDF(w)
             = count(w in d) / |d| * log(N / df(w))
```

其中 `TF` 是文档内的 term frequency，`df` 是 document frequency（包含该词的文档数），`N` 是总文档数。`log` 会让常见词的权重保持有界。

关键性质：二者都会生成稀疏向量，而且每个轴都可解释。你可以查看训练后分类器的权重，直接读出哪些词把文档推向哪个类别。768 维 BERT embedding 做不到这一点。

## 构建它

### 第 1 步：构建词表

```python
def build_vocab(docs):
    vocab = {}
    for doc in docs:
        for token in doc:
            if token not in vocab:
                vocab[token] = len(vocab)
    return vocab
```

输入：分好 token 的文档列表（任何词级 tokenizer 都可以；本课的 `code/main.py` 使用简化的小写版本）。输出：`{word: index}` dict。稳定的插入顺序意味着第一个文档里第一个出现的词索引是 0。约定会变；scikit-learn 按字母排序。

### 第 2 步：bag of words

```python
def bag_of_words(docs, vocab):
    matrix = [[0] * len(vocab) for _ in docs]
    for i, doc in enumerate(docs):
        for token in doc:
            if token in vocab:
                matrix[i][vocab[token]] += 1
    return matrix
```

```python
>>> docs = [["cat", "sat", "on", "mat"], ["cat", "cat", "ran"]]
>>> vocab = build_vocab(docs)
>>> bag_of_words(docs, vocab)
[[1, 1, 1, 1, 0], [2, 0, 0, 0, 1]]
```

行是文档。列是词表索引。条目 `[i][j]` 的含义是“词 `j` 在文档 `i` 中出现了多少次”。文档 1 里 `cat` 是 2，因为它确实出现了两次。文档 0 里 `ran` 是 0，因为它没出现。

### 第 3 步：term frequency 与 document frequency

```python
import math


def term_frequency(doc_bow, doc_length):
    return [c / doc_length if doc_length else 0 for c in doc_bow]


def document_frequency(bow_matrix):
    df = [0] * len(bow_matrix[0])
    for row in bow_matrix:
        for j, count in enumerate(row):
            if count > 0:
                df[j] += 1
    return df


def inverse_document_frequency(df, n_docs):
    return [math.log((n_docs + 1) / (d + 1)) + 1 for d in df]
```

这里有两个值得点名的 smoothing 技巧。`(n+1)/(d+1)` 避免 `log(x/0)`。末尾的 `+1` 确保出现在每篇文档里的词仍然有 IDF 1（而不是 0），这与 scikit-learn 的默认行为一致。其他实现会使用原始的 `log(N/df)`。两者都能工作；平滑版更友好。

### 第 4 步：TF-IDF

```python
def tfidf(bow_matrix):
    n_docs = len(bow_matrix)
    df = document_frequency(bow_matrix)
    idf = inverse_document_frequency(df, n_docs)
    out = []
    for row in bow_matrix:
        length = sum(row)
        tf = term_frequency(row, length)
        out.append([tf_j * idf_j for tf_j, idf_j in zip(tf, idf)])
    return out
```

```python
>>> docs = [
...     ["the", "cat", "sat"],
...     ["the", "dog", "sat"],
...     ["the", "cat", "ran"],
... ]
>>> vocab = build_vocab(docs)
>>> bow = bag_of_words(docs, vocab)
>>> tfidf(bow)
```

三篇文档，五个词表词（`the`、`cat`、`sat`、`dog`、`ran`）。`the` 出现在全部三篇，所以 IDF 很低。`dog` 只出现一次，所以 IDF 很高。向量是稀疏的（大多数条目很小），区分性的词会浮出来。

### 第 5 步：对行做 L2 normalize

```python
def l2_normalize(matrix):
    out = []
    for row in matrix:
        norm = math.sqrt(sum(x * x for x in row))
        out.append([x / norm if norm else 0 for x in row])
    return out
```

如果不做 normalization，长文档会得到更大的向量并主导相似度分数。L2 normalization 会把每篇文档都放到单位超球面上。行之间的 cosine similarity 现在就是点积。

## 使用它

scikit-learn 提供了生产版实现。

```python
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

docs = ["the cat sat on the mat", "the dog sat on the mat", "the cat ran"]

bow_vectorizer = CountVectorizer()
bow = bow_vectorizer.fit_transform(docs)
print(bow_vectorizer.get_feature_names_out())
print(bow.toarray())

tfidf_vectorizer = TfidfVectorizer()
tfidf = tfidf_vectorizer.fit_transform(docs)
print(tfidf.toarray().round(3))
```

`CountVectorizer` 在一次调用里完成 tokenization、词表和 BoW。`TfidfVectorizer` 额外加入 IDF weighting 和 L2 normalization。二者都返回稀疏矩阵。对 10 万篇文档，dense 版本放不进内存；在分类器强制要求 dense 之前，一直保持 sparse。

能改变一切的旋钮：

| 参数 | 影响 |
|-----|--------|
| `ngram_range=(1, 2)` | 加入 bigram。通常会提升分类效果。 |
| `min_df=2` | 丢掉少于 2 篇文档中出现的词。能在噪声数据上收缩词表。 |
| `max_df=0.95` | 丢掉超过 95% 文档中出现的词。相当于不靠硬编码列表近似移除停用词。 |
| `stop_words="english"` | scikit-learn 内置停用词列表。依赖任务；sentiment analysis 不应该删除否定词。 |
| `sublinear_tf=True` | 使用 `1 + log(tf)` 而不是原始 `tf`。当某个词在单篇文档中重复很多次时有帮助。 |

### 2026 年 TF-IDF 仍然赢的场景

- 垃圾邮件检测、主题标注、日志异常标记。词是否出现才是关键；语义细节不是。
- 低数据量场景（数百个标注样本）。TF-IDF 加 logistic regression 没有预训练成本。
- 任何对延迟敏感的地方。TF-IDF 加线性模型能在微秒级回答。把文档送进 transformer 做 embedding 要 10-100ms。
- 必须解释预测的系统。查看分类器系数即可。最强的正向词就是原因。

### TF-IDF 失败的地方

语义盲区。看这两篇文档：

- "The movie was not good at all."
- "The movie was excellent."

一条是负面评论。一条是正面评论。它们的 TF-IDF 重叠恰好是 `{the, movie, was}`。bag-of-words 分类器必须记住 `not` 靠近 `good` 会翻转标签。数据足够多时它可以学到，但永远不如理解语法的模型优雅。

另一个失败：推理时遇到 out-of-vocabulary 词。一个在 IMDb 评论上训练的 BoW 模型，如果训练中没见过 `Zoomer-approved`，就完全不知道该怎么办。Subword embeddings（第 04 课）能处理这个。TF-IDF 不行。

### 混合方案：TF-IDF 加权 embeddings

2026 年中等数据量分类的务实默认方案：用 TF-IDF 权重作为 word embeddings 上的 attention。

```python
def tfidf_weighted_embedding(doc, tfidf_scores, embedding_table, dim):
    vec = [0.0] * dim
    total_weight = 0.0
    for token in doc:
        if token not in embedding_table or token not in tfidf_scores:
            continue
        weight = tfidf_scores[token]
        emb = embedding_table[token]
        for i in range(dim):
            vec[i] += weight * emb[i]
        total_weight += weight
    if total_weight == 0:
        return vec
    return [v / total_weight for v in vec]
```

你同时得到 embeddings 的语义能力，以及 TF-IDF 对稀有词的强调。分类器在 pooled vector 上训练。在 5 万个左右标注样本以下的 sentiment、topic 和 intent classification 中，这通常比二者单独使用都好。

## 交付它

保存为 `outputs/prompt-vectorization-picker.md`：

```markdown
---
name: vectorization-picker
description: 给定文本分类任务，推荐 BoW、TF-IDF、embeddings 或混合方案。
phase: 5
lesson: 02
---

你负责推荐文本向量化策略。给定任务描述后，输出：

1. 表示方式（BoW、TF-IDF、transformer embeddings 或混合方案）。用一句话解释原因。
2. 具体 vectorizer 配置。写出库名。引用参数（`ngram_range`、`min_df`、`max_df`、`sublinear_tf`、`stop_words`）。
3. 发布前要测试的一个失败模式。

当用户的标注样本少于 500 个时，除非他们能证明 TF-IDF baseline 存在语义失败，否则拒绝推荐 embeddings。拒绝为 sentiment analysis 删除停用词（否定词携带信号）。指出类别不平衡需要的不只是换 vectorizer。

示例输入："Classifying 30k customer support tickets into 12 categories. Most tickets are 2-3 sentences. English only. Need explainability for audit logs."

示例输出：

- Representation: TF-IDF。3 万个样本不算少；可解释性要求排除了 dense embeddings。
- Config: `TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.95, sublinear_tf=True, stop_words=None)`。保留停用词，因为类别关键词有时就是停用词（"not working" vs "working"）。
- Failure to test: 验证 `min_df=3` 不会丢掉稀有类别关键词。运行按类别过滤的 `get_feature_names_out` 并人工浏览。
```

## 练习

1. **简单。** 在 L2-normalized TF-IDF 输出上实现 `cosine_similarity(doc_vec_a, doc_vec_b)`。验证相同文档得分为 1.0，词表完全不相交的文档得分为 0.0。
2. **中等。** 给 `bag_of_words` 增加 `n-gram` 支持。参数 `n` 会产出 `n`-grams 的计数。测试 `n=2` 作用在 `["the", "cat", "sat"]` 上时，会产出 `["the cat", "cat sat"]` 的 bigram 计数。
3. **困难。** 使用 GloVe 100d 向量（下载一次并缓存）构建上面的 TF-IDF-weighted-embedding 混合方案。在 20 Newsgroups 数据集上，对比纯 TF-IDF、纯 mean-pooled embeddings 的分类准确率。报告各自在哪些地方胜出。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|-----------------|-----------------------|
| BoW | 词频向量 | 一篇文档中词表词的计数。丢掉顺序。 |
| TF | Term frequency | 一个词在一篇文档中的计数，可选按文档长度归一化。 |
| DF | Document frequency | 至少包含该词一次的文档数量。 |
| IDF | Inverse document frequency | 平滑后的 `log(N / df)`。降低到处都出现的词的权重。 |
| Sparse vector | 大部分是零 | 词表通常有 1 万到 10 万个词；任意一篇文档中大多数都不存在。 |
| Cosine similarity | 向量夹角 | L2-normalized 向量的点积。1 表示相同，0 表示正交。 |

## 延伸阅读

- [scikit-learn — feature extraction from text](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction) — 标准 API 参考，也解释每个旋钮。
- [Salton, G., & Buckley, C. (1988). Term-weighting approaches in automatic text retrieval](https://www.sciencedirect.com/science/article/pii/0306457388900210) — 让 TF-IDF 成为十年默认方案的论文。
- ["Why TF-IDF Still Beats Embeddings" — Ashfaque Thonikkadavan (Medium)](https://medium.com/@cmtwskb/why-tf-idf-still-beats-embeddings-ad85c123e1b2) — 2026 年视角：旧方法什么时候赢，以及为什么赢。
