# Topic Modeling：LDA 与 BERTopic

> LDA：文档是 topics 的混合，topics 是词上的分布。BERTopic：文档在 embedding 空间里聚类，clusters 就是 topics。目标相同，原语不同。

**类型：** 学习
**语言：** Python
**前置要求：** 阶段 5 第 02 课（BoW + TF-IDF）、阶段 5 第 03 课（Word2Vec）
**时间：** ~45 分钟

## 问题

你有 10,000 条客服工单、50,000 篇新闻文章，或 200,000 条 tweets。你需要知道这个集合在讲什么，但不能一篇篇读。你没有标注类别。你甚至不知道有多少类别。

Topic modeling 用无监督方式回答这个问题。给它一个 corpus，得到一小组 coherent topics，以及每篇文档在这些 topics 上的分布。

两类算法家族占主导。LDA（2003）把每篇文档看作 latent topics 的混合，把每个 topic 看作词上的分布。Inference 是 Bayesian 的。当你需要 mixed-membership topic assignments 和可解释的词级概率分布时，它仍然在生产中发布。

BERTopic（2020）用 BERT 编码文档，用 UMAP 降维，用 HDBSCAN 聚类，并通过 class-based TF-IDF 抽取 topic words。它在短文本、社交媒体，以及任何语义相似性比词重叠更重要的地方胜出。每篇文档只得到一个 topic，这是它对长篇内容的限制。

本课会为二者建立直觉，并说明给定 corpus 时该选哪个。

## 概念

![LDA mixture model vs BERTopic clustering](../assets/topic-modeling.svg)

**LDA generative story。** 每个 topic 是词上的分布。每篇文档是 topics 的混合。生成文档中的一个词时，先从文档的 mixture 中采样 topic，再从该 topic 的词分布中采样词。Inference 会反过来：给定观测词，推断每篇文档的 topic distribution 和每个 topic 的 word distribution。Collapsed Gibbs sampling 或 variational Bayes 负责数学。

LDA 的关键输出：

- `doc_topic`：矩阵 `(n_docs, n_topics)`，每行和为 1（文档的 topic mixture）。
- `topic_word`：矩阵 `(n_topics, vocab_size)`，每行和为 1（topic 的 word distribution）。

**BERTopic pipeline。**

1. 用 sentence transformer（例如 `all-MiniLM-L6-v2`）编码每篇文档。384 维向量。
2. 用 UMAP 降到约 5 维。BERT embeddings 对 clustering 来说维度太高。
3. 用 HDBSCAN 聚类。Density-based，产出可变大小 clusters 和一个 "outlier" label。
4. 对每个 cluster，在该 cluster 的文档上计算 class-based TF-IDF，抽取 top words。

输出是每篇文档一个 topic（加一个 -1 outlier label）。也可以通过 HDBSCAN 的 probability vector 得到 soft membership。

## 构建它

### 第 1 步：用 scikit-learn 做 LDA

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np


def fit_lda(documents, n_topics=5, max_features=1000):
    cv = CountVectorizer(
        max_features=max_features,
        stop_words="english",
        min_df=2,
        max_df=0.9,
    )
    X = cv.fit_transform(documents)
    lda = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=42,
        max_iter=50,
        learning_method="online",
    )
    doc_topic = lda.fit_transform(X)
    feature_names = cv.get_feature_names_out()
    return lda, cv, doc_topic, feature_names


def print_top_words(lda, feature_names, n_top=10):
    for idx, topic in enumerate(lda.components_):
        top_idx = np.argsort(-topic)[:n_top]
        words = [feature_names[i] for i in top_idx]
        print(f"topic {idx}: {' '.join(words)}")
```

注意：移除 stopwords，`min_df` 和 `max_df` 过滤稀有词和到处都出现的词，使用 CountVectorizer（不是 TfidfVectorizer），因为 LDA 需要原始计数。

### 第 2 步：BERTopic（生产）

```python
from bertopic import BERTopic

topic_model = BERTopic(
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    min_topic_size=15,
    verbose=True,
)

topics, probs = topic_model.fit_transform(documents)
info = topic_model.get_topic_info()
print(info.head(20))
valid_topics = info[info["Topic"] != -1]["Topic"].tolist()
for topic_id in valid_topics[:5]:
    print(f"topic {topic_id}: {topic_model.get_topic(topic_id)[:10]}")
```

`Topic != -1` 这个过滤会丢掉 BERTopic 的 outlier bucket（HDBSCAN 无法聚类的文档）。`min_topic_size` 控制 HDBSCAN 的最小 cluster size；BERTopic 库默认值是 10。这个例子为了适配本课规模，显式设置为 15。对超过 10,000 篇文档的 corpus，把它增加到 50 或 100。

### 第 3 步：评估

两种方法都会输出 topic words。问题是这些词是否 coherent。

- **Topic coherence（c_v）。** 结合 top-word pairs 在 sliding-window contexts 上的 NPMI（normalized pointwise mutual information），把分数聚合成 topic vectors，并用 cosine similarity 比较这些 vectors。越高越好。使用 `gensim.models.CoherenceModel`，设置 `coherence="c_v"`。
- **Topic diversity。** 所有 topics 的 top words 中 unique words 的比例。越高越好（topics 不重叠）。
- **Qualitative inspection。** 阅读每个 topic 的 top words。它们是否命名了一个真实东西？人类判断仍然是最后防线。

## 什么时候选哪个

| 场景 | 选择 |
|-----------|------|
| 短文本（tweets、reviews、headlines） | BERTopic |
| 带 topic mixtures 的长文档 | LDA |
| 无 GPU / 计算受限 | LDA 或 NMF |
| 需要文档级 multi-topic distributions | LDA |
| 面向 LLM integration 的 topic labeling | BERTopic（直接支持） |
| 资源受限边缘部署 | LDA |
| 最高语义 coherence | BERTopic |

最大的实践考量是文档长度。BERT embeddings 会截断；LDA counts 可处理任意长度。对超过 embedding model context 的文档，要么 chunk + aggregate，要么使用 LDA。

## 使用它

2026 年栈：

- **BERTopic。** 短文本和任何语义重要场景的默认选择。
- **`gensim.models.LdaModel`。** 经典生产 LDA，成熟、经受过实战。
- **`sklearn.decomposition.LatentDirichletAllocation`。** 适合实验的简单 LDA。
- **NMF。** Non-negative matrix factorization。LDA 的快速替代，在短文本上质量相近。
- **Top2Vec。** 和 BERTopic 设计类似。社区更小，但在某些 benchmark 上不错。
- **FASTopic。** 更新，在超大 corpus 上比 BERTopic 更快。
- **LLM-based labeling。** 运行任何 clustering，然后 prompt 模型为每个 cluster 命名。

## 交付它

保存为 `outputs/skill-topic-picker.md`：

```markdown
---
name: topic-picker
description: 为 corpus 选择 LDA 或 BERTopic。指定库、旋钮和评估。
version: 1.0.0
phase: 5
lesson: 15
tags: [nlp, topic-modeling]
---

给定 corpus 描述（document count、avg length、domain、language、compute budget），输出：

1. Algorithm。LDA / NMF / BERTopic / Top2Vec / FASTopic。用一句话说明原因。
2. Configuration。Topic 数：`recommended = max(5, round(sqrt(n_docs)))`，对 40,000 docs 以下的 corpora 上限 clamp 到 200；只有当 corpus 真正很大（>40k）时才允许 >200，并说明计算成本增加。`min_df` / `max_df` filters 和 neural approaches 的 embedding model 也属于这里。
3. Evaluation。通过 `gensim.models.CoherenceModel` 做 topic coherence（c_v）、topic diversity，以及 20-sample human read。
4. 要探测的失败模式。对 LDA，是吸收 stopwords 和高频词的 "junk topics"。对 BERTopic，是 -1 outlier cluster 吞掉 ambiguous documents。

没有 chunking strategy 时，拒绝在超过 embedding model context window 的文档上使用 BERTopic。拒绝在非常短文本（tweets、少于 10 tokens 的 reviews）上使用 LDA，因为 coherence 会崩。把低于 5 的 n_topics 选择标记为很可能错误；把 40k docs 以下 corpus 上 >200 的选择标记为很可能 over-splitting。
```

## 练习

1. **简单。** 在 20 Newsgroups 数据集上用 5 个 topics 拟合 LDA。打印每个 topic 的 top 10 words。手工给每个 topic 命名。算法找到真实类别了吗？
2. **中等。** 在同一个 20 Newsgroups subset 上拟合 BERTopic。和 LDA 比较发现的 topics 数量、top words、定性 coherence。哪个更清晰地浮现真实类别？
3. **困难。** 在你的 corpus 上计算 LDA 和 BERTopic 的 c_v coherence。分别用 5、10、20、50 topics 运行。画 coherence vs topic count。报告哪种方法在 topic counts 上更稳定。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|-----------------|-----------------------|
| Topic | Corpus 讲的东西 | 词上的概率分布（LDA）或相似文档的 cluster（BERTopic）。 |
| Mixed membership | 文档属于多个 topics | LDA 给每篇文档分配所有 topics 上的分布。 |
| UMAP | 降维 | 保留局部结构的 manifold learning；BERTopic 中使用。 |
| HDBSCAN | 密度聚类 | 找到可变大小 clusters；为 outliers 产出 "noise" label（-1）。 |
| c_v coherence | Topic 质量指标 | Sliding windows 内 top topic words 的平均 pointwise mutual information。 |

## 延伸阅读

- [Blei, Ng, Jordan (2003). Latent Dirichlet Allocation](https://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf) — LDA 论文。
- [Grootendorst (2022). BERTopic: Neural topic modeling with a class-based TF-IDF procedure](https://arxiv.org/abs/2203.05794) — BERTopic 论文。
- [Röder, Both, Hinneburg (2015). Exploring the Space of Topic Coherence Measures](https://svn.aksw.org/papers/2015/WSDM_Topic_Evaluation/public.pdf) — 引入 c_v 等指标的论文。
- [BERTopic documentation](https://maartengr.github.io/BERTopic/) — 生产参考。示例非常好。
