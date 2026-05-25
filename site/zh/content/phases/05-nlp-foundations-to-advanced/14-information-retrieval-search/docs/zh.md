# 信息检索与搜索

> BM25 精准但脆弱。Dense 撒网很广但会漏关键词。Hybrid 是 2026 年默认方案。其他都是调参。

**类型：** 构建
**语言：** Python
**前置要求：** 阶段 5 第 02 课（BoW + TF-IDF）、阶段 5 第 04 课（GloVe, FastText, Subword）
**时间：** ~75 分钟

## 问题

用户输入 "what happens if someone lies to get money"，期望找到真正覆盖这个问题的法条："Section 420 IPC." 关键词搜索会完全漏掉它（没有共享词表）。如果 embeddings 没在法律文本上训练，语义搜索也会漏掉。真实搜索必须同时处理二者。

IR 是每个 RAG system、每个搜索框、每个文档站点模糊查找底下的 pipeline。2026 年能在生产中工作的架构不是单一方法。它是一串互补方法，每一层都抓住前一层的失败。

本课会构建每个组件，并说明各自捕获哪些失败。

## 概念

![Hybrid retrieval: BM25 + dense + RRF + cross-encoder rerank](../assets/retrieval.svg)

四层。按需要选择。

1. **Sparse retrieval（BM25）。** 快，对 exact matches 很精准，对语义很差。运行在 inverted index 上。百万文档上每 query 低于 10ms。能正确处理 statute references、product codes、error messages、named entities。
2. **Dense retrieval。** 把 query 和 documents 编码成向量。做 nearest neighbor search。捕捉 paraphrases 和 semantic similarity。会漏掉只差一个字符的精确关键词匹配。使用 FAISS 或 vector DB 时，每 query 50-200ms。
3. **Fusion。** 合并 sparse 和 dense 的 ranked lists。Reciprocal Rank Fusion（RRF）是简单默认方案，因为它忽略 raw scores（分数在不同尺度里），只使用 rank positions。当你知道某个信号在领域中占主导时，weighted fusion 也是选项。
4. **Cross-encoder rerank。** 从 fusion 的 top-30 中取结果。运行 cross-encoder（query + document 一起输入，为每个 pair 打分）。保留 top-5。Cross-encoder 每个 pair 比 bi-encoder 慢得多，但准确得多。通过只在 top-30 上运行来摊销成本。

三路 retrieval（BM25 + dense + SPLADE 这类 learned-sparse）在 2026 benchmarks 上超过两路，但需要 learned-sparse indexes 的基础设施。对多数团队，两路加 cross-encoder rerank 是甜点位。

## 构建它

### 第 1 步：从零实现 BM25

```python
import math
import re
from collections import Counter

TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text):
    return TOKEN_RE.findall(text.lower())


class BM25:
    def __init__(self, corpus, k1=1.5, b=0.75):
        if not corpus:
            raise ValueError("corpus must not be empty")
        self.corpus = [tokenize(d) for d in corpus]
        self.k1 = k1
        self.b = b
        self.n_docs = len(self.corpus)
        self.avg_dl = sum(len(d) for d in self.corpus) / self.n_docs
        self.df = Counter()
        for doc in self.corpus:
            for term in set(doc):
                self.df[term] += 1

    def idf(self, term):
        n = self.df.get(term, 0)
        return math.log(1 + (self.n_docs - n + 0.5) / (n + 0.5))

    def score(self, query, doc_idx):
        q_tokens = tokenize(query)
        doc = self.corpus[doc_idx]
        dl = len(doc)
        freq = Counter(doc)
        score = 0.0
        for term in q_tokens:
            f = freq.get(term, 0)
            if f == 0:
                continue
            numerator = f * (self.k1 + 1)
            denominator = f + self.k1 * (1 - self.b + self.b * dl / self.avg_dl)
            score += self.idf(term) * numerator / denominator
        return score

    def rank(self, query, top_k=10):
        scored = [(self.score(query, i), i) for i in range(self.n_docs)]
        scored.sort(reverse=True)
        return scored[:top_k]
```

两个参数值得知道。`k1=1.5` 控制 term-frequency saturation；越高，term repetition 权重越大。`b=0.75` 控制 length normalization；0 忽略文档长度，1 完全归一化。这些默认值来自 Robertson 在原论文中的建议，几乎不需要调。

### 第 2 步：用 bi-encoder 做 dense retrieval

```python
from sentence_transformers import SentenceTransformer
import numpy as np


def build_dense_index(corpus, model_id="sentence-transformers/all-MiniLM-L6-v2"):
    encoder = SentenceTransformer(model_id)
    embeddings = encoder.encode(corpus, normalize_embeddings=True)
    return encoder, embeddings


def dense_search(encoder, embeddings, query, top_k=10):
    q_emb = encoder.encode([query], normalize_embeddings=True)
    sims = (embeddings @ q_emb.T).flatten()
    order = np.argsort(-sims)[:top_k]
    return [(float(sims[i]), int(i)) for i in order]
```

对 embeddings 做 L2-normalize，这样 dot product 就等于 cosine。`all-MiniLM-L6-v2` 是 384 维、快，对多数英文 retrieval 足够强。多语言工作用 `paraphrase-multilingual-MiniLM-L12-v2`。最高准确率用 `bge-large-en-v1.5` 或 `e5-large-v2`。

### 第 3 步：Reciprocal Rank Fusion

```python
def reciprocal_rank_fusion(rankings, k=60):
    scores = {}
    for ranking in rankings:
        for rank, (_, doc_idx) in enumerate(ranking):
            scores[doc_idx] = scores.get(doc_idx, 0.0) + 1.0 / (k + rank + 1)
    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [(score, doc_idx) for doc_idx, score in fused]
```

`k=60` 常数来自原始 RRF 论文。更高的 `k` 会压平 rank 差异的贡献；更低的 `k` 会让靠前排名主导。60 是论文默认值，几乎不需要调。

### 第 4 步：hybrid search + rerank

```python
from sentence_transformers import CrossEncoder

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def hybrid_search(query, bm25, encoder, dense_embeddings, corpus, top_k=5, pool_size=30, reranker=reranker):
    sparse_ranking = bm25.rank(query, top_k=pool_size)
    dense_ranking = dense_search(encoder, dense_embeddings, query, top_k=pool_size)
    fused = reciprocal_rank_fusion([sparse_ranking, dense_ranking])[:pool_size]

    pairs = [(query, corpus[doc_idx]) for _, doc_idx in fused]
    scores = reranker.predict(pairs)
    reranked = sorted(zip(scores, [doc_idx for _, doc_idx in fused]), reverse=True)
    return reranked[:top_k]
```

三个阶段组合。BM25 找 lexical matches。Dense 找 semantic matches。RRF 合并两个 rankings，不需要校准分数。Cross-encoder 用 query-document pairs 一起重新打 top-30 分，捕捉 bi-encoder 漏掉的细粒度相关性。保留 top-5。

### 第 5 步：评估

| 指标 | 含义 |
|--------|---------|
| Recall@k | 在正确文档存在的 queries 中，有多少比例能在 top-k 中看到它？ |
| MRR（Mean Reciprocal Rank） | 第一个 relevant document 的 1/rank 平均值。 |
| nDCG@k | 考虑相关性的等级，而不只是二元 relevant/not。 |

对 RAG 来说，retriever 的 **Recall@k** 是最重要数字。如果正确 passage 不在 retrieved set 里，reader 就无法回答。

Debugging tip：对失败 queries，diff sparse 和 dense rankings。如果一个找到了正确文档，另一个没有，你要么有 vocabulary mismatch（修复：补上缺失的那一半），要么有 semantic ambiguity（修复：更好的 embeddings 或 reranker）。

## 使用它

2026 年栈：

| 规模 | Stack |
|-------|-------|
| 1k-100k docs | In-memory BM25 + `all-MiniLM-L6-v2` embeddings + RRF。无需单独 DB。 |
| 100k-10M docs | Dense 用 FAISS 或 pgvector，BM25 用 Elasticsearch / OpenSearch。并行运行。 |
| 10M+ docs | Qdrant / Weaviate / Vespa / Milvus，使用 hybrid support。Top-30 上 cross-encoder rerank。 |
| 最佳质量前沿 | 三路（BM25 + dense + SPLADE）+ ColBERT late-interaction reranking |

无论你选什么，都要为评估留预算。先 benchmark retrieval recall，再 benchmark end-to-end RAG accuracy。Reader 无法修复 retriever 漏掉的东西。

### 2026 生产 RAG 的血泪经验

- **80% 的 RAG 失败来自 ingestion 和 chunking，不是模型。** 团队花几周换 LLM、调 prompt，而 retrieval 每三个 query 就悄悄返回错 context。先修 chunking。
- **Chunking strategy 比 chunk size 更重要。** 固定大小切分会打断表格、代码和嵌套标题。Sentence-aware 是默认选择；semantic 或 LLM-based chunking 对技术文档和产品手册值得投入。
- **Parent-doc pattern。** 检索小的 "child" chunks 以获得 precision。当同一个 parent section 的多个 children 出现时，换回 parent block 以保留上下文。无需重新训练就能稳定提升答案质量。
- **k_rerank=3 通常最优。** 超过这个数量的每个额外 chunk 都会增加 token 成本和生成延迟，却不提升答案质量。如果你的 k=8 仍比 k=3 好，reranker 表现不足。
- **HyDE / query expansion。** 从 query 生成 hypothetical answer，embed 它，再检索。弥合短问题和长文档之间的表述差距。无需训练即可免费提升 precision。
- **Context budget 低于 8K tokens。** 如果总是撞到这个限制，说明 reranker threshold 太松。
- **所有东西都版本化。** Prompts、chunking rules、embedding model、reranker。任何漂移都会悄悄破坏答案质量。CI gates 用 faithfulness、context precision 和 unanswered-question rate 阻断回归。
- **三路 retrieval（BM25 + dense + SPLADE 这类 learned-sparse）在 2026 benchmarks 上超过两路**，尤其适合混合 proper nouns 和语义的 queries。当基础设施支持 SPLADE indexes 时发布它。

根据 2026 年行业测量，正确的 retrieval 设计能把 hallucinations 降低 70-90%。多数 RAG 性能收益来自更好的 retrieval，不是模型 fine-tuning。

## 交付它

保存为 `outputs/skill-retrieval-picker.md`：

```markdown
---
name: retrieval-picker
description: 为给定 corpus 和 query pattern 选择 retrieval stack。
version: 1.0.0
phase: 5
lesson: 14
tags: [nlp, retrieval, rag, search]
---

给定需求（corpus size、query pattern、latency budget、quality bar、infra constraints），输出：

1. Stack。BM25 only、dense only、hybrid（BM25 + dense + RRF）、hybrid + cross-encoder rerank，或 three-way（BM25 + dense + learned-sparse）。
2. Dense encoder。写出具体模型。匹配语言、领域和 context length。
3. Reranker。如果使用，写出具体 cross-encoder model。指出 rerank 会在 top-30 上额外增加 30-100ms 延迟。
4. Evaluation plan。Recall@10 是主要 retriever metric。多答案用 MRR。先 baseline，再衡量增量改进。

对于包含 named entities、error codes 或 product SKUs 的 corpus，除非用户有证据证明 dense 能处理 exact matches，否则拒绝推荐 dense-only。对于高风险 retrieval（法律、医疗），最终 top-5 决定用户答案时，拒绝跳过 reranking。
```

## 练习

1. **简单。** 在 500 文档 corpus 上实现上面的 `hybrid_search`。测试 20 个 queries。比较 BM25-only、dense-only 和 hybrid 的 recall at 5。
2. **中等。** 添加 MRR 计算。对每个有已知正确文档的 test query，找出正确 doc 在 BM25、dense 和 hybrid rankings 中的 rank。报告各自 MRR。
3. **困难。** 使用 MultipleNegativesRankingLoss（Sentence Transformers）在你的领域上 fine-tune dense encoder。从 500 对 query-document pairs 构建训练集。比较 fine-tune 前后的 recall。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|-----------------|-----------------------|
| BM25 | 关键词搜索 | Okapi BM25。按 term frequency、IDF 和长度给文档打分。 |
| Dense retrieval | 向量搜索 | 把 query + doc 编码成向量，找最近邻。 |
| Bi-encoder | Embedding model | 独立编码 query 和 doc。查询时快。 |
| Cross-encoder | Reranker model | 把 query + doc 一起编码。慢但准确。 |
| RRF | Rank fusion | 通过求和 `1/(k + rank)` 合并两个 rankings。 |
| Recall@k | Retrieval metric | relevant doc 出现在 top-k 中的 query 比例。 |

## 延伸阅读

- [Robertson and Zaragoza (2009). The Probabilistic Relevance Framework: BM25 and Beyond](https://www.staff.city.ac.uk/~sbrp622/papers/foundations_bm25_review.pdf) — BM25 的权威处理。
- [Karpukhin et al. (2020). Dense Passage Retrieval for Open-Domain QA](https://arxiv.org/abs/2004.04906) — DPR，经典 bi-encoder。
- [Formal et al. (2021). SPLADE: Sparse Lexical and Expansion Model](https://arxiv.org/abs/2107.05720) — 拉近与 dense 差距的 learned-sparse retriever。
- [Cormack, Clarke, Büttcher (2009). Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf) — RRF 论文。
- [Khattab and Zaharia (2020). ColBERT: Efficient and Effective Passage Search](https://arxiv.org/abs/2004.12832) — late-interaction retrieval。
