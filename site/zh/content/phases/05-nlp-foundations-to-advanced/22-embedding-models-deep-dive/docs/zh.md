# Embedding Models：2026 深入解析

> Word2Vec 给每个词一个向量。现代 embedding models 给每个 passage 一个向量，支持 cross-lingual，能以 sparse、dense、multi-vector 视图表示，并按你的索引容量调整维度。选错了，RAG 就会检索错内容。

**类型：** 学习
**语言：** Python
**前置要求：** 阶段 5 · 03（Word2Vec），阶段 5 · 14（信息检索）
**时间：** ~60 分钟

## 问题

你的 RAG 系统 40% 的时候检索错 passage。罪魁祸首很少是 vector database 或 prompt，通常是 embedding model。

2026 年选择 embedding 意味着在五个轴上取舍：

1. **Dense vs sparse vs multi-vector。** 每个 passage 一个 vector，还是每个 token 一个 vector，或 sparse weighted bag of words。
2. **Language coverage。** 纯英语任务上 monolingual English models 仍然胜出。混合语料上 multilingual models 胜出。
3. **Context length。** 512 tokens vs 8,192 vs 32,768，而且真实有效容量通常只有广告 max 的 60-70%。
4. **Dimension budget。** 3,072 个 full precision floats = 每个 vector 12 KB。100M vectors 时，存储每月约 $1,300。Matryoshka truncation 能降 4 倍。
5. **Open vs hosted。** Open-weight 意味着你控制 stack 和数据。Hosted 意味着用控制权换 always-latest。

本课命名这些 tradeoffs，让你基于证据选择，而不是基于上季度流行什么。

## 概念

![Dense, sparse, and multi-vector embeddings](../assets/embedding-modes.svg)

**Dense embeddings。** 每个 passage 一个 vector（通常 384-3,072 维）。Cosine similarity 按语义接近度排序 passages。OpenAI `text-embedding-3-large`、BGE-M3 dense mode、Voyage-3。默认选择。

**Sparse embeddings。** SPLADE 风格。Transformer 预测 vocabulary 每个 token 的权重，然后把大多数置零。结果是一个大小为 |vocab| 的 sparse vector。捕捉 lexical matching（类似 BM25），但 term weights 是学出来的。对 keyword-heavy queries 很强。

**Multi-vector（late interaction）。** ColBERTv2、Jina-ColBERT。每个 token 一个 vector。用 MaxSim 打分：对每个 query token，找到最相似的 document token，求和。存储和 scoring 更贵，但在长 queries 和 domain-specific corpora 上胜出。

**BGE-M3：三者合一。** 单个模型同时输出 dense、sparse 和 multi-vector representations。每种都可独立查询；scores 通过 weighted sum 融合。当你想从一个 checkpoint 获得灵活性时，它是 2026 默认。

**Matryoshka Representation Learning。** 训练目标让 vector 的前 N 维本身就是可用的 standalone embedding。把 1,536-dim vector 截断到 256 dim，只损失约 1% accuracy，却节省 6 倍存储。OpenAI text-3、Cohere v4、Voyage-4、Jina v5、Gemini Embedding 2、Nomic v1.5+ 支持。

### MTEB leaderboard 只讲了部分故事

Massive Text Embedding Benchmark：发布时（2022）包含 8 类任务的 56 个任务，在 MTEB v2 扩展到 100+。2026 年初，Gemini Embedding 2 在 retrieval 上领先（67.71 MTEB-R）。Cohere embed-v4 领先 general（65.2 MTEB）。BGE-M3 领先 open-weight multilingual（63.0）。Leaderboard 必要但不充分：始终在你的领域上 benchmark。

### 三层模式

| 用例 | 模式 |
|----------|---------|
| 快速第一轮 | Dense bi-encoder（BGE-M3、text-3-small） |
| Recall boost | Sparse（SPLADE、BGE-M3 sparse）+ RRF fuse |
| Top-50 precision | Multi-vector（ColBERTv2）或 cross-encoder reranker |

多数生产栈三者都会用。

## 构建它

### 第 1 步：baseline：Sentence-BERT dense embeddings

```python
from sentence_transformers import SentenceTransformer
import numpy as np

encoder = SentenceTransformer("BAAI/bge-small-en-v1.5")
corpus = [
    "The first iPhone launched in 2007.",
    "Apple released the iPod in 2001.",
    "Android is an operating system from Google.",
]
emb = encoder.encode(corpus, normalize_embeddings=True)

query = "When was the iPhone released?"
q_emb = encoder.encode([query], normalize_embeddings=True)[0]
scores = emb @ q_emb
print(sorted(enumerate(scores), key=lambda x: -x[1]))
```

`normalize_embeddings=True` 让 dot product 等价于 cosine similarity。始终设置它。

### 第 2 步：Matryoshka truncation

```python
def truncate(vectors, dim):
    out = vectors[:, :dim]
    return out / np.linalg.norm(out, axis=1, keepdims=True)

emb_256 = truncate(emb, 256)
emb_128 = truncate(emb, 128)
```

截断后重新 normalize。Nomic v1.5、OpenAI text-3 和 Voyage-4 经过训练，前几个层级基本无损。非 Matryoshka 模型（原始 Sentence-BERT）被截断时会明显退化。

### 第 3 步：BGE-M3 multi-functionality

```python
from FlagEmbedding import BGEM3FlagModel

model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)

output = model.encode(
    corpus,
    return_dense=True,
    return_sparse=True,
    return_colbert_vecs=True,
)
# output["dense_vecs"]:    (n_docs, 1024)
# output["lexical_weights"]: list of dict {token_id: weight}
# output["colbert_vecs"]:  list of (n_tokens, 1024) arrays
```

三个索引，一次 inference call。Score fusion：

```python
dense_score = ... # cosine over dense_vecs
sparse_score = model.compute_lexical_matching_score(q_lex, d_lex)
colbert_score = model.colbert_score(q_col, d_col)
final = 0.4 * dense_score + 0.2 * sparse_score + 0.4 * colbert_score
```

在你的领域上调权重。

### 第 4 步：custom task 上的 MTEB eval

```python
from mteb import MTEB

tasks = ["ArguAna", "SciFact", "NFCorpus"]
evaluation = MTEB(tasks=tasks)
results = evaluation.run(encoder, output_folder="./mteb-results")
```

在 *representative* subset 上运行 candidate models。不要只信 leaderboard rank，领域很重要。

### 第 5 步：从零手写 cosine

见 `code/main.py`。Averaged Hashing Trick embeddings（stdlib-only）。它无法与 transformer embeddings 竞争，但展示了形状：tokenize → vector → normalize → dot product。

## 坑

- **Query 和 doc 用同一模型。** 一些模型（Voyage、Jina-ColBERT）使用 asymmetric encoding，query 和 document 走不同路径。始终检查 model card。
- **缺少 prefix。** `bge-*` 模型需要给 query 加 `"Represent this sentence for searching relevant passages: "`。忘记会掉 3-5 点 recall。
- **过度裁剪 Matryoshka。** 1,536 → 256 通常安全。1,536 → 64 不安全。在 eval set 上验证。
- **Context truncation。** 大多数模型会静默截断超过 max length 的输入。长文档需要 chunking（见第 23 课）。
- **忽略 latency tail。** MTEB 分数隐藏 p99 latency。600M 模型可能比 335M 模型高 2 分，但每次 query 贵 3 倍。

## 使用它

2026 年技术栈：

| 场景 | 选择 |
|-----------|------|
| English-only、fast、API | `text-embedding-3-large` 或 `voyage-3-large` |
| Open-weight、English | `BAAI/bge-large-en-v1.5` |
| Open-weight、multilingual | `BAAI/bge-m3` 或 `Qwen3-Embedding-8B` |
| Long context（32k+） | Voyage-3-large、Cohere embed-v4、Qwen3-Embedding-8B |
| CPU-only deployment | Nomic Embed v2（137M params，MoE） |
| Storage-constrained | Matryoshka-truncated + int8 quantization |
| Keyword-heavy queries | 加 SPLADE sparse，用 RRF 与 dense 融合 |

2026 模式：从 BGE-M3 或 text-3-large 开始，用 MTEB 在你的领域上评估；如果 domain-specific model 赢超过 3 分，再替换。

## 交付它

保存为 `outputs/skill-embedding-picker.md`：

```markdown
---
name: embedding-picker
description: 为给定 corpus 和 deployment 选择 embedding model、dimension 和 retrieval mode。
version: 1.0.0
phase: 5
lesson: 22
tags: [nlp, embeddings, retrieval]
---

给定 corpus（size、languages、domain、avg length）、deployment target（cloud / edge / on-prem）、latency budget 和 storage budget，输出：

1. Model。命名 checkpoint 或 API。一句话说明理由。
2. Dimension。Full / Matryoshka-truncated / int8-quantized。理由关联 storage budget。
3. Mode。Dense / sparse / multi-vector / hybrid。说明理由。
4. Query prefix / template（如果 model card 要求）。
5. Evaluation plan。与 domain 相关的 MTEB tasks + held-out domain eval with nDCG@10。

拒绝推荐把 Matryoshka 截到 <64 dims 且没有 domain validation。拒绝为少于 10k passages 的 corpora 使用 ColBERTv2（overhead 不合理）。标记把长文档 corpora（>8k tokens）路由到 512-token windows 模型的方案。
```

## 练习

1. **简单。** 用 `bge-small-en-v1.5` 对 100 个句子做 full dim（384）编码，再用 Matryoshka 128 编码。测量 10 个 queries 上的 MRR drop。
2. **中等。** 在你领域的 500 个 passages 上比较 BGE-M3 dense、sparse 和 colbert。哪个 recall@10 最高？RRF fusion 是否超过最好的 single mode？
3. **困难。** 在 top-2 domain tasks 上对三个 candidate models 运行 MTEB。报告 MTEB score、100-query batch 上的 p99 latency 和 $/1M queries。选择 Pareto-optimal 的一个。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|-----------------|-----------------------|
| Dense embedding | 那个 vector | 每段文本一个固定大小 vector。用 cosine similarity 排序。 |
| Sparse embedding | 学出来的 BM25 | 每个 vocab token 一个权重，大多为零，端到端训练。 |
| Multi-vector | ColBERT-style | 每个 token 一个 vector；MaxSim scoring；索引更大，recall 更好。 |
| Matryoshka | 套娃技巧 | 前 N 维本身就是可用的小 embedding。 |
| MTEB | benchmark | Massive Text Embedding Benchmark：发布时 56 任务，v2 里 100+。 |
| BEIR | retrieval benchmark | 18 个 zero-shot retrieval tasks；常用于 cross-domain robustness。 |
| Asymmetric encoding | Query ≠ doc path | 模型对 queries 和 documents 使用不同 projections。 |

## 延伸阅读

- [Reimers, Gurevych (2019). Sentence-BERT](https://arxiv.org/abs/1908.10084) — bi-encoder 论文。
- [Muennighoff et al. (2022). MTEB: Massive Text Embedding Benchmark](https://arxiv.org/abs/2210.07316) — leaderboard 论文。
- [Chen et al. (2024). BGE-M3: Multi-lingual, Multi-functionality, Multi-granularity](https://arxiv.org/abs/2402.03216) — 统一三种模式的模型。
- [Kusupati et al. (2022). Matryoshka Representation Learning](https://arxiv.org/abs/2205.13147) — dimension-ladder training objective。
- [Santhanam et al. (2022). ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction](https://arxiv.org/abs/2112.01488) — 生产中的 late interaction。
- [MTEB leaderboard on Hugging Face](https://huggingface.co/spaces/mteb/leaderboard) — 实时排名。
