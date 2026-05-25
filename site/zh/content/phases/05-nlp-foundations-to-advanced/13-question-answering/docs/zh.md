# 问答系统

> 三类系统塑造了现代 QA。Extractive 找 span。Retrieval-augmented 把答案 grounded 到文档里。Generative 产出答案。每个现代 AI assistant 都是三者的混合。

**类型：** 构建
**语言：** Python
**前置要求：** 阶段 5 第 11 课（Machine Translation）、阶段 5 第 10 课（Attention Mechanism）
**时间：** ~75 分钟

## 问题

用户输入 "When did the first iPhone launch?"，期待得到 "June 29, 2007."。不是 "Apple's history is long and varied."，也不是孤零零的 "2007"。而是直接、grounded、正确的答案。

过去十年里，三类架构主导了 QA。

- **Extractive QA。** 给定一个问题和一个已知包含答案的 passage，找出 passage 中答案 span 的 start 和 end indices。SQuAD 是经典 benchmark。
- **Open-domain QA。** 不给 passage。先检索相关 passage，再抽取或生成答案。这是今天每条 RAG pipeline 的基石。
- **Generative / Closed-book QA。** 大语言模型从 parametric memory 中回答。无 retrieval。推理最快，事实可靠性最低。

2026 年趋势是 hybrid：检索最好的几个 passages，然后 prompt 生成模型基于这些 passages 回答。这就是 RAG，第 14 课会深入讲 retrieval 这一半。本课构建 QA 这一半。

## 概念

![QA architectures: extractive, retrieval-augmented, generative](../assets/qa.svg)

**Extractive。** 用 transformer（BERT 家族）把 question 和 passage 一起编码。训练两个 heads，分别预测答案的 start token index 和 end token index。Loss 是 valid positions 上的 cross-entropy。输出是 passage 中的 span。按构造不会 hallucinate，也按构造无法处理 passage 不能回答的问题。

**Retrieval-augmented（RAG）。** 两阶段。第一，retriever 从 corpus 中找到 top-`k` passages。第二，reader（extractive 或 generative）使用这些 passages 产出答案。Retriever-reader 分离让二者可以独立训练和评估。现代 RAG 往往在二者之间加 reranker。

**Generative。** Decoder-only LLM（GPT、Claude、Llama）从学到的权重中回答。没有 retrieval step。对常识知识表现优秀，对稀有或近期事实灾难性。Hallucination rate 和 pretraining data 中事实频率负相关。

## 构建它

### 第 1 步：用预训练模型做 extractive QA

```python
from transformers import pipeline

qa = pipeline("question-answering", model="deepset/roberta-base-squad2")

passage = (
    "Apple Inc. released the first iPhone on June 29, 2007. "
    "The device was announced by Steve Jobs at Macworld in January 2007."
)
question = "When was the first iPhone released?"

answer = qa(question=question, context=passage)
print(answer)
```

```python
{'score': 0.98, 'start': 57, 'end': 70, 'answer': 'June 29, 2007'}
```

`deepset/roberta-base-squad2` 在 SQuAD 2.0 上训练，其中包含 unanswerable questions。默认情况下，即使模型的 null score 获胜，`question-answering` pipeline 也会返回最高分 span，它 *不会* 自动返回空答案。要获得显式 "no answer" 行为，需要在 pipeline 调用中传 `handle_impossible_answer=True`：这时只有 null score 超过所有 span score，pipeline 才会返回空答案。不管哪种方式，都要检查 `score` 字段。

### 第 2 步：retrieval-augmented pipeline（草图）

```python
from sentence_transformers import SentenceTransformer
import numpy as np

encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

corpus = [
    "Apple Inc. released the first iPhone on June 29, 2007.",
    "Macworld 2007 featured the iPhone announcement by Steve Jobs.",
    "Android launched in 2008 as Google's mobile operating system.",
    "The first iPod was released in 2001.",
]
corpus_embeddings = encoder.encode(corpus, normalize_embeddings=True)


def retrieve(question, top_k=2):
    q_emb = encoder.encode([question], normalize_embeddings=True)
    sims = (corpus_embeddings @ q_emb.T).squeeze()
    order = np.argsort(-sims)[:top_k]
    return [corpus[i] for i in order]


def answer(question):
    passages = retrieve(question, top_k=2)
    combined = " ".join(passages)
    return qa(question=question, context=combined)


print(answer("When was the first iPhone released?"))
```

两阶段 pipeline。Dense retriever（Sentence-BERT）用 semantic similarity 找相关 passages。Extractive reader（RoBERTa-SQuAD）从合并后的 top passages 中抽取答案 span。适用于小语料。对于百万级文档语料，使用 FAISS 或 vector database。

### 第 3 步：带 RAG 的 generative

```python
def rag_generate(question, llm):
    passages = retrieve(question, top_k=3)
    prompt = f"""Context:
{chr(10).join('- ' + p for p in passages)}

Question: {question}

Answer using only the context above. If the context does not contain the answer, say "I don't know."
"""
    return llm(prompt)
```

Prompt pattern 很重要。明确告诉模型基于 context 回答，并在 context 不足时返回 "I don't know"，相比 naive prompting 能把 hallucination rates 降低 40-60%。更复杂模式会加入 citations、confidence scores 和 structured extraction。

### 第 4 步：反映真实世界的评估

SQuAD 使用 **Exact Match（EM）** 和 **token-level F1**。EM 是 normalization（小写、去标点、去冠词）后的严格匹配，要么完全匹配，要么 0 分。F1 按 prediction 和 reference 的 token overlap 计算，给部分正确答案一些分数。二者都会低估 paraphrases："June 29, 2007" vs "June 29th, 2007" 通常 EM 为 0（序数破坏了 normalization），但仍能从重叠 tokens 中得到可观 F1。

对于生产 QA：

- **Answer accuracy**（LLM-judged 或 human-judged，因为指标无法捕获语义等价）。
- **Citation accuracy。** 引用的 passage 是否真的支持答案？可以用 generated citations 和 retrieved passages 之间的 string match 自动检查。
- **Refusal calibration。** 当 retrieved passages 中没有答案时，系统是否正确说 "I don't know"？衡量 false confidence rate。
- **Retrieval recall。** 在评估 reader 前，先衡量 retriever 是否把正确 passage 放进 top-`k`。Reader 无法修复缺失 passage。

### RAGAS：2026 年生产评估框架

`RAGAS` 是专门为 RAG systems 设计的，也是 2026 年上线默认选择。它不需要 gold references，就能给四个维度打分：

- **Faithfulness。** 答案中的每个 claim 是否来自 retrieved context？用 NLI-based entailment 衡量。你的主要 hallucination metric。
- **Answer relevance。** 答案是否回应了问题？通过从答案生成 hypothetical questions，再和真实问题比较来衡量。
- **Context precision。** Retrieved chunks 中有多少比例真正相关？低 precision = prompt 中有噪声。
- **Context recall。** Retrieved set 是否包含所有所需信息？低 recall = reader 无法成功。

Reference-free scoring 让你能在没有 curated gold answers 的生产流量上评估。对于 exact-match metrics 没用的开放问题，在上面再叠 LLM-as-judge。

`pip install ragas`。接上你的 retriever + reader。每个 query 得到四个 scalar。对 regressions 告警。

## 使用它

2026 年栈。

| 使用场景 | 推荐 |
|---------|-------------|
| 给定 passage，找答案 span | `deepset/roberta-base-squad2` |
| 固定 corpus 上，closed-book 不可接受 | RAG：dense retriever + LLM reader |
| 文档库上的实时 QA | 使用 hybrid（BM25 + dense）retriever + reranker 的 RAG（第 14 课） |
| Conversational QA（后续问题） | 带 conversation history 的 LLM + 每轮 RAG |
| 高事实性、受监管领域 | 在权威 corpus 上做 extractive；绝不单独用 generative |

Extractive QA 在 2026 年不时髦，因为带 LLM 的 RAG 能处理更多情况。但在需要逐字引用的场景中仍然会发布：法律研究、监管合规、审计工具。

## 交付它

保存为 `outputs/skill-qa-architect.md`：

```markdown
---
name: qa-architect
description: 选择 QA 架构、retrieval strategy 和 evaluation plan。
version: 1.0.0
phase: 5
lesson: 13
tags: [nlp, qa, rag]
---

给定需求（corpus size、question type、factuality constraint、latency budget），输出：

1. Architecture。Extractive、RAG with extractive reader、RAG with generative reader，或 closed-book LLM。用一句话说明原因。
2. Retriever。None、BM25、dense（写出 encoder 名称）或 hybrid。
3. Reader。SQuAD-tuned model、具体 LLM 名称，或 "domain-fine-tuned DistilBERT"。
4. Evaluation。Extractive benchmarks 用 EM + F1；生产用 answer accuracy + citation accuracy + refusal calibration。说明你在衡量什么以及如何衡量。

拒绝为监管或合规敏感问题使用 closed-book LLM answers。拒绝任何没有 retrieval-recall baseline 的 QA system（不知道 retriever 是否找到了正确 passage，就无法评估 reader）。指出需要 multi-hop reasoning 的问题需要专门的 multi-hop retrievers，例如 HotpotQA-trained systems。
```

## 练习

1. **简单。** 在 10 个 Wikipedia passages 上搭建上面的 SQuAD extractive pipeline。手写 10 个问题。衡量答案正确频率。如果 passages 和 questions 干净，应该能看到 7-9 个正确。
2. **中等。** 添加 refusal classifier。当 top retrieval score 低于阈值（比如 0.3 cosine）时，返回 "I don't know"，而不是调用 reader。在 held-out set 上调阈值。
3. **困难。** 在你选择的 10,000 文档 corpus 上构建 RAG pipeline。实现 hybrid retrieval（BM25 + dense）和 RRF fusion（见第 14 课）。衡量有无 hybrid step 的 answer accuracy。记录哪些问题类型收益最大。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|-----------------|-----------------------|
| Extractive QA | 找答案 span | 在给定 passage 内预测答案的 start 和 end indices。 |
| Open-domain QA | 在语料上 QA | 不给 passage；必须先检索再回答。 |
| RAG | 先检索再生成 | Retrieval-augmented generation。Retriever + reader pipeline。 |
| SQuAD | 经典 benchmark | Stanford Question Answering Dataset。EM + F1 指标。 |
| Hallucination | 编造答案 | Reader 输出不受 retrieved context 支持。 |
| Refusal calibration | 知道何时闭嘴 | 系统无法回答时正确说 "I don't know"。 |

## 延伸阅读

- [Rajpurkar et al. (2016). SQuAD: 100,000+ Questions for Machine Comprehension of Text](https://arxiv.org/abs/1606.05250) — benchmark 论文。
- [Karpukhin et al. (2020). Dense Passage Retrieval for Open-Domain QA](https://arxiv.org/abs/2004.04906) — DPR，QA 中的经典 dense retriever。
- [Lewis et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401) — 命名 RAG 的论文。
- [Gao et al. (2023). Retrieval-Augmented Generation for Large Language Models: A Survey](https://arxiv.org/abs/2312.10997) — 全面的 RAG survey。
