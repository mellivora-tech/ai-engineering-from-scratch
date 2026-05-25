# 文本摘要

> Extractive 系统告诉你文档说了什么。Abstractive 系统告诉你作者想表达什么。任务不同，坑也不同。

**类型：** 构建
**语言：** Python
**前置要求：** 阶段 5 第 02 课（BoW + TF-IDF）、阶段 5 第 11 课（Machine Translation）
**时间：** ~75 分钟

## 问题

一篇 2,000 词新闻文章出现在你的信息流里。你需要 120 个词概括它。你可以从文章中挑出三个最重要句子（extractive），也可以用自己的话改写内容（abstractive）。二者都叫 summarization。它们是完全不同的问题。

Extractive summarization 是排序问题。给每个句子打分，返回 top-`k`。输出永远是符合语法的，因为它是逐字抬出来的。风险是漏掉分散在文章各处的内容。

Abstractive summarization 是生成问题。Transformer 在输入条件下生成新文本。输出流畅且压缩率高，但可能 hallucinate 源文本中不存在的事实。风险是自信地编造。

本课会构建二者，并说明它们各自拥有的失败模式。

## 概念

![Extractive TextRank vs abstractive transformer](../assets/summarization.svg)

**Extractive。** 把文章看作图：节点是句子，边是相似度。在图上运行 PageRank（或类似算法），按句子和其他句子的连接程度打分。最高分句子就是摘要。经典实现是 **TextRank**（Mihalcea and Tarau, 2004）。

**Abstractive。** 在 document-summary pairs 上 fine-tune transformer encoder-decoder（BART、T5、Pegasus）。推理时，模型读取文档，并通过 cross-attention 逐 token 生成摘要。Pegasus 尤其使用 gap-sentence pretraining objective，所以不需要太多 fine-tuning 就很擅长摘要。

评估使用 **ROUGE**（Recall-Oriented Understudy for Gisting Evaluation）。ROUGE-1 和 ROUGE-2 计算 unigram、bigram overlap。ROUGE-L 计算 longest common subsequence。越高越好，但 40 ROUGE-L 是“好”，50 是“非常好”。每篇论文都会报告三者。使用 `rouge-score` 包。

## 构建它

### 第 1 步：TextRank（extractive）

```python
import math
import re
from collections import Counter


def sentence_split(text):
    return re.split(r"(?<=[.!?])\s+", text.strip())


def similarity(s1, s2):
    w1 = Counter(s1.lower().split())
    w2 = Counter(s2.lower().split())
    intersection = sum((w1 & w2).values())
    denom = math.log(len(w1) + 1) + math.log(len(w2) + 1)
    if denom == 0:
        return 0.0
    return intersection / denom


def textrank(text, top_k=3, damping=0.85, iterations=50, epsilon=1e-4):
    sentences = sentence_split(text)
    n = len(sentences)
    if n <= top_k:
        return sentences

    sim = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                sim[i][j] = similarity(sentences[i], sentences[j])

    scores = [1.0] * n
    for _ in range(iterations):
        new_scores = [1 - damping] * n
        for i in range(n):
            total_out = sum(sim[i]) or 1e-9
            for j in range(n):
                if sim[i][j] > 0:
                    new_scores[j] += damping * sim[i][j] / total_out * scores[i]
        if max(abs(s - ns) for s, ns in zip(scores, new_scores)) < epsilon:
            scores = new_scores
            break
        scores = new_scores

    ranked = sorted(range(n), key=lambda k: scores[k], reverse=True)[:top_k]
    ranked.sort()
    return [sentences[i] for i in ranked]
```

有两件事值得点名。Similarity function 使用 log-normalized word overlap，这是原始 TextRank 变体。TF-IDF vectors 的 cosine 也能用。Damping factor 0.85 和迭代次数是 PageRank 默认值。

### 第 2 步：用 BART 做 abstractive

```python
from transformers import pipeline

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

article = """(long news article text)"""

summary = summarizer(article, max_length=120, min_length=60, do_sample=False)
print(summary[0]["summary_text"])
```

BART-large-CNN 在 CNN/DailyMail 语料上 fine-tuned。开箱会产出新闻风格摘要。对其他领域（科学论文、对话、法律），使用对应 Pegasus checkpoint，或在目标数据上 fine-tune。

### 第 3 步：ROUGE 评估

```python
from rouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
scores = scorer.score(reference_summary, generated_summary)
print({k: round(v.fmeasure, 3) for k, v in scores.items()})
```

永远使用 stemming。没有它，"running" 和 "run" 会被算成不同词，ROUGE 会低估。

### ROUGE 之外（2026 summarization eval）

ROUGE 主导 summarization metric 二十年，但到了 2026 年，单靠它不够。NLG 论文的大规模 meta-analysis 显示：

- **BERTScore**（contextual embedding similarity）在 2023 年前后取得进展，现在多数 summarization 论文会和 ROUGE 一起报告。
- **BARTScore** 把评估视为生成：给定 source，衡量 pretrained BART 给 summary 的 likelihood。
- **MoverScore**（在 contextual embeddings 上做 Earth Mover's Distance）在 2025 summarization benchmarks 中到达第一，因为它比 ROUGE 更好地捕捉语义重叠。
- **FactCC** 和 **QA-based faithfulness** 在 2021-2023 年常见，现在经常被 **G-Eval** 替代（一个用 GPT-4 prompt chain 结合 chain-of-thought reasoning 打分 coherence、consistency、fluency、relevance 的方法）。
- **G-Eval** 及类似 LLM-judge 方法，在 rubrics 设计良好时，与人类判断一致约 80%。

生产建议：为 legacy comparison 报告 ROUGE-L，为语义重叠报告 BERTScore，为 coherence 和 factuality 使用 G-Eval。用 50-100 个人类标注摘要校准。

### 第 4 步：factuality 问题

Abstractive summaries 容易 hallucination。Extractive summaries 的 hallucination 风险低得多，因为输出是从源文本逐字摘出来的；不过，如果源句子被去上下文、过时或引用顺序错乱，仍然可能误导。这是生产系统在合规相邻内容上仍偏好 extractive 方法的最大原因。

要点名的 hallucination 类型：

- **Entity swap。** 源文本说 "John Smith"。摘要说 "John Brown"。
- **Number drift。** 源文本说 "25,000"。摘要说 "25 million"。
- **Polarity flip。** 源文本说 "rejected the offer"。摘要说 "accepted the offer"。
- **Fact invention。** 源文本没有提 CEO。摘要说 CEO 批准了。

有效的评估方法：

- **FactCC。** 在 source sentence 和 summary sentence 的 entailment 上训练的二分类器。预测 factual/not-factual。
- **QA-based factuality。** 让 QA 模型回答答案在 source 中的问题。如果 summary 支持了不同答案，就标记。
- **Entity-level F1。** 比较 source 和 summary 中的 named entities。只出现在 summary 中的实体很可疑。

对任何 factuality 重要的用户可见内容（新闻、医疗、法律、金融），extractive 是更安全默认。Abstractive 需要把 factuality check 放进 loop。

## 使用它

2026 年栈：

| 使用场景 | 推荐 |
|---------|-------------|
| 新闻，3-5 句摘要，英文 | `facebook/bart-large-cnn` |
| 科学论文 | `google/pegasus-pubmed` 或 tuned T5 |
| 多文档、长篇 | 任何 32k+ context 的 LLM，prompted |
| 对话摘要 | `philschmid/bart-large-cnn-samsum` |
| Extractive，按构造低 hallucination 风险 | TextRank 或 `sumy` 的 LSA / LexRank |

当计算不是约束时，长上下文 LLM 在 2026 年经常超过专用模型。取舍是成本和可复现性；专用模型输出更一致。

## 交付它

保存为 `outputs/skill-summary-picker.md`：

```markdown
---
name: summary-picker
description: 选择 extractive 或 abstractive，指定库和 factuality check。
version: 1.0.0
phase: 5
lesson: 12
tags: [nlp, summarization]
---

给定任务（文档类型、合规要求、长度、计算预算），输出：

1. 方法。Extractive 或 abstractive。用一句话说明原因。
2. 起始模型 / library。写出名称。`sumy.TextRankSummarizer`、`facebook/bart-large-cnn`、`google/pegasus-pubmed` 或 LLM prompt。
3. Evaluation plan。ROUGE-1、ROUGE-2、ROUGE-L（使用带 stemming 的 rouge-score）。如果是 abstractive，额外加 factuality check。
4. 要探测的一个失败模式。Entity swap 是 abstractive news summarization 中最常见的；标记 source entities 不出现在 summary 中的样本。

没有 factuality gate 时，拒绝为医疗、法律、金融或受监管内容推荐 abstractive summarization。把超过模型 context window 的输入标记为需要 chunked map-reduce summarization（不能只是截断）。
```

## 练习

1. **简单。** 在 5 篇新闻文章上运行 TextRank。把 top-3 句子和 reference summary 对比。衡量 ROUGE-L。在 CNN/DailyMail 风格文章上应该能看到 30-45 ROUGE-L。
2. **中等。** 实现 entity-level factuality：从 source 和 summary 中抽取 named entities（spaCy），计算 summary 中 source entities 的 recall，以及 summary entities 相对 source 的 precision。高 precision、低 recall 表示安全但简略；低 precision 表示 hallucinated entities。
3. **困难。** 在 50 篇 CNN/DailyMail 文章上对比 BART-large-CNN 和 LLM（Claude 或 GPT-4）。报告 ROUGE-L、factuality（entity F1）和每篇摘要成本。记录各自胜出的地方。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|-----------------|-----------------------|
| Extractive | 挑句子 | 从源文本逐字返回句子。不会 hallucinate。 |
| Abstractive | 改写 | 在 source 条件下生成新文本。可能 hallucinate。 |
| ROUGE | 摘要指标 | 系统输出和 reference 之间的 n-gram / LCS overlap。 |
| TextRank | Graph-based extractive | 在句子相似度图上运行 PageRank。 |
| Factuality | 是否正确 | Summary 中的 claims 是否受到 source 支持。 |
| Hallucination | 编造内容 | Summary 中出现源文本不支持的内容。 |

## 延伸阅读

- [Mihalcea and Tarau (2004). TextRank: Bringing Order into Texts](https://aclanthology.org/W04-3252/) — extractive 经典论文。
- [Lewis et al. (2019). BART: Denoising Sequence-to-Sequence Pre-training](https://arxiv.org/abs/1910.13461) — BART 论文。
- [Zhang et al. (2019). PEGASUS: Pre-training with Extracted Gap-sentences](https://arxiv.org/abs/1912.08777) — Pegasus 和 gap-sentence objective。
- [Lin (2004). ROUGE: A Package for Automatic Evaluation of Summaries](https://aclanthology.org/W04-1013/) — ROUGE 论文。
- [Maynez et al. (2020). On Faithfulness and Factuality in Abstractive Summarization](https://arxiv.org/abs/2005.00661) — factuality landscape 论文。
