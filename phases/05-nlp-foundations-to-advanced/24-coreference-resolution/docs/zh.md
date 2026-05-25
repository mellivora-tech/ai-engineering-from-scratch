# Coreference Resolution

> “She called him. He did not answer. The doctor was at lunch.” 三个指代，两个真人，没有人被点名。Coreference resolution 找出谁是谁。

**类型：** 学习
**语言：** Python
**前置要求：** 阶段 5 · 06（NER），阶段 5 · 07（POS & Parsing）
**时间：** ~60 分钟

## 问题

从一篇 300 词文章中抽取 Apple Inc. 的所有 mention。文章写 “Apple” 时很容易。文章写 “the company”、“they”、“Cupertino's technology giant” 或 “Jobs's firm” 时就很难。若不把这些 mentions 解析到同一 entity，你的 NER pipeline 会漏掉 60-80% 的 mentions。

Coreference resolution 把所有指向同一真实世界 entity 的表达链接成一个 cluster。它是 surface-level NLP（NER、parsing）和 downstream semantics（IE、QA、summarization、KG）之间的胶水。

它在 2026 年重要的原因：

- Summarization：“The CEO announced...” vs “Tim Cook announced...”——摘要应该说出 CEO 的名字。
- Question answering：“Who did she call?” 需要解析 “she”。
- Information extraction：knowledge graph 里把 “PER1 founded Apple” 和 “Jobs founded Apple” 当成两条不同 entries 是错的。
- Multi-document IE：合并多篇文章中关于同一事件的 mentions，就是 cross-document coreference。

## 概念

![Coreference clustering: mentions → entities](../assets/coref.svg)

**任务。** 输入：一个 document。输出：mentions（spans）的 clustering，每个 cluster 指向一个 entity。

**Mention types。**

- **Named entity。** “Tim Cook”
- **Nominal。** “the CEO”, “the company”
- **Pronominal。** “he”, “she”, “they”, “it”
- **Appositive。** “Tim Cook, Apple's CEO,”

**架构。**

1. **Rule-based（Hobbs, 1978）。** 基于 syntactic tree 的 pronoun resolution，使用 grammar rules。强 baseline。在 pronouns 上出人意料地难超越。
2. **Mention-pair classifier。** 对每对 mentions（m_i, m_j）预测它们是否 corefer。通过 transitive closure 聚类。2016 前标准。
3. **Mention-ranking。** 对每个 mention，给 candidate antecedents（包括 “no antecedent”）排序。选 top。
4. **Span-based end-to-end（Lee et al., 2017）。** Transformer encoder。枚举长度上限内所有 candidate spans。预测 mention scores。对每个 span 预测 antecedent probability。贪心聚类。现代默认。
5. **Generative（2024+）。** Prompt LLM：“List every pronoun in this text and its antecedent.” 简单 cases 表现不错，长文档和稀有 referents 上挣扎。

**评估指标。** 有五个标准 metrics（MUC、B³、CEAF、BLANC、LEA），因为没有单一指标能捕捉 clustering quality。通常报告前三者平均作为 CoNLL F1。2026 年 CoNLL-2012 上 SOTA 约 83 F1。

**已知困难 cases。**

- 指向几页前引入 entities 的 definite descriptions。
- Bridging anaphora（“the wheels” → 之前提到的一辆 car）。
- 中文、日语等语言中的 zero anaphora。
- Cataphora（pronoun 在 referent 前）： “When **she** walked in, Mary smiled.”

## 构建它

### 第 1 步：pretrained neural coreference（AllenNLP / spaCy-experimental）

```python
import spacy
nlp = spacy.load("en_coreference_web_trf")   # experimental model
doc = nlp("Apple announced new products. The company said they would ship soon.")
for cluster in doc._.coref_clusters:
    print(cluster, "->", [m.text for m in cluster])
```

在更长文档上，你会得到类似：
- Cluster 1: [Apple, The company, they]
- Cluster 2: [new products]

### 第 2 步：rule-based pronoun resolver（教学）

`code/main.py` 中有 stdlib-only 实现：

1. 抽取 mentions：named entities（capitalized spans）、pronouns（dict lookup）、definite descriptions（“the X”）。
2. 对每个 pronoun，查看前 K 个 mentions，并按以下项打分：
   - gender/number agreement（heuristic）
   - recency（越近越好）
   - syntactic role（subjects 优先）
3. 链接最高分 antecedent。

它无法与 neural models 竞争。但它展示了搜索空间和 end-to-end model 必须做的决策。

### 第 3 步：用 LLM 做 coreference

```python
prompt = f"""Text: {text}

List every pronoun and noun phrase that refers to a person or company.
Cluster them by what they refer to. Output JSON:
[{{"entity": "Apple", "mentions": ["Apple", "the company", "it"]}}, ...]
"""
```

注意两种失败。第一，LLMs 会 over-merge（把指向两个不同人的 “him” 和 “her” 合并）。第二，LLMs 在长文档中会静默漏掉 mentions。始终用 span-offset checks 验证。

### 第 4 步：evaluation

标准 conll-2012 script 计算 MUC、B³、CEAF-φ4 并报告平均值。内部 eval 可以从 annotated test set 上的 span-level precision/recall 开始，再加入 mention-linking F1。

## 坑

- **Singleton explosion。** 一些系统把每个 mention 都报告成自己的 cluster。B³ 宽松，MUC 会惩罚。始终检查三个 metrics。
- **长上下文 pronouns。** 文档超过 2,000 tokens 后，性能掉约 15 F1。要仔细 chunk。
- **Gender assumptions。** 硬编码 gender rules 会在 non-binary referents、organizations、animals 上失效。使用 learned models 或 neutral scoring。
- **LLM drift on long docs。** 单次 API call 无法可靠地聚类 50+ paragraphs 的 mentions。用 sliding-window + merge。

## 使用它

2026 年技术栈：

| 场景 | 选择 |
|-----------|------|
| English, single document | `en_coreference_web_trf`（spaCy-experimental）或 AllenNLP neural coref |
| Multilingual | 在 OntoNotes 或 Multilingual CoNLL 上训练的 SpanBERT / XLM-R |
| Cross-document event coref | Specialized end-to-end models（2025-26 SOTA） |
| 快速 LLM baseline | GPT-4o / Claude + structured-output coref prompt |
| Production dialog systems | Rule-based fallback + neural primary + critical slots manual review |

2026 年实际交付的集成模式：先跑 NER，再跑 coref，把 coref clusters 合并进 NER entities。下游任务看到的是每个 cluster 一个 entity，而不是每个 mention 一个 entity。

## 交付它

保存为 `outputs/skill-coref-picker.md`：

```markdown
---
name: coref-picker
description: 选择 coreference approach、evaluation plan 和 integration strategy。
version: 1.0.0
phase: 5
lesson: 24
tags: [nlp, coref, information-extraction]
---

给定 use case（single-doc / multi-doc、domain、language），输出：

1. Approach。Rule-based / neural span-based / LLM-prompted / hybrid。一句话说明理由。
2. Model。如果是 neural，命名 checkpoint。
3. Integration。操作顺序：tokenize → NER → coref → downstream task。
4. Evaluation。在 held-out set 上测 CoNLL F1（MUC + B³ + CEAF-φ4 average）+ 对 20 个 documents 做 manual cluster review。

拒绝对超过 2,000 tokens 的文档使用 LLM-only coref，除非有 sliding-window merge。拒绝任何没有 mention-level precision-recall report 的 pipeline。标记部署在 demographically diverse text 上的 gender-heuristic systems。
```

## 练习

1. **简单。** 在 5 个手写 paragraphs 上运行 `code/main.py` 中的 rule-based resolver。对照 ground truth 测 mention-link accuracy。
2. **中等。** 在新闻文章上使用 pretrained neural coref model。把 clusters 和你自己的人工标注比较。它在哪里失败？
3. **困难。** 构建 coref-enhanced NER pipeline：先 NER，再通过 coref clusters 合并。在 100 篇文章上测量相对 NER-only 的 entity-coverage improvement。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|-----------------|-----------------------|
| Mention | 一个引用 | 指向 entity 的文本 span（名字、代词、名词短语）。 |
| Antecedent | “it” 指的东西 | 后一个 mention corefer 的更早 mention。 |
| Cluster | Entity 的 mentions | 全部指向同一真实世界 entity 的 mentions 集合。 |
| Anaphora | 向后指代 | 后面的 mention 指向前面的 referent（“he” → “John”）。 |
| Cataphora | 向前指代 | 前面的 mention 指向后面的 referent（“When he arrived, John...”）。 |
| Bridging | 隐式引用 | “I bought a car. The wheels were bad.”（那辆 car 的 wheels。） |
| CoNLL F1 | Leaderboards 上的数字 | MUC、B³、CEAF-φ4 F1 scores 的平均。 |

## 延伸阅读

- [Jurafsky & Martin, SLP3 Ch. 26 — Coreference Resolution and Entity Linking](https://web.stanford.edu/~jurafsky/slp3/26.pdf) — 经典教材章节。
- [Lee et al. (2017). End-to-end Neural Coreference Resolution](https://arxiv.org/abs/1707.07045) — span-based end-to-end。
- [Joshi et al. (2020). SpanBERT](https://arxiv.org/abs/1907.10529) — 提升 coref 的预训练方法。
- [Pradhan et al. (2012). CoNLL-2012 Shared Task](https://aclanthology.org/W12-4501/) — benchmark。
- [Hobbs (1978). Resolving Pronoun References](https://www.sciencedirect.com/science/article/pii/0024384178900064) — rule-based 经典。
