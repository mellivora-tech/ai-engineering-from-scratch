# Relation Extraction 与 Knowledge Graph 构建

> NER 找到了 entities。Entity linking 把它们锚定。Relation extraction 找出它们之间的边。Knowledge graph 是 nodes、edges 和 provenance 的总和。

**类型：** 构建
**语言：** Python
**前置要求：** 阶段 5 · 06（NER），阶段 5 · 25（Entity Linking）
**时间：** ~60 分钟

## 问题

分析师读到：“Tim Cook became CEO of Apple in 2011.” 四个事实：

- `(Tim Cook, role, CEO)`
- `(Tim Cook, employer, Apple)`
- `(Tim Cook, start_date, 2011)`
- `(Apple, type, Organization)`

Relation Extraction（RE）把自由文本变成结构化 triples `(subject, relation, object)`。在整个 corpus 上聚合，你就得到 knowledge graph。继续聚合和查询，你就得到 RAG、analytics 或 compliance audits 的 reasoning substrate。

2026 年的问题：LLMs 会很热情地抽取 relations。太热情了。它们会 hallucinate 源文本不支持的 triples。没有 provenance，你无法区分真实 triples 和听起来合理的虚构。2026 年答案是 AEVS-style anchor-and-verify pipelines。

## 概念

![Text → triples → knowledge graph](../assets/relation-extraction.svg)

**Triple form。** `(subject_entity, relation_type, object_entity)`。Relations 来自 closed ontology（Wikidata properties、FIBO、UMLS）或 open set（OpenIE-style，任何短语都可以）。

**三种 extraction approaches。**

1. **Rule / pattern-based。** Hearst patterns：“X such as Y” → `(Y, isA, X)`。加手写 regex。脆、精确、可解释。
2. **Supervised classifier。** 给定句子中的两个 entity mentions，从固定集合中预测 relation。训练于 TACRED、ACE、KBP。2015-2022 标准。
3. **Generative LLM。** Prompt 模型输出 triples。开箱可用。必须带 provenance，否则会 hallucinate 看似合理的垃圾。

**AEVS（Anchor-Extraction-Verification-Supplement, 2026）。** 当前 hallucination mitigation framework：

- **Anchor。** 标出每个 entity span 和 relation-phrase span 的精确位置。
- **Extract。** 生成 linked to anchor spans 的 triples。
- **Verify。** 把每个 triple 元素匹配回 source text；拒绝任何 unsupported 内容。
- **Supplement。** Coverage pass 确保没有 anchored span 被漏掉。

Hallucinations 会显著下降。它需要更多计算，但可审计。

**Open-vs-closed tradeoff。**

- **Closed ontology。** 固定 property list（例如 Wikidata 11,000+ properties）。可预测、可查询、难以编造。
- **Open IE。** 任何 verbal phrase 都能成为 relation。高 recall、低 precision，难以查询。

生产 KG 通常混合：用 open IE 做 discovery，再把 relations canonicalize 到 closed ontology 后合并进主图。

## 构建它

### 第 1 步：pattern-based extraction

```python
PATTERNS = [
    (r"(?P<s>[A-Z]\w+) (?:is|was) (?:a|an|the) (?P<o>[A-Z]?\w+)", "isA"),
    (r"(?P<s>[A-Z]\w+) (?:is|was) born in (?P<o>\w+)", "bornIn"),
    (r"(?P<s>[A-Z]\w+) works? (?:at|for) (?P<o>[A-Z]\w+)", "worksAt"),
    (r"(?P<s>[A-Z]\w+) founded (?P<o>[A-Z]\w+)", "founded"),
]
```

完整 toy extractor 见 `code/main.py`。Hearst patterns 在 domain-specific pipelines 中仍然交付，因为它们可调试。

### 第 2 步：supervised relation classification

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tok = AutoTokenizer.from_pretrained("Babelscape/rebel-large")
model = AutoModelForSequenceClassification.from_pretrained("Babelscape/rebel-large")

text = "Tim Cook was born in Alabama. He later became CEO of Apple."
encoded = tok(text, return_tensors="pt", truncation=True)
output = model.generate(**encoded, max_length=200)
triples = tok.batch_decode(output, skip_special_tokens=False)
```

REBEL 是 seq2seq relation extractor：text in，triples out，并且已经是 Wikidata property ids。它在 distant-supervision data 上 fine-tuned，是标准 open-weights baseline。

### 第 3 步：带 anchoring 的 LLM-prompted extraction

```python
prompt = f"""Extract (subject, relation, object) triples from the text.
For each triple, include the exact character span in the source text.

Text: {text}

Output JSON:
[{{"subject": {{"text": "...", "span": [start, end]}},
   "relation": "...",
   "object": {{"text": "...", "span": [start, end]}}}}, ...]

Only include triples fully supported by the text. No inference beyond what is stated.
"""
```

验证每个返回 span 是否匹配 source。拒绝任何 `text[start:end] != triple_entity` 的内容。这是最小形式的 AEVS “verify” 步骤。

### 第 4 步：canonicalize 到 closed ontology

```python
RELATION_MAP = {
    "is the CEO of": "P169",       # "chief executive officer"
    "was born in":   "P19",         # "place of birth"
    "founded":        "P112",       # "founded by" (inverted subject/object)
    "works at":       "P108",       # "employer"
}


def canonicalize(relation):
    rel_low = relation.lower().strip()
    if rel_low in RELATION_MAP:
        return RELATION_MAP[rel_low]
    return None   # drop unmapped open relations or route to manual review
```

Canonicalization 往往占工程工作 60-80%。要为它留预算。

### 第 5 步：构建小图并查询

```python
triples = extract(text)
graph = {}
for s, r, o in triples:
    graph.setdefault(s, []).append((r, o))


def neighbors(node, relation=None):
    return [(r, o) for r, o in graph.get(node, []) if relation is None or r == relation]


print(neighbors("Tim Cook", relation="P108"))    # -> [(P108, Apple)]
```

这是每个 RAG-over-KG 系统的原子。用 RDF triple stores（Blazegraph、Virtuoso）、property graphs（Neo4j）或 vector-augmented graph stores 扩展它。

## 坑

- **Coreference before RE。** “He founded Apple”——RE 需要知道 “he” 是谁。先跑 coref（第 24 课）。
- **Entity canonicalization。** “Apple Inc” 和 “Apple” 必须解析到同一个 node。先做 entity linking（第 25 课）。
- **Hallucinated triples。** LLMs 输出文本不支持的 triples。强制 span verification。
- **Relation canonicalization drift。** Open IE relations 不一致（“was born in,” “came from,” “is a native of”）。Collapse 到 canonical ids，否则图不可查询。
- **Temporal errors。** “Tim Cook is CEO of Apple”——现在为真，2005 年为假。很多 relations 有时间边界。使用 qualifiers（Wikidata 中 `P580` start time、`P582` end time）。
- **Domain mismatch。** REBEL 在 Wikipedia 上训练。Legal、medical、scientific text 常常需要 domain-fine-tuned RE models。

## 使用它

2026 年技术栈：

| 场景 | 选择 |
|-----------|------|
| Fast production，general domain | REBEL 或 LlamaPred + Wikidata canonicalization |
| Domain-specific（biomed、legal） | SciREX-style domain fine-tune + custom ontology |
| LLM-prompted，audited output | AEVS pipeline：anchor → extract → verify → supplement |
| High-volume news IE | Pattern-based + supervised hybrid |
| 从零构建 KG | Open IE + manual canonicalization pass |
| Temporal KG | Extract with qualifiers（start/end time、point in time） |

集成模式：NER → coref → entity linking → relation extraction → ontology mapping → graph load。每一阶段都可以是质量门。

## 交付它

保存为 `outputs/skill-re-designer.md`：

```markdown
---
name: re-designer
description: 设计带 provenance 和 canonicalization 的 relation extraction pipeline。
version: 1.0.0
phase: 5
lesson: 26
tags: [nlp, relation-extraction, knowledge-graph]
---

给定 corpus（domain、language、volume）和 downstream use（KG-RAG、analytics、compliance），输出：

1. Extractor。Pattern-based / supervised / LLM / AEVS hybrid。理由关联 precision vs recall target。
2. Ontology。Closed property list（Wikidata / domain）或 open IE + canonicalization pass。
3. Provenance。每个 triple 携带 source char-span + doc id。审计场景不可协商。
4. Merge strategy。Canonical entity id + relation id + temporal qualifiers；dedup policy。
5. Evaluation。在 200 个 hand-labelled triples 上测 precision / recall + LLM-extracted sample 上 hallucination-rate。

拒绝没有 span verification（source provenance）的 LLM-based RE pipeline。拒绝 open-IE output 未经 canonicalization 进入 production graph。标记 time-bounded relations（employer、spouse、position）缺少 temporal qualifier 的 pipelines。
```

## 练习

1. **简单。** 在 5 个新闻句子上运行 `code/main.py` 中的 pattern extractor。手工检查 precision。
2. **中等。** 在相同句子上使用 REBEL（或小型 LLM）。比较 triples。哪个 extractor precision 更高？哪个 recall 更高？
3. **困难。** 构建 AEVS pipeline：用 LLM extract，再把 spans 和 source 验证。测量 50 个 Wikipedia-style sentences 上 verify 前后的 hallucination rate。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|-----------------|-----------------------|
| Triple | Subject-relation-object | `(s, r, o)` tuple，是 KG 的原子单位。 |
| Open IE | 什么都抽 | Open-vocabulary relation phrases；高 recall、低 precision。 |
| Closed ontology | 固定 schema | 有界 relation types（Wikidata、UMLS、FIBO）。 |
| Canonicalization | 全部归一化 | 把 surface names / relations 映射到 canonical ids。 |
| AEVS | Grounded extraction | Anchor-Extraction-Verification-Supplement pipeline（2026）。 |
| Provenance | Source-of-truth link | 每个 triple 携带 source doc id + char-span。 |
| Distant supervision | 廉价标签 | 把文本和已有 KG 对齐来创建训练数据。 |

## 延伸阅读

- [Mintz et al. (2009). Distant supervision for relation extraction without labeled data](https://www.aclweb.org/anthology/P09-1113.pdf) — distant-supervision 论文。
- [Huguet Cabot, Navigli (2021). REBEL: Relation Extraction By End-to-end Language generation](https://aclanthology.org/2021.findings-emnlp.204.pdf) — seq2seq RE workhorse。
- [Wadden et al. (2019). Entity, Relation, and Event Extraction with Contextualized Span Representations (DyGIE++)](https://arxiv.org/abs/1909.03546) — joint IE。
- [AEVS — Anchor-Extraction-Verification-Supplement framework](https://www.mdpi.com/2073-431X/15/3/178) — 2026 hallucination-mitigation design。
- [Wikidata SPARQL tutorial](https://www.wikidata.org/wiki/Wikidata:SPARQL_tutorial) — canonical graph queries。
