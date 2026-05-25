# Entity Linking 与 Disambiguation

> NER 找到了 “Paris”。Entity linking 决定它是：Paris, France？Paris Hilton？Paris, Texas？Paris（Trojan prince）？没有 linking，你的 knowledge graph 仍然含糊。

**类型：** 构建
**语言：** Python
**前置要求：** 阶段 5 · 06（NER），阶段 5 · 24（Coreference Resolution）
**时间：** ~60 分钟

## 问题

一句话写着：“Jordan beat the press.” 你的 NER 把 “Jordan” 标成 PERSON。很好。但这是哪个 Jordan？

- Michael Jordan（篮球）？
- Michael B. Jordan（演员）？
- Michael I. Jordan（伯克利 ML 教授；是的，ML papers 里真有这种混淆）？
- Jordan（国家）？
- Jordan（希伯来 first name）？

Entity linking（EL）把每个 mention 解析到 knowledge base 中的唯一 entry：Wikidata、Wikipedia、DBpedia 或你的 domain KB。两个子任务：

1. **Candidate generation。** 给定 “Jordan”，哪些 KB entries 可能相关？
2. **Disambiguation。** 给定上下文，哪个 candidate 是正确的？

两个步骤都能学习。两个步骤都有 benchmark。组合 pipeline 已稳定十年；变化的是 disambiguator 质量。

## 概念

![Entity linking pipeline: mention → candidates → disambiguated entity](../assets/entity-linking.svg)

**Candidate generation。** 给定 mention surface form（“Jordan”），在 alias index 里查 candidates。Wikipedia alias dictionaries 覆盖大多数 named entities：“JFK” → John F. Kennedy、Jacqueline Kennedy、JFK airport、JFK（movie）。典型 index 每个 mention 返回 10-30 candidates。

**Disambiguation：三种方式。**

1. **Prior + context（Milne & Witten, 2008）。** `P(entity | mention) × context-similarity(entity, text)`。效果好、快、无需训练。
2. **Embedding-based（ESS / REL / Blink）。** 编码 mention + context。编码每个 candidate description。选最大 cosine。2020-2024 默认。
3. **Generative（GENRE, 2021；LLM-based, 2023+）。** 逐 token decode entity canonical name。约束到 valid entity names 的 trie，确保输出一定是 valid KB id。

**End-to-end vs pipeline。** 现代模型（ELQ、BLINK、ExtEnD、GENRE）在一遍中做 NER + candidate generation + disambiguation。生产中 pipeline systems 仍占主导，因为组件可替换。

### 两个度量

- **Mention recall（candidate gen）。** Gold mentions 中正确 KB entry 出现在 candidate list 里的比例。它是整个 pipeline 的地板。
- **Disambiguation accuracy / F1。** 给定正确 candidates，top-1 有多常是对的。

始终同时报告两者。一个 disambiguation 99%、candidate recall 80% 的系统，整体也只是 80% pipeline。

## 构建它

### 第 1 步：从 Wikipedia redirects 构建 alias index

```python
alias_to_entities = {
    "jordan": ["Q41421 (Michael Jordan)", "Q810 (Jordan, country)", "Q254110 (Michael B. Jordan)"],
    "paris":  ["Q90 (Paris, France)", "Q663094 (Paris, Texas)", "Q55411 (Paris Hilton)"],
    "apple":  ["Q312 (Apple Inc.)", "Q89 (apple, fruit)"],
}
```

Wikipedia alias data：约 18M（alias, entity）pairs。从 Wikidata dumps 下载。存成 inverted index。

### 第 2 步：context-based disambiguation

```python
def disambiguate(mention, context, alias_index, entity_desc):
    candidates = alias_index.get(mention.lower(), [])
    if not candidates:
        return None, 0.0
    context_words = set(tokenize(context))
    best, best_score = None, -1
    for entity_id in candidates:
        desc_words = set(tokenize(entity_desc[entity_id]))
        union = len(context_words | desc_words)
        score = len(context_words & desc_words) / union if union else 0.0
        if score > best_score:
            best, best_score = entity_id, score
    return best, best_score
```

Jaccard overlap 是 toy。替换为 embeddings 上的 cosine similarity（见 `code/main.py` step-2 的 transformer 版本）。

### 第 3 步：embedding-based（BLINK-style）

```python
from sentence_transformers import SentenceTransformer
encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def embed_mention(text, mention_span):
    start, end = mention_span
    marked = f"{text[:start]} [MENTION] {text[start:end]} [/MENTION] {text[end:]}"
    return encoder.encode([marked], normalize_embeddings=True)[0]

def embed_entity(entity_id, description):
    return encoder.encode([f"{entity_id}: {description}"], normalize_embeddings=True)[0]
```

Index time 对每个 KB entity 编码一次。Query time 对 mention + context 编码一次，在 candidate pool 上做 dot-product，选择最大。

### 第 4 步：generative entity linking（概念）

GENRE 逐字符 decode entity 的 Wikipedia title。Constrained decoding（见第 20 课）确保只会输出 valid titles。它与 KB-backed trie 紧密集成。现代后代是 REL-GEN 和带 structured output 的 LLM-prompted EL。

```python
prompt = f"""Text: {text}
Mention: {mention}
List the best Wikipedia title for this mention.
Respond with JSON: {{"title": "..."}}"""
```

结合 whitelist（Outlines `choice`），这是 2026 年最简单可交付 EL pipeline。

### 第 5 步：在 AIDA-CoNLL 上评估

AIDA-CoNLL 是标准 EL benchmark：1,393 篇 Reuters articles，34k mentions，Wikipedia entities。报告 in-KB accuracy（`P@1`）和 out-of-KB NIL-detection rate。

## 坑

- **NIL handling。** 一些 mentions 不在 KB 中（新兴 entities、冷门人物）。系统必须预测 NIL，而不是猜错 entity。单独测量。
- **Mention boundary errors。** 上游 NER 漏掉 partial spans（“Bank of America” 只标成 “Bank”）。EL recall 会掉。
- **Popularity bias。** 训练系统过度预测高频 entities。ML paper 里的 “Michael I. Jordan” 常被链接到篮球 Jordan。
- **Cross-lingual EL。** 把中文文本 mentions 映射到 English Wikipedia entities。需要 multilingual encoder 或 translation step。
- **KB staleness。** 新公司、事件、人物不在去年的 Wikipedia dump 里。Production pipelines 需要 refresh loop。

## 使用它

2026 年技术栈：

| 场景 | 选择 |
|-----------|------|
| General-purpose English + Wikipedia | BLINK 或 REL |
| Cross-lingual，KB = Wikipedia | mGENRE |
| LLM-friendly，每天 mentions 少 | Prompt Claude/GPT-4 with candidate list + constrained JSON |
| Domain-specific KB（medical、legal） | Custom BERT + KB-aware retrieval + domain AIDA-style set 上 fine-tune |
| 极低延迟 | Exact-match prior only（Milne-Witten baseline） |
| Research SOTA | GENRE / ExtEnD / generative LLM-EL |

2026 年交付模式：NER → coref → 对每个 mention 做 EL → 把 clusters collapse 成每个 cluster 一个 canonical entity。输出是文档中每个 entity 一个 KB id，而不是每个 mention 一个 id。

## 交付它

保存为 `outputs/skill-entity-linker.md`：

```markdown
---
name: entity-linker
description: 设计 entity linking pipeline：KB、candidate generator、disambiguator、evaluation。
version: 1.0.0
phase: 5
lesson: 25
tags: [nlp, entity-linking, knowledge-graph]
---

给定 use case（domain KB、language、volume、latency budget），输出：

1. Knowledge base。Wikidata / Wikipedia / custom KB。Version date。Refresh cadence。
2. Candidate generator。Alias-index、embedding 或 hybrid。目标 mention recall @ K。
3. Disambiguator。Prior + context、embedding-based、generative 或 LLM-prompted。
4. NIL strategy。Top score threshold、classifier 或 explicit NIL candidate。
5. Evaluation。Mention recall @ 30、top-1 accuracy、held-out set 上 NIL-detection F1。

拒绝没有 mention-recall baseline 的 EL pipeline（不知道 candidate gen 是否找到了正确 entity，就无法评估 disambiguator）。拒绝没有 constrained output 到 valid KB ids 的 LLM-prompted EL。标记 popularity bias 会影响 minority entities（如同名冲突）且没有 domain fine-tuning 的系统。
```

## 练习

1. **简单。** 在 10 个 ambiguous mentions（Paris、Jordan、Apple）上实现 `code/main.py` 中的 prior+context disambiguator。手工标注正确 entity。测量 accuracy。
2. **中等。** 用 sentence transformer 编码 50 个 ambiguous mentions。嵌入每个 candidate description。比较 embedding-based disambiguation 和 Jaccard context overlap。
3. **困难。** 构建一个 1k-entity domain KB（例如你公司的员工 + 产品）。实现 NER + EL end-to-end。在 100 条 held-out sentences 上测 precision 和 recall。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|-----------------|-----------------------|
| Entity linking (EL) | 链到 Wikipedia | 把 mention 映射到唯一 KB entry。 |
| Candidate generation | 可能是谁？ | 为 mention 返回 plausible KB entries shortlist。 |
| Disambiguation | 选正确那个 | 用上下文给 candidates 打分并选 winner。 |
| Alias index | 查找表 | Surface form → candidate entities 的映射。 |
| NIL | 不在 KB 中 | 明确预测没有 KB entry 匹配。 |
| KB | Knowledge base | Wikidata、Wikipedia、DBpedia 或你的 domain KB。 |
| AIDA-CoNLL | benchmark | 带 gold entity links 的 1,393 篇 Reuters articles。 |

## 延伸阅读

- [Milne, Witten (2008). Learning to Link with Wikipedia](https://www.cs.waikato.ac.nz/~ihw/papers/08-DM-IHW-LearningToLinkWithWikipedia.pdf) — foundational prior+context approach。
- [Wu et al. (2020). Zero-shot Entity Linking with Dense Entity Retrieval (BLINK)](https://arxiv.org/abs/1911.03814) — embedding-based workhorse。
- [De Cao et al. (2021). Autoregressive Entity Retrieval (GENRE)](https://arxiv.org/abs/2010.00904) — 带 constrained decoding 的 generative EL。
- [Hoffart et al. (2011). Robust Disambiguation of Named Entities in Text (AIDA)](https://www.aclweb.org/anthology/D11-1072.pdf) — benchmark 论文。
- [REL: An Entity Linker Standing on the Shoulders of Giants (2020)](https://arxiv.org/abs/2006.01969) — open production stack。
