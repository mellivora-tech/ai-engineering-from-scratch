# RAG 的 Chunking 策略

> Chunking 配置对 retrieval 质量的影响和 embedding model 选择一样大（Vectara NAACL 2025）。Chunking 做错，再多 reranking 也救不回来。

**类型：** 构建
**语言：** Python
**前置要求：** 阶段 5 · 14（信息检索），阶段 5 · 22（Embedding Models）
**时间：** ~60 分钟

## 问题

你把一份 50 页合同放进 RAG 系统。用户问：“What is the termination clause?” Retriever 返回封面页。为什么？因为模型在 512-token chunks 上训练，而 termination clause 位于第 20 页，跨越一个 page break，局部没有把它和 query 绑定起来的关键词。

修复方法不是“买更好的 embedding model”。修复方法是 chunking。多大？要不要 overlap？在哪里 split？是否带周围 context？

2026 年 2 月 benchmarks 显示了令人意外的结果：

- Vectara 2026 study：recursive 512-token chunking 击败 semantic chunking，accuracy 69% → 54%。
- SPLADE + Mistral-8B on Natural Questions：overlap 没有提供可测收益。
- Context cliff：response quality 在约 2,500 tokens context 附近急剧下降。

“显而易见”的答案（semantic chunking、20% overlap、1000 tokens）经常是错的。本课建立六种策略的直觉，并告诉你何时使用哪一种。

## 概念

![Six chunking strategies visualized on one passage](../assets/chunking.svg)

**Fixed chunking。** 每 N 个字符或 tokens 切分。最简单 baseline。会切断句子。压缩好，coherence 差。

**Recursive。** LangChain 的 `RecursiveCharacterTextSplitter`。先试 `\n\n`，再试 `\n`，再试 `.`，最后空格。回退干净。2026 默认。

**Semantic。** 编码每个句子。计算相邻句子的 cosine similarity。在 similarity 低于阈值处 split。保留 topic coherence。更慢；有时产生 40-token 小碎片，伤害 retrieval。

**Sentence。** 按句子边界切分。一个句子一个 chunk，或 N 句窗口。到约 5k tokens 前能接近 semantic chunking，成本低得多。

**Parent-document。** 存小 child chunks 用于 retrieval，同时存更大的 parent chunk 用于 context。按 child 检索，返回 parent。降级优雅：即使 child chunks 不完美，也会返回合理 parents。

**Late chunking（2024）。** 先在 token level 编码整个文档，再把 token embeddings pool 成 chunk embeddings。保留跨 chunk context。适用于 long-context embedders（BGE-M3、Jina v3）。计算更高。

**Contextual retrieval（Anthropic, 2024）。** 给每个 chunk 前面加上 LLM 生成的上下文摘要（“This chunk is section 3.2 of the termination clauses...”）。Anthropic 自家 benchmark 中 retrieval 提升 35-50%。索引成本高。

### 胜过所有默认值的规则

让 chunk size 匹配 query type：

| Query type | Chunk size |
|------------|-----------|
| Factoid（“CEO 的名字是什么？”） | 256-512 tokens |
| Analytical / multi-hop | 512-1024 tokens |
| Whole-section comprehension | 1024-2048 tokens |

这是 NVIDIA 2026 benchmark 的结论。Chunk 应该足够大，包含答案和局部 context；也要足够小，让 retriever top-K 聚焦答案而不是 context noise。

## 构建它

### 第 1 步：fixed 和 recursive chunking

```python
def chunk_fixed(text, size=512, overlap=0):
    step = size - overlap
    return [text[i:i + size] for i in range(0, len(text), step)]


def chunk_recursive(text, size=512, seps=("\n\n", "\n", ". ", " ")):
    if len(text) <= size:
        return [text]
    for sep in seps:
        if sep not in text:
            continue
        parts = text.split(sep)
        chunks = []
        buf = ""
        for p in parts:
            if len(p) > size:
                if buf:
                    chunks.append(buf)
                    buf = ""
                chunks.extend(chunk_recursive(p, size=size, seps=seps[1:] or (" ",)))
                continue
            candidate = buf + sep + p if buf else p
            if len(candidate) <= size:
                buf = candidate
            else:
                if buf:
                    chunks.append(buf)
                buf = p
        if buf:
            chunks.append(buf)
        return [c for c in chunks if c.strip()]
    return chunk_fixed(text, size)
```

### 第 2 步：semantic chunking

```python
def chunk_semantic(text, encoder, threshold=0.6, min_chars=200, max_chars=2048):
    sentences = split_sentences(text)
    if not sentences:
        return []
    embs = encoder.encode(sentences, normalize_embeddings=True)
    chunks = [[sentences[0]]]
    for i in range(1, len(sentences)):
        sim = float(embs[i] @ embs[i - 1])
        current_len = sum(len(s) for s in chunks[-1])
        if sim < threshold and current_len >= min_chars:
            chunks.append([sentences[i]])
        else:
            chunks[-1].append(sentences[i])

    result = []
    for group in chunks:
        text_group = " ".join(group)
        if len(text_group) > max_chars:
            result.extend(chunk_recursive(text_group, size=max_chars))
        else:
            result.append(text_group)
    return result
```

在你的领域上调 `threshold`。太高会碎片化；太低会变成一个巨 chunk。

### 第 3 步：parent-document

```python
def chunk_parent_child(text, parent_size=2048, child_size=256):
    parents = chunk_recursive(text, size=parent_size)
    mapping = []
    for p_idx, parent in enumerate(parents):
        children = chunk_recursive(parent, size=child_size)
        for child in children:
            mapping.append({"child": child, "parent_idx": p_idx, "parent": parent})
    return mapping


def retrieve_parent(child_query, mapping, encoder, top_k=3):
    child_embs = encoder.encode([m["child"] for m in mapping], normalize_embeddings=True)
    q_emb = encoder.encode([child_query], normalize_embeddings=True)[0]
    scores = child_embs @ q_emb
    top = np.argsort(-scores)[:top_k]
    seen, parents = set(), []
    for i in top:
        if mapping[i]["parent_idx"] not in seen:
            parents.append(mapping[i]["parent"])
            seen.add(mapping[i]["parent_idx"])
    return parents
```

关键洞见：对 parents 去重。多个 children 可以映射到同一个 parent；全部返回会浪费 context。

### 第 4 步：contextual retrieval（Anthropic pattern）

```python
def contextualize_chunks(document, chunks, llm):
    context_prompts = [
        f"""<document>{document}</document>
Here is the chunk to situate: <chunk>{c}</chunk>
Write 50-100 words placing this chunk in the document's context."""
        for c in chunks
    ]
    contexts = llm.batch(context_prompts)
    return [f"{ctx}\n\n{c}" for ctx, c in zip(contexts, chunks)]
```

索引 contextualized chunks。Query time 时，retrieval 会受益于额外的周边信号。

### 第 5 步：evaluate

```python
def recall_at_k(queries, corpus_chunks, encoder, k=5):
    chunk_embs = encoder.encode(corpus_chunks, normalize_embeddings=True)
    hits = 0
    for q_text, gold_idxs in queries:
        q_emb = encoder.encode([q_text], normalize_embeddings=True)[0]
        top = np.argsort(-(chunk_embs @ q_emb))[:k]
        if any(i in gold_idxs for i in top):
            hits += 1
    return hits / len(queries)
```

始终 benchmark。你的 corpus 的“最佳”策略可能和任何 blog post 都不一样。

## 坑

- **只在 factoid queries 上评估 chunking。** Multi-hop queries 会揭示完全不同的赢家。使用按 query type 分层的 eval set。
- **Semantic chunking 没有最小 size。** 会产生伤害 retrieval 的 40-token fragments。始终强制 `min_tokens`。
- **把 overlap 当信仰。** 2026 研究发现 overlap 往往零收益却让 index cost 翻倍。测量，不要假设。
- **没有 min/max enforcement。** 5 tokens 或 5000 tokens 的 chunks 都会破坏 retrieval。要 clamp。
- **Cross-doc chunking。** 永远不要让一个 chunk 跨两个 documents。始终 per-doc chunk，再 merge。

## 使用它

2026 年技术栈：

| 场景 | 策略 |
|-----------|----------|
| 初次构建，未知 corpus | Recursive，512 tokens，无 overlap |
| Factoid QA | Recursive，256-512 tokens |
| Analytical / multi-hop | Recursive，512-1024 tokens + parent-document |
| Heavy cross-reference（contracts、papers） | Late chunking 或 contextual retrieval |
| Conversational / dialog corpus | Turn-level chunks + speaker metadata |
| Short utterances（tweets、reviews） | 一个 document = 一个 chunk |

从 recursive 512 开始。用 50-query eval set 测 recall@5。然后再调。

## 交付它

保存为 `outputs/skill-chunker.md`：

```markdown
---
name: chunker
description: 为给定 corpus 和 query distribution 选择 chunking strategy、size 和 overlap。
version: 1.0.0
phase: 5
lesson: 23
tags: [nlp, rag, chunking]
---

给定 corpus（document types、avg length、domain）和 query distribution（factoid / analytical / multi-hop），输出：

1. Strategy。Recursive / sentence / semantic / parent-document / late / contextual。说明理由。
2. Chunk size。Token count。理由关联 query type。
3. Overlap。默认 0；如果 >0 必须说明理由。
4. Min/max enforcement。`min_tokens`、`max_tokens` guards。
5. Evaluation plan。在 50-query stratified eval set（factoid、analytical、multi-hop）上测 Recall@5。

拒绝任何没有 min/max chunk size enforcement 的 chunking strategy。拒绝 overlap above 20%，除非有 ablation 证明它有帮助。标记没有 min-token floor 的 semantic chunking recommendations。
```

## 练习

1. **简单。** 用 fixed(512, 0)、recursive(512, 0)、recursive(512, 100) 对一份 20 页文档做 chunking。比较 chunk counts 和 boundary quality。
2. **中等。** 在 5 个文档上构建 30-query eval set。测量 recursive、semantic 和 parent-document 的 recall@5。哪个胜出？是否符合 blog posts？
3. **困难。** 实现 contextual retrieval。测量相对 baseline recursive 的 MRR improvement。报告 index cost（LLM calls）和 accuracy gain。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|-----------------|-----------------------|
| Chunk | 文档的一块 | 被 embed、index、retrieve 的 sub-document unit。 |
| Overlap | 安全边距 | 相邻 chunks 共享 N tokens；2026 benchmarks 中常常没用。 |
| Semantic chunking | 聪明 chunking | 在相邻句子 embedding similarity 下降处 split。 |
| Parent-document | 两层 retrieval | 检索小 children，返回大 parents。 |
| Late chunking | 先 embed 再 chunk | 在 token level embed 整文档，再 pool 成 chunk vectors。 |
| Contextual retrieval | Anthropic 的技巧 | 索引前给每个 chunk 前加 LLM 生成的 summary。 |
| Context cliff | 2500-token 墙 | RAG 中约 2.5k context tokens 处观察到的质量下降（Jan 2026）。 |

## 延伸阅读

- [Yepes et al. / LangChain — Recursive Character Splitting docs](https://python.langchain.com/docs/how_to/recursive_text_splitter/) — 生产默认。
- [Vectara (2024, NAACL 2025). Chunking configurations analysis](https://arxiv.org/abs/2410.13070) — chunking 和 embedding choice 一样重要。
- [Jina AI — Late Chunking in Long-Context Embedding Models (2024)](https://jina.ai/news/late-chunking-in-long-context-embedding-models/) — late chunking 论文。
- [Anthropic — Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval) — LLM-generated context prefixes 带来 35-50% retrieval improvement。
- [NVIDIA 2026 chunk-size benchmark — Premai summary](https://blog.premai.io/rag-chunking-strategies-the-2026-benchmark-guide/) — 按 query type 选择 chunk size。
