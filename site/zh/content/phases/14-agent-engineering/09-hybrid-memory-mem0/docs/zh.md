# Hybrid Memory：Vector + Graph + KV（Mem0）

> Mem0（Chhikara et al., 2025）把 memory 作为三种并行 stores 来处理：vector 用于 semantic similarity，KV 用于快速 fact lookup，graph 用于 entity-relationship reasoning。Retrieval 时用 scoring layer 融合三者。这是 2026 年 external memory 的生产标准。

**类型：** 构建
**语言：** Python (stdlib)
**前置要求：** 阶段 14 · 07（MemGPT），阶段 14 · 08（Letta Blocks）
**时间：** ~75 分钟

## 学习目标

- 解释为什么单一 store（只有 vector、只有 graph、只有 KV）不足以承载 agent memory。
- 说出 Mem0 的三个 parallel stores，以及每个优化什么。
- 描述 Mem0 的 fusion scoring：relevance、importance、recency，以及为什么它是 weighted sum，而不是 hierarchy。
- 用 stdlib 实现一个 toy three-store memory：`add()` 写入三种 store，`search()` 融合结果。

## 问题

对于三类 query，单一 store 总会错一类：

- **Semantic similarity**：“what did we discuss about agent drift last week?” Vector 胜出；KV 和 graph 会漏。
- **Fact lookup**：“what is the user's phone number?” KV 胜出；vector 浪费，graph 过度。
- **Relationship reasoning**：“which customers share the same billing entity?” Graph 胜出；vector 和 KV 答不了。

生产 agents 会在一个 session 中发出三种查询。Single-store memory 总会在其中两类上出错。Mem0 的贡献是把三者接在统一的 `add`/`search` surface 后面，并用 scoring function 融合它们。

## 概念

### 三种 store 并行

Mem0（arXiv:2504.19413, April 2025）在 `add(text, user_id, metadata)` 时：

1. 从 text 抽取 candidate facts（LLM-driven step）。
2. 把每条 fact 写入 vector store（embedding），用于 semantic search。
3. 把每条 fact 写入 KV store，key 是 (user_id, fact_type, entity)，用于 O(1) lookup。
4. 把每条 fact 作为 typed edges 写入 graph store（Mem0g），用于 relationship queries。

在 `search(query, user_id)` 时：

1. Vector store 按 embedding cosine 返回 top-k。
2. KV store 按 query-derived (user_id, type, entity) 返回 direct hits。
3. Graph store 返回从 query entities 可达的 subgraph。
4. Scoring layer 融合三者。

### Fusion scoring

```
score = w_relevance * relevance(q, record)
      + w_importance * importance(record)
      + w_recency * recency(record)
```

- **Relevance**：vector cosine、KV exact match、graph path weight。
- **Importance**：写入时标注或学习得到（某些 facts 更重要：names、IDs、policies）。
- **Recency**：自上次 write 或 read 起按时间做 exponential decay。

Weights 按产品调优。Chat agents 提高 `w_recency`；compliance agents 提高 `w_importance`；retrieval agents 提高 `w_relevance`。

### Mem0g 和 temporal reasoning

Mem0g 添加 conflict detector。当一个新 fact 与现有 edge 矛盾时，现有 edge 会被标记为 invalid，但不会删除。Temporal queries（“what was the user's city in March?”）会遍历 valid-at-time subgraph。

这是 Letta invalidation pattern 泛化后的 compliance-grade behavior。

### Benchmark numbers

Mem0 论文报告（2025）：

- **LoCoMo**（long-form conversation memory）：91.6
- **LongMemEval**（long-horizon episodic memory）：93.4
- **BEAM 1M**（1M-token memory benchmark）：64.1

对比 baselines（full-context 128k LLM、flat vector store、flat KV）都低 10+ 分。Benchmarks 本身不能证明选择正确，operational shape 才能，但这些数字说明 fusion design 不是 rounding error。

### Scope taxonomy

Mem0 按 scope 拆分 memory：

- **User memory**：跨 sessions 持久存在，按 `user_id` key。
- **Session memory**：在一个 thread 内持久存在。
- **Agent memory**：每个 agent instance 的 state。

每次 write 都选择一个 scope。Retrieval 可以用 per-scope weights 跨 scopes 查询。不加思考地混合 scopes，会导致“assistant 把 Bob 的项目告诉 Alice”这类事故。

### 这个模式会在哪里出错

- **Embedding drift。** Vector results 在前一百个 queries 看起来正确，但随着 corpus 增长而退化。对 top-N-used records 定期 re-embedding。
- **KV schema creep。** `(user_id, type, entity)` 看起来简单，直到每个团队都添加自己的 `type`。每季度 audit type set。
- **Graph explosion。** 一个 noisy extractor 每条 message 添加 50 条 edges。限制每次 `add` call 的 graph writes；丢弃 low-confidence edges。

## 构建它

`code/main.py` 用 stdlib 实现 three-store pattern：

- `VectorStore`：用 naive token-overlap similarity 作为 embedding 替身。
- `KVStore`：以 `(user_id, fact_type, entity)` 为 key 的 dict。
- `GraphStore`：typed edges（subject、relation、object、valid）。
- `Mem0`：top-level facade，包含 `add()`、`search()`、fusion scoring 和 scope-aware retrieval。
- 一个 multi-user、multi-session conversation 的 worked trace。

运行它：

```
python3 code/main.py
```

输出会显示三条独立 recall paths 和融合后的 top-k。在 `main()` 顶部翻转 scoring weights，观察 ranking 如何变化。

## 使用它

- **Mem0（Apache 2.0）**：production-ready。可用 Postgres + Qdrant + Neo4j self-host，或使用 managed cloud。
- **Letta**：三层 core/recall/archival；自带 vector 和 graph backends。
- **Zep**：带 temporal KG 和 fact extraction 的 commercial alternative。
- **Custom builds**：当你需要精确控制 extractor（compliance）或 fusion weights（recency 主导的 voice agents）时。

## 发布它

`outputs/skill-hybrid-memory.md` 会生成一个 three-store memory scaffold，接好 fusion scorer、scope taxonomy 和 temporal invalidation。

## 练习

1. 用真实 embedding model（sentence-transformers、Ollama、OpenAI embeddings）替换 toy vector similarity。在 synthetic long conversation 上测 recall@10。1000 次 writes 后 ranking 会漂移吗？
2. 添加 temporal query：`search(query, as_of=timestamp)`。只返回该时间点或之前有效的 records。哪种 store 需要最多工作？
3. 实现 conflict detector：如果 incoming fact 与 graph edge 矛盾，invalidate old edge 并记录两者。用 “user lives in Berlin” -> “user lives in Lisbon” 测试。
4. 把 fusion scorer 扩展一个 `user_feedback` 维度（retrieved records 上的 thumbs-up）。如何防止 gaming（agent 只返回它已经 liked 的 records）？
5. 阅读 Mem0 docs（`docs.mem0.ai`）。把 toy 移植到 `mem0` client calls。在同样 20 条 test queries 上比较 retrieval quality。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|----------------|------------------------|
| Hybrid memory | “Vector plus graph plus KV” | 三种 store 并行写入，retrieval 时融合 |
| Fact extraction | “Memory ingestion” | LLM step，把 text 拆成 (entity, relation, fact) tuples |
| Fusion scoring | “Relevance ranking” | relevance、importance、recency 的 weighted sum |
| Scope | “Memory namespace” | user / session / agent：决定谁能看到什么 |
| Mem0g | “Memory graph” | 带 temporal validity 的 typed edges，用于 relationship queries |
| Temporal invalidation | “Soft delete” | 把被矛盾推翻的 edges 标记 invalid；永远不删除 |
| Embedding drift | “Retrieval rot” | 随 corpus 增长，vector quality 下降；定期 re-embed |

## 延伸阅读

- [Chhikara et al., Mem0 (arXiv:2504.19413)](https://arxiv.org/abs/2504.19413)：原始论文
- [Mem0 docs](https://docs.mem0.ai/platform/overview)：production API、SDKs、managed cloud
- [Packer et al., MemGPT (arXiv:2310.08560)](https://arxiv.org/abs/2310.08560)：virtual-context predecessor
- [Letta, Memory Blocks blog](https://www.letta.com/blog/memory-blocks)：三层 sibling design
