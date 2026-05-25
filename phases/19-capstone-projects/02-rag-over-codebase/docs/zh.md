# Capstone 02 — 面向代码库的 RAG（跨 Repo 语义搜索）

> 到 2026 年，每个严肃的工程组织都会运行内部代码搜索系统，它理解的是意义，而不只是字符串。Sourcegraph Amp、Cursor 的 codebase answers、Augment 的 enterprise graph、Aider 的 repomap、Pinterest 的内部 MCP，形态都一样：摄入多个 repo，用 tree-sitter 解析，嵌入 function 和 class 级别的 chunk，做 hybrid-search、re-rank，并用 citations 回答。这个 capstone 要求你构建一个能处理 10 个 repo、200 万行代码，并能在每次 git push 后承受 incremental re-indexing 的系统。

**类型：** Capstone
**语言：** Python（ingestion）、TypeScript（API + UI）
**前置要求：** 阶段 5（NLP foundations）、阶段 7（transformers）、阶段 11（LLM engineering）、阶段 13（tools）、阶段 17（infrastructure）
**覆盖阶段：** P5 · P7 · P11 · P13 · P17
**时间：** 30 小时

## 问题

到 2026 年，每个前沿 coding agent 都带有 codebase retrieval layer，因为单靠 context window 解决不了跨 repo 问题。Claude 的 1M-token context 有帮助，但并不能消除 ranked retrieval 的需要。对原始 chunk 做 naive cosine search，会在生成代码、monorepo 重复内容，以及很少被 import 的长尾 symbol 上污染结果。生产级答案是：在 AST-aware chunks 上做 hybrid（dense + BM25）search，再接 re-ranker，并由 symbol reference graph 支撑。

你要通过索引一组真实代码资产来学习这一点，而不是只索引一个教程 repo。你要衡量 MRR@10、citation faithfulness 和 incremental freshness。失败模式主要是基础设施问题：10 万文件的 monorepo、一次 push 改动半数文件、一个问题需要跨四个 repo 才能正确回答。

## 概念

AST-aware ingestion pipeline 会用 tree-sitter 解析每个文件，提取 function 和 class node，并在 node boundary 上切分，而不是使用固定 token window。每个 chunk 有三种表示：dense embedding（Voyage-code-3 或 nomic-embed-code）、sparse BM25 terms，以及一段简短自然语言 summary。summary 增加了第三种可检索模态：用户问 “how is X authorized”，summary 提到 “authz”，即使代码里只有 `check_permission`。

Retrieval 是 hybrid。一个 query 同时触发 dense 和 BM25 搜索，合并 top-k，然后把 union 交给 cross-encoder re-ranker（Cohere rerank-3 或 bge-reranker-v2-gemma-2b）。re-ranked list 会进入 long-context synthesizer（带 prompt caching 的 Claude Sonnet 4.7，或自托管 Llama 3.3 70B），并要求每个 claim 都用 file 和 line range 引用。没有 citation 的答案会被 post-filter 拒绝。

Incremental freshness 是基础设施难题。Git push 触发 diff：哪些文件变了，哪些 symbol 变了。只有受影响的 chunk 会重新 embedding。受影响的跨文件 symbol edge（imports、method calls）会重新计算。index 保持一致，而不用每次 commit 都重新处理 200 万行代码。

## 架构

```
git push --> webhook --> ingest worker (LlamaIndex Workflow)
                           |
                           v
             tree-sitter parse + AST chunk
                           |
            +--------------+----------------+
            v              v                v
          dense        BM25 index       summary (LLM)
        (Voyage / bge)  (Tantivy)        (Haiku 4.5)
            |              |                |
            +------> Qdrant / pgvector <----+
                            |
                            v
                      symbol graph (Neo4j / kuzu)
                            |
  query --> LangGraph agent (retrieve -> rerank -> synth)
                            |
                            v
                 Claude Sonnet 4.7 1M context
                            |
                            v
                 answer + file:line citations
```

## 技术栈

- 解析：tree-sitter，带 17 种语言 grammar（Python、TS、Rust、Go、Java、C++ 等）
- Dense embeddings：Voyage-code-3（hosted）或 nomic-embed-code-v1.5（self-host），bge-code-v1 fallback
- Sparse index：Tantivy（Rust），BM25F，对 symbol name vs body 做 field weighting
- Vector DB：Qdrant 1.12 hybrid search；或面向 5000 万向量以下团队的 pgvector + pgvectorscale
- Chunk summary model：Claude Haiku 4.5 或 Gemini 2.5 Flash，使用 prompt cache
- Re-ranker：Cohere rerank-3 或自托管 bge-reranker-v2-gemma-2b
- 编排：ingestion 使用 LlamaIndex Workflows，query agent 使用 LangGraph
- Synthesizer：Claude Sonnet 4.7（1M context）+ prompt caching
- Symbol graph：Neo4j（managed）或 kuzu（embedded），用于 import 和 call edge
- Observability：每个 retrieval + synthesis step 都记录 Langfuse span

## 构建它

1. **Ingestion walker。** 每次 push hook 都遍历 git history，收集 changed files。对每个文件，用 tree-sitter 解析，提取 function 和 class node 及其完整 source span。输出 chunk records `{repo, path, start_line, end_line, symbol, body}`。

2. **Chunk summarizer。** 把 chunks batch 到 Haiku 4.5 调用里，并对 system preamble 使用 prompt caching。Prompt: "Summarize this function in one sentence, naming its public contract and side effects." 把 summary 与 chunk 一起存储。

3. **Embedding pool。** 两个并行队列：dense（Voyage-code-3 batch 128）和 summary（同一个模型，但输入 summary string）。把 vectors 写入 Qdrant，payload 为 `{repo, path, start_line, end_line, symbol, kind}`。

4. **BM25 index。** Field-weighted Tantivy index：symbol name 权重 4，symbol body 权重 1，summary 权重 2。它同时支持 “find the function named X” 和 “find the function that does X”。

5. **Symbol graph。** 对每个 chunk 记录 edges：imports（this file uses symbol Y from repo Z）、calls（this function calls method M on class C）、inheritance。存储到 kuzu。query time 用它扩展跨 repo retrieval。

6. **Query agent。** 三个节点的 LangGraph。`retrieve` 并行触发 dense + BM25，按 (repo, path, symbol) 去重。`rerank` 在 top-50 上运行 cross-encoder 并保留 top-10。`synth` 用 reranked chunks 作为上下文调用 Claude Sonnet 4.7，缓存 system prompt，并要求 file:line citations。

7. **Citation enforcement。** 解析模型输出；任何没有 `(repo/path:start-end)` anchor 的 claim 都标记为 re-ask 或丢弃。只把带 citation 的答案返回给用户。

8. **Incremental re-index。** 每次 webhook 都计算 symbol-level diff。只有文本变化的 chunk 会重新 embed。只有 imports 变化的 chunk 会重算 symbol edges。衡量目标：对一个 200 万 LOC 的代码资产，50-file push 在 60 秒内完成 re-index。

9. **Eval。** 标注 100 个 cross-repo questions，并给出 gold file:line answers。衡量 MRR@10、nDCG@10、citation faithfulness（可验证 anchor 的 claim 比例）和 p50/p99 latency。

## 使用它

```
$ code-rag ask "how is S3 multipart abort wired into our retry budget?"
[retrieve]  12 chunks dense + 7 chunks bm25, 16 unique after dedup
[rerank]    top-5 kept (cohere rerank-3)
[synth]     claude-sonnet-4.7, cache hit rate 68%, 2.1s
answer:
  Multipart aborts are triggered by `AbortMultipartOnFail` in
  services/uploader/retry.go:122-148, which decrements the per-bucket
  retry budget defined in config/budgets.yaml:34-51 ...
  citations: [services/uploader/retry.go:122-148, config/budgets.yaml:34-51,
              libs/s3client/multipart.ts:44-61]
```

## 交付它

可交付 skill 为 `outputs/skill-codebase-rag.md`。给定一组 repo，它会启动 ingestion pipeline、hybrid index 和 query agent，并为任何 cross-repo question 返回带 citation 的答案。评分标准：

| 权重 | 标准 | 衡量方式 |
|:-:|---|---|
| 25 | Retrieval quality | 在 100-question held-out set 上的 MRR@10 和 nDCG@10 |
| 20 | Citation faithfulness | answer claims 中带有可验证 file:line anchor 的比例 |
| 20 | Latency and scale | indexed corpus size 下 10k QPS 的 p95 query latency |
| 20 | Incremental indexing correctness | 50-file commit 从 git push 到可搜索的时间 |
| 15 | UX and answer formatting | Citation clickability、snippet previews、follow-up affordance |
| **100** | | |

## 练习

1. 把 Voyage-code-3 换成自托管 nomic-embed-code。衡量 MRR@10 的变化。报告启用 re-ranking 后差距是否缩小。

2. 向 corpus 注入 20% generated code（LLM 生成的 boilerplate）并重新评估。观察 retrieval poisoning。给 payload 添加 `"generated"` 标记，并降低这些命中的权重。

3. 在你的 corpus size 下，对 Qdrant hybrid search 和 pgvector + pgvectorscale 做基准测试。报告 batch size 1 时的 p99。

4. 添加基于 sampling 的 drift check：每周重新运行 100-question eval。MRR@10 下降 > 5% 时告警。

5. 扩展到跨语言 symbol resolution：一个 Python 函数通过 gRPC 调用 Go service。用 symbol graph 把它们链接起来。

## 关键词汇

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|------------------------|
| AST-aware chunking | “Function-level splits” | 在 tree-sitter node boundary 切分代码，而不是固定 token window |
| Hybrid search | “Dense + sparse” | 并行运行 BM25 和 vector search，合并 top-k，再 rerank |
| Cross-encoder rerank | “Second-stage rank” | 对每个 (query, candidate) pair 一起打分的模型，比 cosine 更准确 |
| Prompt caching | “Cached system prompt” | 2026 Claude / OpenAI 功能：重复 prefix token 最高可折扣 90% |
| Symbol graph | “Code graph” | 跨文件和 repo 的 imports、calls、inheritance edges |
| Citation faithfulness | “Grounded answer rate” | 用户点击 anchor 并阅读 referenced span 后能验证的 claim 比例 |
| Incremental re-index | “Push-to-search time” | 从 git push 到 changed symbols 可查询的 wall-clock 时间 |

## 延伸阅读

- [Sourcegraph Amp](https://ampcode.com) — 生产级 cross-repo code intelligence
- [Sourcegraph Cody RAG architecture](https://sourcegraph.com/blog/how-cody-understands-your-codebase) — 本 capstone 的 reference deep-dive
- [Aider repo-map](https://aider.chat/docs/repomap.html) — tree-sitter ranked repo view
- [Augment Code enterprise graph](https://www.augmentcode.com) — 商业化 symbol-graph RAG
- [Qdrant hybrid search docs](https://qdrant.tech/documentation/concepts/hybrid-queries/) — reference implementation
- [Voyage AI code embeddings](https://docs.voyageai.com/docs/embeddings) — Voyage-code-3 details
- [Cohere rerank-3](https://docs.cohere.com/reference/rerank) — cross-encoder reference
- [Pinterest MCP internal search](https://medium.com/pinterest-engineering) — internal-platform reference
