# ColPali 与 Vision-Native Document RAG

> 传统 RAG 会把 PDF parse 成文本，split 成 chunks，embed chunks，再存 vectors。每一步都会丢信号：OCR 丢掉 chart data，chunking 打断 table rows，text embeddings 忽略 figures。ColPali（Faysse 等人，2024 年 7 月）问了一个更简单的问题：为什么一定要 extract text？直接通过 PaliGemma embed page image，用 ColBERT-style late interaction 做 retrieval，并保留 document 携带的 layout、figures、fonts 和 formatting signal。公开 benchmarks：在 visually-rich documents 上，端到端 accuracy 比 text-RAG 好 20-40%。ColQwen2、ColSmol 和 VisRAG 扩展了这个模式。本课阅读 vision-native RAG thesis，并构建一个小型 ColPali-like indexer。

**类型：** 构建
**语言：** Python（stdlib，multi-vector indexer + MaxSim scorer）
**前置要求：** 阶段 11（LLM Engineering — RAG basics），阶段 12 · 05（LLaVA）
**时间：** ~180 分钟

## 学习目标

- 解释 bi-encoder retrieval（每个 document 一个 vector）与 late-interaction retrieval（每个 document 多个 vectors）的差异。
- 描述 ColBERT 的 MaxSim operation，以及 ColPali 如何把它从 text tokens 推广到 image patches。
- 构建小型 ColPali-like indexer：page -> patch embeddings -> 对 query-term embeddings 做 MaxSim -> top-k pages。
- 在 invoices / financial reports 用例上比较 ColPali + Qwen2.5-VL generator 与 text-RAG + GPT-4。

## 问题

PDF 上的 text-RAG 会丢掉文档的大部分信息。Financial report 的 Q3 revenue growth 通常在图表里；medical report 的 findings 在 annotated images 中；legal contract 的 signature block 是 layout fact，不是 text fact。

Text-RAG pipeline：

1. PDF -> 通过 OCR / pdftotext 转 text。
2. Text -> 300-500 token chunks。
3. Chunk -> bi-encoder embedding（一个 vector）。
4. User query -> embedding -> cosine similarity -> top-k chunks。
5. Chunks + query -> LLM。

五个有损步骤。Charts 没被捕捉。Tables 被 chunk 切开。Multi-column layout 被展平。Figure annotations 消失。

ColPali 的修复：跳过 OCR，直接 embed page image。Retrieval 使用 ColBERT-style late interaction，让模型在 query time attend 到细粒度 patches。

## 概念

### ColBERT（2020）

ColBERT（Khattab & Zaharia，arXiv:2004.12832）是 text retrieval 方法。它不是每个 document 一个 vector，而是每个 token 一个 vector。Query time：

- Query tokens 获得自己的 embeddings（N_q vectors）。
- Document tokens 获得 embeddings（N_d vectors，通常 cached）。
- Score = 对每个 query token，取所有 document tokens 的最大 cosine similarity 后求和：Σ_i max_j cos(q_i, d_j)。

这就是 MaxSim operation。每个 query token “pick” 最匹配的 document token。最终分数是总和。

优点：recall 强，处理 term-level semantics。缺点：每个 document 有 N_d vectors，storage 昂贵。

### ColPali

ColPali（Faysse 等人，arXiv:2407.01449）把 ColBERT 模式应用到图像。

- 每页由 PaliGemma（ViT + language）编码成 patch embeddings：每页 N_p vectors。
- 每个用户 query（text）编码成 query-token embeddings：N_q vectors。
- Score = Σ_i max_j cos(q_i, p_j)，即 query-text-tokens 与 page-image-patches 上的 MaxSim。
- 按 total score 取 top-k pages。

Document-ingestion 时：用 PaliGemma embed 每一页，存储所有 patch embeddings。Query time：embed query tokens，对所有存储的 page embeddings 计算 MaxSim，返回 top-k pages。

优点：在 visually rich documents 上，端到端比 text-RAG 好 20-40%。每个 patch-vector 捕捉 local layout 和 content。

缺点：N_p patches × 4-byte floats × D-dim vectors per page，storage 增长很快。可通过 PQ / OPQ quantization 缓解。

### ColQwen2 与 ColSmol

ColQwen2（illuin-tech，2024-2025）把 PaliGemma 换成 Qwen2-VL。Base encoder 更好，retrieval 更好。

ColSmol 是用于 local / edge 的小规模变体。约 1B params 的 ColSmol retriever 可以在 consumer GPU 上运行。

### VisRAG

VisRAG（Yu 等人，arXiv:2410.10594）是另一种变体：不是在 patches 上做 MaxSim，而是用 VLM 把每页 pool 成一个 vector，然后做 bi-encoder retrieval。Indexing 更快、storage 更小，但 recall 较弱。

Quality-vs-cost trade-off：追求质量用 ColPali，追求规模用 VisRAG。

### M3DocRAG

M3DocRAG（Cho 等人，arXiv:2411.04952）把 multi-modal retrieval 扩展到 multi-page multi-document reasoning。它跨 documents 检索 pages，并为 VLM 组合 multi-page context。

### ViDoRe：benchmark

ColPali 的配套 benchmark。Visual Document Retrieval Evaluation。任务包含 financial reports、scientific papers、administrative documents、medical records、manuals。Metric：nDCG@5。

ColPali-v1 在 ViDoRe 上约 80% nDCG@5；同 documents 上 text-RAG 约 50-60%。

### End-to-end RAG pipeline

Vision-native RAG：

1. Ingest：PDF -> page images -> PaliGemma encoding -> 存储所有 patch embeddings。
2. Query：user text -> query-token embeddings -> 对所有 indexed pages 做 MaxSim -> top-k pages。
3. Generate：top-k page images + query -> VLM（Qwen2.5-VL 或 Claude）-> answer。

全程无 OCR。Figures、charts、fonts、layout 全部流入答案。

### Storage math

一个 50 页 financial report，每页 729 patches，128-dim embeddings：

- ColPali：50 * 729 * 128 * 4 bytes = 约 18 MB raw，PQ 后约 4 MB。
- Text-RAG：50 chunks * 768-dim * 4 bytes = 约 150 kB。

ColPali 每个 document storage 约为 30x。规模化时，OPQ / PQ 可以降到约 5-10x，通常可接受。

### 什么时候 text-RAG 仍然胜出

- 没有 layout signal 的纯文本 documents（wiki articles、chat logs）。Text-RAG 更简单且 storage 更便宜。
- Multi-million-page archives，storage 主导成本。
- 严格监管要求 retrieval 时同时有 extractable OCR text。

除此之外，2026 年的 financial reports、scientific papers、legal contracts、medical records、UX documentation 都更适合 vision-native RAG。

## 使用它

`code/main.py`：

- Toy patch encoder：把一个 “page”（小 feature vector grid）映射到 patch embeddings 数组。
- MaxSim scorer：计算 query token embedding set 与 page patch set 之间的 ColBERT-style score。
- Index 5 个 toy pages，运行 3 个 queries，返回 top-k 和 scores。

## 交付它

本课产出 `outputs/skill-vision-rag-designer.md`。给定 document-RAG project，它会选择 ColPali / ColQwen2 / VisRAG / text-RAG，并估算 storage。

## 练习

1. 一个 200 页 annual report，每页 729 patches、128-dim emb、4-byte floats。计算 raw storage 和 PQ-compressed（8x）storage。

2. MaxSim 是 Σ_i max_j cos(q_i, p_j)。它捕捉了 simple mean similarity 捕捉不到的什么？

3. ColPali 把 pages index 为 patch sets。如果改成 word level index（像 ColBERT 那样），会发生什么变化？Trade-offs？

4. 为 1M-page corpus 设计 end-to-end pipeline，latency budget 为每 query 500ms。选择 ColQwen2 / VisRAG 并解释。

5. 阅读 M3DocRAG（arXiv:2411.04952）。描述 multi-page attention pattern，以及它与 single-page ColPali retrieval 的不同。

## 关键术语

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| Late interaction | “ColBERT-style” | 使用 per-token 或 per-patch embeddings + MaxSim 的 retrieval，而不是单个 doc vector |
| MaxSim | “Max-over-patches” | 对每个 query token，选择最高 similarity document token；跨 query 求和 |
| Bi-encoder | “Single-vector” | 每个 document 一个 vector；更快但丢失 granularity |
| Multi-vector | “Many-vectors-per-doc” | 每个 document / page 存 N_p vectors；storage 成本增加但 recall 改善 |
| Patch embedding | “Page feature” | VLM encoder 为每个 image patch 产生的 vector，按 page cache |
| ViDoRe | “Vision doc bench” | ColPali 的 visual document retrieval benchmark suite |
| PQ quantization | “Product quantization” | 在缩小 storage 约 8x 的同时保持 vector similarity 的压缩 |

## 延伸阅读

- [Faysse et al. — ColPali (arXiv:2407.01449)](https://arxiv.org/abs/2407.01449)
- [Khattab & Zaharia — ColBERT (arXiv:2004.12832)](https://arxiv.org/abs/2004.12832)
- [Yu et al. — VisRAG (arXiv:2410.10594)](https://arxiv.org/abs/2410.10594)
- [Cho et al. — M3DocRAG (arXiv:2411.04952)](https://arxiv.org/abs/2411.04952)
- [illuin-tech/colpali GitHub](https://github.com/illuin-tech/colpali)
