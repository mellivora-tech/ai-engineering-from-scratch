# Capstone 04 — 多模态文档 QA（Vision-First PDF、表格、图表）

> 2026 年 document-QA 的前沿已经从 OCR-then-text 转向 vision-first late interaction。ColPali、ColQwen2.5 和 ColQwen3-omni 把每一页 PDF 当成图像，用 multi-vector late interaction embedding，并让 query 直接 attend 到 patches。在财务 10-K、科学论文和手写笔记上，这种模式大幅胜过 OCR-first。请在 1 万页上 end-to-end 构建这条 pipeline，并发布它与 OCR-then-text 的并排对比。

**类型：** Capstone
**语言：** Python（pipeline）、TypeScript（viewer UI）
**前置要求：** 阶段 4（computer vision）、阶段 5（NLP）、阶段 7（transformers）、阶段 11（LLM engineering）、阶段 12（multimodal）、阶段 17（infrastructure）
**覆盖阶段：** P4 · P5 · P7 · P11 · P12 · P17
**时间：** 30 小时

## 问题

企业沉淀着大量会被 OCR pipeline 搞坏的 PDF：带旋转表格的扫描 10-K、公式密集的科学论文、只有以图像形式才说得通的图表、手写批注。把这些文档当成 text-first，意味着丢掉一半信号。2026 年的答案是在 raw page images 上做 late-interaction multi-vector retrieval。ColPali（Illuin Tech）引入了这种方法；ColQwen2.5-v0.2 和 ColQwen3-omni 进一步提升了准确率。在 ViDoRe v3 上，vision-first retrieval 明显高于 OCR-then-text，且差距会在图表、表格和手写内容上进一步扩大。

代价是 storage 和 latency。一个 ColQwen embedding 约等于每页 2048 个 patch vectors，而不是单个 1024-dim vector。原始存储会膨胀。DocPruner（2026）可以在没有可测 accuracy loss 的情况下剪掉 50%。你将索引 1 万页，衡量 ViDoRe v3 nDCG@5，在 2 秒内服务答案，并与 OCR-then-text baseline 直接对比。

## 概念

Late interaction 意味着每个 query token 都会和每个 patch token 打分，然后对每个 query token 取最大分并求和。你得到的是细粒度匹配，而不需要一个 pooled vector。multi-vector index（Vespa、Qdrant multi-vector 或 AstraDB）会存储 per-patch embeddings，并在 retrieval time 运行 MaxSim。

answerer 是一个 vision-language model，它接收 query 和 top-k retrieved pages 的图像，写出带 evidence regions（bounding boxes 或 page references）的答案。Qwen3-VL-30B、Gemini 2.5 Pro 和 InternVL3 是 2026 年的前沿选择。对于方程和科学记法，可选的 OCR fallback（Nougat、dots.ocr）会作为 text channel 拼接进来。

Evaluation 是一个二维矩阵。一条轴是 content type（plain text paragraphs、dense tables、bar/line charts、handwritten notes、equations）。另一条轴是 retrieval approach（vision-first late interaction vs OCR-then-text vs hybrid）。每个 cell 都得到 nDCG@5 和 answer accuracy。报告就是交付物。

## 架构

```
PDFs -> page renderer (PyMuPDF, 180 DPI)
           |
           v
  ColQwen2.5-v0.2 embed (multi-vector per page, ~2048 patches)
           |
           +------> DocPruner 50% compression
           |
           v
   multi-vector index (Vespa or Qdrant multi-vector)
           |
query ----+----> retrieve top-k pages (MaxSim)
           |
           v
  VLM answerer: Qwen3-VL-30B | Gemini 2.5 Pro | InternVL3
    inputs: query + top-k page images + optional OCR text
           |
           v
  answer with cited page numbers + evidence regions
           |
           v
  Streamlit / Next.js viewer: highlighted boxes on source page
```

## 技术栈

- Page rendering：PyMuPDF（fitz），180 DPI，portrait-normalized
- Late-interaction model：ColQwen2.5-v0.2 或 ColQwen3-omni（Hugging Face 上的 vidore team）
- Index：带 multi-vector field 的 Vespa，或 Qdrant multi-vector，或带 MaxSim 的 AstraDB
- Pruning：DocPruner 2026 policy（保留 high-variance patches，在 < 0.5% accuracy loss 下实现 50% compression）
- OCR fallback（equations / dense tables）：dots.ocr 或 Nougat
- VLM answerer：自托管 Qwen3-VL-30B 或 hosted Gemini 2.5 Pro；InternVL3 作为 fallback
- Evaluation：ViDoRe v3 benchmark，M3DocVQA 用于 multi-page reasoning
- Viewer UI：Next.js 15，用 canvas overlay 展示 evidence regions

## 构建它

1. **Ingest。** 遍历一个包含 10-K、科学论文和扫描文档的 1 万页 PDF corpus。把每页渲染成 1536x2048 PNG。持久化 `{doc_id, page_num, image_path}`。

2. **Embed。** 对每个 page image 运行 ColQwen2.5-v0.2。输出 shape 约为 2048 个 dim 128 的 patch embeddings。应用 DocPruner，保留信号最高的一半。写入 Vespa multi-vector field 或 Qdrant multi-vector。

3. **Query。** 对每个 incoming query，用 query tower 生成 embedding（token-level embeddings）。对 index 运行 MaxSim：对每个 query token，在 page patch embeddings 上取最大 dot-product，再求和。返回 top-k pages。

4. **Synthesize。** 用 query 和 top-5 page images 调用 Qwen3-VL-30B。Prompt: "Answer using only the supplied pages. Cite each claim by (doc_id, page) and name the region (figure, table, paragraph)."

5. **Evidence regions。** 后处理答案，提取 cited regions。如果 VLM 输出 bounding boxes（Qwen3-VL 会输出），就在 viewer 中渲染成 overlays。

6. **OCR fallback。** 对识别为 equation-dense 的页面（基于 image variance 的启发式）运行 Nougat 或 dots.ocr，并把 OCR text 作为额外 channel 与 image 一起传入。

7. **Eval。** 运行 ViDoRe v3（retrieval nDCG@5）和 M3DocVQA（multi-page QA accuracy）。同一 corpus 上也运行 OCR-then-text pipeline，并使用同一个 synthesizer。产出 content-type × approach matrix。

8. **UI。** 先做 Streamlit prototype；生产 viewer 使用 Next.js 15，带逐页 evidence-region overlay。

## 使用它

```
$ doc-qa ask "what was the 2024 operating margin change for segment EMEA?"
[retrieve]   top-5 pages in 320ms (ColQwen2.5, MaxSim, Vespa)
[synth]      qwen3-vl-30b, 1.4s, cited (form-10k-2024, p. 88) + (..., p. 92)
answer:
  EMEA operating margin moved from 18.2% to 16.8%, a 140bp decline.
  cited: 10-K-2024.pdf p.88 (Table 4, Segment Operating Margin)
         10-K-2024.pdf p.92 (MD&A, Operating Performance)
[viewer]     open with highlighted bounding boxes overlaid on p.88 Table 4
```

## 交付它

`outputs/skill-doc-qa.md` 描述交付物：一个针对特定 corpus 调优的 vision-first multimodal document QA system，并在 ViDoRe v3 上与 OCR-then-text baseline 对比评估。

| 权重 | 标准 | 衡量方式 |
|:-:|---|---|
| 25 | ViDoRe v3 / M3DocVQA accuracy | Benchmark numbers vs OCR-text baseline and published leaderboard |
| 20 | Evidence-region grounding | cited regions 中确实包含 answer span 的比例 |
| 20 | Storage and latency engineering | DocPruner compression ratio、index p95、answer p95 |
| 20 | Multi-page reasoning | hand-labeled 100-question multi-page set 上的准确率 |
| 15 | Source-inspection UX | Viewer clarity、overlay fidelity、side-by-side comparison tools |
| **100** | | |

## 练习

1. 在同一个 corpus 上衡量 ColQwen2.5-v0.2 与 ColQwen3-omni。一个做对而另一个漏掉的是哪些页面？向 index 添加 “content class” tag，并按类型 route。

2. 激进剪枝 embeddings（75%、90%）。找到 compression cliff：ViDoRe nDCG@5 掉到 OCR baseline 以下的点。

3. 构建 hybrid：并行运行 OCR-then-text 和 ColQwen，用 RRF 融合，再用 cross-encoder rerank。hybrid 是否胜过任一单独方案？它在哪些地方帮助最大？

4. 把 Qwen3-VL-30B 换成更小的 VLM（Qwen2.5-VL-7B）。衡量 accuracy-per-dollar curve。

5. 添加 handwritten-note support。渲染 handwriting corpus，用 ColQwen embedding，衡量 retrieval。与 handwriting OCR pipeline 对比。

## 关键词汇

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|------------------------|
| Late interaction | “ColPali-style retrieval” | Query tokens 分别与 page patches 打分；MaxSim 聚合 |
| Multi-vector | “Per-patch embedding” | 每个 document 有许多 vectors，而不是一个 pooled vector |
| MaxSim | “Late-interaction scoring” | 对每个 query token，在 document vectors 上取最大相似度并求和 |
| DocPruner | “Patch compression” | 2026 年的 pruning 方法，保留 50% patches 且 accuracy loss 可忽略 |
| ViDoRe v3 | “Document-retrieval benchmark” | 2026 年衡量 visual-document retrieval 的标准 |
| Evidence region | “Cited bounding box” | source page 上定位 answer span 的 bbox |
| OCR fallback | “Equation channel” | 与 vision 并用的 text pipeline，用于 equation-heavy 或 table-heavy 页面 |

## 延伸阅读

- [ColPali (Illuin Tech) repository](https://github.com/illuin-tech/colpali) — late-interaction doc retrieval 参考实现
- [ColPali paper (arXiv:2407.01449)](https://arxiv.org/abs/2407.01449) — foundational method paper
- [ColQwen family on Hugging Face](https://huggingface.co/vidore) — production-ready checkpoints
- [M3DocRAG (Adobe)](https://arxiv.org/abs/2411.04952) — multi-page multimodal RAG baseline
- [Vespa multi-vector tutorial](https://docs.vespa.ai/en/colpali.html) — reference serving stack
- [Qdrant multi-vector support](https://qdrant.tech/documentation/concepts/vectors/#multivectors) — alternate index
- [AstraDB multi-vector](https://docs.datastax.com/en/astra-db-serverless/databases/vector-search.html) — alternate managed index
- [Nougat OCR](https://github.com/facebookresearch/nougat) — equation-capable OCR fallback
