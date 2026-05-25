# Multimodal RAG 与 Cross-Modal Retrieval

> Vision-native document RAG 只是一片。生产中的 multimodal RAG 更宽：跨 text、images、audio 和 video 做 retrieval，用于 trip planning（“find me a quiet vegan brunch with natural light”）、medical triage（“what injury matches this photo + these notes”）、e-commerce（“outfits similar to this selfie, in my size”）、field service（“diagnose this engine sound plus photo of the part”）等 workflow。2025 年三篇 surveys，Abootorabi 等人、Mei 等人、Zhao 等人，把子问题编码成 taxonomy：cross-modal retrieval、retrieval fusion、generation grounding、multimodal evaluation。本课阅读 surveys 并设计 production pipeline。

**类型：** 构建
**语言：** Python（stdlib，cross-modal retriever with fusion + grounded generator）
**前置要求：** 阶段 12 · 23（ColPali），阶段 11（RAG basics）
**时间：** ~180 分钟

## 学习目标

- 设计 cross-modal retrieval：text -> image、image -> text、audio -> video 等。
- 比较三种 fusion strategies：score fusion、attention-based fusion、MoE fusion。
- 解释 generation grounding：当 sources 是多种 modality 混合时，“cite your sources” 是什么样。
- 说出 2025 年三篇 canonical multimodal RAG surveys 及其 sub-problem taxonomy。

## 问题

Single-modality RAG 是已解决模式：embed query，embed chunks，retrieve，stuff into LLM。Multimodal RAG 需要：

1. 多个 retrieval heads（每种 modality 都需要兼容空间中的 embeddings）。
2. 跨 modalities 融合 retrieval results。
3. Generation grounding，能引用跨 modalities 的 sources。
4. 覆盖 cross-modal signal 的 evaluation metrics。

2025 年 surveys 都到达同一 taxonomy。

## 概念

### Cross-modal retrieval

给定 modality A 的 query，检索 modality B 的 documents。三种模式：

1. Shared embedding space。CLIP 和 CLAP 在共享空间中产生 text + image / text + audio embeddings。可以直接跨 modalities 做 cosine similarity。受限于 CLIP-trained pairs。

2. Per-modality encoder + translation。Text encoder + image encoder + 一个小 translator module，在空间之间映射。Gupta 等人的 Sen2Sen 与其他 2024 设计。灵活但增加复杂度。

3. VLM as encoder。使用 VLM hidden states 作为 retrieval representation。VLM 支持的任何 modality 都可用。质量更高，成本更贵。

选择：text+image 用 CLIP / SigLIP 2；text+audio 用 CLAP；frontier-quality cross-modal 用 VLM-hidden-states。

### Fusion strategies

你检索到 10 个结果：5 张 images、3 段 text passages、2 个 audio clips。如何合并？

Score fusion（最便宜）。每种 modality 有自己的 retriever，各自返回 scores。先在 modality 内 normalize scores，再求和。简单，常常有效。

Attention-based fusion。拼接所有 retrieved items，让一个小 attention network 给它们加权。需要训练。

MoE fusion。Gating network 路由到 modality-specific experts。不同 query types 路由不同，visual question 会给 images 更高权重。

生产默认：score fusion，并略微偏向 query 的 dominant modality。如果 A/B 表明 domain 上收益明显，再升级到 MoE。

### Generation grounding

LLM 应当引用哪个 retrieved item 支撑了每条 claim。对于 multi-modal：

- Text source：标准 citation `[1]`。
- Image source：`[img 3]`，带短 caption。
- Audio：`[audio 2 at 0:34]`。

用 grounding-aware data 训练 generator：training target 中每个 claim 都带 source index。推理时，模型自然发出 citations。

### 2025 surveys

Abootorabi 等人（arXiv:2502.08826，"Ask in Any Modality"）：multimodal RAG taxonomy。覆盖 retrieval、fusion、generation。覆盖最广。

Mei 等人（arXiv:2504.08748，"A Survey of Multimodal RAG"）：关注 sub-task benchmarks 和 failure modes。适合 evaluation design。

Zhao 等人（arXiv:2503.18016）：vision-focused survey。ColPali-family work 讲得强。

读完三篇，就能掌握截至 2025 年春的 state of the art。多数子问题仍然开放。

### MuRAG：foundation paper

MuRAG（Chen 等人，2022）是第一篇 multimodal RAG。从 multimodal KB 中检索 image + text，并生成答案。在 VLM 浪潮之前证明了可行性。现代系统（REACT、VisRAG、M3DocRAG）都基于它继续构建。

### 一个 production trip-planner 示例

Query：“find me a quiet vegan brunch with natural light.”

Pipeline：

1. Decompose query。“quiet” -> audio/review keyword；“vegan brunch” -> menu item；“natural light” -> image feature。
2. 每个 modality 检索：
   - 对 reviews 做 text retrieval：“vegan brunch, quiet ambiance.”
   - 对 restaurant photos 做 image retrieval：“natural light, airy.”
   - 对 ambient-sound clips 做 audio retrieval：“low decibel, no music.”
3. Fuse scores。每个 restaurant 有 composite score。
4. Top-k restaurants -> VLM generator with all evidence -> 带 citations 的 answer。

这远超过 text-RAG。每个 modality 都加入了 text alone 捕捉不到的信号。

### Agentic multimodal RAG

Multi-hop：如果第一次 retrieval 没有返回 high-confidence answers，LLM reformulates 并再次检索。阶段 14 的 Agentic RAG patterns 也适用于这里。示例：

- Retrieve initial top-10 -> LLM asks “too noisy, filter for <40 dB” -> re-retrieve。
- Retrieve images -> LLM 看到其中一张有 menu -> retrieve menu text -> answer。

这增加复杂度，但能处理 single-shot retrieval 无法解决的 queries。

### Evaluation

Cross-modal evaluation 仍不成熟。常见 proxies：

- 每个 modality 的 Recall@k。
- Fused top-k accuracy。
- Human-judged end-to-end satisfaction。
- Task-specific（bookings completed、purchases made）。

没有一个 standard benchmark 覆盖所有 modalities。多数论文在 domain-specific tasks 上评估。

## 使用它

`code/main.py`：

- 三个 mock retrievers（text、image、audio），作用于共享 restaurants corpus。
- Score fusion，用 configurable weights 组合 modality scores。
- Generator stub，发出带 citations 的 final answer。
- 一个简单 agentic loop，在 confidence 低时 reformulate query。

## 交付它

本课产出 `outputs/skill-multimodal-rag-designer.md`。给定一个带 multimodal query flow 的 product spec，它会设计 retrievers、fusion、generator 和 evaluation。

## 练习

1. 提出一个 medical-triage multimodal RAG：query = injury photo + text symptoms。哪些 modalities 从哪些 KB 检索？

2. Score fusion 是简单 weighted sum。它有什么 failure mode 是 MoE fusion 可以避免的？

3. 阅读 Abootorabi 等人的 taxonomy（第 3 节）。三个 canonical sub-problems 是什么？如何映射到你选择的产品？

4. 为 trip-planner multimodal RAG 设计 eval spec。什么 metrics 覆盖 image recall、audio recall 和 composite correctness？

5. Agentic multi-hop RAG 每个 round-trip 都有 latency tax。什么 query difficulty 下 accuracy gain 值得这个 latency？

## 关键术语

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| Cross-modal retrieval | “Query one modality, retrieve another” | Text query 检索 images；image query 检索 text；需要 shared space 或 translator |
| Score fusion | “Combine scores” | Per-modality retrieval scores 的 weighted sum；最简单 fusion |
| MoE fusion | “Modality-routed experts” | Gating network 为每个 query 选择信任哪个 modality 的 scores |
| Grounded generation | “Cite your sources” | 答案中每条 claim 都标注 source index |
| MuRAG | “First multimodal RAG” | 2022 论文，建立 multimodal RAG pattern |
| Agentic multi-hop | “Reformulate and retry” | 第一次 retrieval confidence 低时，LLM 重新查询 retrievers |

## 延伸阅读

- [Abootorabi et al. — Ask in Any Modality (arXiv:2502.08826)](https://arxiv.org/abs/2502.08826)
- [Mei et al. — A Survey of Multimodal RAG (arXiv:2504.08748)](https://arxiv.org/abs/2504.08748)
- [Zhao et al. — Vision RAG Survey (arXiv:2503.18016)](https://arxiv.org/abs/2503.18016)
- [Chen et al. — MuRAG (arXiv:2210.02928)](https://arxiv.org/abs/2210.02928)
- [Liu et al. — REACT (arXiv:2301.10382)](https://arxiv.org/abs/2301.10382)
