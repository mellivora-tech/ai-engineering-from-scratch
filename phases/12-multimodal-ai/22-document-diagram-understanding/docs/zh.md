# Document and Diagram Understanding

> 文档不是照片。PDF、scientific paper、invoice 或 handwritten form 有 layout、tables、diagrams、footnotes、headers 和 semantic structure，这些是普通 image understanding 捕捉不到的。VLM 之前的 stack 是 pipeline：Tesseract OCR + LayoutLMv3 + table-extraction heuristics。VLM 浪潮用 OCR-free models 替换了它，例如 Donut（2022）、Nougat（2023）、DocLLM（2023），它们直接输出 structured markup。到 2026 年，frontier 只是“把 page image 以 2576px native 喂给 Claude Opus 4.7”，structured-markup output 几乎免费得到。本课阅读 document AI 的三时代弧线。

**类型：** 构建
**语言：** Python（stdlib，layout-aware document parser skeleton）
**前置要求：** 阶段 12 · 05（LLaVA），阶段 5（NLP）
**时间：** ~180 分钟

## 学习目标

- 解释 document AI 的三个时代：OCR pipeline、OCR-free、VLM-native。
- 描述 LayoutLMv3 的三个输入流：text、layout（bbox）、image patches，以及 unified masking。
- 比较 Donut（OCR-free，image -> markup）、Nougat（scientific paper -> LaTeX）、DocLLM（layout-aware generative）、PaliGemma 2（VLM-native）。
- 为新任务选择 document model（invoices、scientific papers、handwritten forms、Chinese receipts）。

## 问题

“Understand this PDF” 看似简单，其实很难。信息存在于：

- Text content（90% 信号）。
- Layout（headers、footnotes、sidebars、two-column format）。
- Tables（rows、columns、merged cells）。
- Figures and diagrams。
- Handwritten annotations。
- Fonts and typography（title vs body）。

Raw OCR 会导出文本，并丢失其他部分。一个关注 invoices 的系统需要知道 “Total: $1,245” 来自页面右下角，而不是 footnote。

## 概念

### Era 1 — OCR pipeline（2021 年前）

经典 stack：

1. PDF -> 每页 image。
2. Tesseract（或商业 OCR）提取 text 和 per-word bounding boxes。
3. Layout analyzer 识别 blocks（header、table、paragraph）。
4. Table structure recognizer 解析 tables。
5. Domain rules + regex 提取 fields。

对干净 printed text 有效。对 handwriting、skewed scans、complex tables、non-English scripts 会崩。每个 failure mode 都需要定制 exception path。

### TrOCR（2021）

TrOCR（Li 等人，arXiv:2109.10282）用在 synthetic + real text images 上训练的 transformer encoder-decoder，替换了 Tesseract 的经典 CNN-CTC。对 handwritten 和 multilingual text 是明显胜利。它仍然是 pipeline（detector then TrOCR then layout），但 OCR step 大幅改善。

### Era 2 — OCR-free（2022-2023）

第一批 OCR-free models 说：完全跳过 detection，直接把 image pixels 映射到 structured output。

Donut（Kim 等人，arXiv:2111.15664）：
- Encoder-decoder transformer，encoder 是 Swin-B。
- Output 可以是 form understanding 的 JSON、summarization 的 markdown，或任何 task-specific schema。
- 无 OCR、无 layout、无 detection。

Nougat（Blecher 等人，arXiv:2308.13418）：
- 专门在 scientific papers 上训练。
- Output 是 LaTeX / markdown。
- 处理 equations、multi-column layout、figures。
- 每个 arXiv-parser 都会调用的模型。

这些是 specialists，不是 generalists。Donut 处理 scientific paper 会失败；Nougat 处理 invoice 会失败。

### LayoutLMv3（2022）

另一条路线。LayoutLMv3（Huang 等人，arXiv:2204.08387）保留 OCR，但加入 layout understanding：

- 三个输入流：OCR text tokens、per-token 2D bounding boxes、image patches。
- 三种 modality 上的 masked training objective（masked text、masked patches、masked layout）。
- 下游：classification、entity extraction、table QA。

LayoutLMv3 是 OCR-based document understanding 的巅峰。Forms 和 invoices 上很强。需要上游 OCR。标准化 document benchmarks 上是 VLM 前最佳准确率。

### DocLLM（2023）

DocLLM（Wang 等人，arXiv:2401.00908）是 LayoutLM 的 generative sibling。基于 layout tokens 生成 free-form answers。更适合文档 QA；仍依赖 OCR input。

### Era 3 — VLM-native（2024+）

2024 年 VLM 已经足够强，可以完全替代 pipeline。把整页高分辨率图像喂给 VLM，提问，得到答案。

- LLaVA-NeXT 336-tile AnyRes 适合小文档。
- Qwen2.5-VL dynamic-resolution 原生处理 2048+ pixels。
- Claude Opus 4.7 支持 2576px documents。
- PaliGemma 2（2025 年 4 月）专门为 documents + handwriting 训练。

VLM-native 与 OCR-pipeline 的差距迅速闭合。到 2026 年，VLM-native 在这些方面胜出：

- Scene text（hand-written + printed，mixed scripts）。
- 带 merged cells 的 complex tables。
- 嵌入文本的 math equations。
- 带 text annotations 的 figures。

OCR pipelines 仍然在这些地方胜出：

- 大规模 pure-scan workloads，其中 per-page latency 很重要。
- Pipeline reliability（deterministic failures vs VLM hallucinations）。
- 需要 auditable OCR output 的 regulated environments。

### Claude 4.7 / GPT-5 frontier

在 2576-pixel native input 下，frontier VLMs 的 document understanding 接近 human accuracy。2026 年早期 benchmark numbers：

- DocVQA：Claude 4.7 ~95.1，PaliGemma 2 ~88.4，Nougat ~77.3，pipelined LayoutLMv3 ~83。
- ChartQA：Claude 4.7 ~92.2，GPT-4V ~78。
- VisualMRC：Claude 4.7 ~94。

Closed-model gap 主要来自 resolution 和 base-LLM scale。Open models at 7B 落后几分，但正在追赶。

### Math equations 与 LaTeX output

Scientific papers 需要 equations 的精确 LaTeX output。Nougat 专门为此训练。带 LaTeX targets 训练的 VLM（Qwen2.5-VL-Math、Nougat derivatives）能产出可用 LaTeX。没有显式 LaTeX 训练的 VLM 会给出可读但不精确的 transcription。

2026 年 scientific-paper pipeline：先在 PDF 上跑 Nougat，再让 VLM 处理棘手页面。

### Handwriting

仍然是最难子任务。Mixed printed + handwritten（医生笔记、填写表格）是 OCR pipelines 在成本上仍超过 VLMs 的地方。Handwritten-only VLMs 正在改善（Claude 4.7、PaliGemma 2）。

### 2026 recipe

新 document-AI project：

- 大规模 pure-printed invoices：LayoutLMv3 + rules，成本高效。
- Mixed documents（scientific + handwritten + forms）：VLM-native（PaliGemma 2 或 Qwen2.5-VL）。
- Full arXiv ingestion：Nougat 处理 math，VLM 处理 figures。
- Regulatory：OCR pipeline + VLM validator 做 cross-check。

## 使用它

`code/main.py`：

- Toy layout-aware tokenizer：给定（text, bbox）pairs，生成 LayoutLMv3-style input。
- Donut-style task schema generator：forms 的 JSON template。
- 比较 OCR-pipeline、Donut、Nougat 和 VLM-native 每页 token budgets。

## 交付它

本课产出 `outputs/skill-document-ai-stack-picker.md`。给定 document-AI project（domain、scale、quality、regulatory），它会在 OCR pipeline、OCR-free specialist 和 VLM-native 中选择。

## 练习

1. 你的项目每天处理 10M invoices。哪个 stack 在不损失 accuracy 的情况下最小化 cost-per-page？

2. 为什么 LayoutLMv3 在 form QA 上超过 pure-CLIP-VLMs，但在 scene-text 上落后？Bbox stream 放弃了什么？

3. Nougat 生成 LaTeX。提出一个 VLM-native output 在 LaTeX fidelity 上超过 Nougat 的 test case，以及一个 Nougat 胜出的 case。

4. 阅读 PaliGemma 2 论文（Google，2024）。相较 PaliGemma 1，提升 document accuracy 的关键 training-data addition 是什么？

5. 设计一个 regulatory-safe hybrid：OCR pipeline 作为 primary，VLM 作为 secondary cross-check。如何解决 disagreement？

## 关键术语

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| OCR pipeline | “Tesseract-style” | 分阶段 stack：detect -> OCR -> layout -> rules；确定性但脆弱 |
| OCR-free | “Donut-style” | 跳过显式 OCR 的 image-to-output transformer；单模型 |
| Layout-aware | “LayoutLM” | 输入包含 per-token bbox coordinates；跨 modality unified masking |
| VLM-native | “Frontier VLM” | 直接把 page image 以高分辨率喂给 Claude/GPT/Qwen VLM；无 pipeline |
| DocVQA | “Doc benchmark” | Document VQA 标准；最常被引用的分数 |
| Markup output | “LaTeX / MD” | Structured output format，而不是 free-form text；便于下游自动化 |

## 延伸阅读

- [Li et al. — TrOCR (arXiv:2109.10282)](https://arxiv.org/abs/2109.10282)
- [Blecher et al. — Nougat (arXiv:2308.13418)](https://arxiv.org/abs/2308.13418)
- [Huang et al. — LayoutLMv3 (arXiv:2204.08387)](https://arxiv.org/abs/2204.08387)
- [Kim et al. — Donut (arXiv:2111.15664)](https://arxiv.org/abs/2111.15664)
- [Wang et al. — DocLLM (arXiv:2401.00908)](https://arxiv.org/abs/2401.00908)
