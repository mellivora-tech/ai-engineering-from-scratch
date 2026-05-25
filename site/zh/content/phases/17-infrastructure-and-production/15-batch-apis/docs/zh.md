# Batch APIs — 50% Discount 成为行业标准

> 每个主流 provider 都提供 async batch API，带 50% discount 和约 24-hour turnaround。OpenAI、Anthropic、Google，以及多数 inference platforms（Fireworks batch tier、Together batch）都实现了同一模式。把 batch 与 prompt caching 叠加，overnight pipelines 会降到 synchronous-uncached cost 的约 10%。规则残酷而简单：如果不是 interactive，就属于 batch。Content generation pipelines、document classification、data extraction、report generation、bulk labeling、catalog tagging：任何能容忍 24-hour latency 的任务，在迁移到 batch 前都是把钱留在桌上。2026 年 production pattern 是把每个新 LLM workload 分到三条 lane：interactive（synchronous with caching）、semi-interactive（async queue with fallback）、batch（overnight，叠加 cached input）。那些假装 interactive 但其实能容忍分钟级 latency 的 workload 浪费最多。

**类型：** 学习
**语言：** Python（stdlib，玩具版 batch-vs-sync cost simulator）
**前置要求：** 阶段 17 · 14（Prompt & Semantic Caching）
**时间：** ~45 分钟

## 学习目标

- 说出三家 provider batch APIs（OpenAI、Anthropic、Google）以及共同的 50% discount + 24h turnaround guarantees。
- 对 overnight classification workload 计算 batch + cached-input 叠加后的成本，并与 synchronous-uncached baseline 比较。
- 把 workload 分到 interactive / semi-interactive / batch，并说明 lane。
- 说出两个陷阱：partial interactivity（用户期待快于 24h）和 output-schema drift（batch file format 因 provider 而异）。

## 问题

你的团队交付了一个 nightly report generation pipeline。50,000 documents，逐个 summarize，cluster summaries，draft executive brief。同步运行需要 4 小时，每晚 $2,000。你听说了 batch APIs。

Batch 给你 50% off。你还在 system prompt（50k calls 共享）上启用 prompt caching。叠加后，账单降到 $180/night，约 baseline 的 9%。同一 pipeline，三个 config changes。

Batch 是 LLM cost toolkit 中最便宜却最少被使用的杠杆。原因主要是组织性的：团队以为“real-time”，但 SLA 实际是“by morning”。本课就是关于不要把 90% 账单留在桌上。

## 概念

### 三个 batch APIs

**OpenAI Batch API**：上传 JSONL file，内含请求列表。承诺 24-hour turnaround（实践中通常约 2-8 小时）。input 和 output tokens 都 50% discount。`/v1/batches` endpoint。Cache-eligible inputs 还能在此基础上获得 cached-input pricing。

**Anthropic Message Batches**：JSONL upload。24-hour turnaround。50% discount。支持 `cache_control`：cache writes 显式，batch 内自动 reads。

**Google Vertex AI Batch Prediction**：BigQuery 或 GCS input。Gemini 类似 50% discount。与 Vertex pipelines 集成。

### 语义是 asynchronous，不是 slow

Batch 是“我承诺 24 小时内返回”，不是“这会花 24 小时”。典型 P50 是 2-6 小时。Provider 会把你的 batch 安排在 off-peak windows，也就是 GPU inventory 未充分利用的时间。

### 与 caching 叠加

一个 50k-document summarization，共享同一个 4K-token system prompt：

- Synchronous uncached：50000 ×（$input × 4000 + $output × 200），按 full rates。
- Synchronous cached：system prompt 在第一次写入后被缓存；剩余 49999 次 input 便宜 10x。
- Batch cached：以上全部，再对 read 和 write 都打 50% discount。

组合：batch + cache = 约 sync uncached bill 的 10%。任何 overnight 且有 shared system prompt 的 workload 都应该使用它。

### Workload triage

**Interactive** — 用户等待响应。TTFT 重要。Synchronous call with prompt caching。不能 batch。

**Semi-interactive** — 用户提交任务，几分钟后回来查看。Async queue with fallback to sync if batch not available。比如中等 volume 的 RAG indexing。

**Batch** — 用户期望“by morning”或“next hour”。Content pipelines、scale 上的 classification、offline analysis。永远 batch，永远叠加 caching。

常见错误：因为 pipeline 是 production，就把一切归类为 interactive。Production 不是 latency spec，SLA 才是。

### Partial-interactivity 陷阱

有些功能看起来 interactive，但能容忍 5-10 分钟。例子：nightly customer health report 上有个 “refresh” button。用户点击 refresh，等 10 分钟可以接受。团队却把它做成 synchronous。50 个 concurrent refreshes 的成本是 batched-and-delivered-via-email 的 10x。

要问的问题：“24-hour 对这个用户意味着什么？”如果答案是“他们不会注意到”，就 batch。

### Output-schema 陷阱

Batch file formats 因 provider 而异：

- OpenAI：JSONL，每行一个 request。
- Anthropic：JSONL，每行一条 message；response format embedded。
- Vertex：BigQuery table 或 GCS prefix with TFRecord。

跨 providers 写 “one batch client” 意味着每个 provider 都要 adapter code。宣称 multi-provider batch 的 gateways（Portkey、LiteLLM 某些 tiers）仍然只是 thin-wrap raw format。

### 你应该记住的数字

- Providers 的 batch discount：input + output 固定 50%。
- Turnaround SLA：24 小时 guaranteed，2-6 小时 typical P50。
- Stacked batch + cached input：约 sync uncached cost 的 10%。
- Workload triage rule：如果 24h latency 可接受，永远 batch。

## 使用它

`code/main.py` 为 50k-document workload 计算 sync、sync+cache、batch、batch+cache 的成本。报告节省的 $ 和 percent。

## 交付它

本课会产出 `outputs/skill-batch-triager.md`。给定 workload characteristics，它会分流到 interactive/semi/batch 并估算 savings。

## 练习

1. 运行 `code/main.py`。对于 100k-doc pipeline、3K-token system prompt、500-token output，计算 full stack（batch + cache）相对 sync baseline 的节省。
2. 在你熟悉的真实产品中选三个 features。把每个分到 interactive/semi/batch。
3. 一个用户抱怨他们的 report 花了 3 小时。这是 batch mis-triage，还是合理的 interactive？写出 decision criterion。
4. 你的 batch API return SLA 是 24h，但 P99 是 20 小时。如何向用户沟通？edge case 下 downstream system behavior 是什么？
5. 计算 break-even：shared-prefix length 到多少时，batch + cache 比在你自己的 reserved GPU 上 overnight 运行更便宜？

## 关键词汇

| 术语 | 大家常说 | 实际含义 |
|------|----------|----------|
| Batch API | “async discount” | 50% off，24h turnaround |
| JSONL | “batch format” | 每行一个 JSON request；OpenAI/Anthropic 标准 |
| Message Batches | “Anthropic batch” | Anthropic batch API 产品名 |
| Batch prediction | “Vertex batch” | Vertex AI batch API 产品 |
| Turnaround SLA | “24h promise” | Guarantee，不是 typical；typical 是 2-6h |
| Workload triage | “interactivity decision” | Interactive / semi / batch routing decision |
| Output schema | “response format” | 每个 provider 的 JSONL layout；不可移植 |
| Stacked discount | “batch + cache” | 两者都适用时，约为 uncached sync bill 的 10% |

## 延伸阅读

- [OpenAI Batch API](https://platform.openai.com/docs/guides/batch) — JSONL format 和 `/v1/batches` semantics。
- [Anthropic Message Batches](https://docs.anthropic.com/en/docs/build-with-claude/batch-processing) — batch format 和 `cache_control` interaction。
- [Vertex AI Batch Prediction](https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/batch-prediction) — Gemini batch semantics。
- [Finout — OpenAI vs Anthropic API Pricing 2026](https://www.finout.io/blog/openai-vs-anthropic-api-pricing-comparison)
- [Zen Van Riel — LLM API Cost Comparison 2026](https://zenvanriel.com/ai-engineer-blog/llm-api-cost-comparison-2026/)
