# Agent Observability：Langfuse、Phoenix、Opik

> 2026 年，三个 open-source agent observability platforms 最突出。Langfuse（MIT）— 每月 6M+ installs，tracing + prompt management + evals + session replay。Arize Phoenix（Elastic 2.0）— 深度 agent-specific evals、RAG relevancy、OpenInference auto-instrumentation。Comet Opik（Apache 2.0）— automated prompt optimization、guardrails、LLM-judge hallucination detection。

**类型：** 学习
**语言：** Python (stdlib)
**前置要求：** 阶段 14 · 23（OTel GenAI）
**时间：** ~45 分钟

## 学习目标

- 说出三个 top open-source agent observability platforms 及其 licenses。
- 区分它们各自最强的地方：Langfuse（prompt mgmt + sessions）、Phoenix（RAG + auto-instrumentation）、Opik（optimization + guardrails）。
- 解释为什么到 2026 年，89% 的组织报告已经具备 agent observability。
- 实现一个带 LLM-judge evaluation 的 stdlib trace-to-dashboard pipeline。

## 问题

OTel GenAI（第 23 课）给了你 schema。你仍然需要 ingest spans、运行 evaluations、存储 prompt versions、暴露 regressions 的平台。三个竞争者各自强调 lifecycle 的不同部分。

## 概念

### Langfuse (MIT)

- 每月 6M+ SDK installs，19k+ GitHub stars。
- Features：tracing、带 versioning + playground 的 prompt management、evaluations（LLM-as-judge、user feedback、custom）、session replays。
- 2025 年 6 月：原 commercial modules（LLM-as-a-judge、annotation queues、prompt experiments、Playground）以 MIT 开源。
- 最适合：带紧密 prompt-management loop 的 end-to-end observability。

### Arize Phoenix (Elastic License 2.0)

- 更深的 agent-specific evaluation：trace clustering、anomaly detection、RAG 的 retrieval relevancy。
- 原生 OpenInference auto-instrumentation。
- 与 managed Arize AX 搭配用于生产。
- 没有 prompt versioning — 定位是和更广平台并用的 drift/behavioral-regression tool。
- 最适合：RAG relevancy、behavioral drift、anomaly detection。

### Comet Opik (Apache 2.0)

- 通过 A/B experiments 做 automated prompt optimization。
- Guardrails（PII redaction、topical constraints）。
- LLM-judge hallucination detection。
- Comet 自己测量的 benchmark：Opik logs + evals 用 23.44s，Langfuse 用 327.15s（~14x gap）— vendor benchmarks 只当方向性参考。
- 最适合：optimization loop、automated experimentation、guardrail enforcement。

### Industry data

根据 Maxim（2026 field analysis）：89% 的组织已经有 agent observability；质量问题是 production 的首要障碍（32% 受访者提到）。

### 选择一个

| Need | Pick |
|------|------|
| All-in-one with prompt management | Langfuse |
| Deep RAG evaluation + drift | Phoenix |
| Automated optimization + guardrails | Opik |
| Open licensing, no ELv2 | Langfuse (MIT) or Opik (Apache 2.0) |
| Datadog / New Relic integration | Any — they all export OTel |

### 这个模式会在哪里出错

- **没有 eval strategy。** 没有 evaluation 的 tracing 只是昂贵 logging。
- **Self-rolled LLM-judge 没有 grounding。** CRITIC pattern（第 05 课）适用 — judges 需要 external tools 做 factual verification。
- **Prompt versions 没有绑定 traces。** 生产回退时，你无法 bisect 到导致问题的 prompt。

## 构建它

`code/main.py` 实现了一个 stdlib trace collector + LLM-judge evaluator：

- Ingest GenAI-shaped spans。
- 按 session 分组，标记 failed runs（guardrail trips、low-confidence evals）。
- 一个 scripted LLM-judge，根据 rubric 给 agent responses 打分。
- 一个 dashboard-like summary：failure rate、top failure reasons、eval score distribution。

运行它：

```
python3 code/main.py
```

输出：per-session eval scores 和 failure categorization，对应 Langfuse/Phoenix/Opik 会展示的内容。

## 使用它

- **Langfuse** self-hosted 或 cloud；通过 OTel 或它们的 SDK 接入。
- **Arize Phoenix** self-hosted；auto-instrument OpenInference。
- **Comet Opik** self-hosted 或 cloud；automated optimization loop。
- **Datadog LLM Observability** 用于已经运行 Datadog 的 mixed ops+ML teams。

## 发布它

`outputs/skill-obs-platform-wiring.md` 会选择一个平台，并把 traces + evals + prompt versions 接到现有 agent。

## 练习

1. 导出一周 OTel traces 到 Langfuse cloud（free tier）。哪些 sessions 失败？为什么？
2. 为你的领域写一个 LLM-judge rubric（factual correctness、tone、scope adherence）。在 50 条 traces 上测试。
3. 对比 Langfuse prompt versioning 和 Phoenix trace clustering。哪个更快告诉你哪里坏了？
4. 阅读 Opik guardrail docs。把 PII redaction guardrail 接到一次 agent run。
5. 在你的 corpus 上 benchmark 三者。忽略 vendor-published numbers；测你自己的。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|----------------|------------------------|
| Tracing | "Spans collector" | Ingest OTel / SDK spans；按 session 建索引 |
| Prompt management | "Prompt CMS" | 与 traces 绑定的 versioned prompts |
| LLM-as-judge | "Automated eval" | 单独的 LLM 根据 rubric 给 agent output 打分 |
| Session replay | "Trace playback" | 逐步回放过去的 runs 用于 debugging |
| RAG relevancy | "Retrieval quality" | Retrieved context 是否匹配 query |
| Trace clustering | "Behavioral grouping" | 聚类类似 runs，用于 drift detection |
| Guardrail enforcement | "Policy at log time" | 对 logged content 做 PII/toxicity/scope checks |

## 延伸阅读

- [Langfuse docs](https://langfuse.com/) — tracing、evals、prompt mgmt
- [Arize Phoenix docs](https://docs.arize.com/phoenix) — auto-instrumentation、drift
- [Comet Opik](https://www.comet.com/site/products/opik/) — optimization + guardrails
- [OpenTelemetry GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — 三者共同消费的 schema
