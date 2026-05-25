# Capstone 11 — LLM Observability 与 Eval Dashboard

> Langfuse 转向 open-core。Arize Phoenix 发布了 2026 GenAI semconv mappings。Helicone 和 Braintrust 都加码 per-user cost attribution。Traceloop 的 OpenLLMetry 成了事实上的 SDK instrumentation。生产形态是 ClickHouse 存 traces、Postgres 存 metadata、Next.js 做 UI，再加一小支 eval jobs（DeepEval、RAGAS、LLM-judge）在 sampled traces 上运行。构建一个 self-hosted 版本，从至少四个 SDK families 摄入，并演示在五分钟内捕获一个 injected regression。

**类型：** Capstone
**语言：** TypeScript（UI）、Python / TypeScript（ingest + evals）、SQL（ClickHouse）
**前置要求：** 阶段 11（LLM engineering）、阶段 13（tools）、阶段 17（infrastructure）、阶段 18（safety）
**覆盖阶段：** P11 · P13 · P17 · P18
**时间：** 25 小时

## 问题

到 2026 年，每个运行 production traffic 的 AI 团队都会在模型旁边保留一个 observability plane。Cost attribution。Hallucination detection。Drift monitoring。Jailbreak signal。SLO dashboards。PII leak alerts。开源参考实现 Langfuse、Phoenix、OpenLLMetry 都收敛到 OpenTelemetry GenAI semantic conventions，把它作为 ingest schema。现在你可以用一个 SDK instrument OpenAI、Anthropic、Google、LangChain、LlamaIndex 和 vLLM，并发出兼容 spans。

你将构建一个 self-hosted dashboard，从至少四个 SDK families 摄入，针对 sampled traces 运行一小组 eval jobs，检测 drift 并 alert。衡量门槛是：给定一个故意注入的 regression（prompt 开始产出 PII），dashboard 能在五分钟内捕获并发出 alert。

## 概念

Ingest 是 OTLP HTTP。SDK 产出 GenAI-semconv spans：`gen_ai.system`、`gen_ai.request.model`、`gen_ai.usage.input_tokens`、`gen_ai.response.id`、`llm.prompts`、`llm.completions`。Spans 落入 ClickHouse 做 columnar analytics；metadata（users、sessions、apps）落入 Postgres。

Evals 作为 batch jobs 在 sampled traces 上运行。DeepEval 给 faithfulness、toxicity 和 answer relevance 打分。trace 携带 retrieval context 时，RAGAS 会给 retrieval metrics 打分。Custom LLM-judges 运行 domain-specific checks（PII leak、off-policy response）。Eval runs 会写回同一个 ClickHouse，作为 linked to parent trace 的 eval spans。

Drift detection 会随时间观察 embedding-space distributions（prompt embeddings 上的 PSI 或 KL divergence）以及 eval-score trends。Alerts 进入 Prometheus Alertmanager，再到 Slack / PagerDuty。UI 使用 Next.js 15 和 Recharts。

## 架构

```
production apps:
  OpenAI SDK  +  Anthropic SDK  +  Google GenAI SDK
  LangChain + LlamaIndex + vLLM
       |
       v
  OpenTelemetry SDK with GenAI semconv
       |
       v  OTLP HTTP
  collector (ingest, sample, fan-out)
       |
       +-------------+-----------+
       v             v           v
   ClickHouse    Postgres    S3 archive
   (spans)       (metadata)  (raw events)
       |
       +---> eval jobs (DeepEval, RAGAS, LLM-judge)
       |     sampled or all-trace
       |     write eval spans back
       |
       +---> drift detector (PSI / KL on prompt embeddings)
       |
       +---> Prometheus metrics -> Alertmanager -> Slack / PagerDuty
       |
       v
   Next.js 15 dashboard (Recharts)
```

## 技术栈

- Ingest：OpenTelemetry SDKs + GenAI semantic conventions；OTLP HTTP transport
- Collector：带 tail-sampling processor 的 OpenTelemetry Collector（用于 cost control）
- Storage：ClickHouse 存 spans，Postgres 存 metadata，S3 存 raw event archive
- Evals：DeepEval、RAGAS 0.2、Arize Phoenix evaluator pack、custom LLM-judge
- Drift：每周对 pooled prompt embeddings（sentence-transformers）计算 PSI / KL
- Alerting：Prometheus Alertmanager -> Slack / PagerDuty
- UI：Next.js 15 App Router + Recharts + server actions
- 开箱支持的 SDKs：OpenAI、Anthropic、Google GenAI、LangChain、LlamaIndex、vLLM

## 构建它

1. **Collector config。** OpenTelemetry Collector，带 OTLP HTTP receiver、tail-sampler（保留 100% errored traces 和 10% successes），以及到 ClickHouse 和 S3 的 exporters。

2. **ClickHouse schema。** 表 `spans`，列与 GenAI semconv 对齐：`gen_ai_system`、`gen_ai_request_model`、`input_tokens`、`output_tokens`、`latency_ms`、`prompt_hash`、`trace_id`、`parent_span_id`，再加一个存长 payloads 的 JSON bag。按 user_id 和 app_id 添加 secondary indexes。

3. **SDK coverage test。** 用每个 SDK（OpenAI、Anthropic、Google、LangChain、LlamaIndex、vLLM）写一个小 client app，并用 OpenLLMetry auto-instrument。验证每个都会产出 canonical GenAI spans，且能落到 ClickHouse。

4. **Eval jobs。** scheduled job 读取最近 15 分钟 sampled traces，并运行 DeepEval faithfulness、toxicity 和 answer relevance。输出是 linked to parent trace 的 eval spans。

5. **Custom LLM-judge。** PII-leak judge：给定一个 response，调用 guard LLM 给 PII leak likelihood 打分。高分 responses 进入 triage queue。

6. **Drift detection。** weekly job 计算本周 pooled prompt embeddings 与 trailing 4-week baseline 之间的 PSI。如果 PSI 超过阈值，发 alert。

7. **Dashboard。** Next.js 15 页面：overview（spans/sec、cost/user、p95 latency）、traces（search + waterfall）、evals（faithfulness trend、toxicity）、drift（PSI over time）、alerts。

8. **Alerting chain。** Prometheus exporter 读取 eval score aggregates 和 latency percentiles；Alertmanager 将 warnings 路由到 Slack，将 critical breaches 路由到 PagerDuty。

9. **Regression probe。** 注入 bug：被评测的 chatbot 以 1% 概率泄漏 fake SSNs。衡量 MTTR：从 bug deployed 到 Slack alert。

## 使用它

```
$ curl -X POST https://my-otel-collector/v1/traces -d @trace.json
[collector]  accepted 1 trace, 3 spans
[clickhouse] inserted 3 spans (app=chat, user=u_42)
[eval]       DeepEval faithfulness 0.82, toxicity 0.03
[drift]      weekly PSI 0.08 (below 0.2 threshold)
[ui]         live at https://obs.example.com
```

## 交付它

`outputs/skill-llm-observability.md` 是交付物。给定一个 LLM application，dashboard 会摄入 traces、运行 evals、对 drift 告警，并在 Next.js 中展示 cost/user breakdown。

| 权重 | 标准 | 衡量方式 |
|:-:|---|---|
| 25 | Trace-schema coverage | 产出 canonical GenAI spans 的 SDK families 数量（target: 6+） |
| 20 | Eval correctness | DeepEval / RAGAS scores vs hand-labeled set |
| 20 | Dashboard UX | injected regression 上的 MTTR（目标低于 5 分钟） |
| 20 | Cost / scale | 持续 1k spans/sec ingest 且无 backlog |
| 15 | Alerting + drift detection | Prometheus/Alertmanager chain end-to-end 演练 |
| **100** | | |

## 练习

1. 为 Haystack framework 添加 custom instrumentation。验证 canonical spans 带 faithful `gen_ai.*` attributes 落入 ClickHouse。

2. 在相同 traces 上把 DeepEval 换成 Phoenix evaluators。衡量两个 eval engines 之间的 score drift。

3. 锐化 drift detector：按 app-id 计算 PSI，而不是全局计算。展示 per-app drift trails。

4. 添加 “user impact” 页面：cost-per-user 和 failure-rate-per-user，带 sparklines。

5. 构建一个 tail-sampling policy，保留 100% toxicity > 0.5 的 traces，并对其余 traces 做 10% stratified sample。衡量引入的 sampling bias。

## 关键词汇

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|------------------------|
| GenAI semconv | “OTel LLM attributes” | 2025 OpenTelemetry LLM span attributes 规范（system、model、tokens） |
| Tail sampling | “Post-trace sample” | Collector 在 trace 完成后决定保留或丢弃（可以查看 errors） |
| PSI | “Population stability index” | 比较两个 distributions 的 drift metric；> 0.2 通常表示有意义的 drift |
| LLM-judge | “Eval as model” | 用一个 LLM 按 rubric（faithfulness、toxicity、PII）给另一个 LLM 输出打分 |
| Tail-sampling policy | “Keep-rule” | 决定哪些 traces persist vs drop 的规则；errored + sample-rate |
| Eval span | “Linked eval trace” | 携带 eval score 并链接到原始 LLM call span 的 child span |
| Cost per user | “Unit economics” | 某个时间窗口内归因到 user_id 的美元成本；关键 product metric |

## 延伸阅读

- [Langfuse](https://github.com/langfuse/langfuse) — reference open-core observability platform
- [Arize Phoenix](https://github.com/Arize-ai/phoenix) — drift support 很强的 alternate reference
- [OpenLLMetry (Traceloop)](https://github.com/traceloop/openllmetry) — auto-instrumentation SDK family
- [OpenTelemetry GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — ingest schema
- [Helicone](https://www.helicone.ai) — alternate hosted observability
- [Braintrust](https://www.braintrust.dev) — alternate eval-first platform
- [ClickHouse documentation](https://clickhouse.com/docs) — columnar span store
- [DeepEval](https://github.com/confident-ai/deepeval) — evaluator library
