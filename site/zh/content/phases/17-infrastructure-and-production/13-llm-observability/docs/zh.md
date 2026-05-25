# LLM Observability Stack Selection

> 2026 年 observability 市场分成两类。Development platforms（LangSmith、Langfuse、Comet Opik）把 monitoring 与 evals、prompt management、session replays 捆在一起。Gateway/instrumentation tools（Helicone、SigNoz、OpenLLMetry、Phoenix）专注 telemetry。Langfuse 是 MIT-licensed core，OSS 平衡很强（free cloud 每月 50K events）。Phoenix 是 OpenTelemetry-native，采用 Elastic License 2.0：很适合 drift/RAG visualization，但不是持久 production backend。Arize AX 使用 zero-copy Iceberg/Parquet integration，声称比 monolithic observability 便宜 100x。LangSmith 领先于 LangChain/LangGraph，$39/user/mo，self-host 仅 Enterprise。Helicone 基于 proxy，15-30 分钟设置，free 每月 100K req，但 agent traces 深度较浅。常见 production pattern：Gateway（Helicone/Portkey）+ eval platform（Phoenix/TruLens），用 OpenTelemetry 粘起来。

**类型：** 学习
**语言：** Python（stdlib，玩具版 trace-sampling simulator）
**前置要求：** 阶段 17 · 08（Inference Metrics），阶段 14（Agent Engineering）
**时间：** ~60 分钟

## 学习目标

- 区分 development platforms（bundled：evals + prompts + sessions）与 gateway/telemetry tools（仅 traces + metrics）。
- 把六个主流工具（Langfuse、LangSmith、Phoenix、Arize AX、Helicone、Opik）映射到 license、pricing 和 sweet-spot use cases。
- 解释 OpenTelemetry-glue pattern：如何把 gateway tool 与独立 eval platform 组合。
- 说出 2026 年成本差异点（Arize AX 的 zero-copy approach vs monolithic ingest），并说明约 100x multiplier。

## 问题

你交付了一个 LLM feature。它能工作。但你对 prompt failures、tool loops、latency regressions、cost spikes 或 prompt-cache hit rate 没有可见性。你搜索 “LLM observability”，得到八个工具，都声称以三个不同价位解决同一个问题。

它们并不解决同一个问题。LangSmith 回答“这个 LangGraph run 为什么失败？”Phoenix 回答“我的 RAG pipeline 是否在 drift？”Helicone 回答“哪个 app 正在烧 tokens？”Langfuse 回答“我能不能 self-host 整套东西？”工具不同，受众不同。

选择涉及四个轴：stack（LangChain？raw SDK？multi-vendor？）、license tolerance（只要 MIT？Elastic 可接受？commercial 没问题？）、budget（free tier？$100/mo？$1000/mo？），以及 self-host（必须？nice-to-have？永不？）。

## 概念

### 两类工具

**Development platforms** 把 observability 与 evals、prompt management、dataset versioning、session replay 捆绑。你运行 experiments，看哪个 prompt 有效，用 dataset-regression 比较新 prompt 与旧 winners。LangSmith、Langfuse、Comet Opik。

**Gateway/telemetry tools** instrument inference calls：prompt、response、tokens、latency、model、cost。Helicone、SigNoz、OpenLLMetry、Phoenix。更 minimalist。可以通过 OpenTelemetry 与独立 eval tool 组合。

### Langfuse — OSS balance

- Core Apache / MIT licensed；可通过 Docker self-host。
- Cloud free tier：50K events/month。Paid：team $29/mo。
- Evals、prompt management、traces、datasets。四类 dev-platform features 覆盖都合理。
- Sweet spot：你想要 LangSmith-class features，但必须 self-host 或坚持 OSS license。

### Phoenix（Arize）— telemetry-first、OpenTelemetry-native

- Elastic License 2.0；self-host 很简单。
- 非常擅长 RAG 和 drift visualization。Embedding-space scatter plots 是 first-class。
- 不是设计为 persistent production backend，主要是 development-time observability。
- Sweet spot：RAG pipeline development、drift debugging，与独立 production gateway 搭配。

### Arize AX — scale play

- Commercial。通过 Iceberg/Parquet 做 zero-copy data lake integration。
- 声称在 scale 下比 monolithic observability（Datadog-class）便宜约 100x。数学是：你把 traces 存在自己的 S3 Parquet 中，Arize 直接读取。
- Sweet spot：>10M traces/day，已有 data lake，想要 LLM-specific dashboards 但不想付 Datadog pricing。

### LangSmith — LangChain/LangGraph first

- Commercial，$39/user/month。Self-host 仅 Enterprise。
- 对 LangChain 和 LangGraph stacks 是 best-in-class。如果你不在二者上，就没那么有吸引力。
- Sweet spot：团队承诺使用 LangChain，并愿意付费。

### Helicone — proxy-based minimum viable

- 通过把 `OPENAI_API_BASE` 换成 Helicone proxy，15-30 分钟设置。
- MIT licensed；100K req/mo free，paid $20/mo+。
- 包含 failover、caching、rate limits，也扮演 gateway。
- 对 agent / multi-step traces 深度较弱。
- Sweet spot：quick start、single-stack app、需要 gateway + observability 合一。

### Opik（Comet）— OSS dev platform

- Apache 2.0，fully OSS。
- 与 Langfuse 类似的 feature set，带 Comet heritage。
- Sweet spot：已经在 Comet 上的 ML teams，希望在同一 pane 中做 LLM observability。

### SigNoz — OpenTelemetry-first full APM

- Apache 2.0。通过 OpenTelemetry 处理 general APM 和 LLM。
- Sweet spot：跨 services 和 LLM calls 的统一 observability。

### Glue：OpenTelemetry + GenAI semantic conventions

OpenTelemetry 在 2025 年末发布 GenAI semantic conventions（`gen_ai.system`、`gen_ai.request.model`、`gen_ai.usage.input_tokens`）。能消费 OTel 的工具可以互操作。正在形成的 production pattern：

1. 从每个 LLM call 发出带 GenAI conventions 的 OTel。
2. 路由到 gateway（Helicone / Portkey）用于 day-to-day。
3. Dual-ship 到 eval platform（Phoenix / Langfuse）用于 regressions。
4. 存档到 data lake（Iceberg），通过 Arize AX 或 DuckDB 做 long-term analysis。

### 陷阱：在错误层 instrument

在 agent framework 内部 instrument（例如添加 LangSmith traces）会把你耦合到那个 framework。在 HTTP/OpenAI-SDK layer instrument（通过 OpenLLMetry 或你的 gateway）更 portable。

### Sampling — 不能保存一切

超过 1M requests/day 时，full-trace retention 的成本会高于 LLM calls。按规则 sample：100% errors，100% high-cost，5% success。永远保留 aggregates；raw 只保留 long tail。

### 你应该记住的数字

- Langfuse free cloud：50K events/month。
- LangSmith：$39/user/month。
- Helicone free：100K req/month。
- Arize AX claim：scale 下比 monolithic 便宜约 100x。
- OpenTelemetry GenAI conventions：2025 shipping，2026 广泛采用。

## 使用它

`code/main.py` 模拟一个 1M-trace day，比较多种 retention strategies（100% ingest、sampling、sampling + errors）。报告 storage cost 以及每种策略丢失了什么。

## 交付它

本课会产出 `outputs/skill-observability-stack.md`。给定 stack、scale、budget、license posture，它会选择工具组合。

## 练习

1. 你的团队使用 LangChain，并想要 OSS self-hosted observability。选择 Langfuse 或 Opik，并说明理由。
2. 5M traces/day 下，Datadog 报价 $150K/month。计算 Arize AX 的 break-even。
3. 设计一组你们组织应强制每个 LLM call 都带上的 OpenTelemetry GenAI attributes。
4. 论证 Phoenix alone 是否足以用于 production。什么时候不够？
5. Helicone 有 20ms proxy overhead。在 P99 TTFT 300 ms 下可接受吗？如果 SLA 是 100 ms 呢？

## 关键词汇

| 术语 | 大家常说 | 实际含义 |
|------|----------|----------|
| OpenLLMetry | “OTel for LLMs” | 面向 LLMs 的 open-source OpenTelemetry instrumentation |
| GenAI conventions | “OTel attributes” | LLM calls 的标准 OTel attribute names |
| LangSmith | “LangChain observability” | 与 LangChain ecosystem 捆绑的 commercial platform |
| Langfuse | “OSS LangSmith” | MIT OSS，feature set 类似 |
| Phoenix | “Arize dev tool” | OpenTelemetry-native dev/eval platform |
| Arize AX | “scale observability” | Commercial zero-copy Iceberg/Parquet observability |
| Helicone | “proxy observability” | 收集 LLM telemetry + gateway features 的 HTTP proxy |
| Opik | “Comet LLM” | Comet 出品的 Apache 2.0 OSS dev platform |
| Session replay | “trace rerun” | 带 tool calls 的完整 agent session replay |
| Eval | “offline test” | 在 labeled dataset 上运行候选 model/prompt |

## 延伸阅读

- [SigNoz — Top LLM Observability Tools 2026](https://signoz.io/comparisons/llm-observability-tools/)
- [Langfuse — Arize AX Alternative analysis](https://langfuse.com/faq/all/best-phoenix-arize-alternatives)
- [PremAI — Setting Up Langfuse, LangSmith, Helicone, Phoenix](https://blog.premai.io/llm-observability-setting-up-langfuse-langsmith-helicone-phoenix/)
- [OpenTelemetry GenAI Semantic Conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/)
- [Arize Phoenix docs](https://docs.arize.com/phoenix)
- [Helicone docs](https://docs.helicone.ai/)
