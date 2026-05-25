# OpenTelemetry GenAI Semantic Conventions

> OpenTelemetry 的 GenAI SIG（2024 年 4 月启动）定义了 agent telemetry 的标准 schema。Span names、attributes 和 content-capture rules 在 vendors 之间收敛，所以 agent traces 在 Datadog、Grafana、Jaeger 和 Honeycomb 中表达同一件事。

**类型：** 学习 + 构建
**语言：** Python (stdlib)
**前置要求：** 阶段 14 · 13（LangGraph），阶段 14 · 24（Observability Platforms）
**时间：** ~60 分钟

## 学习目标

- 说出 GenAI span categories：model/client、agent、tool。
- 区分 `invoke_agent` CLIENT 与 INTERNAL spans，并说明各自何时适用。
- 列出 top-level GenAI attributes：provider name、request model、data-source ID。
- 解释 content-capture contract：opt-in、`OTEL_SEMCONV_STABILITY_OPT_IN`、external-reference recommendation。

## 问题

每个 vendor 都发明自己的 span names。Ops teams 最后要为每个 framework 单独搭 dashboards。OpenTelemetry 的 GenAI SIG 通过定义一套整个生态都能对齐的标准，修复了这个问题。

## 概念

### Span categories

1. **Model / client spans。** 覆盖原始 LLM calls。由 provider SDKs（Anthropic、OpenAI、Bedrock）和 framework model adapters 发出。
2. **Agent spans。** `create_agent`（构造 agent 时）和 `invoke_agent`（agent 运行时）。
3. **Tool spans。** 每次 tool invocation 一个 span；通过 parent-child relation 连接到 agent span。

### Agent span naming

- Span name：如果有名字，使用 `invoke_agent {gen_ai.agent.name}`；否则 fallback 到 `invoke_agent`。
- Span kind：
  - **CLIENT** — 用于 remote agent services（OpenAI Assistants API、Bedrock Agents）。
  - **INTERNAL** — 用于 in-process agent frameworks（LangChain、CrewAI、本地 ReAct）。

### Key attributes

- `gen_ai.provider.name` — `anthropic`、`openai`、`aws.bedrock`、`google.vertex`。
- `gen_ai.request.model` — model ID。
- `gen_ai.response.model` — resolved model（由于 routing，可能和 request 不同）。
- `gen_ai.agent.name` — agent identifier。
- `gen_ai.operation.name` — `chat`、`completion`、`invoke_agent`、`tool_call`。
- `gen_ai.data_source.id` — 用于 RAG：查询了哪个 corpus 或 store。

Anthropic、Azure AI Inference、AWS Bedrock、OpenAI 都有 technology-specific conventions。

### Content capture

默认规则：instrumentations 默认 SHOULD NOT capture inputs/outputs。Capture 通过以下字段 opt-in：

- `gen_ai.system_instructions`
- `gen_ai.input.messages`
- `gen_ai.output.messages`

推荐的生产模式：把 content 存在外部（S3、你的 log store），在 spans 上记录 references（pointer IDs，而不是 prose）。这就是第 27 课 content-poisoning defense 接入 observability 的方式。

### Stability

截至 2026 年 3 月，大多数 conventions 仍是 experimental。通过下面的变量 opt in 到 stable preview：

```
OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental
```

Datadog v1.37+ 会把 GenAI attributes 原生映射进它的 LLM Observability schema。其他 backends（Grafana、Honeycomb、Jaeger）支持 raw attributes。

### 这个模式会在哪里出错

- **在 spans 中捕获完整 prompts。** PII、secrets、customer data 进入 ops 可读的 traces。应外部存储。
- **没有 `gen_ai.provider.name`。** 缺少 attribution 时，multi-provider dashboards 会断。
- **Spans 没有 parent links。** Tool spans 变成 orphan。始终传播 context。
- **没有设置 stability opt-in。** Backend 升级时 attributes 可能被重命名。

## 构建它

`code/main.py` 实现了一个符合 GenAI conventions 的 stdlib span emitter：

- 带 GenAI attribute schema 的 `Span`。
- 带 `start_span`、nested contexts 的 `Tracer`。
- 一次 scripted agent run 会发出：`create_agent`、`invoke_agent`（INTERNAL）、每个 tool span、LLM calls 的 `chat` spans。
- 一个 content-capture mode，会把 prompts 存到外部，并在 spans 上记录 IDs。

运行它：

```
python3 code/main.py
```

输出：一棵带所有必需 GenAI attributes 的 span tree，以及一个展示 opt-in content references 的 “external store”。

## 使用它

- **Datadog LLM Observability**（v1.37+）原生映射 attributes。
- **Langfuse / Phoenix / Opik**（第 24 课）— auto-instrument ecosystem。
- **Jaeger / Honeycomb / Grafana Tempo** — raw OTel traces；从 GenAI attributes 构建 dashboards。
- **Self-hosted** — 运行带 GenAI processor 的 OTel Collector。

## 发布它

`outputs/skill-otel-genai.md` 会把 OTel GenAI spans 接到现有 agent，并带上 content-capture defaults 和 external-reference storage。

## 练习

1. 用 `invoke_agent`（INTERNAL）+ 每个 tool span 给第 01 课 ReAct loop 加 instrumentation。发送到一个 Jaeger instance。
2. 添加 “references only” mode 的 content capture：prompts 存入 SQLite，span attributes 只携带 row IDs。
3. 阅读 `gen_ai.data_source.id` 的 spec。把它接到第 09 课 Mem0 search。
4. 设置 `OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental`，验证你的 attributes 不会被 collector 重命名。
5. 只从 GenAI attributes 构建一个 dashboard：“哪些 tool errors 和哪些 models 相关？”

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|----------------|------------------------|
| GenAI SIG | "OpenTelemetry GenAI group" | 定义 schema 的 OTel working group |
| invoke_agent | "Agent span" | 表示一次 agent run 的 span name |
| CLIENT span | "Remote call" | 调用 remote agent service 的 span |
| INTERNAL span | "In-process" | in-process agent run 的 span |
| gen_ai.provider.name | "Provider" | anthropic / openai / aws.bedrock / google.vertex |
| gen_ai.data_source.id | "RAG source" | 哪个 corpus/store 产生了 retrieval hit |
| Content capture | "Prompt logging" | Opt-in capture messages；生产中外部存储 |
| Stability opt-in | "Preview mode" | 固定 experimental conventions 的 env var |

## 延伸阅读

- [OpenTelemetry GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — spec
- [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/) — 默认发出 GenAI spans
- [AutoGen v0.4 (Microsoft Research)](https://www.microsoft.com/en-us/research/articles/autogen-v0-4-reimagining-the-foundation-of-agentic-ai-for-scale-extensibility-and-robustness/) — 内置 OTel spans
- [Claude Agent SDK](https://platform.claude.com/docs/en/agent-sdk/overview) — W3C trace context propagation
