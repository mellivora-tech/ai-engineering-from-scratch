# OpenTelemetry GenAI：端到端追踪 Tool Calls

> 一个 agent 调用五个工具、三个 MCP servers 和两个 sub-agents。你需要一条 trace 覆盖所有这些。OpenTelemetry GenAI semantic conventions（v1.37 及以上的 stable attributes）是 2026 年标准，被 Datadog、Langfuse、Arize Phoenix、OpenLLMetry 和 AgentOps 原生支持。本课命名必需 attributes，走通 span hierarchy（agent → LLM → tool），并交付一个可以插入任意 OTel exporter 的 stdlib span emitter。

**类型：** 构建
**语言：** Python（stdlib，OTel span emitter）
**前置要求：** 阶段 13 · 07（MCP server），阶段 13 · 08（MCP client）
**时间：** ~75 分钟

## 学习目标

- 命名 LLM span 和 tool-execution span 所需的 OTel GenAI attributes。
- 构建覆盖 agent loop、LLM call、tool call 和 MCP client dispatch 的 trace hierarchy。
- 决定捕获哪些内容（opt-in）以及默认 redact 哪些内容。
- 在不重写 tool code 的情况下，把 spans 发到本地 collector（Jaeger、Langfuse）。

## 问题

2026 年 2 月的一次 debug：用户报告 “my agent sometimes takes 30 seconds to respond; other times 3 seconds.” 没有 traces。logs 显示了 LLM call，但没有 tool dispatch、没有 MCP server round-trip、没有 sub-agent。你只能猜。最后你发现：一个 MCP server 偶尔 cold-start 挂住。

没有端到端 tracing，你找不到这个。OTel GenAI 修复它。

这些 conventions 在 2025-2026 年由 OpenTelemetry semantic-conventions group 稳定下来。它们定义稳定 attribute names，让 Datadog、Langfuse、Phoenix、OpenLLMetry 和 AgentOps 都能解析同一组 spans。instrument 一次；发送到任意 backend。

## 概念

### Span hierarchy

```
agent.invoke_agent  (top, INTERNAL span)
 ├── llm.chat       (CLIENT span)
 ├── tool.execute   (INTERNAL)
 │    └── mcp.call  (CLIENT span)
 ├── llm.chat       (CLIENT span)
 └── subagent.invoke (INTERNAL)
```

整棵树共享一个 trace id。span ids 链接 parent-child relationships。

### Required attributes

根据 2025-2026 semconv：

- `gen_ai.operation.name` —— `"chat"`、`"text_completion"`、`"embeddings"`、`"execute_tool"`、`"invoke_agent"`。
- `gen_ai.provider.name` —— `"openai"`、`"anthropic"`、`"google"`、`"azure_openai"`。
- `gen_ai.request.model` —— requested model string（例如 `"gpt-4o-2024-08-06"`）。
- `gen_ai.response.model` —— 实际 served 的模型。
- `gen_ai.usage.input_tokens` / `gen_ai.usage.output_tokens`。
- `gen_ai.response.id` —— provider response id，用于 correlation。

tool spans：

- `gen_ai.tool.name` —— tool identifier。
- `gen_ai.tool.call.id` —— 具体 call id。
- `gen_ai.tool.description` —— tool description（optional）。

agent spans：

- `gen_ai.agent.name` / `gen_ai.agent.id` / `gen_ai.agent.description`。

### Span kinds

- 对跨 process boundary 的调用使用 `SpanKind.CLIENT`（LLM provider、MCP server）。
- 对 agent 自己的 loop steps 和 tool execution 使用 `SpanKind.INTERNAL`。

### Opt-in content capture

默认情况下，spans 携带 metrics 和 timing——不携带 prompts 或 completions。大型 payload 和 PII 默认关闭。设置 `OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental` 和具体 content-capture env vars 才包含内容。生产开启前要仔细 review。

### Events on spans

token-level events 可以作为 span events 添加：

- `gen_ai.content.prompt` —— input messages。
- `gen_ai.content.completion` —— output messages。
- `gen_ai.content.tool_call` —— recorded tool call。

events 在 span 内按时间排序，用于详细 replay。

### Exporters

OTel spans 可以 export 到：

- **Jaeger / Tempo。** OSS，on-prem。
- **Langfuse。** LLM-observability-specific；可视化 token usage。
- **Arize Phoenix。** evals + tracing combined。
- **Datadog。** 商业；原生解析 `gen_ai.*` attributes。
- **Honeycomb。** column-oriented；便于 query。

它们都说 OTLP，即 wire format。你的代码不用关心。

### Propagation across MCP

MCP client 调用 server 时，把 W3C traceparent header 注入 request。Streamable HTTP 支持标准 headers。Stdio 不原生携带 HTTP headers；spec 的 2026 roadmap 正讨论在 JSON-RPC calls 上加入 `_meta.traceparent` 字段。

在它发布前：手动把 traceparent 放进每个 request 的 `_meta`。server 记录 trace id。

### Metrics

除 spans 外，GenAI semconv 还定义 metrics：

- `gen_ai.client.token.usage` —— histogram。
- `gen_ai.client.operation.duration` —— histogram。
- `gen_ai.tool.execution.duration` —— histogram。

用这些构建不需要 per-call detail 的 dashboards。

### AgentOps layer

AgentOps（2024 年成立）专注 GenAI observability。它包装热门 frameworks（LangGraph、Pydantic AI、CrewAI），自动发出 OTel spans。如果你的栈使用受支持 framework，它很有用；否则使用 manual instrumentation。

## 使用它

`code/main.py` 会为一个调用 LLM、dispatch 两个 tools、并做一次 MCP round-trip 的 agent，把 OTel-shaped spans 输出到 stdout（类似 OTLP-JSON 的格式）。没有真实 exporter——本课聚焦 span shape 和 attribute set。把输出粘到 OTLP-compatible viewer 中，或直接阅读它。

重点看：

- Trace id 在所有 spans 中共享。
- Parent-child links 通过 `parentSpanId` 编码。
- Required `gen_ai.*` attributes 已填充。
- content capture 默认关闭；其中一个 scenario 通过 env var 打开。

## 交付它

本课产出 `outputs/skill-otel-genai-instrumentation.md`。给定一个 agent codebase，这个 skill 会生成 instrumentation plan：在哪里加 spans，填充哪些 attributes，以及目标 exporters 是哪些。

## 练习

1. 运行 `code/main.py`。统计 spans，并识别哪个是 CLIENT，哪个是 INTERNAL。

2. 打开 content capture（env var），确认出现 `gen_ai.content.prompt` 和 `gen_ai.content.completion` events。注意 PII implications。

3. 添加 tool-execution metric `gen_ai.tool.execution.duration`，并为每次 call 作为 histogram sample 发出。

4. 从 parent agent span 把 traceparent 传播到 MCP request 的 `_meta.traceparent` 字段。验证 MCP server 会看到同一个 trace id。

5. 阅读 OTel GenAI semconv spec。找出 semconv 中列出、但本课代码没有发出的一个 attribute。添加它。

## 关键词

| Term | 大家常说 | 实际含义 |
|------|----------|----------|
| OTel | “OpenTelemetry” | traces、metrics、logs 的开放标准 |
| GenAI semconv | “GenAI semantic conventions” | LLM / tool / agent spans 的稳定 attribute names |
| `gen_ai.*` | “attribute namespace” | 所有 GenAI attributes 共享这个前缀 |
| Span | “Timed operation” | 带 start、end 和 attributes 的 work unit |
| Trace | “Cross-span ancestry” | 共享 trace id 的 span tree |
| SpanKind | “CLIENT / SERVER / INTERNAL” | 关于 span 方向的 hints |
| OTLP | “OpenTelemetry Line Protocol” | exporters 的 wire format |
| Opt-in content | “Prompt / completion capture” | 默认关闭；通过 env var 启用 |
| traceparent | “W3C header” | 在 services 间传播 trace context |
| Exporter | “Backend-specific shipper” | 把 spans 发送到 Jaeger / Datadog / etc. 的组件 |

## 延伸阅读

- [OpenTelemetry — GenAI semconv](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — GenAI spans、metrics 和 events 的权威 conventions
- [OpenTelemetry — GenAI spans](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-spans/) — LLM 和 tool-execution span attribute list
- [OpenTelemetry — GenAI agent spans](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-agent-spans/) — agent-level `invoke_agent` span
- [open-telemetry/semantic-conventions — GenAI spans](https://github.com/open-telemetry/semantic-conventions/blob/main/docs/gen-ai/gen-ai-spans.md) — GitHub-hosted source of truth
- [Datadog — LLM OTel semantic convention](https://www.datadoghq.com/blog/llm-otel-semantic-convention/) — production integration walkthrough
