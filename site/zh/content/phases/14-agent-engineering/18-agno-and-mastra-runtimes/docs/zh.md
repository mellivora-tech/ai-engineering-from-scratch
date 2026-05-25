# Agno 和 Mastra：Production Runtimes

> Agno（Python）和 Mastra（TypeScript）是 2026 年的一组 production-runtime pairing。Agno 目标是 microsecond agent instantiation 和 stateless FastAPI backends。Mastra 在 Vercel AI SDK substrate 上提供 agents、tools、workflows、unified model routing 和 composite storage。

**类型：** 学习
**语言：** Python, TypeScript
**前置要求：** 阶段 14 · 01（Agent Loop），阶段 14 · 13（LangGraph）
**时间：** ~45 分钟

## 学习目标

- 识别 Agno 的 performance targets，以及它们什么时候重要。
- 说出 Mastra 的三个 primitives：Agents、Tools、Workflows，以及支持的 server adapters。
- 解释为什么 stateless session-scoped FastAPI backend 是推荐的 Agno production path。
- 根据给定 stack 选择 Agno vs Mastra（Python-first vs TypeScript-first）。

## 问题

LangGraph、AutoGen、CrewAI 都偏 framework-heavy。想要“just the agent loop, fast, in my runtime”的团队会选择 Agno（Python）或 Mastra（TypeScript）。两者都用一部分 framework-owned primitives 换取原始速度和对周边 stack 的更紧密适配。

## 概念

### Agno

- Python runtime，前身是 Phi-data。
- “No graphs, chains, or convoluted patterns — just pure python.”
- 它们 docs 中的 performance targets：约 2μs agent instantiation、每个 agent 约 3.75 KiB memory、约 23 个 model providers。
- Production path：stateless session-scoped FastAPI backend。每个 request 启动一个 fresh agent；session state 存在 DB 中。
- 原生 multimodal（text、image、audio、video、file）和 agentic RAG。

当你每秒有成千上万个 short-lived agents（chat fan-in、evaluation pipelines）时，速度目标很重要。一个 agent 跑 10 分钟时，它们就没那么重要。

### Mastra

- TypeScript，构建在 Vercel AI SDK 上。
- 三个 primitives：**Agents**、**Tools**（Zod-typed）、**Workflows**。
- Unified Model Router：跨 94 个 providers 的 3,300+ models（March 2026）。
- Composite storage：memory、workflows、observability 可以使用不同 backends；大规模 observability 推荐 ClickHouse。
- Apache 2.0，源码中的 `ee/` directories 使用 source-available enterprise license。
- Server adapters 支持 Express、Hono、Fastify、Koa；first-class Next.js 和 Astro integration。
- 提供 Mastra Studio（localhost:4111）用于 debugging。
- 1.0（Jan 2026）时有 22k+ GitHub stars、300k+ weekly npm downloads。

### Positioning

两者都不试图成为 LangGraph。它们竞争的是：

- **Language fit。** Python-first 团队用 Agno；TypeScript-first 团队用 Mastra。
- **Runtime ergonomics。** Agno = near-zero overhead；Mastra = 与 Vercel ecosystem 集成。
- **Observability。** 两者都集成 Langfuse/Phoenix/Opik（第 24 课），但 Mastra Studio 是 first-party。

### 什么时候选哪一个

- **Agno**：Python backend、很多 short-lived agents、强 perf requirements、FastAPI shop。
- **Mastra**：TypeScript backend、Next.js / Vercel deploy、unified multi-provider model routing、Zod-typed tools。
- **LangGraph**（第 13 课）：当 durable state 和 explicit graph reasoning 比 raw speed 更重要。
- **OpenAI / Claude Agent SDK**：当你想要 provider 的 productized shape（第 16-17 课）。

### 这个模式会在哪里出错

- **Perf-for-perf's-sake。** 只是因为“2μs”听起来好，就在每个 request 只有一次 slow agent call 的 workload 上选择 Agno。Overhead 不是瓶颈。
- **Ecosystem lock-in。** Mastra 的 Vercel-flavored integration 在 Vercel 上是优点，在其他地方可能是缺点。
- **Enterprise license confusion。** Mastra 的 `ee/` directories 是 source-available，不是 Apache 2.0。如果计划 fork，请阅读 licenses。

## 构建它

本课主要是比较性课程。没有一个单一 code artifact 能公正覆盖两个框架。见 `code/main.py` 的 side-by-side toy：一个 minimal “run an agent, stream the output, persist session” flow 被实现了两遍（一遍 Agno-shaped，一遍 Mastra-shaped）。

运行它：

```
python3 code/main.py
```

两条 structurally different 但 functionally equivalent 的 traces。

## 使用它

- **Agno**：需要速度和 FastAPI shape 的 Python backend。
- **Mastra**：带多 providers 和 workflow primitives 的 TypeScript backend。
- 两者都提供 first-party observability hooks。两者都集成 Langfuse。

## 发布它

`outputs/skill-runtime-picker.md` 会根据 stack、latency budget 和 operational shape，在 Agno、Mastra、LangGraph 或 provider SDK 中选择。

## 练习

1. 阅读 Agno docs。把 stdlib ReAct loop（第 01 课）移植到 Agno。什么消失了？什么保留了？
2. 阅读 Mastra docs。把同一个 loop 移植到 Mastra。Tool typing（Zod vs nothing）有什么变化？
3. Benchmark：在你的 stack 上测量 agent instantiation latency。Agno 的 2μs 对你的 workload 重要吗？
4. 设计迁移：如果你一直在 Python 中运行 CrewAI，迁移到 Agno 会坏在哪里？
5. 阅读 Mastra 的 `ee/` license terms。哪些限制会影响 open-source fork？

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|----------------|------------------------|
| Agno | “Fast Python agents” | Stateless session-scoped agent runtime |
| Mastra | “TypeScript agents on Vercel AI SDK” | Agents + Tools + Workflows + Model Router |
| Unified Model Router | “Multi-provider access” | 跨 94 个 providers、3,300+ models 的 single client |
| Composite storage | “Multiple backends” | Memory/workflows/observability 分别使用不同 stores |
| Mastra Studio | “Local debugger” | localhost:4111 UI，用于 introspecting agents |
| Source-available | “Not OSS” | License 允许阅读源码，但限制商业使用 |

## 延伸阅读

- [Agno Agent Framework docs](https://www.agno.com/agent-framework)：performance targets、FastAPI integration
- [Mastra docs](https://mastra.ai/docs)：primitives、server adapters、Model Router
- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview)：stateful-graph alternative
- [Comet Opik](https://www.comet.com/site/products/opik/)：Mastra integrations 引用的 observability comparisons
