# Production Runtimes：Queue、Event、Cron

> Production agents 运行在六种 runtime shapes 上：request-response、streaming、durable execution、queue-based background、event-driven、scheduled。先选 shape，再选 framework。Observability 在每一种 shape 上都是承重结构。

**类型：** 学习
**语言：** Python (stdlib)
**前置要求：** 阶段 14 · 13（LangGraph），阶段 14 · 22（Voice）
**时间：** ~60 分钟

## 学习目标

- 说出六种 production runtime shapes，并把每种匹配到一个 framework / product pattern。
- 解释为什么 durable execution（LangGraph）对 long-horizon tasks 很重要。
- 描述 event-driven runtime，以及 Claude Managed Agents 何时适用。
- 解释 observability-as-load-bearing 对 multi-step agents 的含义。

## 问题

Production agents 的失败方式是 Jupyter notebook 不会暴露的：第 37 步 network timeout，用户在 voice call 中途挂断，cron job 在机器重启时死掉，background worker 内存耗尽。Runtime shape 决定哪些 failures 可恢复。

## 概念

### Request-response

- Synchronous HTTP。用户等待完成。
- 只适合短任务（<30s）。
- Stacks：Agno（Python + FastAPI）、Mastra（TypeScript + Express/Hono/Fastify/Koa）。
- Observability：standard HTTP access logs + OTel spans。

### Streaming

- SSE 或 WebSocket，用于 progressive output。
- LiveKit 把这个扩展到 voice/video 的 WebRTC（第 22 课）。
- Stacks：任何支持 streaming 的 framework + 能处理 SSE/WS 的 frontend。
- Observability：per-chunk timing、first-token latency、tail latency。

### Durable execution

- 每一步之后 checkpoint state；失败后 auto-resumes。
- AutoGen v0.4 actor model 把 failures 隔离到单个 agent（第 14 课）。
- LangGraph 的核心差异点（第 13 课）。
- 当 step count 未知且 recovery cost 高时必不可少。

### Queue-based / background

- Job 进入 queue，workers pick up，结果通过 webhooks 或 pub/sub 回流。
- 对 long-horizon agents 必不可少（Anthropic computer use announcement 中每个 task 可达 dozens-to-hundreds of steps）。
- Stacks：Celery（Python）、BullMQ（Node）、SQS + Lambda（AWS）、custom。
- Observability：queue depth、per-job latency distribution、DLQ size。

### Event-driven

- Agents 订阅 triggers：new email、PR opened、cron fire。
- Claude Managed Agents 开箱覆盖这一点（第 17 课）。
- CrewAI Flows（第 15 课）组织 event-driven deterministic workflows。
- Observability：trigger source、event-to-start latency、agent latency。

### Scheduled

- 周期性运行的 cron-shaped agents。
- 与 durable execution 组合，让失败的 nightly run 在下一 tick resume。
- Stacks：Kubernetes CronJob + durable framework；hosted（Render cron、Vercel cron）。

### 2026 deployment patterns

- **CrewAI Flows** 用于 event-driven production。
- **Agno** stateless FastAPI，用于 Python microservices。
- **Mastra** server adapters（Express、Hono、Fastify、Koa）用于 embedding。
- **Pipecat Cloud / LiveKit Cloud** 用于 managed voice（第 22 课）。
- **Claude Managed Agents** 用于 hosted long-running async。

### Observability is load-bearing

没有 OpenTelemetry GenAI spans（第 23 课）加 Langfuse/Phoenix/Opik backend（第 24 课），你无法 debug 一个在第 40 步失败的 multi-step agent。这对 production 不是可选项。区别在于：“we debug fast” 还是 “we replay from scratch with more logging”。

### Production runtimes 会在哪里失败

- **Wrong shape choice。** 为 5 分钟任务选择 request-response。用户挂断；workers 堆积；retries 复合。
- **No DLQ。** Queue workers 没有 dead-letter。Failed jobs 消失。
- **Opaque background work。** Background agent 运行但不 export trace。直到用户报告，失败都不可见。
- **Skipping durable state。** 任何 > 30 秒且不能承受从头重启的 run，都需要 durable execution。

## 构建它

`code/main.py` 是一个 stdlib multi-shape demo：

- Request-response endpoint（plain function）。
- Streaming handler（generator）。
- 带 DLQ 的 queue-based worker。
- Event trigger registry。
- Cron-shaped scheduler。

运行它：

```bash
python3 code/main.py
```

输出：五条 traces，展示同一 task 在不同 shape 下的行为。同一个 agent logic，不同 outer shells。Durable execution（第六种 shape）有意放在第 13 课 LangGraph checkpointing 中覆盖。

## 使用它

- **Request-response** 用于 chat-style UX。
- **Streaming** 用于 progressive responses。
- **Durable** 用于 long-horizon tasks。
- **Queue** 用于 batch / async / long-running。
- **Event** 用于 agent reactivity。
- **Cron** 用于 housekeeping（memory consolidation、evals、cost reports）。

## 发布它

`outputs/skill-runtime-shape.md` 会为一个 task 选择 runtime shape，并接好 observability requirements。

## 练习

1. 把第 01 课 ReAct loop 移植到你栈中的六种 shapes。哪种 shape 适合哪个 product surface？
2. 给 queue-based demo 添加 DLQ。模拟 10% job failure；暴露 DLQ size。
3. 写一个 nightly 运行的 cron-triggered eval agent，对当天 top 20 traces 做 eval。
4. 实现带 backpressure 的 streaming：如果 client 慢，就 pause agent。这和 turn budget 如何交互？
5. 阅读 Claude Managed Agents docs。什么时候你会把 self-hosted long-horizon agent 迁到 managed？

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|----------------|------------------------|
| Request-response | "Synchronous" | 用户等待；只适合短任务 |
| Streaming | "SSE / WS" | Progressive output；更好的 UX；per chunk 可观测 latency |
| Durable execution | "Resume from failure" | Checkpointed state；从最后一步 restart |
| Queue-based | "Background jobs" | Producer / worker pool / DLQ |
| Event-driven | "Trigger-based" | Agent 响应 external events |
| DLQ | "Dead-letter queue" | Failed jobs 的停车场 |
| Claude Managed Agents | "Hosted harness" | Anthropic-hosted long-running async，带 caching + compaction |

## 延伸阅读

- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview) — durable execution details
- [Claude Managed Agents overview](https://platform.claude.com/docs/en/managed-agents/overview) — hosted long-running async
- [Anthropic, Introducing computer use](https://www.anthropic.com/news/3-5-models-and-computer-use) — “dozens-to-hundreds of steps per task”
- [AutoGen v0.4 (Microsoft Research)](https://www.microsoft.com/en-us/research/articles/autogen-v0-4-reimagining-the-foundation-of-agentic-ai-for-scale-extensibility-and-robustness/) — actor-model fault isolation
