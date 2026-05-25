# AutoGen v0.4：Actor Model 和 Agent Framework

> AutoGen v0.4（Microsoft Research, Jan 2025）围绕 actor model 重设计了 agent orchestration。Async message exchange、event-driven agents、fault isolation、natural concurrency。现在该框架处于 maintenance mode，而 Microsoft Agent Framework（public preview Oct 2025）正在成为继任者。

**类型：** 学习 + 构建
**语言：** Python (stdlib)
**前置要求：** 阶段 14 · 01（Agent Loop），阶段 14 · 12（Workflow Patterns）
**时间：** ~75 分钟

## 学习目标

- 描述 actor model：agents as actors、messages as the only IPC、每个 actor failure isolation。
- 说出 AutoGen v0.4 的三层 API：Core、AgentChat、Extensions，以及每层用途。
- 解释为什么把 message delivery 和 handling 解耦会带来 fault isolation 和 natural concurrency。
- 在 Python 中实现 stdlib actor runtime，并把 two-agent code-review flow 移植到其上。

## 问题

大多数 agent frameworks 是同步的：一个 agent 生产，一个 agent 消费，位于同一个 call stack。Failures 会 crash 整个 stack。Concurrency 是事后补上的。Distribution 需要重写。

AutoGen v0.4 的答案是 actor model。每个 agent 都是带 private inbox 的 actor。Messages 是唯一交互方式。Runtime 把 delivery 和 handling 解耦。Failures 隔离到单个 actor。Concurrency 原生存在。Distribution 只是不同 transport。

## 概念

### Actors

一个 actor 有：

- 私有 state（外部永远不能直接触碰）。
- Inbox（message queue）。
- Handler：`receive(message) -> effects`，其中 effects 可以是“reply”、“send to other actor”、“spawn new actor”、“update state”、“stop self”。

两个 actors 不能共享 memory。它们只能发送 messages。

### AutoGen v0.4 的三层 API

1. **Core。** Low-level actor framework。`AgentRuntime`、`Agent`、`Message`、`Topic`。Async message exchange，event-driven。
2. **AgentChat。** Task-driven high-level API（替代 v0.2 的 ConversableAgent）。`AssistantAgent`、`UserProxyAgent`、`RoundRobinGroupChat`、`SelectorGroupChat`。
3. **Extensions。** Integrations：OpenAI、Anthropic、Azure、tools、memory。

### 为什么 decoupling 很重要

在 v0.2 模型中，同步调用 `agent_a.chat(agent_b)` 会阻塞 agent_a，直到 agent_b 返回。在 v0.4 中，`send(agent_b, msg)` 把 message 放入 agent_b 的 inbox，然后返回。Runtime 稍后 deliver。三个结果：

- **Fault isolation。** Agent B crash 不会 crash Agent A；runtime 会捕获 B handler 中的 failure，并决定怎么做（log、retry、dead-letter）。
- **Natural concurrency。** 同时有许多 in-flight messages；actors 并发处理自己的 inbox。
- **Distribution-ready。** 无论 actor 在进程内还是另一台 host 上，Inbox + transport 都是同一个抽象。

### Topologies

- **RoundRobinGroupChat。** Agents 按固定轮次依次发言。
- **SelectorGroupChat。** Selector agent 根据 conversation context 选择下一个是谁。
- **Magentic-One。** 用于 web browsing、code execution、file handling 的参考 multi-agent team。基于 AgentChat 构建。

### Observability

内置 OpenTelemetry 支持。每条 message 都发出一个 span；tool calls 根据 2026 OTel GenAI semantic conventions（第 23 课）携带 `gen_ai.*` attributes。

### 状态：maintenance mode

2026 年初：AutoGen v0.7.x 对 research 和 prototyping 来说稳定。Microsoft 已经把 active development 转向 Microsoft Agent Framework（public preview Oct 1 2025；1.0 GA 目标是 2026 年 Q1 结束）。AutoGen patterns 可以干净地向前移植，actor model 才是持久想法。

## 构建它

`code/main.py` 实现了一个 stdlib actor runtime：

- `Message`：带 `sender`、`recipient`、`topic`、`body` 的 typed payload。
- `Actor`：abstract，带 `receive(message, runtime)`。
- `Runtime`：event loop，包含 shared queue、delivery、failure isolation。
- Two-actor demo：`ReviewerAgent` review code，`ChecklistAgent` 运行 checklist；它们交换 messages 直到达成 consensus。

运行它：

```
python3 code/main.py
```

Trace 会显示 message delivery、一个 actor 中的 simulated failure 不会 crash 另一个 actor，以及最终收敛到 shared verdict。

## 使用它

- **AutoGen v0.4/v0.7**（maintenance）：适合 research、prototyping、multi-agent patterns。
- **Microsoft Agent Framework**（public preview）：向前路径；同样 actor-model ideas，使用 refreshed API。
- **LangGraph swarm topology**（第 13 课）：通过 shared-tool handoffs 实现类似模式。
- **Custom actor runtime**：当你需要特定 transport（NATS、RabbitMQ、gRPC）时。

## 发布它

`outputs/skill-actor-runtime.md` 会为给定 multi-agent task 生成一个 minimal actor runtime 和 team template（RoundRobin 或 Selector）。

## 练习

1. 添加 dead-letter queue：当 handler 抛错时，把 failing message 暂存给 human inspection。你的 toy 中 DLQ 多久会命中一次？
2. 实现 `SelectorGroupChat`：selector actor 根据 conversation state 选择谁处理下一条 message。
3. 添加 distributed transport：把 in-process queue 换成 JSON-over-HTTP server，让 actors 可以在不同 processes 中运行。
4. 为每条 message 接一个 OTel span（或 no-op stand-in）。按第 23 课发出 `gen_ai.agent.name`、`gen_ai.operation.name`。
5. 阅读 AutoGen v0.4 的 architecture post。把 toy 移植到真实 `autogen_core` API。你跳过了哪些生产中重要的东西？

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|----------------|------------------------|
| Actor | “Agent” | Private state + inbox + handler；无 shared memory |
| Message | “Event” | Typed payload；actors 交互的唯一方式 |
| Inbox | “Mailbox” | 每个 actor 的 pending messages queue |
| Runtime | “Agent host” | 路由 messages 并隔离 failures 的 event loop |
| Topic | “Channel” | Actors 之间的 named publish-subscribe route |
| Fault isolation | “Let it crash” | 一个 actor 失败不会 crash 其他 actors |
| RoundRobinGroupChat | “Fixed-rotation team” | Agents 按顺序轮流发言 |
| SelectorGroupChat | “Context-routed team” | Selector 选择下一个是谁 |
| Magentic-One | “Reference team” | 面向 web + code + files 的 multi-agent squad |

## 延伸阅读

- [AutoGen v0.4, Microsoft Research](https://www.microsoft.com/en-us/research/articles/autogen-v0-4-reimagining-the-foundation-of-agentic-ai-for-scale-extensibility-and-robustness/)：redesign post
- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview)：graph-shaped alternative
- [OpenTelemetry GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/)：AutoGen 默认发出的 spans
