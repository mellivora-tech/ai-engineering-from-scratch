# A2A — Agent-to-Agent Protocol

> Google 在 2025 年 4 月宣布 A2A；到 2026 年 4 月，spec 位于 https://a2a-protocol.org/latest/specification/，并有 150+ organizations 支持。A2A 是 MCP（第 13 课）的水平补充：MCP 是 vertical（agent ↔ tools），A2A 是 peer-to-peer（agent ↔ agent）。它定义 Agent Cards（discovery）、带 artifacts 的 tasks（text、structured data、video）、opaque task lifecycles 和 auth。Production systems 越来越多地把 MCP 与 A2A 配对。Google Cloud 在 2025-2026 年把 A2A support 纳入 Vertex AI Agent Builder。

**类型：** 学习 + 构建
**语言：** Python（stdlib，`http.server`，`json`）
**前置要求：** 阶段 16 · 04（Primitive Model）
**时间：** ~75 分钟

## 问题

你的 agent 需要调用另一个系统上的另一个 agent。怎么做？你可以暴露 HTTP endpoint，定义 bespoke JSON schema，然后希望对方说同一种格式。每一对 agents 都变成 custom integration。

A2A 是这次调用的 universal wire protocol。标准 discovery、标准 task model、标准 transport、标准 artifacts。就像 HTTP+REST，但把 agents 作为 first-class citizens。

## 概念

### 四个 elements

**Agent Card。** 位于 `/.well-known/agent.json` 的 JSON document，描述 agent：name、skills、endpoints、supported modalities、auth requirements。Discovery 通过读取 card 完成。

```
GET https://agent.example.com/.well-known/agent.json
→ {
    "name": "code-review-agent",
    "skills": ["review-python", "review-typescript"],
    "endpoints": {
      "tasks": "https://agent.example.com/tasks"
    },
    "auth": {"type": "bearer"},
    "modalities": ["text", "structured"]
  }
```

**Task。** work unit。一个 async、stateful object，带 lifecycle：`submitted → working → completed / failed / canceled`。client 发送 task，poll 或 subscribe updates。

**Artifact。** task 产生的 result type。Text、structured JSON、image、video、audio。Artifacts 是 typed 的，所以不同 modalities 是 first-class。

**Opaque lifecycle。** A2A 不规定 remote agent *如何* 解决 task。client 只看到 state transitions 和 artifacts；implementation 可以使用任意 framework。

### MCP/A2A 分工

- **MCP**（第 13 课）：agent ↔ tool。agent 通过 JSON-RPC 读写 tool server。默认 stateless。
- **A2A**：agent ↔ agent。Peer protocol；双方都是有自己 reasoning 的 agents。

Production multi-agent systems 两者都用。一个 A2A peer 会在自己那边调用 MCP tools。这个分工保持两个 concerns 清晰。

### Discovery flow

```
Client                     Agent server
  ├──GET /.well-known/agent.json──>
  <──Agent Card JSON─────────────
  ├──POST /tasks {skill, input}──>
  <──201 task_id, state=submitted
  ├──GET /tasks/{id}──────────────>
  <──state=working, 42% done──────
  ├──GET /tasks/{id}──────────────>
  <──state=completed, artifacts──
```

或者 streaming：SSE subscription 到 `/tasks/{id}/events` 获取 push updates。

### Auth

A2A 支持三种常见模式：

- **Bearer token** — OAuth2 或 opaque。
- **mTLS** — mutual TLS；organizations 互相证明身份。
- **Signed requests** — payload 上的 HMAC。

Auth 在 Agent Card 中声明；clients discover 并遵守。

### 2026 年 4 月已有 150+ organizations

Enterprise adoption 推动了 A2A 规模。headline：A2A 成为 enterprise agent systems 跨 trust boundaries 的方式。Google Cloud 发布了 Vertex AI Agent Builder A2A support；Microsoft Agent Framework 支持它；大多数主流 frameworks（LangGraph、CrewAI、AutoGen）都有 A2A adapters。

### A2A 的优势

- **Cross-organization calls。** company A 的 agent 调用 company B 的 agent。没有 A2A，每一对都是 bespoke contract。
- **Heterogeneous frameworks。** LangGraph agent 调 CrewAI agent，再调 custom Python agent。A2A 做 normalizing。
- **Typed artifacts。** Video result、structured JSON、audio — 都是 first-class。
- **Long-running tasks。** Opaque lifecycle + polling 让数小时任务很直接。

### A2A 的不足

- **Latency-sensitive micro-calls。** A2A lifecycle 是 async。亚毫秒 agent-to-agent 不合适；用 direct RPC。
- **Tight-coupled in-process agents。** 如果两个 agents 跑在同一个 Python process，A2A 的 HTTP round-trip 是 overkill。
- **Small teams。** Spec overhead 是真实成本；internal-only agents 可能不需要这种正式性。

### A2A vs ACP、ANP、NLIP

2024-2026 年出现了几个相关 specs：

- **ACP**（IBM/Linux Foundation）— A2A 的前身，scope 更窄。
- **ANP**（Agent Network Protocol）— 强调 peer discovery、decentralized-first。
- **NLIP**（Ecma Natural Language Interaction Protocol，2025 年 12 月标准化）— natural-language content type。

截至 2026 年 4 月，A2A 是采用最广泛的 peer protocol。比较见 arXiv:2505.02279（Liu et al., “A Survey of Agent Interoperability Protocols”）。

## 构建它

`code/main.py` 使用 `http.server` 和 JSON 实现了一个 A2A-minimal server 和 client。server：

- 暴露 `/.well-known/agent.json`，
- 接受 `POST /tasks`，
- 管理 task state，
- 在 `GET /tasks/{id}` 返回 artifacts。

client：

- 获取 Agent Card，
- 提交 task，
- poll 直到 completion，
- 读取 artifact。

运行：

```
python3 code/main.py
```

脚本在 background thread 中启动 server，然后运行 client。你会看到完整 flow：discovery、submit、poll、artifact。

## 使用它

`outputs/skill-a2a-integrator.md` 设计 A2A integration：Agent Card contents、task schemas、auth choice、streaming vs polling。

## 发布它

Checklist：

- **固定 spec version。** A2A 仍在演进；Agent Card 应声明 protocol version。
- **Idempotent task creation。** Duplicate submissions（network retries）应该产生一个 task。
- **Artifact schemas。** 声明 agent 返回的 shapes；consumers 应 validate。
- **Rate limits + auth。** A2A 是 public-facing；应用标准 Web security。
- **Failed tasks 的 dead-letter。** 长期观察 recurring failure types。

## 练习

1. 运行 `code/main.py`。确认 client discover server 并收到正确 artifact。
2. 给 server 添加第二个 skill（例如 “summarize”）。更新 Agent Card。写一个根据 task type 选择 skill 的 client。
3. 实现 SSE streaming endpoint：`/tasks/{id}/events`，emit state changes。client 需要做什么不同的事情？
4. 阅读 A2A spec（https://a2a-protocol.org/latest/specification/）。识别 spec 要求但这个 demo 未实现的三件事。
5. 比较 A2A（Agent Card discovery）和 MCP（通过 `listTools` 做 server-side capability listing）。self-describing agents 与 capability-probing 之间的 tradeoff 是什么？

## 关键词汇

| 术语 | 人们常说 | 实际含义 |
|------|----------------|------------------------|
| A2A | “Agent-to-agent” | agents 跨系统调用其他 agents 的 peer protocol。Google 2025。 |
| Agent Card | “agent 的名片” | `/.well-known/agent.json` 上的 JSON，描述 skills、endpoints、auth。 |
| Task | “work unit” | 带 lifecycle 的 async stateful object；completion 时产生 artifacts。 |
| Artifact | “结果” | Typed output：text、structured JSON、image、video、audio。first-class media。 |
| Opaque lifecycle | “怎么解决是 agent 自己的事” | client 看到 state transitions；server 可自由选择 framework/tools。 |
| Discovery | “找到 agent” | `GET /.well-known/agent.json` 返回 card。 |
| MCP vs A2A | “Tools vs peers” | MCP：vertical agent ↔ tool。A2A：horizontal agent ↔ agent。 |
| ACP / ANP / NLIP | “Sibling protocols” | 相邻 specs；A2A 是 2026 年采用最广的。 |

## 延伸阅读

- [A2A specification](https://a2a-protocol.org/latest/specification/) — canonical spec
- [Google Developers Blog — A2A announcement](https://developers.googleblog.com/en/a2a-a-new-era-of-agent-interoperability/) — 2025 年 4 月 launch post
- [A2A GitHub repo](https://github.com/a2aproject/A2A) — reference implementations and SDKs
- [Liu et al. — A Survey of Agent Interoperability Protocols](https://arxiv.org/html/2505.02279v1) — MCP、ACP、A2A、ANP comparison
