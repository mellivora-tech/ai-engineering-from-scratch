# MCP 基础：Primitives、Lifecycle、JSON-RPC Base

> MCP 之前，每个集成都像一次性的。Model Context Protocol 最早由 Anthropic 在 2024 年 11 月发布，现在由 Linux Foundation 的 Agentic AI Foundation 托管，它标准化了 discovery 和 invocation，让任何 client 都能和任何 server 对话。2025-11-25 spec 命名了六个 primitives（三个 server，三个 client）、三阶段 lifecycle，以及 JSON-RPC 2.0 wire format。掌握这些，本阶段后续 MCP 章节就只是阅读。

**类型：** 学习
**语言：** Python（stdlib，JSON-RPC parser）
**前置要求：** 阶段 13 · 01 到 05（工具接口和 function calling）
**时间：** ~45 分钟

## 学习目标

- 说出全部六个 MCP primitives（server 上的 tools、resources、prompts；client 上的 roots、sampling、elicitation），并各给出一个 use case。
- 走通三阶段 lifecycle（initialize、operation、shutdown），并说明每个阶段谁发送什么消息。
- 解析和发出 JSON-RPC 2.0 request、response 和 notification envelope。
- 解释 `initialize` 时 capability negotiation 是什么，以及没有它会坏在哪里。

## 问题

MCP 之前，每个使用工具的 agent 都有自己的协议。Cursor 有一个 MCP-shaped 但不兼容的工具系统。Claude Desktop 带的是另一个。VS Code 的 Copilot extension 是第三个。一个团队构建 “Postgres query” 工具时，要把同一个工具写三次，分别对接不同 host 的 API。复用它就意味着复制代码。

结果是一场一次性集成的寒武纪大爆发，以及生态速度的天花板。

MCP 通过标准化 wire format 修复这个问题。单个 MCP server 可以在每个 MCP client 中工作：Claude Desktop、ChatGPT、Cursor、VS Code、Gemini、Goose、Zed、Windsurf，2026 年 4 月已有 300+ clients。每月 SDK 下载 1.1 亿次。公开 server 10,000+。Linux Foundation 在 2025 年 12 月通过新的 Agentic AI Foundation 接管治理。

本阶段使用的 spec revision 是 **2025-11-25**。它添加了 async Tasks（SEP-1686）、URL-mode elicitation（SEP-1036）、sampling with tools（SEP-1577）、incremental scope consent（SEP-835）和 OAuth 2.1 resource-indicator semantics。阶段 13 · 09 到 16 会覆盖这些扩展。本课停在 base。

## 概念

### 三个 server primitives

1. **Tools。** 可调用动作。和阶段 13 · 01 的四步循环相同。
2. **Resources。** 暴露的数据。通过 URI 寻址的只读内容：`file:///path`、`db://query/...`、自定义 scheme。
3. **Prompts。** 可复用模板。host UI 中的 slash-command；server 提供模板，client 填参数。

### 三个 client primitives

4. **Roots。** server 被允许触及的 URI 集合。client 声明，server 遵守。
5. **Sampling。** server 请求 client 的模型执行 completion。让 server-hosted agent loop 无需 server-side API key。
6. **Elicitation。** server 在中途向 client 的用户请求结构化输入。表单或 URL（SEP-1036）。

MCP 中的每个 capability 都严格属于这六个之一。阶段 13 · 10 到 14 会逐个深入。

### Wire format：JSON-RPC 2.0

每条消息都是带以下字段的 JSON object：

- Requests：`{jsonrpc: "2.0", id, method, params}`。
- Responses：`{jsonrpc: "2.0", id, result | error}`。
- Notifications：`{jsonrpc: "2.0", method, params}` —— 没有 `id`，不期待 response。

base spec 有约 15 个 method，按 primitive 分组。重要的是：

- `initialize` / `initialized`（handshake）
- `tools/list`、`tools/call`
- `resources/list`、`resources/read`、`resources/subscribe`
- `prompts/list`、`prompts/get`
- `sampling/createMessage`（server-to-client）
- `notifications/tools/list_changed`、`notifications/resources/updated`、`notifications/progress`

### 三阶段 lifecycle

**Phase 1：initialize。**

Client 发送带自身 `capabilities` 和 `clientInfo` 的 `initialize`。Server 回复自己的 `capabilities`、`serverInfo` 和它支持的 spec version。Client 在消化 response 后发送 `notifications/initialized`。从这里开始，双方可以按协商后的 capability 发送 request。

**Phase 2：operation。**

双向。Client 调用 `tools/list` 进行 discovery，然后用 `tools/call` invoke。Server 如果声明了 sampling capability，可以发送 `sampling/createMessage`。Server 在 tool set 变化时可以发送 `notifications/tools/list_changed`。用户改变 root scope 时，client 可以发送 `notifications/roots/list_changed`。

**Phase 3：shutdown。**

任一方关闭 transport。MCP 没有结构化 shutdown method；transport（stdio 或 Streamable HTTP，阶段 13 · 09）承载 end-of-connection 信号。

### Capability negotiation

`initialize` handshake 中的 `capabilities` 就是合约。server 示例：

```json
{
  "tools": {"listChanged": true},
  "resources": {"subscribe": true, "listChanged": true},
  "prompts": {"listChanged": true}
}
```

server 声明它能发 `tools/list_changed` notification，并支持 `resources/subscribe`。client 通过声明自己的 capability 来同意：

```json
{
  "roots": {"listChanged": true},
  "sampling": {},
  "elicitation": {}
}
```

如果 client 没声明 `sampling`，server 就不能调用 `sampling/createMessage`。对称地，如果 server 没声明 `resources.subscribe`，client 就不能尝试 subscribe。

这正是防止生态漂移的机制。不支持 sampling 的 client 仍然是合法 MCP client；不调用 `sampling` 的 server 也仍然是合法 MCP server。它们只是不会一起使用那个 feature。

### Structured content 和 error shapes

`tools/call` 返回一个 typed block 的 `content` array：`text`、`image`、`resource`。阶段 13 · 14 会把 MCP Apps（`ui://` interactive UI）加入这个列表。

错误使用 JSON-RPC error codes。spec-defined additions 包括：`-32002` "Resource not found"、`-32603` "Internal error"，以及放在 `error.data` 中的 MCP-specific error data。

### Client capabilities vs tool call details

一个常见混淆：`capabilities.tools` 指的是 client 是否支持 tool-list-changed notifications。client 是否会调用具体工具，是由模型驱动的 runtime choice，不是 capability flag。capability flag 是 spec-level contract。模型选择与之正交。

### 为什么是 JSON-RPC 而不是 REST？

JSON-RPC 2.0（2010）是轻量双向协议。REST 是 client-initiated。MCP 需要 server-initiated messages（sampling、notifications），因此 JSON-RPC 的对称 request/response 形状自然适合。JSON-RPC 也能在 stdio 和 WebSocket/Streamable HTTP 上干净组合，无需重新发明 HTTP 的 request 形状。

## 使用它

`code/main.py` 提供一个最小 JSON-RPC 2.0 parser 和 emitter，然后手动走过 `initialize` → `tools/list` → `tools/call` → `shutdown` 序列，并打印每条消息。没有真实 transport；只有 message shape。和延伸阅读中的 spec 对照，验证每个 envelope。

重点看：

- `initialize` 双向声明 capabilities；response 有 `serverInfo` 和 `protocolVersion: "2025-11-25"`。
- `tools/list` 返回 `tools` array；每个 entry 有 `name`、`description`、`inputSchema`。
- `tools/call` 使用 `params.name` 和 `params.arguments`。
- response `content` 是 `{type, text}` block array。

## 交付它

本课产出 `outputs/skill-mcp-handshake-tracer.md`。给定一个 pcap-style 的 MCP client-server 交互 transcript，这个 skill 会标注每条消息属于哪个 primitive、哪个 lifecycle phase，以及依赖哪个 capability。

## 练习

1. 运行 `code/main.py`。找出 capability negotiation 发生在哪一行，并描述如果 server 没声明 `tools.listChanged` 会改变什么。

2. 扩展 parser 来处理 `notifications/progress`。消息形状：`{method: "notifications/progress", params: {progressToken, progress, total}}`。在长时间运行的 `tools/call` 进行中发出它，并确认 client handler 会显示 progress bar。

3. 从头到尾阅读 MCP 2025-11-25 spec——整份文档约 80 页。找出大多数 server 不需要的一个 capability flag。提示：它和 resource subscription 有关。

4. 在纸上草拟一个假设的 “cron job” feature 应属于哪个 primitive。（提示：server 想让 client 在计划时间调用它。今天六个 primitive 都不适合。）MCP 2026 roadmap 有这个方向的 draft SEP。

5. 解析 GitHub 上某个 open MCP server 的一段 session log。统计 request、response、notification 消息数。计算 traffic 中 lifecycle vs operation 的比例。

## 关键词

| Term | 大家常说 | 实际含义 |
|------|----------|----------|
| MCP | “Model Context Protocol” | 用于 model-to-tool discovery 和 invocation 的开放协议 |
| Server primitive | “server 暴露什么” | tools（动作）、resources（数据）、prompts（模板） |
| Client primitive | “client 允许 server 用什么” | roots（scope）、sampling（LLM callbacks）、elicitation（user input） |
| JSON-RPC 2.0 | “wire format” | 对称的 request/response/notification envelopes |
| `initialize` handshake | “Capability negotiation” | 第一对 message；server 和 client 声明各自支持的 feature |
| `tools/list` | “Discovery” | client 向 server 请求当前工具集 |
| `tools/call` | “Invocation” | client 要求 server 用参数执行工具 |
| `notifications/*_changed` | “Mutation events” | server 告诉 client primitive list 已变化 |
| Content block | “Typed result” | tool result 中的 `{type: "text" | "image" | "resource" | "ui_resource"}` |
| SEP | “Spec Evolution Proposal” | 命名的 draft proposal（例如 async Tasks 的 SEP-1686） |

## 延伸阅读

- [Model Context Protocol — Specification 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25) — 权威 spec 文档
- [Model Context Protocol — Architecture concepts](https://modelcontextprotocol.io/docs/concepts/architecture) — 六 primitive 心智模型
- [Anthropic — Introducing the Model Context Protocol](https://www.anthropic.com/news/model-context-protocol) — 2024 年 11 月发布文章
- [MCP blog — First MCP anniversary](https://blog.modelcontextprotocol.io/posts/2025-11-25-first-mcp-anniversary/) — 一周年回顾和 2025-11-25 spec changes
- [WorkOS — MCP 2025-11-25 spec update](https://workos.com/blog/mcp-2025-11-25-spec-update) — SEP-1686、1036、1577、835 和 1724 摘要
