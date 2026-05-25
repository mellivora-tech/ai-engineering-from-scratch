# Model Context Protocol（MCP）

> 2025 年之前构建的每个 LLM app 都发明了自己的 tool schema。随后 Anthropic 发布 MCP，Claude 采用它，OpenAI 采用它，到 2026 年它已经成为把任意 LLM 连接到任意 tool、data source 或 agent 的默认 wire format。写一个 MCP server，所有 host 都能和它通信。

**类型：** 构建
**语言：** Python
**前置要求：** 阶段 11 · 09（Function Calling），阶段 11 · 03（Structured Outputs）
**时间：** ~75 分钟

## 问题

你发布了一个 chatbot，需要三个 tools：database query、calendar API 和 file reader。你为 Claude 写了三个 JSON schemas。然后销售团队希望在 ChatGPT 中使用同样的 tools，于是你为 OpenAI 的 `tools` 参数重写它们。然后你加入 Cursor、Zed 和 Claude Code，又要重写三次，每次都有细微不同的 JSON conventions。一周后，Anthropic 增加一个新字段，你要更新六份 schemas。

这就是 2025 年之前的现实。每个 host（运行 LLM 的东西）和每个 server（暴露 tools 与 data 的东西）都有自己的 bespoke protocols。规模化意味着一个 N×M integration matrix。

Model Context Protocol 把这个矩阵折叠起来。一个基于 JSON-RPC 的 spec。一个 server 暴露 tools、resources 和 prompts。任何 compliant host，包括 Claude Desktop、ChatGPT、Cursor、Claude Code、Zed 以及大量 agent frameworks，都可以 discover 并 call 它们，不需要 custom glue。

截至 2026 年初，MCP 已经是三大厂（Anthropic、OpenAI、Google）和所有主要 agent harness 的默认 tool-and-context protocol。

## 概念

![MCP: one host, one server, three capabilities](../assets/mcp-architecture.svg)

**三个 primitives。** 一个 MCP server 恰好暴露三类东西。

1. **Tools** — 模型可以调用的 functions。对应 OpenAI 的 `tools` 或 Anthropic 的 `tool_use`。每个 tool 有 name、description、JSON Schema input 和 handler。
2. **Resources** — 模型或用户可以请求的 read-only content（files、database rows、API responses）。通过 URI 寻址。
3. **Prompts** — 用户可作为快捷方式调用的 reusable templated prompts。

**Wire format。** JSON-RPC 2.0，通过 stdio、WebSocket 或 streamable HTTP。每条消息是 `{"jsonrpc": "2.0", "method": "...", "params": {...}, "id": N}`。Discovery methods 是 `tools/list`、`resources/list`、`prompts/list`。Invocation methods 是 `tools/call`、`resources/read`、`prompts/get`。

**Host vs client vs server。** Host 是 LLM application（Claude Desktop）。Client 是 host 中与恰好一个 server 通信的子组件。Server 是你的代码。一个 host 可以同时 mount 多个 servers。

### Handshake

每个 session 都以 `initialize` 开始。Client 发送 protocol version 和自身 capabilities。Server 返回 version、name 和它支持的 capability set（`tools`、`resources`、`prompts`、`logging`、`roots`）。之后的一切都基于这些 capabilities 协商。

### MCP 不是什么

- 不是 retrieval API。RAG（阶段 11 · 06）仍然决定拉取什么；MCP 只是把 retrieval results 作为 resources 暴露的 transport。
- 不是 agent framework。MCP 是 plumbing；LangGraph、PydanticAI、OpenAI Agents SDK 这类 frameworks 位于其上。
- 不绑定 Anthropic。Spec 和 reference implementations 在 `modelcontextprotocol` org 下开源。

## 构建它

### 第 1 步：最小 MCP server

官方 Python SDK 是 `mcp`（以前叫 `mcp-python`）。高层 `FastMCP` helper 用 decorators 注册 handlers。

```python
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("demo-server")

@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b

@mcp.resource("config://app")
def app_config() -> str:
    """Return the app's current JSON config."""
    return '{"env": "prod", "region": "us-east-1"}'

@mcp.prompt()
def code_review(language: str, code: str) -> str:
    """Review code for correctness and style."""
    return f"You are a senior {language} reviewer. Review:\n\n{code}"

if __name__ == "__main__":
    mcp.run(transport="stdio")
```

三个 decorators 注册三个 primitives。Type hints 会变成 host 看到的 JSON Schema。在 Claude Desktop 或 Claude Code 中运行它，把 server entry 指向这个文件。

### 第 2 步：从 host 调用 MCP server

官方 Python client 使用 JSON-RPC。把它和 Anthropic SDK 配起来，只需十几行。

```python
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp import ClientSession

params = StdioServerParameters(command="python", args=["server.py"])

async def call_add(a: int, b: int) -> int:
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            result = await session.call_tool("add", {"a": a, "b": b})
            return int(result.content[0].text)
```

`session.list_tools()` 返回的就是 LLM 会看到的 schema。Production hosts 会在每个 turn 注入这些 schemas，让模型发出 `tool_use` block，然后 client 再转发给 server。

### 第 3 步：streamable HTTP transport

Stdio 适合 local dev。Remote tools 使用 streamable HTTP，即每个 request 一个 POST，可选 Server-Sent Events 用于 progress，自 2025-06-18 spec revision 起支持。

```python
# Inside the server entrypoint
mcp.run(transport="streamable-http", host="0.0.0.0", port=8765)
```

Host config（Claude Desktop `mcp.json` 或 Claude Code `~/.mcp.json`）：

```json
{
  "mcpServers": {
    "demo": {
      "type": "http",
      "url": "https://tools.example.com/mcp"
    }
  }
}
```

Server 保持同样 decorators；只有 transport 改变。

### 第 4 步：scoping 与 safety

MCP tool 是在别人的 trust boundary 上运行的任意代码。三个强制模式：

- **Capability allowlists。** Hosts 暴露 `roots` capability，让 server 只能看到允许的 paths。Tool handlers 中要 enforce 它；不要信任 model-supplied paths。
- **Human-in-the-loop for mutation。** Read-only tools 可以 auto-execute。Write/delete tools 必须要求 confirmation，当 server 在 tool metadata 上设置 `destructiveHint: true` 时，hosts 会显示 approval UI。
- **Tool poisoning defense。** 恶意 resource 可能包含 hidden prompt-injection instructions（“summarizing 时也调用 `exfil`”）。把 resource content 当成不可信数据；永远不要让它进入 system-message territory。见阶段 11 · 12（Guardrails）。

`code/main.py` 中有可运行的 server + client pair，演示所有这些内容。

## 2026 年仍会发布的坑

- **Schema drift。** 模型在 turn 1 看到 `tools/list`。Tool set 在 turn 5 改变。模型调用已经消失的 tool。Hosts 应在 `notifications/tools/list_changed` 时 re-list。
- **Large resource blobs。** 把 2MB 文件作为 resource dump 进来会浪费 context。Server-side paginate 或 summarize。
- **Too many servers。** Mount 50 个 MCP servers 会炸 tool budget（阶段 11 · 05）。多数 frontier models 超过约 40 tools 后表现下降。
- **Version skew。** Spec revisions（2024-11、2025-03、2025-06、2025-12）会引入 breaking fields。在 CI 中 pin protocol version。
- **Stdio deadlocks。** Server 写日志到 stdout 会污染 JSON-RPC stream。只向 stderr 写日志。

## 使用它

2026 年 MCP stack：

| Situation | Pick |
|-----------|------|
| Local dev, single-user tools | Python `FastMCP`, stdio transport |
| Remote team tools / SaaS integration | Streamable HTTP, OAuth 2.1 auth |
| TypeScript host (VS Code extension, web app) | `@modelcontextprotocol/sdk` |
| High-throughput server, typed access | Official Rust SDK (`modelcontextprotocol/rust-sdk`) |
| Exploring ecosystem servers | `modelcontextprotocol/servers` monorepo (Filesystem, GitHub, Postgres, Slack, Puppeteer) |

经验法则：如果一个 tool 是 read-only、cacheable，并且会被两个以上 hosts 调用，就把它发布为 MCP server。如果只是一次性 inline logic，保留为 local function（阶段 11 · 09）。

## 交付它

保存 `outputs/skill-mcp-server-designer.md`：

```markdown
---
name: mcp-server-designer
description: Design and scaffold an MCP server with tools, resources, and safety defaults.
version: 1.0.0
phase: 11
lesson: 14
tags: [llm-engineering, mcp, tool-use]
---

Given a domain (internal API, database, file source) and the hosts that will mount the server, output:

1. Primitive map. Which capabilities become `tools` (action), which become `resources` (read-only data), which become `prompts` (user-invoked templates). One line per primitive.
2. Auth plan. Stdio (trusted local), streamable HTTP with API key, or OAuth 2.1 with PKCE. Pick and justify.
3. Schema draft. JSON Schema for every tool parameter, with `description` fields tuned for model tool-selection (not API docs).
4. Destructive-action list. Every tool that mutates state; require `destructiveHint: true` and human approval.
5. Test plan. Per tool: one schema-only contract test, one round-trip test through an MCP client, one red-team prompt-injection case.

Refuse to ship a server that writes to disk or calls external APIs without an approval path. Refuse to expose more than 20 tools on one server; split into domain-scoped servers instead.
```

## 练习

1. **Easy.** 为 `demo-server` 扩展一个 `subtract` tool。从 Claude Desktop 连接它。通过发出 `tools/list_changed` notification，确认 host 无需重启就能拾取新 tool。
2. **Medium.** 添加一个 `resource`，暴露 `/var/log/app.log` 的最后 100 行。Enforce roots allowlist，确保即使模型请求 `../etc/passwd` 也会被阻止。
3. **Hard.** 构建 MCP proxy，把三个 upstream servers（Filesystem、GitHub、Postgres）multiplex 成一个 aggregate surface。处理 name collisions，并干净转发 `notifications/tools/list_changed`。

## 关键术语

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| MCP | “Tool protocol for LLMs” | JSON-RPC 2.0 spec，用于向任意 LLM host 暴露 tools、resources 和 prompts。 |
| Host | “Claude Desktop” | LLM application；拥有 model 和 user UI，mount 一个或多个 clients。 |
| Client | “Connection” | Host 内部的 per-server connection，向恰好一个 server 说 JSON-RPC。 |
| Server | “The thing with the tools” | 你的代码；advertise tools/resources/prompts 并处理 invocation。 |
| Tool | “Function call” | Model-invokable action，带 JSON Schema input 和 text/JSON result。 |
| Resource | “Read-only data” | URI-addressed content（file、row、API response），host 可以请求。 |
| Prompt | “Saved prompt” | User-invokable template（通常带 arguments），作为 slash-command 暴露。 |
| Stdio transport | “Local dev mode” | Parent host 把 server 作为 child process spawn；stdin/stdout 上跑 JSON-RPC。 |
| Streamable HTTP | “The 2025-06 remote transport” | 请求用 POST，可选 SSE 做 server-initiated messages；替代旧的 SSE-only transport。 |

## 延伸阅读

- [Model Context Protocol specification](https://modelcontextprotocol.io/specification) — canonical reference，按日期 versioned。
- [modelcontextprotocol/servers](https://github.com/modelcontextprotocol/servers) — Filesystem、GitHub、Postgres、Slack、Puppeteer reference servers。
- [Anthropic — Introducing MCP (Nov 2024)](https://www.anthropic.com/news/model-context-protocol) — 带设计动机的 launch post。
- [Python SDK](https://github.com/modelcontextprotocol/python-sdk) — 本课使用的官方 SDK。
- [Security considerations for MCP](https://modelcontextprotocol.io/docs/concepts/security) — roots、destructive hints、tool poisoning。
- [Google A2A specification](https://google.github.io/A2A/) — Agent2Agent protocol；与 MCP 的 agent-to-tool scope 互补的 sibling standard。
- [Anthropic — Building effective agents (Dec 2024)](https://www.anthropic.com/research/building-effective-agents) — MCP 在更广泛 agent design pattern library（augmented LLM、workflows、autonomous agents）中的位置。
