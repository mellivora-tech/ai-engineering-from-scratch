# 构建 MCP Server：Python + TypeScript SDKs

> 大多数 MCP 教程只展示 stdio hello-world。真正的 server 会暴露 tools、resources 和 prompts，处理 capability negotiation，发出结构化错误，并且在不同 SDK 之间行为一致。本课会端到端构建一个 notes server：stdlib stdio transport、JSON-RPC dispatch、三个 server primitives，以及一种纯函数风格，让你成熟后可以直接迁移到 Python SDK 的 FastMCP 或 TypeScript SDK。

**类型：** 构建
**语言：** Python（stdlib，stdio MCP server）
**前置要求：** 阶段 13 · 06（MCP fundamentals）
**时间：** ~75 分钟

## 学习目标

- 实现 `initialize`、`tools/list`、`tools/call`、`resources/list`、`resources/read`、`prompts/list` 和 `prompts/get` methods。
- 编写一个 dispatch loop，从 stdin 读取 JSON-RPC messages，并向 stdout 写 responses。
- 按 JSON-RPC 2.0 spec 和 MCP 的附加 code 发出 structured error responses。
- 在不重写 tool logic 的情况下，把 stdlib 实现迁移到 FastMCP（Python SDK）或 TypeScript SDK。

## 问题

在使用 remote transport（阶段 13 · 09）或 auth layer（阶段 13 · 16）之前，你需要一个干净的 local server。local 意味着 stdio：server 作为 client 的 child process 被启动，messages 通过 stdin/stdout newline-delimited 传输。

2025-11-25 spec 规定 stdio messages 编码为 JSON object，并用显式 `\n` 分隔。这里没有 SSE；SSE 是旧 remote mode，并会在 2026 年中移除（Atlassian 的 Rovo MCP server 在 2026 年 6 月 30 日弃用它；Keboola 在 2026 年 4 月 1 日弃用）。对 stdio 来说，每行一个 JSON object 就是整个 wire format。

notes server 是一个好形状，因为它会练到三个 server primitives。Tools 做 mutation（`notes_create`）。Resources 暴露数据（`notes://{id}`）。Prompts 提供模板（`review_note`）。本课的形状可泛化到任何 domain。

## 概念

### Dispatch loop

```
loop:
  line = stdin.readline()
  msg = json.loads(line)
  if has id:
    handle request -> write response
  else:
    handle notification -> no response
```

三条规则：

- 不要向 stdout 打印任何非 JSON-RPC envelope 的内容。debug logs 去 stderr。
- 每个 request 必须匹配一个带相同 `id` 的 response。
- Notifications 绝不能被 response。

### 实现 `initialize`

```python
def initialize(params):
    return {
        "protocolVersion": "2025-11-25",
        "capabilities": {
            "tools": {"listChanged": True},
            "resources": {"listChanged": True, "subscribe": False},
            "prompts": {"listChanged": False},
        },
        "serverInfo": {"name": "notes", "version": "1.0.0"},
    }
```

只声明你支持的内容。client 会依赖 capability set 来 gate features。

### 实现 `tools/list` 和 `tools/call`

`tools/list` 返回 `{tools: [...]}`，每个 entry 有 `name`、`description`、`inputSchema`。`tools/call` 接收 `{name, arguments}`，并返回 `{content: [blocks], isError: bool}`。

Content blocks 是 typed。最常见的是：

```json
{"type": "text", "text": "Found 2 notes"}
{"type": "resource", "resource": {"uri": "notes://14", "text": "..."}}
{"type": "image", "data": "<base64>", "mimeType": "image/png"}
```

工具错误有两种形状。Protocol-level errors（unknown method、bad params）是 JSON-RPC errors。Tool-level errors（有效调用，但工具失败）作为 `{content: [...], isError: true}` 返回。这让模型能在 context 中看到失败。

### 实现 resources

Resources 设计上是 read-only。`resources/list` 返回 manifest；`resources/read` 返回内容。URI 可以是 `file://...`、`http://...`，也可以是 `notes://` 这样的自定义 scheme。

当你把数据暴露为 resource 而不是 tool 时：

- 模型不会“调用”它；client 可以在用户请求时把它注入 context。
- Subscriptions 允许 server 在 resource 变化时 push updates（阶段 13 · 10）。
- 阶段 13 · 14 会用 `ui://` 把它扩展成 interactive resources。

### 实现 prompts

Prompts 是带命名参数的模板。host 会把它们显示为 slash-commands。一个 `review_note` prompt 可能接收 `note_id` 参数，并生成一个 multi-message prompt template，client 再把它喂给模型。

### Stdio transport subtleties

- Newline-delimited JSON。没有 length-prefixed framing。
- 不要 buffer。每次写完都 `sys.stdout.flush()`。
- client 控制生命周期。stdin 关闭（EOF）时干净退出。
- 不要静默处理 SIGPIPE；记录并退出。

### Annotations

每个工具可以携带 `annotations` 来描述安全属性：

- `readOnlyHint: true` —— 纯 read，可安全 retry。
- `destructiveHint: true` —— 不可逆副作用；client 应确认。
- `idempotentHint: true` —— 相同输入产生相同输出。
- `openWorldHint: true` —— 与外部系统交互。

client 用这些信息来决定 UX（confirmation dialogs、status indicators）和 routing（阶段 13 · 17）。

### Graduation path

`code/main.py` 中的 stdlib server 大约 180 行。FastMCP（Python）把同样逻辑压缩成 decorator-style：

```python
from fastmcp import FastMCP
app = FastMCP("notes")

@app.tool()
def notes_search(query: str, limit: int = 10) -> list[dict]:
    ...
```

TypeScript SDK 有等价形状。准备好后迁移是 drop-in；概念（capabilities、dispatch、content blocks）保持相同。

## 使用它

`code/main.py` 是一个通过 stdio 运行的完整 notes MCP server，只用 stdlib。它处理 `initialize`、`tools/list`、三个工具（`notes_list`、`notes_search`、`notes_create`）的 `tools/call`、每条 note 的 `resources/list` 和 `resources/read`，以及一个 `review_note` prompt。你可以通过 pipe JSON-RPC messages 驱动它：

```
echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}' | python main.py
```

重点看：

- dispatcher 是一个以 method name 为 key 的 `dict[str, Callable]`。
- 每个 tool executor 返回的是 content block list，而不是裸字符串。
- executor 抛错时会设置 `isError: true`。

## 交付它

本课产出 `outputs/skill-mcp-server-scaffolder.md`。给定一个 domain（notes、tickets、files、database），这个 skill 会 scaffold 一个 MCP server，包含正确的 tools / resources / prompts 划分，以及 SDK graduation path。

## 练习

1. 运行 `code/main.py`，用手写 JSON-RPC messages 驱动它。练习 `notes_create`，然后用 `resources/read` 取回新 note。

2. 添加一个带 `annotations: {destructiveHint: true}` 的 `notes_delete` tool。验证 client 会显示 confirmation dialog（这需要真实 host；Claude Desktop 可用）。

3. 实现 `resources/subscribe`，让 server 在 note 被修改时 push `notifications/resources/updated`。添加 keepalive task。

4. 把 server 移植到 FastMCP。Python 文件应缩到 80 行以内。wire behavior 必须相同；用同一套 JSON-RPC test harness 验证。

5. 阅读 spec 的 `server/tools` 章节，找出本课 server 没有实现的一个 tool definition 字段。（提示：有好几个；选一个并添加。）

## 关键词

| Term | 大家常说 | 实际含义 |
|------|----------|----------|
| MCP server | “暴露工具的东西” | 通过 stdio 或 HTTP 说 MCP JSON-RPC 的进程 |
| stdio transport | “Child process model” | server 由 client 启动；通过 stdin/stdout 通信 |
| Dispatcher | “Method router” | JSON-RPC method name 到 handler function 的映射 |
| Content block | “Tool result chunk” | tool response 的 `content` array 中的 typed element |
| `isError` | “Tool-level failure” | 表示工具失败；与 JSON-RPC error 区分 |
| Annotations | “Safety hints” | readOnly / destructive / idempotent / openWorld flags |
| FastMCP | “Python SDK” | MCP protocol 上层的 decorator-based high-level framework |
| Resource URI | “Addressable data” | 标识 resource 的 `file://`、`db://` 或自定义 scheme |
| Prompt template | “Slash-command brief” | 带 argument slot 的 server-supplied template |
| Capability declaration | “Feature toggle” | `initialize` 中声明的 per-primitive flags |

## 延伸阅读

- [Model Context Protocol — Python SDK](https://github.com/modelcontextprotocol/python-sdk) — reference Python implementation
- [Model Context Protocol — TypeScript SDK](https://github.com/modelcontextprotocol/typescript-sdk) — parallel TS implementation
- [FastMCP — server framework](https://gofastmcp.com/) — decorator-style Python API for MCP servers
- [MCP — Quickstart server guide](https://modelcontextprotocol.io/quickstart/server) — 使用任一 SDK 的端到端教程
- [MCP — Server tools spec](https://modelcontextprotocol.io/specification/2025-11-25/server/tools) — tools/* messages 完整参考
