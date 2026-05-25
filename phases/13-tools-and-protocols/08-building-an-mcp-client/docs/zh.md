# 构建 MCP Client：Discovery、Invocation、Session Management

> 大多数 MCP 内容都在教 server，然后对 client 一笔带过。client 代码才是困难 orchestration 所在：process spawning、capability negotiation、跨多个 server 合并 tool list、sampling callbacks、reconnection、namespace collision resolution。本课构建一个 multi-server client，把三个不同 MCP server 提升成一个给模型使用的扁平 tool namespace。

**类型：** 构建
**语言：** Python（stdlib，multi-server MCP client）
**前置要求：** 阶段 13 · 07（building an MCP server）
**时间：** ~75 分钟

## 学习目标

- 把 MCP server 作为 child process 启动，完成 `initialize`，并发送 `notifications/initialized`。
- 维护 per-server session state（capabilities、tool list、last-seen notification ids）。
- 把多个 server 的 tool list 合并成一个 namespace，并处理 collision。
- 把 tool call 路由到拥有它的 server，并重组 response。

## 问题

真实 agent host（Claude Desktop、Cursor、Goose、Gemini CLI）会同时加载多个 MCP server。用户可能同时运行 filesystem server、Postgres server 和 GitHub server。client 的工作是：

1. 启动每个 server。
2. 分别 handshake。
3. 对每个 server 调用 `tools/list` 并 flatten 结果。
4. 当模型输出 `notes_search` 时，在 merged namespace 中查找它，并路由到正确 server。
5. 处理任意 server 发来的 notification（`tools/list_changed`），且不能阻塞。
6. transport failure 后 reconnect。

手写所有这些，正是 “toy” 和 “serviceable” 的分界线。官方 SDK 会包装这些，但 mental model 必须属于你。

## 概念

### Child-process spawning

使用 `subprocess.Popen`，并设置 `stdin=PIPE, stdout=PIPE, stderr=PIPE`。设置 `bufsize=1` 并使用 text mode 做 line-by-line reads。每个 server 是一个进程；client 为每个 server 持有一个 `Popen` handle。

### Per-server session state

每个 server 一个 `Session` object，包含：

- `process` —— Popen handle。
- `capabilities` —— server 在 `initialize` 时声明的能力。
- `tools` —— 上一次 `tools/list` 的结果。
- `pending` —— request id 到等待 response 的 promise/future 的映射。

request 本质上是 async；发给 server A 的 `tools/call` 不能因为 server B 正在 mid-call 而阻塞。可以用 threads + queues，或 asyncio。

### Merged namespace

client 看到 aggregate tool list 时，名称可能冲突。两个 server 都可能暴露 `search`。client 有三种选择：

1. **按 server name 加前缀。** `notes/search`、`files/search`。清晰但丑。
2. **静默 first-come。** 后加载 server 的 `search` 覆盖前一个。危险；隐藏 collision。
3. **拒绝 collision。** 拒绝加载第二个 server；通知用户。对 security-sensitive host 最安全。

Claude Desktop 使用 prefix-by-server。Cursor 使用 collision rejection 并给出清晰错误。VS Code MCP 也采用 prefix-by-server。

### Routing

合并后，dispatch table 把 `tool_name -> session`。模型按名称输出 call；client 找到 session，把 `tools/call` message 写到该 server 的 stdin，然后等待 response。

### Sampling callback

如果 server 在 `initialize` 时声明了 `sampling` capability，它可能发送 `sampling/createMessage`，要求 client 运行它的 LLM。client 必须：

1. 阻塞发往该 server 的后续 request，直到 sample resolve；如果实现支持 concurrency，也可以 pipeline。
2. 调用自己的 LLM provider。
3. 把 response 发回 server。

第 11 课会端到端覆盖 sampling。本课为完整性 stub 它。

### Notification handling

`notifications/tools/list_changed` 意味着重新调用 `tools/list`。`notifications/resources/updated` 意味着如果该 resource 正在使用，则重新读取。Notifications 不能产生 response——不要试图 ack。

一个常见 client bug：在 `tools/call` 上阻塞 read loop，而 notification 卡在 stream 中。使用 background reader thread，把每条 stdout message 推入 queue；main thread 出队并 dispatch。

### Reconnection

Transport 可能失败：server crashed、OS 杀进程、stdio pipe broken。client 检测 stdout 上的 EOF，并把 session 视为 dead。选项：

- 静默重启 server 并重新 handshake。适合 pure read-only servers。
- 把失败暴露给用户。适合有 user-visible session 的 stateful servers。

阶段 13 · 09 会覆盖 Streamable HTTP reconnection semantics；stdio 更简单。

### Keepalive and session id

Streamable HTTP 使用 `Mcp-Session-Id` header。Stdio 没有 session id——进程身份就是 session。Keepalive ping 是可选的；stdio pipes 不会因为 inactivity 断开。

## 使用它

`code/main.py` 会把三个模拟 MCP server 作为 subprocess 启动，分别 handshake，合并它们的 tool list，并把 tool call 路由到正确 server。“servers” 实际上是其他 Python process，运行 toy responders（没有真实 LLM）。运行它可以看到：

- 三次初始化，每个都有自己的 capability set。
- 三个 `tools/list` 结果合并成 7-tool namespace。
- 根据 tool name 做 routing decision。
- 通过 namespace prefixing 防止 collision。

重点看：

- `Session` dataclass 干净保存 per-server state。
- background reader thread 会读取 stdout 的每一行，不阻塞 main thread。
- dispatch table 是简单的 `dict[str, Session]`。
- collision handling 是显式的：当两个 server 声明相同名称时，后一个会加前缀重命名。

## 交付它

本课产出 `outputs/skill-mcp-client-harness.md`。给定一组 declarative MCP servers（name、command、args），这个 skill 会生成一个 harness：启动它们、合并 tool list，并交付带 collision resolution 的 routing function。

## 练习

1. 运行 `code/main.py`，观察 server spawn log。用 SIGTERM 杀掉一个模拟 server process，观察 client 如何检测 EOF 并把 session 标记为 dead。

2. 实现 namespace prefixing。当两个 server 暴露 `search` 时，把第二个重命名为 `<server>/search`。更新 dispatch table，并验证 tool call 正确路由。

3. 为 server restart 添加 connection-pool-style backoff：连续失败时 exponential backoff，上限 30 秒，三次失败后向用户发 notification。

4. 草拟一个支持 100 个 concurrent MCP servers 的 client。什么数据结构替代简单 dispatch dict？（提示：prefix namespacing 用 trie，再加 tool-count-per-server metric。）

5. 把 client 移植到官方 MCP Python SDK。SDK 会包装 `stdio_client` 和 `ClientSession`。代码应从约 200 行缩到约 40 行，并保留 multi-server routing。

## 关键词

| Term | 大家常说 | 实际含义 |
|------|----------|----------|
| MCP client | “Agent host” | 启动 server 并 orchestrate tool calls 的进程 |
| Session | “Per-server state” | Capabilities、tool list 和 pending-request bookkeeping |
| Merged namespace | “一个 tool list” | 所有 active servers 的扁平 tool name 集合 |
| Namespace collision | “两个 server 同名工具” | client 必须 prefix、reject 或 first-come 处理 duplicate |
| Routing | “谁来处理这个 call？” | 从 tool name dispatch 到 owning server |
| Background reader | “Non-blocking stdout” | 把 server stdout drain 到 queue 的 thread 或 task |
| Sampling callback | “LLM-as-a-service” | server 发来的 `sampling/createMessage` 的 client handler |
| `notifications/*_changed` | “Primitive mutated” | client 必须 re-discover 或 re-read 的信号 |
| Reconnection policy | “server 死了怎么办” | transport 失败时的 restart semantics |
| Stdio session | “Process = session” | 没有 session id；child process 生命周期就是 session |

## 延伸阅读

- [Model Context Protocol — Client spec](https://modelcontextprotocol.io/specification/2025-11-25/client) — canonical client behavior
- [MCP — Quickstart client guide](https://modelcontextprotocol.io/quickstart/client) — 使用 Python SDK 的 hello-world client tutorial
- [MCP Python SDK — client module](https://github.com/modelcontextprotocol/python-sdk) — reference `ClientSession` 和 `stdio_client`
- [MCP TypeScript SDK — Client](https://github.com/modelcontextprotocol/typescript-sdk) — TS parallel
- [VS Code — MCP in extensions](https://code.visualstudio.com/api/extension-guides/ai/mcp) — VS Code 如何在单个 editor host 中 multiplex 多个 MCP servers
