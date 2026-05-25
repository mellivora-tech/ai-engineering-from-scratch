# MCP Transports：stdio vs Streamable HTTP vs SSE Migration

> stdio 只适合本地，不适合其他地方。Streamable HTTP（2025-03-26）是 remote 标准。旧 HTTP+SSE transport 已被 deprecated，并会在 2026 年中移除。选错 transport 会带来迁移成本；选对 transport 则得到一个可远程托管、带 session continuity 和 DNS-rebinding protection 的 MCP server。

**类型：** 学习
**语言：** Python（stdlib，Streamable HTTP endpoint skeleton）
**前置要求：** 阶段 13 · 07、08（MCP server 和 client）
**时间：** ~45 分钟

## 学习目标

- 根据部署形态（local vs remote、single-process vs fleet）在 stdio 和 Streamable HTTP 之间选择。
- 实现 Streamable HTTP single-endpoint pattern：POST 处理 requests，GET 建立 session stream。
- 执行 `Origin` validation 和 session-id semantics，防御 DNS-rebinding。
- 在 2026 年中移除 deadline 前，把 legacy HTTP+SSE server 迁移到 Streamable HTTP。

## 问题

第一个 MCP remote transport（2024-11）是 HTTP+SSE：两个 endpoints，一个接 client 的 POST，一个 Server-Sent-Events channel 负责 server-to-client stream。它能工作，但很笨重：每个 session 两个 endpoints，一些 CDN 前面的 cache 会坏，并且硬依赖长连接 SSE，而某些 WAF 会积极终止它。

2025-03-26 spec 用 Streamable HTTP 替代它：一个 endpoint，POST 处理 client request，GET 建立 session stream，两者共享 `Mcp-Session-Id` header。此后构建或迁移的每个 server 都使用 Streamable HTTP。旧 SSE mode 正在被 deprecated——Atlassian Rovo 在 2026 年 6 月 30 日移除；Keboola 在 2026 年 4 月 1 日移除；大多数剩余 enterprise server 会在 2026 年底前移除。

stdio 仍然对 local server 很重要。Claude Desktop、VS Code 和每个 IDE-shaped client 都通过 stdio 启动 server。正确 mental model 是：stdio 用于 “this machine”，Streamable HTTP 用于 “over the network”。不要交叉使用。

## 概念

### stdio

- Child-process transport。client 启动 server，通过 stdin/stdout 通信。
- 每行一个 JSON object。newline-delimited。
- 没有 session id；进程身份就是 session。
- 不需要 auth（child 继承 parent 的 trust boundary）。
- 永远不要用于 remote servers——你会需要 SSH 或 socat 来 tunnel，而那时应使用 Streamable HTTP。

### Streamable HTTP

单 endpoint `/mcp`（或任意 path）。支持三种 HTTP method：

- **POST /mcp。** Client 发送 JSON-RPC message。Server 回复单个 JSON response，或一个包含一条或多条 response 的 SSE stream（对 batched responses 和该 request 相关 notifications 有用）。
- **GET /mcp。** Client 打开一个长连接 SSE channel。Server 用它发送 server-to-client requests（sampling、notifications、elicitation）。
- **DELETE /mcp。** Client 显式终止 session。

Session 由 `Mcp-Session-Id` header 标识；server 在第一次 response 中设置它，client 在后续每个 request 中回显。Session id 必须是 cryptographically random（128+ bits）；为了安全，拒绝 client-chosen ids。

### Single endpoint vs two

旧 spec 的 two-endpoint mode 在 2026 年仍然 callable——spec 称之为 “legacy compatible”。但所有新 server 都应使用 single-endpoint。官方 SDK 发出 single-endpoint；只有和未迁移 remote 对话时才使用 legacy mode。

### `Origin` validation 和 DNS-rebinding

浏览器今天不是 MCP client，但攻击者可以制作一个网页，让浏览器 POST 到 `localhost:1234/mcp`——用户的本地 MCP server 可能就监听在那里。如果 server 不检查 `Origin`，浏览器的 same-origin policy 救不了它，因为 `Origin: http://evil.com` 是有效的 cross-origin。

2025-11-25 spec 要求 server 拒绝 `Origin` 不在 allowlist 上的 request。allowlist 通常包含 MCP client host（`https://claude.ai`、`vscode-webview://*`）和本地 UI 的 localhost variants。

### Session id lifecycle

1. Client 第一次 request 不带 `Mcp-Session-Id`。
2. Server 分配随机 id，并在 response header 上设置 `Mcp-Session-Id`。
3. Client 在所有后续 request 和用于 stream 的 `GET /mcp` 上回显这个 header。
4. Server 可以 revoke session；client 在后续 request 上看到 404，并必须重新 initialize。
5. Client 可以显式 DELETE session，做 clean shutdown。

### Keepalive and reconnect

SSE 连接会断。client 用同一个 `Mcp-Session-Id` 重新 GET 来重建。Server 必须 queue outage 期间错过的 events（在合理窗口内），并通过 client 回显的 `last-event-id` header replay。

阶段 13 · 13 会覆盖 Tasks，它让 long-running work 即使在 full-session reconnect 后也能存活。

### Backwards compatibility probe

一个想同时支持新旧 server 的 client：

1. POST 到 `/mcp`。
2. 如果 response 是带 JSON 或 SSE 的 `200 OK`，这是 Streamable HTTP。
3. 如果 response 是 `200 OK`，`Content-Type: text/event-stream`，并且 `Location` header 指向 secondary endpoint，这是 legacy HTTP+SSE；follow `Location`。

### Cloudflare、ngrok 和 hosting

2026 年生产 remote MCP servers 跑在 Cloudflare Workers（带 MCP Agents SDK）、Vercel Functions，或 containerized Node/Python 上。关键点：hosting 必须支持用于 SSE GET 的 long-lived HTTP connections。Vercel free tier 上限 10 秒，不适合。Cloudflare Workers 支持 indefinite streams。

### Gateway composition

当你用 gateway（阶段 13 · 17）前置多个 MCP servers 时，gateway 是一个单独的 Streamable HTTP endpoint，会 rewrite session ids 并 multiplex upstream。Tools 在 gateway 层合并；client 看到的是一个 logical server。

### Transport failure modes

- **stdio SIGPIPE。** Child process 在 mid-write 死亡会引发 SIGPIPE；server 应干净退出。client 应检测 EOF 并标记 session dead。
- **HTTP 502 / 504。** Cloudflare、nginx 和其他 proxy 会在 upstream failure 时发出这些。Streamable HTTP client 应短暂 backoff 后 retry 一次。
- **SSE connection drop。** TCP RST、proxy timeout 或 client network change 会关闭 stream。client 用 `Mcp-Session-Id` 和可选的 `last-event-id` reconnect，以 resume。
- **Session revocation。** Server 使 session id 失效；client 在下次 request 看到 404。client 必须重新 handshake。
- **Clock skew。** client 上的 Resource-TTL calculation 与 server 发散。client 应把 server timestamp 视为权威。

### 什么时候绕过 Streamable HTTP

一些企业在内部网络中把 MCP server 部署在 gRPC 或 message-queue transport 后面。这是非标准的——MCP spec 没有正式定义这些。gateway 可以对 MCP client 暴露 Streamable HTTP surface，同时内部使用 gRPC。保持外部界面 spec-compliant；gateway 负责翻译。

## 使用它

`code/main.py` 使用 `http.server`（stdlib）实现一个最小 Streamable HTTP endpoint。它处理 `/mcp` 上的 POST、GET 和 DELETE，在第一次 response 上设置 `Mcp-Session-Id`，验证 `Origin`，并拒绝非 allowlisted origins。handler 复用了第 07 课 notes server 的 dispatch logic。

重点看：

- POST handler 读取 JSON-RPC body、dispatch，并写出 JSON response（single-response variant；SSE variant 结构类似）。
- `Origin` check 拒绝默认的 `http://evil.example` probe，但接受 `http://localhost`。
- Session ids 是随机 128-bit hex strings；server 在内存中保存 per-session state。

## 交付它

本课产出 `outputs/skill-mcp-transport-migrator.md`。给定一个 HTTP+SSE（legacy）MCP server，这个 skill 会生成迁移到 Streamable HTTP 的计划，包含 session-id continuity、Origin checks 和 backwards-compatible probe support。

## 练习

1. 运行 `code/main.py`。用 `curl` POST 一个 `initialize`，观察 `Mcp-Session-Id` response header。第二次 POST 时回显这个 header，并验证 session continuity。

2. 添加一个打开 SSE stream 的 GET handler。每五秒发送一个 `notifications/progress` event。用同一个 session id 重新 GET 来 reconnect，并确认 server 接受它。

3. 实现 `last-event-id` replay logic。reconnect 时，replay 自该 id 以来生成的任何 events。

4. 扩展 `Origin` validation，使其支持 wildcard pattern（`https://*.example.com`），并确认它接受 `https://app.example.com`，但拒绝 `https://evil.example.com.attacker.net`。

5. 从官方 registry 中选择一个 legacy HTTP+SSE server（有好几个），草拟迁移方案：endpoint handling、session id generation 和 header semantics 分别怎么变。

## 关键词

| Term | 大家常说 | 实际含义 |
|------|----------|----------|
| stdio transport | “Local child process” | stdin/stdout 上的 newline-delimited JSON-RPC |
| Streamable HTTP | “remote transport” | single-endpoint POST + GET + optional SSE，2025-03-26 spec |
| HTTP+SSE | “Legacy” | 2026 年中移除的 two-endpoint model |
| `Mcp-Session-Id` | “Session header” | server-assigned random id，后续每个 request 都回显 |
| `Origin` allowlist | “DNS-rebinding defense” | 拒绝 Origin 未批准的 request |
| Single endpoint | “One URL” | `/mcp` 处理所有 session 操作的 POST / GET / DELETE |
| `last-event-id` | “SSE replay” | 用于恢复断开的 stream 而不错过 events 的 header |
| Backwards-compat probe | “Old vs new detection” | client 根据 response shape 自动选择 transport |
| Long-lived HTTP | “SSE streaming” | server 在一个 TCP connection 上 push 数分钟或数小时 events |
| Session revocation | “Force re-init” | server 使 session id 失效；client 必须重新 handshake |

## 延伸阅读

- [MCP — Basic transports spec 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25/basic/transports) — stdio 和 Streamable HTTP 的权威参考
- [MCP — Basic transports spec 2025-03-26](https://modelcontextprotocol.io/specification/2025-03-26/basic/transports) — 引入 Streamable HTTP 的 revision
- [Cloudflare — MCP transport](https://developers.cloudflare.com/agents/model-context-protocol/transport/) — Workers-hosted Streamable HTTP patterns
- [AWS — MCP transport mechanisms](https://builder.aws.com/content/35A0IphCeLvYzly9Sw40G1dVNzc/mcp-transport-mechanisms-stdio-vs-streamable-http) — 不同部署形态的比较
- [Atlassian — HTTP+SSE deprecation notice](https://community.atlassian.com/forums/Atlassian-Remote-MCP-Server/HTTP-SSE-Deprecation-Notice/ba-p/3205484) — 具体迁移 deadline 示例
