# MCP Security II：OAuth 2.1、Resource Indicators、Incremental Scopes

> Remote MCP servers 需要的是 authorization，而不只是 authentication。2025-11-25 spec 与 OAuth 2.1 + PKCE + resource indicators（RFC 8707）+ protected-resource metadata（RFC 9728）对齐。SEP-835 通过 403 WWW-Authenticate 上的 step-up authorization 添加 incremental scope consent。本课把 step-up flow 实现成 state machine，让你看到每一跳。

**类型：** 构建
**语言：** Python（stdlib，OAuth state machine simulator）
**前置要求：** 阶段 13 · 09（transports），阶段 13 · 15（security I）
**时间：** ~75 分钟

## 学习目标

- 区分 resource server 和 authorization server 的职责。
- 走通 PKCE-protected OAuth 2.1 authorization code flow。
- 使用 `resource`（RFC 8707）和 protected-resource metadata（RFC 9728）防止 confused-deputy attacks。
- 实现 step-up authorization：server 用 403 + WWW-Authenticate 请求更高 scope；client 重新提示用户 consent 并 retry。

## 问题

早期 MCP（pre-2025）的 remote server 使用 ad-hoc API keys，甚至没有 auth。2025-11-25 spec 用完整 OAuth 2.1 profile 补上这个缺口。

三个真实需求：

- **普通 remote servers。** 用户安装一个访问其 Notion / GitHub / Gmail 的 remote MCP server。OAuth 2.1 with PKCE 是正确形状。
- **Scope escalation。** 已被授予 `notes:read` 的 notes server，之后可能为某个动作需要 `notes:write`。与其重做完整 flow，不如用 step-up（SEP-835）请求额外 scope。
- **Confused deputy prevention。** client 持有 audience-scoped 给 Server A 的 token。Server A 是恶意的，试图把 token 呈递给 Server B。Resource indicators（RFC 8707）会把 token pin 到预期 audience。

OAuth 2.1 并不新。新的是 MCP 的 profile：明确要求的 flow（只允许 authorization code + PKCE；默认没有 implicit、没有 client credentials）、每次 token request 必须有 resource indicators，以及发布 protected-resource metadata 让 clients 知道去哪。

## 概念

### Roles

- **Client。** MCP client（Claude Desktop、Cursor 等）。
- **Resource server。** MCP server（notes、GitHub、Postgres 等）。
- **Authorization server。** 发行 tokens。它可以和 resource server 是同一个 service，也可以是独立 IdP（Auth0、Keycloak、Cognito）。

在 MCP profile 中，resource 和 authorization servers 可以是同一 host，但应通过 URL 区分。

### Authorization code + PKCE

flow：

1. Client 生成 `code_verifier`（随机）和 `code_challenge`（SHA256）。
2. Client 把用户重定向到 `/authorize?response_type=code&client_id=...&redirect_uri=...&scope=notes:read&code_challenge=...&resource=https://notes.example.com`。
3. 用户 consent。Authorization server 重定向到 `redirect_uri?code=...`。
4. Client POST 到 `/token?grant_type=authorization_code&code=...&code_verifier=...&resource=...`。
5. Authorization server 验证 verifier 的 hash 是否匹配存储的 challenge，并发行 access token。
6. Client 使用 token：在每个发往 resource server 的 request 上加 `Authorization: Bearer ...`。

PKCE 防止 authorization-code interception attacks。Resource indicators 防止 token 在其他地方有效。

### Protected-resource metadata（RFC 9728）

resource server 发布 `.well-known/oauth-protected-resource` document：

```json
{
  "resource": "https://notes.example.com",
  "authorization_servers": ["https://auth.example.com"],
  "scopes_supported": ["notes:read", "notes:write", "notes:delete"]
}
```

client 从 resource server 发现 authorization server。减少配置——client 只需要 resource URL。

### Resource indicators（RFC 8707）

token request 中的 `resource` parameter 会 pin token 的 intended audience。发行的 token 包含 `aud: "https://notes.example.com"`。另一个 MCP server 收到此 token，会检查 `aud` 并拒绝它。

### Scope model

Scopes 是空格分隔的 strings。常见 MCP conventions：

- `notes:read`、`notes:write`、`notes:delete`
- `admin:*` 用于 admin capabilities（谨慎使用）
- `profile:read` 用于 identity

Scope selection 应遵循 least-privilege：现在需要什么就请求什么，需要更多时 step up。

### Step-up authorization（SEP-835）

用户授予 `notes:read`。后来他们要求 agent 删除一条 note。server 响应：

```
HTTP/1.1 403 Forbidden
WWW-Authenticate: Bearer error="insufficient_scope",
    scope="notes:delete", resource="https://notes.example.com"
```

client 看到 insufficient_scope error，向用户显示追加 scope 的 consent dialog，为它执行一次 mini OAuth flow，然后带新 token retry request。

### Token audience validation

每个 request：server 检查 `token.aud == self.resource_url`。不匹配 = 401。这会阻止 cross-server token reuse。

### Short-lived tokens and rotation

Access tokens 应 short-lived（默认 1 小时）。Refresh tokens 每次刷新都 rotate。client 在后台处理 silent refresh。

### No token passthrough

Sampling servers（阶段 13 · 11）绝不能把 client token 传给其他 services。sampling request 是边界。

### Confused deputy prevention

token 绑定到 `aud`。client 绑定到 `client_id`。每个 request 都对两者验证。spec 明确禁止旧 “pass-the-token” 模式，这在 pre-MCP remote tool ecosystems 中很常见。

### Client ID discovery

每个 MCP client 都在固定 URL 发布自己的 metadata。Authorization servers 可以抓取 client metadata document 来发现 redirect URIs 和 contact info。这移除了手动 client registration。

### Gateways and OAuth

阶段 13 · 17 会展示 enterprise gateway 如何处理 OAuth：gateway 持有 upstream server 凭据，给 client 的 token 由 gateway 发行，upstream tokens 永远不离开 gateway。这会翻转 trust model——用户只向 gateway 认证一次；gateway 负责 N 个 server authorizations。

## 使用它

`code/main.py` 把完整 OAuth 2.1 step-up flow 模拟成 state machine。它实现：

- PKCE code-verifier / challenge generation。
- 带 resource indicator 的 authorization code flow。
- Protected-resource metadata endpoint。
- 带 audience check 的 token validation。
- `insufficient_scope` 上的 step-up。

本课没有 HTTP server；state machine 在内存中运行，这样你可以追踪每一跳。阶段 13 · 17 的 gateway lesson 会把它接到真实 transport。

## 交付它

本课产出 `outputs/skill-oauth-scope-planner.md`。给定一个带 tools 的 remote MCP server，这个 skill 会设计 scope set、pinning rules 和 step-up policy。

## 练习

1. 运行 `code/main.py`。追踪 two-scope step-up flow。注意 step-up 时哪些 hop 会重复。

2. 添加 refresh-token rotation：每次 refresh 发行新的 refresh token，并使旧 token 失效。模拟被盗 refresh token 在 rotation 后被使用，并确认失败。

3. 用 stdlib http.server 把 protected-resource metadata endpoint 实现成真实 HTTP response。复用第 09 课的 /mcp endpoint。

4. 为 GitHub MCP server 设计 scope hierarchy：read repo、write PR、approve PR、merge PR、admin。在每一层之间使用 step-up。

5. 阅读 RFC 8707 和 RFC 9728。找出 9728 中一个 MCP 用法不同于 RFC 示例的字段。（提示：它涉及 `scopes_supported`。）

## 关键词

| Term | 大家常说 | 实际含义 |
|------|----------|----------|
| OAuth 2.1 | “Modern OAuth” | 要求 PKCE 并禁止 implicit flow 的 consolidated RFC |
| PKCE | “Proof-of-possession” | code verifier + challenge，防止 authorization-code interception |
| Resource indicator | “Token audience” | RFC 8707 `resource` parameter，把 token pin 到一个 server |
| Protected-resource metadata | “Discovery doc” | RFC 9728 `.well-known/oauth-protected-resource` |
| Step-up authorization | “Incremental consent” | SEP-835 中按需添加 scopes 的 flow |
| `insufficient_scope` | “403 with WWW-Authenticate” | server 发出的重新 consent 更大 scope 的信号 |
| Confused deputy | “Token reuse across services” | 受信任持有者不当转发 token 的攻击 |
| Short-lived token | “Access token TTL” | 快速过期的 bearer；refresh token 用于续期 |
| Scope hierarchy | “Least privilege stack” | 带 step-up 的分级 scope set |
| Client ID metadata | “Client discovery doc” | client 发布自己 OAuth metadata 的 URL |

## 延伸阅读

- [MCP — Authorization spec](https://modelcontextprotocol.io/specification/draft/basic/authorization) — MCP OAuth profile 权威参考
- [den.dev — MCP November authorization spec](https://den.dev/blog/mcp-november-authorization-spec/) — 2025-11-25 changes walkthrough
- [RFC 8707 — Resource indicators for OAuth 2.0](https://datatracker.ietf.org/doc/html/rfc8707) — audience-pinning RFC
- [RFC 9728 — OAuth 2.0 protected resource metadata](https://datatracker.ietf.org/doc/html/rfc9728) — discovery-document RFC
- [Aembit — MCP OAuth 2.1, PKCE and the future of AI authorization](https://aembit.io/blog/mcp-oauth-2-1-pkce-and-the-future-of-ai-authorization/) — practical step-up-flow walkthrough
