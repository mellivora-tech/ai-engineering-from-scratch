# MCP Gateways 和 Registries：Enterprise Control Planes

> 企业不能让每个开发者随意安装随机 MCP servers。gateway 会集中 auth、RBAC、audit、rate limiting、caching 和 tool-poisoning detection，然后把合并后的 tool surface 作为单个 MCP endpoint 暴露。Official MCP Registry（Anthropic + GitHub + PulseMCP + Microsoft，namespace-verified）是 canonical upstream。本课命名 gateway 所在位置，走通一个最小实现，并概览 2026 vendor landscape。

**类型：** 学习
**语言：** Python（stdlib，minimal gateway）
**前置要求：** 阶段 13 · 15（tool poisoning），阶段 13 · 16（OAuth 2.1）
**时间：** ~45 分钟

## 学习目标

- 解释 MCP gateway 位于哪里（在 MCP clients 和多个 backend MCP servers 之间）。
- 实现五项 gateway responsibilities：auth、RBAC、audit、rate limit、policy。
- 在 gateway 层执行 pinned-tool-hash manifest。
- 区分 Official MCP Registry 和 metaregistries（Glama、MCPMarket、MCP.so、Smithery、LobeHub）。

## 问题

一家 Fortune 500 有 30 个 approved MCP servers、5000 名 developers、compliance 和 audit requirements，以及一个希望集中 policy 的安全团队。让每个开发者在自己的 IDE 中安装任意 server 是不可接受的。

gateway pattern：

1. Gateway 作为 developers 连接的单个 Streamable HTTP endpoint 运行。
2. Gateway 持有每个 backend MCP server 的 credentials。
3. 每个 developer request 都通过 gateway 自己的 OAuth 完成 authentication 和 scoping。
4. Gateway 把 call 路由到 backend server，并应用 policy。
5. 所有 calls 记录进 audit。

Cloudflare MCP Portals、Kong AI Gateway、IBM ContextForge、MintMCP、TrueFoundry、Envoy AI Gateway 都在 2025-2026 年发布了 gateways 或 gateway features。

与此同时，Official MCP Registry 作为 canonical upstream 发布：curated、namespace-verified、reverse-DNS-named servers，gateway 可以从中拉取。Metaregistries（Glama、MCPMarket、MCP.so、Smithery、LobeHub）会聚合多个来源的 servers。

## 概念

### 五项 gateway responsibilities

1. **Auth。** OAuth 2.1 用于识别 developer；映射到 user roles。
2. **RBAC。** per-user policy：哪些 servers、哪些 tools、哪些 scopes。
3. **Audit。** 每个 call 记录 who、what、when、result。
4. **Rate limit。** per-user / per-tool / per-server caps，防止 abuse。
5. **Policy。** 拒绝 poisoned descriptions，执行 Rule of Two，redact PII。

### Gateway as a single endpoint

对 developers 来说，gateway 看起来像一个 MCP server。内部它路由到 N 个 backends。Session ids（阶段 13 · 09）会在边界处 rewrite。

### Credential vaulting

Developers 永远看不到 backend tokens。gateway 持有它们（或代理给 identity provider）。一个在 gateway 上拥有 `notes:read` 的 developer，可以 transitively 使用 gateway 自己的 backend credentials 访问 notes MCP server——但只能在绑定 transitive access 的 policy 下。

### Tool-hash pinning at the gateway

gateway 持有 approved tool descriptions 的 manifest（SHA256 hashes）。discovery 时，它抓取每个 backend 的 `tools/list`，把 hashes 与 manifest 比较，并移除任何 description 发生 mutation 的工具。这是阶段 13 · 15 的 rug-pull defense 的集中应用。

### Policy-as-code

高级 gateway 用 OPA/Rego、Kyverno 或 Styra 表达 policy。像 “user `alice` 只能在 org `acme` 的 repos 上调用 `github.open_pr`” 这样的规则会 declaratively 编码。简单 gateway 用手写 Python。两种形状都有效。

### Session-aware routing

当用户 session 包含多个 server 时，gateway 会 multiplex：developer 的单个 MCP session 持有 N 个 backend sessions，每个 server 一个。任意 backend 的 notifications 都经 gateway 路由到 developer session。

### Namespace merging

gateways 会合并所有 backend 的 tool namespaces，通常在 collision 时加前缀。`github.open_pr`、`notes.search`。这让 routing 无歧义。

### Registries

- **Official MCP Registry（`registry.modelcontextprotocol.io`）。** 由 Anthropic、GitHub、PulseMCP、Microsoft 托管发布。Namespace-verified（reverse-DNS：`io.github.user/server`）。预过滤基本质量。
- **Glama。** search-centric metaregistry，聚合许多来源。
- **MCPMarket。** 偏商业的目录，带 vendor listings。
- **MCP.so。** community directory；开放提交。
- **Smithery。** package-manager-style installation flow。
- **LobeHub。** 集成在 LobeChat app 中的 UI registry。

Enterprise gateways 默认从 Official Registry 拉取，允许管理员从 metaregistries 添加 curated additions，并拒绝任何未 pin 的内容。

### Reverse-DNS naming

Official Registry 要求 public servers 使用 reverse-DNS names：`io.github.alice/notes`。Namespaces 防止 squatting，并让 trust delegation 更清晰。

### Vendor survey，2026 年 4 月

| Vendor | Strength |
|--------|----------|
| Cloudflare MCP Portals | Edge-hosted；OAuth integrated；free tier |
| Kong AI Gateway | K8s-native；fine-grained policy；logs to OpenTelemetry |
| IBM ContextForge | Enterprise IAM；compliance；audit export |
| TrueFoundry | DevOps-leaning；metrics-first |
| MintMCP | Developer-platform oriented |
| Envoy AI Gateway | Open-source；customizable filters |

阶段 17（production infrastructure）会更深入 gateway operations。

## 使用它

`code/main.py` 提供一个约 150 行的 minimal gateway：通过 fake Bearer token 验证用户，持有 per-user RBAC policy，向两个 backend MCP servers 路由 requests，把每个 call 写入 audit log，执行 rate limit，并拒绝任何 description hash 不匹配 pinned manifest 的 backend tool。

重点看：

- `RBAC` dict 以 `user_id` 为 key，包含允许的 `server_tool` entries。
- `AUDIT_LOG` 是 append-only event list。
- rate limit 使用 per-user token bucket。
- pinned manifest 是 `server::tool -> hash` 的 dict。

## 交付它

本课产出 `outputs/skill-gateway-bootstrap.md`。给定一个 enterprise MCP plan（users、backends、compliance），这个 skill 会生成 gateway configuration spec。

## 练习

1. 运行 `code/main.py`。以 allowed user 调用一次；再以 disallowed user 调用；再做一次超出 rate limit 的 burst。验证三种 flow。

2. 添加一条 policy，在把结果返回 client 前 redact PII。先用简单 regex 处理 SSN-shaped strings；注明缺口（emails、phone numbers）。

3. 扩展 audit log，发出 OpenTelemetry GenAI spans。阶段 13 · 20 会覆盖确切 attributes。

4. 为一个 50 人 developer team 设计 RBAC policy，包含五个 backends（notes、github、postgres、jira、slack）。谁拥有每个 read-only？谁有 write？

5. 从头到尾阅读 Cloudflare enterprise MCP post。找出 Cloudflare 提供、但这个 stdlib gateway 没有的一个 feature。

## 关键词

| Term | 大家常说 | 实际含义 |
|------|----------|----------|
| Gateway | “MCP proxy” | 位于 clients 和 backends 之间的 centralizing server |
| Credential vaulting | “Backend tokens stay server-side” | developers 永远看不到 upstream tokens |
| Session-aware routing | “Multi-backend session” | gateway 为每个 developer session multiplex N 个 backend sessions |
| Tool-hash pinning | “Approved manifest” | 每个 approved tool description 的 SHA256；集中阻止 rug-pulls |
| RBAC | “Per-user policy” | 面向 tools 和 servers 的 role-based access control |
| Policy-as-code | “Declarative rules” | 在 gateway 执行的 OPA/Rego、Kyverno、Styra policies |
| Audit log | “Who, what, when” | compliance 使用的 append-only event log |
| Rate limit | “Per-user token bucket” | 防止 abuse 的 per-minute caps |
| Official MCP Registry | “Canonical upstream” | namespace-verified 的 `registry.modelcontextprotocol.io` |
| Reverse-DNS naming | “Registry namespace” | `io.github.user/server` convention |

## 延伸阅读

- [Official MCP Registry](https://registry.modelcontextprotocol.io/) — canonical upstream，namespace-verified
- [Cloudflare — Enterprise MCP](https://blog.cloudflare.com/enterprise-mcp/) — 带 OAuth 和 policy 的 gateway pattern
- [agentic-community — MCP gateway registry](https://github.com/agentic-community/mcp-gateway-registry) — open-source reference gateway
- [TrueFoundry — What is an MCP gateway?](https://www.truefoundry.com/blog/what-is-mcp-gateway) — feature comparison article
- [IBM — MCP context forge](https://github.com/IBM/mcp-context-forge) — IBM 的 enterprise gateway
