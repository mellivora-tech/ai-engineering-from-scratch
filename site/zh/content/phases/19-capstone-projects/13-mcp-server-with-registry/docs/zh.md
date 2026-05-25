# Capstone 13 — 带 Registry 与 Governance 的 MCP Server

> Model Context Protocol 在 2026 年不再是未来，而是默认的 tool-use spec。Anthropic、OpenAI、Google 和所有主流 IDE 都提供 MCP clients。Pinterest 发布了内部 MCP servers 生态。AAIF Registry 在 `.well-known` 上正式化了 capability metadata。AWS ECS 发布了 reference stateless deployment。Block 的 goose-agent 把同一协议放进 hosted assistant。2026 年的生产形态是：StreamableHTTP transport、OAuth 2.1 scopes、OPA policy gating，以及一个让 platform teams 能发现、验证、启用 servers 的 registry。请 end-to-end 构建它。

**类型：** Capstone
**语言：** Python（server，经 FastMCP）或 TypeScript（@modelcontextprotocol/sdk）、Go（registry service）
**前置要求：** 阶段 11（LLM engineering）、阶段 13（tools and MCP）、阶段 14（agents）、阶段 17（infrastructure）、阶段 18（safety）
**覆盖阶段：** P11 · P13 · P14 · P17 · P18
**时间：** 25 小时

## 问题

MCP 成了 tool-use 的 lingua franca。Claude Code、Cursor 3、Amp、OpenCode、Gemini CLI 和每个 managed agent 现在都消费 MCP servers。生产挑战不是 authoring servers（FastMCP 已经让这件事很容易），而是带 enterprise requirements 大规模部署：per-tenant OAuth scopes、针对 destructive tools 的 OPA policy、StreamableHTTP stateless scaling、用于 discovery 的 registry、每个 tool call 的 audit logs。Pinterest 的内部 MCP ecosystem 和 AAIF Registry spec 定义了 2026 年的门槛。

你将构建一个暴露 10 个内部工具的 MCP server（Postgres read-only、S3 listing、Jira、Linear、Datadog 等），一个 platform discovery 用的 registry UI，以及 destructive tools 的 human-approval gate。load test 会演示 StreamableHTTP horizontal scaling。audit trail 会满足 enterprise security review。

## 概念

MCP 2026 revision 强制把 StreamableHTTP 作为默认 transport。不同于早期的 stdio-and-SSE 形态，StreamableHTTP 默认 stateless：一个 HTTP endpoint 接收 JSON-RPC requests、流式返回 responses，并支持 notifications 的 long-lived connections。stateless 意味着可以在 load balancer 后面水平扩展。

Authorization 使用 OAuth 2.1 和 per-tool scopes。token 携带类似 `jira:read`、`s3:list`、`postgres:query:readonly` 的 scopes。MCP server 在 tool-call time 检查 scopes，而不是只在 session start 检查。对于 high-risk tools，server 会拒绝任何最近 N 分钟内没有提升到 `approved:by:human` scope 的调用，这个 elevation 来自 Slack review card。

registry 是独立 service。每个 MCP server 都暴露一个 `.well-known/mcp-capabilities` document，包含 tool manifest、transport URL 和 auth requirements。registry 会轮询、验证并索引。platform teams 通过 registry UI 查看可用工具、需要哪些 scopes、由哪个团队拥有。

## 架构

```
MCP client (Claude Code, Cursor 3, ...)
          |
          v
StreamableHTTP over HTTPS (JSON-RPC + streaming)
          |
          v
MCP server (FastMCP) behind load balancer
          |
   +------+------+---------+----------+------------+
   v             v         v          v            v
Postgres    S3 listing  Jira       Linear     Datadog
(read-only) (paged)     (read)     (read)     (query)
          |
   +------+-------------+
   v                    v
 OPA policy gate   destructive tool MCP (separate server)
                        |
                        v
                   human approval via Slack
                        |
                        v
                   audit log (append-only, per-tenant)

  registry service
     |
     v  GET /.well-known/mcp-capabilities from each server
     v
     UI: search / validate / enable-disable / ownership
```

## 技术栈

- Server framework：FastMCP（Python）或 `@modelcontextprotocol/sdk`（TypeScript）
- Transport：基于 HTTPS 的 StreamableHTTP（stateless）
- Auth：OAuth 2.1，结合 SPIFFE / SPIRE 的 workload identity
- Policy：每个 tool 一套 OPA / Rego rules；每个 request 调用 policy decision service
- Registry：self-hosted，消费 `.well-known/mcp-capabilities` manifests
- Human approval：destructive tools 使用 Slack interactive message
- Deployment：AWS ECS Fargate 或 Fly.io，每 tenant 一个 server 或共享 server 加 tenant scoping
- Audit：per-tenant bucket 中的 structured JSONL，带 per-call lineage

## 构建它

1. **Tool surface。** 暴露 10 个 internal tools：Postgres read-only query、S3 list objects、Jira search/fetch、Linear search/fetch、Datadog metric query、PagerDuty on-call lookup、GitHub read-only、Notion search、Slack search、Salesforce read。每个 tool 都有 typed schema 和 scope label。

2. **FastMCP server。** 挂载 tools。配置 StreamableHTTP transport。添加 OAuth token introspection 和 scope enforcement middleware。

3. **OPA policy。** 每个 tool 一条 Rego policy：哪些 scopes 允许 invocation、应用哪些 PII redaction、payload-size caps 是多少。每次 tool call 都调用 decision service。

4. **Registry service。** 独立 Go 或 TS service，轮询 registered servers 的 `.well-known/mcp-capabilities`，用 JSON Schema 验证，并暴露 list / search / validate / enable-disable UI。

5. **Capability manifest。** 每个 server 暴露 `.well-known/mcp-capabilities`，包含：tool list、auth requirements、transport URL、owner team、SLO。

6. **Destructive tool separation。** 会 mutate state 的工具（Jira create、Linear create、Postgres write）放在第二个 MCP server 上，并采用更严格 auth flow：tokens 必须带有最近 15 分钟内通过 Slack card 提升的 `approved:by:human` scope。

7. **Audit log。** Per tenant append-only JSONL：`{timestamp, user, tool, args_redacted, response_redacted, outcome}`。写入前用 Presidio 做 PII redaction。

8. **Load test。** StreamableHTTP 上 100 concurrent clients。通过添加第二个 replica 演示 horizontal scaling；展示 load balancer 在无 session stickiness 下重新分配流量。

9. **Conformance tests。** 对两个 server 运行 official MCP conformance suite。通过所有 mandatory sections。

## 使用它

```
$ curl -H "Authorization: Bearer eyJhbGc..." \
       -X POST https://mcp.internal.example.com/ \
       -d '{"jsonrpc":"2.0","method":"tools/call",
            "params":{"name":"postgres.readonly","arguments":{"sql":"SELECT 1"}}}'
[registry]   capability validated: postgres.readonly v1.2
[policy]    scope postgres:query:readonly present; allowed
[audit]     logged: user=u42 tool=postgres.readonly outcome=ok
response:    { "result": { "rows": [[1]] } }
```

## 交付它

`outputs/skill-mcp-server.md` 描述交付物：一个 production-grade MCP server + registry + audit layer，用于带 OAuth 2.1 scopes 和 OPA gating 的 internal tools。

| 权重 | 标准 | 衡量方式 |
|:-:|---|---|
| 25 | Spec conformance | StreamableHTTP + capability manifest 通过 MCP conformance tests |
| 20 | Security | Scope enforcement、每个 tool 的 OPA coverage、secret hygiene |
| 20 | Observability | 带 PII redaction 的 per-tool-call audit log |
| 20 | Scale | 100-client load test horizontal scale demonstration |
| 15 | Registry UX | Discover / validate / enable-disable workflow |
| **100** | | |

## 练习

1. 添加一个新工具（Confluence search）。通过 registry validation flow 发布它，不触碰 core server。

2. 编写一条 OPA policy，对包含名为 `email`、`ssn` 或 `phone` 的 columns 的 Postgres query results 做 redaction。用 probe query 演练。

3. 在本地 latency 上基准测试 StreamableHTTP vs stdio。报告 per-call p50/p95。

4. 实现 per-tenant quota：每个 tenant 每个 tool 每分钟最多 N 次调用。通过第二条 OPA rule 强制执行。

5. 运行 [mcp-conformance-tests](https://github.com/modelcontextprotocol/conformance) 中的 MCP conformance suite，并修复每个 failure。

## 关键词汇

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|------------------------|
| StreamableHTTP | “2026 MCP transport” | Stateless HTTP + streaming；替代 networked servers 的 SSE + stdio |
| Capability manifest | “Well-known doc” | `.well-known/mcp-capabilities`，包含 tool list、auth、transport URL |
| OPA / Rego | “Policy engine” | Open Policy Agent，用 external rules 授权 tool calls |
| Scope elevation | “Approved-by-human” | 通过 Slack approval 授予的短期 scope，destructive tools 必需 |
| Registry | “Tool discovery” | 从 capability manifests 索引 MCP servers 的 service |
| Workload identity | “SPIFFE / SPIRE” | 用于 OAuth token issuance 的 cryptographic service identity |
| Conformance suite | “Spec tests” | 针对 StreamableHTTP + tool manifest correctness 的 official MCP test battery |

## 延伸阅读

- [Model Context Protocol 2026 Roadmap](https://blog.modelcontextprotocol.io/posts/2026-mcp-roadmap/) — StreamableHTTP、capability metadata、registry
- [AAIF MCP Registry spec](https://github.com/modelcontextprotocol/registry) — 2026 registry spec
- [AWS ECS reference deployment](https://aws.amazon.com/blogs/containers/deploying-model-context-protocol-mcp-servers-on-amazon-ecs/) — reference production deployment
- [Pinterest internal MCP ecosystem](https://www.infoq.com/news/2026/04/pinterest-mcp-ecosystem/) — reference internal deployment
- [Block `goose` MCP usage](https://block.github.io/goose/) — reference agent consumption pattern
- [FastMCP](https://github.com/jlowin/fastmcp) — Python server framework
- [Open Policy Agent](https://www.openpolicyagent.org/) — policy engine reference
- [SPIFFE / SPIRE](https://spiffe.io) — workload identity reference
