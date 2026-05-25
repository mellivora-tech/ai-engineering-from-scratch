# 生产中的 MCP Auth：DCR、JWKS Rotation、iii Primitives 上的 Audience-Pinned Tokens

> 第 16 课在内存中搭起了 OAuth 2.1 state machine。到 2026 年，你交付给真实组织的每个 MCP server 都位于 production auth 之后：dynamic client registration（RFC 7591）、authorization-server metadata discovery（RFC 8414）、不会在凌晨 3 点 token validation 时出问题的 JWKS rotation，以及拒绝 confused-deputy reuse 的 audience-pinned tokens。本课会把这些全部接入 iii primitives——`iii.registerTrigger` 用于 HTTP 和 cron，`iii.registerFunction` 用于 auth logic，`state::set/get` 用于 cached keys——让 auth surface 像 engine 中其他 workload 一样可观测、可重启、可 replay。

**类型：** 构建
**语言：** Python（stdlib，iii primitives mocked for the lesson environment）
**前置要求：** 阶段 13 · 16（OAuth 2.1 state machine），阶段 13 · 17（gateways）
**时间：** ~90 分钟

## 学习目标

- 通过 RFC 8414 metadata 发现 authorization server，并验证合约。
- 实现 RFC 7591 dynamic client registration，让 MCP clients 无需管理员介入即可 enroll。
- 用 cron trigger 缓存并轮换 JWKS keys，让 signature verification 在 key roll-over 后仍能继续。
- 使用 RFC 8707 resource indicators，把 tokens pin 到单个 MCP resource，并拒绝 confused-deputy reuse。
- 把每个 endpoint 和 background job 都接成 iii primitives——HTTP triggers、cron triggers、named functions 和 `state::*` reads——让单次 restart 就能重建 auth surface。
- 读取 IdP capability matrix，并在 IdP 无法满足 MCP auth profile 时拒绝部署。

## 问题

第 16 课的 simulator 在内存中运行 OAuth 2.1。生产环境有三个 memory-only simulator 看不到的 operational gaps。

第一个 gap 是 enrollment。真实组织运行数百个 MCP servers 和数千个 MCP clients。operators 不会手工把每个 Cursor 用户注册成 OAuth client。RFC 7591 dynamic client registration 允许 client 对 authorization server `POST /register`，并当场收到 `client_id`（以及可选的 `client_secret`）。server 在 RFC 8414 metadata 中发布 `registration_endpoint`；client 无需 out-of-band 配置即可发现它。

第二个 gap 是 key rotation。JWT validation 依赖 authorization server 的 signing keys，这些 keys 以 JSON Web Key Set（JWKS）发布。authorization server 会按计划轮换这些 keys（通常每小时一次，事故响应期间可能更快）。如果 MCP server 只在 boot 时 fetch 一次 JWKS，它在 rotation window 前都能验证——然后每个 request 都会失败，直到 restart。生产会把 JWKS 接成带 refresh job 的 cached value，在前一批 key 过期前覆盖 cache；另有 cache miss fallback fetch，用于处理由比 cache 更新的 key 签名的 token 到达的情况。

第三个 gap 是 audience binding。第 16 课介绍了 RFC 8707 resource indicators。生产中，该 indicator 会成为每个 request 上的强制 claim check。MCP server 比较 `token.aud` 和自己的 canonical resource URL，不匹配就以 HTTP 401 拒绝。这是在同一 trust mesh 中防止 upstream MCP server（或持有某个 server token 的恶意 client）把 token replay 到另一个 server 的唯一防御。

本课把每个 gap 都视为 iii primitive。metadata document 是一个 HTTP trigger，返回某个 function 的输出。JWKS rotation 是一个 cron trigger，它调用 `auth::rotate-jwks`，后者写入 `state::set("auth/jwks/<issuer>", ...)`。JWT validation 是其他组件通过 `iii.trigger("auth::validate-jwt", token)` 调用的 function。MCP server 本身只是另一个 HTTP trigger，在 dispatch 前调用 validation。重启 engine：trigger registry 重建；state 保留；auth surface 无需手工 reconciliation 即可运行。

## 概念

### RFC 8414：OAuth Authorization Server Metadata

位于 `/.well-known/oauth-authorization-server` 的 document 描述 client 需要的一切：

```json
{
  "issuer": "https://auth.example.com",
  "authorization_endpoint": "https://auth.example.com/authorize",
  "token_endpoint": "https://auth.example.com/token",
  "jwks_uri": "https://auth.example.com/.well-known/jwks.json",
  "registration_endpoint": "https://auth.example.com/register",
  "response_types_supported": ["code"],
  "grant_types_supported": ["authorization_code", "refresh_token"],
  "code_challenge_methods_supported": ["S256"],
  "scopes_supported": ["mcp:tools.read", "mcp:tools.invoke"],
  "token_endpoint_auth_methods_supported": ["none", "private_key_jwt"]
}
```

给定 MCP resource URL 的 client 会链式 discovery：RFC 9728 的 `oauth-protected-resource`（resource server document）命名 issuer，然后 `oauth-authorization-server`（本 RFC）命名所有 endpoint。client 永远不硬编码 authorization URL。

在把 IdP 交给 MCP 使用前，你需要验证的合约：

- `code_challenge_methods_supported` 包含 `S256`（RFC 7636 的 PKCE）。
- `grant_types_supported` 包含 `authorization_code`，并拒绝 `password` 和 `implicit`。
- `registration_endpoint` 存在（支持 RFC 7591）。
- 对 OAuth 2.1 来说，`response_types_supported` 正好是 `["code"]`。

如果缺少任何一项，MCP server 会拒绝用这个 IdP 部署。错的是 deployment manifest，不是代码。

### RFC 9728（回顾）：Protected Resource Metadata

第 16 课覆盖了 RFC 9728。生产中的差异：这个 document 是 client 查找 *此* MCP server 信任哪些 authorization servers 的唯一地点。一个 MCP server 可以接受多个 IdP 的 token（一个给员工，一个给合作伙伴）。RFC 9728 声明这个集合；RFC 8414 记录每个 IdP 支持什么。

```json
{
  "resource": "https://notes.example.com",
  "authorization_servers": ["https://auth.example.com", "https://partners.example.com"],
  "scopes_supported": ["mcp:tools.invoke"],
  "bearer_methods_supported": ["header"],
  "resource_documentation": "https://notes.example.com/docs"
}
```

### RFC 7591：Dynamic Client Registration

没有 DCR 时，每个 MCP client（Cursor、Claude Desktop、自定义 agent）都需要和 IdP admin 做一次 out-of-band exchange。使用 DCR，client 会 post：

```json
POST /register
Content-Type: application/json

{
  "redirect_uris": ["http://127.0.0.1:7333/callback"],
  "grant_types": ["authorization_code", "refresh_token"],
  "response_types": ["code"],
  "token_endpoint_auth_method": "none",
  "scope": "mcp:tools.invoke",
  "client_name": "Cursor",
  "software_id": "com.cursor.cursor",
  "software_version": "0.42.0"
}
```

server 返回 `client_id` 和用于后续更新的 `registration_access_token`：

```json
{
  "client_id": "c_3e7f1a",
  "client_id_issued_at": 1769472000,
  "redirect_uris": ["http://127.0.0.1:7333/callback"],
  "grant_types": ["authorization_code", "refresh_token"],
  "registration_access_token": "regt_b2...",
  "registration_client_uri": "https://auth.example.com/register/c_3e7f1a"
}
```

`token_endpoint_auth_method: none` 是运行在用户设备上的 MCP clients 的正确默认值。它们只拿到 `client_id`——没有可被 exfiltrate 的 `client_secret`。PKCE 提供 public clients 所需的 proof-of-possession。

三个生产坑：

- registration endpoint 必须按 source IP 做 rate-limit。没有它，敌意 actor 可以脚本化数百万个 fake registrations，耗尽 `client_id` namespace。iii 让这很简单：registration HTTP trigger 在 dispatch 给 registrar 前调用 `auth::rate-limit` function。
- 某些 enterprise IdPs 要求 `software_statement`（一个为 client 背书的 signed JWT）。本课 mock 跳过它；生产要接入 verification step，拒绝除 localhost redirect URIs 以外的 unsigned registrations。
- `registration_access_token` 必须以 hash 存储，而不是 plaintext。这个 token 被盗意味着攻击者可以重写 client redirect URIs。

### RFC 8707（回顾）：Resource Indicators

第 16 课建立了形状。生产规则：每个 token request 都包含 `resource=<canonical-mcp-url>`，MCP server 在每个 call 上验证 `token.aud` 匹配自己的 resource URL。如果 MCP server 可通过 `https://notes.example.com/mcp` 访问，canonical URL 是 `https://notes.example.com`——排除 path component，让单个 server 可以在同一 audience 下托管多个 paths。

### RFC 7636（回顾）：PKCE

PKCE 在 OAuth 2.1 中是强制的。本课的 authorization-code flow 总是携带 `code_challenge` 和 `code_verifier`。server 会拒绝任何没有 verifier，或 verifier 无法 hash 到已存 challenge 的 token request。

### MCP Spec 2025-11-25 Auth Profile

MCP spec（2025-11-25）精确规定了 MCP server 的 authorization layer 必须做什么：

- 发布 `/.well-known/oauth-protected-resource`（RFC 9728）。
- 只通过 `Authorization: Bearer ...` 接受 tokens。
- 每个 request 验证 `aud`、`iss`、`exp` 和 required scopes。
- 对每个 401 和 403 返回带 `Bearer error=...` 的 `WWW-Authenticate`，必要时包含 `scope=` 和 `resource=` parameters。
- 拒绝 `aud` 不匹配 canonical resource 的 token。
- 拒绝 `iss` 不在 protected-resource metadata 的 `authorization_servers` list 中的 token。

OAuth 2.1 draft 是 substrate；RFC 8414/7591/8707/9728 + RFC 7636 是 surface；MCP spec 是 profile。

### IdP capability matrix

不是每个 IdP 都支持完整 MCP profile。下面的矩阵记录截至 2025-11-25 spec 的事实能力声明。它是 *deployment gate*，不是推荐。

| IdP category | RFC 8414 metadata | RFC 7591 DCR | RFC 8707 resource | RFC 7636 S256 PKCE | Notes |
|---|---|---|---|---|---|
| Self-hosted (Keycloak) | yes | yes | yes (since 24.x) | yes | 本课 MCP profile 的 reference IdP；端到端支持每个 RFC。 |
| Enterprise SSO (Microsoft Entra ID) | yes | yes (premium tiers) | yes | yes | DCR availability 因 tenant tier 而异；部署前在目标 tenant 验证。 |
| Enterprise SSO (Okta) | yes | yes (Okta CIC / Auth0) | yes | yes | DCR 在 Auth0（现 Okta CIC）上可用；classic Okta orgs 需要管理员预注册。 |
| Social login IdPs (generic) | varies | rarely | rarely | yes | 大多数 social IdPs 把 clients 视为 static partners；不要依赖 DCR。只把它们作为 identity source，在上层叠加自己的 MCP-aware authorization server。 |
| Custom / homegrown | depends | depends | depends | depends | 如果自己交付，就交付完整 profile。跳过上述四个 RFC 中任意一个，都会破坏 MCP auth contract。 |

deployment manifest 的拒绝规则：如果选定 IdP 没有返回 `registration_endpoint`，并且没有在 `code_challenge_methods_supported` 中列出 `S256`，MCP server 拒绝启动。没有 degraded mode。

### JWKS rotation pattern with iii

生产失败模式是 stale JWKS cache。用 cron trigger 和 `state::*` cache 解决：

```python
iii.registerTrigger(
    "cron",
    {"schedule": "0 */6 * * *", "name": "auth::jwks-refresh"},
    "auth::rotate-jwks",
)
```

每六小时，cron trigger 调用 `auth::rotate-jwks`，后者 fetch `<issuer>/.well-known/jwks.json`，并写入 `state::set("auth/jwks/<issuer>", {keys, fetched_at})`。validator 从 `state::get` 读取。如果某个 token 的 `kid` 在 cache 中缺失，就同步调用一次 `auth::rotate-jwks` 作为 fallback。这同时处理两个 case：计划 rotation（cron）和 key-overlap windows（synchronous fallback）。

state shape：

```json
{
  "auth/jwks/https://auth.example.com": {
    "keys": [
      {"kid": "k_2026_03", "kty": "RSA", "n": "...", "e": "AQAB", "alg": "RS256", "use": "sig"},
      {"kid": "k_2026_04", "kty": "RSA", "n": "...", "e": "AQAB", "alg": "RS256", "use": "sig"}
    ],
    "fetched_at": 1772668800
  }
}
```

稳态下会同时有两个 keys。Authorization servers 通过在 retire 旧 key（`k_2026_03`）前引入 next key（`k_2026_04`）进行 rotation，所以旧 key 下发行的 tokens 在过期前仍有效。cache 保存 union；validator 按 `kid` 选择。

### iii primitive wiring（本课真正要讲的部分）

五个 primitives 组成 auth surface：

```python
# 1. RFC 8414 metadata document
iii.registerTrigger(
    "http",
    {"path": "/.well-known/oauth-authorization-server", "method": "GET"},
    "auth::serve-asm",
)

# 2. RFC 7591 dynamic client registration
iii.registerTrigger(
    "http",
    {"path": "/register", "method": "POST"},
    "auth::register-client",
)

# 3. JWT validation as a callable function (the resource server triggers it)
iii.registerFunction("auth::validate-jwt", validate_jwt_handler)

# 4. Step-up issuance for incremental scope (SEP-835 from L16)
iii.registerFunction("auth::issue-step-up", issue_step_up_handler)

# 5. Cron-driven JWKS rotation
iii.registerTrigger(
    "cron",
    {"schedule": "0 */6 * * *"},
    "auth::rotate-jwks",
)
iii.registerFunction("auth::rotate-jwks", rotate_jwks_handler)
```

MCP server 本身永远不直接调用 validation。它会：

```python
result = iii.trigger("auth::validate-jwt", {"token": bearer_token, "resource": self.resource})
if not result["valid"]:
    return {"status": 401, "WWW-Authenticate": result["www_authenticate"]}
```

这种间接层就是 iii 的赌注。明天你把 validator 换成 fanout，同时 consult 两个 IdP；或添加 span emitter；或 cache positive validations。MCP server 不需要改变。

### Confused-deputy walkthrough with audience binding

Server A（`notes.example.com`）和 Server B（`tasks.example.com`）都向同一个 authorization server 注册。Server A 被攻陷。攻击者拿到用户的 notes token，并 replay 到 Server B。

Server B 的 validator：

1. Decode JWT，按 `kid` fetch JWKS，验证 signature。
2. 根据自己的 protected-resource metadata 的 `authorization_servers` 检查 `iss`。（通过——同一个 IdP。）
3. 检查 `aud == "https://tasks.example.com"`。（失败——token 的 `aud` 是 `https://notes.example.com`。）
4. 返回 401，带 `WWW-Authenticate: Bearer error="invalid_token", error_description="audience mismatch"`。

audience claim 是协议层面对这个攻击的唯一防御。为性能跳过它是最常见的生产错误；validator 必须在每个 request 上运行，而不是只在 session start 运行。

### Failure modes

- **Stale JWKS。** key rotation 后，validator 拒绝有效 tokens。修复是上面的 cron+fallback pattern。永远不要在没有 refresh job 的情况下 cache JWKS。
- **Missing `aud` claim。** 某些 IdP 默认不包含 `aud`，除非 token request 中有 `resource`。validator 必须拒绝缺失 `aud` 的 token，不能把缺失视为 wildcard。
- **Scope upgrade race。** 同一用户的两个并发 step-up flows 都可能成功，并产生两个 scope 不同的 access tokens。validator 必须使用 request 上呈递的 token，而不是查找“用户当前 scope”——否则会产生 TOCTOU window。
- **Registration token theft。** 泄露的 `registration_access_token` 让攻击者可以重写 redirect URIs。以 hash 形式存储；每次 update 要求 client 提供明文；有嫌疑时 rotate。
- **`iss` not pinned。** 接受任意 `iss` 的 validator 允许攻击者搭建自己的 authorization server，为目标 audience 注册 client 并发行 tokens。protected-resource metadata 的 `authorization_servers` list 是 allow-list；必须执行。

## 使用它

`code/main.py` 用 stdlib Python 和一个小型 `iii_mock` registry 走完整生产 flow，mock 了 `iii.registerFunction`、`iii.registerTrigger`、`iii.trigger` 和 `state::set/get`。flow：

1. Authorization server 在 `/.well-known/oauth-authorization-server` 发布 RFC 8414 metadata。
2. MCP client 调用 metadata endpoint，发现 registration endpoint。
3. MCP client POST 到 `/register`（RFC 7591），收到 `client_id`。
4. MCP client 使用 `resource` indicator（RFC 8707）运行 PKCE-protected authorization code flow（RFC 7636）。
5. MCP client 带 `Authorization: Bearer ...` 调用 MCP server 上的工具。
6. MCP server 触发 `auth::validate-jwt`，后者从 `state::get` 读取 JWKS。
7. cron trigger 触发 `auth::rotate-jwks`，替换 state 中的 JWKS。
8. 下一个 call 无需 restart，就能针对新 keys 验证。
9. 对另一个 MCP resource 的 confused-deputy attempt 会因 audience mismatch 收到 401。

这里的 mock JWT 使用 HS256 和 shared secret（这样本课只用 stdlib 就能运行）。生产使用 RS256 或 EdDSA，并配合上面的 JWKS pattern；validation logic 其他部分完全相同。

## 交付它

本课产出 `outputs/skill-mcp-auth-iii.md`。给定一个 MCP server config 和 IdP capability set，这个 skill 会输出需要注册的 iii primitives、JWKS rotation schedule、scope mapping，以及当 IdP 不支持完整 RFC profile 时要应用的 refusal rules。

## 练习

1. 运行 `code/main.py`。追踪 9-step flow。注意 `state::get` 在 `auth::rotate-jwks` 覆盖它前返回 stale data 的位置，以及下一次 request 如何针对新 key 验证通过。

2. 向 protected-resource metadata 的 `authorization_servers` list 添加一个新 IdP。发行一个由新 IdP 签名的 token，并确认 validator 接受。发行一个由未列出 IdP 签名的 token，并确认 validator 拒绝，带 `WWW-Authenticate: Bearer error="invalid_token", error_description="iss not allowed"`。

3. 把 `auth::rate-limit` 实现为 iii function，并在 registration HTTP trigger 内部、registrar 运行前调用它。使用保存在 `state::set("auth/ratelimit/<ip>", ...)` 中的 per-source-IP token-bucket。

4. 阅读 RFC 7591，找出本课 `/register` handler 没有验证的两个字段。添加验证。（提示：`software_statement` 和 `redirect_uris` URI scheme。）

5. 阅读 MCP spec 2025-11-25 authorization section。找出关于 `WWW-Authenticate` headers 的一个 normative requirement，是本课 validator 当前没有发出的。把它添加进去。

## 关键词

| Term | 大家常说 | 实际含义 |
|------|----------|----------|
| ASM | “OAuth metadata document” | RFC 8414 `/.well-known/oauth-authorization-server` JSON |
| DCR | “Self-service client registration” | RFC 7591 `POST /register` flow |
| JWKS | “Public keys for JWT validation” | 从 `jwks_uri` fetch、按 `kid` 索引的 JSON Web Key Set |
| Resource indicator | “Audience parameter” | RFC 8707 `resource` parameter，把 token pin 到一个 server |
| `aud` claim | “Audience” | validator 与 canonical resource URL 比较的 JWT claim |
| Confused deputy | “Token replay” | 为 Server A 发行的 token 被呈递给 Server B 的攻击 |
| `iss` allow-list | “Trusted authorization servers” | protected-resource metadata 的 `authorization_servers` 命名集合 |
| Key rotation | “Rolling JWKS” | 带 overlap windows 的 signing keys 周期替换 |
| Public client | “Native or browser client” | 没有 `client_secret` 的 OAuth client；PKCE 进行补偿 |
| `WWW-Authenticate` | “401/403 response header” | 携带驱动 client recovery 的 `Bearer error=...` directives |

## 延伸阅读

- [MCP — Authorization spec (2025-11-25)](https://modelcontextprotocol.io/specification/draft/basic/authorization) — 本课实现的 MCP auth profile
- [RFC 8414 — OAuth 2.0 Authorization Server Metadata](https://datatracker.ietf.org/doc/html/rfc8414) — discovery contract
- [RFC 7591 — OAuth 2.0 Dynamic Client Registration Protocol](https://datatracker.ietf.org/doc/html/rfc7591) — DCR
- [RFC 7636 — Proof Key for Code Exchange (PKCE)](https://datatracker.ietf.org/doc/html/rfc7636) — public-client proof-of-possession
- [RFC 8707 — Resource Indicators for OAuth 2.0](https://datatracker.ietf.org/doc/html/rfc8707) — audience pinning
- [RFC 9728 — OAuth 2.0 Protected Resource Metadata](https://datatracker.ietf.org/doc/html/rfc9728) — resource server discovery
- [OAuth 2.1 draft](https://datatracker.ietf.org/doc/html/draft-ietf-oauth-v2-1) — consolidated OAuth substrate
