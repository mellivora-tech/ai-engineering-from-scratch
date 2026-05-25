# Roots 和 Elicitation：Scoping 与 Mid-Flight User Input

> 硬编码路径会在用户打开不同项目时立刻失效。预填 tool arguments 会在用户描述不完整时失效。Roots 把 server 限定在一组由用户控制的 URI 内；elicitation 会在 tool-call 中途暂停，通过表单或 URL 向用户请求结构化输入。两个 client primitives，分别修复两类常见 MCP 失败模式。SEP-1036（URL-mode elicitation，2025-11-25）在 2026 H1 仍是实验性的——依赖它前请检查 SDK version。

**类型：** 构建
**语言：** Python（stdlib，roots + elicitation demo）
**前置要求：** 阶段 13 · 07（MCP server）
**时间：** ~45 分钟

## 学习目标

- 声明 `roots` 并响应 `notifications/roots/list_changed`。
- 把 server file operations 限制在声明 root set 内的 URI 上。
- 使用 `elicitation/create` 在 tool-call 中途向用户请求 confirmation 或 structured input。
- 在 form-mode 和 URL-mode elicitation 之间选择（后者是实验性的；已标注漂移风险）。

## 问题

notes MCP server 在生产中会碰到两个具体失败。

**Broken path assumption。** server 针对 `~/notes` 编写。另一个用户的 notes 位于 `~/Documents/Notes`，于是 tool call 静默失败（找不到文件），甚至更糟，写到了错误位置。

**用户知道但缺失的参数。** 用户说“删除旧的 TPS report note”。模型调用 `notes_delete(title: "TPS report")`，但匹配到 2023、2024 和 2025 三条 note。工具不能猜。返回 “ambiguous” 很烦；对三条都执行是灾难。

Roots 修复第一个：client 在 `initialize` 时声明 server 可触及的 URI 集合。Elicitation 修复第二个：server 暂停 tool call，并发送 `elicitation/create`，让用户选择哪一条。

## 概念

### Roots

client 在 `initialize` 时声明 root list：

```json
{
  "capabilities": {"roots": {"listChanged": true}}
}
```

Server 随后可以调用 `roots/list`：

```json
{"roots": [{"uri": "file:///Users/alice/Documents/Notes", "name": "Notes"}]}
```

Servers 必须把 roots 当作边界：任何 root set 之外的 file read 或 write 都会被拒绝。这不是由 client 强制执行的（server 仍然是用户信任的代码），但 spec-compliant servers 会遵守。

当用户添加或移除 root 时，client 发送 `notifications/roots/list_changed`。server 重新调用 `roots/list` 并更新边界。

### 为什么 roots 是 client primitive

Roots 由 client 声明，因为它代表用户的 consent model。用户告诉 Claude Desktop “允许这个 notes server 访问这两个目录”。server 不能扩大这个 scope。

### Elicitation：默认 form-mode

`elicitation/create` 接收 form schema 和自然语言 prompt：

```json
{
  "method": "elicitation/create",
  "params": {
    "message": "Delete 'TPS report'? Multiple notes match; pick one.",
    "requestedSchema": {
      "type": "object",
      "properties": {
        "note_id": {
          "type": "string",
          "enum": ["note-3", "note-7", "note-14"]
        },
        "confirm": {"type": "boolean"}
      },
      "required": ["note_id", "confirm"]
    }
  }
}
```

client 渲染表单，收集用户答案，并返回：

```json
{
  "action": "accept",
  "content": {"note_id": "note-14", "confirm": true}
}
```

三种可能 action：`accept`（用户填写了）、`decline`（用户关闭了）、`cancel`（用户中止整个 tool call）。

Form schemas 是 flat 的——v1 不支持 nested objects。SDK 通常会拒绝超过一层的复杂内容。

### Elicitation：URL mode（SEP-1036，experimental）

2025-11-25 新增。server 不发送 schema，而是发送 URL：

```json
{
  "method": "elicitation/create",
  "params": {
    "message": "Sign in to GitHub",
    "url": "https://github.com/login/oauth/authorize?client_id=..."
  }
}
```

client 在浏览器中打开 URL，等待完成，并在用户返回时响应。适用于 OAuth flows、payment authorization 和 document signing 这类表单不够用的场景。

漂移风险提示：SEP-1036 response shape 仍在稳定；有些 SDK 返回 callback URL，有些返回 completion token。生产使用 URL mode 前阅读你的 SDK release notes。

### 什么时候 elicitation 是正确工具

- destructive action 前的用户确认（destructive hint + elicitation）。
- 消歧（从 N 个 matches 中选一个）。
- 首次运行设置（API keys、directories、preferences）。
- OAuth-style flows（URL mode）。

### 什么时候 elicitation 是错的

- 填写模型本可以通过 prose 追问的 required tool arguments。用普通 re-prompt，不要弹 elicitation dialog。
- 高频调用。Elicitation 会打断对话；不要在 loop 中触发它。
- server 能在事后 validate 的任何内容。validate，返回 error，让模型用文本问用户。

### Human-in-the-loop bridge

Elicitation 加 sampling 共同实现 MCP 的 “human-in-the-loop” model。server 的 agent loop 可以为了用户输入（elicitation）或模型 reasoning（sampling）而暂停。阶段 13 · 11 覆盖了 sampling；本课覆盖 elicitation。把它们合起来，就能获得完整 mid-loop control。

## 使用它

`code/main.py` 扩展 notes server，加入：

- `roots/list` response，server 在 root-list-changed notifications 后重新查询它。
- 一个 `notes_delete` tool，在多个 notes 匹配时使用 `elicitation/create` 消歧。
- 一个 `notes_setup` tool，用 URL-mode elicitation 打开首次运行配置页（模拟）。
- 一个 boundary check，拒绝对声明 roots 之外 URI 的操作。

demo 运行三个场景：happy path（一个 match）、disambiguation（三个 matches，触发 elicitation）、out-of-root-write（被拒绝）。

## 交付它

本课产出 `outputs/skill-elicitation-form-designer.md`。给定一个可能需要用户 confirmation 或 disambiguation 的工具，这个 skill 会设计 elicitation form schema 和 message template。

## 练习

1. 运行 `code/main.py`。触发 disambiguation path；确认模拟用户答案被路由回 tool。

2. 添加一个新工具 `notes_archive`，每次都要求 elicitation confirmation（destructive hint）。检查 UX：这和模型用文本重新提问相比如何？

3. 为首次运行 OAuth flow 实现 URL-mode elicitation。标注漂移风险，并添加 SDK-version guard。

4. 扩展 `roots/list` handling：notification 到达时，server 应 atomically 重新读取并 rescan 可能已经 out of scope 的 open file handles。

5. 阅读 GitHub 上的 SEP-1036 issue discussion thread。找出一个影响 server 如何处理 URL-mode callback 的 open question。

## 关键词

| Term | 大家常说 | 实际含义 |
|------|----------|----------|
| Root | “Consent boundary” | client 允许 server 触及的 URI |
| `roots/list` | “Server asks for scope” | client 返回当前 root set |
| `notifications/roots/list_changed` | “User changed scope” | client 表示 root set 已变化 |
| Elicitation | “Ask the user mid-call” | server-initiated structured user input request |
| `elicitation/create` | “method” | elicitation request 的 JSON-RPC method |
| Form mode | “Schema-driven form” | client UI 中由 flat JSON Schema 渲染的表单 |
| URL mode | “Browser redirect” | SEP-1036 experimental；打开 URL 并等待 |
| `accept` / `decline` / `cancel` | “User response outcomes” | server 处理的三个分支 |
| Disambiguation | “Pick one” | 工具有 N 个候选时常见的 elicitation use case |
| Flat form | “Top-level properties only” | elicitation schema 不能嵌套 |

## 延伸阅读

- [MCP — Client roots spec](https://modelcontextprotocol.io/specification/draft/client/roots) — roots 权威参考
- [MCP — Client elicitation spec](https://modelcontextprotocol.io/specification/draft/client/elicitation) — elicitation 权威参考
- [Cisco — What's new in MCP elicitation, structured content, OAuth enhancements](https://blogs.cisco.com/developer/whats-new-in-mcp-elicitation-structured-content-and-oauth-enhancements) — 2025-11-25 additions walkthrough
- [MCP — GitHub SEP-1036](https://github.com/modelcontextprotocol/modelcontextprotocol) — URL-mode elicitation proposal（experimental，drift-risk）
- [The New Stack — How elicitation brings human-in-the-loop to AI tools](https://thenewstack.io/how-elicitation-in-mcp-brings-human-in-the-loop-to-ai-tools/) — UX walkthrough
