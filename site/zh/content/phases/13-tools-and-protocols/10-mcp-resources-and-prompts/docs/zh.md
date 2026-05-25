# MCP Resources 和 Prompts：工具之外的 Context Exposure

> Tools 拿走了 MCP 90% 的关注度。另外两个 server primitives 解决不同问题。Resources 暴露可读取的数据；prompts 暴露可复用模板作为 slash-commands。很多 server 应该用 resources，而不是把 read 包装成 tools；应该用 prompts，而不是把 workflow 写死在 client prompts 里。本课命名决策规则，并走通 `resources/*` 和 `prompts/*` messages。

**类型：** 构建
**语言：** Python（stdlib，resource + prompt handler）
**前置要求：** 阶段 13 · 07（MCP server）
**时间：** ~45 分钟

## 学习目标

- 针对给定 domain，决定把能力暴露为 tool、resource 还是 prompt。
- 实现 `resources/list`、`resources/read`、`resources/subscribe`，并处理 `notifications/resources/updated`。
- 实现带 argument template 的 `prompts/list` 和 `prompts/get`。
- 识别 host 何时把 prompts 显示为 slash-commands，何时自动注入 context。

## 问题

一个幼稚的 notes app MCP server 会把所有东西都暴露为 tools：`notes_read`、`notes_list`、`notes_search`。这会把每次数据访问都包成 model-driven tool call。后果：

- 模型必须为每个可能受益于 context 的 query 决定是否调用 `notes_read`。
- Read-only content 不能被 subscribed，也不能 stream 到 host 的 side panel。
- Client UI（Claude Desktop 的 resource attachment panel、Cursor 的 “Include file” picker）无法展示这些数据。

正确划分：把数据暴露为 resource，把 mutating 或 computed actions 暴露为 tools，把可复用的 multi-step workflows 暴露为 prompts。每个 primitive 都有自己的 UX affordance 和 access pattern。

## 概念

### Tools vs resources vs prompts：决策规则

| Capability | Primitive |
|------------|-----------|
| 用户想搜索、过滤或转换数据 | tool |
| 用户想让 host 把这份数据作为 context 包含进来 | resource |
| 用户想反复运行一个模板化 workflow | prompt |

Guideline：如果模型在每个相关 query 上都会受益于调用它，它是 tool。如果用户会受益于把它附加到对话，它是 resource。如果一个完整 multi-step workflow 是用户想复用的单位，它是 prompt。

### Resources

`resources/list` 返回 `{resources: [{uri, name, mimeType, description?}]}`。`resources/read` 接收 `{uri}` 并返回 `{contents: [{uri, mimeType, text | blob}]}`。

URI 可以是任何可寻址内容：

- `file:///Users/alice/notes/mcp.md`
- `postgres://my-db/query/SELECT ...`
- `notes://note-14`（custom scheme）
- `memory://session-2026-04-22/recent`（server-specific）

`contents[]` 同时支持 text 和 binary。Binary 使用 `blob` 作为 base64-encoded string，并带 `mimeType`。

### Resource subscriptions

在 capabilities 中声明 `{resources: {subscribe: true}}`。Client 调用 `resources/subscribe {uri}`。Resource 变化时，server 发送 `notifications/resources/updated {uri}`。Client 重新读取。

Use case：一个 notes server，其 resources 是磁盘文件；file watcher 触发 update notifications；Claude Desktop 在 host 外部编辑文件后重新把文件拉进 context。

### Resource templates（2025-11-25 addition）

`resourceTemplates` 允许暴露 parameterized URI pattern：`notes://{id}`，其中 `id` 是 completion target。client 可以在 resource picker 中 autocomplete ids。

### Prompts

`prompts/list` 返回 `{prompts: [{name, description, arguments?}]}`。`prompts/get` 接收 `{name, arguments}`，并返回 `{description, messages: [{role, content}]}`。

prompt 是一个模板，会填充成 host 喂给模型的一组 messages。例如，`code_review` prompt 接收 `file_path` 参数，并返回三条 message：system message、带文件内容的 user message、以及带 reasoning template 的 assistant kickoff。

### Hosts and prompts

Claude Desktop、VS Code 和 Cursor 会在 chat UI 中把 prompts 暴露为 slash-commands。用户输入 `/code_review`，并从表单中选择参数。server 的 prompt 是 “用户 shortcut” 和 “发送给模型的完整 prompt” 之间的合约。

并非每个 client 都支持 prompts——检查 capability negotiation。一个声明 prompt capability 的 server 遇到不支持 prompt 的 client，只是不会显示 slash commands。

### “list changed” notification

resources 和 prompts 都会在集合变化时发出 `notifications/list_changed`。一个刚导入 20 条新 notes 的 notes server 会发出 `notifications/resources/list_changed`；client 会重新调用 `resources/list` 来获取新增项。

### Content type conventions

文本：`mimeType: "text/plain"`、`text/markdown`、`application/json`。
二进制：`image/png`、`application/pdf`，加上 `blob` 字段。
MCP Apps（第 14 课）：`ui://` URI 中的 `text/html;profile=mcp-app`。

### Dynamic resources

resource URI 不一定对应静态文件。`notes://recent` 可以每次读取都返回最新五条 notes。`db://query/users/active` 可以执行参数化 query。server 可以自由动态计算内容。

规则：如果 client 可以按 URI cache，URI 必须稳定。如果 computation 是 one-shot，URI 应包含 timestamp 或 nonce，避免 client cache stale out。

### Subscriptions vs polling

支持 subscription 的 client 通过 `notifications/resources/updated` 获得 server push。pre-subscription clients 或不支持它的 host 通过重新读取来 polling。两者都 spec-compliant。server 的 capability declaration 会告诉 client 它支持哪种。

subscription 的成本：server 上的 per-session state（谁订阅了什么）。保持 subscribed set 有界；disconnected clients 应该 timeout。

### Prompts vs system prompts

MCP 中的 prompts 不是 system prompts。host 自己的 system prompt（它的 operating instructions）和 MCP prompts（server-supplied templates，由用户调用）并排存在。行为良好的 client 永远不会让 server prompt 覆盖自己的 system prompt；它会叠加它们。

## 使用它

`code/main.py` 基于第 07 课的 notes server 添加：

- Per-note resources（`notes://note-1` 等），并支持 `resources/subscribe`。
- 一个 `review_note` prompt，渲染成三条 message 的 template。
- 一个 file-watcher simulation，在 note 被修改时发出 `notifications/resources/updated`。
- 一个 `notes://recent` dynamic resource，总是返回最新五条 notes。

运行 demo 查看完整 flow。

## 交付它

本课产出 `outputs/skill-primitive-splitter.md`。给定一个 proposed MCP server，这个 skill 会把每个 capability 分类为 tool / resource / prompt，并给出理由。

## 练习

1. 运行 `code/main.py`。观察初始 resource list，然后触发 note edit，并验证 `notifications/resources/updated` event 发出。

2. 添加一个 `resources/list_changed` emitter：当新 note 创建时，发送 notification，让 clients 重新 discovery。

3. 为 GitHub MCP server 设计三个 prompts：`summarize_pr`、`triage_issue`、`release_notes`。每个都带 argument schemas。prompt body 应无需进一步编辑即可运行。

4. 选择第 07 课 server 中的一个现有 tool，判断它应继续作为 tool，还是拆成 resource + tool pair。用一句话说明理由。

5. 阅读 spec 的 `server/resources` 和 `server/prompts` 章节。找出 `resources/read` 中一个很少填充但 spec 支持的字段。提示：看 resource content 上的 `_meta`。

## 关键词

| Term | 大家常说 | 实际含义 |
|------|----------|----------|
| Resource | “暴露的数据” | host 可以读取的 URI-addressable content |
| Resource URI | “Pointer to data” | 带 scheme 的标识符（`file://`、`notes://` 等） |
| `resources/subscribe` | “Watch for changes” | client opt-in 的 server-push updates，针对特定 URI |
| `notifications/resources/updated` | “Resource changed” | 告诉 client 已订阅 resource 有新内容的信号 |
| Resource template | “Parameterized URI” | 带 completion hints 的 URI pattern，供 host picker 使用 |
| Prompt | “Slash-command template” | 带 argument slots 的命名 multi-message template |
| Prompt arguments | “Template inputs” | host 在渲染前收集的 typed parameters |
| `prompts/get` | “Render template” | server 返回填充后的 message list |
| Content block | “Typed chunk” | `{type: text | image | resource | ui_resource}` |
| Slash-command UX | “User shortcut” | host 将 prompts 显示为以 `/` 开头的 commands |

## 延伸阅读

- [MCP — Concepts: Resources](https://modelcontextprotocol.io/docs/concepts/resources) — resource URIs、subscriptions 和 templates
- [MCP — Concepts: Prompts](https://modelcontextprotocol.io/docs/concepts/prompts) — prompt templates 和 slash-command integration
- [MCP — Server resources spec 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25/server/resources) — 完整 `resources/*` message reference
- [MCP — Server prompts spec 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25/server/prompts) — 完整 `prompts/*` message reference
- [MCP — Protocol info site: resources](https://modelcontextprotocol.info/docs/concepts/resources/) — 扩展官方文档的社区指南
