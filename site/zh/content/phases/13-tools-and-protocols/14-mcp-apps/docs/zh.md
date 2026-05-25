# MCP Apps：通过 `ui://` 提供交互式 UI Resources

> 纯文本 tool output 限制了 agent 能展示的内容。MCP Apps（SEP-1724，2026 年 1 月 26 日 official）让工具返回 sandboxed interactive HTML，并在 Claude Desktop、ChatGPT、Cursor、Goose 和 VS Code 中 inline 渲染。dashboards、forms、maps、3D scenes，都通过一个 extension 完成。本课会讲 `ui://` resource scheme、`text/html;profile=mcp-app` MIME、iframe-sandbox postMessage protocol，以及允许 server 渲染 HTML 带来的安全面。

**类型：** 构建
**语言：** Python（stdlib，UI resource emitter），HTML（sample app）
**前置要求：** 阶段 13 · 07（MCP server），阶段 13 · 10（resources）
**时间：** ~75 分钟

## 学习目标

- 从 tool call 返回 `ui://` resource，并设置正确 MIME 和 metadata。
- 用 `_meta.ui.resourceUri`、`_meta.ui.csp` 和 `_meta.ui.permissions` 声明工具关联 UI。
- 实现用于 UI-to-host communication 的 iframe sandbox postMessage JSON-RPC。
- 应用 CSP 和 permissions-policy defaults，防御 UI-originated attacks。

## 问题

一个 2025-era 的 `visualize_timeline` tool 可以返回 “Here are 14 notes organized chronologically: ...”。那是一段文字。但用户真正想要的是 interactive timeline。MCP Apps 之前，选项是：client-specific widget APIs（Claude artifacts、OpenAI Custom GPT HTML），或者根本没有 UI。

MCP Apps（SEP-1724，2026 年 1 月 26 日发布）标准化了这份合约。tool result 包含一个 `resource`，URI 是 `ui://...`，MIME 是 `text/html;profile=mcp-app`。host 在 sandboxed iframe 中渲染它，配有受限 CSP，并且除非显式授权，否则没有网络访问。iframe 内的 UI 通过一个很小的 postMessage JSON-RPC dialect 向 host 发消息。

每个兼容 client（Claude Desktop、ChatGPT、Goose、VS Code）都会以同样方式渲染同一个 `ui://` resource。一个 server、一个 HTML bundle、通用 UI。

## 概念

### `ui://` resource scheme

一个工具返回：

```json
{
  "content": [
    {"type": "text", "text": "Here is your notes timeline:"},
    {"type": "ui_resource", "uri": "ui://notes/timeline"}
  ],
  "_meta": {
    "ui": {
      "resourceUri": "ui://notes/timeline",
      "csp": {
        "defaultSrc": "'self'",
        "scriptSrc": "'self' 'unsafe-inline'",
        "connectSrc": "'self'"
      },
      "permissions": []
    }
  }
}
```

host 随后对 `ui://notes/timeline` URI 调用 `resources/read`，并得到：

```json
{
  "contents": [{
    "uri": "ui://notes/timeline",
    "mimeType": "text/html;profile=mcp-app",
    "text": "<!doctype html>..."
  }]
}
```

### Iframe sandbox

host 在 sandboxed `<iframe>` 中渲染 HTML：

- `sandbox="allow-scripts allow-same-origin"`（或按 server declaration 更严格）
- 通过 response headers 应用 server-declared CSP。
- 没有 host origin 的 cookies，也没有 localStorage。
- 网络访问限制在 CSP 的 `connectSrc` 内。

### postMessage protocol

iframe 通过 `window.postMessage` 和 host 通信。一个很小的 JSON-RPC 2.0 dialect：

始终把 `targetOrigin` pin 到 peer 的精确 origin，并在接收侧处理任何 payload 前，根据 allowlist 验证 `event.origin`。这条 channel 两侧都绝不要用 `"*"`——body 会携带 tool calls 和 resource reads。

```js
// iframe to host  (pin to host origin)
window.parent.postMessage({
  jsonrpc: "2.0",
  id: 1,
  method: "host.callTool",
  params: { name: "notes_update", arguments: { id: "note-14", title: "..." } }
}, "https://host.example.com");

// host to iframe  (pin to iframe origin)
iframe.contentWindow.postMessage({
  jsonrpc: "2.0",
  id: 1,
  result: { content: [...] }
}, "https://iframe.example.com");

// receiver on both sides
window.addEventListener("message", (event) => {
  if (event.origin !== "https://expected-peer.example.com") return;
  // safe to process event.data
});
```

UI 可以调用的 host-side methods：

- `host.callTool(name, arguments)` —— invoke server tool。
- `host.readResource(uri)` —— read MCP resource。
- `host.getPrompt(name, arguments)` —— fetch prompt template。
- `host.close()` —— dismiss UI。

每个 call 仍然经过 MCP protocol，并继承 server permissions。

### Permissions

`_meta.ui.permissions` list 请求额外能力：

- `camera` —— 访问用户摄像头（用于 scan-a-document UIs）。
- `microphone` —— voice input。
- `geolocation` —— location。
- `network:*` —— 比 `connectSrc` 单独允许范围更宽的网络访问。

每个 permission 都会成为用户在 UI 渲染前看到的 prompt。

### Security risks

iframe 中的 HTML 仍然是 HTML。新的攻击面：

- **Prompt-injection via UI。** 恶意 server UI 可以显示看起来像 system message 的文本，诱骗用户。host rendering 应清楚区分 server UI 和 host UI。
- **Exfiltration via `connectSrc`。** 如果 CSP 允许 `connect-src: *`，UI 可以把数据发送到任何地方。默认应严格。
- **Clickjacking。** UI 覆盖 host chrome。hosts 必须阻止 z-index manipulation，并执行 opacity rules。
- **Steal focus。** UI 抢占键盘焦点，捕获下一条消息。hosts 必须拦截。

阶段 13 · 15 会把这些作为 MCP security 的一部分深入讲；本课只是引入它们。

### `ui/initialize` handshake

iframe 加载后，会通过 postMessage 发送 `ui/initialize`：

```json
{"jsonrpc": "2.0", "id": 0, "method": "ui/initialize",
 "params": {"theme": "dark", "locale": "en-US", "sessionId": "..."}}
```

host 返回 capabilities 和 session token。UI 在后续每次 host call 上都使用 session token。

### AppRenderer / AppFrame SDK primitives

ext-apps SDK 暴露两个 convenience primitives：

- `AppRenderer`（server side）—— 包装 React / Vue / Solid component，并发出带正确 MIME 和 metadata 的 `ui://` resource。
- `AppFrame`（client side）—— 接收 resource、mount iframe，并协调 postMessage。

你可以使用这些，也可以手写 HTML 和 JSON-RPC。

### Ecosystem status

MCP Apps 于 2026 年 1 月 26 日发布。2026 年 4 月的 client support：

- **Claude Desktop。** 2026 年 1 月起完整支持。
- **ChatGPT。** 通过 Apps SDK 完整支持（底层同一个 MCP Apps protocol）。
- **Cursor。** Beta；需在 settings 中启用。
- **VS Code。** Insider builds only。
- **Goose。** 完整支持。
- **Zed、Windsurf。** Roadmapped。

生产中的 servers：dashboards、map visualizations、data tables、chart builders、sandbox IDE previews。

## 使用它

`code/main.py` 扩展 notes server，添加一个 `visualize_timeline` tool，它返回 `ui://notes/timeline` resource；还添加一个处理该 URI 上 `resources/read` 的 handler，返回一个带 SVG timeline 的小而完整 HTML bundle。HTML 用 stdlib 模板生成——没有 build system。postMessage 在 JS comments 中草拟，因为 stdlib 不能驱动浏览器。

重点看：

- tool response 上的 `_meta.ui` 携带 resourceUri、CSP、permissions。
- HTML 在无网络访问下渲染；所有数据都 inline。
- JS 通过 `window.parent.postMessage` 调用 `host.callTool`（已文档化，但在这个 stdlib demo 中 inert）。

## 交付它

本课产出 `outputs/skill-mcp-apps-spec.md`。给定一个会受益于 interactive UI 的工具，这个 skill 会生成完整 MCP Apps contract：`ui://` URI、CSP、permissions、postMessage entrypoints 和 security checklist。

## 练习

1. 运行 `code/main.py` 并检查输出的 HTML。直接在浏览器中打开 HTML，验证 SVG 渲染。然后草拟 UI 会用来调用 `host.callTool("notes_update", ...)` 的 postMessage contract。

2. 收紧 CSP：移除 `'unsafe-inline'`，改用 nonce-based script policy。HTML generation code 需要怎么改？

3. 添加第二个 UI resource `ui://notes/editor`，包含一个就地编辑 note 的 form。用户提交时，iframe 调用 `host.callTool("notes_update", ...)`。

4. 审计 UI 的攻击面。恶意 server 可以在哪里注入内容？iframe sandbox 防御了什么，又防不住什么？

5. 阅读 SEP-1724 spec，找出 MCP Apps SDK 中一个 toy implementation 没有使用的 capability。（提示：component-level state sync。）

## 关键词

| Term | 大家常说 | 实际含义 |
|------|----------|----------|
| MCP Apps | “Interactive UI resources” | 2026-01-26 发布的 SEP-1724 extension |
| `ui://` | “App URI scheme” | UI bundle 的 resource scheme |
| `text/html;profile=mcp-app` | “MIME” | MCP App HTML 的 content-type |
| Iframe sandbox | “Render container” | 带 CSP 和 permissions 的 UI browser sandboxing |
| postMessage JSON-RPC | “UI-to-host wire” | 用于 host calls 的 tiny JSON-RPC-over-postMessage dialect |
| `_meta.ui` | “Tool-UI binding” | 把 tool result 关联到 UI resource 的 metadata |
| CSP | “Content-Security-Policy” | 声明 script、network、style 的允许来源 |
| AppRenderer | “Server SDK primitive” | 把 framework component 转成 `ui://` resource |
| AppFrame | “Client SDK primitive” | mount iframe 并协调 postMessage 的 helper |
| `ui/initialize` | “Handshake” | UI 发给 host 的第一条 postMessage |

## 延伸阅读

- [MCP ext-apps — GitHub](https://github.com/modelcontextprotocol/ext-apps) — reference implementation 和 SDK
- [MCP Apps specification 2026-01-26](https://github.com/modelcontextprotocol/ext-apps/blob/main/specification/2026-01-26/apps.mdx) — 正式 spec 文档
- [MCP — Apps extension overview](https://modelcontextprotocol.io/extensions/apps/overview) — 高层文档
- [MCP blog — MCP Apps launch](https://blog.modelcontextprotocol.io/posts/2026-01-26-mcp-apps/) — 2026 年 1 月 launch post
- [MCP Apps API reference](https://apps.extensions.modelcontextprotocol.io/api/) — JSDoc-style SDK reference
