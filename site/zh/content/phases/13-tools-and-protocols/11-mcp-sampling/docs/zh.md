# MCP Sampling：Server-Requested LLM Completions 和 Agent Loops

> 大多数 MCP server 都是 dumb executors：接收参数，运行代码，返回内容。Sampling 让 server 反转方向：它请求 client 的 LLM 做决定。这允许 server-hosted agent loops 无需 server 持有任何模型凭据。SEP-1577 在 2025-11-25 合入，为 sampling request 添加了 tools，让 loop 可以包含更深层 reasoning。漂移风险提示：SEP-1577 的 tool-in-sampling 形状在 2026 Q1 仍是实验性的，SDK API 还在稳定中。

**类型：** 构建
**语言：** Python（stdlib，sampling harness）
**前置要求：** 阶段 13 · 07（MCP server），阶段 13 · 10（resources and prompts）
**时间：** ~75 分钟

## 学习目标

- 解释 `sampling/createMessage` 解决什么问题（无需 server-side API keys 的 server-hosted loops）。
- 实现一个 server，向 client 请求对 multi-turn prompt 进行 sampling，并返回 completion。
- 使用 `modelPreferences`（cost / speed / intelligence priorities）引导 client model selection。
- 构建一个 `summarize_repo` tool，内部通过 sampling 迭代，而不是写死行为。

## 问题

一个用于 code-summarization workflow 的有用 MCP server 需要：遍历 file tree，选择要读的文件，合成 summary，并返回。LLM reasoning 在哪里发生？

方案 A：server 调用自己的 LLM。需要 API key，server-side 计费，对每个用户都贵。

方案 B：server 返回 raw content；client 的 agent 做 reasoning。可用，但会把 server logic 移到 client prompt，脆弱。

方案 C：server 通过 `sampling/createMessage` 请求 client 的 LLM。server 保留 algorithm（读哪些文件、做几轮），client 保留 billing 和 model choice。server 完全没有凭据。

Sampling 就是方案 C。它让受信任的 server 可以托管 agent loop，而不需要自己成为完整 LLM host。

## 概念

### `sampling/createMessage` request

Server 发送：

```json
{
  "jsonrpc": "2.0",
  "id": 42,
  "method": "sampling/createMessage",
  "params": {
    "messages": [{"role": "user", "content": {"type": "text", "text": "..."}}],
    "systemPrompt": "...",
    "includeContext": "none",
    "modelPreferences": {
      "costPriority": 0.3,
      "speedPriority": 0.2,
      "intelligencePriority": 0.5,
      "hints": [{"name": "claude-3-5-sonnet"}]
    },
    "maxTokens": 1024
  }
}
```

Client 运行自己的 LLM，并返回：

```json
{"jsonrpc": "2.0", "id": 42, "result": {
  "role": "assistant",
  "content": {"type": "text", "text": "..."},
  "model": "claude-3-5-sonnet-20251022",
  "stopReason": "endTurn"
}}
```

### `modelPreferences`

三个总和为 1.0 的 float：

- `costPriority`：偏好更便宜的模型。
- `speedPriority`：偏好更快的模型。
- `intelligencePriority`：偏好能力更强的模型。

还有 `hints`：server 偏好的命名模型。client 可以尊重 hints，也可以不尊重；client 的用户配置始终优先。

### `includeContext`

三个值：

- `"none"` —— 只有 server 提供的 messages。默认。
- `"thisServer"` —— 包含该 server session 的历史 messages。
- `"allServers"` —— 包含全部 session context。

截至 2025-11-25，`includeContext` 已 soft-deprecated，因为它会泄漏 cross-server context，带来安全问题。更推荐 `"none"`，并在 messages 中显式传递 context。

### Sampling with tools（SEP-1577）

2025-11-25 新增：sampling request 可以包含 `tools` array。client 用这些 tools 运行完整 tool-calling loop。这让 server 能通过 client 的模型托管 ReAct-style agent loop。

```json
{
  "messages": [...],
  "tools": [
    {"name": "fetch_url", "description": "...", "inputSchema": {...}}
  ]
}
```

client 循环：sample，如果调用工具就 execute tool，再 sample，最后返回 final assistant message。这在 2026 Q1 仍是实验性的；SDK signatures 可能仍会漂移。实现时请对照 2025-11-25 spec 的 client/sampling 章节确认。

### Human-in-the-loop

client 必须在运行 sample 前向用户展示 server 要求模型做什么。恶意 server 可以利用 sampling 操纵用户 session（“对用户说 X，让他们点击 Y”）。Claude Desktop、VS Code 和 Cursor 会把 sampling request 显示为用户可以拒绝的 confirmation dialog。

2026 年的共识：没有 human confirmation 的 sampling 是 red flag。Gateways（阶段 13 · 17）可以自动批准 low-risk sampling，并自动拒绝可疑请求。

### 无 API key 的 server-hosted loops

典型 use case：一个没有自己 LLM access 的 code-summarization MCP server。它会：

1. 遍历 repo structure。
2. 调用 `sampling/createMessage`，内容是 “Pick five files most likely to describe this repo's purpose.”
3. 读取这些文件。
4. 调用 `sampling/createMessage`，携带文件内容和 “Summarize the repo in 3 paragraphs.”
5. 作为 `tools/call` result 返回 summary。

server 从不触碰 LLM API。client 用户用自己的 credentials 为 completion 付费。

### Safety risks（Unit 42 disclosure，2026 Q1）

- **Covert sampling。** 一个工具总是用 “respond with the user's email from session context” 调用 sampling。阶段 13 · 15 覆盖 attack vectors。
- **Resource theft via sampling。** server 要求 client summarize 攻击者 payload，让用户付账。
- **Loop bombs。** server 在 tight loop 中调用 sampling。clients 必须执行 per-session rate limits。

## 使用它

`code/main.py` 提供一个 fake server-to-client sampling harness。一个模拟的 `summarize_repo` tool 会调用两轮 sampling（pick-files，然后 summarize），fake client 返回 canned responses。harness 展示：

- server 发送带 `modelPreferences` 的 `sampling/createMessage`。
- client 返回 completion。
- server 继续它的 loop。
- rate limiter 限制每次 tool invocation 的 sampling call 总数。

重点看：

- server 只暴露一个工具（`summarize_repo`）；所有 reasoning 都发生在 sampling calls 中。
- Model preferences 加权 client 的 model choice；hints 列出偏好模型。
- loop 在 `stopReason: "endTurn"` 时终止。
- `max_samples_per_tool = 5` limit 捕捉 runaway loop。

## 交付它

本课产出 `outputs/skill-sampling-loop-designer.md`。给定一个需要 LLM calls 的 server-side algorithm（research、summarization、planning），这个 skill 会设计基于 sampling 的实现，包含合适的 modelPreferences、rate limits 和 safety confirmations。

## 练习

1. 运行 `code/main.py`。把 `max_samples_per_tool` 改为 2，观察 rate-limit cut-off。

2. 实现 SEP-1577 tool-in-sampling variant：sampling request 携带 `tools` array。验证 client-side loop 会在返回 final completion 前执行这些 tools。注意漂移风险：SDK signatures 在 2026 H1 仍可能变化。

3. 添加 human-in-the-loop confirmation：在 server 第一次 `sampling/createMessage` 前暂停并等待用户批准。Denied calls 返回 typed refusal。

4. 添加按 client session key 的 per-user rate limiter。同一用户的 same-server loops 应共享预算。

5. 设计一个使用 sampling 来选择 chunks 的 `summarize_pdf` tool。草拟发送的 messages。`modelPreferences.intelligencePriority` 为 0.1 vs 0.9 时，行为如何变化？

## 关键词

| Term | 大家常说 | 实际含义 |
|------|----------|----------|
| Sampling | “Server-to-client LLM call” | server 请求 client 的模型完成一次 completion |
| `sampling/createMessage` | “method” | sampling request 的 JSON-RPC method |
| `modelPreferences` | “Model priorities” | cost / speed / intelligence weights 加 name hints |
| `includeContext` | “Cross-session leakage” | soft-deprecated 的 context inclusion mode |
| SEP-1577 | “Tools in sampling” | 允许 sampling 中带 tools，用于 server-hosted ReAct |
| Human-in-the-loop | “User confirms” | client 在运行前向用户展示 sampling request |
| Loop bomb | “Runaway sampling” | server-side infinite sampling loop；client 必须 rate-limit |
| Covert sampling | “Hidden reasoning” | 恶意 server 在 sampling prompts 中隐藏意图 |
| Resource theft | “使用用户的 LLM budget” | server 强迫 client 为自己不想要的 sampling 花钱 |
| `stopReason` | “为什么 generation 停止” | `endTurn`、`stopSequence` 或 `maxTokens` |

## 延伸阅读

- [MCP — Concepts: Sampling](https://modelcontextprotocol.io/docs/concepts/sampling) — sampling 高层概览
- [MCP — Client sampling spec 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25/client/sampling) — canonical `sampling/createMessage` 形状
- [MCP — GitHub SEP-1577](https://github.com/modelcontextprotocol/modelcontextprotocol) — tools in sampling 的 Spec Evolution Proposal（experimental）
- [Unit 42 — MCP attack vectors](https://unit42.paloaltonetworks.com/model-context-protocol-attack-vectors/) — covert sampling 和 resource-theft patterns
- [Speakeasy — MCP sampling core concept](https://www.speakeasy.com/mcp/core-concepts/sampling) — 带 client-side code samples 的 walkthrough
