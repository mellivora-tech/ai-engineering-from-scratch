# Async Tasks（SEP-1686）：Call-Now, Fetch-Later 的长任务模式

> 真实 agent 工作会持续数分钟到数小时：CI runs、deep-research synthesis、batch exports。同步 tool calls 会掉连接、超时或阻塞 UI。SEP-1686 在 2025-11-25 合入，添加 Tasks primitive：任何 request 都可以增强为 task，结果可以稍后获取，或通过 state notifications streaming。漂移风险提示：Tasks 在 2026 H1 仍是实验性的；SDK surface 仍围绕 spec 设计中。

**类型：** 构建
**语言：** Python（stdlib，async task state machine）
**前置要求：** 阶段 13 · 07（MCP server），阶段 13 · 09（transports）
**时间：** ~75 分钟

## 学习目标

- 判断何时应把工具从同步提升为 task-augmented（server-side work >30 秒）。
- 走通 task lifecycle：`working` → `input_required` → `completed` / `failed` / `cancelled`。
- 持久化 task state，让 crash 不会丢失 in-flight work。
- 正确 poll `tasks/status` 并 fetch `tasks/result`。

## 问题

一个 `generate_report` tool 运行 multi-minute extraction pipeline。在同步模型下有几种选项：

1. 让连接保持三分钟。Remote transports 会断；clients 会 timeout；UI 会 freeze。
2. 立刻返回 placeholder；要求 client poll 一个 custom endpoint。破坏 MCP uniformity。
3. fire-and-forget；没有结果。

都不好。SEP-1686 添加了第四种：task augmentation。任何 request（通常是 `tools/call`）都可以被标记为 task。server 立即返回 task id。client poll `tasks/status`，完成后 fetch `tasks/result`。server-side state 在 restart 后仍存活。

## 概念

### Task augmentation

通过设置 `params._meta.task.required: true`（或 `optional: true`，由 server 决定），一个 request 会变成 task。server 立即响应：

```json
{
  "jsonrpc": "2.0", "id": 1,
  "result": {
    "_meta": {
      "task": {
        "id": "tsk_9f7b...",
        "state": "working",
        "ttl": 900000
      }
    }
  }
}
```

`ttl` 是 server 保留 state 的承诺；超过 ttl 后 task result 会被丢弃。

### Per-tool opt-in

Tool annotations 可以声明 task support：

- `taskSupport: "forbidden"` —— 该工具总是同步运行。适合 fast tools。
- `taskSupport: "optional"` —— client 可以请求 task-augmentation。
- `taskSupport: "required"` —— client 必须使用 task augmentation。

`generate_report` tool 会是 `required`。`notes_search` tool 会是 `forbidden`。

### States

```
working  -> input_required -> working  (loop via elicitation)
working  -> completed
working  -> failed
working  -> cancelled
```

State machine 是 append-only：一旦进入 `completed`、`failed` 或 `cancelled`，task 就是 terminal。

### Methods

- `tasks/status {taskId}` —— 返回当前 state 和 progress hint。
- `tasks/result {taskId}` —— block，或在尚未完成时返回 404。
- `tasks/cancel {taskId}` —— idempotent；terminal states 会忽略。
- `tasks/list` —— optional；枚举 active 和 recently-completed tasks。

### Streaming state changes

server 支持时，client 可以订阅 state notifications：

```
server -> notifications/tasks/updated {taskId, state, progress?}
```

使用 stream 而不是 poll 的 client 会获得更好的 UX。Polling 永远是最小 surface。

### Durable state

spec 要求声明 task support 的 server 持久化 state。crash 不应丢失 ttl 内的 completed results。store 可以是 SQLite、Redis 或 filesystem。本课 harness 使用 filesystem。

### Cancellation semantics

`tasks/cancel` 是 idempotent。如果 task 正在执行，server 会尝试停止（检查 executor-cooperative cancellation）。如果已经 terminal，请求是 no-op。

### Crash recovery

server process restart 时：

1. 加载所有 persisted task states。
2. 把进程死亡时仍处于 `working` 的 tasks 标记为 `failed`，error 为 `CRASH_RECOVERY`。
3. 在 ttl 内保留 `completed` / `failed` / `cancelled`。

### Async tasks plus sampling

task 本身可以调用 `sampling/createMessage`。long-running research tasks 就是这样工作：server 的 task thread 按需 sample client 模型，同时 client UI 用 periodic progress updates 展示 task 为 `working`。

### 为什么这是 experimental

SEP-1686 已在 2025-11-25 发布，但更广泛的 roadmap 指出三个 open issues：durable subscription primitives、subtasks（parent-child task relationships）和 result-TTL standardization。预计 spec 会在 2026 年继续演进。生产代码应只把 common case 视为稳定，并防备未来 SDK 对 subtasks 的变化。

## 使用它

`code/main.py` 实现一个 durable task store（filesystem-backed）和一个在 background thread 中运行的 `generate_report` tool。client 调用工具，立即获得 task id，在 worker 更新 progress 时 poll `tasks/status`，完成后 fetch `tasks/result`。Cancellation 可用；crash recovery 通过杀 worker thread 并重新加载 state 来模拟。

重点看：

- Task state JSON 持久化到 `/tmp/lesson-13-tasks/<id>.json`。
- Worker thread 更新 `progress` 字段；poll 会显示它前进。
- client-side cancellation 设置 event；worker 检查并提前退出。
- “crash” 后重新加载 state，会把 in-flight task 标记为带 `CRASH_RECOVERY` 的 `failed`。

## 交付它

本课产出 `outputs/skill-task-store-designer.md`。给定一个 long-running tool（research、build、export），这个 skill 会设计 task store（state shape、ttl、durability），选择合适的 taskSupport flag，并草拟 progress notifications。

## 练习

1. 运行 `code/main.py`。启动一个 `generate_report` task，poll status，然后 fetch result。

2. 在 mid-run 添加一次 `tasks/cancel` call。验证 worker 会遵守它，并且 state 变为 `cancelled`。

3. 模拟 crash recovery：杀掉 worker thread，重启 loader，并观察 `CRASH_RECOVERY` failure mode。

4. 把 store 扩展到 SQLite。durability 收益相同；query options 会打开（列出 session X 的所有 tasks）。

5. 阅读 2026 年 MCP roadmap post。找出最可能在下一年影响 SDK API design 的一个 Tasks-related open issue。

## 关键词

| Term | 大家常说 | 实际含义 |
|------|----------|----------|
| Task | “Long-running tool call” | 用 `_meta.task` 增强、异步执行的 request |
| SEP-1686 | “Tasks spec” | 在 2025-11-25 添加 Tasks 的 Spec Evolution Proposal |
| `_meta.task` | “Task envelope” | 包含 id、state、ttl 的 per-request metadata |
| taskSupport | “Tool flag” | 每个 tool 的 `forbidden` / `optional` / `required` |
| `tasks/status` | “Poll method” | 获取当前 state 和可选 progress hint |
| `tasks/result` | “Fetch result” | 返回 completed payload，尚未完成则 404 |
| `tasks/cancel` | “Stop it” | idempotent cancellation request |
| ttl | “Retention budget” | server 承诺保留 task state 的毫秒数 |
| `notifications/tasks/updated` | “State push” | server-initiated state-change event |
| Durable store | “Crash-safe state” | Filesystem / SQLite / Redis persistence layer |

## 延伸阅读

- [MCP — GitHub SEP-1686 issue](https://github.com/modelcontextprotocol/modelcontextprotocol/issues/1686) — 原始 proposal 和完整讨论
- [WorkOS — MCP async tasks for AI agent workflows](https://workos.com/blog/mcp-async-tasks-ai-agent-workflows) — 带 rationale 的设计 walkthrough
- [DeepWiki — MCP task system and async operations](https://deepwiki.com/modelcontextprotocol/modelcontextprotocol/2.7-task-system-and-async-operations) — mechanics 和 state machine
- [FastMCP — Tasks](https://gofastmcp.com/servers/tasks) — SDK-level task implementation patterns
- [MCP blog — 2026 roadmap](https://blog.modelcontextprotocol.io/posts/2026-mcp-roadmap/) — 包括 subtasks 在内的 open issues 和 2026 priorities
