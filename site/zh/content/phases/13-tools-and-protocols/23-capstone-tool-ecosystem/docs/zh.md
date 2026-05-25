# Capstone：构建完整工具生态系统

> 阶段 13 已经教完了每个部件。这个 capstone 会把它们接成一个 production-shaped system：带 tools + resources + prompts + tasks + UI 的 MCP server，边界上的 OAuth 2.1，RBAC gateway，multi-server client，A2A sub-agent call，写入 collector 的 OTel tracing，CI 中的 tool-poisoning detection，以及 AGENTS.md + SKILL.md bundle。到最后，你能为每个架构选择辩护。

**类型：** 构建
**语言：** Python（stdlib，end-to-end ecosystem harness）
**前置要求：** 阶段 13 · 01 到 21
**时间：** ~120 分钟

## 学习目标

- 组合一个暴露 tools、resources、prompts 和带 `ui://` app 的 task 的 MCP server。
- 用执行 RBAC 和 pinned hashes 的 OAuth 2.1 gateway 前置 server。
- 编写一个 multi-server client，用 OTel GenAI attributes 做端到端 tracing。
- 把 workload 的一部分委派给 A2A sub-agent；验证 opacity 被保留。
- 用 AGENTS.md + SKILL.md 打包整个 stack，让其他 agents 可以驱动它。

## 问题

交付 “research and report” system：

- 用户问：“summarize the three most-cited 2026 arXiv papers on agent protocols.”
- 系统：通过 MCP 搜索 arXiv；通过 A2A 把 paper summarization 委派给专门 writer agent；聚合结果；把 interactive report 渲染为 MCP Apps `ui://` resource；把每一步记录到 OTel。

阶段 13 的所有 primitives 都会出现。这不是 toy——Anthropic（Claude Research product）、OpenAI（GPTs with Apps SDK）以及第三方在 2026 年发布的生产 research-assistant systems 都是这个形状。

## 概念

### Architecture

```
[user] -> [client] -> [gateway (OAuth 2.1 + RBAC)] -> [research MCP server]
                                                      |
                                                      +- MCP tool: arxiv_search (pure)
                                                      +- MCP resource: notes://recent
                                                      +- MCP prompt: /research_topic
                                                      +- MCP task: generate_report (long)
                                                      +- MCP Apps UI: ui://report/current
                                                      +- A2A call: writer-agent (tasks/send)
                                                      |
                                                      +- OTel GenAI spans
```

### Trace hierarchy

```
agent.invoke_agent
 ├── llm.chat (kick off)
 ├── mcp.call -> tools/call arxiv_search
 ├── mcp.call -> resources/read notes://recent
 ├── mcp.call -> prompts/get research_topic
 ├── a2a.tasks/send -> writer-agent
 │    └── task transitions (opaque internals)
 ├── mcp.call -> tools/call generate_report (task-augmented)
 │    └── tasks/status polling
 │    └── tasks/result (completed, returns ui:// resource)
 └── llm.chat (final synthesis)
```

一个 trace id。每个 span 都有正确的 `gen_ai.*` attributes。

### Security posture

- OAuth 2.1 + PKCE，resource indicator 把 audience pin 到 gateway。
- Gateway 持有 upstream credentials；用户永远看不到。
- RBAC：`alice` 拥有 `research:read`、`research:write`，可以调用所有 tools。`bob` 拥有 `research:read`，不能调用 `generate_report`。
- pinned description manifest：丢弃任何 tool hash 变化的 server。
- Rule of Two audit：没有 tool 同时组合 untrusted input、sensitive data 和 consequential action。

### Rendering

最终 `generate_report` task 返回 content blocks 加一个 `ui://report/current` resource。client 的 host（Claude Desktop 等）在 sandbox iframe 中渲染 interactive dashboard。dashboard 包含排序后的 paper list、citation counts，以及一个按钮，用户点击任意 paper 时调用 `host.callTool('summarize_paper', {arxiv_id})`。

### Packaging

整个系统按如下方式交付：

```
research-system/
  AGENTS.md                     # project conventions
  skills/
    run-research/
      SKILL.md                  # the top-level workflow
  servers/
    research-mcp/               # the MCP server
      pyproject.toml
      src/
  agents/
    writer/                     # the A2A agent
  gateway/
    config.yaml                 # RBAC + pinned manifest
```

用户用 `docker compose up` 部署。Claude Code、Cursor、Codex 和 opencode 用户可以通过调用 `run-research` skill 来驱动系统。

### 每个阶段 13 课程贡献了什么

| Lesson | What the capstone uses |
|--------|------------------------|
| 01-05 | Tool interface、provider-portability、parallel calls、schemas、linting |
| 06-10 | MCP primitives、server、client、transports、resources + prompts |
| 11-14 | Sampling、roots + elicitation、async tasks、`ui://` apps |
| 15-17 | Tool poisoning、OAuth 2.1、gateway + registry |
| 18 | A2A sub-agent delegation |
| 19 | OTel GenAI tracing |
| 20 | Routing gateway for the LLM layer |
| 21 | SKILL.md + AGENTS.md packaging |

## 使用它

`code/main.py` 把前面课程的模式接成一个 runnable demo。全部 stdlib，全部 in-process，所以你可以端到端阅读。它为 research-and-report scenario 运行完整 flow：和 gateway handshake、模拟 OAuth 2.1、合并 tools/list、把 generate_report 作为 task、A2A call 到 writer、返回 ui:// resource、发出 OTel spans。

重点看：

- 一个 trace id 覆盖每一跳。
- Gateway policy 阻止第二个用户写入。
- Task lifecycle 从 working → completed，并返回 text 和 ui:// content。
- A2A call 的内部状态对 orchestrator 不透明。
- AGENTS.md 和 SKILL.md 是另一个 agent 复现 workflow 所需的唯一文件。

## 交付它

本课产出 `outputs/skill-ecosystem-blueprint.md`。给定一个 product need（research、summarization、automation），这个 skill 会生成完整 architecture：使用哪些 MCP primitives、哪些 gateway controls、哪些 A2A calls、哪些 telemetry、怎样 packaging。

## 练习

1. 运行 `code/main.py`。注意单个 trace id 以及 spans 如何嵌套。统计 demo 触及了阶段 13 的多少 primitives。

2. 扩展 demo：添加第二个 backend MCP server（例如 `bibliography`），并确认 gateway 把它的 tools 合并进同一个 namespace。

3. 用一个运行在 subprocess 上的真实 A2A writer agent 替换 fake A2A writer agent。使用第 19 课 harness。

4. 在 orchestrator 和 LLM 之间的 routing gateway 中添加 PII redaction step。确认 user query 中的 emails 被 scrub。

5. 为将维护这个系统的 teammate 写一份 AGENTS.md。它应在五分钟内读完，并给他们在 Cursor 或 Codex 中驱动 capstone 所需的一切。

## 关键词

| Term | 大家常说 | 实际含义 |
|------|----------|----------|
| Capstone | “Phase-13 integration demo” | 使用每个 primitive 的端到端系统 |
| Research and report | “scenario” | search、summarize、render pattern |
| Ecosystem | “所有部件组合起来” | server + client + gateway + sub-agent + telemetry + package |
| Trace hierarchy | “Single trace id” | 每一跳的 span 共享 trace；通过 span ids 建 parent-child |
| Gateway-issued token | “Transitive auth” | client 只看到 gateway 的 token；gateway 持有 upstream creds |
| Merged namespace | “All tools in one flat list” | gateway 进行 multi-server merge，collision 时加前缀 |
| Opacity boundary | “A2A call hides internals” | sub-agent reasoning 对 orchestrator 不可见 |
| Three-layer stack | “AGENTS.md + SKILL.md + MCP” | project context + workflow + tools |
| Defense-in-depth | “Multiple security layers” | pinned hashes、OAuth、RBAC、Rule of Two、audit log |
| Spec compliance matrix | “What we ship that the spec requires” | deliverables 到 2025-11-25 requirements 的 checklist |

## 延伸阅读

- [MCP — Specification 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25) — consolidated reference
- [MCP blog — 2026 roadmap](https://blog.modelcontextprotocol.io/posts/2026-mcp-roadmap/) — protocol 未来方向
- [a2a-protocol.org](https://a2a-protocol.org/latest/) — A2A v1.0 reference
- [OpenTelemetry — GenAI semconv](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — canonical tracing conventions
- [Anthropic — Claude Agent SDK overview](https://code.claude.com/docs/en/agent-sdk/overview) — production agent runtime patterns
