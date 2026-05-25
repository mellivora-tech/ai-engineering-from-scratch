# Capstone 01 — 终端原生编码 Agent

> 到 2026 年，编码 agent 的形态已经基本定型：一个 TUI harness、一个有状态 plan、一个沙箱化 tool surface，以及一个会计划、行动、观察、恢复的循环。Claude Code、Cursor 3 和 OpenCode 从远处看都长得一样。这个 capstone 要求你 end-to-end 构建一个这样的系统：CLI 输入，pull request 输出，并在 SWE-bench Pro 上用 mini-swe-agent 和 Live-SWE-agent 做对照评测。你会理解，真正困难的不是 model call，而是 tool loop、sandbox，以及 50 轮运行里的成本上限。

**类型：** Capstone
**语言：** TypeScript / Bun（harness）、Python（eval 脚本）
**前置要求：** 阶段 11（LLM engineering）、阶段 13（tools and protocols）、阶段 14（agents）、阶段 15（autonomous systems）、阶段 17（infrastructure）
**覆盖阶段：** P0 · P5 · P7 · P10 · P11 · P13 · P14 · P15 · P17 · P18
**时间：** 35 小时

## 问题

到 2026 年，编码 agent 已经成为最主流的 AI 应用类别。Claude Code（Anthropic）、带 Composer 2 和 Agent Tabs 的 Cursor 3（Cursor）、Amp（Sourcegraph）、OpenCode（11.2 万 star）、Factory Droids 和 Google Jules，都在同一套架构上做变体：终端 harness、带权限的 tool surface、sandbox，以及围绕前沿模型构建的 plan-act-observe 循环。前沿能力的差距很窄：Live-SWE-agent 用 Opus 4.5 在 SWE-bench Verified 上达到 79.2%。但工程手艺的空间很大。大多数失败模式不是模型犯错，而是 tool-loop 不稳定、context poisoning、token 成本失控，以及破坏性的文件系统操作。

你无法从外部真正理解这些 agent。你必须亲手构建一个，看着 loop 在第 47 轮因为 ripgrep 返回 8MB 匹配结果而崩溃，然后重建截断层。这就是这个 capstone 的意义。

## 概念

harness 有四个 surface。**Plan** 维护一个 TodoWrite 风格的状态对象，模型每一轮都会重写它。**Act** 分发工具调用（read、edit、run、search、git）。**Observe** 捕获 stdout / stderr / exit code，截断后把摘要喂回去。**Recover** 处理工具错误，同时避免撑爆 context window 或无限循环。2026 年的形态还多了一样东西：**hooks**。`PreToolUse`、`PostToolUse`、`SessionStart`、`SessionEnd`、`UserPromptSubmit`、`Notification`、`Stop` 和 `PreCompact`，这些都是可配置的扩展点，operator 可以在这里注入 policy、telemetry 和 guardrails。

sandbox 使用 E2B 或 Daytona。每个任务都在一个全新的 devcontainer 里运行，并挂载一个可读写的 git worktree。harness 永远不碰宿主机文件系统。任务成功或失败后，worktree 都会被销毁。成本控制分三层执行：每轮 token 上限、每个 session 的美元预算，以及硬性的轮数上限（通常是 50）。observability 层使用带 GenAI semantic conventions 的 OpenTelemetry spans，并发送到自托管 Langfuse。

## 架构

```
  user CLI  ->  harness (Bun + Ink TUI)
                  |
                  v
           plan / act / observe loop  <--->  Claude Sonnet 4.7 / GPT-5.4-Codex / Gemini 3 Pro
                  |                          (via OpenRouter, model-agnostic)
                  v
           tool dispatcher (MCP StreamableHTTP client)
                  |
     +------------+------------+----------+
     v            v            v          v
  read/edit    ripgrep     tree-sitter   git/run
     |            |            |          |
     +------------+------------+----------+
                  |
                  v
           E2B / Daytona sandbox  (worktree isolated)
                  |
                  v
           hooks: Pre/Post, Session, Prompt, Compact
                  |
                  v
           OpenTelemetry -> Langfuse (spans, tokens, $)
                  |
                  v
           PR via GitHub app
```

## 技术栈

- Harness runtime：Bun 1.2 + Ink 5（React-in-terminal）
- 模型访问：OpenRouter unified API，支持 Claude Sonnet 4.7、GPT-5.4-Codex、Gemini 3 Pro、Opus 4.5（用于最难任务）
- 工具传输：Model Context Protocol StreamableHTTP（MCP 2026 revision）
- Sandbox：E2B sandboxes（JS SDK）或 Daytona devcontainers
- 代码搜索：ripgrep subprocess、17 种语言的 tree-sitter parser（预编译）
- 隔离：每个任务使用 `git worktree add`，成功 / 失败后清理
- Eval harness：SWE-bench Pro（verified subset）+ Terminal-Bench 2.0 + 你自己的 30-task holdout
- Observability：带 `gen_ai.*` semconv 的 OpenTelemetry SDK → 自托管 Langfuse
- PR 发布：使用 fine-grained token 的 GitHub App，scope 限定在目标 repo

## 构建它

1. **TUI 和 command loop。** 用 Ink 搭建 Bun 项目。接受 `agent run <repo> "<task>"`。打印分屏视图：plan pane（顶部）、tool-call stream（中间）、token budget（底部）。加入 Ctrl-C 取消逻辑，退出前触发 `SessionEnd` hook。

2. **Plan state。** 定义 typed TodoWrite schema（pending / in_progress / done items with notes）。模型每轮都以 tool call 的形式重写完整状态，不要让它做增量 mutation。把 plan 持久化到 `.agent/state.json`，这样崩溃后可以 resume。

3. **Tool surface。** 定义六个工具：`read_file`、`edit_file`（带 diff preview）、`ripgrep`、`tree_sitter_symbols`、`run_shell`（带 timeout）、`git`（status / diff / commit / push）。通过 MCP StreamableHTTP 暴露，使 harness 与传输层解耦。每个工具都返回截断后的输出（每次调用上限 4k tokens）。

4. **Sandbox wrapping。** 每个任务启动一个 E2B sandbox。用 `git worktree add -b agent/$TASK_ID` 创建新分支。所有工具调用都在 sandbox 内执行，宿主机文件系统不可达。

5. **Hooks。** 实现 2026 年的全部八种 hook 类型。至少接入四个用户编写的 hook：(a) `PreToolUse` 破坏性命令 guard，阻止 worktree 外的 `rm -rf`；(b) `PostToolUse` token accounting；(c) `SessionStart` budget initialization；(d) `Stop` 写入 final trace bundle。

6. **Eval loop。** 克隆 SWE-bench Pro Python 的 30 个 issue 子集。用你的 harness 跑每个任务。与 mini-swe-agent（最小 baseline）在 pass@1、turns-per-task 和 $-per-task 上比较。把结果写入 `eval/results.jsonl`。

7. **Cost control。** 硬性截断：50 轮、200k context、每任务 $5。`PreCompact` hook 在 150k 处把旧轮次总结成 prior-state block，为新 observation 腾出空间，同时不丢失 plan。

8. **PR posting。** 成功时，最后一步是 `git push` + 调用 GitHub API 打开 PR，并在正文中写入 plan 和 diff summary。

## 使用它

```
$ agent run ./my-repo "Fix the race condition in worker.rs"
[plan]  1 locate worker.rs and enumerate mutex uses
        2 identify shared state under contention
        3 propose fix, verify tests
[tool]  ripgrep mutex.*lock -t rust           (44 matches, truncated)
[tool]  read_file src/worker.rs 120..180
[tool]  edit_file src/worker.rs (+8 -3)
[tool]  run_shell cargo test worker::          (passed)
[plan]  1 done · 2 done · 3 done
[done]  PR opened: #482   turns=9   tokens=38k   cost=$0.41
```

## 交付它

可交付的 skill 位于 `outputs/skill-terminal-coding-agent.md`。给定 repo path 和 task description，它会在 sandbox 中运行完整的 plan-act-observe loop，并返回 PR URL 和 trace bundle。这个 capstone 的评分标准：

| 权重 | 标准 | 衡量方式 |
|:-:|---|---|
| 25 | SWE-bench Pro pass@1 vs baseline | 你的 harness 与 mini-swe-agent 在 30 个匹配 Python 任务上对比 |
| 20 | 架构清晰度 | Plan/act/observe 分离、hook surface、tool schema；对照 Live-SWE-agent layout 审查 |
| 20 | 安全性 | Sandbox escape tests、permission prompts、destructive-command guard 通过 red-team |
| 20 | Observability | Trace 完整性（100% tool call 有 span）、每轮 token accounting |
| 15 | Developer UX | Cold-start < 2s、crash recovery 能 resume plan、Ctrl-C 能干净地取消 mid-tool |
| **100** | | |

## 练习

1. 把 backing model 从 Claude Sonnet 4.7 换成在 vLLM 上服务的 Qwen3-Coder-30B。比较 pass@1 和 $-per-task。报告 open model 表现较差的位置。

2. 添加一个 `reviewer` sub-agent，在 PR 发布前读取 diff，并可以请求一轮 revision。衡量 false-positive review 是否会把 SWE-bench pass rate 拉低到 single-agent baseline 以下（提示：通常会）。

3. 压测 sandbox：写一个尝试 `curl` 外部 URL 的任务，以及一个尝试写入 worktree 外部的任务。确认二者都被 PreToolUse hook 阻止。记录这些尝试。

4. 用更小的模型（Haiku 4.5）实现 `PreCompact` summarization。衡量 3x compaction 会损失多少 plan fidelity。

5. 把 MCP StreamableHTTP transport 换成 stdio。基准测试 cold-start 和 per-call latency。为 local-only use 选出胜者。

## 关键词汇

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|------------------------|
| Harness | “The agent loop” | 包围模型的代码：分发工具、维护 plan state、强制预算 |
| Hook | “Agent event listener” | 用户编写的脚本，由 harness 在八个生命周期事件之一运行 |
| Worktree | “Git sandbox” | 位于独立路径的链接式 git checkout；可丢弃且不会触碰主 clone |
| TodoWrite | “Plan state” | 一个 typed pending/in-progress/done 列表，模型每轮都会重写 |
| StreamableHTTP | “MCP transport” | 2026 MCP revision：带双向 streaming 的长连接 HTTP；替代 SSE |
| Token ceiling | “Context budget” | 输入 + 输出 token 的每轮或每 session 上限；触发 compaction 或 termination |
| pass@1 | “Single-attempt pass rate” | SWE-bench 任务在第一次运行中解决的比例，不重试、不偷看测试集 |

## 延伸阅读

- [Claude Code documentation](https://docs.anthropic.com/en/docs/claude-code) — Anthropic 的 reference harness
- [Cursor 3 changelog](https://cursor.com/changelog) — Agent Tabs 和 Composer 2 product notes
- [mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent) — 用于 SWE-bench harness 对比的 minimal baseline
- [Live-SWE-agent](https://github.com/OpenAutoCoder/live-swe-agent) — 使用 Opus 4.5 达到 79.2% SWE-bench Verified
- [OpenCode](https://opencode.ai) — open harness，11.2 万 star
- [SWE-bench Pro leaderboard](https://www.swebench.com) — 本 capstone 面向的 evaluation
- [Model Context Protocol 2026 roadmap](https://blog.modelcontextprotocol.io/posts/2026-mcp-roadmap/) — StreamableHTTP、capability metadata
- [OpenTelemetry GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — tool call 和 token usage 的 span schema
