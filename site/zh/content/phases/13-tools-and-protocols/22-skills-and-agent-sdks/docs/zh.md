# Skills 和 Agent SDKs：Anthropic Skills、AGENTS.md、OpenAI Apps SDK

> MCP 说明“有哪些工具”。Skills 说明“如何完成任务”。2026 年的 stack 会叠加两者。Anthropic 的 Agent Skills（open standard，2025 年 12 月）以 SKILL.md 和 progressive disclosure 交付。OpenAI 的 Apps SDK 是 MCP 加 widget metadata。AGENTS.md（现在已在 60,000+ repos 中）位于 repo root，作为 project-level agent context。本课命名每一层负责什么，并构建一个可跨 agents 迁移的最小 SKILL.md + AGENTS.md bundle。

**类型：** 学习
**语言：** Python（stdlib，SKILL.md parser and loader）
**前置要求：** 阶段 13 · 07（MCP server）
**时间：** ~45 分钟

## 学习目标

- 区分三层：AGENTS.md（project context）、SKILL.md（reusable know-how）、MCP（tools）。
- 编写带 YAML frontmatter 和 progressive disclosure 的 SKILL.md。
- 以 filesystem-style 把 skills 加载进 agent runtime。
- 把 skill 与 MCP server 和 AGENTS.md 组合，让一个 package 可用于 Claude Code、Cursor 和 Codex。

## 问题

工程师把 release-notes-writing workflow 提炼成 multi-step prompt：“读取最新 merged PRs。按 area 分组。总结每个。按团队风格写 changelog entry。发到 Slack draft。” 他们把它放在团队的 Notion doc 中。

现在他们想从 Claude Code、Cursor 和 Codex CLI 使用这个 workflow。每个 agent 加载 instructions 的方式不同：Claude Code slash-commands、Cursor rules、Codex `.codex.md`。工程师复制了三份 workflow，并维护三份副本。

AGENTS.md 和 SKILL.md 共同修复这个问题：

- **AGENTS.md** 位于 repo root。每个兼容 agent 在 session start 读取它。“这个项目如何工作？有哪些 conventions？哪些命令运行 tests？”
- **SKILL.md** 是 portable bundle：YAML frontmatter（name、description）+ markdown body + optional resources。支持 skills 的 agents 会按需按名称加载它们。
- **MCP**（阶段 13 · 06-14）处理 skill 需要调用的 tools。

三层，一个 portable artifact。

## 概念

### AGENTS.md（agents.md）

2025 年末发布，到 2026 年 4 月被 60,000+ repos 采用。repo root 的一个文件。格式：

```markdown
# Project: my-service

## Conventions
- TypeScript with strict mode.
- Use Pydantic for models on the Python side.
- Tests run with `pnpm test`.

## Build and run
- `pnpm dev` for local dev server.
- `pnpm build` for production bundle.
```

agents 在 session start 读取它，并用它校准自己在该项目上的行为。2026 年每个 coding agent 都支持 AGENTS.md：Claude Code、Cursor、Codex、Copilot Workspace、opencode、Windsurf、Zed。

### SKILL.md format

Anthropic 的 Agent Skills（2025 年 12 月作为 open standard 发布）：

```markdown
---
name: release-notes-writer
description: Write a changelog entry for the latest merged PRs following this project's style.
---

# Release notes writer

When invoked, run these steps:

1. List PRs merged since the last tag. Use `gh pr list --base main --state merged`.
2. Group by label: feature, fix, chore, docs.
3. For each PR in each group, write one line: `- <title> (#<num>)`.
4. Draft the release notes and stage them in CHANGELOG.md.

If the user says "ship", run `git tag vX.Y.Z` and `gh release create`.

## Notes

- Never include commits without a PR.
- Skip "chore" entries from the public changelog.
```

Frontmatter 声明 skill identity。body 是 skill 加载时展示给模型的 prompt。

### Progressive disclosure

Skills 可以引用 sub-resources，agent 只在需要时 fetch。示例：

```
skills/
  release-notes-writer/
    SKILL.md
    style-guide.md
    template.md
    scripts/
      generate.sh
```

SKILL.md 写 “see style-guide.md for the style rules”。agent 只在 skill 活跃运行时拉取 style-guide.md。这避免把模型可能不需要的细节塞满 prompt。

### Filesystem discovery

agent runtimes 会扫描已知目录中的 SKILL.md files：

- `~/.anthropic/skills/*/SKILL.md`
- Project `./skills/*/SKILL.md`
- `~/.claude/skills/*/SKILL.md`

加载按 folder name 和 frontmatter `name`。Claude Code、Anthropic Claude Agent SDK 和 SkillKit（cross-agent）都遵循这个 pattern。

### Anthropic Claude Agent SDK

`@anthropic-ai/claude-agent-sdk`（TypeScript）和 `claude-agent-sdk`（Python）在 session start 加载 skills，并在 runtime 内把它们暴露为可调用 “agents”。当用户调用 skill 时，agent loop 会 dispatch 到该 skill。

### OpenAI Apps SDK

2025 年 10 月发布；直接构建在 MCP 上。把 OpenAI 之前的 Connectors 和 Custom GPT Actions 统一到一个 developer surface。一个 Apps SDK app 是：

- 一个 MCP server（tools、resources、prompts）。
- 加上 ChatGPT UI 的 widget metadata。
- 再加上可选的 MCP Apps `ui://` resource，用于 interactive surfaces。

同一个 protocol，更丰富 UX。

### Cross-agent portability via SkillKit

SkillKit 和类似 cross-agent distribution layers 会把单个 SKILL.md 翻译成 32+ AI agents（Claude Code、Cursor、Codex、Gemini CLI、OpenCode 等）的 native format。一个 source of truth；多个 consumers。

### The three-layer stack

| Layer | File | Loaded when | Purpose |
|-------|------|-------------|---------|
| AGENTS.md | repo root | session start | project-level conventions |
| SKILL.md | skills directory | skill invoked | reusable workflow |
| MCP server | external process | tools needed | callable actions |

三者组合：agent 在 session start 读取 AGENTS.md，用户调用 skill，skill instructions 包含 MCP tool calls，agent 通过 MCP client dispatch。

## 使用它

`code/main.py` 提供一个 stdlib SKILL.md parser 和 loader。它发现 `./skills/` 下的 skills，解析 YAML frontmatter 和 markdown body，并生成按 skill name keyed 的 dict。然后它模拟一个 agent loop，按名称调用 `release-notes-writer`。

重点看：

- YAML frontmatter 用最小 stdlib parser 解析（无 `pyyaml` dependency）。
- Skill body 原样保存；agent 在 invocation 时把它 prepend 到 system prompt。
- Progressive disclosure 通过 `read_subresource` function 演示，按需拉取 referenced files。

## 交付它

本课产出 `outputs/skill-agent-bundle.md`。给定一个 workflow，这个 skill 会生成组合的 SKILL.md + AGENTS.md + MCP-server-blueprint bundle，可跨 agents 迁移。

## 练习

1. 运行 `code/main.py`。在 `skills/` 下添加第二个 skill，并确认 loader 会发现它。

2. 为这个课程 repo 写一个 AGENTS.md。包含 testing commands、style conventions 和 Phase 13 mental model。

3. 把你团队内部 docs 中的 multi-step workflow 移植成 SKILL.md。验证它能在 Claude Code 中加载。

4. 手动把 skill 翻译成 Cursor 和 Codex 的 native rule formats。统计 formats 之间的 diff——这就是 SkillKit 自动化的 translation surface。

5. 阅读 Anthropic Agent Skills blog post。找出 Claude Agent SDK 中一个本课 loader 没有覆盖的 feature。（提示：agent sub-invocation。）

## 关键词

| Term | 大家常说 | 实际含义 |
|------|----------|----------|
| SKILL.md | “skill file” | YAML frontmatter 加 markdown body，由 agent runtime 加载 |
| AGENTS.md | “Repo-root agent context” | session start 时读取的 project-level conventions file |
| Progressive disclosure | “Lazy-load sub-resources” | skill body 引用只在需要时拉取的文件 |
| Frontmatter | “YAML block at top” | `---` delimiters 中的 metadata（name、description） |
| Claude Agent SDK | “Anthropic's skill runtime” | `@anthropic-ai/claude-agent-sdk`，加载 skills 并路由 |
| OpenAI Apps SDK | “MCP + widget meta” | OpenAI 基于 MCP 加 ChatGPT UI hooks 的 dev surface |
| Skill discovery | “Filesystem scan” | 遍历已知目录中的 SKILL.md，并按 name 建索引 |
| Cross-agent portability | “One skill many agents” | 通过 SkillKit-style tools 把一个 SKILL.md 翻译到 32+ agents |
| Agent Skill | “Portable know-how” | MCP tool concept 之外的 reusable task template |
| Apps SDK | “MCP plus ChatGPT UI” | Connectors 和 Custom GPTs 统一到 MCP 上 |

## 延伸阅读

- [Anthropic — Agent Skills announcement](https://www.anthropic.com/engineering/equipping-agents-for-the-real-world-with-agent-skills) — 2025 年 12 月发布
- [Anthropic — Agent Skills docs](https://platform.claude.com/docs/en/agents-and-tools/agent-skills/overview) — SKILL.md format reference
- [OpenAI — Apps SDK](https://developers.openai.com/apps-sdk) — ChatGPT 的 MCP-based developer platform
- [agents.md](https://agents.md/) — AGENTS.md format 和 adoption list
- [Anthropic — anthropics/skills GitHub](https://github.com/anthropics/skills) — official skill examples
