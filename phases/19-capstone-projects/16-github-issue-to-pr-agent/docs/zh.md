# Capstone 16 — GitHub Issue-to-PR 自主 Agent

> AWS Remote SWE Agents、Cursor Background Agents、OpenAI Codex cloud 和 Google Jules 都交付了 2026 年相同的 product shape：给 issue 打 label，得到一个 PR。agent 在 cloud sandbox 中运行，验证 tests 通过，然后发布一个带 rationale、可 review 的 PR。难点是自动复现 repo 的 build environment、防止 credential leakage、强制 per-repo budgets，并确保 agent 不能 force-push。这个 capstone 构建 self-hosted 版本，并在 cost 和 pass rate 上与 hosted alternatives 对比。

**类型：** Capstone
**语言：** Python（agent）、TypeScript（GitHub App）、YAML（Actions）
**前置要求：** 阶段 11（LLM engineering）、阶段 13（tools）、阶段 14（agents）、阶段 15（autonomous）、阶段 17（infrastructure）
**覆盖阶段：** P11 · P13 · P14 · P15 · P17
**时间：** 30 小时

## 问题

async cloud coding agent 是与 interactive coding agents（capstone 01）分开的产品类别。UX 是一个 GitHub label。你给 issue 标上 `@agent fix this`，worker 会在 cloud sandbox 中启动，clone repo，运行 tests，编辑文件，验证，并打开一个 PR，正文中附上 agent 的 rationale。没有 interactive loop，没有 terminal。AWS Remote SWE Agents、Cursor Background Agents、OpenAI Codex cloud、Google Jules 和 Factory Droids 都收敛到这个形态。

工程挑战很具体：environment reproduction（agent 必须在没有 cached dev image 的情况下从头 build repo）、flaky tests（必须 rerun 或 isolate）、credential scoping（使用最小 fine-grained permissions 的 GitHub App）、每个 repo 每天的 budget enforcement，以及 no-force-push policy。capstone 会衡量 pass rate、cost 和 safety，并与 hosted alternatives 对比。

## 概念

trigger 是 GitHub webhook（issue label 或 PR comment）。dispatcher 把 work 入队到 ECS Fargate 或 Lambda。worker 把 repo 拉入 Daytona 或 E2B sandbox，并根据 repo（language、framework）推断出一个 generic Dockerfile。agent 使用 mini-swe-agent 或 SWE-agent v2 loop，底层模型是 Claude Opus 4.7 或 GPT-5.4-Codex。它循环执行：read code、propose fix、apply patch、run tests。

Verification 是 gating step。PR 打开前，full CI 必须在 sandbox 中通过。coverage delta 会被计算；如果超过阈值为负，PR 仍会打开，但会被标注 `needs-review`。agent 会把 rationale 作为 PR description 发布，并开一个 reviewer 可以 @agent 追问 follow-ups 的线程。

Safety 通过两个 GitHub surfaces scoped：App 提供短期 installation token，带 `workflows: read` 和狭窄 repo contents/PR scopes；branch protection（不是 app permissions）强制 “no direct writes to `main`” 和 “no force-push”，app 永远不加入 bypass list。对 `.github/workflows` 的 path-scoped read-only access 不是 GitHub App 原语，所以 worker 必须在 proposed diff 上用 allow-list 强制执行。per repo per day 的 budget ceilings 在 dispatcher 执行（例如每 repo 每天最多 5 个 PR，每 PR $20）。

## 架构

```
GitHub issue labeled `@agent fix` or PR comment
            |
            v
    GitHub App webhook -> AWS Lambda dispatcher
            |
            v
    ECS Fargate task (or GitHub Actions self-hosted runner)
       - pull repo
       - infer Dockerfile (language, package manager)
       - Daytona / E2B sandbox with target runtime
       - clone -> git worktree -> agent branch
            |
            v
    mini-swe-agent / SWE-agent v2 loop
       Claude Opus 4.7 or GPT-5.4-Codex
       tools: ripgrep, tree-sitter, read/edit, run_tests, git
            |
            v
    verify CI passes in-sandbox + coverage delta check
            |
            v (verified)
    git push + open PR via GitHub App
       PR body = rationale + diff summary + trace URL
       label: needs-review
            |
            v
    operator reviews; can @-mention agent for follow-ups
```

## 技术栈

- Trigger：带 fine-grained token 的 GitHub App；通过 Lambda 或 Fly.io 接收 webhook
- Worker：ECS Fargate task（或 GitHub Actions self-hosted runner）
- Sandbox：每个 task 一个 Daytona devcontainer 或 E2B sandbox
- Agent loop：基于 Claude Opus 4.7 / GPT-5.4-Codex 的 mini-swe-agent baseline 或 SWE-agent v2
- Retrieval：tree-sitter repo-map + ripgrep
- Verification：sandbox 中 full CI + coverage delta gate
- Observability：Langfuse，per-PR trace archive 从 PR body 链接
- Budget：per-repo daily dollar ceiling；每 repo 每天最大 PR 数

## 构建它

1. **GitHub App。** Fine-grained installation token：issues read+write、pull_requests write、contents read+write、workflows read。branch protection（唯一能做到这一点的 surface）强制 “no direct push to `main`” 和 “no force-push”；app 不在 bypass list。worker 通过 proposed diff 上的 allow-list check 强制 “no writes under `.github/workflows`”，因为 GitHub App permissions 不是 path-scoped。

2. **Webhook receiver。** Lambda function 接收 issue label / PR comment webhooks。按 label `@agent fix this` 过滤。入队到 SQS。

3. **Dispatcher。** 从 SQS 弹出 tasks。强制 per-repo per-day budget。启动 ECS Fargate task，并传入 repo URL、issue body 和一个 fresh Daytona sandbox。

4. **Environment inference。** 检测 language（Python、Node、Go、Rust）和 package manager（uv、pnpm、go mod、cargo）。如果不存在 Dockerfile，就动态生成一个。

5. **Agent loop。** 使用 Claude Opus 4.7 的 mini-swe-agent 或 SWE-agent v2。Tools：ripgrep、tree-sitter repo-map、read_file、edit_file、run_tests、git。硬限制：$20 cost、30 min wall-clock、30 agent turns。

6. **Verification。** loop 结束后，在 sandbox 中运行 full test suite。通过 jacoco / coverage.py 计算 coverage delta。如果 CI red：停止，不打开 PR。如果 coverage 下降超过 2%：打开 PR 并打 `needs-review` label。

7. **PR posting。** Push agent branch。通过 GitHub API 打开 PR，包含：title、rationale、diff summary、trace URL、cost、turns。

8. **Credential hygiene。** Worker 使用短期 GitHub App installation token。归档前清理 logs 中的 secrets。

9. **Eval。** 30 个 seeded internal issues，难度各异。衡量 pass rate、PR quality（diff size、style、coverage）、cost、latency。与 Cursor Background Agents 和 AWS Remote SWE Agents 在同一 issues 上比较。

## 使用它

```
# on github.com
  - user labels issue #842 with `@agent fix this`
  - PR #1903 appears 14 minutes later
  - body:
    > Fixed NPE in widget.dedupe() caused by null comparator entry.
    > Added regression test widget_test.go::TestDedupeNullComparator.
    > Coverage delta: +0.12%
    > Turns: 7  Cost: $1.80  Trace: langfuse:...
    > Label: needs-review
```

## 交付它

`outputs/skill-issue-to-pr.md` 是交付物。一个 GitHub App + async cloud worker，把 labeled issues 转成 review-ready PR，并控制成本与 credential scope。

| 权重 | 标准 | 衡量方式 |
|:-:|---|---|
| 25 | Pass rate on 30 issues | End-to-end success（CI green + coverage OK） |
| 20 | PR quality | Diff size、coverage delta、style conformance |
| 20 | Cost and latency per resolved issue | 每个 PR 的 $ 和 wall-clock |
| 20 | Safety | Scoped token、per-repo budget、no force-push、credential hygiene |
| 15 | Operator UX | Rationale comments、retry affordance、@-mention follow-up |
| **100** | | |

## 练习

1. 添加 “fix flaky test” mode：label `@agent stabilize-flake TestX` 会在 sandbox 中运行测试 50 次，并提出一个能稳定它的 minimal change。

2. 在三个 shared issues 上与 Cursor Background Agents 比较 cost。报告哪些工具在哪些地方胜出。

3. 实现 budget dashboard：per-repo per-day cost、per-user cost。异常时 alert。

4. 构建 “dry-run” mode：不运行 CI 就打开 draft PR，让 reviewers 低成本检查 plan。

5. 添加 retention policy：超过 7 天未 merge 的 PR branches 自动删除。

## 关键词汇

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|------------------------|
| GitHub App | “Scoped bot identity” | 带 fine-grained permissions 和短期 installation token 的 App |
| Async cloud agent | “Background agent” | 在 cloud sandbox 中运行的非交互 worker，而不是 terminal |
| Environment inference | “Dockerfile synthesis” | 检测 language + package manager，缺失时生成 Dockerfile |
| Verification | “CI-in-sandbox” | 打开 PR 前，在 worker 内运行 full test suite |
| Coverage delta | “Coverage preservation” | 从 base 到 agent branch 的 test coverage % 变化 |
| Per-repo budget | “Daily ceiling” | dispatcher 强制的 dollar 和 PR-count cap |
| Rationale | “PR body explanation” | agent 对改了什么、为什么改的总结；PR body 必需 |

## 延伸阅读

- [AWS Remote SWE Agents](https://github.com/aws-samples/remote-swe-agents) — canonical async cloud agent reference
- [SWE-agent](https://github.com/SWE-agent/SWE-agent) — CLI reference
- [Cursor Background Agents](https://docs.cursor.com/background-agent) — commercial alternative
- [OpenAI Codex (cloud)](https://openai.com/codex) — hosted competitor
- [Google Jules](https://jules.google) — Google hosted version
- [Factory Droids](https://www.factory.ai) — alternate commercial reference
- [GitHub App documentation](https://docs.github.com/en/apps) — scoped bot identity
- [Daytona cloud sandboxes](https://daytona.io) — reference sandbox
