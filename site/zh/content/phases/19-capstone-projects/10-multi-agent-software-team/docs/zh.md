# Capstone 10 — 多 Agent 软件工程团队

> SWE-AF 的 factory architecture、MetaGPT 的 role-based prompting、AutoGen 0.4 的 typed actor graph、Cognition 的 Devin 和 Factory 的 Droids，都收敛到 2026 年的同一种形态：architect 负责 plan，N 个 coders 在 parallel worktrees 中工作，reviewer 做 gate，tester 做 verify。parallel worktrees 把 wall-clock 转化为 throughput。shared state 和 handoff protocols 成为 failure surface。这个 capstone 是构建这个团队，在 SWE-bench Pro 上评估，并报告哪些 handoffs 会坏、坏得有多频繁。

**类型：** Capstone
**语言：** Python / TypeScript（agents）、Shell（worktree scripts）
**前置要求：** 阶段 11（LLM engineering）、阶段 13（tools）、阶段 14（agents）、阶段 15（autonomous）、阶段 16（multi-agent）、阶段 17（infrastructure）
**覆盖阶段：** P11 · P13 · P14 · P15 · P16 · P17
**时间：** 40 小时

## 问题

single-agent coding harnesses 在大型任务上会碰到天花板。不是因为单个 agent 太弱，而是因为 200k-token context 无法同时装下 architecture plan、四个 parallel codebase slices、reviewer commentary 和 test output。multi-agent factories 会拆分问题：architect 拥有 plan，coders 在 parallel worktrees 中分别实现，reviewer 做 gate，tester 做 verify。SWE-AF 的 “factory” architecture、MetaGPT 的 roles、AutoGen 的 typed actor graph，这三种叙述都在描述同一种形态。

failure surface 是 handoff。Architect 规划了 coders 无法实现的东西。Coders 产出冲突 diffs。Reviewer 批准了 hallucinated fix。Tester 与仍在写入的 coder 发生 race。你将构建这样一个团队，在 50 个 SWE-bench Pro issues 上运行，追踪每个 handoff，并发布 post-mortem。

## 概念

Roles 是 typed agents。**Architect**（Claude Opus 4.7）读取 issue，写出 plan，并用明确 interfaces 拆成 subtasks。**Coders**（Claude Sonnet 4.7，N 个并行实例，每个在一个 `git worktree` + Daytona sandbox 中）独立实现 subtasks。**Reviewer**（GPT-5.4）读取 merged diff，批准或请求具体修改。**Tester**（Gemini 2.5 Pro）在隔离环境中运行 test suite，并带 artifacts 报告 pass/fail。

通信通过 shared task board（file-backed 或 Redis）进行。每个 role 消费自己有权限处理的 tasks。Handoffs 是 A2A-protocol-typed messages。协调问题包括：merge-conflict resolution（coordinator role 或 automatic three-way merge）、shared-state synchronization（coders 开始后 plan 冻结；replans 是独立 events）、reviewer gatekeeping（reviewer 不能批准自己写的变更或自己提出的变更）。

Token amplification 是隐藏成本。每个 role boundary 都会增加 summary prompts 和 handoff context。一个 40-turn single-agent run 会变成四个 roles 上总计 160 turns。评分标准会专门权衡 token efficiency vs single-agent baseline，因为问题不是 “multi-agent 是否能工作”，而是 “它是否按美元计价也能赢”。

## 架构

```
GitHub issue URL
      |
      v
Architect (Opus 4.7)
   reads issue, produces plan with subtasks + interfaces
      |
      v
Task board (file / Redis)
      |
   +-- subtask 1 ---+-- subtask 2 ---+-- subtask 3 ---+-- subtask 4 ---+
   v                v                v                v                v
Coder A          Coder B          Coder C          Coder D          (4 parallel)
 (Sonnet)         (Sonnet)         (Sonnet)         (Sonnet)
 worktree A       worktree B       worktree C       worktree D
 Daytona          Daytona          Daytona          Daytona
      |                |                |                |
      +--------+-------+-------+--------+
               v
           merge coordinator  (three-way merge + conflict resolution)
               |
               v
           Reviewer (GPT-5.4)
               |
               v
           Tester  (Gemini 2.5 Pro)  -> passes? -> open PR
                                     -> fails?  -> route back to coder
```

## 技术栈

- Orchestration：带 shared state + per-agent sub-graphs 的 LangGraph
- Messaging：用于 typed inter-agent messages 的 A2A protocol（Google 2025）
- Models：Opus 4.7（architect）、Sonnet 4.7（coders）、GPT-5.4（reviewer）、Gemini 2.5 Pro（tester）
- Worktree isolation：每个 coder 使用 `git worktree add` + Daytona sandbox
- Merge coordinator：custom three-way merge + LLM-mediated conflict resolution
- Eval：SWE-bench Pro（50 issues）、SWE-AF scenarios、HumanEval++ 用于 unit tests
- Observability：Langfuse，带 role-tagged spans 和 per-agent token accounting
- Deployment：K8s，每个 role 是独立 Deployment，并基于 backlog 做 HPA

## 构建它

1. **Task board。** File-backed JSONL，包含 typed messages：`plan_request`、`subtask`、`diff_ready`、`review_needed`、`test_needed`、`approved`、`rejected`、`replan_needed`。Agents 订阅 tags。

2. **Architect。** 读取 GitHub issue，用 Opus 4.7 和 plan template 运行，要求输出明确的 subtask interfaces（files touched、public functions、test impact）。发出一个带 subtask DAG 的 `plan_request`。

3. **Coders。** N 个 parallel workers，每个从 board 认领一个 subtask。每个 worker 都启动一个新的 `git worktree add` branch 加 Daytona sandbox。实现 subtask。发出包含 patch + test deltas 的 `diff_ready`。

4. **Merge coordinator。** all-coders-done 后，把 N 个 branches 通过 three-way merge 合并到 staging branch。只有 file-level overlap 存在时才做 LLM-mediated conflict resolution。

5. **Reviewer。** GPT-5.4 读取 merged diff。不能批准自己写过的 diff。发出 `approved`（no-op）或带具体 change requests 的 `review_feedback`，并 route 回相关 coder。

6. **Tester。** Gemini 2.5 Pro 在 clean sandbox 中运行 test suite。捕获 artifacts。发出带 stacktraces 的 `test_passed` 或 `test_failed`。失败 tests loop back 到拥有 failing subtask 的 coder。

7. **Handoff accounting。** 每个跨 role boundary 的 message 都在 Langfuse 中生成 span，记录 payload size 和 model used。计算 per-subtask token amplification（coder_tokens + reviewer_tokens + tester_tokens + architect_share / coder_tokens）。

8. **Eval。** 在 50 个 SWE-bench Pro issues 上运行。把 pass@1 和 $-per-solved-issue 与 single-agent baseline（一个 Sonnet 4.7，在单个 worktree 中）对比。

9. **Post-mortem。** 对每个失败 issue，识别坏掉的 handoff（plan too vague、merge conflict、reviewer false-approve、tester flake）。产出 handoff-failure histogram。

## 使用它

```
$ team run --issue https://github.com/acme/widget/issues/842
[architect] plan: 4 subtasks (parser, cache, api, migration)
[board]     dispatched to 4 coders in parallel worktrees
[coder-A]   subtask parser  -> 42 lines, tests pass locally
[coder-B]   subtask cache   -> 88 lines, tests pass locally
[coder-C]   subtask api     -> 31 lines, tests pass locally
[coder-D]   subtask migration -> 19 lines, tests pass locally
[merge]     3-way merge: 0 conflicts
[reviewer]  comments on cache (thread pool sizing); routed to coder-B
[coder-B]   revision: 92 lines; submits
[reviewer]  approved
[tester]    all 412 tests pass
[pr]        opened #3382   4 coders, 1 revision, $4.90, 18m
```

## 交付它

`outputs/skill-multi-agent-team.md` 是交付物。给定 issue URL 和 parallelism level，团队会产出一个 merge-ready PR，并带 per-role token accounting。

| 权重 | 标准 | 衡量方式 |
|:-:|---|---|
| 25 | SWE-bench Pro pass@1 | matched 50-issue subset，pass@1 |
| 20 | Parallel speedup | Wall-clock vs single-agent baseline |
| 20 | Review quality | injected-bug probe 上的 false-approval rate |
| 20 | Token efficiency | Total tokens per solved issue vs single-agent |
| 15 | Coordination engineering | Merge-conflict resolution、handoff-failure histogram |
| **100** | | |

## 练习

1. 在运行中向 diff 注入一个明显 bug（main body 前多一个 `return None`）。衡量 reviewer 的 false-approve rate。调整 reviewer prompt，直到 false-approval 低于 5%。

2. 缩减到两个 coders（architect + coder + reviewer + tester，coder sequentially 运行两个 subtasks）。比较 wall-clock 和 pass rate。

3. 把 merge coordinator 换成 single-writer constraint（subtasks 触碰互不相交的 file sets）。衡量 architect 的 planning burden。

4. 把 reviewer 从 GPT-5.4 换成 Claude Opus 4.7。衡量 false-approval rate 和 token cost delta。

5. 添加第五个 role：documenter（Haiku 4.5）。review 之后，它生成 changelog entry。衡量 documentation quality 是否值得额外 token spend。

## 关键词汇

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|------------------------|
| Parallel worktree | “Isolated branch” | `git worktree add` 为每个 coder 生成一个新的 working tree |
| Task board | “Shared message bus” | agents 订阅的 typed messages 文件或 Redis store |
| Handoff | “Role boundary” | 从一个 role 的 context 跨到另一个 role 的任何消息 |
| Token amplification | “Multi-agent overhead” | 同一任务中跨 roles 的 total tokens / single-agent tokens |
| A2A protocol | “Agent-to-agent” | Google 2025 typed inter-agent messages 规范 |
| Merge coordinator | “Integrator” | 运行 three-way merge 并调解 conflicts 的组件 |
| False approval | “Reviewer hallucination” | Reviewer 批准了带 known bugs 的 diff |

## 延伸阅读

- [SWE-AF factory architecture](https://github.com/Agent-Field/SWE-AF) — reference 2026 multi-agent factory
- [MetaGPT](https://github.com/FoundationAgents/MetaGPT) — role-based multi-agent framework
- [AutoGen v0.4](https://github.com/microsoft/autogen) — Microsoft typed actor framework
- [Cognition AI (Devin)](https://cognition.ai) — reference product
- [Factory Droids](https://www.factory.ai) — alternate reference product
- [Google A2A protocol](https://developers.google.com/agent-to-agent) — inter-agent messaging spec
- [git worktree documentation](https://git-scm.com/docs/git-worktree) — isolation substrate
- [SWE-bench Pro](https://www.swebench.com) — evaluation target
