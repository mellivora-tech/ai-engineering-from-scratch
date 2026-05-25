# Eval-Driven Agent Development

> Anthropic 的建议：“start with simple prompts, optimize them with comprehensive evaluation, and add multi-step agentic systems only when needed.” Evaluation 不是最后一步。它是驱动 Phase 14 中所有其他选择的 outer loop。

**类型：** 学习 + 构建
**语言：** Python (stdlib)
**前置要求：** Phase 14 全部内容。
**时间：** ~60 分钟

## 学习目标

- 说出三层 evaluation — static benchmarks、custom offline、online production — 以及每层用途。
- 解释 evaluator-optimizer tight loop。
- 描述 2026 best practice：evals 和 code 放在一起、在 CI 中运行、gate PRs。
- 把 Phase 14 每一课连接到它生成的 eval case。

## 问题

Agents 能通过 demos。它们会以 demos 无法预测的方式在生产失败。Benchmarks 回答 “这个 model broadly capable 吗？” 而不是 “这个 agent 是否为我的产品提交正确 patches？” 答案是三层 evaluation，持续运行，并把每个 guardrail 和 learned rule 映射到 eval case。

## 概念

### 三层 evaluation

1. **Static benchmarks** — 代码用 SWE-bench Verified（第 19 课），browsing / desktop 用 WebArena/OSWorld（第 20 课），generalist 用 GAIA（第 19 课），tool use 用 BFCL V4（第 06 课）。用于 cross-model comparison 和 regression gating。Contamination 真实存在：SWE-bench+ 发现 32.67% solution leakage。始终报告 Verified / +-audited scores。

2. **Custom offline evals** — 你的产品形态：
   - LLM-as-judge（Langfuse、Phoenix、Opik — 第 24 课）。
   - Execution-based（运行 patch，检查 tests）。
   - Trajectory-based（对照 gold 比较 action sequences；OSWorld-Human 显示 top agents 比 gold 多 1.4-2.7x）。

3. **Online evals** — production：
   - Session replays（Langfuse）。
   - Guardrail-triggered alerts（第 16、21 课）。
   - Per-step cost / latency tracking（第 23 课 OTel spans）。

### Evaluator-optimizer (Anthropic)

Tight loop：

1. Proposer 生成 output。
2. Evaluator 判断。
3. Refine 直到 evaluator 通过。

这是 Self-Refine（第 05 课）的泛化。任何你在意的 agent flow 都可以包进 evaluator-optimizer 以提升 reliability。

### 2026 best practice

- Evals 和 code 放在同一个 repo。
- 每个 PR 在 CI 中运行。
- 用 eval scores gate merge（例如 “相比 main 没有 > 5% regression”）。
- 每个 guardrail 映射到一个 eval case。
- 每个 learned rule（Reflexion、pro-workflow learn-rule）映射到一个 failure case。

### 把 Phase 14 串起来

Phase 14 每一课都会生成 eval cases：

| Lesson | Eval case it generates |
|--------|------------------------|
| 01 Agent Loop | Budget-exhausted, infinite-loop guard |
| 02 ReWOO | Planner replans correctly when a tool fails |
| 03 Reflexion | Learned reflections apply on retry |
| 05 Self-Refine/CRITIC | Judge passes refined output |
| 06 Tool Use | Argument coercion works; unknown tools rejected |
| 07-10 Memory | Retrieval citations match sources; stale facts invalidate |
| 12 Workflow Patterns | Each pattern produces correct output |
| 13 LangGraph | Resume reproduces state exactly |
| 14 AutoGen Actors | DLQ catches crashed handlers |
| 16 OpenAI Agents SDK | Guardrail trips on the right inputs |
| 17 Claude Agent SDK | Subagent results return to orchestrator |
| 19-20 Benchmarks | SWE-bench Verified score, WebArena success rate, OSWorld efficiency |
| 21 Computer Use | Per-step safety catches injected DOM |
| 23 OTel | Spans emit required attributes |
| 26 Failure Modes | Detectors tag known failures |
| 27 Prompt Injection | PVE refuses poisoned retrievals |
| 28 Orchestration | Supervisor routes to the right specialist |
| 29 Runtime Shapes | DLQ handles N% failure |

如果你的 eval suite 覆盖这些 cases，你就覆盖了 Phase 14。

### Eval-driven development 会在哪里失败

- **No baseline。** 没有 last-known-good 的 evals 不可读。存储 baselines。
- **LLM-judge without grounding。** Judges 也会 hallucinate。CRITIC pattern（第 05 课）— judge 需要 ground 到 external tools。
- **Over-fitting to evals。** 为 eval 优化偏离 production usefulness。轮换 cases。
- **Flaky evals。** 非确定性 cases 会造成 false alarms。固定 seeds，snapshot state。

## 构建它

`code/main.py` 是一个 stdlib eval harness：

- 带 categories（benchmark、custom、online）的 case registry。
- 一个 scripted agent under test。
- Evaluator-optimizer loop：propose、judge、refine，直到 pass 或 max rounds。
- CI gate：aggregate pass rate + regression against baseline。

运行它：

```
python3 code/main.py
```

输出：per-case pass/fail、regression flag、CI gate verdict。

## 使用它

- 在和 agent code 相同的 repo 中写 eval cases。
- 每个 PR 通过 CI 运行它们。
- regression 时 fail build。
- 跟踪 pass rate over time。
- 把每个 production failure 绑定到一个新 case。

## 发布它

`outputs/skill-eval-suite.md` 会为 agent product 构建三层 eval suite，带 CI gates 和 regression tracking。

## 练习

1. 取一个你的 production failure。写一个能复现它的 eval case。你的 agent 现在能通过吗？
2. 为你的领域构建一个三维 LLM-judge rubric（factual、tone、scope）。给 50 个 sessions 打分。
3. 把 eval suite 接入 CI。在 >=5% regression 时 fail build。
4. 添加 trajectory-efficiency metric：agent 走了多少 steps，相比 gold trajectory 如何？
5. 把 Phase 14 每一课映射到你的 suite 中一个 eval case。有缺失吗？那就是要补的 gap。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|----------------|------------------------|
| Static benchmark | "Off-the-shelf eval" | SWE-bench、GAIA、AgentBench、WebArena、OSWorld |
| Custom offline eval | "Domain eval" | 在你的 product shape 上做 LLM-as-judge / exec / trajectory |
| Online eval | "Production eval" | Session replay、guardrail alerts、cost/latency tracking |
| Evaluator-optimizer | "Propose-judge-refine" | 迭代直到 judge 通过 |
| CI gate | "Merge blocker" | 在 eval regression 时 fail build |
| Baseline | "Last-known-good" | 用于检测 regression 的 reference score |
| Trajectory efficiency | "Steps over gold" | Agent step count 除以 human expert minimum |

## 延伸阅读

- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) — “start simple, optimize with evals”
- [OpenAI, SWE-bench Verified](https://openai.com/index/introducing-swe-bench-verified/) — curated benchmark
- [Berkeley Function Calling Leaderboard](https://gorilla.cs.berkeley.edu/leaderboard.html) — tool-use benchmark
- [Langfuse docs](https://langfuse.com/) — evals + session replay in practice
