# Capstone 09 — 代码迁移 Agent（Repo 级语言 / Runtime 升级）

> Amazon 的 MigrationBench（Java 8 到 17）和 Google 的 App Engine Py2-to-Py3 migrator 定义了 2026 年的门槛。Moderne 的 OpenRewrite 可以大规模执行 deterministic AST rewrites。Grit 用 codemod-style DSL 解决同一类问题。生产模式把两者结合起来：用 deterministic substrate 做安全 rewrite，再用 agent layer 处理模糊案例；每个 branch 在 sandbox 中 build；test harness 变绿后才打开 PR。这个 capstone 要迁移 50 个真实 repo，并发布 pass rate 和 failure taxonomy。

**类型：** Capstone
**语言：** Python（agent）、Java / Python（targets）、TypeScript（dashboard）
**前置要求：** 阶段 5（NLP）、阶段 7（transformers）、阶段 11（LLM engineering）、阶段 13（tools）、阶段 14（agents）、阶段 15（autonomous）、阶段 17（infrastructure）
**覆盖阶段：** P5 · P7 · P11 · P13 · P14 · P15 · P17
**时间：** 30 小时

## 问题

大规模代码迁移是 2026 年 coding agents 最干净的生产应用之一。ground truth 很明确（迁移后 test suite 是否通过？），收益真实（Java-8 fleet migration 是 headcount-scale project），benchmark 也公开（MigrationBench 50-repo subset）。Moderne 的 OpenRewrite 负责 deterministic side。agent layer 处理所有 OpenRewrite recipes 无法处理的问题：ambiguous rewrites、build-system drift、long-tail syntax、transitive dependency breakage。

你将构建一个 agent，输入 Java 8 repo（或 Python 2 repo），输出一个 green-CI migrated branch。你将衡量 pass rate、test-coverage preservation、cost per repo，并构建 failure taxonomy。与 deterministic-only baseline 的 side-by-side 会告诉你 agent 的价值真正出现在哪里。

## 概念

pipeline 有两层。**deterministic substrate**（Java 用 OpenRewrite，Python 用 libcst）安全地执行大部分 mechanical rewrites：imports、method signatures、null-safety edits、try-with-resources、deprecated API replacements。它速度快，diff 可审计。**agent layer**（OpenAI Agents SDK 或基于 Claude Opus 4.7 与 GPT-5.4-Codex 的 LangGraph）处理 recipes 无法覆盖的情况：build-file upgrades（Maven/Gradle/pyproject）、transitive dependency conflicts、test flakes、custom annotations。

每个 repo 都有一个预装目标 runtime 的 Daytona sandbox。agent 迭代执行：run build、classify failures、apply fix、rerun。硬限制：每个 repo 30 分钟、$8、20 个 agent turns。如果所有测试通过且 coverage delta 不为负，就打开 PR。否则，把 repo 归入 failure class 并附 evidence。

failure taxonomy 是交付物。在 50 个 repo 上，是什么坏了？Transitive deps？Custom annotations？Build tool version？与迁移无关的 test flakes？每个 class 都有 count 和 exemplar diff。未来写 recipe 的人可以针对 top three 优先处理。

## 架构

```
target repo
      |
      v
OpenRewrite / libcst deterministic recipes
   (safe, fast, auditable, ~70-80% of fixes)
      |
      v
Daytona sandbox per branch
      |
      v
agent loop (Claude Opus 4.7 / GPT-5.4-Codex):
   - run build -> capture failures
   - classify failures (build, test, lint)
   - apply fix (patch or retry recipe)
   - rerun
   - budget: 30 min, $8, 20 turns
      |
      v
test + coverage delta gate
      |
      v (passed)
open PR
      |
      v (failed)
file under failure class + attach repro
```

## 技术栈

- Deterministic substrate：OpenRewrite（Java）或 libcst（Python）
- Agent：OpenAI Agents SDK，或基于 Claude Opus 4.7 + GPT-5.4-Codex 的 LangGraph
- Sandbox：每个 branch 一个 Daytona devcontainer，预装 target runtime（Java 17 / Python 3.12）
- Build systems：Maven、Gradle、uv（Python）
- Benchmarks：Amazon MigrationBench 50-repo subset（Java 8 到 17）、Google App Engine Py2-to-Py3 repos
- Test harness：parallel runner；Java 用 Jacoco、Python 用 coverage.py 做 coverage
- Observability：Langfuse + 每个 repo 的 trace bundle，包含每个 diff chunk
- Dashboard：failure-taxonomy dashboard，带 per-class counts 和 exemplar diffs

## 构建它

1. **Recipe pass。** 首先运行 OpenRewrite（Java）或 libcst（Python）recipes。抓住 70-80% 的 mechanical migrations。提交为 “recipe” commit。

2. **Build trial。** Daytona sandbox：安装 target runtime，运行 build。如果 green，跳到 tests。如果 red，交给 agent。

3. **Agent loop。** LangGraph tools：`run_build`、`read_file`、`edit_file`、`run_test`、`git_diff`。agent 对 failure 分类（dep、syntax、test、build-tool），并应用 targeted fix。重新运行。

4. **Budget caps。** 每个 repo 30 分钟 wall-clock、$8 cost、20 个 agent turns。任何超限都会停止，并以当前 diff 归入 “budget_exhausted”。

5. **Test + coverage gate。** build 变绿后，运行 test suite。把 coverage 与 base repo 比较。如果 coverage 下降超过 2%，归入 “coverage_regression”。

6. **PR open。** 成功后 push branch，打开 PR，附上 diff、应用了哪些 recipes、哪些 commits 由 agent 编写。

7. **Failure taxonomy。** 对每个失败 repo 打 class tag：`dep_upgrade_required`、`build_tool_drift`、`custom_annotation`、`test_flake`、`syntax_edge_case`、`budget_exhausted`。构建 dashboard。

8. **50-repo run。** 在 MigrationBench subset 上执行。报告 per-class pass rate、cost-per-repo、coverage-preservation，并与 deterministic-only baseline 对比。

## 使用它

```
$ migrate legacy-java-service --target java17
[recipe]   27 rewrites applied (JUnit 4->5, HashMap initializer, try-with-resources)
[build]    FAIL: cannot find symbol sun.misc.BASE64Encoder
[agent]    turn 1 classify: removed_jdk_api
[agent]    turn 2 apply: sun.misc.BASE64Encoder -> java.util.Base64
[build]    OK
[tests]    412/412 passing; coverage 84.1% -> 84.3%
[pr]       opened #1841  cost=$3.20  turns=4
```

## 交付它

`outputs/skill-migration-agent.md` 是交付物。给定一个 repo，它先执行 deterministic recipes，再运行 agent loop，产出一个 green migrated branch，或把 repo 归入 taxonomy class。

| 权重 | 标准 | 衡量方式 |
|:-:|---|---|
| 25 | MigrationBench pass rate | 50-repo subset pass@1 |
| 20 | Test-coverage preservation | 与 base 相比的 mean coverage delta |
| 20 | Cost per migrated repo | passing runs 的 $/repo |
| 20 | Agent / deterministic-tool integration | OpenRewrite handled vs agent authored fixes 的比例 |
| 15 | Failure analysis write-up | 带 exemplars 的 taxonomy completeness |
| **100** | | |

## 练习

1. 只用 OpenRewrite（不使用 agent）运行 migrate pipeline。将 pass rate 与完整 pipeline 对比。识别只有 agent 才能产生差异的案例。

2. 实现一个 “lint-clean” check：迁移后运行 style linter（Java 用 spotless，Python 用 ruff）。如果出现新的 lint errors，就让 PR 失败。衡量 coverage-preserved-but-style-regressed rate。

3. 添加 “minimal-diff” optimizer：agent branch 通过 tests 后，用第二轮 pass 修剪不必要变更。报告 diff-size reduction。

4. 扩展到第三种迁移：Node 18 到 Node 22。复用 sandbox wrapping；把 recipe layer 换成 custom codemod。

5. 把 time-to-first-green-build（TTFGB）作为 UX metric。目标：p50 低于 10 分钟。

## 关键词汇

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|------------------------|
| Deterministic substrate | “Recipe engine” | OpenRewrite / libcst：带 safety guarantees 的 declarative AST rewrites |
| Codemod | “Code-modifying program” | 机械式修改 source code 的 rewrite rule |
| Build drift | “Tool version skew” | major versions 之间 Maven / Gradle / uv 行为的细微变化 |
| Failure class | “Taxonomy bucket” | repo 未迁移成功的标注原因：dep、syntax、test、build-tool、budget |
| Coverage delta | “Coverage preservation” | 从 base 到 migrated branch 的 test coverage % 变化 |
| Agent turn | “Tool-call round” | agent loop 中的一次 plan -> act -> observe cycle |
| Budget exhaustion | “Hit the ceiling” | repo 消耗了 30-min / $8 / 20-turn 上限仍未通过 |

## 延伸阅读

- [Amazon MigrationBench](https://aws.amazon.com/blogs/devops/amazon-introduces-two-benchmark-datasets-for-evaluating-ai-agents-ability-on-code-migration/) — canonical 2026 benchmark
- [Moderne.io OpenRewrite platform](https://www.moderne.io) — deterministic substrate reference
- [OpenRewrite documentation](https://docs.openrewrite.org) — recipe authoring
- [Grit.io](https://www.grit.io) — alternate codemod DSL
- [OpenAI sandboxed migration cookbook](https://developers.openai.com/cookbook/examples/agents_sdk/sandboxed-code-migration/sandboxed_code_migration_agent) — Agents SDK reference
- [Google App Engine Py2 to Py3 migrator](https://cloud.google.com/appengine) — alternate migration benchmark
- [libcst](https://github.com/Instagram/LibCST) — Python deterministic substrate
- [Daytona sandboxes](https://daytona.io) — reference per-branch sandbox
