# Autonomous Coding Agent 版图（2026）

> SWE-bench Verified 在不到三年内从 4% 到 80.9%。同一个 Claude Sonnet 4.5 在 SWE-agent v1 上得分 43.2%，在 Cline autonomous 上得分 59.8%——围绕模型的 scaffolding 现在与模型本身一样重要。OpenHands（原 OpenDevin）是最活跃的 MIT-licensed platform，它的 CodeAct loop 会直接在 sandbox 中执行 Python actions，而不是 JSON tool calls。headline numbers 掩盖了一个方法问题：500 个 SWE-bench Verified tasks 中有 161 个只需要 1–2 行修改，而 SWE-bench Pro（10+ 行任务）上，同样的 frontier models 只有 23–59%。

**类型：** 学习
**语言：** Python（stdlib，CodeAct vs JSON tool-call comparison）
**前置要求：** 阶段 14 · 07（Tool use），阶段 15 · 01（Long-horizon agents）
**时间：** ~45 分钟

## 问题

“哪个 coding agent 最好”是错误问题。正确问题是：在一个匹配我工作的 task distribution 上，使用我会在 production 中运行的 scaffolding，我得到的 end-to-end reliability 是多少？

2022 到 2026 年间，领域学到：scaffolding——retrieval layer、planner、sandbox、edit-verify loop、feedback format——是承重结构。Claude Sonnet 4.5 在 SWE-agent v1 的 SWE-bench Verified 上得分 43.2%；同一个模型放进 Cline 的 autonomous scaffold 得分 59.8%。同样 weights，相差 16.6 个百分点。base model 是组件；loop 才是产品。

伴随问题是 benchmark saturation 会掩盖 regressions。SWE-bench Verified 接近饱和，而且 easy-task tail（500 个任务中有 161 个只需 ≤2 行）把 top scores 拉高。现实质量更适合用 SWE-bench Pro（10+ 行修改）这样的分布衡量，同样的 leaders 在那里仍只有 23–59%。

## 概念

### 用一段话理解 SWE-bench

SWE-bench（Jimenez et al.）取真实 GitHub issues 与 ground-truth patches，要求 agent 产出一个能让 test suite 通过的 patch。SWE-bench Verified（OpenAI, 2024）是一个 500-task 的 human-curated subset，移除了 ambiguous 和 broken tasks。SWE-bench Pro 是更难的后继版本——任务需要 10+ 行修改，当前 frontier agents 得分为 23–59%。

### 2022 → 2026 曲线真正显示了什么

- **2022**：research models 在 raw SWE-bench 上约 4%。
- **2024**：GPT-4 + Devin-style scaffolding 约 14%；SWE-agent 约 12%。
- **2025**：Claude 3.5/3.7 Sonnet 在 Aider 和 SWE-agent 中推进到 40–55% 区间。
- **2026**：Claude Sonnet 4.5 和 frontier competitors 在 SWE-bench Verified 上达到 70–80%+。Epoch AI 的 leaderboard 实时跟踪这一点。

这个斜率来自三个复合来源：更好的 base models，更好的 scaffolding（CodeAct、reflection、verifier loops），以及更好的 benchmarks（Verified 移除了噪声）。

### CodeAct vs JSON tool calls

OpenHands（All-Hands-AI, arXiv:2407.16741，原 OpenDevin）押注了一个具体架构：模型不是发出由 host 解码并执行的 JSON tool calls，而是发出 Python code，由 Jupyter-style kernel 在 sandbox 中运行。agent 可以在一个 action 内遍历 files、链式调用 tools，并捕捉自己的 exceptions。

权衡：

- **JSON tool calls**：每个 action 是一轮；易审计；组合性有限；默认更安全，因为每次调用都经过 explicit validator。
- **CodeAct**：一个 action 可以是一整个 program；组合性强；需要 hardened sandbox（OpenHands 使用 Docker isolation）；failure modes 包括 sandbox runtime 允许的任何事情。

两种架构都在生产中使用。CodeAct 在开放平台中占主导（OpenHands、smolagents）。JSON tool calls 在 managed services 中仍占主导（Anthropic Managed Agents、OpenAI Assistants），因为 provider 控制 executor。

### 2026 版图中的 scaffolds

| Scaffold | License | Execution model | Notable property |
|---|---|---|---|
| OpenHands (OpenDevin) | MIT | CodeAct in Docker | Most active open platform; event-stream replayable |
| SWE-agent | MIT | Agent-Computer Interface (ACI) | First end-to-end SWE-bench scaffold |
| Aider | Apache-2 | edit-via-diff in local repo | Minimal scaffold, strong regression stability |
| Cline | Apache-2 | VS Code agent with tool policy | Highest-scoring open scaffold on Sonnet 4.5 |
| Devin (Cognition) | Proprietary | Managed VM + planner | First "AI software engineer" product category |
| Claude Code | Proprietary | Permission modes + routines | Lesson 10 covers the agent loop in detail |

### 为什么 scaffolding 占主导

一次 coding run 是 long-horizon trajectory（第 1 课）。可靠性会跨步骤复合。scaffolding 能带来分数的三个地方：

1. **Retrieval**：找到正确文件是静默瓶颈。SWE-agent 的 ACI、OpenHands 的 file-index、Aider 的 repo-map 都在攻击这一点。
2. **Verifier loop**：运行 tests、读取 stack traces、重新尝试，是 SWE-bench 上 10+ 点的 delta。
3. **Failure containment**：error 时 rollback 的 sandbox 可以防止 damage 复合。同一个模型，有无 verifier loop 看起来就像两个不同产品。

### Benchmark saturation 和真实分布

OpenHands 作者和 Epoch AI 都指出，SWE-bench Verified 有一个 easy tail：500 个任务中有 161 个只需 1–2 行修改。高分部分由这个 tail 驱动。SWE-bench Pro 限制为 10+ 行修改，即使 frontier systems 也只有 23–59%。你的 production distribution 几乎肯定更接近 Pro，而不是 Verified。

选择 agent 的含义：在你自己的 bug backlog 上运行一个 Pro-like subset。真正重要的分数，是代表你实际交付任务的分布上的分数。

## 使用它

`code/main.py` 在一个固定 mini-task distribution 上比较两个玩具 agent scaffolds：

1. 一个 **JSON tool-call** scaffold，每轮执行一个 action。
2. 一个 **CodeAct** scaffold，每个 action 可以发出一小段 Python snippet。

两者都使用 stub “model”（deterministic rules），因此比较会把 scaffold 与 model quality 分离。输出展示 CodeAct scaffold 会用更少 turns 解决更多 tasks，代价是更大的 per-action blast radius。

## 交付它

`outputs/skill-scaffold-audit.md` 帮你在采用前审计一个拟议 coding-agent scaffold：retrieval quality、verifier presence、sandbox isolation、benchmark-to-distribution fit。

## 练习

1. 运行 `code/main.py`。两个 scaffold 在同一 task set 上各需要多少 turns？各自的 per-action blast radius 是什么？

2. 阅读 OpenHands 论文（arXiv:2407.16741）。论文认为 CodeAct 在复杂任务上优于 JSON tool calls。找出论文承认的一个 failure mode，并写一句说明该 mode 何时会在 production 中占主导。

3. 从你的 bug backlog 中选择一个需要跨两个文件修改 10+ 行的 task。估计一个 frontier model 在（a）JSON tool calls 和（b）CodeAct 下的 end-to-end success probability。解释差距。

4. SWE-bench Verified 有 161 个 single-file、1–2 line tasks。构造一个排除它们的分数。leaderboard 会如何洗牌？

5. 阅读 “Introducing SWE-bench Verified”（OpenAI）。解释移除 ambiguous tasks 的具体方法，并指出一种 curation 会漏掉的类别。

## 关键术语

| Term | What people say | What it actually means |
|---|---|---|
| SWE-bench | “Coding benchmark” | 带 ground-truth patches 和 test suites 的真实 GitHub issues |
| SWE-bench Verified | “Cleaned subset” | 500 个 human-curated tasks，存在 easier-tail |
| SWE-bench Pro | “Harder subset” | 10+ 行修改；frontier 得分 23–59% |
| CodeAct | “Code-as-action” | Agent 发出 Python；Jupyter-style kernel 在 sandbox 中执行 |
| JSON tool call | “Function calling” | 每个 action 是执行前被验证的 structured JSON payload |
| Scaffold | “Agent framework” | base model 周围的 retrieval + planner + executor + verifier loop |
| ACI (Agent-Computer Interface) | “SWE-agent 的 format” | 为 LLM ergonomics 设计的 command set，而不是人类 shell |
| Verifier loop | “Test-and-retry” | 运行 tests、读取 output、修订 patch；最大的非模型 reliability gain |

## 延伸阅读

- [Jimenez et al. — SWE-bench](https://www.swebench.com/) — 原始 benchmark 和 methodology。
- [OpenAI — Introducing SWE-bench Verified](https://openai.com/index/introducing-swe-bench-verified/) — curated subset 的构建方式。
- [Wang et al. — OpenHands: An Open Platform for AI Software Developers](https://arxiv.org/abs/2407.16741) — CodeAct architecture 和 event-stream design。
- [Epoch AI — SWE-bench leaderboard](https://epoch.ai/benchmarks) — 实时跟踪 scores。
- [Anthropic — Measuring agent autonomy](https://www.anthropic.com/research/measuring-agent-autonomy) — long-horizon coding-agent reliability framing。
