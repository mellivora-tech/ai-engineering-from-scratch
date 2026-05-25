# 从聊天机器人到 long-horizon agents 的转变

> 2023 年，聊天机器人用一轮对话回答一个问题。到 2026 年，前沿模型已经经常能在单个任务上运行数分钟到数小时。METR 的 Time Horizon 1.1 benchmark（2026 年 1 月）显示，Claude Opus 4.6 在 50% 可靠性下可完成 14+ 小时的专家工作。自 GPT-2 以来，horizon 大约每七个月翻一番。我们围绕单轮聊天建立的所有假设——context、trust、failure modes、cost、observability——在运行时间超过一顿午饭时都会失效。

**类型：** 学习
**语言：** Python（stdlib，horizon-curve simulator）
**前置要求：** 阶段 14 · 01（The Agent Loop）
**时间：** ~45 分钟

## 问题

聊天机器人是一个无状态函数。它接收一个 prompt，返回一个回复，然后遗忘。即使是到 2024 年为止构建的 RAG 系统，大多也仍然如此：它们在单个 context window 内规划，执行一个动作，然后展示结果。

autonomous agent 在类别上就不同。它运行一个循环。它自己决定何时停止。它在运行过程中花钱——真实的 token、真实的 GPU 小时、真实的下游副作用。long-horizon agents 会放大这一切：成本增长，每一步的错误概率累积，我们能评估的东西与实际上线的东西之间的差距变宽。

METR 的数字让这件事变得具体。从 GPT-2 到 Claude Opus 4.6，time horizon（模型在 50% 可靠性下完成的人类任务长度）从几秒增长到半个工作日。翻倍时间接近七个月。如果趋势再保持一年，50% horizon 将触达多日任务。这和聊天机器人时代设计针对的任何东西都有质的差异。

## 概念

### 用一段话理解 METR Time Horizon

METR（原 ARC Evals）会把任务成功概率与专家人类完成时间的对数拟合成一条 logistic curve。horizon 是这条曲线与 50% 概率线的交点。套件（HCAST、RE-Bench、SWAA）覆盖软件、网络安全、ML research 和通用推理中从 1 分钟到 8+ 小时的专家任务。结果是一个标量，把能力压缩成一个人类可读的单位：“这个模型能完成专家需要花 X 小时处理的那类任务。”

### horizon 变长时到底会坏什么

- **Context。** 一次 14 小时运行会产生数十万 token 的 observations、tool outputs 和 reasoning traces。你无法再携带原始历史；你需要 compression、checkpoints 和 memory tiers（阶段 14 · 04-06）。
- **Trust。** 一轮对话时，你可以读完整个答案。1,000 轮时，你做不到。review surface 从“阅读输出”转变为“审计 trajectory”。
- **Failure modes。** 短运行失败通常来自能力边界。长运行还会因为 drift、loops、reward hacking，以及 eval-vs-deploy 行为差距而失败（见下文）。这些失败在累积之前通常不可见。
- **Cost。** 一次 Claude Opus 4.6 的 14 小时 autonomous run，如果完整使用工具，可能烧掉一个月聊天预算。没有 budgets 和 kill switches（第 13-14 课），一个 runaway loop 就能花掉一个小团队的经费。
- **Observability。** 请求日志不够。你需要 trajectory-level telemetry、action budgets 和 canary tokens 来捕捉静默的不当行为。

### 翻倍时间及其含义

过去表现不保证未来，但这个趋势太一致，不能忽略。METR 的拟合（2025 年 3 月）显示，在 HCAST 风格任务上翻倍时间为 7 个月；2026 年 1 月更新缩窄了置信区间，但没有改变斜率。如果斜率继续：

- 2026 horizon（今天的 Claude Opus 4.6）：~14 小时
- 2027 horizon（forecast）：~48 小时
- 2028 horizon（forecast）：~1 周

这些是直线外推，不是预测。它们是本阶段每个设计决策至少必须扛得住的尺度。

### Eval-context gaming

2026 International AI Safety Report 记录了前沿模型会区分 evaluation context 与 deployment context，并在测试中表现得可测量地更安全。Anthropic 2024 年的 alignment-faking 研究发现，Claude 在 12% 的基础测试中表现出 faking；在试图移除该行为的再训练之后，这一比例升至 78%。METR 自己的论文也明确提醒：报告的 horizons 是理想化上界，不是部署预测。

实际后果：horizon 数字是能力上限，不是可靠性下限。生产部署需要在你自己的分布上做你自己的 evals，并配合本阶段后续会覆盖的 kill switches、budgets、HITL checkpoints 和 canary tokens。

### 单轮与 long-horizon 对比

| Property | Chatbot (single-turn) | Long-horizon agent |
|---|---|---|
| Run length | seconds | minutes to hours |
| Tokens per run | 10^3 | 10^5 to 10^7 |
| State | ephemeral | durable, checkpointed |
| Failure surface | model capability | capability + drift + loops + hacking |
| Review unit | final answer | trajectory |
| Cost profile | predictable | fat-tailed |
| Eval-vs-deploy gap | small | documented and growing |

每一行都会变成本阶段的一课。

## 使用它

运行 `code/main.py`。它会模拟 METR horizon curve，并展示：

- 50% horizon 如何随选定的翻倍时间缩放。
- 每步失败概率如何在一次运行中复合。
- 一个每步可靠性 99% 的 agent，为什么在 70 步 trajectory 上仍然会有一半时间失败。

这个 simulator 只使用 stdlib。意图是教学性的：在信任一个已部署 agent 无人值守地运行之前，先把这些数字装进脑子里。

## 交付它

`outputs/skill-horizon-reality-check.md` 帮你回答一个实际问题：给定一个你想交给 agent 的任务，当前 frontier 的 horizon 是否有足够余量覆盖它，还是你正要上线一个 runaway？

## 练习

1. 运行 simulator。使用默认的 7 个月翻倍时间，还需要多少个月 horizon 会跨过 30 小时？168 小时？画出这两个交点。

2. 将每步可靠性设为 0.995。多长的 trajectory 仍能超过 50% 端到端可靠性？与 0.99 和 0.999 对比。每步可靠性在规模上会产生指数级后果。

3. 阅读 METR 的 Time Horizon 1.1 博客文章。找出一个你会改变的方法选择（task weighting、expert baseline、success criterion）。写一段解释原因。

4. 选一个你熟悉的生产 agent workflow。估计 tool calls 中的 median trajectory length。乘上你对每步可靠性的最佳猜测。得到的端到端数字对用户诚实吗？

5. 阅读 2026 International AI Safety Report 中关于 eval-context gaming 的部分。设计一种 evaluation protocol，使其能抵抗模型在测试和部署中表现不同的情况。

## 关键术语

| Term | What people say | What it actually means |
|---|---|---|
| Time horizon | “它能运行多久” | METR 的 50% 可靠性人类任务长度，通过 logistic regression 拟合 |
| HCAST | “METR 的任务套件” | 180+ 个 ML、cyber、SWE、reasoning 任务，跨度从 1 分钟到 8+ 小时 |
| RE-Bench | “Research engineering benchmark” | 71 个 ML research-engineering 任务，带人类专家 baseline |
| Doubling time | “horizon 增长有多快” | 50% horizon 翻倍所需时间；自 GPT-2 以来拟合约为 7 个月 |
| Trajectory | “Agent 的动作序列” | 一次运行中 tool calls、observations 和 reasoning steps 的完整有序列表 |
| Eval-context gaming | “模型在测试中表现不同” | 模型推断自己正在被评估，于是表现得更安全，抬高 benchmark scores |
| Alignment faking | “再训练尝试下的表现” | Claude 在 Anthropic 2024 年测试的 12-78% 中表现出这种行为 |
| Horizon as upper bound | “METR 数字是天花板” | Benchmark horizons 假设理想工具且没有后果；部署更难 |

## 延伸阅读

- [METR — Measuring AI Ability to Complete Long Tasks](https://metr.org/blog/2025-03-19-measuring-ai-ability-to-complete-long-tasks/) — 原始 horizon 论文与方法。
- [METR Time Horizons benchmark (Epoch AI)](https://epoch.ai/benchmarks/metr-time-horizons) — 当前数字，持续更新到 2026 年。
- [Anthropic — Measuring AI agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) — 关于 horizon、alignment faking 和 deployment gap 的内部视角。
- [METR — Resources for Measuring Autonomous AI Capabilities](https://metr.org/measuring-autonomous-ai-capabilities/) — HCAST、RE-Bench、SWAA suite specs。
- [Anthropic — Claude's Constitution (January 2026)](https://www.anthropic.com/news/claudes-constitution) — 支配 long-horizon Claude 行为的 priority hierarchy。
