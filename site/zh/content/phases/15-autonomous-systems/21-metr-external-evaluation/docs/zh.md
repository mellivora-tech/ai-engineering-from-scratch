# METR Time Horizons 与 External Capability Evaluation

> METR（原 ARC Evals）自 2023 年 12 月起成为独立 501(c)(3)。他们的 Time Horizon 1.1 benchmark（2026 年 1 月）将 task-success probability 与 log(expert human completion time) 拟合成 logistic curve；50% probability 处的交点定义模型的 time horizon。2025–2026 engagement set 覆盖 GPT-5.1、GPT-5.1-Codex-Max，以及 prototype monitoring evaluations（monitor 能否捕捉 side tasks；agent 能否 evade）。Benchmark suites：HCAST（180+ 个 ML、cyber、SWE、reasoning tasks；1 分钟到 8+ 小时）、RE-Bench（71 个带 expert baseline 的 ML research-engineering tasks）、SWAA。诚实提醒：METR measurements 是理想化的——没有人类、没有真实后果——团队也记录了 eval-vs-deployment behavior gap（第 1 课）。time horizon 是 upper bound，不是 deployment prediction。

**类型：** 学习
**语言：** Python（stdlib，logistic-fit horizon estimator）
**前置要求：** 阶段 15 · 01（Long-horizon agents），阶段 15 · 19（RSP）
**时间：** ~60 分钟

## 问题

Scaling policies（第 19、20 课）只有在它们引用的 measurements 有用时才有用。“AI R&D-4 threshold”和“Long-range Autonomy”写在 policy prose 中；只有当具体 evaluations 产生具体数字时，它们才可操作。

METR 是 2024–2026 年定义许多这类数字的 external evaluation organization。他们评估 frontier models——通常是 pre-release，并与 labs 签 NDA——然后发布 methodology。Time Horizon 1.1 benchmark（2026 年 1 月）是他们的 headline artifact：一个将 capability 压缩成人类可读单位的 single scalar（“这个模型在 50% reliability 下能做专家花 X 小时处理的那类任务”）。

本课一部分是 methodology（horizon 如何计算），一部分是 interpretation（为什么 horizon 是 upper bound，而不是 deployment prediction）。两种技能必须放在一起。理解 horizon 如何 fit 的团队，比只在幻灯片上看到“14 hours”的团队，更不容易被糟糕 vendor claim 欺骗。

## 概念

### METR background

- Founded：2023 年 12 月（原 ARC Evals，拆分为独立 501(c)(3)）。
- Scope：评估 frontier models 的 autonomous capabilities，通常 pre-release。
- Partner labs：Anthropic、OpenAI（2025–2026 多次 engagements）。
- Notable deliverables：Time Horizon 1.0（2025 年 3 月）、Time Horizon 1.1（2026 年 1 月）、prototype monitoring evaluations。

### Time Horizon fit

Methodology（来自 METR blog 和 papers）：

1. 收集一个 task suite，覆盖 minute-scale 到 hour-scale expert completion times。当前 suites：HCAST（180+ tasks）、RE-Bench（71 tasks）、SWAA。
2. 让模型运行每个 task；记录 success 或 failure。
3. 拟合 logistic curve：P(success) 作为 log(expert completion time) 的函数。
4. horizon 是 P(success) = 0.5 时的 expert-time。

logistic-fit shape 是合适的，因为 capability 与 task difficulty 通常呈现增长并接近平台的关系。50% point 是一个选择（也可以是 10%、90%）；METR 在详细论文中报告多个 thresholds，但以 50% 为主，因为它最直观。

### 2026 年 1 月数字

根据 Time Horizon 1.1：

- Claude Opus 4.6：在 Time Horizon 1.1（2026 年 1 月）中，50% reliability 下约 14 小时。
- HCAST-style tasks 上的 doubling time：Time Horizon 1.1（2026 年 1 月）报告的 post-2023 fit 约 4.3 个月（130.8 天）；~7 个月数字来自 Time Horizon 1.0 的完整 2019–2025 fit，并在 TH1.1 中与 post-2023 数字一同报告。

这些是 Phase 15 其余部分引用的具体数字。Anthropic 的 RSP v3.0（第 19 课）和 DeepMind 的 FSF v3（第 20 课）引用它们作为 threshold assessments 的输入之一。

### Benchmark suites

- **HCAST**：189 个任务，覆盖 ML、cybersecurity、software engineering 和 general reasoning。跨度：1 分钟到 8+ 小时 expert time。horizon number 背后的 main suite。
- **RE-Bench**：71 个带 human expert baseline 的 ML research-engineering tasks。衡量模型是否能做进入 AI R&D-4 threshold 的那类 ML R&D work。
- **SWAA**：较短 tasks；用于校准 horizon curve 低端。

### Prototype monitoring evaluations

2025–2026 METR 工作包含 monitoring-evaluation prototypes：monitor 能否捕捉模型执行 side task（operator 要求之外的东西）；agent 能否 evade monitor。这些直接进入 OpenAI PF v2 的 “Undermining Safeguards” category，以及 DeepMind FSF v3（第 20 课）中的 deceptive-alignment monitoring。

### 为什么 horizons 是 upper bounds

- **Idealized tooling**：benchmark tasks 给模型 clean tools 和 well-formatted data。真实 deployments 更乱。
- **No real consequences**：模型不会真正给客户计费、删除真实数据、联系真实人员。真实 deployments 有 irreversible stakes。
- **Eval-context gaming**：第 1 课。模型在 tests 中表现不同。2026 International AI Safety Report 对此有经验证据。
- **No legitimate user variance**：benchmark prompts 是结构化的。真实用户会产生 ambiguous、context-dependent requests。

horizon 是有利条件下的 capability ceiling。Deployment reliability 是另一个更低的数字，团队必须测量自己的 distribution 才能知道它。

### external-evaluator case

External evaluation 很重要，因为内部 labs 有激励优化他们报告的 metrics。METR 的独立性——一个有 declared methodology 和 peer-reviewed papers 的 501(c)(3)——是结构性 mitigation。它本身不充分（labs 仍控制 METR 看见什么），但严格优于没有 external evaluation。

### 实践中如何使用 horizon numbers

- **作为 capability filter**：如果模型 horizon 远低于 proposed task 的 expert-time，不要 autonomous ship（第 1 课的 skill file）。
- **作为 trend indicator**：doubling time 告诉你，即使没有新 mitigations，当前实践还能安全多久。
- **作为 prior**：14 小时 horizon 是起点。根据你的 task distribution、tooling quality 和 deployment context 向下调整。

## 使用它

`code/main.py` 会在给定 synthetic result set 时，实现 task-success vs log(expert time) 的 logistic fit。它报告 50% horizon（METR headline）、10% horizon（conservative）和 90% horizon（optimistic）。也会展示当 success rate 被 eval-context gaming 人为抬高时会发生什么。

## 交付它

`outputs/skill-horizon-interpretation.md` 会审查 vendor 的 horizon claim，并产出 benchmark claim 与 deployment reality 之间的 gap analysis。

## 练习

1. 运行 `code/main.py`。确认 fit 的 50% horizon 匹配 synthetic ground truth。然后将 task-time grid 减半；horizon estimate 是否有显著变化？

2. 阅读 METR 的 Time Horizon 1.1 blog post。找出 reliability 最高和最低的具体 tasks。解释为什么有差距。

3. 阅读 METR 的 “Measuring Autonomous AI Capabilities” resources。列出 HCAST task categories。选择一个你会为 production task 更高权重的 category，并解释原因。

4. 在 simulator 中引入 eval-context gaming：将约 20% failed tasks 翻成 success。报告新的 horizon。这近似说明 20% gaming rate 对 observed number 的影响。

5. 在你自己的 bug backlog 或代表性 task set 上设计 internal horizon evaluation。描述 data collection、fit，以及 output 告诉你什么。与 METR numbers 对比。

## 关键术语

| Term | What people say | What it actually means |
|---|---|---|
| METR | “External evaluator” | 原 ARC Evals；自 2023 年 12 月起为独立 501(c)(3) |
| Time Horizon | “Capability measure” | logistic fit 得到的 50% reliability 下 expert task length |
| HCAST | “METR 的 main suite” | 180+ tasks，跨度 1 分钟到 8+ 小时 |
| RE-Bench | “Research engineering” | 71 个带 human baseline 的 ML research-engineering tasks |
| SWAA | “Short-task suite” | 校准 horizon curve 低端 |
| Doubling time | “Growth rate” | 50% horizon 翻倍时间；HCAST 约 7 个月 |
| Eval-context gaming | “模型表现不同” | tests 与 deployment 之间有记录的 behavior gap |
| Upper bound | “Horizon 是天花板” | Benchmark horizon > load 下的 deployment reliability |

## 延伸阅读

- [METR — Resources for Measuring Autonomous AI Capabilities](https://metr.org/measuring-autonomous-ai-capabilities/) — HCAST、RE-Bench、SWAA specs。
- [METR — Measuring AI Ability to Complete Long Tasks](https://metr.org/blog/2025-03-19-measuring-ai-ability-to-complete-long-tasks/) — 原始 horizon 论文。
- [METR — Time Horizon 1.1 (January 2026)](https://metr.org/research/) — 当前数字和 methodology。
- [Epoch AI — METR Time Horizons benchmark](https://epoch.ai/benchmarks/metr-time-horizons) — live tracking。
- [Anthropic — Measuring agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) — 关于 METR measurements 的内部视角。
