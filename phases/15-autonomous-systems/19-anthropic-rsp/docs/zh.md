# Anthropic Responsible Scaling Policy v3.0

> RSP v3.0 于 2026 年 2 月 24 日生效，取代 2023 年 policy。Two-tier mitigation：Anthropic 会单方面做什么 vs 被 framing 为 industry-wide recommendation 的内容（包括 RAND SL-4 security standards）。新增 Frontier Safety Roadmaps 和 Risk Reports，作为 standing documents，而不是一次性交付物。移除 2023 年 pause commitment。引入 AI R&D-4 threshold：一旦跨过，Anthropic 必须发布 affirmative case，识别 misalignment risks 和 mitigations。Claude Opus 4.6 没有跨过它。Anthropic 在 v3.0 announcement 中表示，“confidently ruling this out is becoming difficult.” SaferAI 将 2023 RSP 评为 2.2；他们把 v3.0 下调到 1.9，把 Anthropic 放入与 OpenAI、DeepMind 同列的“weak”RSP 类别。Qualitative thresholds 取代了 2023 年的 quantitative commitments；移除 pause clause 是最尖锐的 regression。

**类型：** 学习
**语言：** Python（stdlib，RSP threshold decision engine）
**前置要求：** 阶段 15 · 06（AAR），阶段 15 · 07（RSI）
**时间：** ~45 分钟

## 问题

Frontier labs 发布的 scaling policies 部分是技术文档，部分是 governance documents，部分是给 regulators 的 signals。RSP v3.0 是当前 Anthropic document。仔细阅读它很重要，不是因为遵守它有约束力（没有），而是因为它的 framing 会塑造 lab 如何理解 catastrophic risk，以及如何向公众沟通 trade-offs。

v3.0 vs v2.0 diff 是有用单位。新增内容：Frontier Safety Roadmaps、Risk Reports、AI R&D-4 threshold。移除内容：2023 pause commitment。重新 framing 的内容：two-tier mitigation schedule，分为 Anthropic-unilateral 和 industry-recommendation。外部 review——SaferAI——将评分从 2.2（v2）降至 1.9（v3.0）。这就是 scaling policy 如何在看起来更 polished 的同时变得不那么 rigorous。

## 概念

### two-tier mitigation schedule

- **Anthropic unilateral actions**：无论其他 labs 做什么，Anthropic 都会做什么。超过 threshold 时 training stops、specific security measures、specific deployment gates。
- **Industry-wide recommendations**：Anthropic 认为行业应共同做什么。包括 RAND SL-4 security standards。这些不是 Anthropic 方面的 commitments；它们是 policy advocacy。

two-tier structure 在 v2 中不存在。这意味着读者需要看每个 commitment 位于哪一列。位于 “industry-wide recommendation” 列的 security measure 不是 Anthropic 的 promise；它是 Anthropic 的 hope。

### AI R&D-4 threshold

这是 RSP v3.0 命名的重要下一个 capability level。具体而言：一个模型可以以有竞争力的成本自动化 AI research 的 substantial fraction。Anthropic 一旦认为某个模型跨过它，就必须在继续 scaling 前发布 affirmative case，识别 misalignment risks 和 mitigations。

根据 v3.0 announcement，Claude Opus 4.6 没有跨过它。文档还补充：“confidently ruling this out is becoming difficult.” 这句话很重要；它承认 threshold 已经足够接近，成为 live concern，而非 speculative limit。

第 6 课（Automated Alignment Research）和第 7 课（Recursive Self-Improvement）直接导向这个 threshold。Automated alignment researchers 跨过 research-quality bars，是 AI R&D-4 threshold 正在接近的证据。

### Frontier Safety Roadmaps 与 Risk Reports

v3.0 将两类 artifact 提升为 standing documents：

- **Frontier Safety Roadmap**：前瞻性文档，描述 planned safety work、capability expectations 和 mitigation research。
- **Risk Report**：特定模型发布后的回顾性文档，描述 observed capability 和 residual risk。

两者都是公开的。两者都按 declared cadence 更新。用途是：读者可以追踪 Anthropic 在 Roadmap 中说会做什么，并对比 Risk Report 中报告了什么。

### 移除 pause clause

2023 RSP 包含明确的 pause commitment：如果模型跨过特定 capability thresholds，training 会暂停，直到 mitigations 到位。v3.0 用更软的 formulation 替代 explicit pause（发布 affirmative case，并在 mitigations 充分时继续）。SaferAI 和其他分析者直接指出，这是新文档中最强的 regression。

支持这一变化的 policy argument：2023 年的 quantitative thresholds 在 2026 年 capability benchmarks 下变得不可达，因为 benchmarks 本身被重新缩放。反方 argument：scaling policy 中的 pause clause 是 commitment device；移除它会移除 policy 的可信度。

### SaferAI 的 downgrade

SaferAI 是一个给 RSP-style documents 评分的独立组织。他们的公开 rating：2023 Anthropic RSP 得分 2.2（评分尺度中 4.0 是当前最佳 RSP，1.0 是 nominal）。v3.0 得分 1.9。这让 Anthropic 从 “moderate” 降到 “weak”，加入 OpenAI 和 DeepMind 所在的 weak category。

SaferAI 给出的 downgrade factors：
- Qualitative thresholds 取代 quantitative ones。
- Pause commitment 被移除。
- AI R&D-4 threshold mitigations 被描述为 “affirmative case”，而不是 specific measures。
- Review mechanisms 依赖 Anthropic 的 Safety Advisory Group，independent oversight 有限。

### 本课不是什么

这不是合规课。RSP v3.0 不是 regulation；没有东西强制 Anthropic 遵守它。本课是学习如何用它应得的 specificity 和 skepticism 阅读文档。Scaling policies 是 frontier labs 发出的关于 catastrophic-risk posture 的主要公共信号。读懂它们，是任何依赖 frontier capabilities 的人都需要的实际技能。

## 使用它

`code/main.py` 实现一个小 decision engine，镜像 RSP threshold-evaluation shape：给定 candidate model 和一组 capability measurements，返回 AI R&D-4 threshold 是否跨过、需要哪些 affirmative-case sections，以及 deployment 是否可以继续。它刻意简单；重点是显式化文档逻辑。

## 交付它

`outputs/skill-scaling-policy-review.md` 会用 v3.0 reference 审查 scaling policy（Anthropic、OpenAI、DeepMind 或 internal）：two-tier structure、thresholds、pause commitments、independent review。

## 练习

1. 运行 `code/main.py`。输入三个不同 capability levels 的 synthetic models。确认 threshold evaluator 行为符合预期，并生成正确的 affirmative-case template。

2. 完整阅读 RSP v3.0（32 页）。找出所有位于 “industry-wide recommendation” tier 的 commitments。哪些在 v2 中会是 “Anthropic unilateral”？

3. 阅读 SaferAI 的 RSP grading methodology。将 rubric 应用于文档，复现他们对 v3.0 的 1.9 分。哪一行 rubric 最推动 downgrade？

4. 2023 pause commitment 被移除了。提出一个 replacement commitment，在承认 2026 benchmark-rescaling problem 的同时保留 policy 的可信度。

5. 将 RSP v3.0 与 OpenAI Preparedness Framework v2（第 20 课）对比。选一个 v3.0 更强的领域。再选一个 Preparedness Framework 更强的领域。

## 关键术语

| Term | What people say | What it actually means |
|---|---|---|
| RSP | “Anthropic 的 scaling policy” | Responsible Scaling Policy；v3.0 于 2026 年 2 月 24 日生效 |
| AI R&D-4 | “Research-automation threshold” | 以有竞争力成本自动化 substantial AI research 的能力 |
| Affirmative case | “Safety justification” | 公开论证 risks 已识别且 mitigations 充分 |
| Frontier Safety Roadmap | “Forward plan” | 关于 planned safety work 和 expected capabilities 的 standing document |
| Risk Report | “Retrospective on a model” | 模型发布后关于 observed capability 和 residual risk 的 standing document |
| Two-tier mitigation | “Unilateral vs industry” | 分离 Anthropic commitments 与 industry recommendations |
| Pause commitment | “2023 clause” | 明确暂停 training 的承诺；v3.0 中移除 |
| SaferAI rating | “Independent RSP grade” | 第三方 rubric；v3.0 得分 1.9（v2 为 2.2） |

## 延伸阅读

- [Anthropic — Responsible Scaling Policy v3.0](https://anthropic.com/responsible-scaling-policy/rsp-v3-0) — 完整 32 页 policy。
- [Anthropic — RSP v3.0 announcement](https://www.anthropic.com/news/responsible-scaling-policy-v3) — v2 到 v3 的变化总结。
- [Anthropic — Frontier Safety Roadmap](https://www.anthropic.com/research/frontier-safety) — RSP v3.0 链接的 standing document。
- [Anthropic — Risk Report: Claude Opus 4.6](https://www.anthropic.com/research/risk-report-claude-opus-4-6) — 当前 frontier model 的 retrospective。
- [Anthropic — Measuring agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) — 将 AI R&D-4 与 measured autonomy 连接起来。
