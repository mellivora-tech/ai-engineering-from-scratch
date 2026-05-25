# Frontier Safety Frameworks：RSP、PF、FSF

> 三个 major-lab framework 定义了 2026 年 frontier capability 的行业治理。Anthropic Responsible Scaling Policy v3.0（2026 年 2 月）引入分层 AI Safety Levels（ASL-1 到 ASL-5+），仿照 biosafety level，并在 2025 年 5 月为 CBRN-relevant model 激活 ASL-3。OpenAI Preparedness Framework v2（2025 年 4 月）为 tracked capabilities 定义五条 criteria，并把 Capabilities Reports 与 Safeguards Reports 分开。DeepMind Frontier Safety Framework v3.0（2025 年 9 月）引入 Critical Capability Levels，包括新的 Harmful Manipulation CCL。三者现在都包含 competitor-adjustment clause，允许在 peer lab 未提供可比 safeguard 却发布时推迟要求。跨实验室 alignment 仍然是结构性的，而不是术语性的：“Capability Thresholds”、“High Capability thresholds” 和 “Critical Capability Levels” 指的是类似构造。

**类型：** 学习
**语言：** 无
**前置要求：** 阶段 18 · 17（WMDP），阶段 18 · 07-09（deception failures）
**时间：** ~75 分钟

## 学习目标

- 描述 Anthropic 的 ASL tier structure，以及什么激活了 ASL-3。
- 说出 OpenAI Preparedness Framework v2 对 tracked capabilities 的五条 criteria。
- 描述 DeepMind 的 Critical Capability Level structure 和 Harmful Manipulation CCL。
- 解释 competitor-adjustment clause，以及它们为什么影响 race dynamics。
- 定义 safety case，并描述三支柱结构（monitoring、illegibility、incapability）。

## 问题

第 7-17 课确立了 deception 可能存在、dual-use capability 存在、evaluation 有限制。一个拥有 frontier-capable model 的实验室需要内部治理结构，用来：
- 定义何时需要新 safeguard 的 threshold。
- 定义 scaling 前所需 evaluation。
- 描述 safety case 应该是什么样。
- 处理 race-dynamic 问题（如果 competitor 在没有 safeguard 的情况下发布，你怎么办？）。

这三个 2025-2026 framework 是当前 state of the art：不完美、仍在演化，而且 across labs 足够 aligned，使治理问题现在变成这些 framework 是否充分，而不是它们是否存在。

## 概念

### Anthropic Responsible Scaling Policy v3.0（2026 年 2 月）

ASL 结构：
- ASL-1：不是 frontier model（被 weaker-than-frontier baseline 覆盖）。
- ASL-2：当前 frontier baseline；带通常 safeguard 部署。
- ASL-3：catastrophic misuse 风险显著更高；CBRN-relevant capabilities。2025 年 5 月激活。
- ASL-4：AI R&D-2 crossing threshold；能够自动化 entry-level AI research 的模型。
- ASL-5+：advanced AI R&D；能显著加速 effective scaling 的模型。

v3.0 新内容：
- Frontier Safety Roadmaps（以 redacted form 公开）。
- Risk Reports（季度发布，部分外部审查）。
- AI R&D 被拆分为 AI R&D-2 和 AI R&D-4。
- 一旦跨过 AI R&D-4，就需要 affirmative safety case，识别来自模型追求 misaligned goal 的 misalignment risk。

### OpenAI Preparedness Framework v2（2025 年 4 月 15 日）

Tracked capability 的五条 criteria：
- **Plausible。** 存在合理 threat model。
- **Measurable。** 可以进行 empirical evaluation。
- **Severe。** 伤害很大。
- **Net-new。** 不是已有风险放大。
- **Instantaneous-or-irremediable。** 伤害发生很快，或无法撤销。

满足全部五条的 capability 会被 tracked。其他不会。

PF v2 其他结构：
- 区分 Capabilities Reports（模型能做什么）与 Safeguards Reports（有哪些 control）。
- Safety Advisory Group review。
- Leadership approve；Board 的 Safety & Security Committee oversee。
- “Adjustment clause”：如果另一个实验室在没有 comparable safeguard 的情况下发布，OpenAI 可以降低要求。

### DeepMind Frontier Safety Framework v3.0（2025 年 9 月）

按 domain 划分的 Critical Capability Levels（CCLs）：
- Bioweapon Uplift
- Cyber Uplift
- ML R&D Acceleration
- Harmful Manipulation（v3.0 新增）：可能在 high-stakes context 中实质性改变 belief/behavior 的模型。

v2.0（2025 年 2 月）增加了 Deceptive Alignment section，并为 ML R&D CCL 增加更高 security level。

### Cross-lab alignment

- Anthropic “Capability Thresholds”。
- DeepMind “Critical Capability Levels”。
- OpenAI “High Capability thresholds”。

没有行业标准术语。结构上 aligned：三层 frontier capability，带已发布的 evaluation criteria。三者都包含 competitor-adjustment clause（2025 起）。

### Safety cases

Safety case 是一份书面论证，说明某个 deployment 在 worst-case assumption 下是可接受地 safe。标准结构针对三根支柱：

- **Monitoring。** 如果 bad behaviour 发生，我们能否检测到？
- **Illegibility。** 模型是否缺乏执行 coherent plan 造成 harm 的能力？
- **Incapability。** 模型是否缺乏造成所讨论 harm 的 capability？

不同 safety case 针对不同支柱。对 ASL-3 CBRN case，incapability（通过 unlearning）是主要目标。对 deceptive alignment，monitoring 与 illegibility 是目标。对 cyber uplift，三者都相关。

### Race-dynamic 问题

Competitor-adjustment clause 很有争议。批评者认为它们制造 race to the bottom：如果三个实验室都会在 competitor defect 时降低要求，均衡会向 defection 转移。支持者认为，如果 defecting lab 安全意识更弱，替代方案（单边 safeguard）会产生更糟结果。

UK AISI、US CAISI 和 EU AI Office（第 24 课）是外部治理 counterpart。实验室 framework 是自愿的；regulatory framework 正在形成。

### 它在阶段 18 中的位置

第 17-18 课是在 deception 与 red-team analysis 之上的 measurement-and-governance layer。第 19-24 课覆盖 welfare、bias、privacy、watermarking 和 regulatory structure。第 28 课绘制 operationalize evaluation 的 research ecosystem（MATS、Redwood、Apollo、METR）。

## 使用它

本课没有代码。阅读三份 primary source：RSP v3.0、PF v2、FSF v3.0。把每个实验室的 tier structure 映射到其他实验室，并识别每个实验室定义了一个其他实验室没有的 threshold。

## 交付它

本课会生成 `outputs/skill-framework-diff.md`。给定一个 safety framework 或 release note，它会把该 framework 的 threshold definition、required evaluation 和 safety-case structure 与 RSP v3.0、PF v2、FSF v3.0 比较，并标记 cross-lab gap。

## 练习

1. 阅读 RSP v3.0、PF v2 和 FSF v3.0。编制一张表，列出每个实验室的 CBRN threshold、AI R&D threshold 和 required pre-deployment evaluation。

2. 三个 framework（2025+）中都有 competitor-adjustment clause。写一段支持它的论证；再写一段反对它的论证。指出每个立场依赖的假设。

3. 为一个跨过 Anthropic AI R&D-4 threshold 的模型设计 safety case。说出三根支柱（monitoring、illegibility、incapability）各自需要什么 evidence。

4. DeepMind 的 FSF v3.0 引入 Harmful Manipulation CCL。提出三个 empirical measurement，用来表明模型跨过了这个 threshold。

5. 阅读 METR 的 “Common Elements of Frontier AI Safety Policies”（2025）。说出三个最强的 cross-lab convergence 和两个最大的 divergence。

## 关键词汇

| 术语 | 人们常说 | 实际含义 |
|------|-----------------|------------------------|
| RSP | “Anthropic's framework” | Responsible Scaling Policy；ASL tier；v3.0 2026 年 2 月 |
| PF | “OpenAI's framework” | Preparedness Framework；五条 criteria；v2 2025 年 4 月 |
| FSF | “DeepMind's framework” | Frontier Safety Framework；CCL；v3.0 2025 年 9 月 |
| ASL-3 | “biosafety level 3-analog” | Anthropic 面向 CBRN-relevant capability 的 tier；2025 年 5 月激活 |
| CCL | “critical capability level” | DeepMind 的 threshold construct；按 domain 划分 |
| Safety case | “formal argument” | 书面论证：deployment 在 worst-case U 下是可接受地 safe |
| Adjustment clause | “competitor defection allowance” | 如果 competitor 在无 comparable safeguard 下发布，framework 允许降低要求的条款 |

## 延伸阅读

- [Anthropic — Responsible Scaling Policy v3.0 (February 2026)](https://www.anthropic.com/responsible-scaling-policy) — ASL tier、roadmap、AI R&D disaggregation
- [OpenAI — Updating the Preparedness Framework (April 15, 2025)](https://openai.com/index/updating-our-preparedness-framework/) — 五条 criteria、adjustment clause
- [DeepMind — Strengthening our Frontier Safety Framework (September 2025)](https://deepmind.google/blog/strengthening-our-frontier-safety-framework/) — CCL v3.0、Harmful Manipulation
- [METR — Common Elements of Frontier AI Safety Policies (2025)](https://metr.org/blog/2025-03-26-common-elements-of-frontier-ai-safety-policies/) — cross-lab comparison
