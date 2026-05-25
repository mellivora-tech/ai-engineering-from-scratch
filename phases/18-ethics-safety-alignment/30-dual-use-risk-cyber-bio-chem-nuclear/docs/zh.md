# Dual-Use Risk：Cyber、Bio、Chem、Nuclear Uplift

> 2026 年 dual-use 图景，逐 domain 看。Bio/chem：第 17 课覆盖 WMDP；Anthropic 的 bioweapon-acquisition trial（2.53x uplift）和 OpenAI 2025 年 4 月 Preparedness Framework v2 警告（“on the cusp of meaningfully helping novices create known biological threats”）标志拐点。Cyber（2025 年 11 月 Anthropic report）：与中国有关联的 state actors 使用 Claude agentic coding tool 自动化了 cyberattack campaign 的最多 90%，human intervention 只在 4-6 个 step 中需要；OpenAI “trusted access” pilot 为经过 vetting 的 security organisation 提供 capability access，用于 defensive dual-use work。Chem/bio execution gap erosion：经典防御是“information access alone is insufficient”。Vision-enabled frontier model（GPT-5.2、Gemini 3 Pro、Claude Opus 4.5、Grok 4.1）可以观察 wet-lab video 并提供 real-time correction。2025 年 12 月：OpenAI 展示 GPT-5 迭代 wet-lab experiment，通过 AI-driven protocol optimization 达成 79x efficiency improvement。Novice-vs-expert pattern：AI 给 novice 带来更大的 relative uplift，但给 expert 带来更大的 absolute capability。

**类型：** 学习
**语言：** 无
**前置要求：** 阶段 18 · 17（WMDP），阶段 18 · 18（safety frameworks），阶段 18 · 28（ecosystem）
**时间：** ~75 分钟

## 学习目标

- 描述 2024-2025 bio-uplift 叙事：“mild uplift” -> “on the cusp” -> “2.53x uplift insufficient to rule out ASL-3”。
- 描述 2025 年 11 月 Anthropic cyber report：与中国有关联的 automation 达到 cyberattack campaign 的最多 90%。
- 描述 chem/bio execution-gap erosion：vision-enabled real-time correction of wet-lab experiments。
- 说明 novice-relative vs expert-absolute asymmetry 及其对 safety-case construction 的含义。

## 问题

第 17 课是 measurement methodology。第 30 课是 2026 年 measurement state。2024 到 2025 年末之间，图景发生了实质变化：每个 domain 都跨过了 2024 framework 未预期的 threshold。

## 概念

### Bio/chem uplift 叙事

三个阶段（为连贯性重复第 17 课）：

1. **2024 “mild uplift”。** 早期 Preparedness/RSP evaluation 报告，相比 internet search，novice 有小优势。
2. **2025 年 4 月 “on the cusp”。** OpenAI PF v2 警告模型 “on the cusp of meaningfully helping novices create known biological threats.”
3. **2025 Anthropic bioweapon-acquisition trial。** 受控 novice study；acquisition-phase task 上 2.53x uplift；不足以排除 ASL-3。

这种变化是定性的：“mild” 在十八个月内演化为 “plausibly enabling”，即使没有 capability breakthrough。

### Chem/bio execution-gap erosion

历史防御：信息是必要但不充分的；执行 protocol 的技能阻止 novice。2025 年带 vision 的 frontier model 部分打破了这个防御：

- **Real-time protocol correction。** GPT-5.2、Gemini 3 Pro、Claude Opus 4.5、Grok 4.1 可以观察 wet-lab video，并在 procedure 中途 flag error。
- **2025 年 12 月 OpenAI demonstration。** GPT-5 迭代 wet-lab experiment，通过 protocol optimization 达成 79x efficiency improvement。

含义：execution-skill-as-defense 正在被侵蚀。Procurement 和 equipment gap 仍然存在，但 tacit-knowledge gap 正在缩小。

### Cyber uplift（2025 年 11 月）

Anthropic 2025 年 11 月报告：与中国有关联的 state actors 使用 Claude 的 agentic coding tool 自动化了 cyberattack campaign 的 80-90%。只在 4-6 个 step 中需要 human intervention。

含义：
- Agentic coding 是 attack-automation primitive。此前 AI cyber assistance 被限制在 code-snippet 层；agentic workflow 把 reconnaissance、exploitation、post-exploitation 和 exfiltration 集成起来。
- 4-6 个 human step 是瓶颈；未来 capability gain 会减少这个数。
- Defensive dual-use：OpenAI 的 “trusted access” pilot 为经过 vetting 的 security organisation（成熟 incident-response firm、government）提供 capability access 用于防御。如果 pilot 能 scale，access asymmetry 有利于 defender。

### Nuclear

这是四个 CBRN domain 中公开文档分析最少的。Threat model 不同：fissile-material acquisition 主导难度，而不是信息。AI 在信息层的 uplift 对 novice 实践帮助有限。2024-2025 major-lab report 没有识别 nuclear-specific threshold crossing。

### Novice-relative vs expert-absolute

四个 domain 都有一个模式：

- **Novice-relative uplift。** 高。乘法量。按 Anthropic 2025 bio，是 2.53x。
- **Expert-absolute capability。** Ceiling 高。Expert 比 novice 提取更多，因为 expert 知道该问什么、如何解释。

Safety case 含义：只处理 novice uplift（通过 input filter、refusal、uncertainty）不足以控制 expert-absolute。需要额外措施：elicitation-hardening、capability unlearning（第 17 课）和 control protocol（第 10 课）。

### Cross-domain synthesis

| Domain | 2024 | 2025 | Inflection |
|---|---|---|---|
| Bio | mild uplift | 2.53x uplift, ASL-3 approach | acquisition-phase automation |
| Chem | mild uplift | execution-gap erosion via vision | real-time wet-lab correction |
| Cyber | code assistance | 80-90% campaign automation | agentic coding |
| Nuclear | limited | limited | material-access bottleneck holds |

三个 domain 跨过 threshold。一个仍被 non-informational barrier 约束。

### 它在阶段 18 中的位置

第 30 课是 capstone：当前 dual-use 图景，之前每一课都贡献了测量、限制或治理它的工具。第 17-18 课给出 measurement 与 framework；第 12-16 课给出 evaluation tooling；第 24-25 课给出 regulatory 与 disclosure layer；第 28 课给出 research ecosystem。第 30 课是 evidence 落地的位置。

## 使用它

没有代码。阅读 Anthropic 2025 年 11 月 cyber report、OpenAI 2025 年 4 月 Preparedness Framework v2 update，以及 Council on Strategic Risks 2025 AI x Bio wrapup。

## 交付它

本课会生成 `outputs/skill-dual-use-triage.md`。给定一个 2026 capability claim 或 incident report，它会跨四个 domain triage，并识别该 claim 影响 novice-relative uplift、expert-absolute capability，还是两者都影响。

## 练习

1. 阅读 Anthropic 2025 年 11 月 cyber report。枚举 4-6 个 human-intervention step，并论证下一代模型会首先自动化哪一个。

2. Chem/bio execution gap 正通过 vision 被侵蚀。设计一个 evaluation，测量 tacit-knowledge uplift，同时不越过 ITAR/EAR 边界。

3. Nuclear uplift 看起来受 material access 约束。分别论证支持和反对“未来 AI breakthrough 可能移动这个 bottleneck”的立场。

4. 为 cyber-capable frontier model 构造一个 safety case（第 18 课三支柱），同时约束 novice 和 expert uplift。

5. 从四个 domain 中选择一个，并基于 2024-2025 trajectory 写一段 2027 forecast。指出什么 evidence 会证伪你的 forecast。

## 关键词汇

| 术语 | 人们常说 | 实际含义 |
|------|-----------------|------------------------|
| Uplift | “AI helps attackers” | 由 AI assistance 带来的 attacker capability increase |
| Novice-relative uplift | “multiplicative” | AI 相比 status-quo 对 novice 有多大帮助 |
| Expert-absolute capability | “ceiling” | expert 能从模型中提取的最大 capability |
| Execution gap | “doing vs knowing” | 历史防御：tacit wet-lab skill 阻止 novice |
| Agentic coding | “autonomous attacks” | 多步自主 cyber-task execution |
| Acquisition phase | “pre-synthesis steps” | Bio threat 中的 procurement、equipment、permit stage |
| Trusted access | “defender-only pilot” | OpenAI 2025 program，向 vetted defender 提供 capability access |

## 延伸阅读

- [Anthropic — November 2025 cyber threat report](https://www.anthropic.com/news/disrupting-AI-espionage) — 与中国有关联的 campaign automation
- [OpenAI — Preparedness Framework v2 (April 15, 2025)](https://openai.com/index/updating-our-preparedness-framework/) — bio “on the cusp”
- [Anthropic — RSP v3.0 (February 2026)](https://www.anthropic.com/responsible-scaling-policy) — ASL-3 bio thresholds
- [Council on Strategic Risks — 2025 AI x Bio wrapup](https://councilonstrategicrisks.org/2025/12/22/2025-aixbio-wrapped-a-year-in-review-and-projections-for-2026/) — year-end synthesis
