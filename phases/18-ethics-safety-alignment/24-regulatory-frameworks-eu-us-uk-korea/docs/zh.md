# Regulatory Frameworks：EU、US、UK、Korea

> 四个主要 regulatory regime 定义了 2026 AI governance landscape。EU AI Act（2024 年 8 月 1 日生效）：prohibited practices 与 AI literacy 自 2025 年 2 月 2 日起适用；GPAI obligations 自 2025 年 8 月 2 日起适用；full applicability 与 Article 50 transparency 于 2026 年 8 月 2 日适用；legacy GPAI 与 embedded high-risk system 于 2027 年 8 月 2 日适用；罚款最高 1500 万欧元或全球 turnover 的 3%。GPAI Code of Practice（2025 年 7 月 10 日）：三章，Transparency、Copyright、Safety and Security，共 12 项 commitment；enforcement 从 2026 年 8 月开始。UK AISI -> AI Security Institute（2025 年 2 月）：rename 表明范围收窄。US AISI -> CAISI（2025 年 6 月）：NIST 下的 Center for AI Standards and Innovation；转向 pro-growth posture。Korean AI Framework Act（2024 年 12 月通过，2026 年 1 月生效）：Article 12 在 MSIT 下设立 AISI；要求 foreign AI company 设置 local representative，并对 high-impact 与 generative AI 进行 risk assessment 和 safety measures。

**类型：** 学习
**语言：** 无
**前置要求：** 阶段 18 · 18（frontier frameworks），阶段 18 · 27（data governance）
**时间：** ~75 分钟

## 学习目标

- 描述 EU AI Act risk tier（prohibited、high-risk、general-purpose、limited-risk）以及 2025 年 8 月 / 2026 年 8 月 / 2027 年 8 月 timeline。
- 描述 GPAI Code of Practice 的三章，以及每章约束哪些 provider。
- 描述 2025 年 rebrand：UK AISI -> AI Security Institute；US AISI -> CAISI；以及每个 rebrand 对 policy direction 的含义。
- 说明 Korea AI Framework Act 的核心 provision。

## 问题

Lab framework（第 18 课）是自愿的。Regulatory framework 是强制的。2024-2026 期间，第一波 comprehensive AI regulation 开始生效。Deployer 必须把 technical control 映射到 regulatory obligation；这种映射因 jurisdiction 而异。

## 概念

### EU AI Act

**2024 年 8 月 1 日生效。** Risk-tier 结构：

- **Prohibited practices**（Article 5）。Social scoring、公共场所实时 remote biometric identification（有 law-enforcement exception）、对 vulnerable group 的 exploitative manipulation。2025 年 2 月 2 日适用。
- **High-risk systems**（Annex III）。Employment、education、credit、law enforcement、justice、migration。要求 conformity assessment、risk management、logging、transparency。
- **General-Purpose AI（GPAI）models**。2025 年 8 月 2 日适用。所有 GPAI provider 都有 obligation；systemic-risk GPAI（>1e25 FLOP training compute）有额外 obligation。
- **Limited-risk systems**。Article 50 下的 transparency obligations（AI-generated content labelling）。2026 年 8 月 2 日适用。

Timeline：
- 2025 年 2 月 2 日：prohibited practices + AI literacy。
- 2025 年 8 月 2 日：GPAI + governance。
- 2026 年 8 月 2 日：full applicability + Article 50 transparency + 最高 1500 万欧元 / 全球 turnover 3% 的 penalty。
- 2027 年 8 月 2 日：legacy GPAI + embedded high-risk。

Commission 在 2025 年末提议把 high-risk timeline 调整为 16 个月。

### GPAI Code of Practice

2025 年 7 月 10 日发布。三章：

- **Transparency。** 所有 GPAI provider。
- **Copyright。** 所有 GPAI provider。
- **Safety and Security。** Systemic-risk GPAI provider（估计 5-15 家公司）。

共 12 项 commitment。由 AI Office 主持的 Signatory Taskforce 管理 implementation。Enforcement 从 2026 年 8 月 2 日开始；在此之前，good-faith compliance 会被接受。

### Article 50 的 Transparency Code

第一稿 2025 年 12 月 17 日。第二稿 2026 年 3 月。最终版 2026 年 6 月。覆盖 AI-generated content labelling，包括 deepfake，也就是要求第 23 课 watermarking technology 的 regulatory layer。

### UK AI Security Institute（2025 年 2 月）

从 AI Safety Institute 改名。Rebrand 收窄范围：去掉 algorithmic bias 和 free-speech framing；聚焦 frontier capability security。开源 Inspect evaluation tool（2024 年 5 月）。与 Redwood（第 10 课）合作 control safety case。

### US CAISI（2025 年 6 月）

Trump administration 把 NIST 的 AI Safety Institute 转为 Center for AI Standards and Innovation。根据 VP Vance 在 Paris AI Action Summit 的发言，转向 “pro-growth AI policies”。减少对 pre-deployment evaluation 的强调；强调标准和 innovation support。作为 EU AI Act regulatory posture 的国内 counterweight。

### Korean AI Framework Act

2024 年 12 月通过。2025 年 1 月颁布。2026 年 1 月生效。整合 19 项独立 AI bill。

Article 12 在 Ministry of Science and ICT（MSIT）下设立 AISI。要求：
- 在韩国运营的 foreign AI company 设置 local representative。
- 对 “high-impact” AI system 进行 risk assessment。
- 对 generative AI 和 high-impact AI 采取 safety measures。

第一个 comprehensive horizontal AI regulation 的亚洲 jurisdiction。

### Cross-jurisdiction dynamics

- EU：严格、risk-tiered、heavy penalties。Privacy-adjacent regulation 的 benchmark。
- US：偏向 innovation、decentralized，州级规则（例如 California AB 2013，第 27 课）填补 federal gap。
- UK：窄 security focus，强 evaluation infrastructure。
- Korea：MSIT-led，聚焦 foreign provider。

相互竞争的 regulatory philosophy。多 jurisdiction deployment 必须遵守最严格的规则，2026 年通常是 EU AI Act。

### 它在阶段 18 中的位置

第 18 课是实验室自愿治理；第 24 课是监管；第 25 课是 AI system 的新兴 CVE 类别；第 26-27 课覆盖 documentation（cards）和 training-data governance。

## 使用它

没有代码。阅读 EU AI Act primary source：regulation text、GPAI Code of Practice、UK AISI Inspect framework。把你的 deployment 映射到每个 jurisdiction 适用的 obligation。

## 交付它

本课会生成 `outputs/skill-regulatory-map.md`。给定一个 deployment description，它会映射适用 jurisdiction、各自的 tier classification、per-jurisdiction obligation 和 deadline structure。

## 练习

1. 阅读 EU AI Act（regulation 2024/1689）和 GPAI Code of Practice（2025 年 7 月 10 日）。识别适用于每个 GPAI provider 的三项 obligation，以及只适用于 systemic-risk GPAI 的三项。

2. 一个 deployment 由美国公司开发、运行在 EU infrastructure 上，并服务韩国用户。哪三个 jurisdiction 的规则适用？每个 substantive question 上哪条规则约束？

3. UK AI Security Institute 的 rename 收窄了范围。分别论证支持和反对更窄 framing。指出每个立场依赖的 policy assumption。

4. CAISI 的 “pro-growth” framing 偏离了 2022-2024 AI safety institute model。识别两个会由这种 framing 导出的可测量 policy shift。

5. Korea AI Framework Act 要求 foreign provider 设置 local representative。描述一个 Bay Area 公司服务韩国用户时的 operational implication。

## 关键词汇

| 术语 | 人们常说 | 实际含义 |
|------|-----------------|------------------------|
| EU AI Act | “regulation” | 基于 risk tier 的 horizontal AI regulation；2024 年 8 月生效 |
| GPAI | “general-purpose AI” | 大型 foundation model；systemic-risk subset 有额外 obligation |
| Article 50 | “transparency obligations” | AI-generated content labelling；2026 年 8 月适用 |
| UK AISI | “AI Security Institute” | 2025 年 2 月改名；更窄的 frontier-security focus |
| CAISI | “US center for AI standards” | 2025 年 6 月从 AI Safety Institute 改名；pro-growth posture |
| Korean AI Framework Act | “MSIT horizontal regulation” | 亚洲第一部 comprehensive AI law；2026 年 1 月生效 |
| Systemic-risk GPAI | “1e25 FLOP threshold” | 额外 obligation tier；估计约束 5-15 家公司 |

## 延伸阅读

- [EU AI Act text (Regulation 2024/1689)](https://digital-strategy.ec.europa.eu/en/policies/regulatory-framework-ai) — regulation 与 timeline
- [GPAI Code of Practice (10 July 2025)](https://digital-strategy.ec.europa.eu/en/library/final-version-general-purpose-ai-code-practice) — 三章 code
- [UK AI Security Institute (renamed Feb 2025)](https://www.gov.uk/government/organisations/ai-security-institute) — 官方页面
- [CSET — South Korea AI Framework Act Analysis (2025)](https://cset.georgetown.edu/publication/south-korea-ai-law-2025/) — Korean framework analysis
