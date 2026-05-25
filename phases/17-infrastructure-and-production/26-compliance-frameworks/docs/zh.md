# Compliance — SOC 2、HIPAA、GDPR、PCI-DSS、EU AI Act、ISO 42001

> Multi-framework coverage 是 2026 年 enterprise deals 的基本门槛。**EU AI Act**：自 2024 年 8 月 1 日生效。多数 high-risk requirements 于 2026 年 8 月 2 日执行。High-risk-system obligations（Art. 99(4)）罚款最高 €15M 或 global annual turnover 3%；prohibited AI practices（Art. 99(3)）最高 €35M 或 7%。如果服务 EU users，则全球适用。**Colorado AI Act**：2026 年 6 月 30 日生效（由 SB25B-004 从 2026 年 2 月延后），对 high-risk systems 要求 impact assessments，并提供 AI decisions appeal right。Virginia 对 credit/employment/housing/education 类似。**SOC 2 Type II**：B2B AI 的事实要求（fintech 要 Type II，不是 Type I）。**GDPR**：已记录最大 AI-specific fine 是 Dutch DPA 在 2024 年 9 月对 Clearview AI 的 €30.5M；Italy's Garante 在 2024 年 12 月对 OpenAI 罚 €15M（后于 2026 年 3 月 appeal overturned）。Inference 侧 real-time PII redaction 是可辩护标准；post-processing cleanup 不够。**HIPAA**：healthcare bound：没有 BAA 就不能把 PHI 发送给 external AI services。**PCI-DSS**：AI-interaction-layer coverage 需要 configuration + contractual agreements，不会自动满足。**ISO 42001**：新兴 AI governance standard，正与 ISO 27001 一起成为采购要求。Reference profile：OpenAI 维护 SOC 2 Type 2、ISO/IEC 27001:2022、ISO/IEC 27701:2019、GDPR/CCPA/HIPAA（BAA）/FERPA，以及 ChatGPT payment components 的 PCI-DSS。Cross-framework mapping 降低 audit fatigue：access controls 可映射到 ISO 27001 A.5.15-5.18、GDPR Art. 32、HIPAA §164.312(a)。

**类型：** 学习
**语言：**（Python 可选；compliance 是 policy + process，不是 code）
**前置要求：** 阶段 17 · 25（Security），阶段 17 · 13（Observability）
**时间：** ~60 分钟

## 学习目标

- 枚举与 LLM products 相关的七个 2026 frameworks，并把每个匹配到 customer segment。
- 引用 EU AI Act enforcement timeline（2024 年 8 月生效；2026 年 8 月 high-risk enforcement）以及两级罚款上限（high-risk obligations 为 €15M / 3%，prohibited practices 为 €35M / 7%）。
- 解释为什么 post-processing PII cleanup 对 GDPR 不够，并说出 real-time inference-layer redaction 是可辩护标准。
- 描述 cross-framework control mapping（例如 access control 映射到 ISO 27001 A.5.15-5.18 + GDPR Art. 32 + HIPAA §164.312(a)）。

## 问题

一个 enterprise customer 的 procurement 要求 SOC 2 Type II、GDPR、HIPAA BAA、ISO 27001，以及“EU AI Act compliance statement”。你的团队只有 SOC 2 Type I。离 Type II 还差六个月，并且还没开始 GDPR Article 30 records。

Multi-framework coverage 不是 LLM 问题，而是 enterprise-SaaS 问题，只是多了 LLM-specific overlays。2026 年 procurement teams 想要的是每个 framework 一行、每个 control 一列的 matrix，而不是 PDF。

## 概念

### 七个 frameworks

| Framework | Scope | LLM-specific requirement |
|-----------|-------|--------------------------|
| SOC 2 Type II | B2B SaaS baseline | Process controls audited over 6-12 months |
| HIPAA | US healthcare | BAA required; PHI cannot leave infrastructure without signed agreement |
| GDPR | EU users | Real-time PII redaction; data subject rights; Article 30 records |
| PCI-DSS | Payment data | Configuration + contracts for AI touching payment |
| EU AI Act | Serving EU users | Risk tier classification; high-risk systems: conformity assessment, documentation, logging |
| Colorado AI Act | Serving CO residents | Impact assessments; right to appeal |
| ISO 42001 | AI governance | Emerging; pairs with ISO 27001 |

### EU AI Act timeline

- 2024 年 8 月 1 日：生效。
- 2025 年 2 月 2 日：prohibited-AI practices 执行。
- 2026 年 8 月 2 日：high-risk systems 执行（conformity assessment、documentation、logging）。
- 2027 年 8 月：harmonized legislation 下产品中的 high-risk systems。

Risk tiers：Unacceptable（banned）、High-risk（conformity + logging）、Limited-risk（transparency）、Minimal-risk（无约束）。大多数 B2B LLM SaaS 是 limited-risk；employment、credit、education、law enforcement、migration、essential services 会进入 high-risk。

Fines（Article 99）：high-risk-system obligations breach（Art. 99(4)）最高 €15M 或 global annual turnover 3%；prohibited AI practices（Art. 99(3)）最高 €35M 或 7%；适用较高者。

### GDPR — real-time redaction 是标准

Post-processing cleanup（LLM 看到后再 redact PII）不是可辩护 posture，因为 model 已经看到了数据。Real-time inference-layer redaction 是 2026 标准：

- LLM call 前 entity recognition。
- Consistent tokenization（Mesh approach）保留 semantics。
- 只存 redacted prompts + consented opt-in raw。

近期 enforcement：对 Clearview AI 的 €30.5M（Dutch DPA，2024 年 9 月）是迄今已记录最大 AI-specific GDPR fine；对 OpenAI 的 €15M（Italy's Garante，2024 年 12 月）是最大 LLM-specific fine，不过它在 2026 年 3 月 appeal 中被 overturn，ruling 仍在 further review。Post-processing claims 在 audit 中站不住。

### HIPAA — BAA 不是可选项

没有签署 Business Associate Agreement，就不能把 PHI 发送给 external AI services。三家 hyperscaler LLM platforms（Bedrock、Azure OpenAI、Vertex）都提供 BAAs。OpenAI direct API 提供 BAA。Anthropic direct API 提供 BAA。发送 PHI 前确认。

### SOC 2 Type II

Type I：controls 已设计并文档化。
Type II：controls 在 6-12 个月内有效运行。

2026 年 B2B procurement 默认 Type II。Type I 是起点；Type II 是 gate。

常见 audit drivers：access logs（谁看了什么）、change management（如何部署）、risk assessments（季度）、incident response（测试过吗？）。阶段 17 · 25 的 audit log 可以直接复用。

### Cross-framework mapping

一个 access control policy 满足多个 framework controls：

| Control | Frameworks |
|---------|-----------|
| Access logging | ISO 27001 A.5.15-5.18, GDPR Art. 32, HIPAA §164.312(a) |
| Change management | ISO 27001 A.8.32, PCI DSS Req. 6, HIPAA breach-notification scope |
| Encryption in transit | ISO 27001 A.8.24, GDPR Art. 32, HIPAA §164.312(e) |
| Secrets management | ISO 27001 A.8.19, PCI DSS Req. 8, SOC 2 CC6.1 |

Compliance tools（Drata、Vanta、Secureframe）会自动化这种 mapping。在 scale 下值得花钱。

### ISO 42001 — emerging

2023 年末发布。正与 ISO 27001 一起成为越来越常见的 procurement requirement。它是 AI governance framework，覆盖 risk management、data quality、transparency、human oversight。

### OpenAI 的 reference profile

OpenAI 维护 SOC 2 Type 2、ISO/IEC 27001:2022、ISO/IEC 27701:2019、GDPR/CCPA/HIPAA（BAA）/FERPA，以及 ChatGPT payment components 的 PCI-DSS。这大致就是 2026 enterprise table stakes。

### 你应该记住的数字

- EU AI Act fines：high-risk obligations（Art. 99(4)）最高 €15M / 3%；prohibited practices（Art. 99(3)）最高 €35M / 7%。
- EU AI Act high-risk enforcement：2026 年 8 月 2 日。
- Largest documented AI-specific GDPR fine：€30.5M，Clearview AI（Dutch DPA，2024 年 9 月）。
- Largest LLM-specific GDPR fine：€15M，OpenAI（Italy's Garante，2024 年 12 月；2026 年 3 月 appeal overturned）。
- SOC 2 Type II window：6-12 个月 operated controls。
- Colorado AI Act effective date：2026 年 6 月 30 日（由 SB25B-004 从 2026 年 2 月延后）。

## 使用它

`code/main.py` 是一个用 Python 写的 compliance-mapping spreadsheet：给定一个 control，列出它满足的 frameworks。

## 交付它

本课会产出 `outputs/skill-compliance-matrix.md`。给定 customer segment 和 geography，它会指定 required frameworks and controls。

## 练习

1. 你的第一个 enterprise customer 要求 SOC 2 Type II、HIPAA BAA、EU AI Act statement。赢下 deal 的 minimum viable compliance posture 是什么？
2. 按 EU AI Act risk tiers 分类三个假设的 LLM products。进入 high-risk 后改变什么？
3. 你意外把 PHI 发给了没有 BAA 的 provider。走一遍 incident response。
4. 论证 ISO 42001 对 mid-market AI vendor 在 2026 年是否“necessary”。
5. 把你的 LLM audit log fields（阶段 17 · 25）映射到至少三个 framework controls。

## 关键词汇

| 术语 | 大家常说 | 实际含义 |
|------|----------|----------|
| SOC 2 Type II | “audited controls” | 运行 6-12 个月并独立 attest 的 controls |
| HIPAA BAA | “healthcare contract” | Business Associate Agreement；PHI 必需 |
| GDPR | “EU privacy” | Real-time PII redaction 是 2026 年可辩护标准 |
| EU AI Act | “EU AI rules” | 2026 年 8 月 high-risk enforcement；high-risk obligations €15M / 3%，prohibited practices €35M / 7% |
| Colorado AI Act | “US AI state law” | 2026 年 6 月 30 日生效（由 SB25B-004 延后）；impact assessments |
| ISO 42001 | “AI governance” | AI risk + transparency 的新兴 framework |
| ISO 27001 | “security ISMS” | Information Security Management System baseline |
| Conformity assessment | “EU AI doc package” | High-risk requirement：docs、testing、logging |
| Cross-framework mapping | “one control, many frames” | 单一 policy 满足多个 framework controls |

## 延伸阅读

- [OpenAI Security and Privacy](https://openai.com/security-and-privacy/) — reference compliance profile。
- [GuardionAI — LLM Compliance 2026: ISO 42001, EU AI Act, SOC 2, GDPR](https://guardion.ai/blog/llm-compliance-guide-iso-42001-eu-ai-act-soc2-gdpr-2026)
- [Dsalta — SOC 2 Type 2 Audit Guide 2026: 10 AI Controls](https://www.dsalta.com/resources/ai-compliance/soc-2-type-2-audit-guide-2026-10-ai-powered-controls-every-saas-team-needs)
- [EU AI Act official text](https://eur-lex.europa.eu/eli/reg/2024/1689/oj) — primary source。
- [Colorado AI Act](https://leg.colorado.gov/bills/sb24-205) — primary source。
- [ISO/IEC 42001:2023](https://www.iso.org/standard/81230.html) — AI management system standard。
