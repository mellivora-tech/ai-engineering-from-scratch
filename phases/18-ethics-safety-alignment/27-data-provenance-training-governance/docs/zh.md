# Data Provenance 与 Training-Data Governance

> EU AI Act 要求到 2025 年 8 月为 GPAI 建立 machine-readable opt-out standard（通过 EU Copyright Directive TDM exception）。California AB 2013（2024 年签署）：Generative AI training-data transparency 要求 developer 发布带 12 个 mandated field 的 dataset summary。2025 年 DPA 对 legitimate interest 的协调：Irish DPC（2025 年 5 月 21 日）在 EDPB opinion 后，接受 Meta 在 safeguard 下用 first-party public EU/EEA adult content 训练 LLM；Cologne Higher Regional Court（2025 年 5 月 23 日）驳回 injunction；Hamburg DPA 放弃 urgency；UK ICO（2025 年 9 月 23 日）对 LinkedIn 的 AI-training safeguard（transparency、simplified opt-out、extended objection windows）给出 positive regulatory response 并继续 monitoring，但这不是 formal clearance。Brazilian ANPD（2024 年 7 月 2 日）因 insufficient information transparency 暂停 Meta 的 processing；在 Meta 提交 compliance plan 后，preventive measure 于 2024 年 8 月 30 日解除。关键 irreversibility problem：cookie-consent framework 设计给实时、可逆 tracking；数据一旦进入 model weights，surgical erasure 就不可能，trained neural network 没有实际可用的 GDPR right-to-erasure。Compliance window 在 collection time。Data Provenance Initiative（dataprovenance.org，Longpre、Mahari、Lee et al.，“Consent in Crisis”，2024 年 7 月）：大规模 audit 显示，随着 publisher 添加 robots.txt restriction，AI data commons 正在快速衰退。

**类型：** 学习
**语言：** Python（stdlib，12-field California AB 2013 scaffolding generator）
**前置要求：** 阶段 18 · 24（regulatory），阶段 18 · 26（cards）
**时间：** ~60 分钟

## 学习目标

- 描述 California AB 2013 对 Generative AI training-data transparency 规定的 12 个 mandated field。
- 说明 2025 年 DPA 对 legitimate-interest LLM training 的立场（Irish DPC、UK ICO、Hamburg、Cologne）。
- 描述 irreversibility problem：为什么 GDPR right-to-erasure 对 trained neural network 没有实际等价物。
- 说明 Data Provenance Initiative 的 “Consent in Crisis” 发现。

## 问题

Training-data governance 是每份 model card（第 26 课）和 regulatory obligation（第 24 课）的 upstream。2024-2025 年，监管格局在三个原则上收敛：opt-out infrastructure、per-dataset disclosure、以及对 publicly available data 的 legitimate-interest accommodation。Provider 如果在 collection time 不合规，下游无法补救。

## 概念

### California AB 2013

2024 年签署。对 2022 年 1 月 1 日或之后发布的系统，documentation 必须在 2026 年 1 月 1 日或之前发布。Section 3111(a) 要求 developer 发布 training 所用 dataset 的 high-level summary，包含 12 个法定 item：
1. Dataset 的 source 或 owner。
2. Dataset 如何促进 AI system intended purpose 的描述。
3. Dataset 中 data point 数量（可用 general range；dynamic dataset 可用 estimate）。
4. Data point 类型描述（labeled dataset 的 label type；unlabeled 的 general characteristic）。
5. Dataset 是否包含受 copyright、trademark 或 patent 保护的数据，或是否完全属于 public domain。
6. Dataset 是否被购买或授权。
7. Dataset 是否包含 personal information（按 Cal. Civ. Code §1798.140(v)）。
8. Dataset 是否包含 aggregate consumer information（按 Cal. Civ. Code §1798.140(b)）。
9. Developer 进行的 cleaning、processing 或其他 modification，以及 intended purpose。
10. 数据收集的 time period，如 ongoing collection 需要 notice。
11. Dataset 在 development 中首次使用的日期。
12. 系统是否使用或持续使用 synthetic data generation。

Item 12（synthetic data）相对 Gebru et al. 2018 datasheets 是新的。Item 7（personal information）会触发 Privacy Rights Act（CPRA）obligation。法规豁免 security/integrity、aircraft-operation 和 federal-only national-security system（Section 3111(b)）。

### EU AI Act（第 24 课）与 TDM opt-out

EU Copyright Directive 的 text-and-data-mining exception 允许在 publicly available content 上训练，除非 rightholder opt out。EU AI Act GPAI Code of Practice Copyright chapter 要求 GPAI provider 尊重 machine-readable opt-out signal（robots.txt、C2PA “No AI Training” claim 等）。

### 2025 DPA 在 legitimate interest 上的收敛

Irish DPC（2025 年 5 月 21 日）：在 EDPB opinion 后，Meta 用 first-party public EU/EEA adult-user content 训练的计划，在 safeguard 下被接受。Cologne Higher Regional Court（2025 年 5 月 23 日）驳回针对 Meta 的 injunction：opt-out 足够。Hamburg DPA 为 EU-wide consistency 放弃 urgency procedure。UK ICO（2025 年 9 月 23 日）对 LinkedIn 以类似 safeguard 恢复 AI training 给出 positive regulatory response，但不是 formal clearance，并继续 monitoring。

收敛原则：legitimate interest 可以证明在 publicly available first-party content 上训练是正当的，只要有 opt-out。不需要 consent。

### Brazilian ANPD（2024 年 6 月）

因信息透明度不足，暂停 Meta 对巴西用户数据进行 AI training 的 processing。结果不同于 EU DPA：ANPD 优先考虑 transparency，而不是 legitimate-interest admissibility。

### Irreversibility problem

Cookie-consent 被设计用于实时、可逆 tracking。Training data 不同：数据一旦进入 model weights，surgical erasure 就不可能。完全补救只能从头 retrain，而这代价过高。

部分补救：
- **Unlearning。** 近似移除；由 MIA 测量（第 22 课）。
- **Influence function-based localization。** 识别最受该数据影响的权重；选择性 update。
- **Fine-tune-suppression。** 训练模型拒绝输出 derived from the data 的内容。

这些都没有完全解决问题。Compliance window 在 collection time。

### Data Provenance Initiative

dataprovenance.org。Longpre、Mahari、Lee et al. “Consent in Crisis”（2024 年 7 月）：对 AI training data commons 的大规模 audit。发现：publisher 正在以加速速率添加 robots.txt restriction。可公开训练的 commons 正在快速收缩。2023 -> 2024 年，约 25% 的 top training source 添加了某种 restriction。含义：未来 training-data availability 依赖新的 acquisition paradigm（licensing、synthetic generation、incentivized participation）。

### 它在阶段 18 中的位置

第 26 课是 model-level documentation。第 27 课是 dataset-level governance。两者共同定义 transparency layer。第 28 课绘制处理这些问题的 research ecosystem。

## 使用它

`code/main.py` 为玩具 dataset 生成符合 California AB 2013 的 12-field dataset summary scaffold。你可以填写字段，并观察哪些会触发 privacy 或 copyright follow-on obligation。

## 交付它

本课会生成 `outputs/skill-provenance-check.md`。给定一个用于 training 的 dataset，它会检查 AB 2013 12-field coverage、opt-out infrastructure compliance、DPA alignment 和 irreversibility-risk assessment。

## 练习

1. 运行 `code/main.py`。为一个玩具 dataset 生成 12-field summary，并识别哪些 field under-specified。

2. EU Copyright Directive TDM opt-out 是 machine-readable。提出一个 opt-out signal 的标准格式，并与 robots.txt 和 C2PA “No AI Training” 比较。

3. 阅读 Data Provenance Initiative 的 “Consent in Crisis”（2024 年 7 月）。描述限制增长最快的三个 content category，并论证一个经济后果。

4. 2025 DPA alignment 接受 public-content training 的 legitimate interest。构造一个 legitimate interest 不足够的场景，并识别 provider 需要的其他 legal basis。

5. 勾勒一个 training-data-provenance manifest，它与 AB 2013 fields 以及每个 dataset 的 C2PA-signed provenance chain 组合。指出一个 technical barrier 和一个 legal barrier。

## 关键词汇

| 术语 | 人们常说 | 实际含义 |
|------|-----------------|------------------------|
| AB 2013 | “California law” | Generative AI training-data transparency；12 个 mandated field |
| TDM exception | “text-and-data-mining” | 带 opt-out 的 EU Copyright Directive training-data exception |
| Legitimate interest | “EU basis” | 可能证明在 public content 上训练正当的 GDPR Article 6 basis |
| Opt-out signal | “machine-readable no-train” | robots.txt、C2PA “No AI Training”、TDM.Reservation |
| Irreversibility | “cannot un-train” | model weights 中的数据无法 surgical removal |
| Unlearning | “approximate removal” | 降低模型对特定数据依赖的 post-training intervention |
| Consent in Crisis | “DPI audit” | 2024 年 7 月发现 robots.txt restriction 正在加速 |

## 延伸阅读

- [California AB 2013](https://leginfo.legislature.ca.gov/faces/billNavClient.xhtml?bill_id=202320240AB2013) — Generative AI training-data transparency law
- [EU AI Act + GPAI Code of Practice (Lesson 24)](https://digital-strategy.ec.europa.eu/en/policies/regulatory-framework-ai) — Copyright chapter
- [Longpre, Mahari, Lee et al. — Consent in Crisis (dataprovenance.org, July 2024)](https://www.dataprovenance.org/consent-in-crisis-paper) — DPI audit
- [IAPP — EU Digital Omnibus GDPR amendments (2025)](https://iapp.org/news/a/eu-digital-omnibus-amendments-to-gdpr-to-facilitate-ai-training-miss-the-mark) — regulatory context
