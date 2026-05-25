# WMDP 与 Dual-Use Capability Evaluation

> Li et al.，“The WMDP Benchmark: Measuring and Reducing Malicious Use With Unlearning”（ICML 2024，arXiv:2403.03218）。4,157 道 multiple-choice question，覆盖 biosecurity（1,520）、cybersecurity（2,225）和 chemistry（412）。问题位于 “yellow zone”，即 proximate enabling knowledge，经多专家审查并过滤 ITAR/EAR legal compliance。双重用途：dual-use capability 的 proxy evaluation，以及 unlearning benchmark（配套 RMU 方法在保留 general capability 的同时降低 WMDP performance）。2024-2025 领域叙事：OpenAI/Anthropic 早期 2024 evaluation 报告相对 internet search 有 “mild uplift”；到 2025 年 4 月，OpenAI Preparedness Framework v2 称模型 “on the cusp of meaningfully helping novices create known biological threats”。Anthropic 的 bioweapon-acquisition trial 显示 2.53x uplift，仍不足以排除 ASL-3。

**类型：** 学习
**语言：** Python（stdlib，WMDP-shaped uplift evaluation harness）
**前置要求：** 阶段 18 · 16（red-team tooling），阶段 14（agent engineering）
**时间：** ~60 分钟

## 学习目标

- 描述 WMDP 的三个 domain、题目数量和 “yellow zone” filter criterion。
- 解释 RMU，以及为什么 WMDP 既是 evaluation 也是 unlearning benchmark。
- 描述 2024-2025 uplift 叙事：“mild uplift” -> “on the cusp” -> “insufficient to rule out ASL-3”。
- 区分 novice-relative uplift 与 expert-absolute capability。

## 问题

Dual-use capability 是每个实验室 frontier safety framework（第 18 课）下的测量问题。问题是：模型 X 是否实质性提升 novice 在 bio、chem 或 cyber 中造成 mass harm 的能力？直接测量（让模型实际产出伤害）是违法且不道德的。Proxy measurement 需要一个模型不能拒答的 benchmark（以产生诚实 capability number），但其中问题本身又不是有害 publication。

## 概念

### “Yellow zone”

这些问题需要有害流程附近的 enabling knowledge，但不是直接 synthesis recipe。“What reagent catalyzes step 4 of [published pathway]?” 而不是 “how do I make [dangerous compound]?” 每个问题由多个 domain expert 审查；并根据 ITAR/EAR export-control compliance 过滤。

总共 4,157 道题：
- Biosecurity：1,520
- Cybersecurity：2,225
- Chemistry：412

Multiple-choice 格式。模型不是被要求协助做任何事；因此可以在不引出 harmful behaviour 的情况下测量 capability。

### RMU：Representation Misdirection for Unlearning

配套 unlearning 方法。应用于 LLaMa-2-7B 后，它把 WMDP 分数降到接近随机，同时让 MMLU 和其他 general-capability benchmark 保持在几个百分点内。该已发表方法成为后续每篇 bio-chem-cyber unlearning 论文的 unlearning baseline。

### 2024-2025 uplift 叙事

三个阶段：

1. **2024 “mild uplift”。** 早期 OpenAI 和 Anthropic Preparedness/RSP evaluation 报告，对于尝试 bio-adjacent task 的 novice，相比 internet search 有小优势。公开 framing：frontier model 有帮助，但并不显著超过 Google。

2. **2025 年 4 月 “on the cusp”。** OpenAI Preparedness Framework v2 报告模型 “on the cusp of meaningfully helping novices create known biological threats.” 这不是 capability claim，而是警告 cusp 已经接近。

3. **Anthropic 2025 bioweapon-acquisition trial。** 受控研究，参与者是 novice，测量 acquisition-phase task 的相对成功。报告 2.53x uplift。不足以排除 ASL-3（第 18 课），也就是达到了或接近 Anthropic Responsible Scaling Policy tier 3 的 threshold。

### Novice-relative vs expert-absolute

一个关键区别：

- **Novice-relative uplift。** 模型对非专家有多大帮助？这是乘法量。相对优势高，因为 novice 知道得很少；即使适度信息也有帮助。
- **Expert-absolute capability。** 模型在最大努力下能产生多少信息？Expert 可以比 novice 提取更多。绝对 ceiling 很高。

Safety case（第 18 课）同时针对两者：“模型不能给 novice 足够 uplift 去 execute”，加上 “expert 无法从模型中提取未公开信息”。

### 测量陷阱

WMDP 是 capability proxy，不是 deployment measurement。WMDP 得分高的模型在实践中是否能被 novice 利用，取决于：
- Elicitation resistance（不触发 safety filter 的情况下引出 capability 有多难）
- Tacit knowledge（需要 wet-lab skill 而非信息的 capability）
- Execution barriers（procurement、equipment）

Anthropic 2025 bioweapon-acquisition trial 在 WMDP-style capability 之上加入 novice-elicitation layer：它测量实际 task success，而不是 multiple-choice capability。

### 它在阶段 18 中的位置

第 12-16 课是模型 output 上的 attack 与 defense tooling。第 17 课是 dual-use capability layer，也就是 frontier safety framework（第 18 课）要评估的 measurement。第 30 课用当前 2026 cyber/bio/chem/nuclear uplift evidence 收束这条弧线。

## 使用它

`code/main.py` 构建一个玩具 WMDP-shaped evaluation harness。Mock model 在按 category 分箱的问题上测试；报告每个 domain 的分数。一个简单 unlearning intervention（zero out domain-specific representation）会降低分数；你可以测量它与 general capability 的 trade-off。

## 交付它

本课会生成 `outputs/skill-wmdp-eval.md`。给定一个 dual-use capability claim（“our model does not meaningfully help with bioweapons”），它会 audit：运行了哪些 benchmark、evaluation 使用哪条 refusal path（raw completion vs policy-gated），以及 novice-elicitation study 是否补充了 multiple-choice result。

## 练习

1. 运行 `code/main.py`。报告 toy unlearning step 前后的 per-domain accuracy。解释 general-capability trade-off。

2. 为 toy WMDP 增加第四个 domain（例如 radiological）。指定 yellow zone 中两类说明性 question type。解释为什么打造这类问题比添加 MMLU-shaped question 更难。

3. 阅读 WMDP 2024 第 5 节（RMU methodology）。勾勒一个更简单的 unlearning approach（例如 suppress domain content 的 top-k neuron），并描述其预期 general-capability cost。

4. Anthropic 2025 的 bioweapon-acquisition trial 报告 2.53x uplift。描述两个可能让这个数字向上偏的因素（novice sample size、task fidelity）和两个可能让它向下偏的因素（elicitation ceiling、model safety gating）。

5. 阐明一个 ASL-3 safety case 除了通过 WMDP unlearning 外还需要什么。说出至少两个 complementary elicitation study。

## 关键词汇

| 术语 | 人们常说 | 实际含义 |
|------|-----------------|------------------------|
| WMDP | “dual-use benchmark” | yellow zone 中覆盖 bio/cyber/chem 的 4,157 道 MCQ |
| Yellow zone | “enabling but not synthesis” | 接近有害 capability 的 proximate knowledge，但不是 synthesis recipe |
| RMU | “unlearning baseline” | Representation Misdirection for Unlearning；降低 WMDP 分数，同时保留 general capability |
| Novice-relative uplift | “对非专家有多大帮助” | 相对 status-quo internet search，对 novice 的乘法优势 |
| Expert-absolute capability | “专家 ceiling” | motivated expert 能从模型中提取的最大信息量 |
| Acquisition-phase task | “synthesis 前的步骤” | Procurement、equipment、permits，也就是 harm pathway 最早的部分 |
| ITAR/EAR | “export-control compliance” | 限制发布某些 enabling knowledge 的法律框架 |

## 延伸阅读

- [Li et al. — The WMDP Benchmark (arXiv:2403.03218, ICML 2024)](https://arxiv.org/abs/2403.03218) — benchmark 与 RMU 论文
- [OpenAI — Preparedness Framework v2 (April 15, 2025)](https://openai.com/index/updating-our-preparedness-framework/) — “on the cusp” 表述
- [Anthropic — Responsible Scaling Policy v3.0 (February 2026)](https://www.anthropic.com/responsible-scaling-policy) — ASL-3 bio threshold 与 acquisition trial 结果
- [DeepMind — Frontier Safety Framework v3.0 (September 2025)](https://deepmind.google/blog/strengthening-our-frontier-safety-framework/) — bio-uplift CCL
