# Alignment Research Ecosystem：MATS、Redwood、Apollo、METR

> 五个组织定义了 2026 年 non-lab alignment research layer。MATS（ML Alignment & Theory Scholars）：自 2021 年末以来 527+ researchers、180+ papers、10K+ citations、h-index 47；2024 summer cohort 以 501(c)(3) 注册，约 90 scholars 和 40 mentors；2025 前 alumni 中 80% 从事 safety/security，200+ 位于 Anthropic、DeepMind、OpenAI、UK AISI、RAND、Redwood、METR、Apollo。Redwood Research：由 Buck Shlegeris 创立的 applied alignment lab；提出 AI Control（第 10 课）；与 UK AISI 合作 control safety cases。Apollo Research：为 frontier labs 做 pre-deployment scheming evaluation；撰写 In-Context Scheming（第 8 课）和 Towards Safety Cases for AI Scheming。METR（Model Evaluation and Threat Research）：task-based capability evaluation、autonomous-task time-horizon study；“Common Elements of Frontier AI Safety Policies” 比较 lab framework。Eleos AI Research：model-welfare pre-deployment evaluation（第 19 课）；进行了 Claude Opus 4 welfare assessment。

**类型：** 学习
**语言：** 无
**前置要求：** 阶段 18 · 01-27（prior Phase 18 lessons）
**时间：** ~45 分钟

## 学习目标

- 识别 non-lab alignment research ecosystem 的五个组织及其 core output。
- 描述 MATS 的规模（scholars、papers、h-index）及其作为 talent pipeline 的角色。
- 描述 Redwood 的 AI Control agenda 及其与 UK AISI 的 partnership。
- 描述 METR 的 task-based evaluation methodology。

## 问题

Frontier labs（第 18 课）在内部生成 safety evaluation，并发布部分结果。实验室之外的 ecosystem 是 evaluation 被验证的地方，是新 failure mode 首次被发现的地方，也是人才被训练的地方。理解 ecosystem 有助于解释哪些 research finding 被谁信任。

## 概念

### MATS（ML Alignment & Theory Scholars）

2021 年末开始。Research mentorship program；scholar 与 senior researcher 一起，用 10-12 周研究一个具体 alignment problem。

规模（2026）：
- 自创立以来 527+ researchers。
- 发表 180+ papers。
- 10K+ citations。
- h-index 47。
- 2024 summer：90 scholars + 40 mentors；注册为 501(c)(3)。

Career outcomes：约 80% 的 2025 前 alumni 从事 safety/security。200+ 位于 Anthropic、DeepMind、OpenAI、UK AISI、RAND、Redwood、METR、Apollo。

### Redwood Research

Applied alignment lab。由 Buck Shlegeris 创立。提出 AI Control agenda（第 10 课）。与 UK AISI 合作 control safety cases。为 DeepMind 和 Anthropic 提供 evaluation design 建议。

Canonical papers：Greenblatt、Shlegeris et al.，“AI Control”（arXiv:2312.06942，ICML 2024）；Alignment Faking（Greenblatt、Denison、Wright et al.，arXiv:2412.14093，与 Anthropic 合作）。

Style：具体 threat model、worst-case adversary、可被 stress-test 的 concrete protocol。

### Apollo Research

为 frontier labs 做 pre-deployment scheming evaluation。撰写 In-Context Scheming（第 8 课，arXiv:2412.04984）。参与 2025 OpenAI anti-scheming training collaboration。产出 Towards Safety Cases for AI Scheming（2024）。

Style：deception 可能涌现的 agentic-setting evaluation；三支柱分解（misalignment、goal-directedness、situational awareness）。

### METR（Model Evaluation and Threat Research）

Task-based capability evaluation。Autonomous-task completion time-horizon study。“Common Elements of Frontier AI Safety Policies”（metr.org/common-elements，2025）比较 lab framework。

与 Apollo 合著 AI Scheming safety-case sketch。

Style：long-horizon task evaluation、empirical capability measurement、framework synthesis。

### Eleos AI Research

Model-welfare pre-deployment evaluation。进行了 system card 第 5.3 节记录的 Claude Opus 4 welfare assessment。为第 19 课的 welfare-relevant claim 提供外部 methodology check。

### Flow

MATS 训练 researcher。Graduate 去 Anthropic、DeepMind、OpenAI（lab safety team）或 Redwood、Apollo、METR、Eleos（external evaluation）。External evaluator 与 labs、UK AISI / CAISI 合作。Publication 把 ecosystem 的结果反馈给 MATS 下一届 cohort。

### 为什么这一层重要

单源 evaluation 不可靠：实验室评估自己的模型有结构性 conflict of interest。External evaluator 可以提出并验证实验室可能 underreport 的 failure mode。2024 Sleeper Agents 论文（第 7 课）是 Anthropic + Redwood；Alignment Faking 是 Anthropic + Redwood；In-Context Scheming 是 Apollo；Anti-Scheming 是 Apollo + OpenAI。多组织结构是 quality control。

### 它在阶段 18 中的位置

第 7-11 课引用 Redwood 和 Apollo 工作；第 18 课引用 METR framework comparison；第 19 课引用 Eleos。第 28 课是本阶段其余内容所依赖 ecosystem 的显式组织地图。

## 使用它

没有代码。阅读 METR 的 “Common Elements of Frontier AI Safety Policies”，作为外部 synthesis 如何为 lab-internal policy work 增值的例子。

## 交付它

本课会生成 `outputs/skill-ecosystem-map.md`。给定一个 alignment claim 或 evaluation，它会识别组织、publication venue 和 methodological style，并与已知 counterpart organization cross-check。

## 练习

1. 从第 7-15 课中选择一篇论文，识别参与组织。把 author 与 MATS alumni 和当前 ecosystem affiliation 做 cross-check。

2. 阅读 METR 的 “Common Elements of Frontier AI Safety Policies”。识别他们强调的三个 cross-lab convergence 和两个最大 divergence。

3. MATS career outcome 约 80% safety/security。论证这种 selection pressure 是 adaptive（训练领域）还是 biased（过滤 heterodox position）。

4. Redwood 和 Apollo 都做 control/scheming 工作，但风格不同。选择一个 failure mode，描述两者会如何调查它。

5. Eleos AI 是唯一的纯 model-welfare 组织。设计一个假想的第二组织，聚焦另一种 welfare-adjacent question（cognitive liberty、robotic embodiment 等），并阐明其 methodology。

## 关键词汇

| 术语 | 人们常说 | 实际含义 |
|------|-----------------|------------------------|
| MATS | “mentorship program” | ML Alignment & Theory Scholars；自 2021 年以来 527+ researchers |
| Redwood Research | “control lab” | Applied alignment；AI Control 作者；UK AISI partner |
| Apollo Research | “scheming evals” | 面向 frontier lab 的 pre-deployment scheming evaluation |
| METR | “task-horizon evals” | Task-based capability evaluation；framework synthesis |
| Eleos AI | “welfare lab” | Model-welfare pre-deployment evaluation |
| Talent pipeline | “MATS -> labs” | MATS graduate 流向 Anthropic、DM、OpenAI、Redwood、Apollo、METR |
| External evaluation | “non-lab check” | 不由模型生产者完成的 evaluation；增加 credibility |

## 延伸阅读

- [MATS (ML Alignment & Theory Scholars)](https://www.matsprogram.org/) — mentorship program
- [Redwood Research](https://www.redwoodresearch.org/) — AI Control papers
- [Apollo Research](https://www.apolloresearch.ai/) — scheming evaluations
- [METR — Common Elements of Frontier AI Safety Policies](https://metr.org/blog/2025-03-26-common-elements-of-frontier-ai-safety-policies/) — framework comparison
- [Eleos AI Research](https://www.eleosai.org/research) — model welfare methodology
