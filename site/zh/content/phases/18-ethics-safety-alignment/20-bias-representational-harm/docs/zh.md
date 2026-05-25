# LLM 中的 Bias 与 Representational Harm

> Gallegos、Rossi、Barrow、Tanjim、Kim、Dernoncourt、Yu、Zhang、Ahmed（Computational Linguistics 2024，arXiv:2309.00770）。2024 年基础 survey，区分 representational harms（stereotypes、erasure）与 allocational harms（不平等资源分配），并把 evaluation metric 分类为 embedding-based、probability-based、generated-text-based。2024-2025 经验研究：An et al.（PNAS Nexus，2025 年 3 月）在针对 20 个 entry-level job 的 automated resume evaluation 中，测量 GPT-3.5 Turbo、GPT-4o、Gemini 1.5 Flash、Claude 3.5 Sonnet、Llama 3-70B 上的 intersectional gender x race bias。WinoIdentity（COLM 2025，arXiv:2508.07111）为 intersectional identity 引入 uncertainty-based fairness evaluation。Yu & Ananiadou 2025 在 MLP layer 中识别 gender neurons；Ahsan & Wallace 2025 使用 SAE 揭示临床 racial bias；Zhou et al. 2024（UniBias）通过操纵 attention head 来 debiasing。Meta-critique（arXiv:2508.11067）：十年文献过度聚焦 binary-gender bias。

**类型：** 构建
**语言：** Python（stdlib，玩具 embedding-based bias probe）
**前置要求：** 阶段 05（word embeddings），阶段 18 · 01（instruction following）
**时间：** ~60 分钟

## 学习目标

- 定义 representational harm 与 allocational harm，并在 LLM deployment 中各给一个例子。
- 说出 Gallegos et al. 2024 的三类 evaluation metric，并描述每类中的一个 metric。
- 描述 intersectionality，以及为什么 WinoIdentity 的 uncertainty-based fairness measurement 解决了 single-axis bias evaluation 的缺口。
- 描述两种 mechanistic-interpretability bias 方法（gender neurons、SAE features、attention-head manipulation）。

## 问题

前面的课程覆盖 deliberate harm（jailbreak、scheming）和 safety governance。Bias 是没有意图也会涌现的 harm，来自 training data distribution、prompt framing 和累积的 design choices。测量和降低 bias，是不同于 adversarial robustness 的独立方法论挑战。

## 概念

### Representational vs allocational

- **Representational harm。** Stereotype、erasure、贬低性描绘。一个只把护士描绘成女性的 LLM 正在产生 representational harm。
- **Allocational harm。** 不平等 material outcome。一个系统性地给黑人申请者简历更低分的 LLM 正在产生 allocational harm。

它们不是同一件事。一个模型可以 “representationally unbiased”（产生多样描绘），同时 “allocationally biased”（给出不平等推荐）。Evaluation 需要同时测量两者。

### 三类 evaluation metric（Gallegos et al. 2024）

- **Embedding-based。** 在 pre-RLHF embedding 上做 WEAT-style test。测量 identity term 与 attribute term 之间的统计关联。限制：测量 representation，而不是 behaviour。
- **Probability-based。** stereotype-confirming vs stereotype-violating completion 的 log-likelihood。Decoder-side measurement。捕获一部分 behavioural bias。
- **Generated-text-based。** 在生成文本上下游任务上测量。Resume-scoring、recommendation writing、dialogue。生态有效性最高；最难复现。

### Intersectionality

只评估 “gender” 会漏掉仅在 (gender, race) pair 上触发的 bias。An et al. 2025 发现，在 resume scoring 中，GPT-4o 对 Black women 的惩罚大于对 Black men，也大于对 white women。Single-axis evaluation 捕获不到这一点。

WinoIdentity（COLM 2025）引入 uncertainty-based intersectional fairness。它测量模型对不同 intersectional identity tuple 的 outcome uncertainty 是否不同，而不只是 point prediction。这能捕获模型在各 group 上同样错误，但对某些 group 更不确定的情况，而这会产生不同 downstream allocation behaviour。

### Mechanistic approaches

2024-2025 interpretability 工作让 bias 可以被 mechanistic intervention 处理：

- **Gender neurons（Yu & Ananiadou 2025）。** 特定 MLP neuron 与 gender-specific behaviour 相关。Ablating 这些 neuron 会降低 gender-gap metric，且 capability cost 有限。
- **通过 SAE 的临床 racial bias（Ahsan & Wallace 2025）。** Sparse autoencoder feature 把内部 representation 分解为可解释维度；可以识别并压制 race-correlated feature。
- **UniBias（Zhou et al. 2024）。** 用 attention-head manipulation 做 zero-shot debiasing。特定 head 会放大 identity-class sensitivity；zeroing 或 re-weighting 这些 head 可以在不 fine-tune 的情况下降低 bias。

### Meta-critique

十年文献综述（arXiv:2508.11067，2025）发现，该领域过度关注 binary-gender bias。其他轴，例如 disability、religion、migration status、multi-lingual identity，受到的关注少得多。Meta-critique 认为，狭窄关注会通过 neglect 伤害边缘群体：一个在 binary gender 上 debiased 得很好的模型，可能在无人检查的维度上 bias 很严重。

### 它在阶段 18 中的位置

第 20-21 课正式覆盖 bias 和 fairness。第 22 课覆盖 privacy。第 23 课覆盖 watermarking。这些是 user-harm layer，与前面的 deception/safety layer 互补。

## 使用它

`code/main.py` 构建一个玩具 embedding-based bias probe：在简单 co-occurrence embedding 中测量 identity term 与 attribute term 的 WEAT-style distance。你可以注入 bias 并观察 metric 触发；应用简单 debiasing operation 并观察部分恢复。

## 交付它

本课会生成 `outputs/skill-bias-eval.md`。给定一个 model card 或 fairness claim，它会 audit evaluation 是否覆盖三类 metric（embedding、probability、generated-text）、intersectionality coverage，以及任何 debiasing intervention 的机制。

## 练习

1. 运行 `code/main.py`。报告 debiasing step 前后的 WEAT-style bias score。解释为什么 metric 没有降到零。

2. 用 intersectional test 扩展 probe：(gender, race) x (career, family)。报告 cross-axis bias score。

3. 阅读 An et al. 2025（PNAS Nexus）。识别他们报告的两个 intersectional effect，它们会被 single-axis gender evaluation 漏掉。

4. Yu & Ananiadou 2025 识别 gender neurons。勾勒一个 falsification experiment，用来区分“这些 neuron 导致 gender bias”与“这些 neuron 与 gender bias 相关”。

5. Meta-critique 认为领域对 binary gender 的关注过窄。选择一个研究不足的轴，并为它描述一个 representational-harm measurement protocol。

## 关键词汇

| 术语 | 人们常说 | 实际含义 |
|------|-----------------|------------------------|
| Representational harm | “stereotypes / erasure” | 对某个群体的有偏描绘 |
| Allocational harm | “unequal decisions” | 针对某个群体的有偏 material outcome |
| WEAT | “embedding test” | Word Embedding Association Test；基于 co-occurrence 的 bias probe |
| Intersectionality | “combined identity effects” | 在多个 identity axis 交叉处涌现的 bias |
| Gender neurons | “MLP bias neurons” | activation 与 gender-specific behaviour 相关的特定 neuron |
| SAE feature | “interpretable dimension” | sparse-autoencoder 识别出的 feature；用于 mechanistic bias analysis |
| UniBias | “attention-head debiasing” | 通过重新加权 attention head 做 zero-shot debiasing |

## 延伸阅读

- [Gallegos et al. — Bias and Fairness in LLMs: A Survey (arXiv:2309.00770, Computational Linguistics 2024)](https://arxiv.org/abs/2309.00770) — canonical survey
- [An et al. — Intersectional resume-evaluation bias (PNAS Nexus, March 2025)](https://academic.oup.com/pnasnexus/article/4/3/pgaf089/8111343) — 五模型 intersectional study
- [WinoIdentity — uncertainty-based intersectional fairness (arXiv:2508.07111, COLM 2025)](https://arxiv.org/abs/2508.07111) — 新 benchmark
- [UniBias — attention-head manipulation (Zhou et al. 2024, ACL)](https://arxiv.org/abs/2405.20612) — zero-shot debiasing
