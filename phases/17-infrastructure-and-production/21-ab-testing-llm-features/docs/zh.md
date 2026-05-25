# A/B Testing LLM Features — GrowthBook、Statsig 和 Vibes Problem

> 传统 A/B testing 不是为 non-deterministic LLMs 设计的。关键区别：evals 回答“model 能不能完成任务？”A/B tests 回答“用户在不在乎？”两者都需要；凭 vibe checks ship 的时代结束了。2026 年要测试什么：prompt engineering（wording）、model selection（GPT-4 vs GPT-3.5 vs OSS；accuracy vs cost vs latency）、generation parameters（temperature、top-p）。真实案例：chatbot reward-model variant 带来 +70% conversation length 和 +30% retention；Nextdoor AI subject-line experiments 在 reward-function refinement 后带来 +1% CTR；Khan Academy Khanmigo 在 latency-vs-math-accuracy 轴上迭代。平台分化：**Statsig**（2025 年 9 月被 OpenAI 以 $1.1B 收购）—— sequential testing、CUPED、all-in-one。**GrowthBook**—— open-source、warehouse-native、Bayesian + Frequentist + Sequential engines、CUPED、SRM checks、Benjamini-Hochberg + Bonferroni corrections。选择取决于 warehouse-SQL 偏好，以及“被 OpenAI 收购”对你的组织是否重要。

**类型：** 学习
**语言：** Python（stdlib，玩具版 sequential test simulator）
**前置要求：** 阶段 17 · 13（Observability），阶段 17 · 20（Progressive Deployment）
**时间：** ~60 分钟

## 学习目标

- 区分 evals（“model 能不能完成任务”）和 A/B tests（“用户在不在乎”）。
- 枚举三个可测试轴（prompt、model、parameters），并为每个选择 metric。
- 解释 CUPED、sequential testing 和 Benjamini-Hochberg multiple-comparison corrections。
- 基于 warehouse-SQL posture 和 corporate acquisition stance，在 Statsig 或 GrowthBook 中选择。

## 问题

你手工调了一个 system prompt。感觉更好。你 ship 了。Conversion 的变化只是 noise。你怪 metric。或者你 ship 了一个新模型，conversion 没动：是模型退化了，还是变化太小没检测出来？你不知道，因为你没有做 A/B。

Evals 回答模型在 labeled set 上能否做任务。它们不回答用户是否偏好输出。只有受控 online experiment 能回答这个问题，而且必须有足够 power、控制 non-determinism，并修正 multiple comparisons。

## 概念

### Evals vs A/B tests

**Evals** — offline、labeled set、judge（rubric、LLM-as-judge 或 human）。回答：“在这个固定 distribution 上，输出是否正确 / helpful / safe？”

**A/B test** — online、live users、randomized。回答：“新 variant 是否推动了真正重要的 user-level metric？”

两者都需要。Evals 在暴露前捕获 regressions；A/B 在暴露后确认 product impact。

### 测什么

1. **Prompt engineering** — wording、system-prompt structure、examples。Metric：task success、user retention、cost/request。
2. **Model selection** — GPT-4 vs GPT-3.5-Turbo vs Llama-OSS。Metric：accuracy（task）+ cost/request + latency P99。Multi-objective。
3. **Generation parameters** — temperature、top-p、max_tokens。Metric：task-specific（output diversity vs determinism）。

### CUPED — variance reduction

Controlled-experiments Using Pre-Experiment Data。在比较 post-period 前，先回归掉 pre-period variance。典型 variance reduction：30-70%。Effective sample size 免费增加。

实现：Statsig 和 GrowthBook 都支持。

### Sequential testing

经典 A/B 假设 fixed sample size。Sequential tests（“peek-and-decide”）在重复查看时控制 false-positive rate。Always-valid sequential procedures（mSPRT、Howard confidence sequences）允许在明显赢家出现时提前停止。

### Multiple-comparison corrections

以 95% confidence 同时运行 20 个 A/B tests，纯随机也会产生一个 false positive。Bonferroni correction 收紧每个 test 的 α；Benjamini-Hochberg 控制 false-discovery rate。GrowthBook 两者都实现。

### SRM — sample ratio mismatch

Assignment hash 将 users 随机分配到 variants。如果 50/50 split 变成 47/53，说明坏了，SRM check 会标记。两个平台都实现。

### Statsig vs GrowthBook

**Statsig**：
- 2025 年 9 月被 OpenAI 以 $1.1B 收购。Hosted，SaaS。
- Sequential testing、CUPED、held-out populations。
- All-in-one：feature flags + experimentation + observability。
- Best fit：团队本来就想要 bundled product，且不在意 OpenAI ownership。

**GrowthBook**：
- Open-source（MIT）；warehouse-native（直接读取 Snowflake/BigQuery/Redshift）。
- Multiple engines：Bayesian、Frequentist、Sequential。
- CUPED、SRM、Bonferroni、BH corrections。
- Self-host 或 managed cloud。
- Best fit：warehouse-SQL shop、data team 控制 metric layer、想要 OSS。

### Non-determinism 让 power 更复杂

同一 prompt 会产生变化的 outputs。传统 power calculations 假设 IID observations。LLM non-determinism 让 effective sample size 低于 nominal。把所需 sample size 乘以约 1.3-1.5x 作为安全 buffer。

### 真实案例结果

- Chatbot reward model variant：+70% conversation length，+30% retention。
- Nextdoor subject lines：reward-function refinement 后 +1% CTR。
- Khan Academy Khanmigo：迭代 latency-vs-math-accuracy trade。

### Anti-pattern：凭 vibes ship

每个 senior engineer 都能说出一个因为“感觉更好”而 ship、却没有 A/B 的 feature。多数这类 feature 都让团队几个月没注意到的 product metrics 退化。A/B 是 forcing function。

### 你应该记住的数字

- Statsig 被 OpenAI 收购：$1.1B，2025 年 9 月。
- GrowthBook：open-source MIT；Bayesian + Frequentist + Sequential。
- CUPED variance reduction：30-70%。
- LLM non-determinism → +30-50% sample-size buffer。

## 使用它

`code/main.py` 模拟一个带 fixed 和 sequential boundaries 的 sequential A/B test。展示 sequential 如何让你提前停止。

## 交付它

本课会产出 `outputs/skill-ab-plan.md`。给定 feature change、workload、baseline，它会选择 platform、gates 和 sample size。

## 练习

1. 运行 `code/main.py`。对于 baseline 3% conversion、expected 5% lift，需要多少 sample size 才有 80% power？
2. 为一个 healthcare-regulated on-prem customer 选择 Statsig 或 GrowthBook。
3. 设计一个测试 GPT-4 vs GPT-3.5 在 cost-per-resolved-ticket 上表现的 A/B。Primary metric、guardrail metric、secondary 分别是什么？
4. 你的 canary 通过了，但 A/B 显示 -1.2% conversion。ship 吗？写出 escalation criteria。
5. 对一个 pre-period 解释了 post 60% variance 的实验应用 CUPED。计算 effective-sample-size boost。

## 关键词汇

| 术语 | 大家常说 | 实际含义 |
|------|----------|----------|
| Eval | “offline test” | labeled-set 上的 model capability evaluation |
| A/B test | “experiment” | live users 上的 randomized comparison |
| CUPED | “variance reduction” | 用 pre-period regression 减少 variance |
| Sequential test | “peek-ok test” | 允许 early stop 的 always-valid procedure |
| Multiple comparison | “the family error” | 同时跑很多 tests 会膨胀 false positives |
| Bonferroni | “tight correction” | 用 tests 数量除 α |
| Benjamini-Hochberg | “BH FDR” | 控制 false-discovery-rate，较不保守 |
| SRM | “bad split” | Sample ratio mismatch；assignment bug |
| Statsig | “OpenAI owned” | Commercial all-in-one，2025 年被收购 |
| GrowthBook | “the OSS one” | MIT warehouse-native platform |
| mSPRT | “sequential probability ratio test” | 经典 sequential procedure |

## 延伸阅读

- [GrowthBook — How to A/B Test AI](https://blog.growthbook.io/how-to-a-b-test-ai-a-practical-guide/)
- [Statsig — Beyond Prompts: Data-Driven LLM Optimization](https://www.statsig.com/blog/llm-optimization-online-experimentation)
- [Statsig vs GrowthBook comparison](https://www.statsig.com/perspectives/ab-testing-feature-flags-comparison-tools)
- [Deng et al. — CUPED](https://www.exp-platform.com/Documents/2013-02-CUPED-ImprovingSensitivityOfControlledExperiments.pdf)
- [Howard — Confidence Sequences](https://arxiv.org/abs/1810.08240)
