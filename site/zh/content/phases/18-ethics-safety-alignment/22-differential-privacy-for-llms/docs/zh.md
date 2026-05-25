# 面向 LLM 的 Differential Privacy

> DP-SGD 仍然是标准方法，注入噪声的 gradient update 提供形式化的 (epsilon, delta) guarantee。Compute、memory 和 utility overhead 都很大；parameter-efficient DP fine-tuning（LoRA + DP-SGD）是 2025 年常见配置（ACM 2025）。两组证据彼此张力很大：canary-based membership inference（Duan et al., 2024）报告对 language model 成功有限；training-data extraction（Carlini et al., 2021; Nasr et al., 2025）恢复了大量 verbatim memorization。解决方案（arXiv:2503.06808，2025 年 3 月）：差距在于测量对象不同，即 inserted canaries vs “most extractable” data。新 canary design 支持不使用 shadow model 的 loss-based MIA，并给出了第一个针对真实数据上训练、具有现实 DP guarantee 的 LLM 的非平凡 DP audit。替代方案：PMixED（arXiv:2403.15638），通过 next-token distribution 上的 mixture of experts 在 inference time 进行 private prediction；DP synthetic data generation（Google Research 2024）。新兴攻击：Differential Privacy Reversal via LLM Feedback，即 confidence-score leakage。

**类型：** 构建
**语言：** Python（stdlib，DP-SGD noise-injection 与 ε-δ accountant demonstration）
**前置要求：** 阶段 01 · 09（information theory），阶段 10 · 01（large-model training）
**时间：** ~60 分钟

## 学习目标

- 定义 (epsilon, delta)-differential privacy，并说明 DP-SGD recipe。
- 解释 2024-2025 的张力：canary MIA 与 training-data extraction 给出不同图景。
- 描述 PMixED，以及为什么 inference-time private prediction 是 DP training 的替代方案。
- 描述 Differential Privacy Reversal via LLM Feedback attack。

## 问题

LLM 会记忆。Carlini et al. 2021 展示 production language model 会按需逐字复现 training text。DP 是形式化防御：训练使 output 对任何单个 training example 都可证明地不敏感。2024-2025 的证据显示 DP-SGD 是必要的，但 deployed ε value 未必匹配 threat model。

## 概念

### (ε, δ)-differential privacy

如果对任意只差一个 example 的两个 dataset，以及任意 event S，随机算法 M 满足：
P(M(D) in S) <= e^ε * P(M(D') in S) + δ.

解释：output distribution 足够接近（由 ε 参数化），使任何单个个体的贡献都不能被可靠推断，除非以概率 δ 例外。

### DP-SGD

Abadi et al. 2016。标准 recipe：
1. 采样 mini-batch。
2. 计算 per-example gradient。
3. 将每个 per-example gradient clip 到 threshold C。
4. 对 clipped gradient 求和，并加入 std 为 σ * C 的 Gaussian noise。
5. 用 noisy sum 更新参数。

Privacy cost 由 accountant 跟踪（Moments Accountant、Rényi DP accountant）。LLM 文献中报告的 ε value 会随 threat model、data sensitivity 和 utility target 大幅变化；不存在普遍“安全”的默认 ε。已发表示例在某些 LLM training setting 中大约跨越 ε ≈ 1-10，但这些只是说明性数字，不是推荐默认值。更低的 ε 通常需要更多噪声，并可能增加 utility loss。

### LoRA + DP-SGD

对 frontier model 做 full DP-SGD 代价过高。LoRA（Hu et al. 2022）把 gradient update 限制在小 adapter 上，减少 per-example gradient storage。LoRA + DP-SGD 是 2025 年常见配置。DP guarantee 适用于 adapter；base model 保持固定。

### 2024-2025 张力

两条证据线：

- **Canary MIA（Duan et al. 2024）。** 向 training data 插入 unique canary，测量 membership-inference attacker 是否能识别它们。报告对 language model 成功有限。暗示 MIA 很难。
- **Training-data extraction（Carlini 2021, Nasr et al. 2025）。** 用 prefix prompt 模型；测量它是否恢复 training 中的 verbatim text。报告大量 memorization。暗示相关意义上的 MIA 很容易。

2025 年 3 月的解决（arXiv:2503.06808）：两者测量不同东西。MIA 在 inserted canary 上问 “example e 是否在 D 中？” Extraction 问 “我能从 D 中恢复什么？” 对 privacy 真正重要的是 “most extractable” example；canary 会低估这一点，因为它们没有被优化为可 extract。

新的 canary design。不需要 shadow model 的 loss-based MIA。针对真实数据上训练、具有现实 DP guarantee 的 LLM 的第一个非平凡 DP audit。

### DP training 的替代方案

- **PMixED（arXiv:2403.15638）。** Inference time 的 private prediction。next-token distribution 上的 mixture of experts；每个 expert 看到一个 training data shard；aggregation 加噪以实现 DP。完全避免 DP training。
- **DP synthetic data generation（Google Research 2024）。** 用 DP-SGD 做 LoRA-fine-tune，采样 synthetic data，在 synthetic data 上训练 downstream classifier。

两者都绕开 full DP training 的 utility cost，但代价是不同 threat model。

### Differential Privacy Reversal via LLM Feedback

2025 年新兴攻击。把 DP-trained model 的 confidence score 当作 oracle，用来重新识别个体。即使 output 不泄漏，confidence distribution 仍可能泄漏。

Defense：不要暴露 confidence，或在暴露前 truncate/quantize。这是 (ε, δ)-DP training 之外的额外要求。

### 它在阶段 18 中的位置

第 20-21 课是 bias/fairness。第 22 课是 privacy。第 23 课是通过 watermarking 的 provenance。第 27 课覆盖 regulatory data-provenance layer。

## 使用它

`code/main.py` 在一个玩具 binary-classification dataset 上模拟 DP-SGD。你可以 sweep noise multiplier σ 和 clipping norm C，跟踪 (ε, δ) budget 与 accuracy cost。“canary attack” 会插入一个 unique training example，并测量 log-loss test 在 DP 前后是否能检测到它。

## 交付它

本课会生成 `outputs/skill-dp-audit.md`。给定一个 language model deployment 的 DP claim，它会 audit：(ε, δ) value、使用的 accountant、MIA evaluation protocol，以及 confidence-exposure vector 是否被评估。

## 练习

1. 运行 `code/main.py`。在 {0.5, 1.0, 2.0} 中 sweep σ，并报告 (ε, δ)-accuracy trade-off。识别 utility 坍缩的点。

2. 实现 canary insertion 和 log-loss test。测量 σ = 1.0 下，DP-SGD 前后的 detection rate。

3. 阅读 Nasr et al. 2025 关于 training-data extraction 的研究。为什么 extraction success 在 moderate ε 下不会坍缩？这对 MIA-as-evaluation 意味着什么？

4. 设计一个使用 PMixED（arXiv:2403.15638）、完全在 inference time 运作的 deployment。PMixED 处理了什么 DP-SGD 没有处理的 threat model？

5. 勾勒 DP Reversal via LLM Feedback attack。设计一个限制 confidence-score leakage 的 countermeasure，并估计其 deployment cost。

## 关键词汇

| 术语 | 人们常说 | 实际含义 |
|------|-----------------|------------------------|
| DP | “(ε, δ)-differential privacy” | 形式化 privacy：在 neighboring-dataset change 下 output distribution 接近 |
| DP-SGD | “noise-injected SGD” | Gradient clipping + Gaussian noise addition；标准 DP training |
| LoRA + DP-SGD | “efficient private fine-tune” | 在 low-rank adapter 上做 DP-SGD；2025 标准配置 |
| MIA | “membership inference” | 判断某个 example 是否在 training data 中的 attack |
| Canary | “inserted watermark example” | 用来测量 DP leakage 的 unique training example |
| PMixED | “private inference mixture” | 通过 next-token distribution 上的 mixture-of-experts 在 inference time 实现 DP |
| DP Reversal | “confidence leakage attack” | 使用模型 confidence 作为 re-identification oracle 的 attack |

## 延伸阅读

- [Abadi et al. — DP-SGD (arXiv:1607.00133)](https://arxiv.org/abs/1607.00133) — 标准 DP training algorithm
- [Carlini et al. — Extracting Training Data (arXiv:2012.07805)](https://arxiv.org/abs/2012.07805) — canonical extraction paper
- [Duan et al. — Canary MIA on LLMs (arXiv:2402.07841, 2024)](https://arxiv.org/abs/2402.07841) — limited-success MIA
- [Kowalczyk et al. — Auditing DP for LLMs (arXiv:2503.06808, March 2025)](https://arxiv.org/abs/2503.06808) — 张力的解决
- [PMixED (arXiv:2403.15638)](https://arxiv.org/abs/2403.15638) — inference-time private prediction
