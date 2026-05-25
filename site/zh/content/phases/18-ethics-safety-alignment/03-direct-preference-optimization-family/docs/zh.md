# Direct Preference Optimization 家族

> Rafailov et al.（2023）表明，RLHF 的最优解可以用偏好数据写成闭式形式，所以你可以跳过显式 reward model，直接优化 policy。这个洞见催生了一个家族：IPO、KTO、SimPO、ORPO、BPO，各自修补 DPO 的一个 failure mode。到 2026 年，direct alignment algorithm 在 frontier post-training run 中的使用已经超过 PPO。但第 2 课的 over-optimization curve 仍然适用：DAA 没有逃离 Goodhart，它们只是改变了被咬的位置。

**类型：** 学习
**语言：** Python（stdlib，六种 preference-loss 比较器）
**前置要求：** 阶段 18 · 01（InstructGPT），阶段 18 · 02（Reward hacking），阶段 10 · 08（DPO basics）
**时间：** ~75 分钟

## 学习目标

- 从带 KL 的 RLHF 最优解推导 DPO 的闭式形式。
- 说明 IPO、KTO、SimPO、ORPO、BPO 各自修复了 DPO 的哪种 failure mode。
- 区分 “implicit reward gap” 和 “preference strength”，并解释为什么 IPO 的 identity mapping 重要。
- 解释为什么 Rafailov et al.（NeurIPS 2024）证明，即使没有显式 RM，DAA 仍然会 over-optimize。

## 问题

RLHF objective（第 1 课）：

```
max_pi E_{x,y~pi} [ r(x, y) ] - beta * KL(pi || pi_ref)
```

有一个已知最优解：

```
pi*(y|x) = (1/Z(x)) * pi_ref(y|x) * exp(r(x, y) / beta)
```

所以 reward 可以由最优 policy 与 reference 的比值隐式定义：

```
r(x, y) = beta * log(pi*(y|x) / pi_ref(y|x)) + beta * log Z(x)
```

把它代入 Bradley-Terry preference likelihood，partition function `Z(x)` 会抵消，因为它只依赖 `x`。剩下的是只含 policy 参数的 loss，不再需要 reward model。这就是 DPO。

问题在于：推导假设最优解可达、偏好数据在分布内、reference policy 是真实的 mode anchor。这些假设都不完全成立。家族中的每个成员都在修复一个不同的被违反假设。

## 概念

### DPO（Rafailov et al., 2023）

```
L_DPO = -log sigmoid(
  beta * log(pi(y_w | x) / pi_ref(y_w | x))
  - beta * log(pi(y_l | x) / pi_ref(y_l | x))
)
```

会出什么问题：

- implicit reward gap `beta * (log(pi/pi_ref)_w - log(pi/pi_ref)_l)` 是无界的。一个很小的偏好也可能产生任意大的 gap。
- loss 会把 chosen 和 rejected log-prob 推向相反方向。只要 rejected 掉得更快，它就可以把 chosen 的绝对 log-prob 也往下推。这就是 Degraded Chosen Response 现象。
- Out-of-distribution preference（rare rare pair vs rare rare pair）会产生任意 implicit reward。

### IPO（Azar et al., 2024）

Identity Preference Optimization 用 preference probability 上的 identity mapping 替换 log-sigmoid。loss 变成有界 target 上的 squared-error：

```
L_IPO = (log(pi(y_w | x) / pi_ref(y_w | x)) - log(pi(y_l | x) / pi_ref(y_l | x)) - 1/(2 beta))^2
```

margin 由 `1/(2 beta)` 约束。Preference strength 与 implicit-reward gap 成比例。不会爆炸。

### KTO（Ethayarajh et al., 2024）

Kahneman-Tversky Optimization 完全丢掉 pairwise structure。给定一个单独标注输出，以及一个二元 “desirable” 或 “undesirable” 信号，它把它映射到 prospect-theory utility：

```
v(x, y) = sigma(beta * log(pi(y|x) / pi_ref(y|x)) - z_ref)
```

并对 gain 和 loss 使用不同权重（loss aversion）。好处：你可以使用 unpaired data，而这类数据丰富得多。

### SimPO（Meng et al., 2024）

Simple Preference Optimization 让训练信号与 generation 对齐。完全移除 reference policy，并按长度归一化 log-likelihood：

```
L_SimPO = -log sigmoid(
  (beta / |y_w|) * log pi(y_w | x)
  - (beta / |y_l|) * log pi(y_l | x)
  - gamma
)
```

用 margin `gamma` 稳定训练。长度归一化移除了利用 DPO length-bias failure mode 的激励（更长的 `y_w` 按构造会给出更大的 log-prob gap）。

### ORPO（Hong et al., 2024）

Odds-Ratio Preference Optimization 在标准 SFT negative log-likelihood 上加入 preference term：

```
L_ORPO = L_NLL(y_w) + lambda * L_OR
L_OR = -log sigmoid(log(odds(y_w) / odds(y_l)))
```

没有 reference policy，SFT 项就是 regularizer。从 base model 到 aligned model 单阶段训练。没有单独的 SFT checkpoint。

### BPO（ICLR 2026 submission, OpenReview id=b97EwMUWu7）

它识别了 Degraded Chosen Responses 问题：DPO 保持排名 `y_w > y_l`，但 `y_w` 的绝对 log-prob 可能下降。BPO 加入一行修正，惩罚 chosen response 上的向下移动。报告称在 Llama-3.1-8B-Instruct 的数学推理上，相比 DPO 准确率 +10.1%。

### 通用结果：DAA 仍然会 over-optimize

Rafailov et al. “Scaling Laws for Reward Model Overoptimization in Direct Alignment Algorithms”（NeurIPS 2024）用 DPO、IPO、SLiC 在多个数据集和 KL budget 上训练 policy。gold-reward-vs-KL 曲线呈现同样的 Gao et al. 峰值-坍缩形状。implicit reward 在训练中查询 out-of-distribution sample；KL regularization 并不能稳定这一点。

DAA 没有逃离 Goodhart。它们把被咬的位置从“reward model 被 over-optimized”换成了“reference policy ratio 被 over-optimized”。通用修复，更好的数据、ensemble、early stopping，对两者都适用。

### 如何选择（2026）

- 如果你有大量 paired preference data：用保守 beta 的 DPO；如果 length bias 明显，用 SimPO。
- 如果你有 unpaired binary feedback：KTO。
- 如果你想要从 base model 开始的单阶段 pipeline：ORPO。
- 如果你在 DPO 日志中看到 degraded chosen log-prob：BPO。
- 如果 preference strength 变化很大且 DPO 正在 saturating：IPO。

每个实验室都会在一组 battery 上跑完五种方法，并按任务选择赢家。没有理由认为数学推理和 safety 的最优方法相同。

## 使用它

`code/main.py` 在一个玩具偏好数据集上比较六种 loss（DPO、IPO、KTO、SimPO、ORPO、BPO），其中真实 preference strength 随 pair 变化。每个 loss 都针对同一个 500-pair 样本，用一个小 softmax policy 优化。它会绘制每种方法的最终 win rate、chosen-log-prob drift 和 implicit-reward spread。

## 交付它

本课会生成 `outputs/skill-preference-loss-selector.md`。给定数据集统计（paired vs unpaired、variable vs uniform preference strength、长度分布）和目标（单阶段或 SFT-then-preference），推荐一种 preference loss，并报告它防护的 failure mode。

## 练习

1. 运行 `code/main.py`。报告 DPO 和 BPO 的最终 chosen-log-prob drop。BPO 应该保留更高的 chosen absolute probability，请验证这一点。

2. 修改偏好数据，使所有 pair 都有相同强度。六种方法中哪一种最 robust？哪一种退化？解释 IPO 在这里的优势。

3. 让 rejected response 平均比 chosen 长 2 倍。在不改变其他设置的情况下，用数值展示 DPO 的长度 exploitation 和 SimPO 的修复。

4. Rafailov et al.（NeurIPS 2024）声称 DAA 会 over-optimize。复现一个单点版本：画出 chosen-minus-rejected KL divergence，并观察 DPO 在大 beta 下的 over-optimization。

5. 阅读 BPO 论文摘要（OpenReview b97EwMUWu7）。写下 BPO 加到 DPO 上的一行修正。与 `code/main.py` 中的实现核对。

## 关键词汇

| 术语 | 人们常说 | 实际含义 |
|------|-----------------|------------------------|
| DPO | “没有 reward model 的 RLHF” | 从闭式 RLHF 最优解推导出的 loss；只含 policy 参数 |
| Implicit reward | “log-ratio” | `beta * log(pi(y|x) / pi_ref(y|x))`，DPO 暗含的 reward |
| IPO | “bounded DPO” | 用 identity 替换 log-sigmoid；implicit reward gap 被 `1/(2 beta)` 封顶 |
| KTO | “unpaired DPO” | 对单标签使用 prospect-theory utility，并带 loss aversion |
| SimPO | “reference-free DPO” | 长度归一化 log-likelihood + margin；没有 reference policy |
| ORPO | “one-stage DPO” | NLL + odds-ratio preference term；从 base model 一次训练 |
| BPO | “chosen-preserving DPO” | DPO 加上对降低 chosen response 绝对 log-prob 的惩罚 |
| Degraded Chosen | “chosen goes down” | 只要 rejected 掉得更快，DPO 就会降低 chosen log-prob |
| DAA | “direct alignment algorithm” | 任何跳过显式 RM 的 preference-loss 方法 |

## 延伸阅读

- [Rafailov et al. — Direct Preference Optimization (NeurIPS 2023, arXiv:2305.18290)](https://arxiv.org/abs/2305.18290)
- [Azar et al. — A General Theoretical Paradigm to Understand Learning from Human Preferences (AISTATS 2024, arXiv:2310.12036)](https://arxiv.org/abs/2310.12036) — IPO
- [Ethayarajh et al. — KTO: Model Alignment as Prospect Theoretic Optimization (arXiv:2402.01306)](https://arxiv.org/abs/2402.01306)
- [Meng, Xia, Chen — SimPO (NeurIPS 2024, arXiv:2405.14734)](https://arxiv.org/abs/2405.14734)
- [Hong, Lee, Thorne — ORPO (EMNLP 2024, arXiv:2403.07691)](https://arxiv.org/abs/2403.07691)
- [BPO — Behavior Preservation Optimization (ICLR 2026 OpenReview b97EwMUWu7)](https://openreview.net/forum?id=b97EwMUWu7)
- [Rafailov et al. — Scaling Laws for RM Overoptimization in DAAs (NeurIPS 2024, arXiv:2406.02900)](https://arxiv.org/abs/2406.02900)
