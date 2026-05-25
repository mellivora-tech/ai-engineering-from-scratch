# Fairness Criteria：Group、Individual、Counterfactual

> 三个家族构成 fairness 文献结构。Group fairness：demographic parity、equalized odds、conditional use accuracy equality，也就是 protected group 之间平均 rate 相等。Individual fairness（Dwork et al. 2012）：similar individuals receive similar decisions；decision map 满足 Lipschitz condition。Counterfactual fairness（Kusner et al. 2017）：如果在 counterfactually altered sensitive attribute 时 decision 不变，那么这个 decision 对个体公平。2024 理论结果（NeurIPS 2024）：存在内在 CF-vs-accuracy trade-off；一个 model-agnostic method 可以把 optimal-but-unfair predictor 转换为 CF predictor，并有 bounded accuracy loss。Backtracking counterfactuals（arXiv:2401.13935，2024 年 1 月）：一种新范式，避免要求对 legally protected attribute 进行 intervention。Philosophical reconciliation（ICLR Blogposts 2024）：在 causal graph 下，满足某些 group fairness measure 会 entail counterfactual fairness。

**类型：** 学习
**语言：** Python（stdlib，three-criteria comparison）
**前置要求：** 阶段 18 · 20（bias），阶段 02（classical ML）
**时间：** ~60 分钟

## 学习目标

- 说出三种 group-fairness criterion（demographic parity、equalized odds、conditional use accuracy equality）和一个 impossibility result。
- 通过 Dwork et al. 2012 的 Lipschitz formulation 描述 individual fairness。
- 描述 counterfactual fairness 及其对 causal graph 的依赖。
- 解释 backtracking counterfactuals，以及为什么它们绕过了 intervention-on-protected-attribute 问题。

## 问题

第 20 课讨论的是测量 bias。第 21 课讨论的是 measurement 应该服务于哪一种 fairness standard。三个家族给出结构上不同的标准：一个模型可以 group-fair 但 individual-unfair，也可以 counterfactually fair 但 group-unfair。选择标准是政策决策；没有一种标准是普遍最优的。

## 概念

### Group fairness

- **Demographic parity。** P(Y=1 | A=a) = P(Y=1 | A=a') 对所有 group 成立。Acceptance rate 相等。
- **Equalized odds。** P(Y=1 | Y*=y, A=a) = P(Y=1 | Y*=y, A=a')。Group 之间 TPR 和 FPR 相等。
- **Conditional use accuracy equality。** P(Y*=y | Y=y, A=a) = P(Y*=y | Y=y, A=a')。Group 之间 predictive value 相等。

Impossibility（Chouldechova, Kleinberg-Mullainathan-Raghavan 2017）：在 base rate 不相等时，这三者不能同时满足。

### Individual fairness

Dwork et al. 2012。如果对于某个 task-specific similarity metric d，decision map f 满足 |f(x) - f(x')| <= L * d(x, x')，其中 L 是某个 Lipschitz constant，则 f 是 individually fair。相似个体得到相似 decision。

它需要定义 d。这是政策问题，不是统计问题。

### Counterfactual fairness

Kusner et al. 2017。在 population 的 causal model 下，如果当个体 i 的 sensitive attribute 被 counterfactually altered 时，decision 不变，则该 decision 对个体 i 是 counterfactually fair。

它需要 causal DAG。DAG 是建模选择。Counterfactual fairness 的正当性只与 DAG 一样强。

### CF-vs-accuracy trade-off

NeurIPS 2024 理论结果：counterfactual fairness 与 predictive accuracy 之间存在内在 trade-off。一个 model-agnostic method 可以把 optimal-but-unfair predictor 转换成 CF predictor，且 accuracy cost 有界。Accuracy cost 取决于 optimal unfair predictor 中 sensitive-attribute coefficient 的大小。

### Backtracking counterfactuals

arXiv:2401.13935（2024 年 1 月）。传统 counterfactual 需要对 sensitive attribute 做 intervention，比如“如果这个人是另一种 gender，decision 会改变吗”。在法律上，这有问题：classification law 中不能对 protected attribute 做 intervention。

Backtracking counterfactuals 反转方向：不是对 attribute 做 intervention，而是问这个个体的哪些实际 feature 组合会产生 counterfactual outcome。这绕过了法律上的 objection。

### Philosophical reconciliation

ICLR Blogposts 2024。有了 causal graph，满足某些 group-fairness measure 会 entail counterfactual fairness。三个家族并非正交；它们是同一底层 causal structure 的不同侧面。

这不能解决 impossibility theorem（base rate 不相等仍然阻止 simultaneous group fairness）。但它表明，“group” 与 “individual / counterfactual” 之间的表面对立，部分是因为没有显式说明 causal model。

### 它在阶段 18 中的位置

第 20 课是 bias measurement。第 21 课是 fairness definition。第 22 课是 privacy（differential privacy）。第 23 课是 watermarking。这些是 allocation-adjacent 课程，与 deception-adjacent 的第 7-11 课互补。

## 使用它

`code/main.py` 构建一个带 sensitive attribute 和 unequal base rate 的玩具 binary-classification dataset。在简单 classifier 上计算 demographic parity、equalized odds 和 conditional use accuracy equality。观察三个 metric 不一致。应用一个针对 demographic parity 的 re-weighting，并观察它对另外两个 metric 的代价。

## 交付它

本课会生成 `outputs/skill-fairness-criterion.md`。给定一个 fairness claim 或 policy，它会识别被声明的是哪种 criterion，在所声称的 unequal base rate 下模型能否满足其余 criterion，以及该 claim 依赖什么 causal DAG。

## 练习

1. 运行 `code/main.py`。报告默认数据上的三个 group metric。应用针对 demographic-parity 的 re-weighting，并再次报告。

2. 使用 non-sensitive feature 上的 L2，实现 Dwork et al. 2012 individual-fairness metric。报告在 L=1 时有多少 pair 违反 Lipschitz。

3. 阅读 Kusner et al. 2017。为 resume scoring 构造一个简单 two-feature causal DAG，并识别其暗示的 counterfactual-fairness condition。

4. 2024 backtracking-counterfactuals 论文避免对 protected attribute 做 intervention。描述一个这对 legal compliance 很重要的场景。

5. ICLR 2024 reconciliation 认为 group fairness 与 counterfactual fairness 是同一结构的不同侧面。选择 `code/main.py` 中三个 criterion 的两个，并说明什么 causal assumption 会让它们等价。

## 关键词汇

| 术语 | 人们常说 | 实际含义 |
|------|-----------------|------------------------|
| Demographic parity | “equal rates” | P(Y=1 | A=a) 在 group 之间相等 |
| Equalized odds | “equal TPR/FPR” | group 之间 true-positive 和 false-positive rate 相等 |
| Conditional use accuracy | “equal PPV/NPV” | group 之间 predictive value 相等 |
| Individual fairness | “Lipschitz condition” | 相似个体得到相似 decision |
| Counterfactual fairness | “causal alteration invariance” | counterfactual attribute alteration 下 decision 不变 |
| Backtracking counterfactual | “explain via actuals” | 从 outcome 向后推理 counterfactual，而不是从 attribute 向前推理 |
| Impossibility theorem | “the three conflict” | Chouldechova / KMR 2017：base rate 不等时 group criterion 互斥 |

## 延伸阅读

- [Dwork et al. — Fairness through Awareness (arXiv:1104.3913)](https://arxiv.org/abs/1104.3913) — individual fairness
- [Kusner, Loftus, Russell, Silva — Counterfactual Fairness (arXiv:1703.06856)](https://arxiv.org/abs/1703.06856) — counterfactual fairness
- [Chouldechova — Fair prediction with disparate impact (arXiv:1703.00056)](https://arxiv.org/abs/1703.00056) — impossibility
- [Backtracking Counterfactuals (arXiv:2401.13935)](https://arxiv.org/abs/2401.13935) — protected-attribute intervention 的新范式
