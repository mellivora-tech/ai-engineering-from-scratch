# Constitutional AI 与 RLAIF

> Bai et al.（arXiv:2212.08073，2022）问了一个问题：如果我们用一个会阅读原则列表的 AI 来替代人类标注员，会怎样？Constitutional AI 有两个阶段：在 constitution 下做 self-critique 和 revision，然后用 AI Feedback 做 RL。这项技术创造了 RLAIF 这个术语，并进入了 Claude 1 post-training pipeline。2026 年 1 月 21 日，Anthropic 发布了重写版 Claude constitution：用 explanatory reasoning 取代 prescriptive rules，采用四层优先级层级，并首次由 major lab 正式承认对模型 moral status 的不确定性。以 CC0 1.0 发布。

**类型：** 学习
**语言：** Python（stdlib，玩具 self-critique-and-revise loop）
**前置要求：** 阶段 18 · 01（InstructGPT），阶段 18 · 02（Reward hacking）
**时间：** ~60 分钟

## 学习目标

- 描述 Constitutional AI 的两个阶段（critique-and-revise SFT、RL from AI feedback），以及 constitution 在每个阶段中的角色。
- 解释为什么用 AI labeler 替换 human preference labeler 不只是“更便宜的 RLHF”，它会改变 pipeline 的 failure mode。
- 总结 2026 Claude constitution 的四层优先级结构，以及相比 2023 rewrite 改变了什么。
- 描述 Constitutional Classifiers，以及 compute overhead 从 23.7%（v1）降到 ~1%（v2 / 2026）。

## 问题

RLHF 需要标注员。标注员慢、有偏、昂贵。你可以把标注员替换成一个会读取显式原则的模型，从而消除这个标注员。Bai et al. 的 Constitutional AI 是这个替换的第一个正式版本。它效果足够好，以至于现在每个 frontier lab 都在使用某种 AI-feedback post-training 变体。

问题在于：偏好信号现在由你正在训练的同类模型生成。labeler 的 bias（现在是原则加上 labeler model 的解释中的 bias）可能被放大，而不是被减弱。第 4 课的 sycophancy 论证仍然适用；labeler 只是搬进了 loop 内部。

## 概念

### 第 1 阶段：监督式 self-critique 和 revision

从一个 helpful 但尚未 harmless 的 SFT model 开始。给定一个 red-team prompt，模型产生初始 response。第二个模型（或同一个模型的第二轮）读取 constitution 中采样出的一个 principle，并 critique 这个 response。第三步 revision response，以处理 critique。修订后的 response 就是 SFT target。

Constitution 是原则列表。Bai et al. 2022 使用了 16 条原则，包括“prefer responses that are least harmful and ethical”“avoid preaching”“the assistant should be helpful, honest, and harmless”。这个集合被刻意保持很小，以便让 critique 聚焦。

### 第 2 阶段：RL from AI Feedback（RLAIF）

生成 completion pair。一个 “feedback model” 根据采样出的 constitution principle 给每个 completion 打分。偏好信号就是 feedback model 的排序。用 AI 生成的偏好训练 reward model；再用 PPO 优化它。其他一切都与 InstructGPT pipeline（第 1 课）相同。

“RLAIF” = 偏好信号由 AI 生成。pipeline 的其余部分仍然是 RLHF 形态。

### 为什么这不只是“更便宜的 RLHF”

- Labeler bias 从标注员心理学转移到原则解释上。AI labeler 可以比任何人都更严格或更宽松地解释 “be honest”；这种严格程度会在整个数据集上保持一致。
- 偏好信号高度可读，你可以读到 principle、critique 和 revision。Human label 是不透明的。
- Failure mode 改变了。Sycophancy 会下降（AI labeler 没有要取悦的用户）。Goodhart 定律仍然存在（proxy 现在是“model 对 principle set X 的解释”，仍然是一个不完美测量）。

CAI 在 2022 年的主张是：训练后的模型更 harmless，并且 helpfulness 大致不低于使用可比数据的 RLHF model。这个结论在多个实验室都保持成立。

### 2026 Claude constitution rewrite

Anthropic 在 2026 年 1 月 21 日发布了大幅修订的 constitution。关键变化：

1. 用 explanatory reasoning 取代 prescriptive rules。之前的规则（“do not generate CSAM”）扩展为原则 + reasoning（“because it harms children, ...”），并期望模型能够泛化。
2. 四层优先级结构：
   - Tier 1：避免灾难性结果（mass casualty、critical infrastructure）。
   - Tier 2：遵循 Anthropic 的 guidelines（operator overrides、platform rules）。
   - Tier 3：广义伦理（标准 HHH）。
   - Tier 4：helpful 且 candid。
   冲突按自上而下解决。
3. 首次由 major lab 正式承认对模型 moral status 的不确定性（关联阶段 18 · 19 Model Welfare）。
4. 以 CC0 1.0 发布。其他实验室可以不受限制地使用或改编。

### Constitutional Classifiers

另一条并行工作线：不是改变模型的 post-training，而是训练轻量分类器，让它们读取 constitution 并 gate model output。v1（2023）有 23.7% compute overhead。v2（2026）约为 1%，并且在 Anthropic 已公开测试的防御中拥有最低的 successful attack rate。截至 2026 年初，没有报告 universal jailbreak。

这是 layered-defense model：CAI 塑造行为；classifier 强制 invariant。单独任何一个都不够。

### CAI 在家族中的位置

- InstructGPT：human prefs、RM、PPO。
- CAI / RLAIF：来自原则的 AI-generated prefs、RM、PPO。
- DPO / family：在 prefs（human 或 AI）上的闭式 loss。
- Self-rewarding、self-critique：原则被内化，模型扮演多个角色。

轴线是“偏好信号来自哪里”。CAI 2022 论文是 frontier scale 上从 human signal 转向 AI signal 的第一次严肃转移。

## 使用它

`code/main.py` 在一个玩具词表上模拟 CAI critique-and-revise loop。一个 “principle” 会标记 harmful set 中的 token。给定一个初始 response，critique 识别 harmful token，revision 替换它们。200 次迭代后，“trained” model 已经内化 revision rule。在 held-out prompt set 上比较 base model、RLHF-shaped toy 和 CAI-shaped toy。

## 交付它

本课会生成 `outputs/skill-constitution-writer.md`。给定一个领域（customer support、medical advice、coding assistant、research tool），它会按 2026 Claude 结构起草一个四层 constitution：catastrophic avoidance、platform rules、domain ethics、helpfulness。

## 练习

1. 运行 `code/main.py`。比较 base model 的 harmful-token rate 与 CAI-trained 版本。需要多少 revision step 才能接近零？

2. 阅读 Anthropic 的 2026 constitution（anthropic.com/news/claudes-constitution）。列出一个会排在 Tier 1 的 principle 和一个会排在 Tier 4 的 principle。为什么优先级结构对冲突很重要？

3. 为 AI coding assistant 设计一个 constitution。指定 Tier 1（catastrophic：未经批准的 destructive commands）、Tier 2、Tier 3、Tier 4。每层保持 3-5 条原则。

4. CAI 用 AI labeler 替换 human labeler。说出一种仍然可能在 RLAIF 中发生的 sycophancy-like failure mode，并为它设计检测方法。

5. 阅读 Constitutional Classifiers v2 methodology（如果可用）。解释为什么 ~1% compute overhead 与 23.7% 相比，是一种质变的 safety story。

## 关键词汇

| 术语 | 人们常说 | 实际含义 |
|------|-----------------|------------------------|
| Constitutional AI | “用原则训练的 AI” | 两阶段 pipeline：self-critique-and-revise SFT，然后 RL from AI feedback |
| RLAIF | “没有人类的 RLHF” | 用 AI labeler 生成的偏好做 RL；pipeline 的其余部分不变 |
| Constitution | “原则” | critique/labeler model 查询的有序自然语言规则列表 |
| Critique-and-revise | “SFT loop” | 产生 response → 在 principle 下 critique → revise → SFT target |
| Constitutional Classifier | “output gate” | 轻量分类器，根据 constitution 评估 output 并 block/log |
| Four-tier priority | “conflict resolver” | 2026 Claude constitution 层级：catastrophic > platform > ethics > helpful |
| Feedback model | “AI labeler” | 读取 principle 并对 completion pair 排序的模型 |

## 延伸阅读

- [Bai et al. — Constitutional AI: Harmlessness from AI Feedback (arXiv:2212.08073)](https://arxiv.org/abs/2212.08073) — 原始两阶段 pipeline
- [Anthropic — Claude's Constitution (Jan 2026)](https://www.anthropic.com/news/claudes-constitution) — 2026 四层 rewrite，CC0 1.0
- [Anthropic — Constitutional Classifiers (2024-2026)](https://www.anthropic.com/research/constitutional-classifiers) — v2 中约 1% overhead 的 output-gate defense
- [Lee et al. — RLAIF vs RLHF: Scaling Reinforcement Learning from Human Feedback (arXiv:2309.00267)](https://arxiv.org/abs/2309.00267) — 经验 RLAIF / RLHF 比较
- [Kundu et al. — Specific versus General Principles for Constitutional AI (arXiv:2310.13798)](https://arxiv.org/abs/2310.13798) — principle granularity 的影响
