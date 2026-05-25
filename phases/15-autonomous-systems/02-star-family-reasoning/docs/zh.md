# STaR、V-STaR、Quiet-STaR — 自我教学推理

> 最小的 self-improvement loop 位于 rationale 内部。模型生成一段 chain of thought，保留那些得到正确答案的样本，并在这些样本上 fine-tune。这就是 STaR。V-STaR 增加了 verifier，让 inference-time selection 更好。Quiet-STaR 把 rationale 下沉到每个 token。三者都有效。三者都不是魔法——这个 loop 会保留任何碰巧到达正确答案的捷径。

**类型：** 学习
**语言：** Python（stdlib，bootstrap-loop simulator）
**前置要求：** 阶段 13 · 01-03（Reasoning and CoT），阶段 15 · 01（long-horizon framing）
**时间：** ~60 分钟

## 问题

教模型推理的直接方式是收集人类写好的 reasoning traces。这很贵，很慢，而且受限于人类愿意写多少高质量 chain-of-thought。

STaR（Self-Taught Reasoner，Zelikman et al., 2022）问的是：如果模型自己写 rationales，并根据已知答案给它们打分，会怎样？这个 loop 是：

1. 采样一个 reasoning trace 加答案。
2. 如果最终答案正确，保留这条 trace。
3. 在保留下来的 traces 上 fine-tune。
4. 重复。

它有效。GSM8K 和 CommonsenseQA 都在没有新增人工标注的情况下提升了。但这个 loop 有一个内置偏差：任何产生正确答案的 rationale 都会被保留，无论推理本身是否可靠。V-STaR（Hosseini et al., 2024）用 learned verifier 修补这一点；Quiet-STaR（Zelikman et al., 2024）把这个想法推广到 per-token internal rationales。

## 概念

### STaR：在有效样本上 bootstrap

从一个具备一些弱推理能力的 base model 开始。对每个训练问题，采样 rationale 加答案。如果答案匹配标签，就保留这个（problem, rationale, answer）三元组。在保留集合上 fine-tune 模型。重复。

有一个关键变化。如果模型永远无法答对某个问题，loop 就无法从中学习。STaR 增加了 **rationalization**：对模型失败的问题，把正确答案作为提示注入，并重新 prompt 模型生成一条通向它的 rationale。rationalized rationales 会加入训练集。

原论文结果（Zelikman et al., 2022）：一个 GPT-J base model 通过带 rationalization 的重复 STaR rounds，在 GSM8K 上从 5.8% 提升到 10.7%——绝对提升约 5 个百分点。在 CommonsenseQA 上，STaR 训练的 GPT-J 6B 达到 72.5%，与 fine-tuned GPT-3 175B（~73%）相当——后者是一个大约 30 倍大的模型，并在人工标注 rationales 上训练。

### V-STaR：用 DPO 训练 verifier

STaR 会丢弃错误 rationales。Hosseini et al.（2024）观察到，这些也同样是数据：每一对（rationale，“这是否正确”）都可以训练 verifier。他们在正确和错误解答上使用 Direct Preference Optimization 来构建 ranker。推理时，采样 N 条 rationales，并选择 verifier 排名最高的一条。

报告的提升：在 GSM8K 和 MATH 上，相比之前 self-improvement baselines 提升 +4 到 +17 个百分点，其中大部分收益来自用 verifier 做 inference-time selection，而不是做额外的 generator fine-tuning。

### Quiet-STaR：per-token internal rationales

Zelikman et al.（2024）提出：如果模型学会在每个 token 位置生成一段短的 internal rationale，而不是只在问题和答案之间生成，会怎样？Quiet-STaR 训练模型在每个预测 token 前发出一个隐藏的“thought”，再通过 learned weight 将 thought-aware prediction 与 baseline prediction 混合。

结果：Mistral 7B 在不做 task-specific fine-tuning 的情况下，在 GSM8K 上 zero-shot 从 5.9% 提升到 10.9%，在 CommonsenseQA 上从 36.3% 提升到 47.2%。模型学会了“何时思考”——难 token 会得到更长的 internal rationales；容易 token 几乎没有。

### 为什么三者都有共同的安全担忧

三种方法都把最终答案作为 gradient signal。一条通过错误推理得到正确答案的 rationale——利用捷径、猜测、或者使用无法泛化的模式——会得到正向强化。在 in-distribution 问题上，捷径有效。在 out-of-distribution 问题上，它会静默失效。

V-STaR 的 verifier 通过学习排序 rationales 来缓解问题，但 verifier 训练在同一组标签上。它可能学会偏好格式良好但错误的推理，而不是诚实的不确定性。更安全的设计是把 STaR-style 数据与（a）process-supervised reward models（奖励中间步骤，而不只是答案）以及（b）会打破简单捷径的 held-out OOD evaluation 结合。

### 对比

| Method | Training signal | Inference cost | Data waste | Known failure mode |
|---|---|---|---|---|
| STaR | keep (rationale, answer) if correct | 1x | discards all incorrect rationales | shortcut rationales |
| STaR + rationalization | above + correct-answer hinted retries | 1x | less | rationalized rationales may be implausible |
| V-STaR | STaR + DPO verifier from both classes | Nx (best-of-N) | minimal | verifier can reinforce confident wrongness |
| Quiet-STaR | per-token rationale + mixing weight | 1.5-3x | minimal | still answer-conditioned gradient |

### 它在 2026 技术栈中的位置

STaR 已经不新了。但这个模式在 2025-2026 年到处重现。对可验证数学问题做 RL（DeepSeek-R1、Kimi-k1.5、o1）就是 STaR 的 answer-conditioned gradient signal 的放大版。Process reward models（Lightman et al., 2023；OpenAI 的“Let's verify step by step”）是 process-supervised 的替代方案。AlphaEvolve（第 3 课）是代码版 STaR，只是用 program evaluator 代替标签。Darwin Godel Machine（第 4 课）是 agent scaffolding 本身的 STaR。

理解 STaR 会让这些东西都豁然开朗。它是最低可行的 self-improvement loop。

## 使用它

`code/main.py` 会在一个玩具算术任务上运行模拟的 STaR loop。你可以观察：

- accuracy 如何随着 bootstrap rounds 上升。
- 捷径如何混入：simulator 包含一个“lazy”rationale class，它有 40% 的时间得到正确答案，但泛化很差。观察 STaR 是否会保留它们。
- verifier（V-STaR 风格）如何在 inference 时帮忙，但无法完全剪掉训练中引入的捷径。

## 交付它

`outputs/skill-star-loop-reviewer.md` 帮你在训练前审计一个拟议的 self-taught-reasoning pipeline。

## 练习

1. 运行 simulator。将 shortcut frequency 设为 0，再设为 0.4。尽管两次都在训练分布上达到 >90%，最终 accuracy 会分化多少？

2. 给 simulator 添加 held-out OOD test。从不同分布抽取问题，并在 in-distribution 和 OOD sets 上评估 bootstrapped model。量化差距。

3. 阅读 Quiet-STaR 论文（arXiv:2403.09629）第 3 节。分别用三句话解释“end-of-thought”token 和 mixing-weight head。

4. 将 STaR 的 keep-if-correct filter 与 process-supervised alternative 对比，后者独立奖励每个 rationale step。指出 labelling cost 差异和可能的质量差异。

5. 设计一个 evaluation 来捕捉 deployed model 中的 shortcut rationales。它不需要完美——只需要打破 STaR loop 会强化的最简单捷径。

## 关键术语

| Term | What people say | What it actually means |
|---|---|---|
| STaR | “Self-Taught Reasoner” | 在得到正确答案的 model-generated rationales 上 fine-tune；重复 |
| Rationalization | “Hinted retry” | 对 base model 失败的问题注入正确答案，并重新 prompt 生成 rationale |
| V-STaR | “Verifier STaR” | 在正确和错误 rationales 上用 DPO 训练 verifier，并用于 inference-time selection |
| Quiet-STaR | “Per-token rationales” | 在每个 token 位置生成隐藏 thoughts；与 baseline prediction 混合 |
| Answer-conditioned gradient | “Outcome-based signal” | 训练 loop 奖励最终答案，而不是推理步骤 |
| Process reward model | “Step-level verifier” | 基于每步正确性训练的 reward model，而不是基于 outcome——与 STaR 相对 |
| Shortcut rationale | “正确答案，错误推理” | 通过无法泛化的模式得到标签的 rationale；STaR 会保留它 |

## 延伸阅读

- [Zelikman et al. (2022). STaR: Bootstrapping Reasoning With Reasoning](https://arxiv.org/abs/2203.14465) — 原始论文。
- [Hosseini et al. (2024). V-STaR: Training Verifiers for Self-Taught Reasoners](https://arxiv.org/abs/2402.06457) — 增加用于 inference-time selection 的 DPO verifier。
- [Zelikman et al. (2024). Quiet-STaR: Language Models Can Teach Themselves to Think Before Speaking](https://arxiv.org/abs/2403.09629) — per-token internal rationales。
- [Lightman et al. (2023). Let's Verify Step by Step](https://arxiv.org/abs/2305.20050) — process reward models，即替代的 gradient signal。
- [DeepSeek-R1 paper (arXiv:2501.12948)](https://arxiv.org/abs/2501.12948) — 在可验证任务上做 RL，把 STaR 扩展到 frontier training。
