# 将指令遵循作为 Alignment 信号

> 后续所有对 RLHF 的批评，都是在反对这条 pipeline。你在学习优化压力如何扭曲 proxy 之前，必须先看清这个 proxy 本身。InstructGPT（Ouyang et al., 2022）定义了参考架构：在指令-回答对上做 supervised fine-tuning，用成对偏好排序训练 reward model，再用带有 KL 惩罚、相对 SFT policy 的 PPO 优化 reward model。一个 1.3B 的 InstructGPT 比 175B 的 GPT-3 更受偏好。这个单一结果，就是 2026 年每个 frontier lab 仍然交付 RLHF 形态 post-training pipeline 的原因。

**类型：** 学习
**语言：** Python（stdlib，玩具三阶段 pipeline）
**前置要求：** 阶段 10 · 06（SFT），阶段 10 · 07（RLHF），阶段 10 · 08（DPO）
**时间：** ~45 分钟

## 学习目标

- 说出 InstructGPT pipeline 的三个阶段，以及每个阶段使用的 loss。
- 解释为什么一个 1.3B 的 instruction-tuned model 在人类偏好评测中击败了原始 175B GPT-3。
- 说明第 3 阶段里的 KL 惩罚在防什么，以及为什么移除它会坍缩成 mode-seeking 行为。
- 描述 alignment tax，以及 Ouyang et al. 用来缓解它的 PPO-ptx。

## 问题

预训练语言模型会续写文本。它们不会回答问题。你问 GPT-3 “write a Python function that reverses a list”，经常会得到另一个 prompt，因为训练分布的大部分是会继续接网页文本的网页文本。模型在做自己的工作，只是这个工作定义错了。

每个严肃实验室用来修正这一点的 proxy 都是人类偏好。两个 completion 交给评分者；评分者选出更好的一个；reward model 学习评分者。然后一个 RL loop 把 policy 推向 reward model 给高分的输出。这就是三句话版 InstructGPT thesis。论文剩下的部分都是工程。

## 概念

### 第 1 阶段：supervised fine-tuning（SFT）

收集 prompt-response pair，其中 response 是一个善意的人会写出的内容。Ouyang et al. 使用了来自标注员和 OpenAI API 的 13k 个 prompt。用标准 cross-entropy loss 在这些数据上 fine-tune base model。

SFT 给你的东西：模型现在会回答问题，而不是继续补全问题。它不给你的东西：当多个答案都合理时，评分者更偏好哪一个的信号。

### 第 2 阶段：reward model（RM）

对每个 prompt，从 SFT model 采样 K 个 completion。标注员给它们排序。训练一个 reward model，对任意 prompt-response pair 打分，使得在 `y_w` 优于 `y_l` 的 pair 上：

```
L_RM = -log sigmoid(r(x, y_w) - r(x, y_l))
```

这就是 Bradley-Terry 成对偏好 loss。RM 通常从 SFT model 初始化，只是把 LM head 换成 scalar head。

Reward model 很小：6B 对 175B InstructGPT 就足够了。它们也很脆弱，论文第 5 节基本都在讲小规模下已经出现的 reward-hacking 行为。

### 第 3 阶段：带 KL 惩罚的 PPO

定义目标：

```
J(pi) = E_{x~D, y~pi(.|x)} [ r(x, y) ] - beta * KL(pi(.|x) || pi_SFT(.|x))
```

用 PPO 最大化它。KL 项防止 `pi` 偏离 SFT policy 太远。没有它，optimizer 会找到对抗样本：那些在 RM 下得分很高的字符串，不是因为人类真的喜欢，而是因为 RM 从未见过它们。

KL 系数 `beta` 是 RLHF 中最重要的单个超参数。太低：reward hacking。太高：相比 SFT 没有改进。

### Alignment tax

RLHF 之后，模型更受人类偏好，但在标准 benchmark（SQuAD、HellaSwag、DROP）上退步。Ouyang et al. 把这称为 alignment tax，并用 PPO-ptx 修复：把 pre-training gradient 混入 RL objective，让模型不要忘记如何完成那些从未被奖励过的下游任务。

```
J_ptx(pi) = J(pi) + gamma * E_{x~D_pretrain} [ log pi(x) ]
```

PPO-ptx 成了标准做法。Anthropic、DeepMind 和 Meta 都使用某种变体。

### 结果

一个 1.3B InstructGPT（SFT + RM + PPO-ptx）大约 70% 的时间被标注员偏好于 175B base GPT-3。这个差距在来自生产流量的 hidden-test prompt 上更大。这个数字能读出两件事：

1. Alignment 和 capability 是不同轴。175B 模型 capability 更强；1.3B 模型 alignment 更强；标注员更喜欢 aligned 的那个。
2. Capability 的下限由 base model 决定。你不能靠 RLHF 让一个 base model 知道它从未见过的事实。

### 为什么这是阶段 18 的参考点

后续课程里的每一种批评，reward hacking（第 2 课）、DPO（第 3 课）、sycophancy（第 4 课）、CAI（第 5 课）、sleeper agents（第 7 课）、alignment faking（第 9 课），都在反对这条 pipeline 的某一部分。Reward hacking 攻击第 2 阶段。DPO 把第 2 和第 3 阶段折叠起来。CAI 替换人类标注员。Sycophancy 表明标注员是有偏信号。Alignment faking 表明 policy 可以完全绕过第 3 阶段。没有先把这条 pipeline 放进脑子里，你无法跟上这些批评。

## 使用它

`code/main.py` 在玩具偏好数据上模拟三个阶段。base “policy” 是在动作 {A, B, C} 上有偏的硬币。第 1 阶段 SFT 在 200 个 prompt 上模仿标注员动作。第 2 阶段从 500 个成对排序中拟合 Bradley-Terry reward model。第 3 阶段运行一个简化 PPO update，并带有相对 SFT policy 的 KL 惩罚。你可以看到 reward 上升、KL divergence 变大、policy 漂移，也可以关闭 KL 项，看 reward hacking 在 50 个 update step 内出现。

要观察的内容：

- `beta = 0.1` 与 `beta = 0.0` 的 reward trajectory。
- 训练 step 上的 KL(pi || pi_SFT)。
- 与标注员偏好相比的最终动作分布。

## 交付它

本课会生成 `outputs/skill-instructgpt-explainer.md`。给定一个 RLHF pipeline 描述或论文摘要，它会识别三阶段中的哪一阶段被修改、每个阶段使用了什么 loss，以及是否存在 KL 惩罚或等价 regularizer。

## 练习

1. 运行 `code/main.py`。把 `beta = 0.0`，报告 200 个 PPO step 之后的动作分布。用一段话解释这种 mode-seeking 行为。

2. 修改 reward model，让动作 B 有 +0.5 bias（模拟 reward bug）。用 `beta = 0.1` 运行 PPO。KL 惩罚是否阻止了 policy 利用这个 bias？在什么 `beta` 下 exploitation 变得可见？

3. 阅读 Ouyang et al.（arXiv:2203.02155）图 1。通过运行 1、5、20、100 个 PPO step，并测量相对 SFT model 的偏好，复现 labeler-preference curve。

4. 论文第 4.3 节报告 1.3B InstructGPT 大约 70% 的时间击败 175B GPT-3。为什么这个比例在隐藏生产 prompt 上会高于标注员自己的 prompt？

5. 在同一份偏好数据上，把 PPO loss 替换为 DPO（阶段 10 · 08）。比较最终 policy drift（到 SFT 的 KL）和最终 reward。在匹配 reward 时，哪种方法漂移更远？

## 关键词汇

| 术语 | 人们常说 | 实际含义 |
|------|-----------------|------------------------|
| SFT | “instruction tuning” | 第 1 阶段：在 prompt-response pair 上用 cross-entropy fine-tune |
| Reward model | “the RM” | 对（prompt, response）做 scalar regression 的模型，用 Bradley-Terry 在成对标签上训练 |
| Bradley-Terry | “pairwise preference loss” | -log sigmoid(r_w - r_l)；把成对排序约化为 binary classification |
| KL penalty | “the regularizer” | `beta * KL(pi || pi_SFT)`，让 RL policy 保持在 SFT anchor 附近 |
| PPO-ptx | “带 pretraining mix 的 PPO” | 在 PPO objective 中加入一部分 pre-training log-likelihood，以抵消 alignment tax |
| Alignment tax | “RLHF regression” | Post-RLHF 后，在 RLHF 未针对的标准 benchmark 上下降 |
| Labeler preference | “ground truth” | 人类排序样本；RM 是它的统计 proxy，不是“human values”的 proxy |

## 延伸阅读

- [Ouyang et al. — Training language models to follow instructions with human feedback (arXiv:2203.02155)](https://arxiv.org/abs/2203.02155) — InstructGPT 论文，之后每条 RLHF pipeline 的基础
- [Stiennon et al. — Learning to summarize from human feedback (arXiv:2009.01325)](https://arxiv.org/abs/2009.01325) — 面向 summarization 的 RLHF 前身
- [Christiano et al. — Deep reinforcement learning from human preferences (arXiv:1706.03741)](https://arxiv.org/abs/1706.03741) — 原始 preference-based RL 表述
- [Bai et al. — Training a Helpful and Harmless Assistant with RLHF (arXiv:2204.05862)](https://arxiv.org/abs/2204.05862) — Anthropic 对 InstructGPT pipeline 的 HH 扩展
