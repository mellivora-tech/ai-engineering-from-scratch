# Reward Modeling 与 RLHF

> 人类无法为“好的助手回复”手写 reward function，但可以比较两个回复并选出更好的那个。把 reward model 拟合到这些比较上，然后用 RL 让语言模型针对它优化。Christiano 2017。InstructGPT 2022。这套配方把 GPT-3 变成了 ChatGPT。到 2026 年，它大多被 DPO 替代，但心智模型仍然重要。

**类型：** 构建
**语言：** Python
**先修：** Phase 5 · 05（Sentiment），Phase 9 · 08（PPO）
**时间：** 约 45 分钟

## 问题

你用 next-token-prediction objective 训练了一个语言模型。它能写出合乎语法的英文。它也会撒谎、啰嗦，以及该拒绝时不拒绝。你无法靠更多 pretraining 修复它：网络文本是问题的一部分，不是解药。

你想要一个 *scalar reward*，能说明“对于 instruction X，response A 比 response B 更好”。手写这个 reward function 不可能。“Helpfulness” 不是 token 上的闭式表达式。但人类可以比较两个输出并标记偏好。这种数据可以相对低成本地大规模收集。

RLHF（Christiano et al. 2017；Ouyang et al. 2022）把偏好转换成 reward model，然后用 PPO 针对这个 reward 优化 LM。三步：SFT → RM → PPO。这是 2023–2025 年推出 ChatGPT、Claude、Gemini 以及其他 aligned-LLM 的配方。

到 2026 年，PPO 步骤大多被 DPO（Phase 10 · 08）替代，因为它更便宜，而且在 alignment tuning 上几乎同样好。但 *reward model* 这部分仍然支撑着每个 Best-of-N sampler、每个 RL-from-verifiable-rewards pipeline，以及每个使用 process reward model 的 reasoning model。理解 RLHF，你就理解了整个 alignment stack。

## 核心概念

![Three-stage RLHF: SFT, RM training on pairwise prefs, PPO with KL penalty](../assets/rlhf.svg)

**阶段 1：Supervised Fine-Tuning（SFT）。** 从 pretrained base model 开始。用人类写出的目标行为 demonstration 做 fine-tune（instruction-following responses、helpful replies 等）。结果是一个 `π_SFT` 模型，它*偏向好行为*，但 action space 仍然无界。

**阶段 2：Reward Model training。**

- 收集 prompt `x` 下的 response 对 `(y_+, y_-)`，由人类标注为“y_+ 优于 y_-”。
- 训练 reward model `R_φ(x, y)`，让它给 `y_+` 更高分。
- Loss：**Bradley-Terry pairwise logistic**：

  `L(φ) = -E[ log σ(R_φ(x, y_+) - R_φ(x, y_-)) ]`

  σ 是 sigmoid。reward 差值隐含 preference 的 log-odds。BT 从 1952 年（Bradley-Terry）起就是标准方法，也是现代 RLHF 的主流选择。

- `R_φ` 通常从 SFT model 初始化，并在顶部加 scalar head。同一个 transformer backbone；一个线性层输出 reward。

**阶段 3：带 KL penalty 针对 RM 做 PPO。**

- 从 `π_SFT` 初始化可训练 policy `π_θ`。保留一个冻结的 *reference* `π_ref = π_SFT`。
- Response `y` 结束处的 reward：

  `r_total(x, y) = R_φ(x, y) - β · KL(π_θ(·|x) || π_ref(·|x))`

  KL penalty 防止 `π_θ` 任意漂离 `π_SFT`。它是 *regularizer*，不是硬 trust region。`β` 通常为 `0.01`-`0.05`。
- 用这个 reward 运行 PPO（第 08 课）。Advantages 在 token-level trajectory 上计算，但 RM 只给完整 response 打分。

**为什么需要 KL？** 没有它，PPO 会很乐意找到 reward-hacking 策略：RM 只在 in-distribution completion 上训练过。Out-of-distribution response 可能比任何人类写的回复得分都高。KL 让 `π_θ` 留在 RM 训练过的 manifold 附近。它是 RLHF 中最重要的旋钮。

**2026 状态：**

- **DPO**（Rafailov 2023）：用闭式代数把阶段 2+3 折叠成一个针对偏好数据的 supervised loss。没有 RM，没有 PPO。只用一小部分 compute，就能在 alignment benchmark 上达到相近质量。Phase 10 · 08 覆盖。
- **GRPO**（DeepSeek 2024–2025）：PPO，但用 group-relative baseline 替代 critic，reward 来自 *verifier*（代码运行 / 数学答案匹配），而不是人类训练的 RM。它是 reasoning model 的主流。Phase 9 · 12 覆盖。
- **Process reward models（PRMs）：** 给部分解答（每个 reasoning step）打分，用于 RLHF 和 GRPO 的 reasoning 变体。
- **Constitutional AI / RLAIF：** 用 aligned LLM 生成偏好，而不是用人类。扩展 preference 预算。

## 动手构建

本课使用很小的合成“prompts”和“responses”，用字符串表示。RM 是 bag-of-tokens 表示上的线性打分器。没有真实 LLM：pipeline 的*形状*重要，规模不重要。见 `code/main.py`。

### 第 1 步：synthetic preference data

```python
PROMPTS = ["help me", "answer me", "explain this"]
GOOD_WORDS = {"clear", "specific", "kind", "thorough"}
BAD_WORDS = {"vague", "rude", "wrong", "short"}

def make_pair(rng):
    x = rng.choice(PROMPTS)
    y_good = rng.choice(list(GOOD_WORDS)) + " " + rng.choice(list(GOOD_WORDS))
    y_bad = rng.choice(list(BAD_WORDS)) + " " + rng.choice(list(BAD_WORDS))
    return (x, y_good, y_bad)
```

在真实 RLHF 中，这会被人类标注者替代。形状 `(prompt, preferred_response, rejected_response)` 完全相同。

### 第 2 步：Bradley-Terry reward model

线性分数：`R(x, y) = w · bag(y)`。训练目标是最小化 BT pairwise log-loss：

```python
def rm_train_step(w, x, y_pos, y_neg, lr):
    r_pos = dot(w, bag(y_pos))
    r_neg = dot(w, bag(y_neg))
    p = sigmoid(r_pos - r_neg)
    for tok, cnt in bag(y_pos).items():
        w[tok] += lr * (1 - p) * cnt
    for tok, cnt in bag(y_neg).items():
        w[tok] -= lr * (1 - p) * cnt
```

几百次更新后，`w` 会给 good-word token 正权重，给 bad token 负权重。

### 第 3 步：RM 之上的 PPO-like policy

我们的玩具 policy 从词表中生成单个 token。我们用 RM 给 token 打分，计算 `log π_θ(token | prompt)`，加上到 reference 的 KL penalty，并应用 clipped PPO surrogate。

```python
def rlhf_step(theta, ref, w, prompt, rng, eps=0.2, beta=0.1, lr=0.05):
    logits_theta = policy_logits(theta, prompt)
    probs = softmax(logits_theta)
    token = sample(probs, rng)
    logits_ref = policy_logits(ref, prompt)
    probs_ref = softmax(logits_ref)
    reward = dot(w, bag([token])) - beta * kl(probs, probs_ref)
    # ppo-style update on theta, treating reward as the return
    ...
```

### 第 4 步：监控 KL

每次更新跟踪 mean `KL(π_θ || π_ref)`。如果它爬过 `~5-10`，说明 policy 已经远离 `π_SFT`：`β` 太低，漂移正在加剧，或 reward hacking 正在开始。这是现实 RLHF 中最重要的诊断项。

### 第 5 步：使用 TRL 的生产配方

理解玩具 pipeline 后，下面是真实库用户写出的同一个循环。Hugging Face 的 [TRL](https://huggingface.co/docs/trl) 是参考实现：`RewardTrainer` 用于阶段 2，`PPOTrainer`（内置到 reference 的 KL）用于阶段 3。

```python
# Stage 2: reward model from pairwise preferences
from trl import RewardTrainer, RewardConfig
from transformers import AutoModelForSequenceClassification, AutoTokenizer

tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
rm = AutoModelForSequenceClassification.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct", num_labels=1
)

# dataset rows: {"prompt", "chosen", "rejected"} — Bradley-Terry format
trainer = RewardTrainer(
    model=rm,
    tokenizer=tok,
    train_dataset=preference_data,
    args=RewardConfig(output_dir="./rm", num_train_epochs=1, learning_rate=1e-5),
)
trainer.train()
```

```python
# Stage 3: PPO against the RM with KL penalty to the SFT reference
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead

policy = AutoModelForCausalLMWithValueHead.from_pretrained("./sft-checkpoint")
ref    = AutoModelForCausalLMWithValueHead.from_pretrained("./sft-checkpoint")  # frozen

ppo = PPOTrainer(
    config=PPOConfig(learning_rate=1.41e-5, batch_size=64, init_kl_coef=0.05,
                     target_kl=6.0, adap_kl_ctrl=True),
    model=policy, ref_model=ref, tokenizer=tok,
)

for batch in dataloader:
    responses = ppo.generate(batch["query_ids"], max_new_tokens=128)
    rewards   = rm(torch.cat([batch["query_ids"], responses], dim=-1)).logits[:, 0]
    stats     = ppo.step(batch["query_ids"], responses, rewards)
    # stats includes: mean_kl, clip_frac, value_loss — the three PPO diagnostics
```

库替你做了三件事。`adap_kl_ctrl=True` 实现 adaptive-β schedule：如果观察到的 KL 超过 `target_kl`，β 翻倍；如果低于一半，β 减半。Reference model 按惯例冻结：你绝不能意外地让它和 `policy` 共享参数。Value head 与 policy 位于同一个 backbone 上（`AutoModelForCausalLMWithValueHead` 附加一个 scalar MLP head），所以 TRL 会分别报告 `policy/kl` 和 `value/loss`。

## 常见坑

- **过度优化 / reward hacking。** RM 不完美；`π_θ` 会找到得分高但实际很差的 adversarial completions。症状：reward 无限上升，而 human eval score 持平或下降。修复：早停、提高 `β`、扩展 RM 训练数据。
- **Length hacking。** 在 helpful response 上训练的 RM 常常隐式奖励长度。Policy 学会填充回复。补救：length-normalized reward，或使用 length-aware RM 的 RLAIF。
- **RM 太小。** RM 至少需要和 policy 一样大。太小的 RM 无法忠实评估 policy 输出。
- **KL tuning。** β 太低 → drift 和 reward hacking。β 太高 → policy 几乎不变。标准技巧是使用针对固定 KL per step 的 *adaptive* β。
- **Preference-data 噪声。** 约 30% 的人类标签有噪声或歧义。用 agreement-filtered data 训练 RM，或在 BT 中使用 temperature 做校准。
- **Off-policy 问题。** 第一轮 epoch 后，PPO 数据略微 off-policy。像第 08 课一样监控 clip fraction。

## 使用它

2026 年的 RLHF 是分层的：

| 层 | 目标 | 方法 |
|----|------|------|
| Instruction following, helpfulness, harmlessness | Alignment | DPO（Phase 10 · 08）优先于 RLHF-PPO。 |
| Reasoning correctness（math, code） | Capability | 带 verifier reward 的 GRPO（Phase 9 · 12）。 |
| Long-horizon multi-step tasks | Agentic | PPO / GRPO，加上针对步骤的 process reward models。 |
| Safety / refusal behavior | Safety | 带独立 safety RM 的 RLHF-PPO，或 Constitutional AI。 |
| Best-of-N at inference | Fast alignment | 解码时使用 RM；不需要 policy training。 |
| Reward distillation | Inference compute | 在冻结 LM 顶部训练一个小 “reward head”。 |

RLHF 是 2022–2024 年的*核心*方法。到 2026 年，生产 alignment pipeline 首先考虑 DPO，只有 RM 密集或 safety-critical 步骤才使用 PPO。

## 交付

保存为 `outputs/skill-rlhf-architect.md`：

```markdown
---
name: rlhf-architect
description: Design an RLHF / DPO / GRPO alignment pipeline for a language model, including RM, KL, and data strategy.
version: 1.0.0
phase: 9
lesson: 9
tags: [rl, rlhf, alignment, llm]
---

Given a base LM, a target behavior (alignment / reasoning / refusal / agent), and a preference or verifier budget, output:

1. Stage. SFT? RM? DPO? GRPO? With justification.
2. Preference or verifier source. Humans, AI feedback, rule-based, unit-test-pass, or reward distillation.
3. KL strategy. Fixed β, adaptive β, or DPO (implicit KL).
4. Diagnostics. Mean KL, reward stability, over-optimization guard (holdout human eval).
5. Safety gate. Red-team set, refusal rate, safety RM separate from helpfulness RM.

Refuse to ship RLHF-PPO without a KL monitor. Refuse to use an RM smaller than the target policy. Refuse length-only rewards. Flag any pipeline that does not hold back a blind human-eval set as lacking over-optimization protection.
```

## 练习

1. **简单。** 在 `code/main.py` 中用 500 个合成 preference pair 训练 Bradley-Terry reward model。在保留的 100 个 pair 上测量 pairwise accuracy。应超过 90%。
2. **中等。** 用 `β ∈ {0.0, 0.1, 1.0}` 运行玩具 PPO-RLHF 循环。对每个设置，绘制 RM score vs KL-to-reference。哪些 run 发生了 reward-hack？
3. **困难。** 在同一份 preference data 上实现 DPO（闭式 preference-likelihood loss），并与 RLHF-PPO pipeline 比较使用的 compute 和最终达到的 RM score。

## 关键术语

| 术语 | 人们通常怎么说 | 它实际的含义 |
|------|----------------|--------------|
| RLHF | “Alignment RL” | 三阶段 SFT + RM + PPO pipeline（Christiano 2017，Ouyang 2022）。 |
| Reward Model（RM） | “打分网络” | 通过 Bradley-Terry 拟合 pairwise preferences 的 learned scalar function。 |
| Bradley-Terry | “Pairwise logistic loss” | `P(y_+ ≻ y_-) = σ(R(y_+) - R(y_-))`；标准 RM objective。 |
| KL penalty | “留在 reference 附近” | reward 中的 `β · KL(π_θ || π_ref)`；防 reward-hacking 的 regularizer。 |
| Reward hacking | “Goodhart's law” | Policy 利用 RM 缺陷；症状是 reward 上升、human eval 持平。 |
| RLAIF | “AI-labeled preferences” | 标签来自另一个 LM 而不是人类的 RLHF。 |
| PRM | “Process Reward Model” | 给部分 reasoning steps 打分；用于 reasoning pipeline。 |
| Constitutional AI | “Anthropic 的方法” | 由显式规则引导的 AI-generated preferences。 |

## 延伸阅读

- [Christiano et al. (2017). Deep Reinforcement Learning from Human Preferences](https://arxiv.org/abs/1706.03741) — 开创 RLHF 的论文。
- [Ouyang et al. (2022). InstructGPT — Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) — ChatGPT 背后的配方。
- [Stiennon et al. (2020). Learning to summarize with human feedback](https://arxiv.org/abs/2009.01325) — 较早的摘要 RLHF。
- [Rafailov et al. (2023). Direct Preference Optimization](https://arxiv.org/abs/2305.18290) — DPO；2026 年 post-RLHF 默认选择。
- [Bai et al. (2022). Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073) — RLAIF 与 self-critique loop。
- [Anthropic RLHF paper (Bai et al. 2022). Training a Helpful and Harmless Assistant](https://arxiv.org/abs/2204.05862) — HH 论文。
- [Hugging Face TRL library](https://huggingface.co/docs/trl) — 生产级 `RewardTrainer` 和 `PPOTrainer`。阅读 trainer 源码可了解 adaptive-KL 与 value-head 细节。
- [Hugging Face — Illustrating Reinforcement Learning from Human Feedback](https://huggingface.co/blog/rlhf) by Lambert, Castricato, von Werra, Havrilla — 三阶段 pipeline 的经典图文 walkthrough。
- [von Werra et al. (2020). TRL: Transformer Reinforcement Learning](https://github.com/huggingface/trl) — 该库；`examples/` 中有 Llama、Mistral 和 Qwen 的端到端 RLHF 脚本。
- [Sutton & Barto (2018). Ch. 17.4 — Designing Reward Signals](http://incompleteideas.net/book/RLbook2020.pdf) — reward-hypothesis 视角；思考 reward hacking 的必要前提。
