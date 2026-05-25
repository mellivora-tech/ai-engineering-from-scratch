# 游戏中的 RL — AlphaZero、MuZero 与 LLM Reasoning 时代

> 1992 年：TD-Gammon 用纯 TD 在双陆棋上击败人类冠军。2016 年：AlphaGo 击败李世石。2017 年：AlphaZero 从零开始统治国际象棋、将棋和围棋。2024 年：DeepSeek-R1 证明同一配方，把 PPO 换成 GRPO，也能用于 reasoning。游戏是推动本阶段每次突破的 benchmark。

**类型：** 构建
**语言：** Python
**先修：** Phase 9 · 05（DQN），Phase 9 · 08（PPO），Phase 9 · 09（RLHF），Phase 9 · 10（MARL）
**时间：** 约 120 分钟

## 问题

游戏拥有 RL 想要的一切。干净的 reward（胜/负）。无限 episode（self-play reset）。完美 simulation（游戏*本身*就是模拟器）。离散或小型连续动作空间。迫使 adversarial robustness 的 multi-agent 结构。

而且，几乎每个重大 RL 突破都用游戏测试过。TD-Gammon（双陆棋，1992）。Atari-DQN（2013）。AlphaGo（2016）。AlphaZero（2017）。OpenAI Five（Dota 2，2019）。AlphaStar（StarCraft II，2019）。MuZero（learned model，2019）。AlphaTensor（矩阵乘法，2022）。AlphaDev（排序算法，2023）。DeepSeek-R1（数学 reasoning，2025）——它是最新证明：game-RL 技术可以用于文本。

这个 capstone 通过一个统一视角梳理三种里程碑架构：AlphaZero、MuZero 和 GRPO：**self-play + search + policy improvement**。每个都推广了前一个；尤其是 GRPO，它把 AlphaZero 的配方应用到 LLM reasoning 上，把 token 作为动作，把数学验证作为胜利信号。

## 核心概念

![AlphaZero ↔ MuZero ↔ GRPO: same loop, different environments](../assets/rl-games.svg)

**统一循环。**

```
while True:
    trajectory = self_play(current_policy, search)     # play game against self
    policy_target = search.improved_policy(trajectory) # search improves raw policy
    policy_net.update(policy_target, value_target)     # supervised on search output
```

**AlphaZero（2017）。** Silver 等人。给定一个规则已知的游戏（国际象棋、将棋、围棋）：

- Policy-value network：一个 tower `f_θ(s) → (p, v)`。`p` 是 legal moves 上的 prior。`v` 是 expected game outcome。
- Monte Carlo Tree Search（MCTS）：在每一步，扩展可能 continuation 的树。用 `(p, v)` 作为 prior + bootstrap。通过 UCB（PUCT）选择节点：`a* = argmax Q(s, a) + c · p(a|s) · √N(s) / (1 + N(s, a))`。
- Self-play：agent-vs-agent 地下棋。第 `t` 步时，MCTS visit distribution `π_t` 成为 policy training target。
- Loss：`L = (v - z)² - π · log p + c · ||θ||²`。`z` 是游戏结果（+1 / 0 / -1）。

零人类知识。零手写启发式。一个配方，在每种游戏几千万 self-play games 后掌握国际象棋、将棋和围棋。

**MuZero（2019）。** Schrittwieser 等人。移除“规则已知”的要求。

- 不使用固定环境，而是学习一个 *latent dynamics model* `(h, g, f)`：
  - `h(s)`：把 observation 编码成 latent state。
  - `g(s_latent, a)`：预测下一个 latent state + reward。
  - `f(s_latent)`：预测 policy prior + value。
- MCTS 在 *learned latent space* 中运行。同一个 search，同一个 training loop。
- 可用于围棋、国际象棋、将棋*以及* Atari：一个算法，不需要规则知识。

**Stochastic MuZero（2022）。** 加入 stochastic dynamics 和 chance nodes；扩展到双陆棋这类游戏。

**Muesli、Gumbel MuZero（2022-2024）。** 改进样本效率和 deterministic search。

**GRPO（2024-2025）。** DeepSeek-R1 配方。同样是 AlphaZero 形状的循环，但应用到 language-model reasoning：

- “游戏”：回答数学 / 编程 / reasoning 问题。“胜利” = verifier（测试用例通过、数值答案匹配）返回 1。
- Policy：LLM。Actions：tokens。State：prompt + response-so-far。
- 没有 critic（PPO-style V_φ）。相反，对每个 prompt，从 policy 采样 `G` 个 completions。计算每个 completion 的 reward。使用 **group-relative advantage** `A_i = (r_i - mean_r) / std_r` 作为 REINFORCE-style update 的信号。
- 到 reference policy 的 KL penalty，用来防止 drift（类似 RLHF）。
- 完整 loss：

  `L_GRPO(θ) = -E_{q, {o_i}} [ (1/G) Σ_i A_i · log π_θ(o_i | q) ] + β · KL(π_θ || π_ref)`

没有 reward model，没有 critic，没有 MCTS。Group-relative baseline 替代了三者。在 reasoning benchmark 上，以一小部分 compute 达到或超过 PPO-RLHF 质量。

**完整 R1 配方。** DeepSeek-R1（DeepSeek 2025）在一篇论文里有两个模型：

- **R1-Zero。** 从 DeepSeek-V3 base model 开始。没有 SFT。直接用两个 reward components 应用 GRPO：*accuracy reward*（rule-based：最终答案是否能解析成正确数字 / 代码是否通过 unit tests）和 *format reward*（completion 是否把 chain-of-thought 包进 `<think>…</think>` 标签）。经过数千步，平均 response length 从约 100 增长到约 10,000 tokens，数学 benchmark 分数爬升到接近 o1-preview 水平。模型从零学会 reasoning。缺点：它的 chains of thought 常常不可读、混合语言，并缺少风格润色。
- **R1。** 用四阶段 pipeline 修复 R1-Zero 的可读性问题：
  1. **Cold-start SFT。** 收集几千条长 CoT demonstrations，格式干净。用它们对 base model 做 supervised-finetune。这给了模型一个可读的起点。
  2. **Reasoning-oriented GRPO。** 使用 accuracy+format rewards，并加上 *language-consistency* reward 防止 code-switching，应用 GRPO。
  3. **Rejection sampling + SFT round 2。** 从 RL checkpoint 采样约 600K 条 reasoning trajectories，只保留最终答案正确且 CoT 可读的样本，并与约 200K 条非 reasoning SFT examples（writing、QA、self-cognition）合并。再次 fine-tune base。
  4. **Full-spectrum GRPO。** 再进行一轮 RL，覆盖 reasoning（rule-based rewards）和 general alignment（helpfulness/harmlessness preference-based rewards）。

结果是在开放权重下，在 AIME 和 MATH-500 上匹配 o1，并且足够小，可以 distill。同一篇论文还发布了六个 distilled dense models（从 Qwen-1.5B 到 Llama-70B），它们通过在 R1 的 reasoning traces 上做 SFT 得到，student 不做 RL。强 RL teacher 的 distillation 在 student 尺度上稳定胜过从零做 RL。

**为什么 reasoning 用 GRPO 而不是 PPO。** DeepSeekMath 论文（2024 年 2 月）给出三个原因：（1）没有 value network 要训练，内存减半；（2）group baseline 天然适配 reasoning task 产生的 sparse end-of-trajectory reward；（3）per-prompt normalization 让不同难度问题之间的 advantage 可比，而 PPO 的单个 critic 做不到。

**Search-free vs search-based。** 游戏领域已经分支：

- *Perfect-information games with long horizons*（围棋、国际象棋）：仍然是 search-based。AlphaZero / MuZero 占主导。
- *LLM reasoning*：生产中还没有 MCTS；使用完整 rollout 上的 GRPO，以及推理时的 best-of-N。Process reward models（PRMs）暗示 step-level search 会被重新加回来。

## 动手构建

`code/main.py` 中的代码实现了一个**微型 GRPO**：一个带多组样本的 bandit。算法和 LLM 上的一样；只是 policy 和环境更简单。它教的是 *loss* 和 *group-relative advantage*，这是 2025 年的创新点。

### 第 1 步：tiny verifier environment

```python
QUESTIONS = [
    {"prompt": "q1", "correct": 3},
    {"prompt": "q2", "correct": 1},
]

def verify(prompt_idx, answer_token):
    return 1.0 if answer_token == QUESTIONS[prompt_idx]["correct"] else 0.0
```

真实 GRPO 中，verifier 会运行 unit tests 或检查数学等价性。

### 第 2 步：policy：每个 prompt 上对 K 个 answer tokens 做 softmax

```python
def policy_probs(theta, p_idx):
    return softmax(theta[p_idx])
```

等价于 LLM conditioned on prompt 后的 final-layer output。

### 第 3 步：group sampling 与 group-relative advantage

```python
def grpo_step(theta, p_idx, G=8, beta=0.01, lr=0.1, rng=None):
    probs = policy_probs(theta, p_idx)
    samples = [sample(probs, rng) for _ in range(G)]
    rewards = [verify(p_idx, s) for s in samples]
    mean_r = sum(rewards) / G
    std_r = stddev(rewards) + 1e-8
    advs = [(r - mean_r) / std_r for r in rewards]

    for a, A in zip(samples, advs):
        grad = onehot(a) - probs
        for i in range(len(probs)):
            theta[p_idx][i] += lr * A * grad[i]
    # KL penalty: pull theta toward reference
    for i in range(len(probs)):
        theta[p_idx][i] -= beta * (theta[p_idx][i] - reference[p_idx][i])
```

Group-relative advantage 是 2024 年 DeepSeek 技巧。无需 critic。“Baseline” 是组均值，归一化使用组标准差。

### 第 4 步：与 REINFORCE baseline（value-free）对比

同一设置、同一 compute，plain REINFORCE。GRPO 收敛更快、更稳定。

### 第 5 步：观察 entropy 和 KL

和 RLHF 一样的 diagnostics：到 reference 的 mean KL、policy entropy、reward-over-time。一旦这些稳定，训练就完成。

## 常见坑

- **通过 verifier gaming 做 reward hacking。** GRPO 继承了 RLHF 的风险：如果 verifier 错误或可被利用，LLM 会找到漏洞。稳健的 verifiers（多个 test cases、formal proofs）很重要。
- **Group size 太小。** Group baseline 的方差约按 `1/√G` 下降。低于 `G = 4` 时，advantage signal 噪声很大；标准选择是 `G = 8` 到 `64`。
- **Length bias。** 不同长度的 LLM completions 有不同 log-probabilities。按 token count 归一化，或使用 sequence-level log-prob，或截断到 max length。
- **Pure self-play cycles。** AlphaZero-style training 在 general-sum games 上可能卡在 dominance loops 中。通过多样对手池缓解（league play，第 10 课）。
- **Search-policy mismatch。** AlphaZero 训练 policy 去模仿 search output。如果 policy net 太小，无法表示 search 的分布，训练会停滞。
- **Compute floor。** MuZero / AlphaZero 需要巨大 compute。一次 ablation 往往就是数百 GPU-hours。学习用的微型 demos 存在（例如 Connect Four 上的 AlphaZero）。
- **Verifier coverage。** 对 bug solution 也能通过的 unit tests 会强化这个 bug。设计能捕捉边界情况的 verifiers。

## 使用它

2026 年 game-RL 版图，按领域：

| 领域 | 主导方法 |
|------|----------|
| Two-player zero-sum board games（Go, chess, shogi） | AlphaZero / MuZero / KataGo |
| Imperfect info card games（poker） | CFR + deep learning（DeepStack, Libratus, Pluribus） |
| Atari / pixel games | Muesli / MuZero / IMPALA-PPO |
| Large multiplayer strategy（Dota, StarCraft） | PPO + self-play + league（OpenAI Five, AlphaStar） |
| LLM math/code reasoning | GRPO（DeepSeek-R1, Qwen-RL, open replications） |
| LLM alignment | DPO / RLHF-PPO（不是 GRPO；verifier 是 preference，不是可验证信号） |
| Robotics | PPO + DR（不是 game-RL，但使用相同 policy-gradient 工具） |
| Combinatorial problems | AlphaZero variants（AlphaTensor, AlphaDev） |

这套*配方*：self-play、search-augmented improvement、policy distillation，横跨文本、像素和物理控制。GRPO 是最年轻的实例；还会有更多。

## 交付

保存为 `outputs/skill-game-rl-designer.md`：

```markdown
---
name: game-rl-designer
description: Design a game-RL or reasoning-RL training pipeline (AlphaZero / MuZero / GRPO) for a given domain.
version: 1.0.0
phase: 9
lesson: 12
tags: [rl, alphazero, muzero, grpo, self-play]
---

Given a target (perfect-info game / imperfect-info / Atari / LLM reasoning / combinatorial), output:

1. Environment fit. Known rules? Markov? Stochastic? Multi-agent? Informs AlphaZero vs MuZero vs GRPO.
2. Search strategy. MCTS (PUCT with learned prior), Gumbel-sampled, best-of-N, or none.
3. Self-play plan. Symmetric self-play / league / offline data / verifier-generated.
4. Target signal. Game outcome / verifier reward / preference / learned model. Include robustness plan.
5. Diagnostics. Win rate vs baseline, ELO curve, verifier pass rate, KL to reference.

Refuse AlphaZero on imperfect-info games (route to CFR). Refuse GRPO without a trusted verifier. Refuse any game-RL pipeline without a fixed baseline opponent set (self-play ELO is uncalibrated otherwise).
```

## 练习

1. **简单。** 在 `code/main.py` 中实现 GRPO bandit。训练 2 个 prompt × 每个 4 个 answer tokens。使用 `G=8` 在 < 1,000 次更新内收敛。
2. **中等。** 接入 PPO（clipped）和 vanilla REINFORCE。在同一个 bandit 上比较它们与 GRPO 的样本效率和 reward 方差。
3. **困难。** 扩展到长度为 2 的“reasoning chain”：agent 发出两个 token，verifier 奖励这个 pair。测量 GRPO 如何处理两步序列中的 credit assignment。（提示：对*完整 sequence* 计算 group advantage，并传播到两个 token 位置。）

## 关键术语

| 术语 | 人们通常怎么说 | 它实际的含义 |
|------|----------------|--------------|
| MCTS | “带 learned net 的 tree search” | Monte Carlo Tree Search；使用 learned `(p, v)` priors 的 UCB1/PUCT selection。 |
| AlphaZero | “Self-play + MCTS” | Policy-value net 训练来匹配 MCTS visits 和 game outcome。 |
| MuZero | “Learned-model AlphaZero” | 同一循环，但通过 learned dynamics 在 latent space 中运行。 |
| GRPO | “Critic-free PPO” | Group Relative Policy Optimization；带 group-mean baseline + KL 的 REINFORCE。 |
| PUCT | “AlphaZero 的 UCB” | `Q + c · p · √N / (1 + N_a)`：平衡 value estimate 和 prior。 |
| Self-play | “Agent vs past self” | Zero-sum 的标准做法；对称训练信号。 |
| League play | “Population-based self-play” | 过去 + 当前 + exploiters 被采样为对手。 |
| Verifier reward | “Verifiable RL” | Reward 来自确定性 checker（tests pass、answer matches）。 |
| Process reward | “PRM” | 给每个 reasoning step 打分，而不只最终答案。 |

## 延伸阅读

- [Silver et al. (2017). Mastering the game of Go without human knowledge (AlphaGo Zero)](https://www.nature.com/articles/nature24270).
- [Silver et al. (2018). A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play (AlphaZero)](https://www.science.org/doi/10.1126/science.aar6404).
- [Schrittwieser et al. (2020). Mastering Atari, Go, chess and shogi by planning with a learned model (MuZero)](https://www.nature.com/articles/s41586-020-03051-4).
- [Vinyals et al. (2019). Grandmaster level in StarCraft II (AlphaStar)](https://www.nature.com/articles/s41586-019-1724-z).
- [DeepSeek-AI (2024). DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models (GRPO)](https://arxiv.org/abs/2402.03300) — 介绍 GRPO 和 group-relative baseline 的论文。
- [DeepSeek-AI (2025). DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948) — 完整四阶段 R1 配方，以及 R1-Zero ablation。
- [Brown et al. (2019). Superhuman AI for multiplayer poker (Pluribus)](https://www.science.org/doi/10.1126/science.aay2400) — 大规模 CFR + deep-learning。
- [Tesauro (1995). Temporal Difference Learning and TD-Gammon](https://dl.acm.org/doi/10.1145/203330.203343) — 开启这一切的论文。
- [Hugging Face TRL — GRPOTrainer](https://huggingface.co/docs/trl/main/en/grpo_trainer) — 用 custom reward functions 应用 GRPO 的生产参考。
- [Qwen Team (2024). Qwen2.5-Math — GRPO replication](https://github.com/QwenLM/Qwen2.5-Math) — 多尺度 R1 配方开源复现。
- [Sutton & Barto (2018). Ch. 17 — Frontiers of Reinforcement Learning](http://incompleteideas.net/book/RLbook2020.pdf) — self-play、search 和“designed reward”的教科书框架；R1 在 LLM 尺度上实例化了它。
