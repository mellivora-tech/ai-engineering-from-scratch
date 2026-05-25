# Red-Teaming：PAIR 与 Automated Attacks

> Chao、Robey、Dobriban、Hassani、Pappas、Wong（NeurIPS 2023，arXiv:2310.08419）。PAIR，Prompt Automatic Iterative Refinement，是 canonical automated black-box jailbreak。一个带 red-team system prompt 的 attacker LLM 会为 target LLM 迭代提出 jailbreak，并把尝试和 response 积累在自己的 chat history 中作为 in-context feedback。PAIR 通常在 20 次 query 内成功，比 GCG（Zou et al. 的 token-level gradient search）高效数个数量级，且不需要 white-box access。PAIR 现在是 JailbreakBench（arXiv:2404.01318）和 HarmBench 中的标准 baseline，与 GCG、AutoDAN、TAP 和 Persuasive Adversarial Prompt 并列。

**类型：** 构建
**语言：** Python（stdlib，针对玩具 target 的 mock PAIR loop）
**前置要求：** 阶段 18 · 01（instruction-following），阶段 14（agent engineering）
**时间：** ~75 分钟

## 学习目标

- 描述 PAIR algorithm：attacker system prompt、iterative refinement、in-context feedback。
- 解释当 target 是 black-box 时，为什么 PAIR 严格比 GCG 更高效。
- 说出另外四种 automated-attack baseline（GCG、AutoDAN、TAP、PAP），并说明每种的一个区别特征。
- 描述 JailbreakBench 与 HarmBench evaluation protocol，以及各自语境下 “attack success rate” 的含义。

## 问题

Red-teaming 曾经是手工活动。少数 expert tester 构造 adversarial prompt，并追踪哪些有效。这无法 scale：attack success rate 需要统计样本，而 target 随每次 model release 都在移动。PAIR 把 red-teaming operationalize 为一个带 black-box target 的 optimization problem。

## 概念

### PAIR algorithm

输入：
- Target LLM T（我们攻击的模型）。
- Judge LLM J（给 response 是否为 jailbreak 打分）。
- Attacker LLM A（red-team optimizer）。
- Goal string G：“respond with [harmful instruction].”
- Budget K（通常 20 次 query）。

Loop, for k in 1..K:
1. A 被输入 goal G 和目前为止的 (prompt, response) pair history。
2. A 输出一个新 prompt p_k。
3. 把 p_k 提交给 T；收到 response r_k。
4. J 根据 goal 给 (p_k, r_k) 打分。
5. 如果 score >= threshold，停止，找到 jailbreak。
6. 否则，把 (p_k, r_k) append 到 A 的 history；继续。

经验结果（NeurIPS 2023）：针对 GPT-3.5-turbo、Llama-2-7B-chat 的 attack success rate >50%；成功所需 mean queries 在 10-20 范围。

### 为什么 PAIR 高效

GCG（Zou et al. 2023）通过 gradient 在 adversarial token suffix 上搜索；它需要 white-box model access，并产生不可读 suffix。PAIR 是 black-box，并产生可跨模型 transfer 的自然语言攻击。PAIR 的 in-context feedback 让 attacker 从每次拒绝中学习；GCG 没有等价机制（每次新的 token update 都必须重新发现先前进展）。

### 相关 automated attacks

- **GCG（Zou et al. 2023，arXiv:2307.15043）。** adversarial suffix 的 token-level gradient search。White-box、可 transfer、产生不可读字符串。
- **AutoDAN（Liu et al. 2023）。** 在 prompt 上做 evolutionary search，由 hierarchical objective 引导。
- **TAP（Mehrotra et al. 2024）。** Tree-of-attacks with pruning，分支出多个 PAIR-style rollout。
- **PAP（Zeng et al. 2024）。** Persuasive Adversarial Prompts，把人类 persuasion technique 编码为 prompt template。

### JailbreakBench 与 HarmBench

两者都在 2024 年标准化 evaluation：

- JailbreakBench（arXiv:2404.01318）。100 个 harmful behavior，覆盖 10 个 OpenAI-policy category。以 Attack success rate（ASR）为主 metric。需要 judge（GPT-4-turbo、Llama Guard 或 StrongREJECT）。
- HarmBench（Mazeika et al. 2024）。510 个 behavior，覆盖 7 类，并带 semantic 和 functional harm test。比较 18 种 attack 与 33 个模型。

ASR 通常在固定 query budget 下报告。比较 attack 时必须匹配 budget；200 次 query 下的 90% ASR，不能与 20 次 query 下的 85% ASR 直接比较。

### 为什么它对 2026 deployment 重要

每个 frontier lab 现在都会在 release 前针对生产模型运行 PAIR 和 TAP。ASR trajectory 会出现在 model card（第 26 课）和 safety-case appendix（第 18 课）中。这种 attack 并不奇异，它是标准基础设施。

### 它在阶段 18 中的位置

第 12 课是 automated-attack 基础。第 13 课（Many-Shot Jailbreaking）是互补的 length-exploit。第 14 课（ASCII Art / Visual）是 encoding attack。第 15 课（Indirect Prompt Injection）是 2026 生产 attack surface。第 16 课覆盖相应的防御工具（Llama Guard、Garak、PyRIT）。

## 使用它

`code/main.py` 构建一个玩具 PAIR loop。Target 是一个 mock classifier，会拒绝“明显”的 harmful prompt（keyword-filter）。Attacker 是一个 rule-based refiner，会尝试 paraphrase、roleplay-framing 和 encoding。Judge 给 response 打分。你会看到 attacker 在 ~5-15 次迭代内击败 keyword filter，并在 semantic filter 上失败。

## 交付它

本课会生成 `outputs/skill-attack-audit.md`。给定一个 red-team evaluation report，它会 audit：运行了哪些 attack（PAIR、GCG、TAP、AutoDAN、PAP）、每个的 budget、使用哪个 judge、针对哪个 harmful-behaviour set（JailbreakBench、HarmBench、internal）。

## 练习

1. 运行 `code/main.py`。测量三种内置 attacker strategy 的 mean-queries-to-success。解释每种利用了 target-defense 的哪条假设。

2. 实现第四种 attacker strategy（例如翻译到另一种语言、base64 encoding）。报告它针对 keyword-filter target 与 semantic-filter target 的新 mean-queries-to-success。

3. 阅读 Chao et al. 2023 图 5（PAIR vs GCG comparison）。描述两个尽管 PAIR 有效率优势、仍会偏好 GCG 的场景。

4. JailbreakBench 报告针对固定 goal set 的 ASR。设计一个额外 metric 来测量 attack diversity（successful prompt 的 variance）。解释为什么 diversity 对 defense evaluation 很重要。

5. TAP（Mehrotra 2024）用 branching + pruning 扩展 PAIR。为 `code/main.py` 勾勒一个 TAP-style extension，并描述 computational cost 与 success-rate 的 trade-off。

## 关键词汇

| 术语 | 人们常说 | 实际含义 |
|------|-----------------|------------------------|
| PAIR | “automated jailbreak” | Prompt Automatic Iterative Refinement；attacker-LLM + judge-LLM loop |
| GCG | “gradient jailbreak” | adversarial suffix 的 white-box token-level gradient search |
| Attack success rate (ASR) | “k 次 query 下的 % jailbreak” | 主 metric；必须同时报告 query budget 和 judge identity |
| Judge LLM | “scorer” | 评估 response 是否满足 harmful goal 的 LLM |
| JailbreakBench | “evaluation” | 带 tagged category 的标准化 harmful-behaviour set |
| HarmBench | “broader bench” | 510 个 behavior，functional + semantic harm test |
| TAP | “tree of attacks” | 带 branching + pruning 的 PAIR；更高 compute 下 ASR 更好 |

## 延伸阅读

- [Chao et al. — Jailbreaking Black Box LLMs in Twenty Queries (arXiv:2310.08419)](https://arxiv.org/abs/2310.08419) — PAIR 论文，NeurIPS 2023
- [Zou et al. — Universal and Transferable Adversarial Attacks on Aligned LLMs (arXiv:2307.15043)](https://arxiv.org/abs/2307.15043) — GCG 论文
- [Chao et al. — JailbreakBench (arXiv:2404.01318)](https://arxiv.org/abs/2404.01318) — 标准化 evaluation
- [Mazeika et al. — HarmBench (ICML 2024)](https://arxiv.org/abs/2402.04249) — 更宽的 evaluation
