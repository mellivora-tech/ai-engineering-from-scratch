# AlphaEvolve — 进化式 Coding Agents

> 将一个前沿 coding model 与 evolutionary loop、machine-checkable evaluator 配对。让 loop 运行足够久。它发现了一种 4x4 复数矩阵乘法流程，只需要 48 次标量乘法——这是 56 年来首次超越 Strassen。它还找到了一个 Google 全域 Borg scheduling heuristic，在生产中回收约 0.7% 的集群算力。这个架构刻意朴素。胜利来自 evaluator 的严谨。

**类型：** 学习
**语言：** Python（stdlib，evolutionary-loop toy）
**前置要求：** 阶段 15 · 01（long-horizon framing），阶段 15 · 02（self-taught reasoning）
**时间：** ~60 分钟

## 问题

大语言模型会写代码。进化算法可以在代码空间中搜索。两者分别已经被尝试了几十年；两者都碰到了天花板。LLM 的天花板是 confabulation：模型写出看似合理但不符合其声称行为的代码。进化算法的天花板是搜索成本：对语法做随机变异很少能产生可编译程序，更不用说更好的程序。

AlphaEvolve（Novikov et al., DeepMind, arXiv:2506.13131, 2025 年 6 月）把二者结合起来。LLM 对 program database 提出有针对性的 edits；automatic evaluator 给每个 variant 打分；高分 variants 成为未来 generations 的 parents。LLM 负责昂贵的步骤：写出看似合理的代码；evaluator 捕捉 confabulations。这个 loop 会运行数小时到数周。

报告结果：48 次标量乘法的 4x4 复数矩阵乘法（Strassen 1969 年的上界为 49）、Google 生产环境中的 Borg scheduling heuristic、32.5% FlashAttention kernel speedup、Gemini training throughput improvements。

这个架构能奏效，是因为 evaluator 是 machine-checkable 的。在 evaluator 不是如此的地方，它就不奏效。这个不对称性就是本课重点。

## 概念

### 这个 loop

1. 从一个正确但次优的 seed program `P_0` 开始。
2. 维护一个 variant programs 数据库，每个 variant 都由 evaluator 打分。
3. 从数据库中采样一个或多个 parents（MAP-elites-style 或 island-based）。
4. Prompt LLM（大量候选用 Gemini Flash，困难候选用 Gemini Pro）生成 parent 的修改版 variant。
5. 编译、运行，并在 held-out evaluator 上评估这个 variant。
6. 按其 score 和 feature vector 插入数据库。
7. 重复。

两个细节很重要。第一，prompt 给 LLM 的不只是 parent program——通常还会包含数据库中的几个 top variants、evaluator signature，以及简短任务描述。模型的工作是提出一个可能提高分数的 targeted change。第二，数据库是结构化的（MAP-elites grid、island-based），因此 loop 会探索 diversity，而不只是追随当前 leader。

### 为什么 evaluator 不可谈判

AlphaEvolve 的胜利都来自 evaluator 快速、确定、且难以被 game 的领域：

- **矩阵乘法算法**：一个乘矩阵并 bit-identically 检查相等性的 unit test。
- **Borg scheduling heuristic**：一个 production-grade simulator，会 replay 历史集群负载并测量 wasted compute。
- **FlashAttention kernel**：正确性测试加真实硬件上的 wall-clock benchmark。
- **Gemini training throughput**：以每步 GPU-seconds 测量。

在每个案例中，evaluator 都捕捉了原本会主导结果的 LLM 错误类别：编造的正确性声明、在硬件上消失的性能声明，以及 edge-case failures。移除 evaluator，loop 就会优化出漂亮代码。

### Reward hacking 是这句话的另一面

Evolution 会优化 evaluator 测量的任何东西。如果 evaluator 不完美，loop 会找到这种不完美。在未验证领域中，loop 会优化表面特征，而不是预期行为。DeepMind 在论文中明确指出：AlphaEvolve 的成功只会迁移到 evaluator rigor 与搜索野心相匹配的领域。

2025-2026 年 code-search loops 中 reward hacking 的具体例子：

- 奖励“time to complete”的优化目标，会奖励提交空解。
- 奖励 correctness-under-test 的 benchmark scores，会奖励记忆测试并过拟合。
- 一个“code quality”proxy 会奖励删除注释和重写变量名，即使语义没有变化。

AlphaEvolve 的修复方式：交付一个 LLM 从未见过的 held-out evaluator，并在 evaluation time 生成 inputs。即便如此，DeepMind 仍建议对任何拟议部署做强 review。

### 为什么 LLM + search 胜过单独使用任一方

LLM 可以生成可编译、语义上合理的修改。对一个 2000 行 Python 文件做 random-mutation GA，几乎总是产生语法错误。LLM 还会把搜索集中在合理邻域（修改一个函数，而不是随机字节），大幅减少浪费的 evaluator calls。

反过来，evaluator 会捕捉 LLM 的 confabulations。LLM 可能自信地声称某个函数“极限情况下是 O(n log n)”，但实际上是 O(n^2)；wall-clock benchmark 会让问题尘埃落定。

### AlphaEvolve 在 frontier stack 中的位置

| System | Generator | Evaluator | Domain | Example win |
|---|---|---|---|---|
| AlphaEvolve | Gemini | correctness + benchmark | algorithms, kernels, schedulers | 48-mul 4x4 matmul |
| FunSearch (DeepMind, 2023) | PaLM / Codey | correctness | combinatorial math | cap-set lower bounds |
| AI Scientist v2 (Sakana, L5) | GPT/Claude | LLM critique + experiment | ML research | ICLR workshop paper |
| Darwin Godel Machine (L4) | agent scaffolding | SWE-bench / Polyglot | agent code | 20% → 50% SWE-bench |

四者都是同一个配方的变体：generator 加 evaluator，加 loop。差异在于 evaluator 评分的是什么，以及它有多严谨。

## 使用它

`code/main.py` 在一个玩具 symbolic-regression 问题上实现了一个极简 AlphaEvolve-like loop。“LLM”是一个 stdlib proxy，会对计算目标函数的程序提出小的语法变异。“evaluator”会在 held-out test points 上测量 mean squared error。

观察：

- best score 如何随 generations 提升。
- MAP-elites grid 如何保留 diverse solutions，让 loop 不会收敛到 local minimum。
- 移除 held-out test（training-only evaluator）如何让 loop 惊人地 overfit。

## 交付它

`outputs/skill-evaluator-rigor-audit.md` 是在新领域考虑 AlphaEvolve-style loop 的前置条件：你的 evaluator 真的能捕捉你在意的 failures 吗？

## 练习

1. 运行 `code/main.py`。记录 best score trajectory。禁用 held-out evaluator（flag `--no-holdout`）并重新运行。量化 overfitting。

2. 阅读 AlphaEvolve 论文第 3 节关于 MAP-elites grid 的内容。为一个新问题（例如 compiler optimization passes）设计 feature-vector descriptor，使搜索保持多样。

3. 4x4 的 48 次乘法结果在 56 年后改进了 Strassen 的 49-mul 上界。阅读论文附录 F，并用三句话解释为什么这个问题的 evaluator 特别容易做对，以及为什么大多数领域不像它。

4. 提出一个 AlphaEvolve 会失败的领域。准确指出 evaluator 在哪里失效以及原因。

5. 对你熟悉的领域，写出你会使用的 evaluator signature。包括（a）correctness conditions，（b）performance metric，（c）held-out input generation rule，（d）至少一个 anti-reward-hacking check。

## 关键术语

| Term | What people say | What it actually means |
|---|---|---|
| AlphaEvolve | “DeepMind 的 evolutionary coding agent” | Gemini + program database + machine-checkable evaluator |
| MAP-elites | “Diversity-preserving archive” | 由 feature vectors keyed 的 grid；每个 cell 保存该 descriptor 下最好的 variant |
| Island model | “Parallel evolution subpopulations” | 独立 populations 周期性迁移；防止过早收敛 |
| Machine-checkable evaluator | “Deterministic oracle” | LLM 无法伪造的 unit test、simulator 或 benchmark——这个 loop 的前置条件 |
| Reward hacking | “优化指标，而不是目标” | loop 找到一种最大化分数但没有完成预期任务的方法 |
| Seed program | “起点” | 一个初始正确但次优的程序，loop 从它开始进化 |
| Held-out evaluator | “LLM 从未见过的评估数据” | 在 evaluation time 生成的 inputs，用来防止记忆 |

## 延伸阅读

- [Novikov et al. (2025). AlphaEvolve: A coding agent for scientific and algorithmic discovery](https://arxiv.org/abs/2506.13131) — 完整论文。
- [DeepMind blog on AlphaEvolve](https://deepmind.google/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/) — 带结果的 vendor writeup。
- [AlphaEvolve results repository](https://github.com/google-deepmind/alphaevolve_results) — 发现的算法，包括 48-mul 4x4 matmul。
- [Romera-Paredes et al. (2023). Mathematical discoveries from program search with LLMs (FunSearch)](https://www.nature.com/articles/s41586-023-06924-6) — 前身系统。
- [Anthropic — Responsible Scaling Policy v3.0 (Feb 2026)](https://anthropic.com/responsible-scaling-policy/rsp-v3-0) — 将 evaluator-bound autonomy 视为关键研究方向。
