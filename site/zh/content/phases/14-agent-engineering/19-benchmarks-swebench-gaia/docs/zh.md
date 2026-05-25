# Benchmarks：SWE-bench、GAIA、AgentBench

> 三个 benchmarks 支撑了 2026 年的 agent evaluation。SWE-bench 测 code patching。GAIA 测 generalist tool use。AgentBench 测 multi-environment reasoning。你要知道它们的 composition、contamination story，以及它们不测什么。

**类型：** 学习
**语言：** Python (stdlib)
**前置要求：** 阶段 14 · 06（Tool Use）
**时间：** ~60 分钟

## 学习目标

- 说出 SWE-bench 的 test harness（FAIL_TO_PASS），并解释为什么它以 unit tests 为 gate。
- 解释 SWE-bench Verified（OpenAI，500 tasks）为什么存在，以及它移除了什么。
- 描述 GAIA 的设计：对人类简单，对 AI 困难；三个 difficulty levels。
- 说出 AgentBench 的八个 environments，以及它对 open-source LLMs 的主要 blocker。
- 总结 SWE-bench+ 的 contamination finding 及其影响。

## 问题

Leaderboards 告诉你哪个模型在某个 benchmark 上赢了。它们不会告诉你：

- Benchmark 是否 contaminated（solutions 在 training data 中、test leakage）。
- Benchmark 是否测量你关心的东西（code vs browsing vs generalist）。
- Evaluator 是否 robust（AST matching、state checks、human review）。

在引用数字前，先了解三个 anchoring benchmarks 和它们的 failure modes。

## 概念

### SWE-bench（Jimenez et al., ICLR 2024 oral）

- 来自 12 个流行 Python repos 的 2,294 个真实 GitHub issues。
- Agent 得到：pre-fix commit 处的 codebase + natural-language issue description。
- Agent 产出：patch。
- Evaluator：apply patch，运行 repo 的 test suite。Patch 必须让 FAIL_TO_PASS tests（之前失败、现在通过）翻转，同时不能破坏 PASS_TO_PASS tests。

SWE-agent（Yang et al., 2024）发布时达到 12.5%，重点在 agent-computer interfaces（file editor commands、模型能理解的 search syntax）。

### SWE-bench Verified

OpenAI，Aug 2024。500-task 的 human-curated subset。移除 ambiguous issues、unreliable tests，以及 fix 不清楚的 tasks。这是“你的 agent 是否能交付真实 patches”的 primary benchmark。

### Contamination

- 超过 94% 的 SWE-bench issues 早于大多数 model cutoffs。
- **SWE-bench+** 发现 32.67% 的 successful patches 在 issue text 中泄露了 solutions（模型在 description 中看到了 fix），31.08% 因 weak test coverage 而可疑。
- Verified 更干净，但不是 contamination-free。

实际影响：一个在 SWE-bench 上 50% 的模型，在 SWE-bench+ 上可能是 35%。如果你声称 SWE-bench performance，请总是同时报告两者。

### GAIA（Mialon et al., Nov 2023）

- 466 个问题；300 个保留给 huggingface.co/gaia-benchmark 的 private leaderboard。
- 设计哲学：“conceptually simple for humans (92%) but hard for AI (GPT-4 with plugins: 15%).”
- 测 reasoning、multi-modality、web、tool use。
- 三个 difficulty levels；Level 3 需要跨 modalities 的 long tool chains。

GAIA 用来测“generalist capability”。不要把它和 code-specific benchmarks 混淆。

### AgentBench（Liu et al., ICLR 2024）

- 8 个 environments，覆盖 code（Bash、DB、KG）、games（Alfworld、LTP）、web（WebShop、Mind2Web）和 open-ended generation。
- Multi-turn，每个 split 约 4k-13k turns。
- 主要发现：long-term reasoning、decision-making 和 instruction following 是 OSS LLMs 追上 commercial 的 blocker。

### 它们不测什么

- 真实世界 operational cost（tokens、wall-clock）。
- 对抗条件下的 safety behavior。
- 你的 domain 上的 performance（使用自己的 evals，第 30 课）。
- Tail failures（benchmarks 看平均值；production operators 关心最差 1%）。

### Benchmarking 会在哪里出错

- **Single-number fixation。** SWE-bench 50% 不如 P50/P75/P95 cost + step distribution 有信息量。
- **Contaminated claims。** 报告 SWE-bench 而不提 Verified 或 SWE-bench+ 是 misleading。
- **Benchmark-as-development-target。** 为 benchmark 优化会偏离 production usefulness。

## 构建它

`code/main.py` 实现了一个 toy SWE-bench-like harness：

- Synthetic bug-fix tasks（3 个 tasks）。
- 一个 scripted “agent” 提出 patches。
- 一个 test runner，检查 FAIL_TO_PASS（bug 现在修好）和 PASS_TO_PASS（没有破坏其他东西）。
- 一个基于 question decomposition depth 的 GAIA-style difficulty classifier。

运行它：

```
python3 code/main.py
```

输出会显示每个 task + 每种 difficulty 的 resolution rate，并让 evaluator rules 具体化。

## 使用它

- **SWE-bench Verified** 用于 code agents。总是报告 Verified scores。
- **GAIA** 用于 generalist agents。使用 private leaderboard split。
- **AgentBench** 用于 multi-environment comparison。
- **Custom evals**（第 30 课）用于你的产品实际形态。

## 发布它

`outputs/skill-benchmark-harness.md` 会为任意 codebase-task pair 构建 SWE-bench-style harness，带 FAIL_TO_PASS / PASS_TO_PASS gating。

## 练习

1. 把 toy harness 移植到真实 repo（选你自己的一个）。为已知 bugs 写 3 个 FAIL_TO_PASS tests。
2. 添加 step-count metric。在你的 3 个 tasks 上，每次 resolution 需要多少 agent steps？
3. 阅读 SWE-bench+ 论文。实现一个 solution-leakage check（把 issue text 和 diff 做 pattern-match）。
4. 从 public split 下载一个 GAIA question。Trace 一个 GPT-4-class agent 会怎么做。它需要哪些 tools？
5. 阅读 AgentBench 的 per-environment breakdown。哪个 environment 映射你的 product surface？那里的“SOTA”是什么样？

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|----------------|------------------------|
| SWE-bench | “Code agent benchmark” | 2,294 个 GitHub issues；patch 必须翻转 FAIL_TO_PASS tests |
| SWE-bench Verified | “Clean SWE-bench” | OpenAI human-curated 500 tasks |
| FAIL_TO_PASS | “Fix gate” | Patch 后必须通过的、之前失败的 tests |
| PASS_TO_PASS | “No-regression gate” | 之前通过且 patch 后仍必须通过的 tests |
| GAIA | “Generalist benchmark” | 466 个 human-easy / AI-hard multi-tool questions |
| AgentBench | “Multi-env benchmark” | 8 个 environments；long-horizon multi-turn |
| Contamination | “Training-set leak” | Benchmark tasks 出现在 model training 中 |
| SWE-bench+ | “Contamination audit” | 在 successful SWE-bench patches 中发现 32.67% solution leakage |

## 延伸阅读

- [Jimenez et al., SWE-bench (arXiv:2310.06770)](https://arxiv.org/abs/2310.06770)：原始 benchmark
- [OpenAI, SWE-bench Verified](https://openai.com/index/introducing-swe-bench-verified/)：curated subset
- [Mialon et al., GAIA (arXiv:2311.12983)](https://arxiv.org/abs/2311.12983)：generalist benchmark
- [Liu et al., AgentBench (arXiv:2308.03688)](https://arxiv.org/abs/2308.03688)：multi-environment suite
