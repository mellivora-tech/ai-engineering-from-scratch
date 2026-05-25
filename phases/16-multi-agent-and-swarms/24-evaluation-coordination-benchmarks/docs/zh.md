# Evaluation 与 Coordination Benchmarks

> 五个 2025-2026 benchmarks 覆盖 multi-agent evaluation space。**MultiAgentBench / MARBLE**（ACL 2025, arXiv:2503.01935）用 milestone KPIs 评估 star/chain/tree/graph topologies；**graph 最适合 research**，cognitive planning 增加约 3% milestone achievement。**COMMA** 评估 multimodal asymmetric-information coordination；包括 GPT-4o 在内的 state-of-the-art models 难以超过 random baseline。**MedAgentBoard**（arXiv:2505.12371）覆盖四类 medical tasks，经常发现 multi-agent 不支配 single-LLM。**AgentArch**（arXiv:2509.10769）benchmark 结合 tool-use + memory + orchestration 的 enterprise agent architectures。**SWE-bench Pro**（[arXiv:2509.16941](https://arxiv.org/abs/2509.16941)）有 41 个 repos、1865 个 problems，覆盖 business apps、B2B services 和 developer tools；frontier models 在 Pro 上约 23%，而 Verified 上 70%+ — 这是 contamination 的 reality check。据报告，Claude Opus 4.7（2026 年 4 月）在 Pro 上 **64.3%**，使用 explicit agent-teams coordination（尚无 Anthropic primary source — 视为 preliminary）；Verdent（agent scaffold）在 Verified 上达到 **76.1% pass@1**（[Verdent technical report](https://www.verdent.ai/blog/swe-bench-verified-technical-report)）。**AAAI 2026 Bridge Program WMAC**（https://multiagents.org/2026/）是 2026 community focal point。本课基于 MARBLE metrics，运行 topology-vs-metric sweep，并固定 “just passing SWE-bench Verified is not evidence of generalization” 规则。

**类型：** 学习
**语言：** Python（stdlib）
**前置要求：** 阶段 16 · 15（Voting and Debate Topology），阶段 16 · 23（Failure Modes）
**时间：** ~75 分钟

## 问题

当论文声称 “our multi-agent system is better” 时，问题是：比什么好，在什么任务上好，如何测量？2023-2024 年 multi-agent evaluation 很混乱 — 每个人选择自己的 metrics、baselines 和 task sets。2025-2026 benchmarks 加入了结构。

没有 shared benchmarks，就无法有意义地比较两个 multi-agent systems。更糟的是，没有 hold-out benchmarks，frontier models 可能 contaminated。SWE-bench Verified 到 2025 年中已部分进入 training corpora；frontier scores 膨胀；Pro 被设计为 uncontaminated reality check。

本课列出 2026 年五个 canonical benchmarks，说明每个测什么，并教你带怀疑地阅读 benchmark claims。

## 概念

### MultiAgentBench（MARBLE）— ACL 2025

arXiv:2503.01935。评估四种 coordination topologies（star、chain、tree、graph）在 research、coding、planning tasks 上的表现。Milestone-based KPIs 跟踪 partial progress，而不只看 final success。

测量结果：

- **Graph** topology 最适合 research scenarios；支持 any-to-any critique。
- **Chain** 最适合 stepwise-refinement coding。
- **Star** 最适合 fast-factual consolidation。
- **Coordination tax** 在 graph 超过约 4 agents 后出现。
- **Cognitive planning** 在 topologies 上增加约 3% milestone achievement。

使用场景：你想 apples-to-apples 比较 coordination topologies。MARBLE repo（https://github.com/ulab-uiuc/MARBLE）提供 evaluator。

### COMMA — multimodal asymmetric information

覆盖 agents 有不同 observation modalities 且必须在不完全共享信息下协调的 tasks。报告结果不舒服：包括 GPT-4o 在内的 frontier models 在 COMMA 的 agent-agent collaboration 中很难超过 **random baseline**。signal 是 multi-agent modalities 训练和评估不足 — LLMs 对 single-modality cooperation 尚可；multi-modality coordination 会 collapse。

使用场景：你的系统有 multimodal 或 asymmetric-information coordination。COMMA 的 null result 是“先测量再宣称”的警告。

### MedAgentBoard — domain stress test

arXiv:2505.12371。四个 medical task categories：diagnosis、treatment planning、report generation、patient communication。比较 multi-agent vs single-LLM vs conventional rule-based systems。

发现：multi-agent 在多数 categories 上并不支配 single-LLM。multi-agent advantage 很窄 — 当 subtasks 清晰可分（diagnosis + treatment）时 task decomposition 有帮助；当 coordination overhead 超过 specialization gain（report generation）时有害。

使用场景：你的 domain 有 clear-cut single-LLM baselines。如果 MedAgentBoard 的 lesson 泛化，许多 proposed multi-agent systems 都 over-engineered。

### AgentArch — enterprise architectures

arXiv:2509.10769。带 tool use、memory 和 orchestration layer 的 enterprise settings。benchmark 隔离每一层贡献：加 tools 帮多少？加 memory？加 multi-agent orchestration？

使用场景：你在设计 enterprise agent stack，并需要证明每层的价值。AgentArch 帮助避免购买无法衡量价值的 features。

### SWE-bench Pro — reality check

arXiv:2509.16941。41 个 repositories、1865 个 problems，覆盖 business apps、B2B services 和 developer tools。设计为 **uncontaminated**，避开后续 training cutoffs。Frontier models 在 Pro 上约 23%，而 Verified 上 70%+。这个 gap 是 contamination signal。

2026 年 4 月 scores：
- Claude Opus 4.7 on Pro：**64.3%**（据报告使用 explicit agent-teams coordination；尚无 Anthropic primary source — 视为 preliminary）。
- Verdent（agent scaffold）on Verified：**76.1% pass@1**（[technical report](https://www.verdent.ai/blog/swe-bench-verified-technical-report)）。
- 没有 agent scaffolding 的 frontier raw scores on Pro：约 23-35%（[SWE-bench Pro paper](https://arxiv.org/abs/2509.16941)）。

结论：“we beat SWE-bench Verified” 已不再是 capability 证据。Pro 是当前 gating test。Agent-team scaffolding 在 Pro 上产生可测 gains（约 30-40 points delta），这是 2026 年 multi-agent coordination 的最强 empirical arguments 之一。

### AAAI 2026 WMAC

AAAI 2026 Bridge Program — Workshop on Multi-Agent Coordination（https://multiagents.org/2026/）。这是 2026 年 multi-agent AI research 的 community focal point。accepted papers 和 workshop proceedings 是评估新 methods 的 canonical venue；production decisions 优先参考 WMAC-accepted claims，而不是 arXiv preprints。

### 带怀疑阅读 benchmark claims — 2026 checklist

当有人声称 multi-agent result：

1. **哪个 benchmark，哪个 split？** SWE-bench Verified vs Pro 差别很大。在错误 split 上报告的数字没有价值。
2. **Contamination check。** benchmark 是否在 model training cutoff 后发布？如果不是，谨慎对待。
3. **Baseline comparison。** vs single-LLM baseline、vs random、vs prior multi-agent work。不是 “vs untuned version of the same system”。
4. **Statistical significance。** N trials、p-value、confidence interval。Frontier models 高 variance；single runs 会误导。
5. **Task diversity。** 一个任务还是很多任务？production 需要 generalization。
6. **Cost disclosure。** tokens per task、wall-clock。20x cost 换 90% solution 是 business decision，不是 capability claim。

### 当前 benchmarks 都测不好什么

- **Long-horizon coordination。** 数天 wall-clock interaction。当前 benchmarks 都短。
- **Adversarial resilience。** 当一个 agent malicious 或 compromised 时会怎样？
- **Drift under deployment。** Benchmarks 是 static；production distributions 会 shift。
- **Cost-normalized performance。** 多数 benchmarks 报 raw accuracy，而不是 accuracy-per-dollar。

为你真正关心的轴构建 internal benchmark 往往是正确做法。

## 构建它

`code/main.py` 是一个 non-interactive walk-through：

- 在 toy task 上模拟 3 个 multi-agent systems。
- 为每个计算 MARBLE-style milestone metrics。
- 通过从 “training” set withholding tasks 运行 contamination check。
- 明确与 random baseline 比较。
- 打印 benchmark-claims scorecard。

运行：

```bash
python3 code/main.py
```

预期输出：system scorecard，包含 raw accuracy、milestone achievement、cost-per-task、vs-random baseline delta，以及 contamination-check note。

## 使用它

`outputs/skill-benchmark-reader.md` 读取任何 multi-agent benchmark claim 并应用 scrutiny checklist。输出：grade 和 caveats。

## 发布它

Production evaluation discipline：

- **构建 internal benchmark**，反映真实 production distribution。public benchmarks 提供信息，但不能替代。
- **每次比较都包含 random baseline。** 如果 coordination task 上不能大幅超过 random，任务可能 ill-posed。
- **accuracy 旁边报告 cost。** Token cost 和 wall-clock。Ops teams 两者都需要。
- **每季度重建 benchmark。** Production distribution 会 shift；stale benchmarks 会误导。
- **避免 published-benchmark overfitting。** 如果 team 专门优化 SWE-bench Pro numbers，production 会退步。

## 练习

1. 运行 `code/main.py`。识别三个 simulated systems 中哪个有 best cost-per-milestone。它与最高 raw-accuracy system 相同吗？
2. 阅读 MultiAgentBench（arXiv:2503.01935）。对你自己的 task domain，决定 MARBLE 会推荐四种 topologies 中哪一种。根据论文结果论证。
3. 阅读 SWE-bench Pro paper。它具体如何做到 contamination-resistant？同样技术能应用到你关心的其他 benchmarks 吗？
4. 阅读 COMMA 关于 multimodal coordination 的 finding。设计一个可加入 internal benchmark 的简单 multimodal coordination task。什么才算 useful signal？
5. 用 benchmark-claims checklist 评估一篇 recent multi-agent paper 的 headline result。你会给这个 claim 什么 grade？

## 关键词汇

| 术语 | 人们常说 | 实际含义 |
|------|----------------|------------------------|
| MARBLE | “MultiAgentBench” | ACL 2025；star/chain/tree/graph topologies 和 milestone KPIs。 |
| COMMA | “Multimodal benchmark” | multimodal asymmetric-info coordination；frontier models 难以超过 random。 |
| MedAgentBoard | “Domain stress test” | 四个 medical categories；常发现 multi-agent 不支配 single-LLM。 |
| AgentArch | “Enterprise benchmark” | tools + memory + orchestration layered。 |
| SWE-bench Pro | “Contamination-resistant” | 1865 problems、41 repos；Pro ~23% vs Verified 70%+（contamination signal）。 |
| Milestone achievement | “Partial credit” | 奖励 progress，而不只奖励 final success 的 benchmarks。 |
| Contamination | “benchmark 泄漏到 training” | benchmark 发布后漂入 training corpora；scores 膨胀。 |
| WMAC | “AAAI 2026 Bridge Program” | Workshop on Multi-Agent Coordination；community focal point。 |

## 延伸阅读

- [MultiAgentBench / MARBLE](https://arxiv.org/abs/2503.01935) — 带 milestone KPIs 的 topology benchmark
- [MARBLE repository](https://github.com/ulab-uiuc/MARBLE) — reference implementation
- [MedAgentBoard](https://arxiv.org/abs/2505.12371) — domain stress test；multi-agent 经常不支配
- [AgentArch](https://arxiv.org/abs/2509.10769) — enterprise agent architectures
- [SWE-bench leaderboards](https://www.swebench.com/) — frontier models 的 Verified 和 Pro scores
- [AAAI 2026 WMAC](https://multiagents.org/2026/) — 2026 community focal point
