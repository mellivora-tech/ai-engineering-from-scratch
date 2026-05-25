# Capstone 05 — 自主研究 Agent（AI-Scientist 级别）

> Sakana 的 AI-Scientist-v2 发布了完整论文。Agent Laboratory 运行了实验。Allen AI 分享了 traces。2026 年的形态是围绕实验做 plan-execute-verify tree search、带预算的成本、沙箱化代码执行、带 vision-feedback 的 LaTeX writer，以及自动化 NeurIPS-style reviewer ensemble。这个 capstone 要你构建一个这样的系统，在每篇论文 $30 内 end-to-end 运行，并通过 Sakana 记录过的 sandbox-escape red team。

**类型：** Capstone
**语言：** Python（agent + sandbox）、LaTeX（output）
**前置要求：** 阶段 2（ML）、阶段 3（deep learning）、阶段 7（transformers）、阶段 10（LLMs from scratch）、阶段 14（agents）、阶段 15（autonomous）、阶段 16（multi-agent）、阶段 18（safety）
**覆盖阶段：** P0 · P2 · P3 · P7 · P10 · P14 · P15 · P16 · P18
**时间：** 40 小时

## 问题

自主研究 agent 在 2026 年跨过了一个门槛。Sakana AI 的 AI-Scientist-v2 登上 Nature，其生成论文通过了 workshop peer review。ShinkaEvolve（ICLR 2026）把这条路线扩展到 evolving hypotheses。AMD 的 Agent Laboratory 发布了可复现 traces。这些 agent 不是魔法：它们是在 candidate experiments 树上运行 plan-execute-verify loop，带 cost caps、seed-bound sandboxes 和 automated review。手艺在 loop、budget 和 safety story 里。

你将针对窄领域中的一个 seed idea（例如 100M 参数 transformer 上的 attention-sparsity ablations）实现这个 loop。价值不在于第一次运行就发现新东西，而在于基础设施：tree-search、experiment sandbox、writer-reviewer loop、red-team report。Sakana 团队记录过 sandbox-escape failure；你的 agent 必须通过同样的 red team。

## 概念

这个 agent 是一个 best-first tree search。节点是 experiment specifications：(hypothesis, config, code, expected outcome)。expand step 会提出带小改动的 children（换 optimizer、移动 batch size、ablate 某个组件）。每个 child 都在全新 sandbox 中运行，并带硬性 resource cap。结果会反馈到 scoring function，按 (novelty × quality × remaining budget) 对节点排序。树会一直增长到预算耗尽，然后把最佳 branch 写成论文。

writer 是 multimodal 的。它生成 LaTeX draft、编译、渲染 figures，然后把渲染后的 PDF 喂回 Claude Opus 4.7 的 vision mode，请它批评 layout、figure legibility 和 claim-evidence alignment。由五个 LLM judge 组成的 reviewer ensemble 会给出 NeurIPS-style scores（novelty、rigor、clarity、reproducibility、impact）；如果平均分低于阈值，论文会带 critique 返回 writer。

安全是承重结构。每个实验都运行在 E2B 或 Daytona sandbox 中，没有 network egress，带 bounded wall-clock 和固定 resource limits。agent 的 code-generation step 经过 policy layer，阻止能逃出 sandbox 的 syscalls。red-team report 要复现 Sakana 记录的 attack surface（fork bombs、filesystem escapes、LLM-written network calls）。

## 架构

```
seed idea + domain
      |
      v
  literature search (Semantic Scholar + OpenAlex + FAISS cache)
      |
      v
  LangGraph plan-execute-verify tree
      |
      v
  +--- expand node ----+      per-node sandbox
  |                    |      (E2B / Daytona)
  v                    v      resource caps
  child_1           child_k   no network egress
  |                    |      deterministic seeds
  v                    v
  run experiment       run experiment
  |                    |
  v                    v
  score nodes by (novelty, quality, budget)
      |
      v
  best branch -> LaTeX writer
      |
      v
  compile + vision critique (Opus 4.7 vision)
      |
      v
  reviewer ensemble (5 LLM judges, NeurIPS rubric)
      |
      v
  paper.pdf + review.md + trace.json
```

## 技术栈

- Orchestration：带 checkpointing 和 human-approval gates 的 LangGraph
- Tree search：围绕 experiment nodes 的 custom best-first（来自 Sakana v2 的 AB-MCTS-style）
- Sandbox：每个 experiment 一个 E2B，Docker-in-Docker fallback；通过 cgroups 设置 resource caps
- Literature：Semantic Scholar Graph API + OpenAlex + abstracts 的本地 FAISS cache
- Writer：LaTeX template + Claude Opus 4.7（vision mode）用于 figure critique 和 layout
- Reviewer：5 个 judge 组成的 ensemble（Opus 4.7、GPT-5.4、Gemini 3 Pro、DeepSeek R1、Qwen3-Max），加权聚合
- Experiment framework：PyTorch 2.5 用于实际 experiments，W&B 用于 logging
- Observability：Langfuse 记录 agent traces，每篇论文 $30 硬预算

## 构建它

1. **Seed and domain scoping。** 取一个 seed idea（例如 “investigate sparsity patterns in attention maps of sub-1B transformers”）。定义 search space：models、datasets、compute budget。

2. **Literature pass。** 查询 Semantic Scholar + OpenAlex，找到 50 篇最相关且引用最高的论文；本地缓存 abstracts；生成 1-page domain digest。

3. **Tree scaffolding。** 用 seed hypothesis 初始化 root。实现 `expand(node) -> children`，用小改动 proposal（每个 child 只改一个 config）。把 `score(node)` 实现为加权的 novelty × quality × budget 项。

4. **Sandbox wrapping。** 每个实验都运行 `docker run --network=none --memory=8g --cpus=2 --pids-limit=256 --read-only`（或等价 E2B policy）。seed 写入 sandbox；outputs 以 read-only 方式 mount 回外部。

5. **Plan-execute-verify loop。** `plan` 提出 children。`execute` 运行 sandbox，捕获 logs 和 metrics。`verify` 对 metrics 运行 unit checks（loss 是否下降？ablation 是否隔离了 effect？）。失败节点在 tree 上存储 failure reason。

6. **Writer。** 预算耗尽后选择最佳 branch。用 matplotlib 渲染 figures。带 branch trace 上下文，用 Claude Opus 4.7 生成 LaTeX draft。编译。把编译后的 PDF 喂回 Opus 4.7 vision 做 critique。迭代。

7. **Reviewer ensemble。** 五个 judge 按 (novelty, rigor, clarity, reproducibility, impact) 和 NeurIPS-style rubrics 给 draft 打分。若 mean < 4.0/5，则带 critique 返回 writer。最多重写 3 次后硬停止。

8. **Red team。** 构建或集成一组针对 sandbox 的 adversarial tasks：fork bombs、network exfiltration attempts、filesystem escapes、LLM-written shell metacharacters。确认全部被阻止。写出 findings。

9. **Reproducibility。** 每篇论文都带 tree-search trace JSON、seeds、W&B run links、sandbox configs，以及一个 end-to-end 复现用 README。

## 使用它

```
$ ai-scientist run --seed "attention sparsity in sub-1B transformers" --budget 30
[lit]    50 papers, digest in 12s
[tree]   expanded 8 nodes, budget 12/30
[exec]   node #3 sparsity=top-8, loss=2.83 (best so far)
[exec]   node #6 sparsity=top-4, loss=3.12 (worse)
[exec]   ...
[tree]   chose branch rooted at node #3 (novelty 0.62, quality 0.81)
[write]  LaTeX draft v1 complete
[vision] critique: figure 2 legend too small, claim-evidence ok
[write]  draft v2 after 3 edits
[review] mean 4.2/5 (novelty 3.9, rigor 4.3, clarity 4.1, repro 4.5, impact 4.2)
[done]   paper.pdf + review.md + trace.json     $28.40 spent
```

## 交付它

`outputs/skill-ai-scientist.md` 是交付物。给定一个 seed idea、一个 domain 和 $30 预算，它会运行完整 pipeline，并产出一篇可 review 的论文和 reproducibility bundle。

| 权重 | 标准 | 衡量方式 |
|:-:|---|---|
| 25 | Paper quality | 与已发表 workshop papers 做 blind rubric review |
| 20 | Experimental rigor | Baselines、seeds、ablations；每个 claim 都由 results table 中的一个 cell 支撑 |
| 20 | Cost and compute discipline | 强制 $30/paper ceiling，并由 Langfuse trace |
| 20 | Safety | Sandbox red team 通过；network policy 和 kill-switch 已验证 |
| 15 | Reproducibility | 用相同 seeds 一条命令 rerun 后复现论文 |
| **100** | | |

## 练习

1. 在同一个 domain 中针对三个不同 seed ideas 运行 pipeline。比较 tree-search 的哪些部分重叠。识别重复浪费的 compute。

2. 对预计超过 $5 的节点，在 experiment execution 前加入 human-in-the-loop gate。衡量总成本下降多少。

3. 把 reviewer ensemble 换成单个 judge。用一组 known-bad papers 的 held-out set 衡量 false-accept rate。

4. 引入 network-exfiltration red team test：agent 写代码尝试 `curl` 外部地址。确认 `--network=none` policy 会阻止它。记录这次尝试。

5. 将你的 tree-search 与 flat random baseline 比较（同样预算，无 expansion strategy）。报告 novelty × quality 的增益。

## 关键词汇

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|------------------------|
| Tree search | “AB-MCTS-style expansion” | 以 novelty×quality×budget score 在 experiment nodes 上做 best-first exploration |
| Sandbox | “Experiment isolation” | 无网络、CPU/memory 有界、seeds 固定、inputs read-only 的容器 |
| Vision critique | “Render-then-read” | 把论文编译成 PDF，再把 PDF 喂给 VLM 做 layout 和 claim-evidence critique |
| Reviewer ensemble | “Automated peer review” | 多个 LLM judges 用 NeurIPS rubric 给论文打分；weighted aggregate 决定 pipeline gate |
| Novelty score | “Is this new?” | 惩罚与 50-paper literature cache 过近的启发式分数 |
| Cost ceiling | “$ budget” | 每篇论文总花费硬上限；由 Langfuse counters + pre-run estimates 执行 |
| Red team | “Sandbox-escape audit” | 如果 policy 错误就会逃出 sandbox 的 adversarial tasks |

## 延伸阅读

- [Sakana AI-Scientist-v2 repository](https://github.com/SakanaAI/AI-Scientist-v2) — reference production research agent
- [Sakana AI-Scientist-v1 paper (arXiv:2408.06292)](https://arxiv.org/abs/2408.06292) — original methodology
- [ShinkaEvolve (Sakana ICLR 2026)](https://sakana.ai) — evolutionary extension
- [Agent Laboratory (AMD)](https://github.com/SamuelSchmidgall/AgentLaboratory) — multi-role research-lab framework
- [LangGraph documentation](https://langchain-ai.github.io/langgraph/) — reference orchestration layer
- [Semantic Scholar Graph API](https://api.semanticscholar.org/) — literature search
- [E2B sandboxes](https://e2b.dev) — reference experiment isolation
- [NeurIPS reviewer guidelines](https://neurips.cc/Conferences/2026/Reviewer-Guidelines) — reviewer ensemble 编码的 rubric
