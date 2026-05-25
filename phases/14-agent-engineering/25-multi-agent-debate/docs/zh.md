# Multi-Agent Debate 和 Collaboration

> Du et al.（ICML 2024, "Society of Minds"）运行 N 个 model instances，让它们独立提出答案，然后经过 R 轮互相 critique 来收敛。它提升 factuality、rule-following、reasoning。Sparse topology 在 token cost 上优于 full mesh。

**类型：** 学习 + 构建
**语言：** Python (stdlib)
**前置要求：** 阶段 14 · 12（Workflow Patterns），阶段 14 · 05（Self-Refine and CRITIC）
**时间：** ~60 分钟

## 学习目标

- 解释 debate protocol：N 个 proposers，R 轮，收敛到共享答案。
- 描述为什么 debate 能提升 factuality、rule-following 和 reasoning。
- 解释 sparse topology：不是每个 debater 都需要看到所有其他 debaters。
- 基于 scripted LLM 实现一个 stdlib debate，包含 full-mesh 和 sparse variants；测量 token cost vs accuracy。

## 问题

Self-Refine（第 05 课）是一个 model critique 自己 — 有 groupthink 风险。CRITIC（第 05 课）把 critique ground 到 external tools — 但 tools 不总是可用。Debate 引入第三种模式：多个 instances、cross-critique、通过 disagreement 收敛。

## 概念

### Society of Minds (Du et al., ICML 2024)

- N 个 model instances 独立回答同一个问题。
- 经过 R 轮，每个 model 读取其他 proposals 并 critique。
- Models 根据 critiques 更新自己的答案。
- R 轮后，返回收敛答案。

原始实验因成本使用 N=3、R=2。对于困难问题（MMLU、GSM8K、Chess Move Validity、biography generation），更多 agents 和更多 rounds 会提升 accuracy。

Cross-model combinations 胜过 single-model debates：ChatGPT + Bard together > 任一单独模型。

### Sparse topology

"Improving Multi-Agent Debate with Sparse Communication Topology"（arXiv:2406.11776，2024-2025）表明 full-mesh debate 不总是最优。Sparse topologies（star、ring、hub-and-spoke）能以更低 token cost 匹配 accuracy。每个 debater 只看一部分 peers。

Implications：

- Full mesh N=5, R=3 = 5 × 3 = 15 proposals，每个读取 4 个 peers = 60 critique ops。
- Star N=5, R=3（一个 hub + 4 个 spokes）= 15 proposals，spokes 只读 hub = 12 critique ops。

### Debate 什么时候有帮助

- **Factuality。** N 个独立 proposals，cross-check 减少 hallucination。
- **Rule-following。** Chess move validity — 一个 model 漏掉规则，其他 model 抓住。
- **Open-ended reasoning。** 多种 framing 逐步缩小到正确答案。

### Debate 什么时候有害

- **Latency-sensitive UX。** N × R 串行 rounds 是你可能没有的延迟。
- **Cost-sensitive scale。** 每个问题 N × R tokens。
- **Simple factual lookups。** 一次 lookup 比五个 debates 便宜。

### 2026 practical instantiations

- **Anthropic orchestrator-workers**（第 12 课）— 带 synthesis step 的 debate 变体。
- **LangGraph supervisor**（第 13 课）— central router + specialist agents 可以把 debate 实现为一个 node。
- **OpenAI Agents SDK**（第 16 课）— agents 通过 handoff 来回进行 iterative critique。
- **Multi-agent evals** — debate + evaluator-optimizer 组合成 eval signal。

### 这个模式会在哪里出错

- **Convergence collapse。** 所有 agents 收敛到第一个错误答案。用 required disagreement rounds 缓解。
- **Hub failure。** 在 star topology 中，一个坏 hub 会污染所有人。轮换 hub 或使用多个 hubs。
- **Prompt homogenization。** 所有 agents 使用同一个 prompt；它们产出同样答案。使用 diverse prompts 和/或 models。

## 构建它

`code/main.py` 实现 stdlib debate：

- `Debater` class（带 per-debater opinion drift 的 scripted LLM）。
- `FullMeshDebate` 和 `SparseDebate` runners。
- 三个问题：一个 factual、一个 rule-based、一个 reasoning。
- Metrics：convergent answer、rounds to convergence、total critique ops。

运行它：

```
python3 code/main.py
```

输出：每个 protocol 的 accuracy 和 cost；sparse 在更低成本下在 2/3 questions 上匹配 full mesh。

## 使用它

- **Anthropic orchestrator-workers** 用于简单 2-3-worker debates。
- **LangGraph** 用于带 checkpointing 的 stateful multi-round debate。
- **Custom** 用于 research 或 specialized correctness guarantees。

## 发布它

`outputs/skill-debate.md` 会 scaffold 一个 multi-agent debate，支持 configurable topology、N、R 和 convergence rule。

## 练习

1. 实现一个 "forced disagreement" 规则：第 1 轮每个 debater 必须产出 distinct proposal。测量对 convergence speed 的影响。
2. 添加 confidence-weighted aggregation：debaters 返回 (answer, confidence)；aggregator 按 confidence 加权。有帮助吗？
3. 把一个 “agent” 换成意见不同的 scripted LLM。Heterogeneity 会提升 accuracy 吗？
4. 在你的 3 个问题上测量 full mesh vs sparse 的 token cost。绘制 cost vs accuracy。
5. 阅读 Society of Minds paper。把 toy 移植到 N=5、R=3。哪里会坏？哪里会更好？

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|----------------|------------------------|
| Debate | "Multi-agent critique" | N 个 proposers，R 轮 cross-critique，收敛 |
| Full mesh | "Everyone reads everyone" | 每个 debater 每轮都读取每个 peer |
| Sparse topology | "Limited peer view" | Debaters 只读取 peers 的一个子集 |
| Hub-and-spoke | "Star topology" | 一个 central debater，N-1 个 spokes 只读 hub |
| Convergence | "Agreement" | Debaters 收敛到一个共享答案 |
| Society of Minds | "Du et al. debate paper" | ICML 2024 multi-agent debate method |

## 延伸阅读

- [Du et al., Society of Minds (arXiv:2305.14325)](https://arxiv.org/abs/2305.14325) — canonical multi-agent debate
- [Sparse Communication Topology (arXiv:2406.11776)](https://arxiv.org/abs/2406.11776) — sparse topology results
- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) — orchestrator-workers as a debate variant
- [Madaan et al., Self-Refine (arXiv:2303.17651)](https://arxiv.org/abs/2303.17651) — single-model self-critique counterpart
