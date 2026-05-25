# Agents 的 Consensus 与 Byzantine Fault Tolerance

> Classical distributed-systems BFT 遇上 stochastic LLMs。2025-2026 年出现三条 research directions：**CP-WBFT**（arXiv:2511.10400）用 confidence probe 给每票加权；**DecentLLMs**（arXiv:2507.14928）采用 leaderless parallel worker proposals 和 geometric-median aggregation；**WBFT**（arXiv:2505.05103）把 weighted voting 与 Hierarchical Structure Clustering 结合，划分 Core 和 Edge nodes。“Can AI Agents Agree?”（arXiv:2603.01213）给出的诚实 empirical result 是：今天即便 scalar agreement 也很脆弱 — 一个 deceptive agent 就能破坏 Mixture-of-Agents。BFT 必要但不充分。本课构建一个 minimal BFT protocol，注入三种 agent-specific attacks（byzantine lie、sycophantic conformity、correlated-error monoculture），并测量每种 consensus variant 如何应对。

**类型：** 学习 + 构建
**语言：** Python（stdlib）
**前置要求：** 阶段 16 · 07（Society of Mind and Debate），阶段 16 · 13（Shared Memory）
**时间：** ~75 分钟

## 问题

你有 N 个 LLM agents，每个产生一个 answer。它们不一致。Majority vote 选错了，因为两个 agents 是相关的（同 base model、同 training data、同 failure modes）。第三个 agent 则以一种新方式碰巧错了 — 所以 majority 是 false majority。

现在加入一个 deceptive agent：它故意说谎。或者一个 sycophantic agent：它同意上一个发言的人。在 classical BFT 中，假设 Byzantine nodes 是 `f < n/3` 的比例，并可任意行为。2026 年现实是：LLM nodes 即便 honest 也是 stochastic，跨 models 相关，并且会受彼此 outputs 影响。你不能把它们当 independent Bernoulli voters。

Classical BFT（PBFT, 1999）并不是错，而是不完整。它处理 arbitrary bit-flipping。它不处理“三个 honest agents 因共享 training data 而共享 hallucination”。本课从 PBFT 基础出发，并叠加三种 2025-2026 adaptations。

## 概念

### Classical BFT 给了你什么

Practical Byzantine Fault Tolerance（Castro & Liskov, OSDI 1999）容忍 `f < n/3` Byzantine nodes。protocol 有三阶段（pre-prepare、prepare、commit）和两个 primitives（signed messages、quorum certificates）。在 `n >= 3f + 1` 的 honest-or-malicious nodes 中对单个 value 达成 agreement。

这些 guarantees 很强，但假设：

1. **Independent faults。** Byzantines 不协同。
2. **Honest nodes 真正 honest。** Honest outputs 的 correctness 不成问题；protocol 只 align disagreement。
3. **问题有 ground-truth answer。** 错 fact 上的 consensus 仍然是 consensus。

LLM agents 违反三点。两个运行同一个 base model 的 agents 共享 faults。一个 “honest” LLM 仍会 hallucinate。在 ambiguous questions 上，“truth” 是 agents 决定的东西 — 没有 external oracle。

### 三种 LLM-specific attacks

**Byzantine lie。** 一个 agent 输出故意错误的答案。如果 `f < n/3`，classical BFT 能处理。

**Sycophantic conformity。** 一个 agent 在 voting 前读取其他 answers，并对齐到上一个发言的人。不恶意，但与最大声 voice 相关。Classical BFT 无法阻止，因为这个 agent 通过所有 signature check。

**Correlated-error monoculture。** 三个 agents 共享一个 base model。它们 hallucinate 同一个错误答案。majority 错了。Classical BFT 无济于事，因为三者都“honestly” agree。

### 2025-2026 responses

**CP-WBFT**（arXiv:2511.10400）— Confidence-Probed Weighted BFT。每个 voter 给自己的 answer 附加 confidence probe（self-reported probability，或 separate calibration model 的预测）。vote weights 随 confidence 缩放。在 complete graphs 上报告 +85.71% BFT improvement。缓解：sycophantic conformity（conforming agents 对自己自愿 position 往往 confidence 较低）。

**DecentLLMs**（arXiv:2507.14928）— Leaderless。Worker agents 并行 propose，evaluator agents score proposals，final answer 是 scored positions 的 geometric median。在 `f < n/2` 时 robust。缓解：Byzantine lie 和 correlated errors（geometric median 对 outliers robust，并拉向 dense cluster，而不是 model-biased average）。

**WBFT**（arXiv:2505.05103）— Weighted BFT with Hierarchical Structure Clustering。vote weights 由 response quality 加上从历史学到的 trust score 分配。把 agents cluster 成 Core 和 Edge；Core agents 必须先达成 consensus，Edge agents 跟随。缓解：scalability（Core consensus 小且快），并部分缓解 monoculture（Core 可按 diversity 选择）。

### Empirical：“Can AI Agents Agree?”（arXiv:2603.01213）

该论文测量多个 frontier models 之间的 scalar agreement（LLM agents 对单个 numeric value 达成一致）。发现并不舒服：

- 即便没有 adversaries，LLM agents 在许多 benchmarks 的 scalar questions 上 disagreement rates 超过 30%。
- 一个采用 deceptive persona 的 agent 就能把 Mixture-of-Agents consensus 从 honest baseline 拉偏 40+ percentage points。
- disagreement rates 与 model diversity 相关 — heterogeneous ensembles 分歧更多（好处：uncorrelated errors），但 drift 更慢（坏处：longer time-to-agreement）。

结论：BFT 给你 align outputs 的 machinery，但不告诉你 aligned output 是否正确。结合 verification（阶段 16 · 08 role specialization）、diversity（阶段 16 · 15 debate variants）和 evaluator agents（阶段 16 · 24 benchmarks）。

### 核心 protocol，精简版

一个 minimal BFT round for LLM agents：

```
1. task arrives; each agent i produces answer a_i
2. each agent attaches confidence probe c_i in [0, 1]
3. aggregator collects (a_i, c_i) from all n agents
4. aggregator groups by semantic cluster (equivalent answers)
5. aggregator computes weight for each cluster C:
     w(C) = sum_{i in C} c_i
6. winner = cluster with max weight, if max > threshold * sum(c_i)
   else: retry or escalate
7. minority clusters logged with provenance for post-hoc audit
```

semantic clustering step 是 LLM-specific twist。两个答案 “the study reports 4.2%” 和 “4.2% improvement” 是同一 cluster。naive string-equality check 会错过。在 production 中，使用便宜 embedding model 或显式 canonicalization。

### Threshold tuning

`threshold` 参数决定什么时候 accept，什么时候 retry。太低：接受 weak majorities。太高：永远不接受。经验范围：`n=5-7` agents 时 0.5-0.67；小 n 时更高。低于 threshold 时，escalate to human 或不同 agent ensemble。

### Consensus 不帮忙的地方

- **Ambiguous questions。** 如果问题没有 ground truth，consensus 是 opinion。要这样称呼它。
- **Compound questions。** “Write code and explain it” — 两个 answers。分别投票。
- **Adversarial multi-round。** 如果 agents 可观察 prior rounds 并 mimic（Du 2023 debate），它们会开始无论真相如何都互相同意。限制 rounds（通常 2-3）。

## 构建它

`code/main.py` 实现：

- `AgentVoter` — 带 (answer, confidence) 的 scripted policy。
- `MajorityVote` — classical plurality。
- `CPWBFT` — semantic clustering 后的 confidence-weighted voting。
- `DecentLLMs` — scored proposals 上的 geometric-median aggregation。
- `Scenario` — 在三种 attack patterns 下运行每个 aggregator。

实现的 attack patterns：

1. `byzantine`：一个 agent 高 confidence 说谎。
2. `sycophancy`：一个 agent copy 它看到的第一个 answer，并匹配 confidence。
3. `monoculture`：三个 agents 共享一个 wrong answer（correlated error），confidence 中等。

运行：

```
python3 code/main.py
```

预期输出：一个 (attack, aggregator) -> final answer 的表，并高亮 correct answer。Plurality 在 monoculture case 失败。CPWBFT 的 confidence weighting 缓解 sycophancy。当 monoculture 少于半数 population 时，DecentLLMs 的 geometric-median 拉向 honest cluster。

## 使用它

`outputs/skill-consensus-designer.md` 为 multi-agent ensemble 设计 consensus protocol：clustering method、weighting、threshold，以及 sub-threshold rounds 的 escalation policy。

## 发布它

发布任何 consensus mechanism 前：

- **至少用上述三种 patterns 做 attack-test。** protocol 应该可预测地失败，而不是 silently。
- **记录每个 minority cluster** 及 provenance。minority clusters 是 correlated errors 的 early-warning system。
- **强制 bounded rounds。** 不要 “keep debating until agreement” — 那会奖励 sycophancy。
- **区分 agreement 和 correctness。** consensus output 交给 verifier；verifier 独立于 ensemble。
- **监控 agreement rate。** 急剧上升意味着 conformity bias；急剧下降意味着 model drift。

## 练习

1. 运行 `code/main.py`。确认 plurality 在 monoculture attack 失败，但当 monoculture confidence 低于 0.7 时 CPWBFT 能部分缓解。
2. 添加第四种 attack pattern：**silent abstention** — 一个 agent 拒绝回答（“I don't know”）。每个 aggregator 应如何处理 abstentions？实现你的选择。
3. 把 semantic clustering 从 string canonicalization 换成 embedding-similarity（使用任意 open-source embedding model）。sycophancy attack 会发生什么？
4. 阅读 CP-WBFT（arXiv:2511.10400）。实现 confidence-probe calibration step（一个 separate calibration model 检查每个 agent 的 self-reported confidence）。测量 monoculture scenario 上的 accuracy gain。
5. 阅读 “Can AI Agents Agree?”（arXiv:2603.01213）。复现一个简化 scalar-agreement experiment：三个 agents、一个 scalar question、deceptive-persona prompt。CPWBFT 或 DecentLLMs 能捕捉它吗？

## 关键词汇

| 术语 | 人们常说 | 实际含义 |
|------|----------------|------------------------|
| BFT | “Byzantine fault tolerance” | Castro-Liskov 1999 protocol，用于在 `f < n/3` arbitrary faults 下达成 consensus。 |
| Byzantine | “任何坏行为” | 可以说谎、丢消息、静默失败的 node — 任何非安全 crash 行为。 |
| Confidence probe | “你有多确定？” | 附在 vote 上的 self-reported 或 calibrator-predicted probability。 |
| Semantic clustering | “同答案，不同说法” | 计票前把 equivalent answers 分组。 |
| Geometric median | “Robust center” | 最小化到 sample points 距离和的点。与 mean 不同，对 outliers robust。 |
| Monoculture | “同 model，同 failures” | agents 共享 training data 或 base model 时的 correlated errors。 |
| Sycophantic conformity | “同意最大声的人” | agent 的 vote 偏向最先/最大声发言者。 |
| Core/Edge | “Hierarchical BFT” | WBFT split：小 Core 先 consensus，Edge nodes 跟随。限制 latency。 |

## 延伸阅读

- [Castro & Liskov — Practical Byzantine Fault Tolerance (OSDI 1999)](https://pmg.csail.mit.edu/papers/osdi99.pdf) — 基础论文
- [CP-WBFT — Confidence-Probe Weighted BFT](https://arxiv.org/abs/2511.10400) — 按 confidence 加权 voting
- [DecentLLMs — leaderless multi-agent consensus](https://arxiv.org/abs/2507.14928) — geometric-median aggregation
- [WBFT — Weighted BFT with Hierarchical Structure Clustering](https://arxiv.org/abs/2505.05103) — 用 Core/Edge split 限制 latency
- [Can AI Agents Agree?](https://arxiv.org/abs/2603.01213) — scalar-agreement fragility 和 deceptive-persona attack
