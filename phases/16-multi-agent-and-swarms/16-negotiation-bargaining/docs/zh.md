# Negotiation 与 Bargaining

> Agents 会 negotiation resources、prices、task allocations 和 terms。2026 年 benchmark set 很清楚：NegotiationArena（arXiv:2402.05863）显示 LLMs 能通过 persona manipulation（“desperation”）把 payoffs 提高约 20%；“Measuring Bargaining Abilities”（arXiv:2402.15813）显示 buyer 比 seller 更难，scale 没有帮助 — 他们的 **OG-Narrator**（deterministic offer generator + LLM narrator）把 deal rate 从 26.67% 推到 88.88%；Large-Scale Autonomous Negotiation Competition（arXiv:2503.06416）运行约 180k negotiations，发现 **chain-of-thought-concealing** agents 通过向 counterparts 隐藏 reasoning 获胜；Bhattacharya et al. 2025 按 Harvard Negotiation Project metrics 排名：Llama-3 最 effective，Claude-3 aggressive，GPT-4 fairest。本课实现 Contract Net Protocol（FIPA ancestor，第 02 课），接入 LLM-style buyer/seller，运行 OG-Narrator-style decomposition，并测量每个 structural choice 如何改变 deal rate。

**类型：** 学习 + 构建
**语言：** Python（stdlib）
**前置要求：** 阶段 16 · 02（FIPA-ACL Heritage），阶段 16 · 09（Parallel Swarm Networks）
**时间：** ~75 分钟

## 问题

两个 agents 需要就价格达成一致。让它们用纯 language prompts 自行 negotiation 时，2024-2026 LLMs 在 tightly-parameterized bargains 上 close deals 的比例出奇地低（arXiv:2402.15813 中约 27%）。Scale 不能修复它：GPT-4 在 bargaining 结构上不比 GPT-3.5 好；它只是更擅长 bargaining 的 *language*。

根本问题是 LLMs 混淆了两个工作 — 决定 offer 和叙述 offer。OG-Narrator 把两者分开：deterministic offer generator 计算 numeric moves；LLM 只做 narration。Deal rate 跳到约 89%。

这呼应了 classical multi-agent 发现：decoupling the mechanism from the communication layer 会赢。Contract Net Protocol（FIPA, 1996; Smith, 1980）是参考 task-market mechanism。把 LLM 插到 narration slot，就得到 modern LLM-powered task market。

## 概念

### 一段话理解 Contract Net

Smith 1980 的 Contract Net Protocol：**manager** broadcast 一个 **call for proposals (cfp)**；**bidders** 用包含 offers 的 **propose** messages 响应；manager 选择 winner，并给 winner 发送 **accept-proposal**，给 losers 发送 **reject-proposal**。winner 执行 work。可选 message：**refuse**（bidder 拒绝 propose）。FIPA 将其编码为 `fipa-contract-net` interaction protocol。

### 为什么 OG-Narrator 胜出

“Measuring Bargaining Abilities of Language Models”（arXiv:2402.15813）观察到：

- LLMs 经常破坏 bargaining rules（报出 nonsensical prices，忽略对方 ZOPA）。
- 它们 anchor 很差（接受糟糕 first offers；counter-offer 按 symbolic 而非 strategic amounts）。
- Scale alone 不能修复。更大的 models 写出更合理的语言，但 strategic error 类似。

OG-Narrator decomposition：

```
           ┌──────────────────┐        ┌──────────────────┐
  state  → │ offer generator  │ price → │  LLM narrator    │ → message
           │  (deterministic) │        │  (writes the     │
           │                  │        │   human-style    │
           └──────────────────┘        │   accompaniment) │
                                       └──────────────────┘
```

offer generator 是 classical negotiation strategy：Rubinstein bargaining model、Zeuthen strategy，或简单的 price tit-for-tat。LLM narrates。message 包含 deterministic price 和 natural-language framing。

Deal rate 跳升是因为：
- Prices 保持在 bargaining zone 内。
- Anchors 是 strategic，而不是 emotional。
- LLM 做它擅长的事：写作。

### NegotiationArena findings

arXiv:2402.05863 提供 canonical benchmark。headline findings：

- LLMs 可以通过 adoption personas（“I am desperate to sell this by Friday”）把 payoffs 提高约 20% — persona manipulation 是真实 tactic。
- Fair/cooperative agents 会被 adversarial ones exploit；防御需要显式 counter-posturing。
- Symmetric pair-ups 在约 40% benchmark scenarios 中收敛到 inequitable outcomes。

这不是 “LLMs are bad negotiators”。而是 “LLMs negotiate too much like humans, including the exploitable parts.”

### Chain-of-thought concealment

Large-Scale Autonomous Negotiation Competition（arXiv:2503.06416）在许多 LLM strategies 上运行约 180k negotiations。赢家隐藏 reasoning：

- 如果一个 agent 把 “I will only go to $75; my reservation price is $70” 写进 publicly visible scratchpad，对手会读到。
- 赢家私下计算 strategy；output channel 只包含 offer 和 minimum required narration。

这是 2026 年对 classical game theory 的呼应（Aumann 1976 关于 rationality and information）：暴露 private valuation 会损失 payoff。LLMs 不会直觉到这一点，会高兴地把 reservations 写进对 counterpart 可见的 reasoning traces。

Engineering takeaway：分离 private-scratchpad context 和 public-message context。不是可选项。

### Bhattacharya et al. 2025 — model rankings

按 Harvard Negotiation Project metrics（principled negotiation、BATNA respect、interest reciprocity）：

- **Llama-3** 最擅长达成 bargains（deal rate + payoff）。
- **Claude-3** 是最 aggressive negotiator（high anchors、late concessions）。
- **GPT-4** 最 fair（pairings 之间 payoff variance 最小）。

这是 2025 snapshot。重点不是 2026 年 4 月哪个 model 赢，而是不同 base models 有持久 negotiation styles。Heterogeneous ensembles（第 15 课）把它作为 diversity source。

### 通过 Contract Net + LLM 做 task allocation

LLM multi-agent 中对 Contract Net 的现代复用：

1. Manager agent 把 task 分解为 units。
2. 对 worker agents broadcast 带 task description 的 `cfp`。
3. 每个 worker 返回 offer：`(price, eta, confidence)`，price 可以是 tokens、compute units 或 dollars。
4. Manager 选择 winners（单个或多个，取决于 task）并 award。
5. 被 reject 的 workers 可以自由 bid 其他 tasks。

这能很好扩展到 100+ workers，因为 coordination 是 broadcast-and-respond，不是 synchronous chat。production 中使用：Microsoft Agent Framework 的 orchestration patterns，一些 LangGraph implementations。

### LLM-Stakeholders Interactive Negotiation

NeurIPS 2024（https://proceedings.neurips.cc/paper_files/paper/2024/file/984dd3db213db2d1454a163b65b84d08-Paper-Datasets_and_Benchmarks_Track.pdf）引入带 **secret scores** 和 **minimum-acceptance thresholds** 的 multi-party scorable games。每个 stakeholder 有 private utilities；LLM 必须从 messages 中推断。这是 two-party bargaining 到 N-party coalition formation 的泛化。对具有 heterogeneous worker capabilities 的 production task markets 有关。

### Narration-vs-mechanism rule

在所有 2024-2026 negotiation benchmarks 中，一条一致 engineering rule 是：

> Let the LLM narrate. Do not let the LLM compute the offer.

如果 offer 需要是数字（price、ETA、quantity），从 negotiation state deterministic 地生成，并让 LLM 产出 framing。如果 offer 需要是 proposal structure（task decomposition、role assignment），可以让 LLM draft，但在发送前必须用 schema validate 和 constraint-check。

## 构建它

`code/main.py` 实现：

- `ContractNetManager`、`ContractNetTask`、`Bid` — manager + bidders，broadcast cfp、collect proposals、award。
- `og_narrator_bargain(state, rng)` — OG-Narrator buyer：deterministic Zeuthen-style concession toward midpoint。
- `seller_response(state, rng)` — deterministic seller counter-offer policy（两种风格的 structural ground truth）。
- `naive_llm_bargain(state, rng)` — 模拟 all-LLM bargainer：高方差地选 prices，经常落在 ZOPA 外。
- Measurement：每个 trial 重新 sample reservation prices，运行 1000 trials 的 deal rate。

运行：

```
python3 code/main.py
```

预期输出：naive-LLM deal rate 约 65-75%；OG-Narrator deal rate 约 85-95%；15-25 point gap 是把 offer-generation 与 narration 分解的 structural advantage。另有一个包含三个 bidders 和一个 task 的 Contract Net task-market allocation example。

## 使用它

`outputs/skill-bargainer-designer.md` 设计 bargaining protocol：谁生成 offers（deterministic 或 LLM）、谁 narrates、private scratchpads 如何与 public messages 分离，以及如何监控 deal rate。

## 发布它

Production bargaining checklist：

- **Separate scratchpad。** Private state 永远不进入 counterpart 的 context。这一点不可谈判。
- **Deterministic offer generation。** Prices、quantities、ETAs：compute，不要 prompt。
- **Validate all incoming offers** against schema。在 protocol boundary 拒绝 out-of-ZOPA offers。
- **Bound rounds。** 最多 3-5 rounds；deadlock 时 escalate to mediator。
- **持续测量 deal rate 和 payoff variance。** deal rate 下降是 symptom — 通常是 prompt drift 或 counterpart-side attack。
- **记录所有 rejected proposals** 及 deterministic rationale。对 Contract Net managers，losing bidders 需要理解为什么。

## 练习

1. 运行 `code/main.py`。确认 OG-Narrator 在 deal rate 上超过 naive-LLM。差距是多少？
2. 实现 **persona-based payoff improvement**（arXiv:2402.05863）— buyer 仅在 narration 中采用 “desperate to buy this week” persona，offer generator 不变。deal rate 或 payoff 变化了吗？
3. 实现 chain-of-thought **concealment**：维护一个不传给 counterpart 的 private scratchpad string。如果意外泄露它会怎样（通过交换 channels 来模拟）？
4. 把 Contract Net 扩展为带 reserve price 的 N-bidder auction。当所有 bids 都超过 reserve 时，manager 如何在 lowest-price 与 highest-quality 之间选择？你选哪个 award rule，为什么？
5. 阅读 Bhattacharya et al. 2025 的 Harvard Negotiation Project metrics。实现两个不同 styles（aggressive vs fair）的 bargainers。测量 symmetric 和 asymmetric pairings 下的 payoff variance。

## 关键词汇

| 术语 | 人们常说 | 实际含义 |
|------|----------------|------------------------|
| Contract Net | “Task market” | Smith 1980, FIPA 1996。cfp + propose + accept/reject。canonical task-market。 |
| ZOPA | “Zone of possible agreement” | buyer max 与 seller min 的重叠。外部 offers 不可能 close。 |
| BATNA | “Best alternative to a negotiated agreement” | 交易失败时的 fallback。决定 reservation price。 |
| OG-Narrator | “Offer generator + narrator” | decomposition：deterministic offer，LLM narration。 |
| Zeuthen strategy | “Risk-minimizing concession” | 基于 risk limits 做 concession 的 classical offer-generator。 |
| Rubinstein bargaining | “Alternating-offer equilibrium” | 带 discounting 的 infinite-horizon bargaining game-theoretic model。 |
| CoT concealment | “隐藏你的 reasoning” | arXiv:2503.06416 的赢家使用 private scratchpads；public channel 只显示 offer。 |
| Persona manipulation | “Emotional posturing” | arXiv:2402.05863：desperation/urgency personas 带来约 20% payoff gain。 |

## 延伸阅读

- [NegotiationArena](https://arxiv.org/abs/2402.05863) — benchmark；persona manipulation 和 exploitation findings
- [Measuring Bargaining Abilities of Language Models](https://arxiv.org/abs/2402.15813) — OG-Narrator 和 buyer-harder-than-seller result
- [Large-Scale Autonomous Negotiation Competition](https://arxiv.org/abs/2503.06416) — 约 180k negotiations；chain-of-thought concealment 获胜
- [LLM-Stakeholders Interactive Negotiation (NeurIPS 2024)](https://proceedings.neurips.cc/paper_files/paper/2024/file/984dd3db213db2d1454a163b65b84d08-Paper-Datasets_and_Benchmarks_Track.pdf) — 带 secret utilities 的 multi-party scorable games
- [Smith 1980 — The Contract Net Protocol](https://ieeexplore.ieee.org/document/1675516) — classical mechanism，IEEE Transactions on Computers
