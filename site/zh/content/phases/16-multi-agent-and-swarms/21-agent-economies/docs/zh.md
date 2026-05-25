# Agent Economies、Token Incentives、Reputation

> Long-horizon autonomous agents（METR 的 1-hour 到 8-hour work-curve）需要 economic agency。正在成形的 **5-layer stack** 是：**DePIN**（physical compute）→ **Identity**（W3C DIDs + reputation capital）→ **Cognition**（RAG + MCP）→ **Settlement**（account abstraction）→ **Governance**（Agentic DAOs）。Production agent-incentive networks 包括 **Bittensor**（TAO subnets 奖励 task-specific models）、**Fetch.ai / ASI Alliance**（ASI-1 Mini LLM + FET token）和 **Gonka**（transformer-based PoW，把 compute 重新分配到 productive AI tasks）。学术工作：AAMAS 2025 的 decentralized LaMAS 使用 **Shapley-value credit attribution** 公平奖励 contributing agents；Google Research “Mechanism design for large language models” 提出带 monotone aggregation 的 **token auctions** 和 second-price payment。本课构建 minimal agent marketplace，将 Shapley-value credit attribution 应用于 multi-agent pipeline，并运行 second-price token auction，让 game-theory machinery 具体落地。

**类型：** 学习
**语言：** Python（stdlib）
**前置要求：** 阶段 16 · 16（Negotiation and Bargaining），阶段 16 · 09（Parallel Swarm Networks）
**时间：** ~75 分钟

## 问题

当 agents 共同产生价值却需要 individually rewarded 时，multi-agent systems 会变复杂。classical mechanisms — equal split、last-contributor-takes-all — 要么不公平，要么容易被 game。通过 Shapley values 做 coalition-based rewarding 在构造上公平，但计算昂贵。2025-2026 literature 推出有用 approximations：Shapley sampling、monotone aggregation auctions，以及从 confirmed contributions 中累积的 on-chain reputation。

除了 credit attribution，field 已经转向真实 economic agents：Bittensor TAO 奖励 mining compute 来 fine-tune subnet-specific models，Fetch.ai/ASI 用 FET tokens 奖励 ASI-1 Mini LLM usage，Gonka 将 transformer proof-of-work 重新分配到 productive AI tasks。自主交易的 agents 今天已经存在；问题是如何 align incentives。

本课把 agent economies 当作具体 problem family — credit attribution、mechanism design 和 reputation — 并用最少数学构建它们，让 ideas 记住。

## 概念

### 5-layer agent-economy stack

1. **DePIN（physical compute）。** 租用 GPU、storage、bandwidth 的 decentralized infrastructure。Bittensor subnets、Render Network、Akash。不是 agent-specific；agents 使用它。
2. **Identity。** W3C Decentralized Identifiers（DIDs）给每个 agent 一个独立于平台的 durable ID。Reputation 累积到 DID 上。Agent Network Protocol（ANP）使用 DID 作为 discovery layer。
3. **Cognition。** agent 的 reasoning loop：LLM + RAG + MCP。这是其他 phases 构建的内容。
4. **Settlement。** Account abstraction（ERC-4337）让 agents 从自己的 balances 支付 gas，而不必持有 ETH。Agents 可以为 services、彼此或 compute 支付。
5. **Governance。** Agentic DAOs：humans *and* agents 对 protocol changes 投票，voting power 与 reputation 绑定。

不是每个 production system 都用五层。Bittensor 使用 1、2，部分使用 3、部分使用 4，没有 5。OpenAI agents 除了 3 之外都不用。这个 stack 是 reference map，不是 requirement。

### Bittensor、Fetch.ai、Gonka — 运行中的东西

**Bittensor（TAO）。** Subnets 是 specialized tasks（language modeling、image generation、forecasting）。Miners 提交 model outputs。Validators rank them；stake-weighted scoring 分配 TAO rewards。每个 subnet 有自己的 evaluation。经济 lesson：按 task-specific output quality 付费，而不是按 compute used。

**Fetch.ai / ASI Alliance。** ASI-1 Mini LLM 运行在 Fetch.ai network 上；users 用 FET tokens 为 inference 付费。agents-as-peers narrative 在这里更强：Fetch 上的 agent 可以 call 另一个 agent 做任务并用 FET 支付。

**Gonka。** Transformer proof-of-work：“work” 是 transformer forward passes。miners 通过运行有已知正确 outputs（来自 training data）的 inference tasks 获得收益。resource-productive PoW，而不是 hash-based PoW。

截至 2026 年 4 月，三者都是 production-grade。payoff distribution 不同。Bittensor 根据 subnet validators 奖励相对质量；Fetch 根据 paying users 衡量 utility；Gonka 奖励 verifiable inference work。

### Shapley-value credit attribution

三个 agents 协作完成一个 task。output 得分 0.8。谁贡献了什么？

Shapley value：满足四个 axioms（efficiency、symmetry、linearity、null）的唯一 credit allocation。对 agent `i`：

```
shapley(i) = (1/N!) * sum over all orderings O of (v(S_i_O ∪ {i}) - v(S_i_O))
```

其中 `S_i_O` 是 ordering `O` 中 `i` 之前的 agents 集合。实践中：枚举所有 permutations，记录每个 agent 在每个 permutation 中的 marginal contribution，取平均。

N=3 agents 时有 6 permutations。N=10 时有 3.6M — 所以实践中用 sample orderings，而不是 enumerate。

### Aggregation 的 second-price auction

Google Research（“Mechanism design for large language models”）提出用于 aggregating LLM outputs 的 second-price token auctions。设置：N 个 agents 各自 propose 一个 completion；每个都对 being selected 有 private value。auctioneer 选择 highest-value proposal，并支付 *second-highest* value。在 monotone aggregation 下（value 取决于哪个 proposal 被选，而不是有多少 bid），这是 truthful — agents 会 bid true value。

这对 LLM systems 重要：你可以把 completion tasks 外包给多个不同 pricing 的 agents；auction 选择 best + fair payment，agents 没有 misreport 的 incentive。

### Reputation capital

DID-bound reputation score 从 confirmed contributions 中累积。一个简单 update rule：

```
rep(i, t+1) = alpha * rep(i, t) + (1 - alpha) * contribution_quality(i, t)
```

decay factor `alpha` 接近 1。Reputation：

- 对 routing decisions 便宜可读（“send hard tasks to high-rep agents”）。
- 伪造成本高（随时间累积，绑定 DID）。
- 可 slash：verification 失败的 contributions 会扣分。

### AAMAS 2025 decentralized LaMAS

LaMAS proposal（AAMAS 2025）结合了：DID identity、Shapley-value credit attribution 和简单 auction mechanism。核心 claim：decentralizing credit attribution step 让系统可 audit，并免疫 single-point manipulation。

### 经济机制在哪里崩坏

- **Price oracle manipulation。** 如果 credit function 可被 game，agents 就会 game 它。每个 mechanism 都需要 adversarial test。
- **Sybil attacks。** 一个 operator 启动 N 个 fake agents 来膨胀自己的 contribution。DIDs 会减慢但不会阻止；reputation cost-to-forge 是 mitigation。
- **Verification cost。** credit attribution 的公平性取决于 verifier。verification 如果便宜（small LLM），可被 game；如果昂贵（human panel），系统无法扩展。
- **Regulatory overhang。** Agent economies 与 financial regulation 相交。截至 2026 年，Bittensor、Fetch 和 Gonka 在一些 jurisdictions 处于 legal gray areas。

### Agent economies 什么时候合理

- **有 heterogeneous operators 的 open networks。** 没有单一 team 控制所有 agents。
- **Verifiable outputs。** 没有 verification，credit attribution 就是猜测。
- **Long-horizon workflows。** One-shot tasks 不受益于 reputation accumulation。
- **Tokenized payments 在你 jurisdiction legally viable。**

在 closed corporate systems 中，economics 会让位于更简单的 allocation（managers assign work，metrics are internal）。economics literature 主要适用于 open networks。

## 构建它

`code/main.py` 实现：

- `shapley(value_fn, agents)` — 对小 N exact enumeration 的 Shapley computation。
- `second_price_auction(bids)` — truthful mechanism；winner pays second-highest。
- `Reputation` — 带 exponential decay 和 slashing 的 DID-bound reputation。
- Demo 1：三个 agents 协作，exact Shapley 分配 credit。
- Demo 2：五个 agents 为一个 task slot 出价；second-price auction 选择 winner + payment。
- Demo 3：100 轮 task assignment 给 heterogeneous rep agents；rep-weighted routing warmup 后比 random 更好。

运行：

```
python3 code/main.py
```

预期输出：每个 agent 的 Shapley values；auction result 展示 truthful-bid equilibrium；rep-weighted routing 在 warmup 后比 random 有 10-20% quality gain。

## 使用它

`outputs/skill-economy-designer.md` 设计 minimal agent economy：identity layer、credit attribution mechanism、payment mechanism、reputation rule 的选择。

## 发布它

2026 年运行 agent economy：

- **从 reputation 开始，而不是 tokens。** Reputation 便宜且单独有价值；tokens 增加 legal 和 economic complexity。
- **reward 前先 verify。** 永远不要没有 independent verification 就分配 credit。self-reported quality 会累积 sybil games。
- **Shapley-sample，不要 Shapley-exact。** sample 100-1000 orderings；exact enumeration 不可扩展。
- **限制 decay factor 并设置 reputation floor。** 无界 decay 会抹掉合法 contributors；太慢的 decay 会奖励 stale high-rep agents。
- **对 mechanisms 做 adversarial audit。** 开放 network 前先运行 red-team scenarios。每个 mechanism 都有 game theory；你要在 attackers 之前找到洞。

## 练习

1. 运行 `code/main.py`。确认 Shapley values 加总为 total value（efficiency axiom）。改变 value function；Shapley allocations 是否按预期方向变化？
2. 实现 Shapley *sampling*（对 K 个 orderings 做 Monte Carlo）。K 如何影响 approximation accuracy？与 N=4 的 exact 比较。
3. 在 auction 前实现 coalition-forming step：agents 可以合并为 teams 并作为 unit bid。哪些 coalitions 形成？结果是否比 individual bidding Pareto-better？
4. 阅读 Google Research mechanism-design post。识别一个违反后会破坏 truthfulness 的 assumption。在 LLM setting 中这个 failure mode 长什么样？
5. 阅读 AAMAS 2025 decentralized LaMAS paper。在 10 agents synthetic task 上实现它们的 Shapley step。exact computation 需要多久？100 draws 的 sampling 有多接近？

## 关键词汇

| 术语 | 人们常说 | 实际含义 |
|------|----------------|------------------------|
| DePIN | “Decentralized physical infrastructure” | token-incentivized compute/storage/bandwidth。Bittensor、Akash、Render。 |
| DID | “Decentralized identifier” | portable IDs 的 W3C spec。Agent reputation 绑定到 DID，而不是 platform。 |
| ERC-4337 | “Account abstraction” | 可 sponsor gas 的 contract accounts，使 agent payments 成为可能。 |
| Shapley value | “Fair credit attribution” | 满足 efficiency、symmetry、linearity、null 的唯一 allocation。 |
| Second-price auction | “Vickrey auction” | truthful mechanism：winner pays second-highest bid。兼容 monotone aggregation。 |
| Reputation capital | “Accumulated quality score” | DID-bound score，来自 confirmed contributions；随时间 decay。 |
| Agentic DAO | “Agents + humans govern” | 把 agent voters 作为 first-class、voting power 与 reputation 绑定的 DAO。 |
| TAO / FET / GPU credits | “Token denominations” | Bittensor TAO、Fetch.ai FET、各种 DePIN tokens。 |

## 延伸阅读

- [The Agent Economy](https://arxiv.org/abs/2602.14219) — 2026 年 5-layer agent-economy stack survey
- [Google Research — Mechanism design for large language models](https://research.google/blog/mechanism-design-for-large-language-models/) — 带 monotone aggregation 的 token auctions
- [AAMAS 2025 — decentralized LaMAS](https://www.ifaamas.org/Proceedings/aamas2025/pdfs/p2896.pdf) — Shapley-value credit attribution
- [Bittensor TAO documentation](https://docs.bittensor.com/) — subnet structure 和 reward distribution
- [Fetch.ai / ASI Alliance](https://fetch.ai/) — ASI-1 Mini LLM 和 FET token
- [W3C Decentralized Identifiers (DIDs) spec](https://www.w3.org/TR/did-core/) — identity foundation
