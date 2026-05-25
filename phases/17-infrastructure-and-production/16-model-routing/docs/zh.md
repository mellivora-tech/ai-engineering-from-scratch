# Model Routing 作为 Cost-Reduction Primitive

> Dynamic broker 会评估每个 request（task type、token length、embedding similarity、confidence），把简单 queries 发给便宜模型，把复杂 queries 升级到 frontier model。也叫 model cascading。Production case studies 显示，在 US/UK/EU deployments 中 iso-quality 下可降低 20-60% 成本；高 volume SaaS 中 30% routing efficiency improvement 会变成六位数 annual savings。2026 年背景是 LLM inference prices 每年约下降 10x：GPT-4-class token 从 2022 年末的 $20/M 降到 2026 年约 $0.40/M。大部分下降来自更好的 serving stacks（阶段 17 · 04-09），而不是硬件。Routing 是你在不造成 product regression 的情况下，把这轮 price drop 转化成 margin 的方式。Failure mode 是 cheap-model drift：route 把 40% 推给较弱模型，reasoning tasks 质量下降 3-5%，一个季度没人注意。用 online quality metrics gate routes，而不只是 offline eval sets。

**类型：** 学习
**语言：** Python（stdlib，玩具版 cascading router simulator）
**前置要求：** 阶段 17 · 01（Managed LLM Platforms），阶段 17 · 19（AI Gateways）
**时间：** ~60 分钟

## 学习目标

- 解释 model cascading：cheap-first with confidence check，在 low confidence 时 escalate。
- 枚举四个 routing signals（task classification、prompt length、embedding similarity to known-hard set、self-confidence from first-pass）。
- 在目标 routing split 和 quality loss tolerance 下计算 expected blended cost。
- 说出捕获 cheap-model creep 的 drift-monitoring metric（online quality gate）。

## 问题

你的服务在 GPT-5 上每月花 $80k。Analytics 显示 70% queries 很简单：“what time is it in Paris?” “rephrase this sentence.” Haiku-class model 能以 3% 成本完美处理它们。30% 需要 GPT-5 的 reasoning：coding、math、multi-step planning。

如果把 70% 路由到 cheap，30% 路由到 expensive，你的账单在相同 product quality 下会降低约 65%。这就是 routing。技巧是构建 broker，同时不造成质量回退。

## 概念

### 四个 routing signals

1. **Task classification**：simple/complex/codegen/math/chat。可以是 rules-based classifier、小 LLM（Haiku-class at $0.25/M），或与 labeled buckets 的 embedding similarity。输出：route = cheap / balanced / frontier。

2. **Prompt length**：prompts >4K tokens 往往需要 frontier 保持 coherence。Prompts <500 tokens 通常不需要。

3. **Embedding similarity to known-hard set**：如果 query 接近（cosine > 0.88）known-hard bucket，直接 escalate 到 frontier。

4. **Self-confidence from first-pass**：先发给 cheap；如果模型 log-probs 显示 low confidence，或者 refuse，或者输出 hedging language，就在 frontier 上 retry。对约 10% 流量增加 P95 latency，但在其他 90% 上节省 50%+。

### 三种 patterns

**Pre-route**（前置 classifier）：增加约 5-10ms latency；整体最快。

**Cascade**（cheap-first，low confidence 时 escalate）：median latency 约 1.2x（cheap run plus verify），escalated 时约 2x。质量地板最好。

**Ensemble route**（对样本并行运行 cheap 和 frontier，由 reward-model 选择）：质量最高、成本最高；只用于关键 A/B。

### 实现

AI gateways（阶段 17 · 19）暴露 routing。LiteLLM 有带 fallback 和 cost-routing 的 `router` config。Portkey 有 guards + routing。Kong AI Gateway 有 plugin-based routing。OpenRouter 的 model marketplace 暴露 recommendation API。

Open-source：RouteLLM（LMSYS）、Not Diamond（commercial）、Prompt Mule。

### 2026 price curve

| Model class | Late 2022 | 2026 | Change |
|-------------|-----------|------|--------|
| GPT-4-level quality | ~$20/M | ~$0.40/M | 50x cheaper |
| Frontier (GPT-5, Claude 4) | — | ~$3-10/M | new tier |

大部分改进来自 serving efficiency：阶段 17 · 04-09 的核心课程变成了 provider-side cost drops。Routing 让你在 app layer 捕获这些收益，而不是等待所有用户迁移到 cheap tier。

### Drift 才是真正风险

你的 route 把 40% 发给 cheap model。六个月后，task distribution 变化（用户更复杂，问题更长）。Router 没注意，因为 classifier 是用 Q1 data 训练的。质量静默下降。没人抱怨到足够响。直到 competitor benchmark 中你输了才发现。

用 online quality metrics gate routes：

- 每条 route 的 user thumbs-up / thumbs-down。
- 每条 route 上 held-out sample（5%）的 automated LLM-judge。
- Escalation rate：如果 cascade up-route >30%，说明 cheap model 被过度路由。
- 每条 route 的 refusal rate。

### 你应该记住的数字

- 2026 年 iso-quality routing savings：case studies 中 20-60%。
- LLM price drop 2022-2026：总计约每年 10x。
- GPT-4-level 2022 vs 2026：~$20/M → ~$0.40/M。
- Cascade latency impact：median ~1.2x，escalated ~2x（约 10% 流量）。

## 使用它

`code/main.py` 在 mixed workload 上模拟 pre-route、cascade 和 ensemble。报告 blended cost、quality loss 和 escalation rate。

## 交付它

本课会产出 `outputs/skill-router-plan.md`。给定 workload 和 quality budget，它会选择 routing pattern 和 signals。

## 练习

1. 运行 `code/main.py`。在什么 accuracy floor 下 cascade 胜过 pre-route？
2. 你的用户群是 30% enterprise（复杂 queries）、70% free tier（简单）。设计 routing split。用什么 online metric gate 它？
3. 一条 route 让质量下降 2%，但节省 40%。能 ship 吗？取决于产品，分别论证。
4. 使用 OpenAI / Anthropic APIs 的 logprobs 实现 confidence check。起始 threshold 设多少？
5. 六个月内 escalation rate 从 8% 升到 22%。诊断三个原因，并给出每个的修复。

## 关键词汇

| 术语 | 大家常说 | 实际含义 |
|------|----------|----------|
| Model routing | “cost broker” | 对每个 request 动态选择 model |
| Model cascade | “cheap-first escalate” | 先跑 cheap，low confidence 时 fall through 到 frontier |
| Pre-route | “classify first” | 前置 classifier；不 rerun |
| Ensemble route | “parallel pick” | 运行多个模型，由 reward-model 选最佳 |
| Escalation rate | “uprouted %” | cascade requests 中被升级的比例 |
| RouteLLM | “LMSYS router” | OSS router library |
| Not Diamond | “commercial router” | SaaS model-routing product |
| Drift | “cheap creep” | Distribution shift 而 router 未察觉 |
| Online quality gate | “live check” | 对 live traffic 做 automated LLM-judge sampling |

## 延伸阅读

- [AbhyashSuchi — Model Routing LLM 2026 Best Practices](https://abhyashsuchi.in/model-routing-llm-2026-best-practices/)
- [Lukas Brunner — Rise of Inference Optimization 2026](https://dev.to/lukas_brunner/the-rise-of-inference-optimization-the-real-llm-infra-trend-shaping-2026-4e4o)
- [RouteLLM paper / code](https://github.com/lm-sys/RouteLLM)
- [Not Diamond — model routing](https://www.notdiamond.ai/)
- [OpenRouter](https://openrouter.ai/) — 带 routing primitives 的 multi-model gateway。
