# LLM Production 的 Chaos Engineering

> 2026 年，LLM 的 chaos engineering 是自己的 discipline。在 production 中运行 experiments 前的 prerequisites：已定义 SLI/SLO、trace+metric+log observability、automated rollback、runbooks、on-call。Architecture 有四个 planes：control（experiment scheduler）、target（services、infra、data stores）、safety（guards + abort + traffic filters）、observability（metrics + traces + logs）、feedback（进入 SLO adjustments）。Guardrails 是必需的：如果 daily error-budget burn > 2x expected，burn-rate alerts 会暂停 experiments；suppression windows + trace-ID correlation 会去重 alert noise。Cadence：每周 small canary + SLO review；每月 game day + postmortem；每季度 cross-team resilience audit + dependency mapping。LLM-specific experiments：memory overload、network failures、provider outages、malformed prompts、KV cache eviction storms。Tooling：Harness Chaos Engineering（LLM-derived recommendations、blast-radius downscaling、MCP tool integration）；LitmusChaos（CNCF）；Chaos Mesh（CNCF Kubernetes-native）。

**类型：** 学习
**语言：** Python（stdlib，玩具版 chaos experiment runner）
**前置要求：** 阶段 17 · 23（SRE for AI），阶段 17 · 13（Observability）
**时间：** ~60 分钟

## 学习目标

- 说出五个 chaos engineering prerequisites（SLI/SLO、observability、rollback、runbooks、on-call），并解释为什么跳过任何一个都会破坏实践。
- 画出四个 planes（control、target、safety、observability）以及进入 SLO 的 feedback loop。
- 枚举五个 LLM-specific experiments（memory overload、network fail、provider outage、malformed prompt、KV eviction storm）。
- 给定 stack，选择工具：Harness、LitmusChaos、Chaos Mesh。

## 问题

传统 stack 中的 chaos testing 已经成熟。LLM stacks 增加了新的 failure modes。一个带 poison character 的 4K-token prompt 会让 tokenizer 卡 12 秒。上游 provider 429；gateway retries；你的 service 在 retry-amplified concurrency 下 OOM。Burst load 下的 KV cache eviction storm 引发 re-prefill cascades，饱和 compute。

这些都不会出现在 unit tests 里。Chaos engineering 是你在用户发现之前发现它们的方法。

## 概念

### Prerequisites

没有以下条件，不要在 production 中运行 chaos：

1. **SLI/SLO** — 已定义 service-level indicators 和 objectives。
2. **Observability** — traces、metrics、logs，接到 dashboards。
3. **Automated rollback** — 阶段 17 · 20 的 policy-flag rollback。
4. **Runbooks** — structured，阶段 17 · 23。
5. **On-call** — 有人响应。

缺任何一个，chaos 都会变成真实 incident。

### Four planes + feedback

**Control plane** — experiment scheduler（Litmus workflow、Chaos Mesh schedule、Harness UI）。

**Target plane** — services、pods、nodes、load balancers、data stores。

**Safety plane** — kill switch、suppression windows、blast-radius limits、error-budget gates。

**Observability plane** — 正常 metrics + trace-ID correlation，用于区分 chaos-induced 与 natural failures。

**Feedback loop** — findings 回流到 SLO adjustment、runbook updates、code fixes。

### Guardrails 是必需的

- **Burn-rate alert**：如果 daily error-budget burn 超过 expected 的 2x，暂停 experiment。
- **Suppression windows**：在 experiment 期间，静默 blast radius 内的非 experiment alerts。
- **Trace-ID correlation**：所有 experiment-induced errors 都带 tag，让 on-call 能去重。

### 五个 LLM-specific experiments

1. **Memory overload** — 通过发送 high concurrency 的 long-context requests 强制 KV cache preemption storm。观察：service 是 graceful shed 还是 crash？

2. **Network failure** — 切断 inference gateway 与 provider 之间的 connectivity。观察：fallback 是否在 SLA 内启动？（阶段 17 · 19）

3. **Provider outage simulation** — OpenAI 100% 429。观察：routing 是否 failover 到 Anthropic？（阶段 17 · 16、19）

4. **Malformed prompt** — 注入 tokenizer-stalling payload（例如 deeply nested unicode、huge UTF-8 codepoint）。观察：单个请求是否 lock up worker？

5. **KV eviction storm** — 通过饱和 vLLM block budget 强制 eviction。观察：LMCache 是否恢复，还是 service degrade？

### Cadence

- **Weekly** — staging 中的小 canary experiments，也许 5% prod。
- **Monthly** — 针对具体 scenario 的 scheduled game day；cross-team attendance；postmortem。
- **Quarterly** — cross-team resilience audit；dependency map update。

### Tooling

- **Harness Chaos Engineering** — commercial；AI-derived experiment recommendations；blast-radius downscaling；MCP tool integration。
- **LitmusChaos** — CNCF graduated；基于 Kubernetes workflow。
- **Chaos Mesh** — CNCF sandbox；Kubernetes-native CRD style。
- **Gremlin** — commercial；广泛支持。
- **AWS FIS** / **Azure Chaos Studio** — managed cloud offerings。

### 从小开始

第一个 experiment：在 steady traffic 下 pod-kill 一个 decode replica。观察 rerouting 和 recovery。如果这安全且有效，再升级到 network chaos。

第一个 LLM-specific experiment：注入一个 provider 429，持续 5 分钟。观察 fallback。多数团队会发现自己的 fallback 并没有被完整测试过。

### 你应该记住的数字

- 四个 planes：control、target、safety、observability。
- Burn-rate pause：expected daily budget burn 的 2x。
- Cadence：weekly canary、monthly game day、quarterly audit。
- 五个 LLM experiments：memory、network、provider、malformed prompt、KV storm。

## 使用它

`code/main.py` 模拟三个带 safety plane gates 的 chaos experiments。报告哪些 experiments 会触发 burn-rate abort。

## 交付它

本课会产出 `outputs/skill-chaos-plan.md`。给定 stack 和 maturity，它会选择前三个 experiments 和 tooling。

## 练习

1. 运行 `code/main.py`。哪个 experiment 触发 burn-rate gate，为什么？
2. 为 vLLM-based RAG service 设计前五个 chaos experiments。包括 success criteria。
3. 你的 burn-rate alert 暂停了一个 experiment。如何判断 root cause 是 chaos 还是 natural？
4. 论证 chaos 应该在 production 还是只在 staging 运行。什么时候 production 是正确答案？
5. 说出三个 generic network-chaos 无法复现的 LLM-specific failure modes。

## 关键词汇

| 术语 | 大家常说 | 实际含义 |
|------|----------|----------|
| SLI / SLO | “service targets” | Indicator + objective；必需 prerequisite |
| Blast radius | “scope” | experiment 影响的 services / users 集合 |
| Burn-rate alert | “budget gate” | error-budget burn rate > 2x expected 时触发 |
| Game day | “monthly drill” | Scheduled cross-team chaos exercise |
| LitmusChaos | “CNCF workflow” | Graduated CNCF Kubernetes chaos tool |
| Chaos Mesh | “CNCF CRD” | CNCF sandbox Kubernetes-native chaos |
| Harness CE | “commercial AI-assisted” | 带 AI recommendations 的 Harness chaos |
| Malformed prompt | “tokenizer bomb” | 会卡住 tokenization 的输入 |
| KV eviction storm | “preemption cascade” | 大量 eviction 触发 re-prefills |

## 延伸阅读

- [DevSecOps School — Chaos Engineering 2026 Guide](https://devsecopsschool.com/blog/chaos-engineering/)
- [Ankush Sharma — Observability for LLMs (book)](https://www.amazon.com/Observability-Large-Language-Models-Engineering-ebook/dp/B0DJSR65TR)
- [LitmusChaos (CNCF)](https://litmuschaos.io/)
- [Chaos Mesh (CNCF)](https://chaos-mesh.org/)
- [Harness Chaos Engineering](https://www.harness.io/products/chaos-engineering)
- [AWS FIS](https://aws.amazon.com/fis/)
