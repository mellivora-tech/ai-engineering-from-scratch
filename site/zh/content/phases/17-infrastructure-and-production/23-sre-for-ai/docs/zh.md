# AI SRE — Multi-Agent Incident Response、Runbooks、Predictive Detection

> AI SRE 使用 LLM，通过 RAG 绑定基础设施数据（logs、runbooks、service topology），自动化 investigation、documentation 和 coordination 阶段。2026 年 architecture pattern 是 multi-agent orchestration：specialized agents（logs、metrics、runbooks）由 supervisor 协调；AI 提出 hypotheses 和 queries，人类批准 judgment calls。Datadog Bits AI 和 Azure SRE Agent 以 managed products 形式交付。Runbooks 正在演进：NeuBird Hawkeye 使用 adversarial evaluation（两个模型分析同一 incident；agreement = confidence，disagreement = uncertainty）；operational memory 跨团队变动持久保存。Auto-remediation 保持谨慎：AI 建议，人类批准。Fully autonomous action 很窄（restart pod、rollback specific deploy）且有严格 guardrails；任何销售 “set it and forget it” 的都在过度承诺。Emerging frontier 是 pre-incident prediction。MIT 研究报告称，一个基于 historical logs + GPU temps + API error patterns 训练的 LLM 能提前 10-15 min 预测 89% outages。预测：到 2026 年底，95% enterprise LLMs 拥有 automated failover。

**类型：** 学习
**语言：** Python（stdlib，玩具版 multi-agent incident triage simulator）
**前置要求：** 阶段 17 · 13（Observability），阶段 17 · 24（Chaos Engineering）
**时间：** ~60 分钟

## 学习目标

- 画出 multi-agent AI SRE architecture：supervisor + specialized agents（logs、metrics、runbooks）+ human approval gate。
- 解释为什么 auto-remediation 是 narrow（restart pod、revert deploy），而不是 broad（re-architect service）。
- 说出 adversarial evaluation pattern（NeuBird Hawkeye）：两个模型同意 = confidence；不同意 = escalate。
- 引用 MIT 89% early-detection result 和 operational constraint：没有 actuation 的 predictions 只是 dashboards。

## 问题

一名 on-call engineer 凌晨 3 点被 page：“High error rate in checkout。”他们检查 Datadog、Loki、三个 runbooks、deploy log。30 分钟后，他们意识到 root cause 是 KV cache spike 导致 vLLM OOM。重启 pod 后 error 消失。

2026 年，这段调查的前 20 分钟可以自动化。按 service 分组 logs、关联 recent deploys、匹配 runbooks，都是 RAG + tool-use。一个 supervised agent 可以做 first-pass triage，并在人类打开 Datadog 前展示 hypothesis。

Fully autonomous remediation 是另一个问题。Restart pod：安全。Scale GPU pool：policy 允许时安全。Re-architect service：绝对不行。Discipline 就是画出这条窄线。

## 概念

### Multi-agent architecture

```
          Incident
             │
             ▼
        Supervisor
        /    |    \
       ▼     ▼     ▼
  Log agent  Metric agent  Runbook agent
       │     │     │
       └─────┴─────┘
             │
             ▼
        Hypothesis + evidence
             │
             ▼
        Human approval
             │
             ▼
        Action (narrow set)
```

Supervisor 把 incident 拆成 sub-queries。Specialized agents 拥有 tool access（log search、PromQL、doc retrieval）。Supervisor 综合结果，把 hypothesis + evidence 呈现给人类。人类批准或重定向。

### Auto-remediation scope

**Safe（narrow）**：restart pod、revert specific deploy、在 pre-approved bounds 内 scale pool、启用 pre-approved feature flag。

**Not safe（broad）**：change service topology、modify resource limits、deploy new code、change IAM、alter databases。

任何销售 “set it and forget it” 的都在过度承诺。随着 AI SRE 成熟，safe set 会增长，但边界是真实的。

### Adversarial evaluation（NeuBird Hawkeye）

两个模型独立分析同一 incident。如果它们在 root cause 上一致，confidence 高。如果不同意，就把两个 hypotheses 都展示给人类并 escalate。模式简单，是过滤 hallucinated root causes 的有效方式。

### Operational memory

团队流动是传统 SRE 的 silent kill：tribal knowledge 会离开。AI SRE 把 runbooks + post-mortems 存到 vector DB；agents 在每个新 incident 上 retrieve。新工程师加入时，AI 拥有完整历史。

### Pre-incident prediction

MIT 2025 研究：基于 historical logs、GPU temperatures、API error patterns 训练的 LLM，在 test set 上能提前 10-15 分钟预测 89% outages。

Reality check：没有 actuation 的 predictions 只是 dashboards。Operational question 是“当我们预测时，要做什么？”Pre-emptive drain？Pager？Auto-scale？答案取决于 policy。

### 2026 年产品

- **Datadog Bits AI** — Datadog 内的 managed SRE copilot。
- **Azure SRE Agent** — Azure-native。
- **NeuBird Hawkeye** — adversarial eval + operational memory。
- **PagerDuty AIOps** — triage + deduplication。
- **Incident.io Autopilot** — incident commander + coordination。

### Runbooks as code

Runbooks 正在从 Confluence pages 演进为带 structured sections（symptom、hypothesis、verify、act）的 versioned markdown。Structured runbooks 让 RAG retrieval 更好。任何 AI-SRE rollout 都应从把 unstructured runbooks 转为 structured 开始。

### 你应该记住的数字

- MIT early-detection：89% outages，10-15 min lead time。
- Multi-agent triage：supervisor +（logs、metrics、runbooks）+ human。
- Safe auto-remediation set：restart pod、revert deploy、bounded scale。
- Adversarial eval：两个模型独立；agreement = confidence。

## 使用它

`code/main.py` 模拟 multi-agent triage：log agent 找到 error，metric agent 找到 CPU spike，runbook agent 匹配 known issue。Supervisor 对 hypotheses 排序。

## 交付它

本课会产出 `outputs/skill-ai-sre-plan.md`。给定当前 on-call、incident volume、team maturity，它会设计 AI SRE rollout。

## 练习

1. 运行 `code/main.py`。如果 log 和 metric agents 不一致，会发生什么？Supervisor 如何 resolve？
2. 为你的 service 定义三个 “safe” auto-remediation actions。说明每个理由。
3. 写一个 structured runbook template：sections、required fields、verification commands。
4. Predictive detection 以 12 min lead 触发。你的 policy 是什么：pager、pre-drain，还是二者？
5. 论证一个 3-person team 是否应该在 2026 年采用 AI SRE，还是等待。考虑 maturity、volume、risk。

## 关键词汇

| 术语 | 大家常说 | 实际含义 |
|------|----------|----------|
| AI SRE | “agent for on-call” | LLM-backed incident investigation + coordination |
| Supervisor agent | “the orchestrator” | 把 incidents 拆成 sub-queries 的 top-level agent |
| Specialized agent | “domain agent” | 有 tool access 的 sub-agent（logs、metrics、runbooks） |
| Auto-remediation | “AI fixes it” | 狭窄、pre-approved action；不是 broad re-architecture |
| Operational memory | “vector runbooks” | 为 RAG 存入 vector DB 的 post-mortems + runbooks |
| Adversarial eval | “two-model check” | 独立 analyses；agreement = confidence |
| NeuBird Hawkeye | “the adversarial one” | 带 adversarial-eval + memory pattern 的产品 |
| Bits AI | “Datadog's SRE agent” | Datadog-managed AI SRE |
| Pre-incident prediction | “early detection” | outage prediction 的 10-15 min lead time |

## 延伸阅读

- [incident.io — AI SRE Complete Guide 2026](https://incident.io/blog/what-is-ai-sre-complete-guide-2026)
- [InfoQ — Human-Centred AI for SRE](https://www.infoq.com/news/2026/01/opsworker-ai-sre/)
- [DZone — AI in SRE 2026](https://dzone.com/articles/ai-in-sre-whats-actually-coming-in-2026)
- [Datadog Bits AI](https://www.datadoghq.com/product/bits-ai/)
- [NeuBird Hawkeye](https://www.neubird.ai/)
- [awesome-ai-sre](https://github.com/agamm/awesome-ai-sre)
