# Failure Modes：Agents 为什么会坏

> MASFT（Berkeley, 2025）把 14 种 multi-agent failure modes 归入 3 类。Microsoft 的 Taxonomy 记录了已有 AI failures 如何在 agentic settings 中放大。Industry field data 收敛到五种反复出现的模式：hallucinated actions、scope creep、cascading errors、context loss、tool misuse。

**类型：** 学习 + 构建
**语言：** Python (stdlib)
**前置要求：** 阶段 14 · 05（Self-Refine and CRITIC），阶段 14 · 24（Observability）
**时间：** ~60 分钟

## 学习目标

- 说出 MASFT 的三类 failure categories，以及每类至少四个具体 modes。
- 解释为什么 agentic failure 会放大已有 AI failure modes（bias、hallucination）。
- 描述五个 industry-recurring modes 及其 mitigations。
- 实现一个 stdlib detector，为 agent traces 打上 failure-mode labels。

## 问题

团队发布的 agents 在 90% traces 上可用。那 10% 失败不是随机噪声 — 它们落在少数反复出现的类别里。一旦能命名它们，你就能监控它们并修复它们。

## 概念

### MASFT (Berkeley, arXiv:2503.13657)

Multi-Agent System Failure Taxonomy。14 种 failure modes 聚成 3 类。Inter-annotator Cohen's Kappa 0.88 — 这些类别可以被可靠地区分。

核心主张：failures 是 multi-agent systems 的基础设计缺陷，而不是能靠更好的 base models 修好的 LLM limitations。

### Microsoft Taxonomy of Failure Mode in Agentic AI Systems

- 已有 AI failures（bias、hallucination、data leakage）在 agentic settings 中放大。
- Autonomy 带来新 failures：unintended action at scale、tool misuse、mission drift。
- 这份 whitepaper 是 agentic products 的 risk register。

### Characterizing Faults in Agentic AI (arXiv:2603.06847)

- Failures 来自 orchestration、internal state evolution 和 environment interaction。
- 不只是 “bad code” 或 “bad model output”。

### LLM Agent Hallucinations Survey (arXiv:2509.18970)

两种主要表现：

1. **Instruction-following Deviation** — agent 没有遵循 system prompt。
2. **Long-range Contextual Misuse** — agent 忘记或误用早期 turns 中的 context。

Sub-intention errors：Omission（漏步骤）、Redundancy（重复步骤）、Disorder（步骤乱序）。

### 五个 industry-recurring modes

Arize、Galileo、NimbleBrain 2024-2026 field analyses 收敛到：

1. **Hallucinated actions。** Agent 调用不存在的 tool，或编造 arguments。
2. **Scope creep。** Agent 把任务扩大到 user ask 之外（创建额外 PRs、发送额外 emails）。
3. **Cascading errors。** 一个错误 call 触发下游 effects。一个 phantom SKU hallucination 触发四个 API calls — 变成 multi-system incident。
4. **Context loss。** Long-horizon tasks 忘记 early-turn constraints。
5. **Tool misuse。** 调用了正确 tool 但参数错，或直接调用了错误 tool。

Cascading 是杀手。Agents 无法区分 “我失败了” 和 “任务不可能完成”，经常在 400 errors 上 hallucinate success message 来闭环。

### Mitigation：每一步都有 gates

在 reasoning chain 的每一步设置 automated verification gates，对照 environment state 检查 factual grounding。具体来说：

- Per-step safety classifier（第 21 课）。
- Tool-call argument validation（第 06 课）。
- 把 retrieved content 与 known facts cross-check（第 05 课，CRITIC）。
- 通过重新探测 state 检测 success hallucination（文件真的创建了吗？）。

### Failure monitoring 会在哪里出错

- **只标记 crashes。** 大多数 agent failures 会产出看起来有效的 output。需要 content-level checks。
- **没有 baseline。** Drift detection 需要 last-known-good；没有它就无法说 “正在变糟”。
- **Over-alerting。** 每个 failure 都 page。要 cluster 并 rate-limit。

## 构建它

`code/main.py` 实现了一个 stdlib failure-mode tagger：

- 覆盖五种 modes 的 synthetic trace dataset。
- 每个 mode 一个 detector function（针对 tool calls、outputs、repeat actions 的 signature patterns）。
- 一个 tagger，为每条 trace 打标签并报告 mode distribution。

运行它：

```
python3 code/main.py
```

输出：per-trace labels + aggregate distribution，这是 Phoenix trace clustering 所暴露内容的低成本复现。

## 使用它

- **Phoenix** 用于 production drift clustering（第 24 课）。
- **Langfuse** 用于 session replay + annotation。
- **Custom** 用于 observability platform 检测不到的 domain-specific signatures。

## 发布它

`outputs/skill-failure-detector.md` 会生成适合你领域的 failure-mode detectors，并接到 trace store。

## 练习

1. 添加一个 “success hallucination” detector：agent 返回 success，但 target state 没有变化。
2. 标记你构建过的产品中的 100 条真实 traces。哪种 mode 占主导？修复成本是多少？
3. 实现一个 “cascade radius” metric：给定第 N 步的 failure，它影响了多少下游步骤？
4. 阅读 MASFT 的 14 个 failure modes。选三个适用于你产品的，写 detectors。
5. 把一个 detector 接入 CI job：如果 >=5% traces 标记某个 mode，就 fail build。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|----------------|------------------------|
| MASFT | "Multi-agent failure taxonomy" | Berkeley 14-mode categorization |
| Cascading error | "Ripple failure" | 一个早期错误传播到 N 个 steps |
| Context loss | "Forgot the constraint" | Long-horizon turn 丢失 early-turn facts |
| Tool misuse | "Wrong tool / wrong args" | 有效 call，但 invocation 错 |
| Success hallucination | "Faked completion" | Agent 在 400 上声称 success；state 没变 |
| Scope creep | "Overreach" | Agent 做了超出要求的事 |
| Instruction-following deviation | "Disobedience" | 忽略 system prompt 或 user constraint |
| Sub-intention errors | "Plan bugs" | Plan execution 中的 omission、redundancy、disorder |

## 延伸阅读

- [Cemri et al., MASFT (arXiv:2503.13657)](https://arxiv.org/abs/2503.13657) — 14 failure modes，3 categories
- [Microsoft, Taxonomy of Failure Mode in Agentic AI Systems](https://cdn-dynmedia-1.microsoft.com/is/content/microsoftcorp/microsoft/final/en-us/microsoft-brand/documents/Taxonomy-of-Failure-Mode-in-Agentic-AI-Systems-Whitepaper.pdf) — risk register
- [Arize Phoenix](https://docs.arize.com/phoenix) — drift clustering in practice
- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) — 什么时候 simpler patterns 可以完全避开这些 modes
