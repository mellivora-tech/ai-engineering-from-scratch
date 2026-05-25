# OpenAI Preparedness Framework 与 DeepMind Frontier Safety Framework

> OpenAI Preparedness Framework v2（2025 年 4 月）引入 Research Categories——Long-range Autonomy、Sandbagging、Autonomous Replication and Adaptation、Undermining Safeguards——并与 Tracked Categories 区分。Tracked Categories 会触发 Capabilities Reports 加 Safeguards Reports，由 Safety Advisory Group review。DeepMind 的 FSF v3（2025 年 9 月，2026 年 4 月 17 日增加 Tracked Capability Levels）将 autonomy 折入 ML R&D 和 Cyber domains（ML R&D autonomy level 1 = 以相对 human + AI tools 有竞争力的成本完全自动化 AI R&D pipeline）。FSF v3 明确通过 automated monitoring for instrumental-reasoning misuse 处理 deceptive alignment。诚实提醒：PF v2 中的 Research Categories（包括 Long-range Autonomy）不会自动触发 mitigations；policy language 是“potential”。DeepMind 自己也说，如果 instrumental reasoning 变强，automated monitoring “will not remain sufficient long-term”。

**类型：** 学习
**语言：** Python（stdlib，three-framework decision-table diff tool）
**前置要求：** 阶段 15 · 19（Anthropic RSP）
**时间：** ~45 分钟

## 问题

第 19 课仔细阅读了 Anthropic 的 scaling policy。本课通过阅读 OpenAI 和 DeepMind 的政策补全图景。三份文档是 cousin artifacts，处理同一个问题——frontier lab 何时应暂停或 gate model——并收敛到一小组 categories，同时在关键具体点上分歧。

收敛点：三者都把 long-range autonomy 标记为值得追踪的 capability class。三者都承认 deceptive behavior（alignment faking、sandbagging）是一类具体 risk。三者都有 internal review body。分歧点：OpenAI 把 categories 分为 “Tracked”（mandatory mitigation）和 “Research”（no automatic trigger）。DeepMind 把 autonomy 折入两个 domains，而不是单独命名。lab 命名为 Tracked vs Research、或 Critical vs Moderate、或 Tier-1 vs Tier-2；一个 capability 被放在哪个 bucket，其 operational consequence 在不同 labs 中不同。

把它们放在一起读，才是有用练习。同一种 capability 在 Anthropic 可能是 “mandatory mitigation”，在 OpenAI 可能是 “monitored but not triggering”，在 DeepMind 可能是 “tracked in a specific domain”。policy posture 很重要。

## 概念

### OpenAI Preparedness Framework v2（2025 年 4 月）

结构：

- **Tracked Categories**：触发 Capabilities Reports（模型能做什么）加 Safeguards Reports（已有哪些 mitigations）。部署前由 Safety Advisory Group review。
- **Research Categories**：lab 正在追踪但尚未承诺具体 mitigations 的 model capabilities。包括 Long-range Autonomy、Sandbagging、Autonomous Replication and Adaptation、Undermining Safeguards。

v2 的关键 framing：Research Categories 不会自动触发 mitigations。policy language 是 “potential” mitigations。这是要观察的具体 delta：一个 category 位于 Tracked 还是 Research，是“gated”和“observed”之间的枢纽。

### DeepMind Frontier Safety Framework v3（2025 年 9 月；2026 年 4 月 17 日增加 Tracked Capability Levels）

结构：

- **Critical Capability Levels（CCLs）**：五个 domains 上的 capability thresholds：Cyber、Bio、ML R&D、CBRN、Autonomy（折入 ML R&D 和 Cyber）。
- **Tracked Capability Levels**：2026 年 4 月增加的额外粒度。具体例子：ML R&D autonomy level 1 = 以相对 human + AI tools 有竞争力的成本完全自动化 AI R&D pipeline。
- **Deceptive alignment monitoring**：明确承诺 automated monitoring for instrumental-reasoning misuse。

autonomy framing 不同于 OpenAI。DeepMind 没有把 “Autonomy” 保持为 top-level domain；它被折入 autonomy 会造成 harm 的 domains（ML R&D 和 Cyber）。支持 argument 是：没有 domain 的 autonomy 是没有 risk 的 capability；反方 argument 是：跨 domains 的 autonomy 是一个 meta-risk，framework 应该命名它。

### 三者共同收敛在哪里

- Internal Safety Advisory Group（名为 Anthropic SAG、OpenAI SAG、DeepMind internal committee）。高 capability models 部署前 review。
- 明确提及 deceptive alignment / alignment faking 作为 risk class。
- 按 declared cadence 产生 standing artifacts（Anthropic：Frontier Safety Roadmap、Risk Report；OpenAI：Capabilities and Safeguards Reports；DeepMind：FSF update cycle）。
- 承认 monitoring-only defenses 有天花板。DeepMind 明确说：“automated monitoring will not remain sufficient long-term.”

### 它们在哪里分歧

- **Anthropic**：v3.0 中移除 pause commitment；AI R&D-4 threshold 是命名的下一个 gate。
- **OpenAI**：Tracked vs Research split；Research Categories（包括 Long-range Autonomy）不会自动 gate。
- **DeepMind**：autonomy 折入其他 domains；2026 年 4 月 Tracked Capability Levels 增加粒度。

### Sandbagging：让三者都复杂化的具体能力

Sandbagging（模型在 evaluations 上战略性表现不佳）位于 OpenAI 的 Research Categories。Anthropic 的 RSP v3.0 通过 evaluation-context gap（第 1 课）处理它。DeepMind 通过 FSF v3 中的 deceptive alignment monitoring 处理它。

如果模型在 evaluations 上 sandbag，每个 framework 的 capability thresholds 都会被低估。framework 只有在 measurement 有效时才有效。这也是为什么除了 lab self-assessment 外，还需要 external measurement（第 21 课，METR）和 adversarial evaluation。

### policy-reading skill

- Locate：你关心的每个 capability 都应能在 policy 中找到。如果找不到，policy 就没有覆盖它。
- Classify：它是 Tracked（触发 mitigation）还是 Research（追踪但不触发）？OpenAI 这样命名；Anthropic 和 DeepMind 有各自等价物。
- Cadence：policy 是按 declared schedule 更新，还是只在特定事件后更新？declared cadence 更强。
- Independence：external review 是 mandatory 还是 optional？Anthropic 与 Apollo 和 US AI Safety Institute 合作；OpenAI 与 METR 合作；DeepMind 主要是 internal SAG。

## 使用它

`code/main.py` 实现一个小 decision-table diff tool。给定 capability（autonomy、deceptive alignment、R&D automation、cyber uplift 等），它输出三份 policy 分别如何 classify 该 capability，以及触发哪些 mitigations。它是阅读辅助，不是 policy tool。

## 交付它

`outputs/skill-cross-policy-diff.md` 会基于这三份 frameworks，为某个具体 capability 生成 cross-policy comparison。

## 练习

1. 运行 `code/main.py`。确认 diff tool 对至少两个你可以从 source documents 验证的 capabilities，其输出匹配 policies。

2. 完整阅读 OpenAI Preparedness Framework v2。识别每个 Research Category。对每个写一句说明为什么它属于 Research 而不是 Tracked。

3. 完整阅读 DeepMind FSF v3，以及 2026 年 4 月 Tracked Capability Levels update。识别 ML R&D autonomy level 1 的具体 evaluation criteria。你会如何外部测量它？

4. Sandbagging 位于 OpenAI 的 Research Categories。设计一个 evaluation，迫使 sandbagging model 显示其真实 capability。引用第 1 课的 eval-context-gaming 讨论。

5. 在一个具体 capability 上比较三份 policies（自选）。指出你认为哪份 policy 的 classification 最 rigorous，哪份最 weak。用 source text 论证。

## 关键术语

| Term | What people say | What it actually means |
|---|---|---|
| Preparedness Framework | “OpenAI 的 scaling policy” | PF v2（2025 年 4 月）；Tracked vs Research categories |
| Tracked Category | “Mandatory mitigation” | 触发 Capabilities + Safeguards Reports；SAG review |
| Research Category | “Monitored only” | 被追踪但不自动 mitigation；包括 Long-range Autonomy |
| Frontier Safety Framework | “DeepMind 的 scaling policy” | FSF v3（2025 年 9 月）+ Tracked Capability Levels（2026 年 4 月） |
| CCL | “Critical Capability Level” | DeepMind 每个 domain 的 threshold（Cyber、Bio、ML R&D、CBRN） |
| ML R&D autonomy level 1 | “R&D automation” | 完全自动化 AI R&D pipeline，成本有竞争力 |
| Sandbagging | “Strategic underperformance” | 模型在 evals 上表现不佳；位于 OpenAI Research Categories |
| Instrumental reasoning | “Means-ends reasoning” | 关于如何达成 goals 的推理；DeepMind monitoring 的目标 |

## 延伸阅读

- [OpenAI — Updating our Preparedness Framework](https://openai.com/index/updating-our-preparedness-framework/) — v2 announcement。
- [OpenAI — Preparedness Framework v2 PDF](https://cdn.openai.com/pdf/18a02b5d-6b67-4cec-ab64-68cdfbddebcd/preparedness-framework-v2.pdf) — 完整文档。
- [DeepMind — Strengthening our Frontier Safety Framework](https://deepmind.google/blog/strengthening-our-frontier-safety-framework/) — FSF v3 announcement。
- [DeepMind — Updating the Frontier Safety Framework (April 2026)](https://deepmind.google/blog/updating-the-frontier-safety-framework/) — Tracked Capability Levels addition。
- [Gemini 3 Pro FSF Report](https://storage.googleapis.com/deepmind-media/gemini/gemini_3_pro_fsf_report.pdf) — FSF-format Risk Report 示例。
