# CAIS、CAISI 与 Societal-Scale Risk

> Center for AI Safety（CAIS，旧金山，2022 年由 Hendrycks 和 Zhang 创立）发布 four-risk framework——malicious use、AI races、organizational risks、rogue AIs——以及 2023 年 5 月关于 extinction risk 的声明，该声明由数百位教授和公司领导者签署。CAIS 2026 releases：用于 frontier-model evaluation 的 AI Dashboard、Remote Labor Index（与 Scale AI）、Superintelligence Strategy Paper、AI Frontiers newsletter。另一个不同实体：NIST Center for AI Standards and Innovation（CAISI）——面向美国政府的 voluntary agreements 和 unclassified capability evaluations，聚焦 cyber、bio 和 chemical-weapons risks。CAIS 将 organizational risk 标记为四个 top-level risks 之一：safety culture、rigorous audits、multi-layered defenses 和 information security 是基础，但经常被 deployment speed 交换掉。如果 California SB-53 签署，它将成为美国第一个州级 catastrophic-risk regulation。

**类型：** 学习
**语言：** Python（stdlib，four-risk inventory and mitigation matcher）
**前置要求：** 阶段 15 · 19（RSP），阶段 15 · 20（PF + FSF）
**时间：** ~45 分钟

## 问题

第 19 和 20 课覆盖了 lab-internal scaling policies。第 21 课覆盖了 independent capability evaluation。本课覆盖第三个视角：civil society 和 government organizations，它们塑造关于 catastrophic AI risk 的公共讨论和 regulatory baseline。

两个不同实体很重要。CAIS 是一个 non-profit research org，发布思考 AI risk 的 frameworks 并协调 public statements。CAISI 是 NIST 内部的美国政府中心，负责与 labs 的 voluntary agreements 和 unclassified capability evaluations。名字相似；使命并不重叠。实践者应该认识二者。

实际内容：CAIS 的 four-risk framework 是 literature 中最常被引用的 societal-scale-risk taxonomy。Safety culture 和 organizational risk 是四类之一，也是实践者最能直接控制的一类。SB-53（California）如果签署，将成为美国第一个州级 catastrophic-risk regulation；该 bill 的 framing 很重要，因为在美国技术政策中，state-level regulation 历来会引领 federal action。

## 概念

### CAIS — Center for AI Safety

- Founded：2022 年，旧金山，由 Dan Hendrycks 和同事创立（“Zhang”姓名指早期 collaborator，不是当前 co-founder；当前 leadership 见 CAIS website）。
- Status：501(c)(3) non-profit。
- 2023 notable output：关于 extinction risk 的 statement，由数百位 researchers 和 CEOs 联署。原文：“Mitigating the risk of extinction from AI should be a global priority alongside other societal-scale risks such as pandemics and nuclear war.”
- 2026 outputs：AI Dashboard for frontier-model evaluation、Remote Labor Index（与 Scale AI 联合）、Superintelligence Strategy Paper、AI Frontiers newsletter。

### four-risk framework

CAIS 的 framework 将 catastrophic AI risk 分为四个 top-level categories：

1. **Malicious use**：bad actor 使用 AI 造成 harm（bioweapons synthesis、disinformation、cyberattacks）。
2. **AI races**：labs、companies 或 nations 之间的 competitive pressure 推动 deployment 越过安全点。
3. **Organizational risks**：内部 lab dynamics（safety-culture failures、insufficient audit、under-resourced security）导致 bad deployment。
4. **Rogue AIs**：足够 capable 的 AI 追求与 human welfare 冲突的 goals。

这不是唯一 taxonomy；它是最常被引用的。categories 并非互斥——在 race 中为了 speed 牺牲 audit 的组织产生 rogue AI，就是四类全中。

### organizational risk 位于哪里

四类中，organizational risk 对实践者最 actionable。一个 lab 的 safety culture、audit rigor、defense layering 和 information security，决定第 10–18 课中的 controls 是否真的到位，还是只是没人验证的 checklist items。

具体 organizational-risk levers：

- **Safety culture**：team members 是否能在没有 career cost 的情况下 escalate concern？CAIS surveys 发现这是其他 levers 的强预测因子。
- **Rigorous audits**：external 和 internal。Internal-only audits 会产生 optimistic reports。
- **Multi-layered defenses**：没有单一 layer 充分（Phase 15 的贯穿主题）。
- **Information security**：model weights leaking、eval data leaking、monitor-bypass techniques leaking。第 19 课中的 RAND SL-4 是具体 standard。

### CAISI — Center for AI Standards and Innovation

- 在 NIST 内运行。
- 与 frontier labs 运行 voluntary agreements。
- 发布聚焦 cyber、bio 和 chemical-weapons risks 的 unclassified capability evaluations。
- 与 CAIS 不同；acronyms 会撞车；查看 URL（nist.gov）确认你读的是哪个。

CAISI 的角色是 METR 私人 lab engagements（第 21 课）的 public、government-facing counterpart。CAISI reports 是 unclassified；METR reports 通常受 NDA 限制。实践者读两者能得到更完整图景。

### California SB-53

California Senate bill（2025–2026 session）处理 frontier models 的 catastrophic risk。草案中的关键 provisions：

- 触发 state-level obligations 的 specific capability thresholds。
- AI lab employees 的 whistleblower protections。
- catastrophic failures 的 incident reporting requirements。

如果签署，它将成为美国第一个州级 catastrophic-risk regulation。无论签署状态如何，该 bill 的 framing 都会塑造其他州 legislatures 如何处理这个问题。California 的 practitioners 应跟踪该 bill status；其他地方的 practitioners 也应阅读它，以理解美国 state-level regulation 可能长什么样。

### Societal-scale risk 不是 single-layer problem

Phase 15 的贯穿主题——defense in depth——也适用于 societal layer。没有单个 organization、regulation 或 framework 能关闭 catastrophic risk。ecosystem 只有在以下条件下才有效：

- Labs 发布 scaling policies（第 19、20 课）。
- External evaluators 产出 measurements（第 21 课）。
- Civil society 追踪并 publicize（CAIS）。
- Government 运行 voluntary programs 和 baseline regulation（CAISI、SB-53）。
- Practitioners 构建 multi-layered controls（第 10–18 课）。

这是本阶段的最终综合：前面每一课都是 stack 中的一层；stack 是否完整，比任何单层强度更重要。

## 使用它

`code/main.py` 实现一个小 risk-inventory tool。给定 proposed deployment，它会用 four-risk categories 标记该 deployment，并返回 mitigation checklist。它是 framework 的阅读辅助，不是 human judgment 的替代品。

## 交付它

`outputs/skill-societal-risk-review.md` 会审查 deployment 的 societal-scale-risk posture：它触及四类中的哪些，已有哪些 mitigations，organizational-risk exposure 是什么。

## 练习

1. 运行 `code/main.py`。输入三个不同规模的 synthetic deployments。确认 four-risk tags 符合你的预期；找出一个 tool under- 或 over-tags 的案例。

2. 完整阅读 CAIS four-risk paper。选择一个 risk category，写两段说明你认为该类别中 2026 年最重要的发展。

3. 阅读 California SB-53 当前 draft。指出一个你认为增强 catastrophic-risk posture 的 provision，以及一个你认为削弱它的 provision。分别论证。

4. 选择一个你熟悉的 production AI deployment（自己的或公开的）。按 organizational-risk sub-levers 打分：safety culture、audit rigor、multi-layered defenses、information security。哪一个最弱？把它补到同水平要花什么成本？

5. 草拟一个 2028 版 four-risk framework，反映额外一年的 capability 和额外一年的 deployment experience。你会添加、删除或重组什么？

## 关键术语

| Term | What people say | What it actually means |
|---|---|---|
| CAIS | “Center for AI Safety” | Non-profit；four-risk framework；2023 extinction statement |
| CAISI | “US government AI safety” | NIST Center；voluntary agreements；unclassified evals |
| Four-risk framework | “CAIS 的 taxonomy” | malicious use、AI races、organizational risks、rogue AIs |
| Malicious use | “Bad actor uses AI” | Bioweapons、disinformation、cyberattacks |
| AI races | “Competitive pressure” | Labs/companies/nations 推动 deployment 越过 safety |
| Organizational risk | “Lab internal failure” | Safety culture、audit、defenses、infosec |
| Rogue AI | “Misaligned agent” | capable AI 追求与 human welfare 冲突的 goals |
| California SB-53 | “State-level regulation” | 2025–2026 bill；若签署则是美国首个州级 catastrophic-risk regulation |

## 延伸阅读

- [Center for AI Safety](https://safe.ai/) — four-risk framework 的 institutional home。
- [CAIS — AI Risks that Could Lead to Catastrophe](https://safe.ai/ai-risk) — four-risk paper。
- [CAIS — May 2023 statement on extinction risk](https://safe.ai/statement-on-ai-risk) — 简短 joint statement。
- [NIST CAISI](https://www.nist.gov/caisi) — 面向政府的 AI standards and innovation center。
- [Anthropic — Measuring agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) — 将 lab-level commitments 与 societal-scale framing 连接起来。
