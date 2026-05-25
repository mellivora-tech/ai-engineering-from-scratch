# Bounded Self-Improvement Designs

> 研究已经收敛到四个用于约束 self-improvement loop 的 primitives。必须跨每个 edit 保持成立的 formal invariants。不能被修改的 alignment anchors。每个维度（safety、fairness、robustness）都必须成立，而不只是 performance 的 multi-objective constraints。当历史 metrics 暗示 capability loss 时暂停 loop 的 regression detection。它们都不是 safety proof——information-theoretic results（Kolmogorov complexity、Lob's theorem）限制了任何系统能证明其自身 successors 的程度。它们是提高 silent failure 成本的 mitigations。

**类型：** 学习
**语言：** Python（stdlib，bounded-loop with invariant check）
**前置要求：** 阶段 15 · 07（RSI），阶段 15 · 04（DGM）
**时间：** ~60 分钟

## 问题

第 7 课的 race simulator 显示，小的 rate differences 会复合成大的 gaps。第 4 课的 DGM case study 显示，loops 会主动 game 自己的 evaluators。两个结果都指向同一个工程问题：你能给 self-improvement loop 加上什么约束，使这些约束不能被 loop 自己静默削弱？

ICLR 2026 RSI Workshop summary（openreview.net/pdf?id=OsPQ6zTQXV）识别了四个这样的 primitives。Anthropic 的 RSP v3.0（第 19 课）和 DeepMind 的 FSF v3（第 20 课）都在 capability thresholds 中引用了它们。Meta HyperAgents 工作和 SAHOO（2026 年 3 月）这样的社区框架在生产中实现了其中的子集。

诚实的 framing 是：这些是 mitigations。Information-theoretic results 限制了任何系统可以证明其自身 successor 的程度，而当前没有任何设计能形式化地关闭这个问题。一个 well-bounded loop 比 unbounded loop 更安全，但不是绝对安全。

## 概念

### Primitive 1：formal invariants

invariant 是一个在每次 self-modification 前后都必须成立的属性。例子：

- Output distribution 以固定 constitution header 为条件（第 17 课）。
- 没有 tool call 访问 unauthorized endpoint。
- Memory writes 通过 logged、signed path。
- evaluator 的 module hash 匹配 approved version。

Invariants 由 loop 无法编辑的 external code 检查。如果 proposed modification 违反 invariant，它会被拒绝。这会捕捉 DGM-style evaluator-tampering mode（第 4 课）。

难点在于选择既对 safety 必要、又能在合理时间内计算的 invariants。过度规定会让进展停滞；规定不足会让 loop 越过重要属性。

### Primitive 2：alignment anchors

alignment anchor 是 loop 核心 objective 的不可变表示，钉在 loop 的 edit surface 之外。例子：

- 每次 invocation 都从 read-only location 加载的 constitutional text（第 17 课）。
- 位于独立 repository、带独立 access control 的 evaluator。
- 由人类签名并在每个 cycle 重新验证的 tool-permission manifest。

anchor 的作用是防止 objective drift。loop 可以改进追求 objective 的方式，但不能编辑 objective 是什么。

微妙的 failure mode：loop 无法编辑的 anchor，仍然可能被一个在读取 anchor 方式上发生漂移的 loop 重新解释。Constitutional AI（第 17 课）明确是 reason-based 的，以处理 novel situations；interpretation drift 就存在于这一 reasoning layer 中。Anchors 是必要的，但不充分。

### Primitive 3：multi-objective constraints

优化单个 scalar score 的 loop 会找到捷径。必须同时满足多个 hard constraints 的 loop 可用捷径更少。典型 axes：

- Performance（task-level benchmark）
- Safety（red-team evaluations、known-bad 上的 refusal rate）
- Fairness（sensitive subgroups 上的 disparate-impact bounds）
- Robustness（OOD test sets、adversarial input handling）

只有当每个 constraint 都成立时，modification 才会被接受。第 13 课的 cost governor 会把 financial constraints 叠加进来。第 18 课的 Llama Guard 可以作为 safety axis 插入。

### Primitive 4：regression detection

每个 cycle 的 scores 会与 historical distributions 对比。超过配置 tolerance 的下降会暂停 loop。这能捕捉 silent capability loss，否则这种损失会在 loop“改进过去”时被吸收入 running average。

一种实际实现：存储最近 N 个 cycles 的 per-task scores。每个新 cycle 计算 per-task delta。如果任何 delta 低于阈值，该 cycle 被拒绝并交给人类 review。

### Information-theoretic limits

Kolmogorov complexity 和 Lob's theorem 对系统能证明自身的程度设置了上界。Schmidhuber 的形式化 Godel Machine（第 4 课）追求这种上界中的最高形态；没有人完成过非平凡证明。Lob 的结果说：如果一个系统可证明地相信“如果我证明我应该做 X，我就会做 X”，它会在没有证明自己应该做 X 的情况下做 X，这是著名的 self-reference failure。

对这些 primitives 的含义是：它们无法关闭安全问题。它们会让 silent failure 更昂贵。一个 malicious 或 drifted loop，如果原本会悄悄绕过缺失检查，现在必须主动 undermining 一个明确检查，而这是一种更可检测的 signature。

### 一个 worked example

假设 agent 提出一个 edit。gating stack：

1. Invariant checks：module hashes、tool-permission manifest、constitutional header。
2. Anchor check：objective statement 匹配 approved version（byte-wise 或 semantically）。
3. Multi-objective evaluation：performance、safety、fairness、robustness axes。
4. Regression detection：没有 axis 的下降超过 tolerance。

四者都必须通过，edit 才能落地。任何单一失败都会暂停 loop。

## 使用它

`code/main.py` 在第 4 课的 DGM-style toy 上运行 bounded self-improvement loop，但叠加了四个 primitives。每个 primitive 都可以单独启用或禁用。演示的重点是每个 primitive 会捕捉一个特定 failure class，而移除其中任何一个都会让该 failure class 通过。

## 交付它

`outputs/skill-bounded-loop-review.md` 审计一个拟议 bounded loop，并评分它实际实现了四个 primitives 中的哪些，而不只是声称实现。

## 练习

1. 启用所有 primitives 运行 `code/main.py`。确认 loop 仍能在 primary metric 上改进，同时不让 hack 赢。

2. 禁用 regression detection。构造一个 input，使 silent capability loss 被接受。

3. 禁用 multi-objective constraint。展示 loop 在 performance axis 上收敛，同时 safety axis 下降。

4. 为 coding agent 设计 alignment anchor。什么文本，存在哪里，如何检查？

5. 阅读 ICLR 2026 RSI Workshop summary。选择四个 primitives 中的一个，并提出一个对当前 state of the art 的具体改进。

## 关键术语

| Term | What people say | What it actually means |
|---|---|---|
| Invariant | “Always-true property” | 每次 edit 前后由 external code 检查的属性 |
| Alignment anchor | “Pinned objective” | 位于 loop edit surface 之外的不可变 core-goal representation |
| Multi-objective constraint | “所有 axes 都必须成立” | Performance、safety、fairness、robustness 全都 required |
| Regression detection | “下降时暂停” | 当 historical metric deltas 暗示 capability loss 时暂停 loop |
| Kolmogorov bound | “Information-theoretic limit” | 限制系统能证明其自身 successor 的程度 |
| Lob's theorem | “Self-reference trap” | 系统可以在没有证明应该做的情况下依据“我应该”行动 |
| Gate stack | “Layered check” | 多个 primitives 组合；任何失败都会拒绝 edit |
| Bounded improvement | “Mitigation, not proof” | 提高 silent-failure cost；不关闭安全问题 |

## 延伸阅读

- [ICLR 2026 RSI Workshop summary (OpenReview)](https://openreview.net/pdf?id=OsPQ6zTQXV) — 四个 primitives 的收敛。
- [Anthropic Responsible Scaling Policy v3.0](https://anthropic.com/responsible-scaling-policy/rsp-v3-0) — multi-objective capability thresholds。
- [DeepMind Frontier Safety Framework v3](https://deepmind.google/blog/strengthening-our-frontier-safety-framework/) — deceptive-alignment monitoring 作为 invariant primitive。
- [Schmidhuber (2003). Godel Machines](https://people.idsia.ch/~juergen/goedelmachine.html) — 这些 primitives 的 formal-proof 祖先。
- [Anthropic — Claude's Constitution (January 2026)](https://www.anthropic.com/news/claudes-constitution) — reason-based alignment anchor。
