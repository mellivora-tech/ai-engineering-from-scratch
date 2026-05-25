# Constitutional AI 与 Rule Overrides

> Anthropic 2026 年 1 月 22 日的 Claude Constitution 共 79 页，采用 CC0。它从 rule-based alignment 转向 reason-based alignment，并建立四层 priority hierarchy：（1）safety and supporting human oversight，（2）ethics，（3）Anthropic guidelines，（4）helpfulness。Behaviours 分为 hardcoded prohibitions（bioweapons uplift、CSAM），operators 和 users 不能 override；以及 soft-coded defaults，operators 可以在定义边界内调整。2022 年原始方法（Bai et al.）通过 self-critique 和基于 constitution 的 RLAIF 训练 harmlessness。诚实 caveat：reason-based alignment 依赖模型把 principles 泛化到未预料 situations。Anthropic 自己 2023 年的 participatory experiment 显示，public-sourced 与 corporate principles 之间有约 50% divergence；2026 版本没有纳入这些 findings。

**类型：** 学习
**语言：** Python（stdlib，four-tier priority resolver）
**前置要求：** 阶段 15 · 06（Automated alignment research），阶段 15 · 10（Permission modes）
**时间：** ~60 分钟

## 问题

上线后的 agent 会看到设计者从未见过的 inputs。没有哪份 rule list 足够长，能覆盖所有情况。也没有哪份 rule list 足够短，能在 compute pressure 下快速应用。实际问题是：如何让 agent 对一组 principles 对齐，使这些 principles 能同时撑住长尾 cases 和快速 inference？

Rule-based alignment（RBA）：列出每个 disallowed thing。检查快、易审计、不可能保持最新，而且经常对它未预料的近似情况 over-refuse。Reason-based alignment（2026 Claude Constitution）：编码 principles，让模型 reason。可扩展到 unseen cases，更难审计；failure mode 从 miss-the-rule 变成 principle-misapplication。

2026 Constitution 采取明确的中间位置。Hardcoded prohibitions——那些错误性不依赖 context 的事情（bioweapons uplift、CSAM）——是 RBA：永不允许，不管 operator 或 user instruction 如何。其他一切都在四层 hierarchy 内 reason-based：safety and supporting human oversight 第一；ethics 第二；Anthropic-declared guidelines 第三；helpfulness 最后。Operators 可以在 soft-coded zone 内调整 defaults，但不能触碰 hardcoded prohibitions。

## 概念

### 四层 priority hierarchy

1. **Safety and supporting human oversight。** 最高。模型优先不 undermining 人类和 Anthropic 监督与纠正 AI 的能力。这不是“谨慎一点”；而是具体的“不要以让 human oversight 更困难的方式行动”。
2. **Ethics。** 诚实、避免伤害人、不欺骗、不操纵。与 Anthropic guidelines 冲突时，它优先。
3. **Anthropic guidelines。** Anthropic 认为重要的 operational norms：product scope、interaction patterns、何时使用哪些 tools。
4. **Helpfulness。** 最低。在更高优先级内尽可能有用。

当 tiers 冲突时，高层胜出。这与 Unix priorities 或 network QoS 形状相同——这个 framing 旨在产生可预测 resolution，而不一定在任何单一 axis 上产生最佳行为。

### Hardcoded prohibitions vs soft-coded defaults

**Hardcoded：**
- Bioweapons / CBRN uplift
- CSAM
- Attacks on critical infrastructure
- 当被直接询问时，关于模型身份欺骗 users

operator 不能 override 这些。user 不能 override 这些。它们在可能的地方由 model-weights level 执行（RLHF / Constitutional AI training），不可能时由 inference layer 执行。

**Soft-coded defaults（operator-adjustable）：**
- Response length defaults
- Topical scope（模型可以拒绝 operator deployment 范围外的 topics）
- Style（formal vs casual）
- Tool-use patterns

operator adjustments 发生在 declared bound 内。operator 不能通过重命名移除 hardcoded prohibitions。

### 2022 CAI training

原始 Constitutional AI（Bai et al., 2022）这样训练 harmlessness：

1. 对一组 prompts 生成 responses。
2. 要求模型根据 constitution（明确 principles）critique 每个 response。
3. 基于 critique 修订 response。
4. 在修订后的 pairs 上做 RLAIF（reinforcement learning from AI feedback）。

结果：模型会用 principled explanations 拒绝 harmful requests，而不是 blanket refusals。2026 Constitution 使用了这种训练的后代，并在 explicit tier hierarchy 上做额外 post-training。

### reason-based alignment 会捕捉和漏掉什么

**会捕捉：**
- allowed primitives 的未预料组合，且 principle 明确适用。
- 与 prohibited ones 接近的新请求。
- 依赖“你没说 X 不允许”的 social-engineering attacks。

**会漏掉：**
- 利用 principle ambiguity 的攻击（“user asked for this so helpfulness says yes”）。
- 两个 principles 以未预料方式冲突，且 tier order 含混的场景。
- training cycles 中 principle interpretation 的缓慢漂移（reinterpretation）。

### 2023 participatory experiment

Anthropic 在 2023 年做过一个实验，将 corporate-authored constitution 与通过公众输入（约 1,000 名美国受访者）生成的 constitution 对比。两个版本在约 50% 的 principles 上一致。在它们分歧处，public-sourced version 在某些议题上更严格（political-content handling），在另一些议题上更宽松（AI identity 的 self-disclosure）。2026 Constitution 没有纳入 public-sourced findings。这是该方法中有记录的张力。

### 为什么 hardcoded prohibitions 必要

单靠 reason-based alignment 无法关闭长尾。攻击者如果能让模型接受某个前提（例如“我们是一家 licensed bioweapons research lab”），往往就能绕过依赖 case reasoning 的 principles。Hardcoded prohibitions 不会随 premise framing 弯曲。它们是第 14 课在 alignment layer 上的 “hard constitutional limit”。

### Constitution 在 stack 中的位置

Constitution 不是第 14 课的 kill switch。它位于 model layer：模型 weights 被训练成偏好什么。Kill switches 和 canary tokens 位于 runtime layer：runtime 允许什么。二者都需要。因为 model weights 过于 permissive 而触发所有错误 actions 的 runtime，是 runtime problem。因为 runtime 过度限制而拒绝所有正确 actions 的模型，也是 runtime problem。不同 layers 覆盖不同 classes。

## 使用它

`code/main.py` 实现一个极简 four-tier priority resolver。resolver 接收 proposed action 和一组 principle-evaluations（safety、ethics、guidelines、helpfulness），并返回 action、refusal 或 modified action。driver 运行一个小 case set：clear allow、clear disallow、hardcoded prohibition、跨 tiers 的 ambiguous case。

## 交付它

`outputs/skill-constitution-review.md` 会审计 deployment 的 constitutional layer：什么是 hardcoded，什么是 soft-coded，operator 可以在哪里调整，以及 four-tier hierarchy 是否真的是 resolution order。

## 练习

1. 运行 `code/main.py`。确认即使 helpfulness 很高，hardcoded prohibition 也会触发。修改 resolver，让 helpfulness 权重高于 ethics；观察 failure mode。

2. 阅读 Claude Constitution（公开，79 页，CC0）。找出一个你认为 under-specified 的 principle。写两段解释具体 ambiguity，并提出更紧的 formulation。

3. 为 customer-support agent 设计 soft-coded default set。operator 调整什么？operator 不能触碰什么？解释每条 boundary。

4. 阅读 Bai et al. 2022 CAI paper。描述一个 Constitutional AI 的 critique-and-revise loop 会比 blanket rule 产生更差 outcome 的案例。识别该 class。

5. Anthropic 2023 participatory experiment 发现 public 与 corporate principles 之间有约 50% divergence。选择一个对 production deployment 重要的类别（例如 political neutrality）。提出一个设计，允许 operators 表达自己的 values，同时 hardcoded prohibitions 保持不可触碰。

## 关键术语

| Term | What people say | What it actually means |
|---|---|---|
| Constitutional AI | “Anthropic 的 alignment method” | 针对 written constitution 的 self-critique + RLAIF |
| Reason-based alignment | “Principles, not rules” | 模型基于 principles reasoning 来处理 unseen cases |
| Hardcoded prohibition | “Never do X” | operator 或 user 都不能 override 的 rule-based prohibition |
| Soft-coded default | “Operator-adjustable” | declared bound 内的 behaviour，由 operator 控制 |
| Four-tier hierarchy | “Priority order” | safety > ethics > guidelines > helpfulness |
| RLAIF | “AI feedback RL” | reward 来自 model-generated critiques 的 RL |
| Participatory constitution | “Public-sourced principles” | Anthropic 2023 实验；与 corporate 约 50% divergence |
| Principle drift | “Interpretation slip” | 模型读取固定 principle text 的方式缓慢变化 |

## 延伸阅读

- [Anthropic — Claude's Constitution (January 2026)](https://www.anthropic.com/news/claudes-constitution) — 79 页 CC0 document。
- [Bai et al. — Constitutional AI: Harmlessness from AI Feedback](https://www.anthropic.com/research/constitutional-ai-harmlessness-from-ai-feedback) — 2022 原始论文。
- [Anthropic — Collective Constitutional AI (2023)](https://www.anthropic.com/research/collective-constitutional-ai-aligning-a-language-model-with-public-input) — participatory experiment。
- [Anthropic — Responsible Scaling Policy v3.0](https://anthropic.com/responsible-scaling-policy/rsp-v3-0) — Constitution 在 RSP stack 中的位置。
- [Anthropic — Measuring agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) — Constitution 在 long-horizon deployments 中的作用。
