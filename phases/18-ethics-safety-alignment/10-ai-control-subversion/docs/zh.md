# AI Control：在 Subversion 下保持 Safety

> Greenblatt、Shlegeris、Sachan、Roger（Redwood Research，arXiv:2312.06942，ICML 2024）。Control 重新框定 safety 问题：给定一个可能在对抗性地反优化你的 untrusted strong model U，什么 protocol 能让你提取有用工作，同时约束不可接受结果的概率？资源包括一个 trusted weaker model T、一小笔 trusted human labor H，以及编辑 U 输出的能力。在这个框架中，Trusted Editing 在 code-backdoor task 上 Pareto-dominates Untrusted Monitoring。该框架继承了 security engineering 的 worst-case assumption：safety 是在 red-team pressure 下测量的概率，也就是 adversarially chosen U 无法造成具体 unacceptable outcome 的概率；red team 代表 U。

**类型：** 学习
**语言：** Python（stdlib，code-backdoor task 上的 safety-vs-usefulness frontier）
**前置要求：** 阶段 18 · 07（sleeper agents），阶段 18 · 08（scheming），阶段 18 · 09（alignment faking）
**时间：** ~75 分钟

## 学习目标

- 定义 AI Control threat model：untrusted strong model U、trusted weaker model T、trusted human budget H。
- 描述 Trusted Editing protocol，以及为什么它在 ICML 2024 code-backdoor task 上 Pareto-dominates Untrusted Monitoring。
- 解释 alignment（U 可信）与 control（U 不可信但我们仍然安全）的区别。
- 说明 control 的三个限制：人类不能 scale、trusted weak model 最终会落后、control 不会产生 trustworthy strong model。

## 问题

第 7-9 课确立了 frontier model 可能 deceptively aligned、scheming 或 faking alignment。Safety 问题重新框定为：我们能否从一个不信任的模型中提取有用工作，同时不给它造成 unacceptable outcome 的机会？这就是 control problem，它像经典 security 一样构造：假设 worst-case adversary，设计 protocol，在 red-team pressure 下测量 adversary 的 success rate。

## 概念

### Threat model

Control 假设：
- U：untrusted strong model。有能力。可能 adversarial（evaluation 会把它当作 adversarial 来对待）。
- T：trusted weaker model。能力较弱，但已知不是 adversarial。
- H：trusted human labor。小预算；不能 scale。

系统必须使用 U 产生有用输出。Adversary 试图造成具体 unacceptable outcome（例如 shipped code 中的 backdoor），同时保持一定比例的合法有用工作，以通过 usefulness threshold。

Control 放弃信任 U。它问的是在 U 不可信的情况下，什么 protocol 能维持 safety。

### ICML 2024 backdoor task

Prompt 是竞赛级 coding problem。U 产生 candidate solution。一个隐藏 oracle 标记 solution 是否包含 backdoor（某个特定输入会让代码产生错误输出）。Red team 控制 U，并且允许看到所有 protocol internals。Protocol 必须在保持 backdoor-shipping rate 低于阈值的同时，最大化 solved-problem rate。

### 比较的四种 protocol

- **Baseline：** 直接 ship U 的 output。Usefulness 最高。Safety 最低。
- **Upfront auditing：** 对所有 output 使用 H。Safety 最高。预算很快耗尽。
- **Untrusted Monitoring：** 用第二个 U instance 给第一个 U instance 的 output 打分；H audit 被 flag 的 output。风险：两个 U instance 可能 collude。
- **Trusted Editing：** T 在每次 query 上编辑 U 的 output。如果 T 不 confident，则 route to H。

Greenblatt et al. 发现 Trusted Editing Pareto-dominates Untrusted Monitoring：在每个 safety level 上都有更高 usefulness。

### 为什么 Trusted Editing 会赢

Untrusted Monitoring 要求 U 的第二个 instance 不能与第一个协调。Collusion 可能通过 steganographic cue 或共享 training prior 发生。Protocol 必须把预算投入 collusion-resistance（paraphrasing、obfuscation），这会降低 monitor signal。Trusted Editing 完全避开 collusion，因为 T 是 trusted 的；按假设，T 提议的任何 edit 都不是 adversarial。代价是 T 能力较弱，所以可能 over-reject 或漏掉 subtle backdoor。

### 四个 protocol 轴

1. U 扮演什么角色：只做 producer、monitor，还是两者兼任。
2. T 扮演什么角色：editor、monitor，还是 resampler。
3. H 如何分配：uniform audit、random audit，还是 flag-directed audit。
4. Protocol 在 collusion-resistance 上花费多少 compute。

### Alignment vs control

- Alignment：“U 是否 safe to trust？”第 2、4、5、6、7、8、9 课都与此有关。
- Control：“我们能否在不信任 U 的情况下安全使用 U？”第 10 课。

Alignment 是模型的性质。Control 是 deployment protocol 的性质。Control 不产生 aligned model；它产生对可能 unaligned model 的安全使用。两个 agenda 都需要。

### 三个根本限制

- 人类不能 scale。H budget 受愿意 audit 困难 AI output 的人类数量限制。
- Trusted weak model 最终会落后。当 T 远弱于 U 时，T 的 editing signal 会变成噪声。
- Control 不是 alignment。能在 ASL-3（第 18 课）工作的 protocol 可能无法在 ASL-4 工作；如果 U 能操纵 T，或推理 H 的 audit distribution，control 就会失败。

### 2025 年进展

UK AISI 与 Redwood 合作研究 Control Safety Cases（metr.org / redwoodresearch.org 2025）：control safety case 论证在关于 U 的 worst-case assumption 下，protocol 能约束 unacceptable outcome 的概率。Safety case（第 18 课）继承 control frame。

### 它在阶段 18 中的位置

第 6-9 课描述威胁（deception 可能存在）。第 10 课在该威胁为真的假设下描述防御范式。第 11 课描述互补的进攻范式（通过 scalable oversight 让 U 变得 trustworthy）。

## 使用它

`code/main.py` 在玩具 backdoor task 上模拟四种 protocol。U 产生 output；一部分带 backdoor（red-team policy）。T approve 或 edit。H 在预算内被调用。你可以 sweep budget 和 collusion parameter，复现 Greenblatt et al. 2024 图 2 中 Pareto frontier 的形状（不是数值）。

## 交付它

本课会生成 `outputs/skill-control-protocol-audit.md`。给定一份 deployment protocol description，它会识别 U / T / H 角色，说出 collusion vector，并估计 protocol 在 red-team pressure 下的 usefulness-safety frontier 位置。

## 练习

1. 运行 `code/main.py`。复现 Trusted Editing 与 Untrusted Monitoring 的 usefulness-safety frontier。识别 Untrusted Monitoring 的 collusion assumption 在哪个 budget regime 中变得关键。

2. 添加一个新 protocol：“T 用 low temperature resample，然后 U 编辑 T 的 output。”测量它在 frontier 上的位置。为什么它比 Trusted Editing 更灾难性地失败？

3. 2024 论文假设 U 可以看到 protocol。设计一个对 U 隐藏哪些 output 正在被 audit 的 protocol。这个 protocol 对 deployment infrastructure 有什么要求？

4. 阅读 Greenblatt et al. 2024 第 7 节（collusion）。提出一种 collusion-resistance measure，并解释其预期 usefulness cost。

5. 为一个使用 U = GPT-4-class、T = GPT-3.5-class、H = audit 1% output 的 deployment 阐明 control safety case。指出该 case 依赖的最强单一假设。

## 关键词汇

| 术语 | 人们常说 | 实际含义 |
|------|-----------------|------------------------|
| AI Control | “subversion 下的 safety” | 在 worst-case U 下约束 unacceptable outcome 的 protocol design |
| Untrusted model (U) | “strong model” | 我们想从中提取工作、但不信任的 capable model |
| Trusted model (T) | “weak model” | 一个较弱且已知不 adversarial 的模型，用作 editor/monitor |
| Trusted human (H) | “audit budget” | 有界 expert labor；昂贵；ground-truth signal |
| Trusted Editing | “T edits U” | T 作为 U output gate 的 protocol；ambiguous output 调用 H |
| Untrusted Monitoring | “U monitors U” | 第二个 U instance 给 output 打分的 protocol；只有额外投资后才 collusion-resistant |
| Control safety case | “formal argument” | 书面论证：一个 protocol 在 worst-case U 下约束 adversary success |

## 延伸阅读

- [Greenblatt, Shlegeris, Sachan, Roger — AI Control (arXiv:2312.06942)](https://arxiv.org/abs/2312.06942) — canonical ICML 2024 paper
- [Redwood Research — Recent AI Control papers index](https://redwoodresearch.substack.com/) — 2024-2025 follow-up
- [UK AISI + METR — Control safety cases](https://metr.org/blog/2025-03-26-common-elements-of-frontier-ai-safety-policies/) — frontier lab operationalization
- [Hubinger et al. — Sleeper Agents (Lesson 7, arXiv:2401.05566)](https://arxiv.org/abs/2401.05566) — control 所假设的 threat model
