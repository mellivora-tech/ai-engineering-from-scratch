# Scalable Oversight 与 Weak-to-Strong Generalization

> Burns et al.（OpenAI Superalignment，“Weak-to-Strong Generalization”，2023）提出了 superalignment problem 的一个 proxy：用较弱模型产生的标签来 fine-tune 强模型。如果强模型能从不完美的 weak supervision 中正确泛化，那么当前 human-scale alignment 方法也许能扩展到 superhuman system。Scalable oversight 与 W2SG 是互补的。Scalable oversight（debate、recursive reward modeling、task decomposition）提高 overseer 的有效能力，使其能跟上被监督模型。W2SG 确保强模型能从 overseer 提供的任何不完美 supervision 中正确泛化。Debate Helps W2SG（arXiv:2501.13124，2025 年 1 月）把两者结合起来。

**类型：** 学习
**语言：** Python（stdlib，W2SG gap 模拟器）
**前置要求：** 阶段 18 · 01（instruction-following），阶段 18 · 10（AI Control），阶段 09（RL foundations）
**时间：** ~60 分钟

## 学习目标

- 定义 scalable oversight 和 weak-to-strong generalization，并解释它们如何互补。
- 描述 Burns et al. 2023 的实验设置：用 GPT-2 的标签 fine-tune GPT-4。
- 解释 performance gap recovered（PGR）metric 以及它测量什么。
- 说出三种主要 scalable-oversight mechanism（debate、recursive reward modeling、task decomposition），并说明每种的一个优势。

## 问题

阶段 18 到目前为止的每种 alignment technique 都假设 overseer 能评估模型行为。当模型是 superhuman 时，overseer 就是薄弱环节。Superalignment 问题是：一个较弱 supervisor 能否可靠地产生一个更强且 aligned 的模型？

Burns et al. 把它缩减为一个 operationalized empirical setup：用 weak 监督 strong，测量 strong model 的 capability 有多少在 weak supervision 下存活。这不是 superalignment 的解决方案，而是一个让我们测量进展的 proxy。

## 概念

### W2SG：Burns et al. 设置

- Weak model：GPT-2 class。
- Strong model：GPT-4 class。
- Target：任务上 strong GPT-4 ceiling。

流程：
1. 获取 weak model 在任务上的 zero-shot prediction。
2. 在 weak-labeled data 上 fine-tune strong model。
3. 测量 strong fine-tuned model 的 accuracy。

Baseline：weak model 自身 accuracy、strong model 在 gold label 监督下的 ceiling。

Gap metric：Performance Gap Recovered（PGR）= (fine-tuned - weak) / (ceiling - weak)。PGR 为 1.0 意味着 weak supervision 完全关闭 gap；PGR 为 0 意味着 weak supervision 没有帮助。

### Burns et al. 经验发现

在 NLP、chess-puzzle 和 reward-modeling 任务上，PGR 一直为正：strong model 能泛化越过 weak supervisor 的错误。Task-specific PGR 从 ~20% 到 ~80% 不等。Strong model 从不完美标签中“理解”了目标任务，并用自己的 pre-trained prior 填补空缺。

Burns et al. 指出的限制：
- 这里的 weak-to-strong gap 是 capability，不是 alignment。Weak model 是 aligned 的；真正 superhuman model 也许由人类 alignment，但 capability gap 大得多。
- Strong model 的泛化可能更多来自 task prior，而不是恢复 ground-truth intent。
- 引出 strong model 的 latent knowledge 才是真正问题；PGR 测量的是一个具体 operationalization。

### Scalable oversight：三种机制

- **Debate（Irving et al. 2018）。** 两个 U instance 为相反立场辩论；weak judge 做决定。假设是：找到一个有说服力的真答案，比找到一个有说服力的假答案更容易。近期 2024-2025 工作（Khan et al., Michael et al.）显示 debate 有时有帮助、有时没有，是否有帮助取决于 task structure。
- **Recursive Reward Modeling（Leike et al. 2018）。** U 帮助 human 训练 U+1 的 reward model。Overseer 的有效能力随模型增长。
- **Task Decomposition（Christiano, Shlegeris, Amodei 2018）。** 递归地把 hard task 分解为 human 可以检查的 sub-task。假设任务可 decomposable。

每种机制都对任务结构或中间组件的 alignment 有某种假设。

### 为什么 scalable oversight 与 W2SG 互补

Scalable oversight 提高 overseer 的有效信号质量。
W2SG 从 overseer 能提供的任何不完美信号中关闭 gap。

Lang et al. — Debate Helps Weak-to-Strong Generalization（arXiv:2501.13124）把两者结合起来：debate protocol 提供更好的 weak label，然后 strong model 在这些标签上训练。报告称 NLP 任务上的 PGR 提高。

### 组织层面的戏剧

OpenAI 的 Superalignment team 在 Jan Leike 离开加入 Anthropic 后，于 2024 年 5 月解散。这个 agenda（scalable oversight、W2SG、automated alignment research）在 Anthropic 和学术实验室继续推进，包括 MATS（第 28 课）、Redwood（第 10 课）、Apollo（第 8 课）、METR（第 28 课）。组织结构变了；研究问题没有变。

### 它在阶段 18 中的位置

第 6-10 课描述威胁，以及在假设 U 不可信时的防御范式。第 11 课是进攻范式：让 overseer 足够强，可以验证 U 的 alignment。第 12-16 课随后转向 adversarial evaluation 的实用工具。

## 使用它

`code/main.py` 在合成任务上模拟 W2SG fine-tune。Weak labeler 有 70% accuracy 且带 structured errors；strong model 在 gold label 上有 95% ceiling。你在 weak label 上 fine-tune strong model，测量 PGR，并与 strong-on-gold 和 weak-alone 比较。

## 交付它

本课会生成 `outputs/skill-w2sg-pgr.md`。给定一个 oversight setup description，它会识别 weak supervisor、strong model、supervision quality，并计算（或请求）PGR。它会标记该 claim 是“weak can supervise strong”，还是“weak + oversight mechanism can supervise strong”。

## 练习

1. 运行 `code/main.py`。报告 weak_accuracy = 0.60、0.70、0.80 时的 PGR。解释 PGR curve 的形状。

2. 修改 weak labeler，让它有 structured error（例如总是在某个特定 input class 上错误）。PGR 是增加、减少还是保持不变？解释原因。

3. 阅读 Burns et al. 2023 第 4.3 节（NLP tasks）。复现 “confidence auxiliary loss” 直觉：当 strong model 比 weak label 更 confident 时，谁赢？

4. 设计一个把 debate 和 task decomposition 结合起来，用于 software-engineering task 的 scalable-oversight protocol。说出每个组件的一个 failure mode，并解释组合如何处理或未能处理它们。

5. 阐明什么会证伪 “weak-to-strong generalization is a viable path to superalignment” 这个 claim。具体说明你需要看到什么经验 signature。

## 关键词汇

| 术语 | 人们常说 | 实际含义 |
|------|-----------------|------------------------|
| Scalable oversight | “让 overseer 更强” | 提高 overseer 评估更 capable model 能力的机制 |
| W2SG | “weak supervises strong” | 在 weak label 上 fine-tune strong model，并测量恢复的 capability |
| PGR | “performance gap recovered” | (fine-tuned - weak) / (ceiling - weak)；1.0 = 完全关闭，0 = 无帮助 |
| Debate | “两个 U instance 争论” | scalable oversight 机制，由 weak judge 在两个 U defender 之间选择 |
| RRM | “recursive reward modeling” | U 帮助训练 U+1 的 reward model；overseer capability 跟踪 U |
| Task decomposition | “human 检查的 sub-task” | 把 hard task 递归拆成 human 可验证的 sub-task |
| Superalignment | “aligning superhuman AI” | 关注让人类无法直接评估的模型 aligned 的研究 agenda |

## 延伸阅读

- [Burns et al. — Weak-to-Strong Generalization (OpenAI 2023)](https://openai.com/index/weak-to-strong-generalization/) — W2SG 论文
- [Irving, Christiano, Amodei — AI safety via debate (arXiv:1805.00899)](https://arxiv.org/abs/1805.00899) — debate mechanism
- [Leike et al. — Scalable agent alignment via reward modeling (arXiv:1811.07871)](https://arxiv.org/abs/1811.07871) — recursive reward modeling
- [Khan et al. — Debating with More Persuasive LLMs Leads to More Truthful Answers (arXiv:2402.06782)](https://arxiv.org/abs/2402.06782) — 2024 年关于 stronger debater 的 debate 实证研究
- [Lang et al. — Debate Helps Weak-to-Strong Generalization (arXiv:2501.13124)](https://arxiv.org/abs/2501.13124) — 2025 年 debate + W2SG 组合
