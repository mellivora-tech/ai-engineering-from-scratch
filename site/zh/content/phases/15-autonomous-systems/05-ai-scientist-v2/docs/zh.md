# AI Scientist v2 — Workshop 级 Autonomous Research

> Sakana 的 AI Scientist v2（Yamada et al., arXiv:2504.08066）运行完整 research loop：hypothesis、code、experiments、figures、writeup、submission。它是第一个让生成论文通过 ICLR 2025 workshop peer review 的系统。独立评估（Beel et al.）发现 42% 的实验因 coding errors 失败，literature review 也经常把已有概念误标为新颖。Sakana 自己的 docs 警告该 codebase 会执行 LLM 写的代码，并建议 Docker isolation。这两半图景合起来才是重点。

**类型：** 学习
**语言：** Python（stdlib，research-loop state-machine toy）
**前置要求：** 阶段 15 · 03（AlphaEvolve），阶段 15 · 04（DGM）
**时间：** ~60 分钟

## 问题

研究是开放式任务。不同于 AlphaEvolve 的 algorithmic search 或 DGM 的 benchmark-bounded self-modification，研究结果没有 machine-checkable correctness criterion。论文由 reviewers 判断，而不是 unit tests。这使得 loop 更难闭合——但一旦闭合也更有价值，因为研究正是 compounding progress 所在。

AI Scientist v1（Sakana, 2024）通过从人类写好的 templates 开始来闭合 loop。LLM 在固定 scaffolding 内填入 experiments。AI Scientist v2（Yamada et al., 2025）通过使用 agentic tree search 和 vision-language model critique loop，移除了 template requirement。系统生成 ideas、实现 experiments、产出 figures、写 paper，并根据 reviewer feedback 迭代。

Peer review 结论：一篇 v2 生成的论文被 ICLR 2025 workshop 接收（并有 disclosure）。独立评估结论：系统远不可靠。两者都是真的。

## 概念

### 架构

1. **Idea generation。** LLM 基于 topic 和 prior literature 提出 research ideas。v1 使用 templates；v2 在 hypotheses 空间上使用 agentic search。
2. **Novelty check。** literature retrieval step 检查 idea 是否已经发表。这正是 Beel et al. 评估发现 mislabeling 的步骤——已建立方法经常被归类为 novel。
3. **Experiment plan。** agent 起草 experimental protocol 并写代码。
4. **Execution。** 代码在 sandbox 中运行。失败会反馈到 retry loop。Beel et al. 的测量中，42% 的 experiments 在此阶段因 coding errors 失败。
5. **Figure generation。** vision-language model 阅读生成的 figures，并为清晰度重写它们。这是 v2 的关键技术新增点。
6. **Writeup。** LLM 起草 paper，并与 internal reviewer 迭代。
7. **Optional: submission。** paper 提交到 venue。

### workshop acceptance 结果意味着什么

一篇 v2 生成的论文通过了 ICLR 2025 workshop 的 peer review。作者向 program committee 披露了论文来源。这个接收是一个 data point；它不是声称系统“会做研究”的许可证。

重要背景：workshop papers 的门槛低于 main-conference papers。Peer review 有噪声；任何给定一天都有小比例 submissions 会被接收。一次成功是 proof of concept，不是可靠性声明。Nature 2026 论文记录了 end-to-end loop，且它本身由人类研究者共同署名；这不是“系统写了一篇 Nature paper”。

### 独立评估发现了什么

Beel et al.（arXiv:2502.14297）做了外部评估。主要发现：

- **Experiment failures。** 42% 的 experiments 因 coding errors（错误 imports、shape mismatches、undefined variables）失败。retry loop 捕捉了一部分，但不是全部。
- **Novelty mislabeling。** literature-retrieval step 经常把已有概念标记为 novel。这是研究等价物中的 hallucination。
- **Presentation-quality gap。** vision-language figure critique 产出了 publication-grade visuals，掩盖了底层实验弱点。

最后一项对本阶段最重要。一个能产出 convincing outputs 但没有做 convincing research 的系统，比一个明显失败的系统更危险，而不是更安全。Evaluation 必须抵达底层 claims，而不能止步于 figure。

### sandbox-escape 担忧

Sakana 自己的 repository README 警告：

> Due to the nature of this software, which executes LLM-generated code, we cannot guarantee safety. There are risks of dangerous packages, uncontrolled web access, and spawning of unintended processes. Use at your own risk and consider Docker isolation.

这就是未验证领域中 autonomy 的 operational shape。LLM 写代码；代码运行；代码可以做任何该进程被允许做的事情。如果没有对 filesystem、network 和 process actions 做硬限制的 sandbox，任何 self-directed research agent 都可能 exfiltrate data、烧掉 compute，或者重写自己。

AlphaEvolve 的 sandbox story 更容易，因为它的 evaluator 很紧。AI Scientist v2 的 loop 运行 open-ended code，并追求 open-ended goals。因此它需要更强隔离（Docker 是最低要求；更推荐 seccomp / gVisor），并且在任何 submission 离开系统之前进行 manual review。

### v2 在 frontier stack 中的位置

| System | Target | Output kind | Evaluator | Known failure |
|---|---|---|---|---|
| AlphaEvolve | algorithms | code | unit + benchmark | bounded by evaluator rigor |
| DGM | agent scaffolding | code | SWE-bench | reward hacking |
| AI Scientist v2 | research papers | text + code + figures | peer review (weak) | experiment failures, mislabeling, polish masking weakness |

在三者中，v2 的 automatic evaluator 最弱，output surface 最宽，并且到 public artifacts 的路径最短。operational controls（sandbox、review、disclosure）承担了大部分安全工作。

## 使用它

`code/main.py` 把 v2 loop 模拟成 state machine：idea → novelty check → experiment → figure → writeup → review → accept-or-iterate。每个 state 都有一个可配置 failure probability，来自 Beel et al. 的发现。运行 N 个 loops 并统计：

- 有多少 ideas 走到 submission。
- 有多少 submissions 会存在 polished paper 掩盖的 critical experimental flaw。
- retry budgets 如何在 quality 与 yield 之间权衡。

## 交付它

`outputs/skill-ai-scientist-sandbox-review.md` 是一个 two-gate review checklist，用于任何 research-loop agent 产物离开 sandbox 之前。

## 练习

1. 使用默认参数运行 `code/main.py`。有多少比例的 loop runs 产出“clean”paper？有多少比例产出的 paper 存在被 figure critique 掩盖的 experiment-failure flaw？

2. 默认值已经使用 Beel et al. 的 42% / 25%。分别用 `--experiment-failure 0.20 --novelty-mislabel 0.10` 和 `--experiment-failure 0.60 --novelty-mislabel 0.40` 重新运行。polished-but-flawed share 在两次运行之间如何变化？

3. 阅读 Sakana 的 AI Scientist v2 repo README 关于 sandbox requirements 的部分。说出你会为 multi-day autonomous run 添加的两个额外限制（Docker 之外）。

4. 阅读 Beel et al. 第 4 节关于 presentation-quality gap 的内容。设计一个额外 evaluator，用来捕捉看起来 polished 但实验有问题的 papers。

5. 为 research-agent outputs 提出一个 human-review protocol，使其比“每篇 paper 都由 PhD 阅读”更可扩展。找出瓶颈并围绕它设计。

## 关键术语

| Term | What people say | What it actually means |
|---|---|---|
| AI Scientist v1 | “Sakana 的 templated research agent” | 将 experiments 填入固定 scaffold |
| AI Scientist v2 | “Template-free research agent” | 带 VLM figure critique 的 agentic tree search |
| Agentic tree search | “Branching research agent” | 并行扩展多个 experiment plans；由 internal critic 剪枝 |
| Vision-language critique | “VLM polish on figures” | multimodal model 阅读 figures 并为清晰度重写 |
| Literature retrieval | “Novelty check” | 搜索 prior work 来确认 idea novelty——已有 mislabel 记录 |
| Polish masking | “漂亮 paper，坏 research” | 展示质量超过实验质量；掩盖弱点 |
| Sandbox escape | “LLM code breaks out” | agent-executed code 做了 loop designer 不希望它做的事 |

## 延伸阅读

- [Yamada et al. (2025). The AI Scientist-v2](https://arxiv.org/abs/2504.08066) — 论文。
- [Sakana blog on the Nature 2026 publication](https://sakana.ai/ai-scientist-nature/) — 带 peer-review context 的 vendor summary。
- [Beel et al. (2025). Independent evaluation of The AI Scientist](https://arxiv.org/abs/2502.14297) — external evaluation numbers。
- [Sakana AI Scientist v1 paper](https://arxiv.org/abs/2408.06292) — templated predecessor。
- [Anthropic — Measuring AI agent autonomy](https://www.anthropic.com/research/measuring-agent-autonomy) — open-ended research agents 的更宽 framing。
