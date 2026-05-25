# Reflexion：Verbal Reinforcement Learning

> Gradient-based RL 需要上千次试验和一个 GPU 集群来修复一种 failure mode。Reflexion（Shinn et al., NeurIPS 2023）用自然语言做到这件事：每次失败试验后，agent 写一段 reflection，把它存入 episodic memory，并让下一次试验以这段 memory 为条件。这就是 Letta 的 sleep-time compute、Claude Code 的 CLAUDE.md learnings，以及 pro-workflow 的 learn-rule 背后的模式。

**类型：** 构建
**语言：** Python (stdlib)
**前置要求：** 阶段 14 · 01（Agent Loop），阶段 14 · 02（ReWOO）
**时间：** ~60 分钟

## 学习目标

- 说出 Reflexion 的三个组件（Actor、Evaluator、Self-Reflector），以及 episodic memory 的作用。
- 用 stdlib 实现 Reflexion loop，包含 binary evaluator、reflection buffer 和 fresh re-attempts。
- 为给定任务选择 scalar、heuristic 和 self-evaluated feedback source。
- 解释为什么 verbal reinforcement 能捕捉到 gradient-based RL 需要上千次试验才能修复的错误。

## 问题

一个 agent 任务失败了。标准 RL 会继续跑几千次试验，计算 gradients，更新 weights。昂贵、缓慢，而且大多数生产 agents 没有预算为每次失败训练模型。

Reflexion（Shinn et al., arXiv:2303.11366）问了另一个问题：如果 agent 只是思考自己为什么失败，然后把这段思考放进 prompt 里重试，会怎样？没有 weight updates。没有 gradient。只有 trial 之间保存的自然语言。

结果是：在 ALFWorld 上它击败了 ReAct 和其他非 fine-tuned baselines。在 HotpotQA 上它比 ReAct 有提升。在代码生成（HumanEval/MBPP）上，它达到了当时的 state of the art。全程没有一步 gradient。

## 概念

### 三个组件

```
Actor         : generates a trajectory (ReAct-style loop)
Evaluator     : scores the trajectory — binary, heuristic, or self-eval
Self-Reflector: writes a natural-language reflection on the failure
```

再加一个数据结构：

```
Episodic memory: list of prior reflections, prepended to the next trial's prompt
```

一次 trial 由 Actor 运行。Evaluator 给它打分。如果分数低，Self-Reflector 生成一段 reflection（“I picked the wrong tool because I misread the question as asking about X when it was asking about Y”）。这段 reflection 进入 episodic memory。下一次 trial 从头开始，但会看到这段 reflection。

### 三种 evaluator 类型

1. **Scalar**：外部 binary signal。ALFWorld 成功或失败。HumanEval 测试通过或失败。最简单、信号最高。
2. **Heuristic**：预定义 failure signatures。“如果 agent 连续两次产生同一个 action，标记为 stuck。”“如果 trajectory 超过 50 步，标记为 inefficient。”
3. **Self-evaluated**：LLM 给自己的 trajectory 打分。当没有 ground truth 时需要它。信号较弱；适合搭配 tool-grounded verification（第 05 课：CRITIC）。

2026 年默认是混合使用：有 scalar 就用 scalar，没有就用 self-eval，heuristics 作为 safety rails。

### 为什么它能泛化

Reflexion 与其说是新算法，不如说是一个被命名的模式。几乎每个生产中的“self-healing” agent 都运行某种变体：

- Letta 的 sleep-time compute（第 08 课）：一个独立 agent 会反思过去对话，并写入 memory blocks。
- Claude Code 的 `CLAUDE.md` / “save memory” 模式：reflections 作为 learnings 被捕获，并前置到未来 sessions。
- pro-workflow 的 `/learn-rule` 命令：corrections 被捕获为显式 rules。
- LangGraph 的 reflection nodes：一个 node 会给 output 打分，如果需要 refine 就 route 过去。

它们都来自同一个洞见：自然语言足够丰富，可以在 runs 之间携带“我从失败中学到了什么”。

### 什么时候有效，什么时候无效

Reflexion 在这些情况下有效：

- 有清晰的 failure signal（test failure、tool error、wrong answer）。
- 任务类别可复现（同类型问题会再次出现）。
- Reflection 有空间改善 trajectory（足够的 action budget）。

Reflexion 在这些情况下无效：

- Agent 第一次就已经成功。
- 失败来自外部（network down、tool broken）：反思“网络挂了”不会帮助未来 runs。
- Reflection 变成迷信：为一次偶发 flaky run 存储一段叙事。

2026 年的坑：memory rot。Reflections 会积累；有些过时或错误；episodic buffer 变大后 rerun 会越来越慢。缓解方式：定期 compaction（第 06 课）、给 reflections 加 TTL，或者使用单独的 sleep-time cleanup agent（Letta）。

## 构建它

`code/main.py` 在一个 toy puzzle 上实现 Reflexion：生成一个长度为 3、和为目标值的 list。Actor 生成候选 lists；Evaluator 检查总和；Self-Reflector 写一行诊断说明错在哪里。Reflection 进入 episodic memory，供下一次 trial 使用。

组件：

- `Actor`：一个 scripted policy，在看到 reflections 后会改进。
- `Evaluator.binary()`：针对 target sum 的 pass/fail。
- `SelfReflector`：生成一行 failure diagnosis。
- `EpisodicMemory`：带 TTL 语义的 bounded list。

运行它：

```
python3 code/main.py
```

Trace 会显示三次 trials。Trial 1 失败，存入一条 reflection；trial 2 看到 reflection 后改进但仍失败；trial 3 成功。和 baseline run（无 reflection）对比，它会一直卡在 trial 1 的答案上。

## 使用它

LangGraph 把 reflection 作为 node pattern 提供。Claude Code 的 `/memory` 命令和 pro-workflow 的 `/learn-rule` 会把 episodic buffer 外部化成 markdown 文件。Letta 的 sleep-time compute 在 downtime 运行 Self-Reflector，让 primary agent 保持 latency-bound。OpenAI Agents SDK 不直接内置 Reflexion；你可以用一个按 score 拒绝 trajectories 的 custom Guardrail，以及一个跨 runs 存活的 memory `Session` 来构建它。

## 发布它

`outputs/skill-reflexion-buffer.md` 会创建并维护一个 episodic buffer，支持 reflection capture、TTL 和 deduplication。给定一个 task class 和一次 failure，它会生成真正能帮助下一次 trial 的 reflection，而不是泛泛地说“be more careful”。

## 练习

1. 从 binary evaluator 切换为返回距离指标的 scalar evaluator（距离目标多远）。它会更快收敛吗？
2. 给 reflections 添加 10 次 trials 的 TTL。超过这个点以后，旧 reflections 是伤害还是帮助？
3. 实现 heuristic evaluator：如果同一个 action 重复，就把 trial 标记为 stuck。这会如何和 Self-Reflector 交互？
4. 用一个忽略 reflections 的 adversarial Actor 运行 Reflexion。让 Actor 注意到它们的最小 reflection prompt engineering 是什么？
5. 阅读 Reflexion 论文关于 AlfWorld 的第 4 节。概念性复现 130% success-rate improvement：相比 vanilla ReAct，关键差异是什么？

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|----------------|------------------------|
| Reflexion | “Self-correction” | Shinn et al. 2023：Actor、Evaluator、Self-Reflector 加 episodic memory |
| Verbal reinforcement | “Learning without gradients” | 前置到下一次 trial prompt 的 natural-language reflection |
| Episodic memory | “Per-task reflections” | 一个任务类别的 prior reflections bounded buffer |
| Scalar evaluator | “Binary success signal” | 来自 ground truth 的 pass/fail 或 numeric score |
| Heuristic evaluator | “Pattern-based detector” | 预定义 failure signatures（例如 stuck-loop、too-many-steps） |
| Self-evaluator | “LLM-as-judge on own trace” | 没有 ground truth 时的低信号 fallback；搭配 tool-grounded verification 使用 |
| Memory rot | “Stale reflections” | Episodic buffer 被过时条目填满；用 compaction/TTL 修复 |
| Sleep-time reflection | “Async self-reflection” | 在 hot path 之外运行 Self-Reflector，让 primary agent 保持快速 |

## 延伸阅读

- [Shinn et al., Reflexion: Language Agents with Verbal Reinforcement Learning (arXiv:2303.11366)](https://arxiv.org/abs/2303.11366)：经典论文
- [Letta, Sleep-time Compute](https://www.letta.com/blog/sleep-time-compute)：生产中的 async reflection
- [Anthropic, Effective context engineering for AI agents](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)：把 episodic buffer 作为 context 管理的一部分
- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview)：reflection node pattern
