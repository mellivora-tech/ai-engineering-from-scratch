# Anthropic 的 Workflow Patterns：Simple Over Complex

> Schluntz 和 Zhang（Anthropic, Dec 2024）区分 workflows（预定义路径）和 agents（dynamic tool-use）。五种 workflow patterns 覆盖了大多数场景。从 direct API calls 开始。只有当步骤无法预测时，才添加 agents。

**类型：** 学习 + 构建
**语言：** Python (stdlib)
**前置要求：** 阶段 14 · 01（Agent Loop）
**时间：** ~60 分钟

## 学习目标

- 说出 Anthropic 的五种 workflow patterns：prompt chaining、routing、parallelization、orchestrator-workers、evaluator-optimizer。
- 解释 agent-vs-workflow 区别，以及各自的工程成本。
- 识别什么时候该选 workflow 而不是 agent，反之亦然。
- 用 stdlib 和 scripted LLM 实现全部五种模式。

## 问题

团队经常为了本该用单个 function call 解决的问题引入 multi-agent frameworks。成本是真实的：frameworks 会添加层级，遮蔽 prompts、隐藏 control flow，并诱发过早复杂化。Schluntz 和 Zhang 在 2024 年 12 月的文章是最常被引用的行业反驳：从简单开始，只有当复杂度赚回成本时才添加它。

## 概念

### Workflows vs agents

- **Workflow。** LLMs 和 tools 通过预定义代码路径编排。Engineers 拥有 graph。
- **Agent。** LLMs 动态指导自己的 tools，并自行采取步骤。Model 拥有 graph。

两者都有位置。Workflows 更便宜、更快、更易调试。Agents 解锁 open-ended problems，但让 failure modes 更难推理。

### Augmented LLM

五种模式的基础：一个 LLM 接入三种能力：search（retrieval）、tools（actions）、memory（persistence）。任何 API call 都可以使用这些能力。

### 五种模式

1. **Prompt chaining。** Call 1 的输出是 call 2 的输入。适用于任务有清晰线性分解的情况。步骤之间可以加 optional programmatic gates。

2. **Routing。** 一个 classifier LLM 选择调用哪个 downstream LLM 或 tool。适用于分类上不同的输入需要不同处理（tier-1 support vs refund vs bug vs sales）。

3. **Parallelization。** 并发运行 N 个 LLM calls，聚合结果。两种形状：sectioning（不同 chunks）和 voting（同一个 prompt，N 次运行，majority/synthesis）。

4. **Orchestrator-workers。** 一个 orchestrator LLM 动态决定运行哪些 workers（也是 LLMs），并综合它们的输出。类似 agent loops，但 orchestrator 不会无限 loop。

5. **Evaluator-optimizer。** 一个 LLM 提出答案，另一个 LLM 评估它。迭代直到 evaluator 通过。这是泛化后的 Self-Refine（第 05 课）。

### Workflows 胜过 agents 的地方

- **Predictable tasks。** 如果你能枚举步骤，就应该枚举。
- **Cost-bound tasks。** Workflows 有有界 step count；agents 可能螺旋上升。
- **Compliance-bound tasks。** Auditors 想读 graph，而不是从 trajectories 里推断它。

### Agents 胜过 workflows 的地方

- **Open-ended research。** 下一步取决于上一步返回了什么。
- **Variable-length tasks。** 持续数分钟到数小时、step count 未知的工作。
- **Novel domains。** 当你还不知道正确 workflow 时：先探索，再固化。

### Context-engineering companion

“Effective context engineering for AI agents”（Anthropic 2025）把相邻学科形式化：200k window 是预算，不是容器。包含什么、什么时候 compact、什么时候让 context grow。它会在阶段 14 关于 context compression 的课程中详细覆盖（本课程重编号前的阶段 14 早期第 06 课）。

## 构建它

`code/main.py` 针对 `ScriptedLLM` 实现全部五种 workflow patterns：

- `prompt_chain(input, steps)`：sequential。
- `route(input, classifier, handlers)`：classification + dispatch。
- `parallel_vote(prompt, n, aggregator)`：N runs，aggregate。
- `orchestrator_workers(task, workers)`：orchestrator 选择 workers。
- `evaluator_optimizer(task, proposer, evaluator, max_iter)`：loop until pass。

运行它：

```
python3 code/main.py
```

每种模式都会打印自己的 trace。每个 pattern 约 10-15 行代码；一个 framework 的成本则以千行计。

## 使用它

- 大多数任务使用 direct API calls。
- 只有当 pattern 真正需要 durable state（LangGraph）、actor-model concurrency（AutoGen v0.4）或 role templating（CrewAI）时才用 framework。
- 当你想要 Claude Code harness shape，又不想重建它时，选择 Claude Agent SDK。

## 发布它

`outputs/skill-workflow-picker.md` 会为给定 task description 选择正确模式，包含 decision rationale，以及当 workflows 不够用时重构到 agent 的路径。

## 练习

1. 用 confidence threshold 实现 routing。低于 threshold -> escalate to human。对 tier-1 support use case 来说 threshold 会落在哪里？
2. 给 `parallel_vote` 添加 timeout。当一个 call hang 住时会怎样？缺少 votes 时如何 aggregate？
3. 把 `evaluator_optimizer` 变成 bandit：跨 iterations 保留 top-2 outputs，避免一个 late good result 被一个 late bad one 覆盖。
4. 组合 prompt chaining 和 routing：router 从三条 chains 中选择一条。测量 token cost，并和一个 single big-prompt alternative 对比。
5. 选择你的一个 production feature。画出 workflow graph。数步骤。这里 agent 真的更好吗？

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|----------------|------------------------|
| Workflow | “Predefined flow” | Engineer-owned LLM 和 tool calls graph |
| Agent | “Autonomous AI” | Model-owned graph；dynamic tool direction |
| Augmented LLM | “LLM with tools” | LLM + search + tools + memory；原子单元 |
| Prompt chaining | “Sequential calls” | Call N 的输出是 call N+1 的输入 |
| Routing | “Classifier dispatch” | 选择哪个 chain/model 处理输入 |
| Parallelization | “Fan out” | N 个并发 calls；按 sectioning 或 voting 聚合 |
| Orchestrator-workers | “Dispatcher agent” | Orchestrator LLM 动态选择 specialist LLMs |
| Evaluator-optimizer | “Proposer + judge” | 迭代直到 evaluator 通过；Self-Refine 的泛化 |

## 延伸阅读

- [Anthropic, Building Effective Agents (Dec 2024)](https://www.anthropic.com/research/building-effective-agents)：五种 workflow patterns
- [Anthropic, Effective context engineering for AI agents](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)：配套学科
- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview)：什么时候 stateful graphs 值得其成本
- [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/)：产品化的 orchestrator-workers pattern
