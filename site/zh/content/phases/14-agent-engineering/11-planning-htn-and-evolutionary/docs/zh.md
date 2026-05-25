# 使用 HTN 和 Evolutionary Search 做规划

> Symbolic planning 处理 plan 可以被证明正确的场景。Evolutionary code search 处理 fitness function 可以机器检查的场景。ChatHTN（2025）和 AlphaEvolve（2025）展示了它们各自和 LLM 搭配时能解锁什么。

**类型：** 构建
**语言：** Python (stdlib)
**前置要求：** 阶段 14 · 02（ReWOO and Plan-and-Execute）
**时间：** ~75 分钟

## 学习目标

- 解释 Hierarchical Task Networks：tasks、methods、operators、preconditions、effects。
- 描述 ChatHTN 的 hybrid loop：symbolic search 加 LLM fallback decomposition。
- 解释 AlphaEvolve 的 evolutionary loop，以及为什么它只在有 programmatic evaluator 时有效。
- 用 stdlib 实现一个 toy HTN planner 和一个 toy evolutionary search。

## 问题

ReWOO（第 02 课）、Plan-and-Execute 和 ReAct 覆盖了大多数 agent planning。它们不擅长两个场景：

1. **Plans with provable correctness。** Scheduling、flight pathing、compliance workflows：plan 必须按构造保证 sound。偶尔 hallucinate 一个 step 的流畅 LLM plan 不能接受。
2. **Optimizations with a machine-checkable fitness function。** Matrix multiplication、scheduling heuristics、compiler passes：目标不是“一个正确 plan”，而是“最好的 plan”。

HTN planning 和 AlphaEvolve 解决的是两个不同问题。两者都把 LLM 当作放大器，而不是替代品。

## 概念

### Hierarchical Task Networks

一个 HTN 包含：

- **Tasks**：compound（需要分解）和 primitive（可直接执行）。
- **Methods**：把 compound task 分解为 subtasks 的方式，带 preconditions。
- **Operators**：带 preconditions 和 effects 的 primitive actions。
- **State**：一组 facts。

Planning：给定 goal task 和 initial state，找出一组 primitive operators 的分解，使得它们的 preconditions 按顺序满足。

HTN 比 LLM 更早出现，并且仍然是 provably-correct plans 的参考方法。

### ChatHTN（Gopalakrishnan et al., 2025）

ChatHTN（arXiv:2505.11814）把 symbolic HTN 和 LLM queries 交错：

1. 尝试用现有 methods 分解当前 compound task。
2. 如果没有 method 适用，询问 LLM：“how would you decompose `task` in state `s`?”
3. 把 LLM response 翻译为 candidate subtasks。
4. 根据 operator schema 校验；拒绝 invalid decompositions。
5. 递归。

论文的核心主张：所有生成的 plan 都是 provably sound，因为 LLM suggestions 只作为 candidate decompositions 进入，永远不直接编辑 plan。Symbolic layer 拥有 correctness；LLM 扩展 method library。

Online method learning（OpenReview `gwYEDY9j2x`, 2025 follow-up）添加了一个 learner，通过 regression 泛化 LLM-produced decompositions，最多减少 75% LLM query frequency。

### AlphaEvolve（Novikov et al., 2025）

AlphaEvolve（arXiv:2506.13131, DeepMind, June 2025）是另一种东西：由 Gemini 2.0 Flash/Pro ensemble 编排的 evolutionary code search。

Loop：

1. 从 seed program + programmatic evaluator（返回 fitness score）开始。
2. LLM ensemble 提出 mutations。
3. 用 evaluator 运行 mutations。
4. 保留最好的，再继续 mutate。

公开成果：

- 56 年来首次改进 4x4 complex matrix multiplication 的 Strassen 结果（48 scalar multiplications）。
- 通过 Borg scheduling heuristic 找回 0.7% Google compute。
- 在 frontier workload 上实现 32% FlashAttention speedup。

硬约束：fitness function 必须 machine-checkable。对 prose answers 做 evolutionary search 不会收敛。

### 什么时候用哪一个

| Problem class | Use | Why |
|---------------|-----|-----|
| Scheduling with hard constraints | HTN + ChatHTN | Provable soundness |
| Compiler optimization | AlphaEvolve | Machine-checkable fitness |
| Multi-step task execution | ReAct / ReWOO | LLM in the loop, no formal guarantees |
| Code improvement with tests | AlphaEvolve | Tests are the evaluator |
| Policy-bound automation | HTN | Preconditions encode policy |

### 这个模式会在哪里出错

- **HTN without operators。** 没有 precondition/effect schemas，soundness claim 就会崩塌。ChatHTN 的“LLM suggests decomposition”要求 schema 拒绝 invalid moves。
- **AlphaEvolve without a real evaluator。** “问 LLM 这段代码是否更好”不是 fitness function。Evaluator 必须 deterministic 且 fast。
- **Over-engineering。** 大多数 agent tasks 不需要两者。先使用 ReAct 或 ReWOO。

## 构建它

`code/main.py` 实现了两个 toys：

- 一个 stdlib HTN planner，包含 operators、methods、preconditions、effects，以及在没有 method 匹配 compound task 时触发的 `LLMFallback`。“LLM” 是一个 scripted decomposer，因此 planner 可以离线运行。
- 一个 stdlib evolutionary search over arithmetic programs：生成 expressions，让输出在 test set 上最小化 `|f(x) - target|`。Evaluator 是 deterministic。

运行它：

```
python3 code/main.py
```

Trace 会显示 HTN planner 分解一个 compound task（中途带一次 LLM fallback），以及 evolutionary loop 收敛到 target expression。

## 使用它

- **HTN planners**：`pyhop`、`SHOP3`，或为 domain-specific policy enforcement 自建。
- **ChatHTN**：research code；这个模式（symbolic + LLM fallback）可以干净地移植到任何 HTN planner。
- **AlphaEvolve**：DeepMind 论文；这个模式（ensemble + evaluator）可复现。OpenEvolve 和类似 open-source forks 正在出现。
- **Agent frameworks**：目前还没有一等支持 HTN 或 AlphaEvolve。把它构建为 subagent 或 background worker。

## 发布它

`outputs/skill-hybrid-planner.md` 会生成一个 hybrid planner scaffold（HTN 或 evolutionary），并明确限定 LLM 的角色。

## 练习

1. 给 HTN planner 添加 backtracking：当 operator 的 postcondition 在 runtime 失败时，回滚并尝试下一个 method。
2. 给 ChatHTN 添加 LLM-method cache：当 LLM 在 state pattern `P` 中分解 task `T` 时，存储结果。下一次调用时先重新检查 method library。
3. 把 evolutionary search evaluator 换成真实 test suite。Evolve 一个通过 20 个 test cases 的 sort function；报告收敛需要的 generations。
4. 阅读 AlphaEvolve 的 evaluator design notes。为你关心的 domain 设计一个 evaluator（SQL query optimization、test-suite minimization、deployment YAML）。
5. 组合使用：用 HTN 把 compound task 分解成 subtasks，再对每个 subtask 的 primitive operator 使用 evolutionary search。它在哪里发光？在哪里过度工程？

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|----------------|------------------------|
| HTN | “Hierarchical planner” | 带 operators、preconditions、effects 的 task decomposition |
| Method | “Decomposition rule” | 把 compound task 拆成 subtasks 的方式 |
| Operator | “Primitive action” | 带 precondition 和 effect 的具体步骤 |
| ChatHTN | “LLM + HTN” | 没有 method 匹配时，symbolic planner 询问 LLM |
| AlphaEvolve | “Evolutionary code search” | LLM ensemble mutate code；deterministic evaluator 选择 |
| Fitness function | “Evaluator” | 对 outputs 做 deterministic、machine-checkable scoring |
| Online method learning | “Cached LLM decomposition” | 存储并泛化 LLM plans，以降低 query cost |

## 延伸阅读

- [Gopalakrishnan et al., ChatHTN (arXiv:2505.11814)](https://arxiv.org/abs/2505.11814)：symbolic + LLM hybrid planner
- [Novikov et al., AlphaEvolve (arXiv:2506.13131)](https://arxiv.org/abs/2506.13131)：带 LLM mutations 的 evolutionary code search
- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents)：什么时候使用 planner，什么时候使用 simple loop
