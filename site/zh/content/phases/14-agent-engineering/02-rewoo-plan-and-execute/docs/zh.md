# ReWOO 和 Plan-and-Execute：解耦规划

> ReAct 把 thought 和 action 交错放在同一条流里。ReWOO 把它们分开：先做一个完整的大计划，再执行。tokens 少 5 倍，HotpotQA accuracy 提升 4%，而且你可以把 planner 蒸馏进 7B 模型。Plan-and-Execute 泛化了它；Plan-and-Act 把它扩展到 web navigation。

**类型：** 构建
**语言：** Python (stdlib)
**前置要求：** 阶段 14 · 01（Agent Loop）
**时间：** ~60 分钟

## 学习目标

- 解释为什么 ReWOO 的 Planner / Worker / Solver 拆分比 ReAct 的 interleaved loop 更省 tokens、更稳健。
- 实现一个 plan DAG、一个按依赖顺序执行的 executor，以及一个组合 worker outputs 的 solver，全部只用 stdlib。
- 使用 2026 年“五种 workflow patterns”的框架（Anthropic），判断任务应该用 plan-then-execute 还是 interleaved ReAct。
- 识别什么时候需要 Plan-and-Act 的 synthetic plan data 来处理 long-horizon web 或 mobile tasks。

## 问题

ReAct 的 interleaved thought-action-observation loop 简单且灵活，但每次 tool call 都必须带上完整的历史上下文，包括之前的每个 thought。Token 使用量会随深度按二次方增长。更糟的是，当工具在 loop 中途失败时，模型必须从 error observation 重新推导整个计划。

ReWOO（Xu et al., arXiv:2305.18323, May 2023）注意到这一点，并做了一个取舍：先把整件事规划出来，并行获取 evidence，最后组合答案。一次 LLM call 做 plan，N 次 tool calls 获取 evidence（可以并行），一次 LLM call 做 solve。代价是灵活性较低（plan 是静态的），换来更好的 token efficiency 和更清晰的 failure modes。

## 概念

### 三个角色

```
Planner:  user_question -> [plan_dag]
Workers:  [plan_dag]     -> [evidence]        (tool calls, possibly parallel)
Solver:   user_question, plan_dag, evidence -> final_answer
```

Planner 生成 DAG。每个 node 指定一个 tool、它的 arguments，以及它依赖哪些更早的 nodes（例如 `#E1`、`#E2`）。Workers 按拓扑顺序执行 nodes。Solver 把所有东西拼起来。

### 为什么 tokens 少 5 倍

ReAct 的 prompt 长度会随 step count 线性增长。到第 10 步，prompt 里包含 thought 1 加 action 1 加 observation 1，再加 thought 2、action 2、observation 2，如此类推。每个中间步骤还会冗余包含原始 prompt。

ReWOO 支付一次 planner prompt（大）、N 个小 worker prompts（每个只是 tool call，没有 chain）和一次 solver prompt。HotpotQA 上，论文测得 tokens 少约 5 倍，同时 absolute accuracy 高 4 个点。

### 为什么它更稳健

如果 worker 3 在 ReAct 中失败，loop 必须在流中途从错误里推理出来。在 ReWOO 中，worker 3 返回一个 error string；solver 会把它和原始计划一起看见，并能 graceful degrade。Failure localization 是 per-node，而不是 per-step。

### Planner distillation

论文的第二个结果：由于 planner 看不到 observations，你可以用 175B teacher 的 planner outputs 微调一个 7B 模型。小模型负责 planning；推理时不需要大模型。这现在已经是标准做法。很多 2026 年生产 agents 使用小 planner 加大 executor，或者反过来。

### Plan-and-Execute（LangChain, 2023）

LangChain 团队 2023 年 8 月的文章把 ReWOO 泛化成一个模式名：Plan-and-Execute。前置 planner 发出 step list，executor 运行每一步，可选 replanner 在观察结果后修订。这比 ReWOO 更接近 ReAct（replanner 会把 observations 带回 planning），但保留了 token savings。

### Plan-and-Act（Erdogan et al., arXiv:2503.09572, ICML 2025）

Plan-and-Act 把这个模式扩展到 long-horizon web 和 mobile agents。关键贡献是 synthetic plan data：一个带标签 trajectory generator 生成显式 plan 的训练数据。它用来微调 planner models，让模型在 WebArena-like tasks 中超过 30-50 步后仍能工作，而单条 ReAct trajectory 通常会失去 coherence。

### 什么时候选哪种

| 模式 | 适用场景 |
|---------|------|
| ReAct | 短任务、未知环境、需要 reactive exception handling |
| ReWOO | 结构化任务、已知工具、tokens 敏感、evidence 可并行 |
| Plan-and-Execute | 类似 ReWOO，但 partial execution 后需要 replanning |
| Plan-and-Act | Long-horizon（>30 步）、web/mobile/computer-use |
| Tree of Thoughts | Search 值得付费（第 04 课） |

Anthropic 2024 年 12 月的建议：从最简单的开始。如果任务只是一次 tool call 加一个 summary，不要构建 ReWOO。如果任务是一个 40 步 research assignment，不要只用 ReAct。

## 构建它

`code/main.py` 实现了一个 toy ReWOO：

- `Planner`：一个 scripted policy，从 prompt 生成 plan DAG。
- `Worker`：通过 registry 分发每个 node 的 tool call。
- `Solver`：scripted composition，读取 evidence 并生成 final answer。
- Dependency resolution：像 `#E1` 这样的引用会替换成更早的 worker outputs。

Demo 回答：“What is the population of the capital of France, rounded to millions?” 它使用两步计划：(1) 查首都，(2) 查人口，然后 solve。

运行它：

```
python3 code/main.py
```

Trace 会先显示完整 plan，然后是 worker results，最后是 solver composition。把 token count（我们打印一个粗略 character count）和 ReAct-style interleaved run 对比，在这种结构化任务上 ReWOO 会赢。

## 使用它

LangGraph 把 Plan-and-Execute 作为 recipe 提供（`create_react_agent` 用于 ReAct，custom graphs 用于 plan-execute）。CrewAI 的 Flows 直接编码这个模式：你先定义 tasks，Flow DAG 执行它们。Plan-and-Act 的 synthetic data 方法目前仍主要是研究；runtime pattern（显式 plan DAG）通过 LangGraph 和 CrewAI Flows 在生产中落地。

## 发布它

`outputs/skill-rewoo-planner.md` 会在给定 tool catalog 的情况下，从用户请求生成 ReWOO plan DAG。它会先校验 plan（无环、每个引用已解析、每个工具存在），再交给 executor。

## 练习

1. 为独立 plan nodes 并行化 worker execution。在一个 6-node DAG、2 个 parallel groups 上，它会带来什么？
2. 添加一个 replanner node，在任何 worker 返回 error 时触发。让 ReWOO 变成 Plan-and-Execute 的最小改动是什么？
3. 用小模型（7B class）替换 `Planner`，让 `Solver` 继续使用 frontier model。比较 end-to-end quality。这个拆分在哪里失败？
4. 阅读 ReWOO 论文关于 planner distillation 的第 4 节。概念性复现 175B -> 7B 结果：你需要什么训练数据？如何给 plan quality 打分？
5. 把 toy 移植到 Plan-and-Act 的 trajectory 形状：plan 是 sequence，不是 DAG。哪些 tradeoffs 会改变？

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|----------------|------------------------|
| ReWOO | “Reasoning without observations” | 先计划，再并行获取 evidence，最后 solve；planning prompt 中没有 observations |
| Plan-and-Execute | “LangChain 的 plan-execute pattern” | ReWOO 加上 execution 后可选的 replanner node |
| Plan-and-Act | “扩展版 plan-execute” | 显式 planner/executor 拆分，并用 synthetic plan training data 支持 long-horizon tasks |
| Evidence reference | “#E1, #E2, ...” | Plan-node placeholder，在 dispatch 时替换成之前的 worker output |
| Planner distillation | “Small planner, big executor” | 用大 teacher 的 planner traces 微调小模型 |
| Token efficiency | “Fewer round trips” | 论文中 HotpotQA 上比 ReAct 少 5 倍 tokens |
| DAG executor | “Topological dispatcher” | 按依赖顺序运行 plan nodes；每一层可以并行 |

## 延伸阅读

- [Xu et al., ReWOO: Decoupling Reasoning from Observations (arXiv:2305.18323)](https://arxiv.org/abs/2305.18323)：经典论文
- [Erdogan et al., Plan-and-Act (arXiv:2503.09572)](https://arxiv.org/abs/2503.09572)：带 synthetic plans 的扩展 planner-executor
- [LangGraph Plan-and-Execute tutorial](https://docs.langchain.com/oss/python/langgraph/overview)：框架 recipe
- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents)：选择能工作的最简单模式
