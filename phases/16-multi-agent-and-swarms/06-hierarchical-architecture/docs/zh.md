# Hierarchical Architecture 及其 Failure Mode

> Hierarchical 是嵌套的 supervisor。Manager agents 管 sub-managers，sub-managers 管 workers。CrewAI `Process.hierarchical` 是教科书版本：一个 `manager_llm` 动态委派 tasks 并验证 outputs。LangGraph 等价形式是 `create_supervisor(create_supervisor(...))`。当任务真的是 org chart 时，这是自然 pattern。它也是最容易坍缩成 managerial looping 的 pattern — manager agents 分配工作不当、误解 sub-outputs，或无法达成 consensus。Sequential 往往胜过它。

**类型：** 学习 + 构建
**语言：** Python（stdlib）
**前置要求：** 阶段 16 · 05（Supervisor Pattern）
**时间：** ~60 分钟

## 问题

一旦 supervisor pattern 想通了，自然下一步就是“如果 workers 自己也是 supervisors 呢？”团队有 sub-teams；公司有 departments of departments。Hierarchical architectures 就是在镜像这一点。

问题是：LLM managers 不等同于人类 managers。人类 manager 对自己的 reports 知道什么有稳定先验。LLM manager 每轮都从 context 中重新推理 org。context 里一点点 drift，整棵树就会误分配工作。

## 概念

### 形状

```
                 Manager
                 ┌─────┐
                 └──┬──┘
           ┌────────┴────────┐
           ▼                 ▼
       Sub-Mgr A         Sub-Mgr B
       ┌─────┐           ┌─────┐
       └──┬──┘           └──┬──┘
         ┌┴──┬──┐          ┌┴──┐
         ▼   ▼  ▼          ▼   ▼
       W1  W2  W3         W4  W5
```

每个 internal node 负责 plan、delegate 和 synthesize。只有 leaves 做实际工作。

### 它擅长的地方

- **清晰的 org mapping。** 如果真实任务是 departmental（“legal review the doc, finance review the doc, engineering review the doc, then summarize for exec”），hierarchy 是显式的。
- **Local summarization。** 每个 sub-manager 在 top manager 看到之前综合自己团队的 output。Top manager 看到三个 sub-manager summaries，而不是十五个 worker outputs。

### 它坏掉的地方

2026 post-mortems 反复发现三种 failure modes：

1. **Task assignment error。** manager 读取 goal，hallucinate 一个 decomposition，并委派给错误 sub-manager。因为 sub-manager 会顺从地做被分配的事，这个错误只会在 top synthesis 时暴露 — 离人类本可捕捉的位置已经隔了一层。
2. **Output misinterpretation。** sub-manager 返回 “unable to verify claim X”。top manager 总结成 “claim X not confirmed”。每一层 meaning 都会 drift。
3. **Consensus loops。** 两个 sub-managers 不一致；top manager 让他们 reconcile；他们重新向下委派；workers 重跑；sub-managers 返回略有不同的答案；循环。CrewAI 的 `Process.hierarchical` 用 step limits 防护，但这个 limit 本身现在成了 hyperparameter。

### 决定性问题

Sequential（linear pipeline）vs hierarchical：你的任务真的有 independent sub-teams，还是一个 linear flow 假装成 tree？如果是后者，用 sequential。如果是前者，用 hierarchical，但要预算显式 reconciliation rules。

### CrewAI 的实现

`Process.hierarchical` 把 manager LLM 接到 specialist crews 之上。manager：

- 接收 top-level task，
- 给 crews 分配 subtasks，
- 评估 crew outputs，
- 决定 accept、re-delegate 或 iterate。

文档：https://docs.crewai.com/en/introduction（在 Core Concepts 下找 “Hierarchical Process”）。

### LangGraph 的实现

LangGraph 使用嵌套的 `create_supervisor` calls。inner supervisor 有自己的 graph；outer supervisor 把 inner graph 当成 opaque node。这比 CrewAI 更容易 debug（可以分别 step through 每个 graph），但更难表达 tree 的动态重塑。

参考：https://reference.langchain.com/python/langgraph-supervisor。

## 构建它

`code/main.py` 运行一个 3-level hierarchy：

- top manager：把 task 拆成 “engineering” 和 “legal” 分支，
- engineering sub-manager：拆成 “frontend” 和 “backend” workers，
- legal sub-manager：一个 worker。

Demo 对比 happy path（所有人一致）和一个 **perturbed path**：top manager 的 decomposition 把 “legal” 错标为 “finance”，然后观察 error cascade — sub-manager 顺从地做 finance work，top synthesizer 报告 finance findings，原始 legal question 没有回答。

运行：

```
python3 code/main.py
```

输出展示两条路径，并清楚对比 “what was asked” 与 “what was delivered”。

## 使用它

`outputs/skill-hierarchy-fitness.md` 评估给定任务应该使用 hierarchical、sequential 还是 flat supervisor。输入：task description、org structure、reconciliation budget。输出：pattern recommendation，以及需要防护的具体 failure modes。

## 发布它

如果你要发布 hierarchical：

- **把 tree depth 限制在 2。** 三层已经会隐藏大多数 observability errors。
- **显式 reconciliation budget。** 设置 top manager 必须 commit 前的 max rounds。通常是 2。
- **每次 synthesis 都带 provenance。** 每个 node 的 summary 必须 cite 由哪些 leaf outputs 产生。
- **对 decomposition drift 告警。** 记录 manager 每步的 decomposition；与 user query diff。如果 decomposition 不再覆盖 query，就触发 alert。

## 练习

1. 运行 `code/main.py` 并比较 happy vs perturbed。需要多少层 manager hand-off，top output 才会完全偏离 user question？
2. 添加第三层（top → sub → sub-sub → worker）。测量随着 depth 增长，perturbed path 有多常自我修正 vs 完全发散。
3. 在每个 sub-manager 下实现一个 “canary” worker，它总是被问原始 user question，不做改写。用 canary answer 检测 decomposition drift。当 canary 不同意 synthesized answer 时，manager 应该如何反应？
4. 阅读 CrewAI 的 `Process.hierarchical` docs。识别一个具体 guardrail（step limit、manager_llm constraint），并描述它针对什么 failure mode。
5. 比较 nested LangGraph supervisors 与 CrewAI hierarchical。哪个更便宜地检测 reconciliation loops？

## 关键词汇

| 术语 | 人们常说 | 实际含义 |
|------|----------------|------------------------|
| Hierarchical | “Org chart pattern” | supervisors over supervisors；只有 leaves 做工作。 |
| Manager LLM | “The boss” | 在 internal node 负责 decomposes、assigns、validates 的 LLM。 |
| Decomposition drift | “老板跑偏了” | top manager 的拆分不再覆盖原始问题。 |
| Reconciliation loop | “无尽会议” | sub-managers disagree；top re-delegates；workers re-run；循环直到 budget exhausted。 |
| Depth-2 ceiling | “不要超过 2 层” | 经验 guardrail：3+ levels 会让 observability 坍缩。 |
| Canary question | “每层的 ground truth” | 一个始终被问原始 query 的 worker，用来检测 drift。 |
| Provenance chain | “谁说了什么” | 从每次 synthesis 回溯到产生它的 leaf outputs。 |

## 延伸阅读

- [CrewAI introduction — Process.hierarchical](https://docs.crewai.com/en/introduction) — 带 manager LLM 的教科书 hierarchical
- [LangGraph supervisor reference](https://reference.langchain.com/python/langgraph-supervisor) — 通过 `create_supervisor` 嵌套 supervisor
- [Anthropic engineering — Research system](https://www.anthropic.com/engineering/multi-agent-research-system) — Anthropic 为什么刻意选择 flat supervisor 而不是 hierarchical
- [Cemri et al. — Why Do Multi-Agent LLM Systems Fail?](https://arxiv.org/abs/2503.13657) — MAST taxonomy；coordination failures 章节记录 decomposition drift
