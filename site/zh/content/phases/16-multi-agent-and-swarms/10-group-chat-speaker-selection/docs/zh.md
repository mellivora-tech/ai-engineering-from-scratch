# Group Chat 与 Speaker Selection

> AutoGen GroupChat 和 AG2 GroupChat 在 N 个 agents 之间共享一个 conversation；selector function（LLM、round-robin 或 custom）选择谁下一个发言。这是 emergent multi-agent conversation 的原型 — agents 不知道自己在 static graph 中的角色，只是对 shared pool 作出反应。AutoGen v0.2 的 GroupChat semantics 在 AG2 fork 中保留；AutoGen v0.4 将其重写为 event-driven actor model。Microsoft 在 2026 年 2 月将 AutoGen 置于 maintenance mode，并把它与 Semantic Kernel 合并进 Microsoft Agent Framework（RC February 2026）。GroupChat primitive 在 AG2 和 Microsoft Agent Framework 中都存活下来 — 学一次，到处用。

**类型：** 学习 + 构建
**语言：** Python（stdlib）
**前置要求：** 阶段 16 · 04（Primitive Model）
**时间：** ~60 分钟

## 问题

当 workflow 已知时，static graphs（LangGraph）很棒。真实 conversation 不是 static：有时 coder 问 reviewer，有时问 researcher，有时问 writer。硬编码每一种可能 handoff 会造成 edge explosion。你想要的是 *agents reacting to a shared pool*，并由某个 function 决定谁下一个说话。

这正是 AutoGen GroupChat 做的事。

## 概念

### 形状

```
              ┌─── shared pool ────┐
              │   m1  m2  m3  ...  │
              └─────────┬──────────┘
                        │ (everyone reads all)
      ┌───────┬─────────┼─────────┬───────┐
      ▼       ▼         ▼         ▼       ▼
    Agent A  Agent B  Agent C  Agent D  Selector
                                           │
                                           ▼
                                  "next speaker = C"
```

每个 agent 都看到每条 message。每一轮都会调用 selector function 来选择谁下一个说话。

### 三种 selector flavors

**Round-robin。** 固定循环。Deterministic。随 N 线性扩展，但忽略 context — 即使 topic 是 legal review，coder 也会轮到发言。

**LLM-selected。** 调用一个 LLM，读取 recent pool 并返回最佳 next speaker。Context-aware 但慢：每一轮都多一次 LLM call。AutoGen 默认。

**Custom。** 使用任意逻辑的 Python function。典型做法：LLM-selected 加 fallback rules（例如 “coder 之后总是让 verifier 发言”）。

### ConversableAgent API

```
agent = ConversableAgent(
    name="coder",
    system_message="You write Python.",
    llm_config={...},
)
chat = GroupChat(agents=[coder, reviewer, tester], messages=[])
manager = GroupChatManager(groupchat=chat, llm_config={...})
```

`GroupChatManager` 持有 selector。当一个 agent 完成一轮，manager 调用 selector，返回下一个 agent。loop 持续直到 termination condition。

### Termination

三种常见模式：

- **Max rounds。** 对总 turns 设置硬上限。
- **`TERMINATE` token。** Agents 可以发出 sentinel message；manager 看到后停止。
- **Goal-reached check。** 每轮运行 lightweight verifier，当 chat 完成目标时停止。

### AutoGen → AG2 分叉与 Microsoft Agent Framework 合并

2025 年初，Microsoft 开始围绕 event-driven actor model 对 AutoGen 做重大重写（v0.4）。社区把 AutoGen v0.2 的 GroupChat semantics fork 为 AG2，保留早期采用者已集成的 API。

2026 年 2 月，Microsoft 宣布 AutoGen 进入 maintenance mode，event-driven actor model 合并进 **Microsoft Agent Framework**（RC February 2026，现已与 Semantic Kernel 合并）。GroupChat concept 在两条路径中都保留；实现细节不同。AG2 是 v0.2-compatible code 的首选 upstream。

### GroupChat 适合什么时候

- **Emergent conversations。** 你不想预先连接每个 possible next-speaker。
- **Role-mixing tasks。** coder 问 researcher，researcher 问 archivist，archivist 又问 coder。flow 不是 DAG。
- **Exploratory problem-solving。** 想象 “brainstorm meeting”，而不是 “assembly line”。

### 它什么时候失败

- **Strict determinism。** LLM selector 可能不一致。同样 prompt，不同 runs，不同 next speakers。
- **Sycophancy cascades。** Agents 服从最自信的人。需要显式 counter-prompt。
- **Context bloat。** 每个 agent 读取每条 message；10 turns 后 context 很大。使用 projections（第 15 课）来 scope views。
- **Hot speakers。** 一个 agent 因 selector 偏爱其 specialty 而主导 conversation。把 speaker balance 引入 selector feature。

### Group chat vs supervisor

同样 primitives，不同 defaults：

- Supervisor：一个 agent 计划，其他 agents 执行。selector 是“问 planner 要做什么”。
- Group chat：所有 agents 都是 peers；selector 是 over shared pool 的 function。

两者都使用第 04 课的 four primitives。Group chat 默认 LLM-selected orchestration 和 full-pool shared state。

## 构建它

`code/main.py` 用 stdlib 从零实现 GroupChat。三个 agents（coder、reviewer、manager）、round-robin 和 LLM-selected variants，以及基于 `TERMINATE` token 的 termination。

demo 打印 conversation transcript，以及两个 variants 的 selector decision trace。

运行：

```
python3 code/main.py
```

## 使用它

`outputs/skill-groupchat-selector.md` 为给定任务配置 GroupChat selector — round-robin vs LLM-selected vs custom，以及使用什么 selector inputs（recent messages、agent specialties、turn counts）。

## 发布它

Checklist：

- **Max rounds cap。** 永远需要。典型任务 10-20。
- **Speaker-balance metric。** 追踪每个 agent 的 turns；imbalance 超过 threshold 时告警。
- **Termination token。** `TERMINATE` 或 dedicated verifier agent。
- **Projection or scoped memory。** 约 10 messages 后，考虑只给每个 agent scoped view，防止 context bloat。
- **Selector logging。** 对 LLM-selected variants，记录 selector input 和 choice。否则无法调试。

## 练习

1. 运行 `code/main.py`。比较 round-robin 与 LLM-selected 下的 conversation。每种情况下哪个 agent 主导？
2. 在 selector 中添加 “max-speaks-per-agent” rule。它如何影响 transcript？
3. 实现 goal-reached termination：当 reviewer 返回 “approved” 时停止。它多常在 round cap 前触发？
4. 阅读 AutoGen stable docs on GroupChat（https://microsoft.github.io/autogen/stable/user-guide/core-user-guide/design-patterns/group-chat.html）。识别 `GroupChatManager` 使用的 default selector。
5. 阅读 AG2 repo（https://github.com/ag2ai/ag2），比较它的 v0.2 GroupChat 与 v0.4 event-driven version。v0.4 添加了什么具体 property（throughput、fault-tolerance、composability）？

## 关键词汇

| 术语 | 人们常说 | 实际含义 |
|------|----------------|------------------------|
| GroupChat | “Agents 在一个聊天室” | Shared message pool + selector function。AutoGen / AG2 primitive。 |
| Speaker selection | “谁下一个说话” | 选择下一个 agent 的 function。Round-robin、LLM-selected 或 custom。 |
| GroupChatManager | “会议主持人” | AutoGen 组件，拥有 selector 并循环 turns。 |
| ConversableAgent | “base agent” | AutoGen base class；能 send 和 receive messages 的 agent。 |
| Termination token | “停止词” | 结束 chat 的 sentinel string（通常是 `TERMINATE`）。 |
| Hot speaker | “一个 agent 主导” | selector 持续选择同一个 agent 的 failure mode。 |
| Context bloat | “pool 无界增长” | 每个 agent 读取每条 prior message；context 随 turns 增长。 |
| Projection | “Scoped view” | shared pool 的 role-specific view，用来防止 context bloat。 |

## 延伸阅读

- [AutoGen group chat docs](https://microsoft.github.io/autogen/stable/user-guide/core-user-guide/design-patterns/group-chat.html) — 参考实现
- [AG2 repo](https://github.com/ag2ai/ag2) — community AutoGen v0.2 continuation
- [Microsoft Agent Framework docs](https://microsoft.github.io/agent-framework/) — 合并后的继任者，RC February 2026
- [AutoGen v0.4 release notes](https://microsoft.github.io/autogen/stable/) — event-driven actor model rewrite details
