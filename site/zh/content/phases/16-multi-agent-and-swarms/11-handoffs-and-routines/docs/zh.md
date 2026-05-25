# Handoffs and Routines — Stateless Orchestration

> OpenAI 的 Swarm（2024 年 10 月）把 multi-agent orchestration 提炼为两个 primitives：**routines**（instructions + tools 作为 system prompt）和 **handoffs**（返回另一个 Agent 的 tool）。没有 state machine，没有 branching DSL — LLM 通过调用正确的 handoff tool 来 route。OpenAI Agents SDK（2025 年 3 月）是 production successor。Swarm 本身仍然是最清晰的 conceptual reference — 全部源码只有几百行。这个 pattern 流行，是因为 API surface 大约就是 “agent = prompt + tools; handoff = function returning agent”。限制：stateless，所以 memory 是 caller 的问题。

**类型：** 学习 + 构建
**语言：** Python（stdlib）
**前置要求：** 阶段 16 · 04（Primitive Model）
**时间：** ~60 分钟

## 问题

每个 multi-agent framework 都想让你学习它的 DSL：LangGraph nodes and edges、CrewAI crews and tasks、AutoGen GroupChat and managers。这些 DSL 是真实抽象，但它们让事情感觉比实际更重。

Swarm 反方向推进：使用 model 已经拥有的 tool-calling capability。Handoffs 变成 tool calls。orchestrator 就是当前持有 conversation 的 agent。state machine 隐含在 agents 的 system prompts 中。

## 概念

### 两个 primitives

**Routine。** 定义 agent 角色和可用 tools 的 system prompt。可以把它理解为 scoped instructions：“你是 triage agent；如果用户询问退款，就 hand off 给 refund agent。”

**Handoff。** agent 可以调用的 tool，返回一个新的 Agent object。Swarm runtime 检测到 Agent return value，并在下一轮切换 active agent。

这就是全部抽象。

```
def transfer_to_refunds():
    return refund_agent  # Swarm sees Agent return → switch active agent

triage_agent = Agent(
    name="triage",
    instructions="Route the user to the right specialist.",
    functions=[transfer_to_refunds, transfer_to_sales, transfer_to_support],
)
```

triage agent 的 system prompt 让它根据 user message 选择正确 handoff。LLM 的 tool-calling 完成 routing。

### 为什么它流行

- **Small API。** 只需要学两个概念。
- **使用 model 已经会的东西。** Tool calling 已经在各 providers 中 production-grade。
- **没有 state-machine burden。** 你不描述 graph；agents 的 prompts 描述它们会 hand off 给谁。

### Stateless trade

Swarm 明确在 runs 之间 stateless。framework 在一次 run 中保存 message history，但不 persist 任何东西。Memory、continuity、long-running tasks — 都是 caller 的问题。

在 production（OpenAI Agents SDK，2025 年 3 月）中，这是主要变化之一：SDK 添加了 built-in session management、guardrails 和 tracing，同时保留 handoff primitive。

### Swarm/handoffs 适合什么时候

- **Triage patterns。** Front-line agent 把用户 route 到 specialist。
- **Skill-based handoffs。** “如果任务需要 code，call coder；如果需要 research，call researcher。”
- **短且有界的 conversations。** Customer support、FAQ-to-ticket、simple workflows。

### Swarm 吃力的地方

- **带 shared memory 的 long sessions。** Handoffs 把 conversation state 重置为新 agent 的 prompt 加 history。没有 caller-managed memory 就没有跨 agents 的 persistent state。
- **Parallel execution。** Handoff 一次只能一个 — active agent 被切换。并行需要 caller orchestrating multiple Swarm runs。
- **Audit and replay。** Stateless runs 很难精确 replay；LLM 的 handoff choice 不是 deterministic。

### OpenAI Agents SDK（2025 年 3 月）

production successor 添加：

- **Session state。** runs 之间的 persistent thread。
- **Guardrails。** Input/output validation hooks。
- **Tracing。** 每个 tool call 和 handoff 都会被记录。
- **Handoff filters。** 控制 handoff 时传递什么 context。

handoff primitive 仍然存在；production ergonomics 加在它周围。

### Swarm vs GroupChat

两者都使用 LLM-driven routing，但 **谁选择 next** 不同：

- GroupChat：selector（function 或 LLM）从外部选择 next speaker。
- Swarm：当前 agent 通过调用 handoff tool 选择 successor。

Swarm 是“agent decides what's next”；GroupChat 是“manager decides what's next”。Swarm 的 decision 在 active agent 的 tool call 中；GroupChat 的在 `GroupChatManager` 中。

## 构建它

`code/main.py` 从零实现 Swarm：Agent dataclass、handoff mechanism（tool returns Agent），以及检测 agent switches 的 run loop。

Demo：triage agent route 到 refund、sales 或 support specialists。每个 specialist 有自己的 tools。run loop 打印每次 handoff。

运行：

```
python3 code/main.py
```

## 使用它

`outputs/skill-handoff-designer.md` 为给定任务设计 handoff topology：有哪些 agents、它们可以调用哪些 handoffs、传递什么 context。

## 发布它

Checklist：

- **Handoff logging。** 每次 handoff 写 trace event，包含 from-agent、to-agent、context snapshot。
- **Context transfer rules。** 决定 handoff 时移动什么：full history（昂贵）、last N messages，或 summary。
- **Guardrail on handoff。** handoff 到拥有不同 tool permissions 的 specialist 必须 authenticated — 否则 prompt injection 可以强迫不该发生的 handoffs。
- **Loop detection。** 两个 agents 来回 handoff 是常见 failure；用简单 last-K ring check 检测。
- **Fallback agent。** 如果 handoff target 不存在，退回 safe default。

## 练习

1. 运行 `code/main.py`，triage 到 refund agent。确认第二轮 active agent 是 refund。
2. 添加 loop-detection rule：如果同一对 agents 连续 handoff 3 次，就强制退出。设计 fallback。
3. 阅读 OpenAI Agents SDK docs on handoff filters。实现 “summarize-on-handoff” 版本：outgoing agent 在 incoming agent 接手前把 context 压缩为 bullet summary。
4. 比较 Swarm handoff 和 GroupChatManager selector。哪个 pattern 让 prompt injection 更严重，为什么？
5. 阅读 Swarm cookbook（https://developers.openai.com/cookbook/examples/orchestrating_agents）。识别 Swarm 做出的一个明确 design decision，OpenAI Agents SDK 改了还是保留了它。

## 关键词汇

| 术语 | 人们常说 | 实际含义 |
|------|----------------|------------------------|
| Routine | “agent prompt” | System prompt + tool list。定义 role 和 available handoffs。 |
| Handoff | “Transfer to another agent” | active agent 可以调用的 tool，返回一个 new Agent。runtime 切换 active agent。 |
| Stateless | “runs 之间无 memory” | Swarm 不 persist 任何东西；memory 是 caller 责任。 |
| Active agent | “现在谁在说话” | 当前持有 conversation 的 agent。Handoff 改变它。 |
| Context transfer | “handoff 时移动什么” | incoming agent 看到什么 history 的 policy：full、last N 或 summarized。 |
| Handoff loop | “agents 乒乓” | 两个 agents 不断把控制权交回给对方的 failure mode。 |
| OpenAI Agents SDK | “Production Swarm” | 2025 年 3 月 successor；在 handoff primitive 之上添加 sessions、guardrails、tracing。 |
| Handoff filter | “transfer gate” | SDK feature，用来在 handoff boundary inspect 和 modify context。 |

## 延伸阅读

- [OpenAI cookbook — Orchestrating Agents: Routines and Handoffs](https://developers.openai.com/cookbook/examples/orchestrating_agents) — 参考阐述
- [OpenAI Swarm repo](https://github.com/openai/swarm) — 原始实现，作为 conceptual reference 保留
- [OpenAI Agents SDK docs](https://openai.github.io/openai-agents-python/) — 带 sessions 和 tracing 的 production successor
- [Anthropic handoff-in-Claude notes](https://docs.anthropic.com/en/docs/claude-code) — Claude Code subagents 如何通过 `Task` 使用 handoff-like pattern
