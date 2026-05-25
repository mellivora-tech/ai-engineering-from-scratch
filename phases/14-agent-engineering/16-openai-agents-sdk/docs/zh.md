# OpenAI Agents SDK：Handoffs、Guardrails、Tracing

> OpenAI Agents SDK 是构建在 Responses API 上的 lightweight multi-agent framework。五个 primitives：Agent、Handoff、Guardrail、Session、Tracing。Handoffs 是命名为 `transfer_to_<agent>` 的 tools。Guardrails 会在 input 或 output 上 trip。Tracing 默认开启。

**类型：** 学习 + 构建
**语言：** Python (stdlib)
**前置要求：** 阶段 14 · 01（Agent Loop），阶段 14 · 06（Tool Use）
**时间：** ~75 分钟

## 学习目标

- 说出 OpenAI Agents SDK 的五个 primitives。
- 解释 handoffs：为什么它们被建模为 tools、模型看到的 name shape 是什么、context 如何转移。
- 区分 input guardrails、output guardrails 和 tool guardrails；解释 `run_in_parallel` vs blocking mode。
- 实现一个 stdlib runtime，包含 handoffs + guardrails + span-style tracing。

## 问题

不能干净 delegation 的 agents 最终会把所有东西塞进一个 prompt。没有 guardrails 的 agents 会发布 PII、policy-violating output，或者永远 loop。OpenAI 的 SDK 把让 multi-agent work 可处理的三个 primitives 固化下来。

## 概念

### 五个 primitives

1. **Agent。** LLM + instructions + tools + handoffs。
2. **Handoff。** 委托给另一个 agent。对模型表示为名为 `transfer_to_<agent_name>` 的 tool。
3. **Guardrail。** 对 input（仅第一个 agent）、output（仅最后一个 agent）或 tool invocation（每个 function tool）做 validation。
4. **Session。** 跨 turns 自动保存 conversation history。
5. **Tracing。** LLM generations、tool calls、handoffs、guardrails 的 built-in spans。

### Handoffs as tools

模型会在 tool list 里看到 `transfer_to_billing_agent`。调用它会通知 runtime：

1. 复制 conversation context（或通过 `nest_handoff_history` beta 折叠）。
2. 用 target agent 的 instructions 初始化它。
3. 用 target agent 继续 run。

这是产品化的 supervisor pattern（第 13 课 / 第 28 课）。

### Guardrails

三种风味：

- **Input guardrails。** 在第一个 agent 的 input 上运行。任何 LLM call 前拒绝 unsafe 或 out-of-scope requests。
- **Output guardrails。** 在最后一个 agent 的 output 上运行。捕获 PII leaks、policy violations、malformed responses。
- **Tool guardrails。** 对每个 function-tool 运行。校验 arguments、检查 permissions、审计 execution。

模式：

- **Parallel**（默认）。Guardrail LLM 和 main LLM 一起运行。Tail latency 更低。如果 trip，main LLM 的工作会被丢弃（token waste）。
- **Blocking**（`run_in_parallel=False`）。Guardrail LLM 先运行。如果 trip，就不会浪费 main call 的 tokens。

Tripwires 会抛出 `InputGuardrailTripwireTriggered` / `OutputGuardrailTripwireTriggered`。

### Tracing

默认开启。每个 LLM generation、tool call、handoff 和 guardrail 都会发出 span。`OPENAI_AGENTS_DISABLE_TRACING=1` 可以关闭。`add_trace_processor(processor)` 会把 spans 同时 fan out 到你自己的 backend 和 OpenAI。

### Sessions

`Session` 把 conversation history 存在 backend 中（SQLite、Redis、custom）。`Runner.run(agent, input, session=session)` 会自动 load 并 append。

### 这个模式会在哪里出错

- **Handoff drift。** Agent A hand off 给 Agent B，Agent B 又 hand back 给 Agent A。添加 hop counter。
- **Guardrail bypass。** Tool guardrails 只对 function tools 触发；built-in tools（file reader、web fetch）需要单独 policy。
- **Over-tracing。** Spans 中包含 sensitive content。和 OTel GenAI content-capture rules（第 23 课）搭配：外部存储内容，通过 ID 引用。

## 构建它

`code/main.py` 用 stdlib 实现 SDK shape：

- `Agent`、`FunctionTool`、`Handoff`（作为带 transfer semantics 的 function tool）。
- `Runner`，包含 input/output/tool guardrails、handoff dispatch 和 hop counter。
- 一个 simple span emitter，用来展示 trace shape。
- 一个 triage agent，会根据用户 query hand off 到 billing 或 support；其中一个 input 会触发 guardrail。

运行它：

```
python3 code/main.py
```

Trace 会显示两次成功 handoffs、一次 input guardrail trip，以及一个和真实 SDK 发出内容相似的 span tree。

## 使用它

- **OpenAI Agents SDK** 用于 OpenAI-first products。
- **Claude Agent SDK**（第 17 课）用于 Claude-first products。
- **LangGraph**（第 13 课）用于需要 explicit state 和 durable resume 的场景。
- **Custom** 用于需要精确控制的场景（voice、multi-provider、federated deployments）。

## 发布它

`outputs/skill-agents-sdk-scaffold.md` 会 scaffold 一个 Agents SDK app，包含 triage agent、handoffs、input/output/tool guardrails、session store 和 trace processor。

## 练习

1. 添加 handoff hop counter：N 次 transfers 后拒绝。Trace 这个行为。
2. 把 `nest_handoff_history` 实现为一个选项：transfer 前把 prior messages 折叠成一个 summary。
3. 写一个 blocking output guardrail。比较会 trip 的 prompts 和会 pass 的 prompts 的 latency。
4. 把 `add_trace_processor` 接到 JSON logger。每个 span 发出的 shape 是什么？
5. 阅读 SDK docs。把 stdlib toy 移植到 `openai-agents-python`。你建模错了什么？

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|----------------|------------------------|
| Agent | “LLM + instructions” | SDK 中的 Agent type；拥有 tools 和 handoffs |
| Handoff | “Transfer” | 模型调用以委托给另一个 agent 的 tool |
| Guardrail | “Policy check” | 对 input / output / tool invocation 的 validation |
| Tripwire | “Guardrail trip” | Guardrail 拒绝时抛出的 exception |
| Session | “History store” | 在 runs 之间持久化的 conversation memory |
| Tracing | “Spans” | 覆盖 LLM + tool + handoff + guardrail 的 built-in observability |
| Blocking guardrail | “Sequential check” | Guardrail 先运行；trip 时不浪费 tokens |
| Parallel guardrail | “Concurrent check” | Guardrail 并行运行；latency 更低，trip 时浪费 tokens |

## 延伸阅读

- [OpenAI Agents SDK docs](https://openai.github.io/openai-agents-python/)：primitives、handoffs、guardrails、tracing
- [Claude Agent SDK overview](https://platform.claude.com/docs/en/agent-sdk/overview)：Claude-flavored counterpart
- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents)：什么时候需要 handoffs
- [OpenTelemetry GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/)：Agents SDK spans 映射到的标准
