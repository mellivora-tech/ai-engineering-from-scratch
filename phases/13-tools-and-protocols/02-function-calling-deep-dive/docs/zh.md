# Function Calling 深入解析：OpenAI、Anthropic、Gemini

> 三家 frontier provider 在 2024 年收敛到了同一个 tool-call 循环，然后在其他所有细节上分叉。OpenAI 使用 `tools` 和 `tool_calls`。Anthropic 使用 `tool_use` 和 `tool_result` block。Gemini 使用 `functionDeclarations` 和 unique-id correlation。本课把三者并排 diff，让一份在某个 provider 上上线的代码，在迁移到另一个 provider 时不会被 plumbing 绊倒。

**类型：** 构建
**语言：** Python（stdlib，schema translators）
**前置要求：** 阶段 13 · 01（工具接口）
**时间：** ~75 分钟

## 学习目标

- 说出 OpenAI、Anthropic 和 Gemini function-calling payload 的三类形状差异（declaration、call、result）。
- 在三个 provider 格式之间翻译同一个工具声明，并预测 strict-mode 约束会在哪里不同。
- 在每个 provider 中使用 `tool_choice` 来强制、禁止或自动选择工具调用。
- 了解每个 provider 的硬限制（工具数量、schema 深度、argument 长度）以及违反限制时的错误特征。

## 问题

function-calling request 的形状因 provider 而异。2026 年生产栈里的三个具体例子：

**OpenAI Chat Completions / Responses API。** 你传入 `tools: [{type: "function", function: {name, description, parameters, strict}}]`。模型响应包含 `choices[0].message.tool_calls: [{id, type: "function", function: {name, arguments}}]`，其中 `arguments` 是你必须解析的 JSON 字符串。Strict mode（`strict: true`）通过 constrained decoding 强制 schema compliance。

**Anthropic Messages API。** 你传入 `tools: [{name, description, input_schema}]`。响应返回为 `content: [{type: "text"}, {type: "tool_use", id, name, input}]`。`input` 已经是解析后的 object，不是字符串。你需要回复一个新的 `user` message，其中包含 `{type: "tool_result", tool_use_id, content}` block。

**Google Gemini API。** 你传入 `tools: [{functionDeclarations: [{name, description, parameters}]}]`（嵌套在 `functionDeclarations` 下）。响应位于 `candidates[0].content.parts: [{functionCall: {name, args, id}}]`，其中 `id` 在 Gemini 3 及以上版本中是 unique，用于 parallel-call correlation。你回复 `{functionResponse: {name, id, response}}`。

同一个循环。不同字段名、不同嵌套、不同 string-vs-object 约定、不同 correlation 机制。一个团队在 OpenAI 上写了天气 agent，移植到 Anthropic 要为 plumbing 付出两天，再到 Gemini 又要一天。

本课会构建一个 translator，把三种格式统一为一个 canonical tool declaration，并在边界处路由。阶段 13 · 17 会把同一个模式泛化成 LLM gateway。

## 概念

### 共同结构

每个 provider 都需要五件事：

1. **Tool list。** 每个工具的名称、description 和 input schema。
2. **Tool choice。** 强制特定工具、禁止工具，或让模型决定。
3. **Call emission。** 命名工具和参数的结构化输出。
4. **Call id。** 把 response 关联到正确的 call（对并行尤其重要）。
5. **Result injection。** 一条 message 或 block，把结果绑定回 call。

### 逐字段形状差异

| Aspect | OpenAI | Anthropic | Gemini |
|--------|--------|-----------|--------|
| Declaration envelope | `{type: "function", function: {...}}` | `{name, description, input_schema}` | `{functionDeclarations: [{...}]}` |
| Schema field | `parameters` | `input_schema` | `parameters` |
| Response container | assistant message 上的 `tool_calls[]` | `content[]` 中 type 为 `tool_use` 的 block | type 为 `functionCall` 的 `parts[]` entry |
| Arguments type | stringified JSON | parsed object | parsed object |
| Id format | `call_...`（OpenAI 生成） | `toolu_...`（Anthropic） | UUID（Gemini 3+） |
| Result block | role `tool`，`tool_call_id` | 带 `tool_result`、`tool_use_id` 的 `user` | 带匹配 `id` 的 `functionResponse` |
| Force-a-tool | `tool_choice: {type: "function", function: {name}}` | `tool_choice: {type: "tool", name}` | `tool_config: {function_calling_config: {mode: "ANY"}}` |
| Forbid tools | `tool_choice: "none"` | `tool_choice: {type: "none"}` | `mode: "NONE"` |
| Strict schema | `strict: true` | schema-is-schema（始终执行） | request level 的 `responseSchema` |

### 你真的会撞到的限制

- **OpenAI。** 每个 request 128 个工具。Schema 深度 5。Argument string <= 8192 bytes。Strict mode 要求没有 `$ref`，没有重叠的 `oneOf`/`anyOf`/`allOf`，每个 property 都列在 `required` 中。
- **Anthropic。** 每个 request 64 个工具。Schema 深度理论上不设限，实践上 10 左右。没有 strict-mode flag；schema 是 contract，模型通常会遵守。
- **Gemini。** 每个 request 64 个 function。Schema 类型是 OpenAPI 3.0 子集（和 JSON Schema 2020-12 有轻微差异）。Gemini 3 起，parallel call 有 unique id。

### `tool_choice` 行为

所有 provider 都支持三种模式，只是命名不同。

- **Auto。** 模型选择工具或文本。默认。
- **Required / Any。** 模型必须调用至少一个工具。
- **None。** 模型不能调用工具。

每个 provider 还有一个独有模式：

- **OpenAI。** 按名称强制特定工具。
- **Anthropic。** 按名称强制特定工具；`disable_parallel_tool_use` flag 分离 single vs multi。
- **Gemini。** `mode: "VALIDATED"` 会让每个响应都经过 schema validator，而不管模型意图。

### Parallel calls

OpenAI 的 `parallel_tool_calls: true`（默认）会在一个 assistant message 中输出多个调用。你全部运行，然后用一个 batched tool-role message 回复，每个条目对应一个 `tool_call_id`。Anthropic 过去偏向 single-call；`disable_parallel_tool_use: false`（Claude 3.5 起默认）启用 multi。Gemini 2 允许 parallel calls 但没有稳定 id；Gemini 3 加入 UUID，让乱序响应能干净关联。

### Streaming

三者都支持 streamed tool calls。wire format 不同：

- **OpenAI。** `tool_calls[i].function.arguments` 的 delta chunk 增量到达。你累积到 `finish_reason: "tool_calls"`。
- **Anthropic。** Block-start / block-delta / block-stop events。`input_json_delta` chunk 携带部分参数。
- **Gemini。** `streamFunctionCallArguments`（Gemini 3 新增）发出带 `functionCallId` 的 chunk，因此多个 parallel call 可以交错。

阶段 13 · 03 会深入 parallel + streaming reassembly。本课聚焦声明和 single-call 形状。

### Errors and repair

invalid-argument error 的样子也不同。

- **OpenAI（non-strict）。** 模型返回 `arguments: "{bad json}"`，你的 JSON parse 失败，你注入错误消息并重新调用。
- **OpenAI（strict）。** 验证发生在 decoding 期间；invalid JSON 不可能出现，但 `refusal` 可能出现。
- **Anthropic。** `input` 可能包含意外字段；schema 偏 advisory。要在 server side 验证。
- **Gemini。** OpenAPI 3.0 怪癖：object field 上的 `enum` 可能被静默忽略；自己验证。

### Translator pattern

你代码中的 canonical tool declaration 大致长这样（形状由你决定）：

```python
Tool(
    name="get_weather",
    description="Use when ...",
    input_schema={"type": "object", "properties": {...}, "required": [...]},
    strict=True,
)
```

三个小函数把它翻译成三种 provider 形状。`code/main.py` 中的 harness 正是这么做的，然后把一个 fake tool call 通过每个 provider 的响应形状 round-trip 一遍。无需网络——本课教的是形状，不是 HTTP。

生产团队会把这个 translator 包进 `AbstractToolset`（Pydantic AI）、`UniversalToolNode`（LangGraph）或 `BaseTool`（LlamaIndex）。阶段 13 · 17 会交付一个 gateway，在任意三家 provider 前面暴露 OpenAI-shaped API。

## 使用它

`code/main.py` 定义了一个 canonical `Tool` dataclass 和三个 translator，用来输出 OpenAI、Anthropic、Gemini 的 declaration JSON。它随后把每种形状的手写 provider response 解析成同一个 canonical call object，证明表层下面的语义相同。运行它，并把三种 declaration 并排 diff。

重点看：

- 三个 declaration block 只在 envelope 和字段名上不同。
- 三个 response block 只在 call 所在位置上不同（top-level `tool_calls`、`content[]` block、`parts[]` entry）。
- 一个 `canonical_call()` 函数从三种 response shape 中提取 `{id, name, args}`。

## 交付它

本课产出 `outputs/skill-provider-portability-audit.md`。给定一个基于某个 provider 的 function-calling integration，这个 skill 会生成 portability audit：它依赖哪些 provider limit，哪些字段需要重命名，移植到其他 provider 时会坏在哪里。

## 练习

1. 运行 `code/main.py`，验证三种 provider declaration JSON 都序列化了同一个底层 `Tool` object。修改 canonical tool，添加一个 enum 参数，并确认只有 Gemini translator 需要处理 OpenAPI 怪癖。

2. 为每个 provider 添加一个 `ListToolsResponse` parser，用来提取模型在 `list_tools` 或 discovery call 之后返回的工具列表。OpenAI 原生没有这个接口；记下这个不对称。

3. 实现 `tool_choice` 转换：把 canonical `ToolChoice(mode="force", tool_name="x")` 映射到三种 provider 形状。然后映射 `mode="any"` 和 `mode="none"`。对照本课 diff 表检查。

4. 选择三家 provider 中的一家，从头到尾阅读它的 function-calling guide。找出一个 schema spec 字段是另外两家不支持的。候选项：OpenAI `strict`、Anthropic `disable_parallel_tool_use`、Gemini `function_calling_config.allowed_function_names`。

5. 写一个 test vector：一个参数违反声明 schema 的 tool call。把它通过每个 provider 的 validator（第 01 课的 stdlib validator 可做代理）并记录触发了哪些错误。说明你会在生产中选择哪个 provider 来获得严格性。

## 关键词

| Term | 大家常说 | 实际含义 |
|------|----------|----------|
| Function calling | “Tool use” | provider-level API，用于输出结构化 tool-call |
| Tool declaration | “Tool spec” | Name + description + JSON Schema input payload |
| `tool_choice` | “Force / forbid” | Auto / required / none / specific-name 模式 |
| Strict mode | “Schema enforcement” | OpenAI flag，约束 decoding 以匹配 schema |
| `tool_use` block | “Anthropic 的调用形状” | 带 id、name、input 的 inline content block |
| `functionCall` part | “Gemini 的调用形状” | 包含 name、args、id 的 `parts[]` entry |
| Arguments-as-string | “Stringified JSON” | OpenAI 返回的是 JSON string，不是 object |
| Parallel tool calls | “一个 turn 里 fan-out” | 一个 assistant message 中的多个 tool call |
| Refusal | “模型拒绝” | strict-mode-only refusal block，替代调用 |
| OpenAPI 3.0 subset | “Gemini schema 怪癖” | Gemini 使用一种类似 JSON Schema、但略有差异的 dialect |

## 延伸阅读

- [OpenAI — Function calling guide](https://platform.openai.com/docs/guides/function-calling) — 包括 strict mode 和 parallel calls 的权威参考
- [Anthropic — Tool use overview](https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/overview) — `tool_use` 和 `tool_result` block 语义
- [Google — Gemini function calling](https://ai.google.dev/gemini-api/docs/function-calling) — parallel calls、unique ids 和 OpenAPI subset
- [Vertex AI — Function calling reference](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/multimodal/function-calling) — Gemini 的企业级界面
- [OpenAI — Structured outputs](https://platform.openai.com/docs/guides/structured-outputs) — strict-mode schema enforcement 细节
