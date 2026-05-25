# 工具接口：为什么 Agent 需要结构化 I/O

> 语言模型会生成 token。程序会采取行动。两者之间的空隙就是工具接口：一份合约，让模型能够请求一个动作，并让宿主执行它。2026 年的每个栈都是同一个四步循环的不同编码方式：OpenAI、Anthropic 和 Gemini 上的 function calling，MCP 的 `tools/call`，A2A 的 task parts。本课会命名这个循环，并展示运行它所需的最小机制。

**类型：** 学习
**语言：** Python（stdlib，无 LLM）
**前置要求：** 阶段 11（LLM completion APIs）
**时间：** ~45 分钟

## 学习目标

- 解释为什么一个只能生成文本的 LLM 本身无法对真实世界采取行动。
- 画出四步 tool-call 循环（describe → decide → execute → observe），并说出每一步由谁负责。
- 把一个工具描述写成三部分：名称、JSON Schema 输入，以及确定性的 executor 函数。
- 区分纯工具和有副作用的工具，并说明这种区分为什么影响安全。

## 问题

LLM 输出的是下一个 token 的概率分布。这就是它的全部输出界面。如果你问聊天模型“班加罗尔现在天气怎么样”，它可以写出一句看似合理的话，但它不能真的拨进天气 API。这句话可能碰巧正确，也可能已经过期三天。

工具接口的目的就是补上这个空隙。宿主程序——你的 agent runtime、Claude Desktop、ChatGPT、Cursor，或者自定义脚本——向模型公布一组可调用工具。模型在判断需要动作时，会输出一个结构化 payload，里面写明工具名和参数。宿主解析这个 payload，真实运行工具，再把结果喂回去。循环会一直继续，直到模型判断不再需要更多调用。

这份合约的第一个版本在 2023 年 6 月以 OpenAI 的 `functions` 参数形式发布。Anthropic 随后在 Claude 2.1 中加入 `tool_use` block。Gemini 几个月后加入 `functionDeclarations`。如今每个 provider 都暴露同样的形状：输入一组用 JSON Schema 标注类型的工具列表，输出一个 JSON payload 的工具调用。Model Context Protocol（2024 年 11 月）把这份合约泛化为一个工具注册表服务所有模型。A2A（2026 年 4 月，v1.0）又把同一个 primitive 用于 agent-to-agent 委派。

四步循环是所有这些东西下面不变的结构。阶段 13 的其他内容都是它的延展。

## 概念

### 第一步：describe

宿主用三个字段声明每个工具。

- **Name。** 稳定、机器可读的标识符。用 `get_weather`，不要用 "weather thing"。
- **Description。** 一段自然语言简介。“当用户询问某个具体城市的当前天气时使用。不要用于历史数据。”
- **Input schema。** 一个 JSON Schema object（draft 2020-12），描述工具的参数。

模型会收到这份列表。现代 provider 会用 provider-specific 模板把这些声明序列化进 system prompt，所以作为调用方，你只需要处理结构化形式。

### 第二步：decide

给定用户消息和可用工具，模型会选择三种行为之一。

1. **直接用文本回答。** 不调用工具。
2. **调用一个或多个工具。** 输出结构化 call object。在 `parallel_tool_calls: true` 下（OpenAI 和 Gemini 默认，Anthropic 需 opt-in），模型可以在一个 turn 里输出多个调用。
3. **拒绝。** strict-mode structured outputs 可以生成带类型的 `refusal` block，而不是调用。

一个工具调用 payload 有三个稳定字段：call `id`、工具 `name` 和 JSON `arguments` object。id 的存在是为了让宿主把后续结果和具体调用关联起来；当并行调用乱序返回时，这一点尤其重要。

### 第三步：execute

宿主收到调用后，会按声明的 schema 验证参数，并运行 executor。参数无效意味着模型幻觉了字段或用了错误类型——这在弱模型上非常常见。生产宿主面对无效参数通常有三种做法：快速失败并把错误暴露给模型，用受约束 parser 修复 JSON，或者把验证错误放进 prompt 后重试模型。

executor 本身就是普通代码。Python、TypeScript、shell 命令、数据库查询都可以。它产生一个结果，通常是字符串，但也可以是任意 JSON 值或结构化 content block（MCP 中的 text、image 或 resource reference）。结果必须可序列化。

### 第四步：observe

宿主把工具结果追加到对话中（作为带匹配 `id` 的 `tool` role message），然后重新调用模型。模型现在在 context 中看到了工具输出，可以给出最终答案，也可以请求更多调用。这会一直继续，直到模型停止输出调用，或宿主触及迭代次数的安全上限。

### 信任分割

从安全角度看，工具分两类。

- **Pure。** 只读、确定性、无副作用。`get_weather`、`search_docs`、`get_current_time`。可以安全地试探性调用。
- **Consequential。** 会修改状态、花钱、触及用户数据。`send_email`、`delete_file`、`execute_trade`。必须加 gate。

Meta 2026 年面向 agent security 的 “Rule of Two” 说，一个 turn 最多只能同时包含以下三者中的两者：不可信输入、敏感数据、有后果的动作。工具接口正是你执行这条规则的地方——通过拒绝调用、要求用户确认或提升 scope。完整安全章节见阶段 13 · 15，agent-level 权限策略见阶段 14 · 09。

### 循环在哪里发生

| Context | 谁 describe | 谁 decide | 谁 execute |
|---------|-------------|-----------|------------|
| Single-turn function calling（OpenAI/Anthropic/Gemini） | App developer | LLM | App developer |
| MCP | MCP server | 通过 MCP client 的 LLM | MCP server |
| A2A | Agent Card publisher | Calling agent | Called agent |
| Web browser（function-calling agent） | Browser extension / WebMCP | LLM | Browser runtime |

到处都是同样的四步。列名会变，结构不会变。

### 为什么不直接 prompt 模型输出 JSON？

“要求模型用 JSON 回复”是 function calling 之前的模式。即使在 frontier model 上，它也会有约 5% 到 15% 的失败率，在小模型上更高。失败模式包括缺少花括号、尾随逗号、幻觉字段和错误类型。然后你就需要 JSON repair pass、retry 或 constrained decoder。

原生 function calling 更好有三个原因。第一，provider 会用精确调用形状对模型做端到端训练，所以 strict mode 下 valid-JSON 率会升到 98% 到 99%。第二，call payload 位于自己的协议槽位里，不在自由文本中，所以工具调用不会泄漏进用户可见回复。第三，provider 会用 constrained decoding 强制 schema compliance（OpenAI strict mode、Anthropic `tool_use`、Gemini `responseSchema`）。输出保证能通过验证。

阶段 13 · 02 会并排讲三个 provider API。阶段 13 · 04 会深入 structured outputs。

### Circuit breakers

当模型停止输出调用，或宿主达到最大 turn 数时，循环终止。生产宿主通常把这个值设在 5 到 20 个 turn 之间。超过这个范围，你几乎肯定进入了模型无法退出的循环。Claude Code 默认 20，OpenAI Assistants 默认 10，Cursor 的 agent mode 默认 25。

另一种选择——无限循环——每隔六个月就会以“agent 一夜之间花了 400 美元 API 调用费”的事故复盘出现。没有边界不要上线。

阶段 14 · 12 会深入 error recovery 和 self-healing；阶段 17 会讲生产 rate limits。

### 阶段 13 接下来要做什么

- 第 02 到 05 课打磨 provider-level tool-call surface。
- 第 06 到 14 课把这个循环泛化到 MCP。
- 第 15 到 18 课防御敌意 server、对抗性用户和未认证的 remote auth surface。
- 第 19 到 22 课把模式扩展到 agent-to-agent collaboration、observability、routing 和 packaging。
- 第 23 课会交付一个使用所有 primitive 的完整生态系统。

后面每一课都是这个四步循环的延展。把它当作不变量记在脑子里。

## 使用它

`code/main.py` 在没有 LLM 的情况下运行四步循环。一个假的 “decider” 函数通过 pattern matching 用户消息来模拟模型；executor、schema validator 和 observe-step harness 都是真实的。运行它，观察完整 request/response 编排和可打印的中间状态，然后在后续课程中把 fake decider 换成任意真实 provider。

重点看：

- 工具注册表为每个工具保存三个字段：name、description、schema，以及 executor reference。
- validator 是一个最小 JSON Schema 子集（types、required、enum、min/max），只用 stdlib 编写。阶段 13 · 04 会提供更完整的版本。
- 循环把迭代次数限制为 5。生产 agent 正需要这种 circuit breaker。

## 交付它

本课产出 `outputs/skill-tool-interface-reviewer.md`。给定一份草稿工具定义（name + description + schema + executor outline），这个 skill 会审计它是否适合进入循环：名称是否机器稳定，description 是否完整说明使用场景，schema 是否正确使用 JSON Schema 2020-12，以及 pure-vs-consequential 分类是否明确。

## 练习

1. 在 `code/main.py` 中添加第四个工具 `get_stock_price(ticker)`。把它的 description 写成 "Use when the user asks for a current stock price by ticker. Do not use for historical prices or market summaries." 运行 harness，并确认 fake decider 会把提到 ticker 的查询路由到新工具。

2. 故意破坏 schema validator。传入一个缺少 required field 的 `arguments` object，并确认宿主会在执行前拒绝它。然后传入一个额外的未知字段。决定：宿主应该拒绝还是忽略？用安全论证说明你的选择。

3. 把 harness 中的每个工具分类为 pure 或 consequential。给需要的 registry entry 添加 `consequential: true` flag，并修改循环，让它在选择 consequential tool 时打印一行 "would confirm with user"。这就是每个生产宿主都需要的 confirmation gate 形状。

4. 在纸上画出四步循环，并为你最喜欢的 client（Claude Desktop、Cursor、ChatGPT 或自定义栈）填上上面的 provider-column 表。和阶段 13 · 06 的 MCP-specific 版本交叉对照。

5. 从头到尾阅读 OpenAI 的 function-calling guide。找出一个位于 request 中、但不在本课四步循环里的字段。解释它增加了什么，以及为什么它方便但不是本质。

## 关键词

| Term | 大家常说 | 实际含义 |
|------|----------|----------|
| Tool | “模型可以调用的东西” | name + JSON-Schema-typed input + executor function 三元组 |
| Function calling | “原生工具使用” | provider-level API 支持输出结构化工具调用，而不是 prose |
| Tool call | “模型的行动请求” | 模型输出的 JSON payload，包含 `id`、`name`、`arguments` |
| Tool result | “工具返回了什么” | executor 的输出，包装成带匹配 id 的 `tool` role message |
| Parallel tool calls | “一次多个调用” | 一个 model turn 中的多个 call object，彼此独立，并可按 id 排序 |
| Strict mode | “保证 JSON” | 强制模型输出按声明 schema 验证通过的 constrained decoding |
| Pure tool | “只读工具” | 无副作用；可以安全重跑 |
| Consequential tool | “动作工具” | 修改外部状态；需要 gate、audit 或用户确认 |
| Four-step loop | “tool-call cycle” | describe → decide → execute → observe |
| Host | “Agent runtime” | 持有工具注册表、调用模型并运行 executor 的程序 |

## 延伸阅读

- [OpenAI — Function calling guide](https://platform.openai.com/docs/guides/function-calling) — OpenAI 风格工具声明和调用形状的权威参考
- [Anthropic — Tool use overview](https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/overview) — Claude 的 `tool_use` / `tool_result` block 格式
- [Google — Gemini function calling](https://ai.google.dev/gemini-api/docs/function-calling) — Gemini 中的 `functionDeclarations` 和 parallel-call 语义
- [Model Context Protocol — Specification 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25) — 工具接口的 provider-agnostic 泛化
- [JSON Schema — 2020-12 release notes](https://json-schema.org/draft/2020-12/release-notes) — 每个现代工具 API 都在使用的 schema dialect
