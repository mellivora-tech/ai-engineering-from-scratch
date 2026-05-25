# Parallel Tool Calls 和工具 Streaming

> 三个彼此独立的天气查询如果串行执行，就是三次 round trip。并行执行时，总耗时会收缩到最慢的单个调用。每个 frontier provider 现在都能在一个 turn 中输出多个 tool call。收益是真实的；plumbing 很微妙。本课会讲两半：并行 fan-out 和 streamed-argument reassembly，重点强调 id-correlation 这个坑。

**类型：** 构建
**语言：** Python（stdlib，thread pool + streaming harness）
**前置要求：** 阶段 13 · 02（function calling deep dive）
**时间：** ~75 分钟

## 学习目标

- 解释为什么存在 `parallel_tool_calls: true`，以及什么时候要禁用它。
- 在 parallel fan-out 期间把 streamed argument chunk 关联到正确的 tool-call id。
- 在不提前解析的情况下，把部分 `arguments` string 重新组装成完整 JSON。
- 运行一个三城市天气 benchmark，展示 sequential vs parallel latency。

## 问题

没有 parallel calls 时，一个回答“班加罗尔、东京和苏黎世天气怎么样”的 agent 会这样做：

```
user -> LLM
LLM -> call get_weather(Bengaluru)
host -> run executor, reply with result
LLM -> call get_weather(Tokyo)
host -> run executor, reply with result
LLM -> call get_weather(Zurich)
host -> run executor, reply with result
LLM -> final text answer
```

三次 LLM round trip，每次还要付 executor latency。大约是理想 wall-clock time 的 4 倍。

有 parallel calls 时：

```
user -> LLM
LLM -> call get_weather(Bengaluru); call get_weather(Tokyo); call get_weather(Zurich)
host -> run all three executors concurrently, reply with three results
LLM -> final text answer
```

一次 LLM round trip。executor time 是三者的最大值，不是总和。OpenAI、Anthropic 和 Gemini 的生产 benchmark 显示，fan-out workload 的 wall-clock 会下降 60% 到 70%。

代价是 correlation complexity。当三个调用乱序完成时，你的结果必须带上匹配的 `tool_call_id`，让模型能对齐。当结果 streaming 时，你必须把部分 argument fragment 组装成完整 JSON 后才能执行。Gemini 3 加入 unique id，部分原因就是解决真实世界里两个并行调用同一工具时无法区分的问题。

## 概念

### 启用 parallel

- **OpenAI。** `parallel_tool_calls: true` 默认开启。设为 `false` 强制串行。
- **Anthropic。** 通过 `disable_parallel_tool_use: false` 启用 parallel（Claude 3.5 及以上默认）。设为 `true` 串行。
- **Gemini。** 始终具备 parallel 能力；`tool_config.function_calling_config.mode = "AUTO"` 让模型决定。

当工具有顺序依赖（`create_file` 再 `write_file`）、一个调用的输出会影响另一个调用的输入，或 rate limiter 承受不了 fan-out 时，禁用 parallel。

### Id correlation

模型输出的每个 call 都有一个 `id`。宿主返回的每个 result 都必须包含同一个 id。没有它，结果就会含糊。

- **OpenAI。** 每条 tool-role message 上的 `tool_call_id`。
- **Anthropic。** 每个 `tool_result` block 上的 `tool_use_id`。
- **Gemini。** 每个 `functionResponse` 上的 `id`（Gemini 3 及以上；Gemini 2 按名称匹配，同名 parallel call 会坏）。

### 并发运行调用

宿主用独立线程、coroutine 或 remote worker 运行每个 call 的 executor。最简单的 harness 用 thread pool；生产中通常用 asyncio 的 `asyncio.gather` 或 structured concurrency。完成顺序不可预测——id 才是标识符。

一个常见 bug：按 call-list 顺序回复结果，而不是按完成顺序。这通常能工作，因为模型只关心 `tool_call_id`，但如果某个结果被丢弃或重复，乱序提交会让调试更难。更推荐按完成顺序回复，并显式携带 id。

### Streaming tool calls

模型 streaming 时，`arguments` 会分片到达。三个 parallel call 的三条 chunk stream 会在线路上交错。你需要每个 id 一个 accumulator。

provider 形状如下：

- **OpenAI。** 每个 chunk 是 `choices[0].delta.tool_calls[i].function.arguments`（partial string）。chunk 携带 `index`（在 call list 中的位置）。你按 index 累积，在 id 第一次出现时读取它，并在 `finish_reason = "tool_calls"` 时解析 JSON。
- **Anthropic。** Stream event 是 `message_start`，然后每个 block 有一个 `content_block_start`，type 为 `tool_use`（包含 id、name、空 input）。`content_block_delta` event 携带 `input_json_delta` chunk。`content_block_stop` 关闭每个 block。
- **Gemini。** `streamFunctionCallArguments`（Gemini 3 及以上）会发出带 `functionCallId` 的 chunk，因此调用可以干净交错。Gemini 3 之前，streaming 一次返回一个完整 call。

### Partial JSON 和 parse-early trap

在 `arguments` 完整前不能解析它。像 `{"city": "Beng` 这样的 partial JSON 无效，会抛错。正确的 gate 是 provider 的 end-of-call 信号：OpenAI 的 `finish_reason = "tool_calls"`、Anthropic 的 `content_block_stop`、Gemini 的 stream-end event。只有那时才尝试 `json.loads`。更稳健的做法是使用增量 JSON parser，在结构完成时 yield event；OpenAI streaming guide 推荐这用于显示实时 “thinking” indicator 的 UX。用括号计数判断完整性并不可靠（quoted string 或 escaped content 中的括号会误报），只能当作非正式 debug heuristic。

### Out-of-order completion

```
call_A: fast API, returns first
call_B: slow API, returns second
call_C: median API, returns third
```

宿主回复仍然必须引用 id：

```
[{role: "tool", tool_call_id: "call_A", content: ...},
 {role: "tool", tool_call_id: "call_B", content: ...},
 {role: "tool", tool_call_id: "call_C", content: ...}]
```

对 OpenAI 或 Anthropic 来说，回复顺序不影响正确性。Gemini 也接受任意顺序，只要 id 匹配。

### Benchmark：sequential vs parallel

`code/main.py` 中的 harness 模拟三个 executor，latency 分别是 400、600 和 800 ms。Sequential 总共 1800 ms。Parallel 是 max(400, 600, 800) = 800 ms。差值是常数，不是比例，因此工具数量越多收益越大。

现实 caveat：parallel calls 会给下游 API 施压。对 rate-limited 服务做 10-way fan-out 会失败。阶段 13 · 17 会讲 gateway-level backpressure；retry semantics 计划在未来阶段覆盖。

### Streaming fan-out wall-clock

如果模型本身在 streaming，你可以在某个 call 的 arguments 完整后立即开始执行，而不是等待所有调用 finalized。这是 OpenAI 文档中提到的优化，但并非所有 SDK 都暴露。本课 harness 会这么做：只要模拟 stream 产出完整 argument object，宿主就启动对应调用。

## 使用它

`code/main.py` 有两部分。第一部分用 `concurrent.futures.ThreadPoolExecutor` 顺序和并行运行三个模拟天气调用，并打印 wall-clock time。第二部分 replay 一个 fake streaming response——三个 parallel call 的 `arguments` chunk 在同一条 stream 上交错——并用 `StreamAccumulator` 按 id 重组。没有 LLM，没有网络，只有 reassembly logic。

重点看：

- sequential timer 达到 1.8 秒。同样 fake latency 下，parallel timer 达到 0.8 秒。
- accumulator 通过按 id buffer 并且只在每个 call 的 JSON 完整时解析，处理 out-of-order chunk。
- executor 会在某个 id 的 arguments finalize 后立即启动，而不是等所有 stream 结束。

## 交付它

本课产出 `outputs/skill-parallel-call-safety-check.md`。给定一个工具注册表，这个 skill 会审计哪些工具可以安全 parallelize，哪些有 ordering dependencies，哪些会压垮下游 rate limits，并返回带有 per-tool `parallel_safe` flag 的修订 registry。

## 练习

1. 运行 `code/main.py` 并改变模拟 latency。确认 parallel-to-sequential ratio 近似 `max/sum`（真实运行会因为线程调度、序列化和 harness overhead 与理想值略有偏差）。什么 latency 分布下 parallel 不再重要？

2. 扩展 accumulator 来处理 “call was cancelled mid-stream” 情况：丢弃它的 buffer 并发出 `cancelled` event。哪个 provider 明确记录了这种情况？检查 Anthropic 的 `content_block_stop` 语义和 OpenAI 的 `finish_reason: "length"` 行为。

3. 用 `asyncio.gather` 替换 thread pool。benchmark 两者。只有当 executor 做真实 I/O 时，async 才会因为较低 context-switch cost 看到小幅收益。

4. 选择两个不应该 parallelize 的工具（例如 `create_file` 再 `write_file`）。给 registry 添加 `ordering_dependency` graph，并用它 gate parallel fan-out。这是 dependency-aware scheduling 的最小机制，未来的 agent-engineering 阶段会将其形式化。

5. 阅读 OpenAI 的 parallel-function-calling 章节和 Anthropic 的 `disable_parallel_tool_use` 文档。找出 Anthropic 建议禁用 parallelism 的一种真实世界工具类型。（提示：同一资源上的 consequential mutations。）

## 关键词

| Term | 大家常说 | 实际含义 |
|------|----------|----------|
| Parallel tool calls | “一个 turn 里 fan-out” | 模型在单个 assistant message 中输出多个 tool call |
| `parallel_tool_calls` | “OpenAI 的 flag” | 启用或禁用 multi-call emission |
| `disable_parallel_tool_use` | “Anthropic 的反向 flag” | opt-out flag；默认启用 parallel |
| Tool call id | “Correlation handle” | result message 必须回显的 per-call 标识符 |
| Accumulator | “Stream buffer” | 存放 partial `arguments` chunk 的 per-id string buffer |
| Out-of-order completion | “最快先回” | parallel call 完成顺序不可预测；id 是胶水 |
| Dependency graph | “顺序约束” | 输出会喂给其他工具输入的工具；不能 parallelize |
| Parse-early trap | “JSON.parse 炸了” | 尝试解析不完整的 `arguments` string |
| `streamFunctionCallArguments` | “Gemini 3 feature” | 每个 call 带 unique id 的 streamed argument chunks |
| Completion-order reply | “不要等全部” | 结果到达就按 id 回复 |

## 延伸阅读

- [OpenAI — Parallel function calling](https://platform.openai.com/docs/guides/function-calling#parallel-function-calling) — 默认行为和 opt-out flag
- [Anthropic — Tool use: implementing tool use](https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/implementing-tool-use) — `disable_parallel_tool_use` 和 result batching
- [Google — Gemini function calling parallel section](https://ai.google.dev/gemini-api/docs/function-calling) — Gemini 3 起基于 id 关联的 parallel calls
- [OpenAI — Streaming responses with tools](https://platform.openai.com/docs/api-reference/responses-streaming) — OpenAI stream 的 chunked argument reassembly
- [Anthropic — Streaming messages](https://docs.anthropic.com/en/api/messages-streaming) — 带 `input_json_delta` 的 `content_block_delta`
