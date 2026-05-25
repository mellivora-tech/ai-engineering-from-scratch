# Structured Output：JSON Schema、Pydantic、Zod、Constrained Decoding

> “好好要求模型返回 JSON” 即使在 frontier model 上也会有 5% 到 15% 的失败率。Structured outputs 用 constrained decoding 补上这个缺口：模型从字面上被阻止输出任何会违反 schema 的 token。OpenAI 的 strict mode、Anthropic 的 schema-typed tool use、Gemini 的 `responseSchema`、Pydantic AI 的 `output_type`、Zod 的 `.parse`，都是同一个想法的五种表面形态。本课会构建 schema validator 和 strict-mode contract，后续每条生产 extraction pipeline 都会用到它们。

**类型：** 构建
**语言：** Python（stdlib，JSON Schema 2020-12 子集）
**前置要求：** 阶段 13 · 02（function calling deep dive）
**时间：** ~75 分钟

## 学习目标

- 使用正确约束（enum、min/max、required、pattern）为 extraction target 编写 JSON Schema 2020-12。
- 解释 strict mode 和 constrained decoding 为什么提供了不同于 “validate after generation” 的保证。
- 区分三种失败模式：parse error、schema violation、model refusal。
- 交付一条带 typed repair 和 typed refusal handling 的 extraction pipeline。

## 问题

一个 agent 读取 purchase-order email 时，需要把自由文本变成 `{customer, line_items, total_usd}`。有三种方法。

**方法一：prompt for JSON。** “用 JSON 回复，字段包括 customer、line_items、total_usd。” 在 frontier model 上 85% 到 95% 的时间可用。会以六种方式失败：缺少花括号、尾随逗号、类型错误、幻觉字段、在 token limit 截断、泄漏 “Here is your JSON:” 这样的 prose。

**方法二：validate after generation。** 自由生成，parse，根据 schema validate，失败后 retry。可靠但昂贵——每次 retry 都要付费，truncation bug 每次都会多花一个 turn。

**方法三：constrained decoding。** provider 在 decode time 强制 schema。无效 token 会从 sampling distribution 中被 mask 掉。输出保证能 parse，也保证能 validate。失败收敛为一种模式：refusal（模型判断输入不适合该 schema）。

2026 年每个 frontier provider 都提供了某种形式的方法三。

- **OpenAI。** `response_format: {type: "json_schema", strict: true}`，如果模型拒绝，响应中会有 `refusal`。
- **Anthropic。** 在 `tool_use` input 上执行 schema；没有 `stop_reason: "refusal"` 这种东西，但 `end_turn` 且无 tool call 就是信号。
- **Gemini。** request level 的 `responseSchema`；2026 年 Gemini 对部分类型提供 token-level grammar constraints。
- **Pydantic AI。** `output_type=InvoiceModel` 输出一个带 `InvoiceModel` 类型的 structured `RunResult`。
- **Zod（TypeScript）。** runtime parser，根据 Zod schema 验证 provider output；可与 OpenAI 的 `beta.chat.completions.parse` 搭配。

共同主线是：schema 只声明一次，并端到端执行。

## 概念

### JSON Schema 2020-12：通用语

每个 provider 都接受 JSON Schema 2020-12。最常用的构造：

- `type`：`object`、`array`、`string`、`number`、`integer`、`boolean`、`null` 之一。
- `properties`：字段名到 subschema 的映射。
- `required`：必须出现的字段名列表。
- `enum`：封闭的允许值集合。
- `minimum` / `maximum`（数字），`minLength` / `maxLength` / `pattern`（字符串）。
- `items`：应用到每个 array element 的 subschema。
- `additionalProperties`：`false` 禁止额外字段（默认值因模式而异）。

OpenAI strict mode 额外要求三件事：每个 property 都必须列在 `required` 中，到处都要 `additionalProperties: false`，并且没有未解析的 `$ref`。如果违反这些，API 会在 request time 返回 400。

### Pydantic：Python binding

Pydantic v2 通过 `model_json_schema()` 从 dataclass-shaped model 生成 JSON Schema。Pydantic AI 包装了这一点，所以你可以写：

```python
class Invoice(BaseModel):
    customer: str
    line_items: list[LineItem]
    total_usd: Decimal
```

agent framework 会在边界处把 schema 翻译成 OpenAI strict mode、Anthropic `input_schema` 或 Gemini `responseSchema`。模型输出会以 typed `Invoice` instance 返回。验证错误会抛出带 typed error path 的 `ValidationError`。

### Zod：TypeScript binding

Zod（`z.object({customer: z.string(), ...})`）是 TS 对等物。OpenAI 的 Node SDK 暴露 `zodResponseFormat(Invoice)`，会翻译成 API 的 JSON Schema payload。

### Refusals

Strict mode 不能强迫模型回答。如果输入无法适配 schema（“这封 email 是诗，不是 invoice”），模型会输出一个包含原因的 `refusal` 字段。你的代码必须把它当作一等 outcome，而不是失败。refusal 也是有用的安全信号：当模型被要求从受保护内容 email 中抽取信用卡号时，它会返回附带 safety reason 的 refusal。

### 开放环境中的 constrained decoding

open-weights 实现使用三种技术。

1. **Grammar-based decoding**（`outlines`、`guidance`、`lm-format-enforcer`）：从 schema 构建 deterministic finite automaton；每一步 mask 掉会违反 FSM 的 token logits。
2. **Logit masking with a JSON parser**：让 streaming JSON parser 与模型同步运行；每一步计算合法 next-token 集合。
3. **Speculative decoding with a verifier**：便宜的 draft model 提议 token，verifier 执行 schema。

商业 provider 会在幕后选择其中一种。2026 年最先进的状态是：短 structured output 比普通生成更快，长输出速度大致相同。

### 三种失败模式

1. **Parse error。** 输出不是有效 JSON。strict mode 下不可能发生。non-strict provider 仍可能发生。
2. **Schema violation。** 输出可以 parse，但违反 schema。strict mode 下不可能发生。在 strict 之外很常见。
3. **Refusal。** 模型拒绝。必须作为 typed outcome 处理。

### Retry strategy

当你不在 strict mode 中（Anthropic tool use、non-strict OpenAI、旧 Gemini），恢复模式是：

```
generate -> parse -> validate -> if fail, inject error and retry, max 3x
```

通常一次 retry 就够。三次 retry 能兜住弱模型抖动。超过三次说明 schema 有问题：模型无法在某些输入上满足它，prompt 或 schema 需要修。

### 小模型支持

Constrained decoding 对小模型也有效。一个带 grammar enforcement 的 3B 参数 open model，在 structured task 上会胜过一个 raw prompting 的 70B 参数模型。这是 structured outputs 对生产如此重要的主要原因：它把可靠性和模型大小解耦。

## 使用它

`code/main.py` 提供一个只用 stdlib 写的最小 JSON Schema 2020-12 validator（types、required、enum、min/max、pattern、items、additionalProperties）。它包装一个 `Invoice` schema，并让 fake LLM output 经过 validator，演示 parse error、schema violation 和 refusal path。在生产中，把 fake output 换成任意 provider 的真实 response。

重点看：

- validator 返回带 path 和 message 的 typed `[ValidationError]` list。这正是你想暴露给 retry prompt 的形状。
- refusal branch 不会 retry。它记录日志并返回 typed refusal。阶段 14 · 09 会把 refusals 用作 safety signal。
- `additionalProperties: false` 检查会在 adversarial test input 上触发，展示为什么 strict mode 能关上 hallucinated fields 的门。

## 交付它

本课产出 `outputs/skill-structured-output-designer.md`。给定一个 free-text extraction target（invoice、support ticket、resume 等），这个 skill 会生成一个 strict-mode-compatible 的 JSON Schema 2020-12，以及与之镜像的 Pydantic model，并 stub 出 typed refusal 和 retry handling。

## 练习

1. 运行 `code/main.py`。添加第四个 test case，让 `total_usd` 是负数。确认 validator 用 `minimum` 约束路径拒绝它。

2. 扩展 validator，支持带 discriminator 的 `oneOf`。常见场景：`line_item` 要么是 product，要么是 service，并由 `kind` 标记。Strict mode 在这里有微妙规则；查看 OpenAI 的 structured outputs guide。

3. 把同一个 Invoice schema 写成 Pydantic BaseModel，并比较 `model_json_schema()` 输出和手写 schema。找出 Pydantic 默认设置、但手写版本省略的一个字段。

4. 测量 refusal rates。构造十个不应该可抽取的输入（一段歌词、一个数学证明、一封空 email），用 strict mode 跑过真实 provider。统计 refusals vs hallucinated outputs。这是你做 refusal-aware retry 的 ground truth。

5. 从头到尾阅读 OpenAI 的 structured outputs guide。找出一个 strict mode 明确禁止、但普通 JSON Schema 允许的构造。然后设计一个非必要地使用该禁止构造的 schema，并重构成 strict-compatible。

## 关键词

| Term | 大家常说 | 实际含义 |
|------|----------|----------|
| JSON Schema 2020-12 | “schema spec” | 每个现代 provider 都使用的 IETF-draft schema dialect |
| Strict mode | “Guaranteed schema” | OpenAI flag，通过 constrained decoding 强制 schema |
| Constrained decoding | “Logit masking” | decode-time enforcement，mask 无效 next-token |
| Refusal | “模型拒绝” | 输入无法适配 schema 时的 typed outcome |
| Parse error | “Invalid JSON” | 输出无法解析为 JSON；strict 下不可能 |
| Schema violation | “形状错了” | 能 parse，但违反 type / required / enum / range |
| `additionalProperties: false` | “不允许额外字段” | 禁止未知字段；OpenAI strict 必需 |
| Pydantic BaseModel | “Typed output” | 生成并验证 JSON Schema 的 Python class |
| Zod schema | “TypeScript output type” | 用于 provider output validation 的 TS runtime schema |
| Grammar enforcement | “Open-weights constrained decode” | 基于 FSM 的 logit masking，例如 outlines / guidance |

## 延伸阅读

- [OpenAI — Structured outputs](https://platform.openai.com/docs/guides/structured-outputs) — strict mode、refusals 和 schema requirements
- [OpenAI — Introducing structured outputs](https://openai.com/index/introducing-structured-outputs-in-the-api/) — 2024 年 8 月发布文章，解释 decoding guarantee
- [Pydantic AI — Output](https://ai.pydantic.dev/output/) — 会序列化到各 provider 的 typed output_type bindings
- [JSON Schema — 2020-12 release notes](https://json-schema.org/draft/2020-12/release-notes) — 权威 spec
- [Microsoft — Structured outputs in Azure OpenAI](https://learn.microsoft.com/en-us/azure/foundry/openai/how-to/structured-outputs) — 企业部署说明和 strict-mode caveats
