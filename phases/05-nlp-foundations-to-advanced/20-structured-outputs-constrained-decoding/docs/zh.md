# 结构化输出与 Constrained Decoding

> 让 LLM 返回 JSON。大多数时候你会得到 JSON。在生产中，“大多数”就是问题。Constrained decoding 通过在 sampling 前修改 logits，把“大多数”变成“总是”。

**类型：** 构建
**语言：** Python
**前置要求：** 阶段 5 · 17（Chatbots），阶段 5 · 19（Subword Tokenization）
**时间：** ~60 分钟

## 问题

一个 classifier prompt LLM：“Return one of {positive, negative, neutral}.” 模型返回：“The sentiment is positive — this review is overwhelmingly favorable because the customer explicitly states that they ...”。你的 parser 崩了。Classifier 的 F1 变成 0.0。

Free-form generation 不是契约，只是建议。生产系统需要契约。

2026 年有三层方案。

1. **Prompting。** 好好请求。“Return only the JSON object.” 对 frontier models 大约 80% 有效，对小模型更差。
2. **Native structured output APIs。** OpenAI `response_format`、Anthropic tool use、Gemini JSON mode。对支持的 schema 可靠。Vendor-locked。
3. **Constrained decoding。** 在每个生成步骤修改 logits，让模型无法发出 invalid tokens。构造上 100% 有效。适用于任何 local model。

本课建立三者直觉，并说明什么时候该用哪一个。

## 概念

![Constrained decoding masking invalid tokens at each step](../assets/constrained-decoding.svg)

**Constrained decoding 如何工作。** 在每个生成步骤，LLM 会产生一个覆盖完整 vocabulary（~100k tokens）的 logit vector。一个 *logit processor* 坐在模型和 sampler 之间。它根据目标 grammar 中当前位置（JSON Schema、regex、context-free grammar）计算哪些 tokens 有效，并把所有 invalid tokens 的 logits 设为负无穷。剩余 logits 的 softmax 只会给 valid continuations 分配概率质量。

2026 年实现：

- **Outlines。** 把 JSON Schema 或 regex 编译成 finite-state machine。每个 token 都能 O(1) 查 valid-next-token。基于 FSM，所以 recursive schemas 需要 flattening。
- **XGrammar / llguidance。** Context-free grammar engines。处理 recursive JSON Schema。Decoding overhead 接近零。OpenAI 在 2025 年 structured output 实现中引用了 llguidance。
- **vLLM guided decoding。** 通过 Outlines、XGrammar 或 lm-format-enforcer backends 内置 `guided_json`、`guided_regex`、`guided_choice`、`guided_grammar`。
- **Instructor。** 基于 Pydantic 的任意 LLM wrapper。Validation failure 后 retry。跨 provider，但不修改 logits，依赖 retries + structured-output-aware prompts。

### 反直觉结果

Constrained decoding 往往比 unconstrained generation 更快。原因有两个。第一，它缩小了 next-token 搜索空间。第二，聪明实现会完全跳过 forced tokens 的生成（脚手架如 `{"name": "`，每个 byte 都已经确定）。

### 会让你付出代价的坑

字段顺序很重要。把 `answer` 放在 `reasoning` 前面，模型会在思考前先承诺答案。JSON 有效，但答案错了。没有 validation 能抓到。

```json
// BAD
{"answer": "yes", "reasoning": "because ..."}

// GOOD
{"reasoning": "... therefore ...", "answer": "yes"}
```

Schema field order 是逻辑，不只是格式。

## 构建它

### 第 1 步：从零做 regex-constrained generation

独立 FSM 实现见 `code/main.py`。30 行核心思路：

```python
def mask_logits(logits, valid_token_ids):
    mask = [float("-inf")] * len(logits)
    for tid in valid_token_ids:
        mask[tid] = logits[tid]
    return mask


def generate_constrained(model, tokenizer, prompt, fsm):
    ids = tokenizer.encode(prompt)
    state = fsm.initial_state
    while not fsm.is_accept(state):
        logits = model.next_token_logits(ids)
        valid = fsm.valid_tokens(state, tokenizer)
        logits = mask_logits(logits, valid)
        tok = sample(logits)
        ids.append(tok)
        state = fsm.transition(state, tok)
    return tokenizer.decode(ids)
```

FSM 跟踪我们已经满足了 grammar 的哪些部分。`valid_tokens(state, tokenizer)` 计算哪些 vocabulary tokens 能推进 FSM 且仍保留到达接受状态的路径。

### 第 2 步：Outlines for JSON Schema

```python
from pydantic import BaseModel
from typing import Literal
import outlines


class Review(BaseModel):
    sentiment: Literal["positive", "negative", "neutral"]
    confidence: float
    evidence_span: str


model = outlines.models.transformers("meta-llama/Llama-3.2-3B-Instruct")
generator = outlines.generate.json(model, Review)

result = generator("Classify: 'The wait staff was attentive and the food arrived hot.'")
print(result)
# Review(sentiment='positive', confidence=0.93, evidence_span='attentive ... hot')
```

Validation errors 永远为零。FSM 让 invalid output 不可达。

### 第 3 步：Instructor 做 provider-agnostic Pydantic

```python
import instructor
from anthropic import Anthropic
from pydantic import BaseModel, Field


class Invoice(BaseModel):
    vendor: str
    total_usd: float = Field(ge=0)
    line_items: list[str]


client = instructor.from_anthropic(Anthropic())
invoice = client.messages.create(
    model="claude-opus-4-7",
    max_tokens=1024,
    response_model=Invoice,
    messages=[{"role": "user", "content": "Extract from: 'Acme Corp $420. Widget, Gizmo.'"}],
)
```

机制不同。Instructor 不碰 logits。它把 schema 格式化进 prompt，解析输出，并在 validation failure 后 retry（默认 3 次）。适用于任何 provider。Retries 会增加延迟和成本。跨 provider 可移植性是它的卖点。

### 第 4 步：native vendor APIs

```python
from openai import OpenAI

client = OpenAI()
response = client.responses.create(
    model="gpt-5",
    input=[{"role": "user", "content": "Classify: 'The food was cold.'"}],
    text={"format": {"type": "json_schema", "name": "sentiment",
          "schema": {"type": "object", "required": ["sentiment"],
                     "properties": {"sentiment": {"type": "string",
                                                  "enum": ["positive", "negative", "neutral"]}}}}},
)
print(response.output_parsed)
```

Server-side constrained decoding。对支持的 schemas，与 Outlines 可靠性相当。不需要管理 local model。代价是锁定 vendor。

## 坑

- **Recursive schemas。** Outlines 会把 recursion flatten 到固定深度。Tree-structured outputs（nested comments、AST）需要 XGrammar 或 llguidance（CFG-based）。
- **巨大 enums。** 10,000-option enum 编译慢或超时。换成 retriever：先预测 top-k candidates，再 constrain 到这些。
- **Grammar 太严格。** 强制 `date: "YYYY-MM-DD"` regex，模型就无法为缺失日期输出 `"unknown"`。模型会通过编造日期来补偿。允许 `null` 或 sentinel。
- **过早承诺。** 见上面的字段顺序坑。始终把 reasoning 放前面。
- **Vendor JSON mode without schema。** 纯 JSON mode 只保证 JSON 有效，不保证对你的 use case 有效。始终提供完整 schema。

## 使用它

2026 年技术栈：

| 场景 | 选择 |
|-----------|------|
| OpenAI/Anthropic/Google model，简单 schema | Native vendor structured output |
| 任意 provider，Pydantic workflow，可容忍 retries | Instructor |
| Local model，需要 100% validity，flat schema | Outlines（FSM） |
| Local model，recursive schema | XGrammar 或 llguidance |
| Self-hosted inference server | vLLM guided decoding |
| Batch processing，retries 可接受 | Instructor + 最便宜模型 |

## 交付它

保存为 `outputs/skill-structured-output-picker.md`：

```markdown
---
name: structured-output-picker
description: 选择 structured output approach、schema design 和 validation plan。
version: 1.0.0
phase: 5
lesson: 20
tags: [nlp, llm, structured-output]
---

给定 use case（provider、latency budget、schema complexity、failure tolerance），输出：

1. Mechanism。Native vendor structured output、Instructor retries、Outlines FSM 或 XGrammar CFG。一句话说明理由。
2. Schema design。字段顺序（reasoning first, answer last）、"unknown" 的 nullable fields、enum vs regex、required fields。
3. Failure strategy。Max retries、fallback model、graceful `null` handling、out-of-distribution refusal。
4. Validation plan。Schema compliance rate（目标 100%）、semantic validity（LLM-judge）、field-coverage rate、latency p50/p99。

拒绝任何把 `answer` 或 `decision` 放在 reasoning fields 前面的设计。拒绝没有 schema 的 bare JSON mode。标记放在 FSM-only library 后面的 recursive schemas。
```

## 练习

1. **简单。** 在没有 constrained decoding 的情况下 prompt 一个小型 open-weights model（如 Llama-3.2-3B）输出 `Review(sentiment, confidence, evidence_span)`。测量 100 条 reviews 中能 parse 成 valid JSON 的比例。
2. **中等。** 同一 corpus 使用 Outlines JSON mode。比较 compliance rate、latency 和 semantic accuracy。
3. **困难。** 从零为电话号码（`\d{3}-\d{3}-\d{4}`）实现 regex-constrained decoder。在 1000 个 samples 上验证 0 invalid outputs。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|-----------------|-----------------------|
| Constrained decoding | 强制有效输出 | 在每个生成步骤 mask invalid-token logits。 |
| Logit processor | 约束的东西 | 函数：`(logits, state) -> masked_logits`。 |
| FSM | Finite-state machine | 编译后的 grammar representation；O(1) valid-next-token lookup。 |
| CFG | Context-free grammar | 能处理 recursion 的 grammar；比 FSM 慢但表达力更强。 |
| Schema field order | 这重要吗？ | 重要。第一个字段会承诺；始终把 reasoning 放在 answer 前面。 |
| Guided decoding | vLLM 的名字 | 同一概念，集成进 inference server。 |
| JSON mode | OpenAI 早期版本 | 保证 JSON 语法；不保证 schema match。 |

## 延伸阅读

- [Willard, Louf (2023). Efficient Guided Generation for LLMs](https://arxiv.org/abs/2307.09702) — Outlines 论文。
- [XGrammar paper (2024)](https://arxiv.org/abs/2411.15100) — 快速 CFG-based constrained decoding。
- [vLLM — Structured Outputs](https://docs.vllm.ai/en/latest/features/structured_outputs.html) — inference server 集成。
- [OpenAI — Structured Outputs guide](https://platform.openai.com/docs/guides/structured-outputs) — API reference + gotchas。
- [Instructor library](https://python.useinstructor.com/) — 跨 providers 的 Pydantic + retries。
- [JSONSchemaBench (2025)](https://arxiv.org/abs/2501.10868) — 6 个 constrained decoding frameworks 的 benchmark。
