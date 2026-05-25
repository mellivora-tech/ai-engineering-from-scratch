# Structured Outputs：JSON、Schema Validation、Constrained Decoding

> LLM 返回的是字符串。你的应用需要 JSON。这个缺口让崩溃的生产系统比任何模型幻觉都多。Structured output 是自然语言与 typed data 之间的桥。做对了，LLM 就像可靠 API。做错了，你凌晨 3 点会用 regex 解析 free-text。

**类型：** 构建
**语言：** Python
**前置要求：** 阶段 10，第 01-05 课（LLMs from Scratch）
**时间：** ~90 分钟
**相关：** 阶段 5 · 20（Structured Outputs & Constrained Decoding）讲 decoder-level theory（FSM/CFG logit processors、Outlines、XGrammar）。本课关注生产 SDK surface（OpenAI `response_format`、Anthropic tool use、Instructor）；如果你想理解 API 下方发生了什么，先读阶段 5 · 20。

## 学习目标

- 使用 OpenAI 和 Anthropic API 参数实现 JSON-mode 和 schema-constrained outputs
- 构建 Pydantic validation layer，拒绝 malformed LLM outputs，并通过 error feedback retry
- 解释 constrained decoding 如何在 token level 强制 valid JSON，而不依赖 post-processing
- 设计健壮 extraction prompts，把 unstructured text 可靠转换成 typed data structures

## 问题

你问 LLM：“Extract the product name, price, and availability from this text.” 它回答：

```
The product is the Sony WH-1000XM5 headphones, which cost $348.00 and are currently in stock.
```

这是完全正确的答案。但对你的应用完全没用。Inventory system 需要 `{"product": "Sony WH-1000XM5", "price": 348.00, "in_stock": true}`。你需要带有特定 keys、特定 types、特定 value constraints 的 JSON object。你不需要一个句子。

天真的做法是给 prompt 加一句 “Respond in JSON”。这 90% 时间有效。剩下 10% 模型会把 JSON 包在 markdown code fences 中，或加一句 “Here's the JSON:”，或因为提前关 bracket 生成 syntactically invalid JSON。JSON parser 崩。Pipeline 断。你加 try/except 和 retry loop。Retry 有时产生不同数据。现在你在 parsing problem 上又叠了 consistency problem。

这不是 prompt engineering problem，而是 decoding problem。模型从左到右生成 tokens。每个位置，它从 100K+ vocabulary options 中选择最可能的下一个 token。任意位置大部分 options 都会产生 invalid JSON。如果模型刚输出 `{"price":`，下一个 token 必须是 digit、quote（string）、`null`、`true`、`false` 或 negative sign。其他任何东西都会让 JSON invalid。没有 constraints，模型可能选择一个合理英文词，但语法上灾难性错误。

## 概念

### Structured Output Spectrum

Structured output control 有四个层级，越往后越可靠。

```mermaid
graph LR
    subgraph Spectrum["Structured Output Spectrum"]
        direction LR
        A["Prompt-based\n'Return JSON'\n~90% valid"] --> B["JSON Mode\nGuaranteed valid JSON\nNo schema guarantee"]
        B --> C["Schema Mode\nJSON + matches schema\nGuaranteed compliance"]
        C --> D["Constrained Decoding\nToken-level enforcement\n100% compliance"]
    end

    style A fill:#1a1a2e,stroke:#ff6b6b,color:#fff
    style B fill:#1a1a2e,stroke:#ffa500,color:#fff
    style C fill:#1a1a2e,stroke:#51cf66,color:#fff
    style D fill:#1a1a2e,stroke:#0f3460,color:#fff
```

**Prompt-based**（“Respond in valid JSON”）：没有 enforcement。模型通常遵守，但有时不遵守。可靠性约 90%。Failure modes：markdown fences、preamble text、truncated output、wrong structure。

**JSON mode**：API 保证输出是 valid JSON。OpenAI 的 `response_format: { type: "json_object" }` 启用它。输出一定能 parse，但可能不匹配你的 expected schema：extra keys、wrong types、missing fields。

**Schema mode**：API 接收 JSON Schema，并保证输出匹配它。2026 年每个主要 provider 都原生支持：OpenAI 的 `response_format: { type: "json_schema", json_schema: {...} }`（也可用 `tool_choice="required"`）、Anthropic 带 `input_schema` 的 tool use、Gemini 的 `response_schema` + `response_mime_type: "application/json"`。输出有你指定的准确 keys、types 和 constraints。

**Constrained decoding**：generation 时每个 token position，decoder 都 mask 掉会产生 invalid output 的 tokens。如果 schema 要求 number，而模型准备输出 letter，该 token 概率被设为零。模型只能生成会通向 valid output 的 tokens。这就是 OpenAI structured output mode 和 Outlines、Guidance 这类库的底层做法。

### JSON Schema：Contract Language

JSON Schema 是你告诉模型（或 validation layer）输出必须是什么 shape 的方式。所有主流 structured output system 都使用它。

```json
{
  "type": "object",
  "properties": {
    "product": { "type": "string" },
    "price": { "type": "number", "minimum": 0 },
    "in_stock": { "type": "boolean" },
    "categories": {
      "type": "array",
      "items": { "type": "string" }
    }
  },
  "required": ["product", "price", "in_stock"]
}
```

这个 schema 表示：输出必须是 object，包含 string `product`、non-negative number `price`、boolean `in_stock`，以及可选 string array `categories`。不匹配的任何输出都会被 reject。

Schemas 能处理难点：nested objects、带 typed items 的 arrays、enums（把 string 约束到特定 values）、pattern matching（strings 上的 regex）以及 combinators（oneOf、anyOf、allOf 用于 polymorphic outputs）。

### Pydantic Pattern

在 Python 中，你不会手写 JSON Schema。你定义 Pydantic model，让它为你生成 schema。

```python
from pydantic import BaseModel

class Product(BaseModel):
    product: str
    price: float
    in_stock: bool
    categories: list[str] = []
```

这会产生与上面相同的 JSON Schema。Instructor library（以及 OpenAI SDK）可以直接接收 Pydantic models：传入 model class，拿回 validated instance。如果 LLM output 不匹配，Instructor 会自动 retry。

### Function Calling / Tool Use

这是同一个问题的另一种接口。你不是要求模型直接生成 JSON，而是定义带 typed parameters 的 “tools”（functions）。模型输出一个带 structured arguments 的 function call。OpenAI 称为 “function calling”。Anthropic 称为 “tool use”。结果相同：structured data。

```mermaid
graph TD
    subgraph ToolUse["Tool Use Flow"]
        U["User: Extract product info\nfrom this review text"] --> M["Model processes input"]
        M --> TC["Tool Call:\nextract_product(\n  product='Sony WH-1000XM5',\n  price=348.00,\n  in_stock=true\n)"]
        TC --> V["Validate against\nfunction schema"]
        V --> R["Structured Result:\n{product, price, in_stock}"]
    end

    style U fill:#1a1a2e,stroke:#0f3460,color:#fff
    style TC fill:#1a1a2e,stroke:#e94560,color:#fff
    style V fill:#1a1a2e,stroke:#ffa500,color:#fff
    style R fill:#1a1a2e,stroke:#51cf66,color:#fff
```

当模型需要选择调用哪个 function，而不仅是填参数时，tool use 更合适。如果你有 10 种 extraction schemas，并且模型必须根据 input 选择正确 schema，tool use 同时给你 schema selection 和 structured output。

### Common Failure Modes

即便有 schema enforcement，structured outputs 仍会以微妙方式失败。

**Hallucinated values**：输出匹配 schema，但包含编造数据。文本说 $348，模型输出 `{"price": 299.99}`。Schema validation 抓不到，因为 type 对，value 错。

**Enum confusion**：你把字段约束为 `["in_stock", "out_of_stock", "preorder"]`。模型输出 `"available"`，语义正确但不在 allowed set。好的 constrained decoding 会防止这个，prompt-based 做不到。

**Nested object depth**：深层嵌套 schemas（4+ levels）更容易出错。每一层 nesting 都是模型可能丢失结构的位置。

**Array length**：模型可能在 array 中生成过多或过少 items。Schemas 支持 `minItems` 和 `maxItems`，但不是所有 provider 都在 decoding level enforce。

**Optional field omission**：模型省略技术上 optional、但对你的 use case 很重要的字段。即使数据有时缺失，也把它们设为 required，强迫模型显式输出 `null`。

## 构建它

### 第 1 步：JSON Schema Validator

从零构建 validator，检查 Python object 是否匹配 JSON Schema。这就是输出侧验证 compliance 的东西。

```python
import json

def validate_schema(data, schema):
    errors = []
    _validate(data, schema, "", errors)
    return errors

def _validate(data, schema, path, errors):
    schema_type = schema.get("type")

    if schema_type == "object":
        if not isinstance(data, dict):
            errors.append(f"{path}: expected object, got {type(data).__name__}")
            return
        for key in schema.get("required", []):
            if key not in data:
                errors.append(f"{path}.{key}: required field missing")
        properties = schema.get("properties", {})
        for key, value in data.items():
            if key in properties:
                _validate(value, properties[key], f"{path}.{key}", errors)

    elif schema_type == "array":
        if not isinstance(data, list):
            errors.append(f"{path}: expected array, got {type(data).__name__}")
            return
        min_items = schema.get("minItems", 0)
        max_items = schema.get("maxItems", float("inf"))
        if len(data) < min_items:
            errors.append(f"{path}: array has {len(data)} items, minimum is {min_items}")
        if len(data) > max_items:
            errors.append(f"{path}: array has {len(data)} items, maximum is {max_items}")
        items_schema = schema.get("items", {})
        for i, item in enumerate(data):
            _validate(item, items_schema, f"{path}[{i}]", errors)

    elif schema_type == "string":
        if not isinstance(data, str):
            errors.append(f"{path}: expected string, got {type(data).__name__}")
            return
        enum_values = schema.get("enum")
        if enum_values and data not in enum_values:
            errors.append(f"{path}: '{data}' not in allowed values {enum_values}")

    elif schema_type == "number":
        if not isinstance(data, (int, float)):
            errors.append(f"{path}: expected number, got {type(data).__name__}")
            return
        minimum = schema.get("minimum")
        maximum = schema.get("maximum")
        if minimum is not None and data < minimum:
            errors.append(f"{path}: {data} is less than minimum {minimum}")
        if maximum is not None and data > maximum:
            errors.append(f"{path}: {data} is greater than maximum {maximum}")

    elif schema_type == "boolean":
        if not isinstance(data, bool):
            errors.append(f"{path}: expected boolean, got {type(data).__name__}")

    elif schema_type == "integer":
        if not isinstance(data, int) or isinstance(data, bool):
            errors.append(f"{path}: expected integer, got {type(data).__name__}")
```

### 第 2 步：Pydantic-Style Model to Schema

构建一个最小 class-to-schema converter。定义 Python class，并自动生成它的 JSON Schema。

```python
class SchemaField:
    def __init__(self, field_type, required=True, default=None, enum=None, minimum=None, maximum=None):
        self.field_type = field_type
        self.required = required
        self.default = default
        self.enum = enum
        self.minimum = minimum
        self.maximum = maximum

def python_type_to_schema(field):
    type_map = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
    }

    schema = {}

    if field.field_type in type_map:
        schema["type"] = type_map[field.field_type]
    elif field.field_type == list:
        schema["type"] = "array"
        schema["items"] = {"type": "string"}
    elif isinstance(field.field_type, dict):
        schema = field.field_type

    if field.enum:
        schema["enum"] = field.enum
    if field.minimum is not None:
        schema["minimum"] = field.minimum
    if field.maximum is not None:
        schema["maximum"] = field.maximum

    return schema

def model_to_schema(name, fields):
    properties = {}
    required = []

    for field_name, field in fields.items():
        properties[field_name] = python_type_to_schema(field)
        if field.required:
            required.append(field_name)

    return {
        "type": "object",
        "properties": properties,
        "required": required,
    }
```

### 第 3 步：Constrained Token Filter

模拟 constrained decoding。给定 partial JSON string 和 schema，判断当前位置哪些 token categories 是 valid。

```python
def next_valid_tokens(partial_json, schema):
    stripped = partial_json.strip()

    if not stripped:
        return ["{"]

    try:
        json.loads(stripped)
        return ["<EOS>"]
    except json.JSONDecodeError:
        pass

    last_char = stripped[-1] if stripped else ""

    if last_char == "{":
        return ['"', "}"]
    elif last_char == '"':
        if stripped.endswith('":'):
            return ['"', "0-9", "true", "false", "null", "[", "{"]
        return ["a-z", '"']
    elif last_char == ":":
        return [" ", '"', "0-9", "true", "false", "null", "[", "{"]
    elif last_char == ",":
        return [" ", '"', "{", "["]
    elif last_char in "0123456789":
        return ["0-9", ".", ",", "}", "]"]
    elif last_char == "}":
        return [",", "}", "]", "<EOS>"]
    elif last_char == "]":
        return [",", "}", "<EOS>"]
    elif last_char == "[":
        return ['"', "0-9", "true", "false", "null", "{", "[", "]"]
    else:
        return ["any"]

def demonstrate_constrained_decoding():
    partial_states = [
        '',
        '{',
        '{"product"',
        '{"product":',
        '{"product": "Sony"',
        '{"product": "Sony",',
        '{"product": "Sony", "price":',
        '{"product": "Sony", "price": 348',
        '{"product": "Sony", "price": 348}',
    ]

    print(f"{'Partial JSON':<45} {'Valid Next Tokens'}")
    print("-" * 80)
    for state in partial_states:
        valid = next_valid_tokens(state, {})
        display = state if state else "(empty)"
        print(f"{display:<45} {valid}")
```

### 第 4 步：Extraction Pipeline

把所有东西组合成 extraction pipeline：定义 schema，模拟 LLM 生成 structured output，validate output，并处理 retries。

```python
def simulate_llm_extraction(text, schema, attempt=0):
    if "headphones" in text.lower() or "sony" in text.lower():
        if attempt == 0:
            return '{"product": "Sony WH-1000XM5", "price": 348.00, "in_stock": true, "categories": ["audio", "headphones"]}'
        return '{"product": "Sony WH-1000XM5", "price": 348.00, "in_stock": true}'

    if "laptop" in text.lower():
        return '{"product": "MacBook Pro 16", "price": 2499.00, "in_stock": false, "categories": ["computers"]}'

    return '{"product": "Unknown", "price": 0, "in_stock": false}'

def extract_with_retry(text, schema, max_retries=3):
    for attempt in range(max_retries):
        raw = simulate_llm_extraction(text, schema, attempt)

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            print(f"  Attempt {attempt + 1}: JSON parse error -- {e}")
            continue

        errors = validate_schema(data, schema)
        if not errors:
            return data

        print(f"  Attempt {attempt + 1}: Schema validation errors -- {errors}")

    return None

product_schema = {
    "type": "object",
    "properties": {
        "product": {"type": "string"},
        "price": {"type": "number", "minimum": 0},
        "in_stock": {"type": "boolean"},
        "categories": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["product", "price", "in_stock"],
}
```

### 第 5 步：Run the Full Pipeline

```python
def run_demo():
    print("=" * 60)
    print("  Structured Output Pipeline Demo")
    print("=" * 60)

    print("\n--- Schema Definition ---")
    product_fields = {
        "product": SchemaField(str),
        "price": SchemaField(float, minimum=0),
        "in_stock": SchemaField(bool),
        "categories": SchemaField(list, required=False),
    }
    generated_schema = model_to_schema("Product", product_fields)
    print(json.dumps(generated_schema, indent=2))

    print("\n--- Schema Validation ---")
    test_cases = [
        ({"product": "Test", "price": 10.0, "in_stock": True}, "Valid object"),
        ({"product": "Test", "price": -5.0, "in_stock": True}, "Negative price"),
        ({"product": "Test", "in_stock": True}, "Missing price"),
        ({"product": "Test", "price": "ten", "in_stock": True}, "String as price"),
        ("not an object", "String instead of object"),
    ]

    for data, label in test_cases:
        errors = validate_schema(data, product_schema)
        status = "PASS" if not errors else f"FAIL: {errors}"
        print(f"  {label}: {status}")

    print("\n--- Constrained Decoding Simulation ---")
    demonstrate_constrained_decoding()

    print("\n--- Extraction Pipeline ---")
    texts = [
        "The Sony WH-1000XM5 headphones are priced at $348 and currently available.",
        "The new MacBook Pro 16-inch laptop costs $2499 but is sold out.",
        "This is a random sentence with no product info.",
    ]

    for text in texts:
        print(f"\n  Input: {text[:60]}...")
        result = extract_with_retry(text, product_schema)
        if result:
            print(f"  Output: {json.dumps(result)}")
        else:
            print(f"  Output: FAILED after retries")
```

## 使用它

### OpenAI Structured Outputs

```python
# from openai import OpenAI
# from pydantic import BaseModel
#
# client = OpenAI()
#
# class Product(BaseModel):
#     product: str
#     price: float
#     in_stock: bool
#
# response = client.beta.chat.completions.parse(
#     model="gpt-5-mini",
#     messages=[
#         {"role": "system", "content": "Extract product information."},
#         {"role": "user", "content": "Sony WH-1000XM5, $348, in stock"},
#     ],
#     response_format=Product,
# )
#
# product = response.choices[0].message.parsed
# print(product.product, product.price, product.in_stock)
```

OpenAI 的 structured output mode 在内部使用 constrained decoding。模型生成的每个 token 都保证能产生匹配 Pydantic schema 的 output。不需要 retries。不需要 validation。Constraint 被烘进 decoding process。

### Anthropic Tool Use

```python
# import anthropic
#
# client = anthropic.Anthropic()
#
# response = client.messages.create(
#     model="claude-opus-4-7",
#     max_tokens=1024,
#     tools=[{
#         "name": "extract_product",
#         "description": "Extract product information from text",
#         "input_schema": {
#             "type": "object",
#             "properties": {
#                 "product": {"type": "string"},
#                 "price": {"type": "number"},
#                 "in_stock": {"type": "boolean"},
#             },
#             "required": ["product", "price", "in_stock"],
#         },
#     }],
#     messages=[{"role": "user", "content": "Extract: Sony WH-1000XM5, $348, in stock"}],
# )
```

Anthropic 通过 tool use 实现 structured output。模型发出带 structured arguments 的 tool call，并匹配 `input_schema`。结果相同，只是 API surface 不同。

### Instructor Library

```python
# pip install instructor
# import instructor
# from openai import OpenAI
# from pydantic import BaseModel
#
# client = instructor.from_openai(OpenAI())
#
# class Product(BaseModel):
#     product: str
#     price: float
#     in_stock: bool
#
# product = client.chat.completions.create(
#     model="gpt-5-mini",
#     response_model=Product,
#     messages=[{"role": "user", "content": "Sony WH-1000XM5, $348, in stock"}],
# )
```

Instructor 包装任意 LLM client，并加入 validation + automatic retries。如果第一次尝试 validation 失败，它会把 errors 作为 context 发回模型，请它修正 output。这适用于任何 provider，不只 OpenAI。

## 交付它

本课产出 `outputs/prompt-structured-extractor.md`：一个 reusable prompt template，给定 schema definition，从任意文本中抽取 structured data。输入 JSON Schema 和 unstructured text，返回 validated JSON。

它还产出 `outputs/skill-structured-outputs.md`：一个 decision framework，用于根据 provider、reliability requirements 和 schema complexity 选择 structured output strategy。

## 练习

1. 扩展 schema validator，支持 `oneOf`（data 必须精确匹配多个 schemas 中的一个）。这能处理 polymorphic outputs，例如一个字段可以是形状不同的 `Product` 或 `Service` object。

2. 构建 “schema diff” tool，比较两个 schemas，并识别 breaking changes（移除 required fields、改变 types）与 non-breaking changes（添加 optional fields、放宽 constraints）。这是生产中 versioning extraction schemas 的必要能力。

3. 实现更真实的 constrained decoding simulator。给定 JSON Schema 和 100-token vocabulary（letters、digits、punctuation、keywords），逐步走过 generation，在每个位置 mask invalid tokens。测量每步 vocabulary 中 valid 的百分比。

4. 构建 extraction eval suite。创建 50 条 product descriptions，并手工标注 JSON outputs。对全部 50 条运行 extraction pipeline，测量 exact match、field-level accuracy 和 type compliance。找出哪些字段最难正确抽取。

5. 给 extraction pipeline 加 “confidence scores”。对每个 extracted field，估计模型 confidence（基于 token probabilities，或运行 3 次 extraction 并测一致性）。把 low-confidence fields 标记给 human review。

## 关键术语

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| JSON mode | “Returns JSON” | API flag，保证 syntactically valid JSON output，但不 enforce 特定 schema |
| Structured output | “Typed JSON” | 匹配特定 JSON Schema 的 output，带正确 keys、types 和 constraints |
| Constrained decoding | “Guided generation” | 在每个 token position mask 掉会产生 invalid output 的 tokens；保证 100% schema compliance |
| JSON Schema | “A JSON template” | 描述 JSON data structure、types 和 constraints 的声明式语言（OpenAPI、JSON Forms 等使用） |
| Pydantic | “Python dataclasses+” | 定义带 type validation 的 data models 的 Python 库，FastAPI 和 Instructor 用它生成 JSON Schemas |
| Function calling | “Tool use” | LLM 输出 structured function invocation（name + typed arguments），而不是 free text；OpenAI 和 Anthropic 都支持 |
| Instructor | “Pydantic for LLMs” | 包装 LLM clients 并返回 validated Pydantic instances 的 Python 库，validation failure 时自动 retry |
| Token masking | “Filtering the vocabulary” | Generation 时把特定 token 概率设为零，使模型不能产生它们 |
| Schema compliance | “Matches the shape” | Output 有所有 required fields、正确 types、constraints 内 values，且没有 disallowed extra fields |
| Retry loop | “Try again until it works” | 把 validation errors 发回模型并要求修复 output；Instructor 会自动执行，直到 configurable max |

## 延伸阅读

- [OpenAI Structured Outputs Guide](https://platform.openai.com/docs/guides/structured-outputs) -- OpenAI API 中基于 JSON Schema 的 constrained decoding 官方文档
- [Willard & Louf, 2023 -- "Efficient Guided Generation for Large Language Models"](https://arxiv.org/abs/2307.09702) -- Outlines 论文，描述如何把 JSON Schemas 编译成 finite state machines 以实现 token-level constraints
- [Instructor documentation](https://python.useinstructor.com/) -- 用 Pydantic validation 和 retries 从任意 LLM 获取 structured outputs 的标准库
- [Anthropic Tool Use Guide](https://docs.anthropic.com/en/docs/tool-use) -- Claude 如何用 JSON Schema `input_schema` 通过 tool use 实现 structured output
- [JSON Schema specification](https://json-schema.org/) -- 每个主流 structured output system 使用的 schema language 完整 spec
- [Outlines library](https://github.com/outlines-dev/outlines) -- 用 regex 和编译为 finite state machines 的 JSON Schema 进行 open-source constrained generation
- [Dong et al., "XGrammar: Flexible and Efficient Structured Generation Engine for Large Language Models" (MLSys 2025)](https://arxiv.org/abs/2411.15100) -- 当前 state-of-the-art grammar engine；pushdown-automaton compilation，以约 100 ns / token mask tokens。
- [Beurer-Kellner et al., "Prompting Is Programming: A Query Language for Large Language Models" (LMQL)](https://arxiv.org/abs/2212.06094) -- LMQL 论文，把 constrained decoding 表述为带 type 和 value constraints 的 query language。
- [Microsoft Guidance (framework docs)](https://github.com/guidance-ai/guidance) -- template-driven constrained generation；Outlines 和 XGrammar 的 vendor-agnostic 补充。
