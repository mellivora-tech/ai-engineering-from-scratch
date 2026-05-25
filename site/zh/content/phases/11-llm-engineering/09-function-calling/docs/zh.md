# Function Calling 与 Tool Use

> LLM 不能真正“做”任何事。它们生成文本。这就是全部能力。它们不能查天气、查询数据库、发送邮件、运行代码或读取文件。你见过的每个“AI agent”，本质上都是 LLM 生成一段 JSON，说应该调用哪个 function，然后你的代码真正去调用它。模型是大脑。工具是双手。Function calling 是连接它们的神经系统。

**类型：** 构建
**语言：** Python
**前置要求：** Phase 11 Lesson 03（Structured Outputs）
**时间：** 约 75 分钟
**相关：** Phase 11 · 14（Model Context Protocol），当一个 tool 需要跨 host 共享时，就应该从 inline function-calling 升级到 MCP server。本课覆盖 inline case，MCP 覆盖 protocol case。

## 学习目标

- 实现 function calling loop：定义 tool schemas、解析模型的 tool-call JSON、执行 functions，并返回结果
- 设计带清晰 descriptions 和 typed parameters 的 tool schemas，让模型可靠调用
- 构建 multi-turn agent loop，通过串联多个 function calls 回答复杂查询
- 处理 function calling 边界情况：parallel tool calls、error propagation，以及防止 infinite tool loops

## 问题

你构建了一个 chatbot。用户问：“What's the weather in Tokyo right now?”

模型回答：“I don't have access to real-time weather data, but based on the season, Tokyo is likely around 15 degrees Celsius...”

这是披着免责声明的幻觉。模型不知道天气。它永远也不会知道。天气每小时都在变化。模型训练数据已经是几个月前的。

正确答案需要调用 OpenWeatherMap API，获取当前温度，并返回真实数字。模型不能调用 API。你的代码可以。缺失的一环是一个结构化协议，让模型能说“我需要用这些参数调用 weather API”，然后让你的代码执行它，并把结果反馈回来。

这就是 function calling。模型输出结构化 JSON，描述要用哪些 arguments 调用哪个 function。你的 application 执行 function。结果回到 conversation。模型使用这个结果生成最终答案。

没有 function calling，LLM 是百科全书。有了它，它们才成为 agents。

## 概念

### Function Calling Loop

每个 tool-use interaction 都遵循同一个 5 步循环。

```mermaid
sequenceDiagram
    participant U as User
    participant A as Application
    participant M as Model
    participant T as Tool

    U->>A: "What's the weather in Tokyo?"
    A->>M: messages + tool definitions
    M->>A: tool_call: get_weather(city="Tokyo")
    A->>T: Execute get_weather("Tokyo")
    T->>A: {"temp": 18, "condition": "cloudy"}
    A->>M: tool_result + conversation
    M->>A: "It's 18C and cloudy in Tokyo."
    A->>U: Final response
```

Step 1：用户发送消息。Step 2：模型收到消息和 tool definitions（用 JSON Schema 描述可用 functions）。Step 3：模型不直接回复文本，而是输出 tool call，也就是包含 function name 和 arguments 的结构化 JSON object。Step 4：你的代码执行 function 并捕获结果。Step 5：结果回到模型，模型现在拥有真实数据，可以生成最终答案。

模型从不执行任何东西。它只决定调用什么，以及用什么 arguments 调用。你的代码才是 executor。

### Tool Definitions：JSON Schema Contract

每个 tool 都由一个 JSON Schema 定义，告诉模型 function 做什么、接受哪些 arguments、这些 arguments 必须是什么类型。

```json
{
  "type": "function",
  "function": {
    "name": "get_weather",
    "description": "Get current weather for a city. Returns temperature in Celsius and conditions.",
    "parameters": {
      "type": "object",
      "properties": {
        "city": {
          "type": "string",
          "description": "City name, e.g. 'Tokyo' or 'San Francisco'"
        },
        "units": {
          "type": "string",
          "enum": ["celsius", "fahrenheit"],
          "description": "Temperature units"
        }
      },
      "required": ["city"]
    }
  }
}
```

`description` 字段非常关键。模型会读取它们来决定何时以及如何使用 tool。像 “gets weather” 这样模糊的描述，会比 “Get current weather for a city. Returns temperature in Celsius and conditions.” 产生更差的 tool selection。Description 是用于 tool selection 的 prompt。

### Provider 对比

所有主流 provider 都支持 function calling，但 API surface 不同。

| Provider | API Parameter | Tool Call Format | Parallel Calls | Forced Calling |
|----------|--------------|-----------------|---------------|----------------|
| OpenAI (GPT-5, o4) | `tools` | `tool_calls[].function` | Yes（每轮多个） | `tool_choice="required"` |
| Anthropic (Claude 4.6/4.7) | `tools` | `content[].type="tool_use"` | Yes（多个 blocks） | `tool_choice={"type":"any"}` |
| Google (Gemini 3) | `function_declarations` | `functionCall` | Yes | `function_calling_config` |
| Open-weight (Llama 4, Qwen3, DeepSeek-V3) | Llama 4 原生 `tools`；其他使用 Hermes 或 ChatML | Mixed | 取决于模型 | Prompt-based 或支持时用 `tool_choice` |

到 2026 年，三个 closed providers 已经收敛到几乎相同的 JSON-Schema-based 格式。Llama 4 自带与 OpenAI 形状匹配的原生 `tools` 字段。Open-weight fine-tunes 仍然有差异，Hermes format（NousResearch）是第三方 fine-tunes 中最常见的格式。对于跨 hosts 共享的 tools，优先使用 MCP（Phase 11 · 14），而不是 inline function-calling，因为 server 对所有 host 都相同。

### Tool Choice：Auto、Required、Specific

你可以控制模型何时使用 tools。

**Auto**（默认）：模型决定是否调用 tool 或直接回答。“What's 2+2?” 会直接回答。“What's the weather?” 会调用 tool。

**Required**：模型必须至少调用一个 tool。当你知道用户意图需要 tool 时使用它。它能防止模型猜测，而不是查询真实数据。

**Specific function**：强制模型调用某个特定 function。`tool_choice={"type":"function", "function": {"name": "get_weather"}}` 保证调用 weather tool，无论 query 是什么。把它用于 routing，当上游逻辑已经确定需要哪个 tool 时。

### Parallel Function Calling

GPT-4o 和 Claude 可以在单轮中调用多个 functions。用户问：“What's the weather in Tokyo and New York?” 模型会同时输出两个 tool calls：

```json
[
  {"name": "get_weather", "arguments": {"city": "Tokyo"}},
  {"name": "get_weather", "arguments": {"city": "New York"}}
]
```

你的代码执行两者（最好并发），返回两个结果，模型再合成为一个回复。这会把 round trips 从 2 次降到 1 次。对于每个 query 有 5 到 10 次 tool calls 的 agents，parallel calling 能把延迟降低 60% 到 80%。

### Structured Outputs vs Function Calling

Lesson 03 覆盖了 structured outputs。Function calling 使用同一套 JSON Schema 机制，但目的不同。

**Structured outputs**：强制模型以特定 shape 生成数据。输出就是最终产物。示例：从文本中抽取 product info 为 `{name, price, in_stock}`。

**Function calling**：模型声明执行 action 的意图。输出是中间步骤。示例：`get_weather(city="Tokyo")`，模型是在请求 action，而不是生成最终答案。

当你想要 data extraction 时，使用 structured outputs。当你想让模型与外部系统交互时，使用 function calling。

### 安全：不可协商的规则

Function calling 是你能给 LLM 的最危险能力。模型选择执行什么。如果你的 tool set 包含 database queries，模型会构造 queries。如果包含 shell commands，模型会写 commands。

**规则 1：永远不要把模型生成的 SQL 直接传给数据库。** 模型能够而且会生成 DROP TABLE、UNION injections，或返回每一行的 queries。始终参数化。始终验证。始终使用 operation allowlist。

**规则 2：Allowlist functions。** 模型只能调用你显式定义的 functions。永远不要构建一个泛用的“按名称执行任意 function”工具。如果你有 50 个内部 functions，只暴露用户需要的 5 个。

**规则 3：验证 arguments。** 模型可能传入 city name 为 `"; DROP TABLE users; --"`。执行前对每个 argument 按预期类型、范围和格式验证。

**规则 4：清理 tool results。** 如果 tool 返回 sensitive data（API keys、PII、internal errors），先过滤再发回模型。模型会原样把 tool results 包含在响应中。

**规则 5：限制 tool calls 速率。** 循环中的模型可能调用工具数百次。设置最大值（每个 conversation 10 到 20 次比较合理）。打断 infinite loops。

### Error Handling

Tools 会失败。API 会超时。数据库会宕机。文件不存在。模型需要知道 tool 何时失败以及为什么失败。

以结构化 tool results 返回错误，而不是 exceptions：

```json
{
  "error": true,
  "message": "City 'Toky' not found. Did you mean 'Tokyo'?",
  "code": "CITY_NOT_FOUND"
}
```

模型会读取它，调整 arguments，然后重试。模型很擅长从结构化 error messages 中自我纠正。它们不擅长从空响应或泛泛的 “something went wrong” 中恢复。

### MCP：Model Context Protocol

MCP 是 Anthropic 面向 tool interoperability 的开放标准。不是每个 application 都定义自己的 tools，MCP 提供了通用协议：tools 由 MCP servers 提供，由 MCP clients（如 Claude Code、Cursor 或你的 application）消费。

一个 MCP server 可以把 tools 暴露给任何兼容 client。Postgres MCP server 给任何 MCP-compatible agent 数据库访问能力。GitHub MCP server 给任何 agent 仓库访问能力。Tools 只定义一次，到处使用。

MCP 之于 function calling，就像 HTTP 之于 networking。它标准化 transport layer，让 tools 变得可移植。

## 构建

### Step 1：定义 Tool Registry

构建一个 registry，存储 tool definitions 和它们的 implementations。每个 tool 都有一个 JSON Schema definition（模型看到的内容）和一个 Python function（你的代码执行的内容）。

```python
import json
import math
import time
import hashlib


TOOL_REGISTRY = {}


def register_tool(name, description, parameters, function):
    TOOL_REGISTRY[name] = {
        "definition": {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": parameters,
            },
        },
        "function": function,
    }
```

### Step 2：实现 5 个 Tools

构建 calculator、weather lookup、web search simulator、file reader 和 code runner。

```python
def calculator(expression, precision=2):
    allowed = set("0123456789+-*/.() ")
    if not all(c in allowed for c in expression):
        return {"error": True, "message": f"Invalid characters in expression: {expression}"}
    try:
        result = eval(expression, {"__builtins__": {}}, {"math": math})
        return {"result": round(float(result), precision), "expression": expression}
    except Exception as e:
        return {"error": True, "message": str(e)}


WEATHER_DB = {
    "tokyo": {"temp_c": 18, "condition": "cloudy", "humidity": 72, "wind_kph": 14},
    "new york": {"temp_c": 22, "condition": "sunny", "humidity": 45, "wind_kph": 8},
    "london": {"temp_c": 12, "condition": "rainy", "humidity": 88, "wind_kph": 22},
    "san francisco": {"temp_c": 16, "condition": "foggy", "humidity": 80, "wind_kph": 18},
    "sydney": {"temp_c": 25, "condition": "sunny", "humidity": 55, "wind_kph": 10},
}


def get_weather(city, units="celsius"):
    key = city.lower().strip()
    if key not in WEATHER_DB:
        suggestions = [c for c in WEATHER_DB if c.startswith(key[:3])]
        return {
            "error": True,
            "message": f"City '{city}' not found.",
            "suggestions": suggestions,
            "code": "CITY_NOT_FOUND",
        }
    data = WEATHER_DB[key].copy()
    if units == "fahrenheit":
        data["temp_f"] = round(data["temp_c"] * 9 / 5 + 32, 1)
        del data["temp_c"]
    data["city"] = city
    return data


SEARCH_DB = {
    "python function calling": [
        {"title": "OpenAI Function Calling Guide", "url": "https://platform.openai.com/docs/guides/function-calling", "snippet": "Learn how to connect LLMs to external tools."},
        {"title": "Anthropic Tool Use", "url": "https://docs.anthropic.com/en/docs/tool-use", "snippet": "Claude can interact with external tools and APIs."},
    ],
    "MCP protocol": [
        {"title": "Model Context Protocol", "url": "https://modelcontextprotocol.io", "snippet": "An open standard for connecting AI models to data sources."},
    ],
    "weather API": [
        {"title": "OpenWeatherMap API", "url": "https://openweathermap.org/api", "snippet": "Free weather API with current, forecast, and historical data."},
    ],
}


def web_search(query, max_results=3):
    key = query.lower().strip()
    for db_key, results in SEARCH_DB.items():
        if db_key in key or key in db_key:
            return {"query": query, "results": results[:max_results], "total": len(results)}
    return {"query": query, "results": [], "total": 0}


FILE_SYSTEM = {
    "data/config.json": '{"model": "gpt-4o", "temperature": 0.7, "max_tokens": 4096}',
    "data/users.csv": "name,email,role\nAlice,alice@example.com,admin\nBob,bob@example.com,user",
    "README.md": "# My Project\nA tool-use agent built from scratch.",
}


def read_file(path):
    if ".." in path or path.startswith("/"):
        return {"error": True, "message": "Path traversal not allowed.", "code": "FORBIDDEN"}
    if path not in FILE_SYSTEM:
        available = list(FILE_SYSTEM.keys())
        return {"error": True, "message": f"File '{path}' not found.", "available_files": available, "code": "NOT_FOUND"}
    content = FILE_SYSTEM[path]
    return {"path": path, "content": content, "size_bytes": len(content), "lines": content.count("\n") + 1}


def run_code(code, language="python"):
    if language != "python":
        return {"error": True, "message": f"Language '{language}' not supported. Only 'python' is available."}
    forbidden = ["import os", "import sys", "import subprocess", "exec(", "eval(", "__import__", "open("]
    for pattern in forbidden:
        if pattern in code:
            return {"error": True, "message": f"Forbidden operation: {pattern}", "code": "SECURITY_VIOLATION"}
    try:
        local_vars = {}
        exec(code, {"__builtins__": {"print": print, "range": range, "len": len, "str": str, "int": int, "float": float, "list": list, "dict": dict, "sum": sum, "min": min, "max": max, "abs": abs, "round": round, "sorted": sorted, "enumerate": enumerate, "zip": zip, "map": map, "filter": filter, "math": math}}, local_vars)
        result = local_vars.get("result", None)
        return {"success": True, "result": result, "variables": {k: str(v) for k, v in local_vars.items() if not k.startswith("_")}}
    except Exception as e:
        return {"error": True, "message": f"{type(e).__name__}: {e}"}
```

### Step 3：Register All Tools

```python
def register_all_tools():
    register_tool(
        "calculator", "Evaluate a mathematical expression. Supports +, -, *, /, parentheses, and decimals. Returns the numeric result.",
        {"type": "object", "properties": {"expression": {"type": "string", "description": "Math expression, e.g. '(10 + 5) * 3'"}, "precision": {"type": "integer", "description": "Decimal places in result", "default": 2}}, "required": ["expression"]},
        calculator,
    )
    register_tool(
        "get_weather", "Get current weather for a city. Returns temperature, condition, humidity, and wind speed.",
        {"type": "object", "properties": {"city": {"type": "string", "description": "City name, e.g. 'Tokyo' or 'San Francisco'"}, "units": {"type": "string", "enum": ["celsius", "fahrenheit"], "description": "Temperature units, defaults to celsius"}}, "required": ["city"]},
        get_weather,
    )
    register_tool(
        "web_search", "Search the web for information. Returns a list of results with title, URL, and snippet.",
        {"type": "object", "properties": {"query": {"type": "string", "description": "Search query"}, "max_results": {"type": "integer", "description": "Maximum results to return", "default": 3}}, "required": ["query"]},
        web_search,
    )
    register_tool(
        "read_file", "Read the contents of a file. Returns the file content, size, and line count.",
        {"type": "object", "properties": {"path": {"type": "string", "description": "Relative file path, e.g. 'data/config.json'"}}, "required": ["path"]},
        read_file,
    )
    register_tool(
        "run_code", "Execute Python code in a sandboxed environment. Set a 'result' variable to return output.",
        {"type": "object", "properties": {"code": {"type": "string", "description": "Python code to execute"}, "language": {"type": "string", "enum": ["python"], "description": "Programming language"}}, "required": ["code"]},
        run_code,
    )
```

### Step 4：构建 Function Calling Loop

这是核心 engine。它模拟模型决定调用哪个 tool，执行 tool，并把结果反馈回去。

```python
def simulate_model_decision(user_message, tools, conversation_history):
    msg = user_message.lower()

    if any(word in msg for word in ["weather", "temperature", "forecast"]):
        cities = []
        for city in WEATHER_DB:
            if city in msg:
                cities.append(city)
        if not cities:
            for word in msg.split():
                if word.capitalize() in [c.title() for c in WEATHER_DB]:
                    cities.append(word)
        if not cities:
            cities = ["tokyo"]
        calls = []
        for city in cities:
            calls.append({"name": "get_weather", "arguments": {"city": city.title()}})
        return calls

    if any(word in msg for word in ["calculate", "compute", "math", "what is", "how much"]):
        for token in msg.split():
            if any(c in token for c in "+-*/"):
                return [{"name": "calculator", "arguments": {"expression": token}}]
        if "+" in msg or "-" in msg or "*" in msg or "/" in msg:
            expr = "".join(c for c in msg if c in "0123456789+-*/.() ")
            if expr.strip():
                return [{"name": "calculator", "arguments": {"expression": expr.strip()}}]
        return [{"name": "calculator", "arguments": {"expression": "0"}}]

    if any(word in msg for word in ["search", "find", "look up", "google"]):
        query = msg.replace("search for", "").replace("look up", "").replace("find", "").strip()
        return [{"name": "web_search", "arguments": {"query": query}}]

    if any(word in msg for word in ["read", "file", "open", "cat", "show"]):
        for path in FILE_SYSTEM:
            if path.split("/")[-1].split(".")[0] in msg:
                return [{"name": "read_file", "arguments": {"path": path}}]
        return [{"name": "read_file", "arguments": {"path": "README.md"}}]

    if any(word in msg for word in ["run", "execute", "code", "python"]):
        return [{"name": "run_code", "arguments": {"code": "result = 'Hello from the sandbox!'", "language": "python"}}]

    return []


def execute_tool_call(tool_call):
    name = tool_call["name"]
    args = tool_call["arguments"]

    if name not in TOOL_REGISTRY:
        return {"error": True, "message": f"Unknown tool: {name}", "code": "UNKNOWN_TOOL"}

    tool = TOOL_REGISTRY[name]
    func = tool["function"]
    start = time.time()

    try:
        result = func(**args)
    except TypeError as e:
        result = {"error": True, "message": f"Invalid arguments: {e}"}

    elapsed_ms = round((time.time() - start) * 1000, 2)
    return {"tool": name, "result": result, "execution_time_ms": elapsed_ms}


def run_function_calling_loop(user_message, max_iterations=5):
    conversation = [{"role": "user", "content": user_message}]
    tool_definitions = [t["definition"] for t in TOOL_REGISTRY.values()]
    all_tool_results = []

    for iteration in range(max_iterations):
        tool_calls = simulate_model_decision(user_message, tool_definitions, conversation)

        if not tool_calls:
            break

        results = []
        for call in tool_calls:
            result = execute_tool_call(call)
            results.append(result)

        conversation.append({"role": "assistant", "content": None, "tool_calls": tool_calls})

        for result in results:
            conversation.append({"role": "tool", "content": json.dumps(result["result"]), "tool_name": result["tool"]})

        all_tool_results.extend(results)
        break

    return {"conversation": conversation, "tool_results": all_tool_results, "iterations": iteration + 1 if tool_calls else 0}
```

### Step 5：Argument Validation

构建 validator，在执行前根据 JSON Schema 检查 tool call arguments。

```python
def validate_tool_arguments(tool_name, arguments):
    if tool_name not in TOOL_REGISTRY:
        return [f"Unknown tool: {tool_name}"]

    schema = TOOL_REGISTRY[tool_name]["definition"]["function"]["parameters"]
    errors = []

    if not isinstance(arguments, dict):
        return [f"Arguments must be an object, got {type(arguments).__name__}"]

    for required_field in schema.get("required", []):
        if required_field not in arguments:
            errors.append(f"Missing required argument: {required_field}")

    properties = schema.get("properties", {})
    for arg_name, arg_value in arguments.items():
        if arg_name not in properties:
            errors.append(f"Unknown argument: {arg_name}")
            continue

        prop_schema = properties[arg_name]
        expected_type = prop_schema.get("type")

        type_checks = {"string": str, "integer": int, "number": (int, float), "boolean": bool, "array": list, "object": dict}
        if expected_type in type_checks:
            if not isinstance(arg_value, type_checks[expected_type]):
                errors.append(f"Argument '{arg_name}': expected {expected_type}, got {type(arg_value).__name__}")

        if "enum" in prop_schema and arg_value not in prop_schema["enum"]:
            errors.append(f"Argument '{arg_name}': '{arg_value}' not in {prop_schema['enum']}")

    return errors
```

### Step 6：运行 Demo

```python
def run_demo():
    register_all_tools()

    print("=" * 60)
    print("  Function Calling & Tool Use Demo")
    print("=" * 60)

    print("\n--- Registered Tools ---")
    for name, tool in TOOL_REGISTRY.items():
        desc = tool["definition"]["function"]["description"][:60]
        params = list(tool["definition"]["function"]["parameters"].get("properties", {}).keys())
        print(f"  {name}: {desc}...")
        print(f"    params: {params}")

    print(f"\n--- Argument Validation ---")
    validation_tests = [
        ("get_weather", {"city": "Tokyo"}, "Valid call"),
        ("get_weather", {}, "Missing required arg"),
        ("get_weather", {"city": "Tokyo", "units": "kelvin"}, "Invalid enum value"),
        ("calculator", {"expression": 123}, "Wrong type (int for string)"),
        ("unknown_tool", {"x": 1}, "Unknown tool"),
    ]
    for tool_name, args, label in validation_tests:
        errors = validate_tool_arguments(tool_name, args)
        status = "VALID" if not errors else f"ERRORS: {errors}"
        print(f"  {label}: {status}")

    print(f"\n--- Tool Execution ---")
    direct_tests = [
        {"name": "calculator", "arguments": {"expression": "(10 + 5) * 3 / 2"}},
        {"name": "get_weather", "arguments": {"city": "Tokyo"}},
        {"name": "get_weather", "arguments": {"city": "Mars"}},
        {"name": "web_search", "arguments": {"query": "python function calling"}},
        {"name": "read_file", "arguments": {"path": "data/config.json"}},
        {"name": "read_file", "arguments": {"path": "../etc/passwd"}},
        {"name": "run_code", "arguments": {"code": "result = sum(range(1, 101))"}},
        {"name": "run_code", "arguments": {"code": "import os; os.system('rm -rf /')"}},
    ]
    for call in direct_tests:
        result = execute_tool_call(call)
        print(f"\n  {call['name']}({json.dumps(call['arguments'])})")
        print(f"    -> {json.dumps(result['result'], indent=None)[:100]}")
        print(f"    time: {result['execution_time_ms']}ms")

    print(f"\n--- Full Function Calling Loop ---")
    test_queries = [
        "What's the weather in Tokyo?",
        "Calculate (100 + 250) * 0.15",
        "Search for MCP protocol",
        "Read the config file",
        "Run some Python code",
        "Tell me a joke",
    ]
    for query in test_queries:
        print(f"\n  User: {query}")
        result = run_function_calling_loop(query)
        if result["tool_results"]:
            for tr in result["tool_results"]:
                print(f"    Tool: {tr['tool']} ({tr['execution_time_ms']}ms)")
                print(f"    Result: {json.dumps(tr['result'], indent=None)[:90]}")
        else:
            print(f"    [No tool called -- direct response]")
        print(f"    Iterations: {result['iterations']}")

    print(f"\n--- Parallel Tool Calls ---")
    multi_city_query = "What's the weather in tokyo and london?"
    print(f"  User: {multi_city_query}")
    result = run_function_calling_loop(multi_city_query)
    print(f"  Tool calls made: {len(result['tool_results'])}")
    for tr in result["tool_results"]:
        city = tr["result"].get("city", "unknown")
        temp = tr["result"].get("temp_c", "N/A")
        print(f"    {city}: {temp}C, {tr['result'].get('condition', 'N/A')}")

    print(f"\n--- Security Checks ---")
    security_tests = [
        ("read_file", {"path": "../../etc/passwd"}),
        ("run_code", {"code": "import subprocess; subprocess.run(['ls'])"}),
        ("calculator", {"expression": "__import__('os').system('ls')"}),
    ]
    for tool_name, args in security_tests:
        result = execute_tool_call({"name": tool_name, "arguments": args})
        blocked = result["result"].get("error", False)
        print(f"  {tool_name}({list(args.values())[0][:40]}): {'BLOCKED' if blocked else 'ALLOWED'}")
```

## 使用

### OpenAI Function Calling

```python
# from openai import OpenAI
#
# client = OpenAI()
#
# tools = [{
#     "type": "function",
#     "function": {
#         "name": "get_weather",
#         "description": "Get current weather for a city",
#         "parameters": {
#             "type": "object",
#             "properties": {
#                 "city": {"type": "string"},
#                 "units": {"type": "string", "enum": ["celsius", "fahrenheit"]}
#             },
#             "required": ["city"]
#         }
#     }
# }]
#
# response = client.chat.completions.create(
#     model="gpt-4o",
#     messages=[{"role": "user", "content": "Weather in Tokyo?"}],
#     tools=tools,
#     tool_choice="auto",
# )
#
# tool_call = response.choices[0].message.tool_calls[0]
# args = json.loads(tool_call.function.arguments)
# result = get_weather(**args)
#
# final = client.chat.completions.create(
#     model="gpt-4o",
#     messages=[
#         {"role": "user", "content": "Weather in Tokyo?"},
#         response.choices[0].message,
#         {"role": "tool", "tool_call_id": tool_call.id, "content": json.dumps(result)},
#     ],
# )
# print(final.choices[0].message.content)
```

OpenAI 把 tool calls 返回为 `response.choices[0].message.tool_calls`。每个 call 都有一个 `id`，你在返回结果时必须包含它。模型用这个 ID 把 results 与 calls 对齐。GPT-4o 可以在单个 response 中返回多个 tool calls，遍历并执行全部。

### Anthropic Tool Use

```python
# import anthropic
#
# client = anthropic.Anthropic()
#
# response = client.messages.create(
#     model="claude-sonnet-4-20250514",
#     max_tokens=1024,
#     tools=[{
#         "name": "get_weather",
#         "description": "Get current weather for a city",
#         "input_schema": {
#             "type": "object",
#             "properties": {
#                 "city": {"type": "string"},
#                 "units": {"type": "string", "enum": ["celsius", "fahrenheit"]}
#             },
#             "required": ["city"]
#         }
#     }],
#     messages=[{"role": "user", "content": "Weather in Tokyo?"}],
# )
#
# tool_block = next(b for b in response.content if b.type == "tool_use")
# result = get_weather(**tool_block.input)
#
# final = client.messages.create(
#     model="claude-sonnet-4-20250514",
#     max_tokens=1024,
#     tools=[...],
#     messages=[
#         {"role": "user", "content": "Weather in Tokyo?"},
#         {"role": "assistant", "content": response.content},
#         {"role": "user", "content": [{"type": "tool_result", "tool_use_id": tool_block.id, "content": json.dumps(result)}]},
#     ],
# )
```

Anthropic 把 tool calls 返回为 `type: "tool_use"` 的 content blocks。Tool result 放在带有 `type: "tool_result"` 的 user message 中。注意关键区别：Anthropic 使用 `input_schema` 定义 tool parameters，而 OpenAI 使用 `parameters`。

### MCP Integration

```python
# MCP servers expose tools over a standardized protocol.
# Any MCP-compatible client can discover and call these tools.
#
# Example: connecting to a Postgres MCP server
#
# from mcp import ClientSession, StdioServerParameters
# from mcp.client.stdio import stdio_client
#
# server_params = StdioServerParameters(
#     command="npx",
#     args=["-y", "@modelcontextprotocol/server-postgres", "postgresql://localhost/mydb"],
# )
#
# async with stdio_client(server_params) as (read, write):
#     async with ClientSession(read, write) as session:
#         await session.initialize()
#         tools = await session.list_tools()
#         result = await session.call_tool("query", {"sql": "SELECT count(*) FROM users"})
```

MCP 将 tool implementation 与 tool consumption 解耦。Postgres server 懂 SQL。GitHub server 懂 API。你的 agent 只需要发现并调用 tools，不需要为每个 integration 编写 provider-specific code。

## 交付

本课会产出 `outputs/prompt-tool-designer.md`，这是一个可复用 prompt template，用于设计 tool definitions。给它一个你希望 tool 做什么的描述，它会生成包含 descriptions、types 和 constraints 的完整 JSON Schema definition。

它还会产出 `outputs/skill-function-calling-patterns.md`，这是一个 production function calling 决策框架，覆盖 tool design、error handling、security 和 provider-specific patterns。

## 练习

1. **添加第 6 个 tool：database query。** 实现一个模拟 SQL tool，使用 in-memory table。Tool 接受 table name 和 filter conditions（不是 raw SQL）。验证 table name 在 allowlist 中，并且 filter operators 限制为 `=`、`>`、`<`、`>=`、`<=`。以 JSON 返回匹配 rows。

2. **实现带 error feedback 的 retry。** 当 tool call 失败时（例如 city not found），把 error message 反馈给 model decision function，让它纠正 arguments。记录每个 call 需要多少次 retries。设置每个 tool call 最多 3 次 retries。

3. **构建 multi-step agent。** 有些查询需要串联 tool calls：“Read the config file and tell me what model is configured, then search the web for that model's pricing.” 实现一个 loop，持续运行直到模型决定不再需要 tools，并把累计结果传入每个 decision step。限制为 10 iterations 以防 infinite loops。

4. **测量 tool selection accuracy。** 创建 30 个 test queries，每个都有预期 tool name。对所有 30 个运行你的 decision function，测量它选择正确 tool 的比例。识别哪些 queries 最容易在 tools 之间混淆。

5. **实现 tool call caching。** 如果同一个 tool 在 60 秒内以相同 arguments 被调用，返回 cached result 而不是重新执行。使用以 `(tool_name, frozenset(args.items()))` 为 key 的 dictionary。测量一个包含 20 个 queries 的 conversation 中的 cache hit rate。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|----------------|----------------------|
| Function calling | “Tool use” | 模型输出结构化 JSON，描述要用具体 arguments 调用某个 function，由你的代码执行它，而不是模型执行 |
| Tool definition | “Function schema” | 描述 tool name、purpose、parameters 和 types 的 JSON Schema object，模型读取它来决定何时以及如何使用 tool |
| Tool choice | “Calling mode” | 控制模型必须调用 tool（required）、可以调用 tool（auto），或必须调用具体 tool（named） |
| Parallel calling | “Multi-tool” | 模型在单轮中输出多个 tool calls，减少 round trips，GPT-4o 和 Claude 都支持 |
| Tool result | “Function output” | 执行 tool 的返回值，以 message 形式发回模型，使它能在响应中使用真实数据 |
| Argument validation | “Input checking” | 在执行 tool 前，验证模型生成的 arguments 是否匹配预期 types、ranges 和 constraints |
| MCP | “Tool protocol” | Model Context Protocol，Anthropic 的开放标准，通过 servers 暴露 tools，让任何兼容 client 都能发现并调用 |
| Agent loop | “ReAct loop” | model-decides-tool、code-executes-tool、result-feeds-back 的迭代循环，直到模型拥有足够信息可响应 |
| Tool poisoning | “通过 tools 做 prompt injection” | Tool results 中包含操纵模型行为的指令。必须清理所有 tool outputs |
| Rate limiting | “Call budget” | 为每个 conversation 设置最大 tool calls 数量，以防 infinite loops 和 runaway API costs |

## 延伸阅读

- [OpenAI Function Calling Guide](https://platform.openai.com/docs/guides/function-calling)：GPT-4o tool use 的权威参考，包括 parallel calls、forced calling 和 structured arguments。
- [Anthropic Tool Use Guide](https://docs.anthropic.com/en/docs/tool-use)：Claude 的 tool use 实现，包含 input_schema、multi-tool responses 和 tool_choice configuration。
- [Model Context Protocol Specification](https://modelcontextprotocol.io)：跨 AI applications 的 tool interoperability 开放标准，采用 server/client architecture。
- [Schick et al., 2023 -- "Toolformer: Language Models Can Teach Themselves to Use Tools"](https://arxiv.org/abs/2302.04761)：训练 LLM 决定何时以及如何调用外部 tools 的基础论文。
- [Patil et al., 2023 -- "Gorilla: Large Language Model Connected with Massive APIs"](https://arxiv.org/abs/2305.15334)：在 1,645 个 APIs 上 fine-tune LLM，以提升 API calls 准确性并减少 hallucination。
- [Berkeley Function Calling Leaderboard](https://gorilla.cs.berkeley.edu/leaderboard.html)：实时 benchmark，比较 GPT-4o、Claude、Gemini 和 open models 的 function calling accuracy。
- [Yao et al., "ReAct: Synergizing Reasoning and Acting in Language Models" (ICLR 2023)](https://arxiv.org/abs/2210.03629)：Thought-Action-Observation loop，这是每个 tool call 外层 agent loop 的基础。本课结束的地方，Phase 14 会接上。
- [Anthropic — Building effective agents (Dec 2024)](https://www.anthropic.com/research/building-effective-agents)：由单一 tool-use primitive 构建的五种可组合 patterns（prompt chaining、routing、parallelization、orchestrator-workers、evaluator-optimizer）。
