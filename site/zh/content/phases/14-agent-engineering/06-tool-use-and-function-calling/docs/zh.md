# Tool Use 和 Function Calling

> Toolformer（Schick et al., 2023）开启了 self-supervised tool annotation。Berkeley Function Calling Leaderboard V4（Patil et al., 2025）设定了 2026 年的标准：40% agentic、30% multi-turn、10% live、10% non-live、10% hallucination。Single-turn 已经基本解决。Memory、dynamic decision-making 和 long-horizon tool chains 还没有。

**类型：** 构建
**语言：** Python (stdlib)
**前置要求：** 阶段 14 · 01（Agent Loop），阶段 13 · 01（Function Calling Deep Dive）
**时间：** ~60 分钟

## 学习目标

- 解释 Toolformer 的 self-supervised training signal：只有当执行结果降低 next-token loss 时，才保留 tool annotations。
- 说出 BFCL V4 的五个 evaluation categories，以及每一类测量什么。
- 用 stdlib 实现一个 tool registry，包含 schema validation、argument coercion 和 execution sandboxing。
- 诊断 2026 年的三个开放问题：long-horizon tool chaining、dynamic decision-making 和 memory。

## 问题

早期 tool use 问的是：模型能不能预测一个正确的 function call？现代 tool use 问的是：模型能不能跨 40 步串联工具，带 memory、带 partial observability、能从 tool failures 中恢复，并且不会 hallucinate 不存在的工具？

Toolformer 建立了 baseline：模型可以通过 self-supervision 学会什么时候调用工具。BFCL V4 定义了 2026 年的 evaluation target。两者之间的空白，就是生产 agents 所处的空间。

## 概念

### Toolformer（Schick et al., NeurIPS 2023）

想法：让模型用 candidate API calls 标注自己的 pretraining corpus。对每个 candidate 执行它。只有当包含 tool result 能降低 next token 的 loss 时，才保留这个 annotation。然后在 filtered corpus 上 fine-tune。

覆盖的工具：calculator、QA system、search engines、translator、calendar。Self-supervision signal 只关心工具是否有助于预测文本，不需要 human labels。

Scale result：tool use 在规模上涌现。小模型会被 tool annotations 伤害；大模型会受益。这就是为什么 2026 年 frontier models 内置了很强的 tool use，而大多数 7B 模型需要显式 tool-use fine-tuning 才可靠。

### Berkeley Function Calling Leaderboard V4（Patil et al., ICML 2025）

BFCL 是 2026 年事实标准 evaluation。V4 构成：

- **Agentic (40%)**：完整 agent trajectories：memory、multi-turn、dynamic decisions。
- **Multi-Turn (30%)**：带 tool chains 的交互式对话。
- **Live (10%)**：用户提交的真实 prompts（更难的分布）。
- **Non-Live (10%)**：synthetic test cases。
- **Hallucination (10%)**：检测什么时候不应该调用工具。

V3 引入了 state-based evaluation：在一串 tool sequence 后，检查 API 的实际状态（例如“文件是否创建了？”），而不是匹配 tool calls 的 AST。V4 添加了 web search、memory 和 format sensitivity categories。

2026 年关键发现：single-turn function calling 已经接近解决。Failures 集中在 memory（跨 turns 携带 context）、dynamic decision-making（基于之前结果选择工具）、long-horizon chains（20+ 步后漂移）和 hallucination detection（没有合适工具时拒绝调用）。

### Tool schema

每个 provider 都有 schema。细节不同，但形状相同：

```
name: string
description: string (what it does, when to use it)
input_schema: JSON Schema (properties, required, types, enums)
```

Anthropic 直接使用 `input_schema`。OpenAI 使用 `function.parameters`。两者都接受 JSON Schema。Descriptions 是承重结构，因为模型会读取它们来选择正确工具。糟糕的 tool descriptions 是 wrong-tool-picked failures 的首要根因。

### Argument validation

不要信任任何 tool call。校验：

1. **Type coercion。** 模型可能在 schema 要求 int 时返回字符串 `"5"`。明确无歧义时可以 coerce；否则 reject。
2. **Enum validation。** 如果 schema 说 `status in {"open", "closed"}`，模型却发出 `"in_progress"`，用描述性错误 reject。
3. **Required fields。** 缺少必填字段 -> 立即把 error observation 返回给模型，而不是 crash。
4. **Format validation。** 日期、邮箱、URL：用具体 parsers 校验，不要用 regex。

每个 validation failure 都应该返回 structured observation，让模型可以用正确形状重试。

### Parallel tool calls

现代 providers 支持在一个 assistant turn 中发出 parallel tool calls。Loop：

1. 模型发出 3 个 tool calls，每个都有不同的 `tool_use_id`。
2. Runtime 执行它们（如果独立，可以并行）。
3. 每个结果作为 `tool_result` block 返回，并通过 `tool_use_id` 关联。

工程规则：把 correlation IDs 当作承重结构。交换它们会导致 wrong-tool-to-wrong-result routing。

### Sandboxing

Tool execution 是 sandbox boundary。详见第 09 课。简短版本：每个工具都应该指定 read/write surface、network access、timeout、memory cap。通用 `run_shell(cmd)` 是危险信号；具体的 `git_status()` 更安全。

## 构建它

`code/main.py` 实现了一个生产形态的 tool registry：

- JSON Schema subset validator（只用 stdlib）。
- Tool registration，包含 description、input schema、timeout 和 executor。
- Argument coercion 和 enum validation。
- 带 correlation IDs 的 parallel tool dispatch。
- 作为 structured strings 的 error observations。

运行它：

```
python3 code/main.py
```

Trace 会显示一个 mini agent 在一轮中调用三个工具，其中一个刻意 malformed call 会被拒绝，并返回模型可以处理的描述性错误。

## 使用它

每个 provider 都有自己的 tool schema：Anthropic、OpenAI、Gemini、Bedrock。如果需要 multi-provider，使用 translation layer（OpenAI Agents SDK、Vercel AI SDK、LangChain tool adapter）。BFCL 是参考 benchmark：如果 tool use 是产品核心，发布前对你的 agent 跑它。

## 发布它

`outputs/skill-tool-registry.md` 会为给定 task domain 生成 tool catalog、schema 和 registry。包含 description-quality checks（每个工具的 description 是否告诉模型什么时候使用它？）。

## 练习

1. 添加一个 “no-op” tool，让模型可以显式拒绝使用任何其他工具。在 BFCL-like hallucination test 上测量。
2. 为 int-as-string 和 float-as-string 实现 argument coercion。Coercion 从哪里开始会掩盖真实 bugs？
3. 添加 per-tool timeout 和 circuit breaker（连续失败 3 次后 60 秒内拒绝该工具）。这会如何改变模型的恢复方式？
4. 阅读 BFCL V4 描述。选择一个 category（例如 “multi-turn”），用你的 agent 跑 10 个 example prompts。报告 pass rate。
5. 把 stdlib validator 移植到 Pydantic 或 Zod。Pydantic/Zod 抓到了 toy 漏掉的什么？

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|----------------|------------------------|
| Function calling | “Tool use” | 带 validated schema 的 structured-output tool invocation |
| Toolformer | “Self-supervised tool annotation” | Schick 2023：保留结果能降低 next-token loss 的 tool calls |
| BFCL | “Berkeley Function Calling Leaderboard” | 2026 benchmark：40% agentic、30% multi-turn、10% live、10% non-live、10% hallucination |
| Tool schema | “Function signature for the model” | name、description、arguments 的 JSON Schema |
| tool_use_id | “Correlation ID” | 把 tool call 和 result 绑定；parallel dispatch 的关键 |
| Hallucination detection | “Know when not to call” | V4 category：没有合适工具时拒绝调用 |
| Argument coercion | “String-to-int repair” | 针对可预测 schema mismatch 的窄修复；有歧义就 reject |
| Sandboxing | “Tool execution boundary” | 每个工具的 read/write surface、network、timeout、memory cap |

## 延伸阅读

- [Schick et al., Toolformer (arXiv:2302.04761)](https://arxiv.org/abs/2302.04761)：self-supervised tool annotation
- [Berkeley Function Calling Leaderboard (V4)](https://gorilla.cs.berkeley.edu/leaderboard.html)：2026 eval benchmark
- [Anthropic, Tool use documentation](https://platform.claude.com/docs/en/agent-sdk/overview)：Claude Agent SDK 中的 production tool schema
- [OpenAI Agents SDK docs](https://openai.github.io/openai-agents-python/)：function tool type 和 Guardrails
