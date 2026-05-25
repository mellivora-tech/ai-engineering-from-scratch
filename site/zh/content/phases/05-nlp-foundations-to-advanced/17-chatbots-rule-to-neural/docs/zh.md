# Chatbots：从规则到神经网络再到 LLM Agents

> ELIZA 用 pattern match 回复。DialogFlow 映射 intents。GPT 从权重里回答。Claude 调用工具并验证。每个时代都解决了上一个时代最糟的失败。

**类型：** 学习
**语言：** Python
**前置要求：** 阶段 5 · 13（问答），阶段 5 · 14（信息检索）
**时间：** ~75 分钟

## 问题

用户说：“I want to change my flight.” 系统必须判断他们想做什么、缺少什么信息、如何获取它，以及如何完成动作。然后用户又说：“wait, what if I cancel instead?” 系统必须记住上下文，切换任务，并保留状态。

对 ML 系统来说，对话很难。输入是开放的。输出必须在多轮中保持连贯。系统可能需要对真实世界采取行动（改航班、扣款）。每一步错误都会被用户直接看到。

Chatbot 架构经历了四种范式。每一种都是因为上一种失败得太明显才出现的。本课按顺序走一遍它们。2026 年的生产格局是后两者的混合。

## 概念

![Chatbot evolution: rule-based → retrieval → neural → agent](../assets/chatbot.svg)

**Rule-based（ELIZA、AIML、DialogFlow）。** 手写 pattern 匹配用户输入并生成回复。Intent classifiers 路由到预定义流程。Slot-filling state machines 收集必需信息。在设计好的窄范围内表现极好。一旦超出范围就立刻失败。仍然用于安全关键领域（银行认证、航空订票），因为这些场景不能容忍 hallucination。

**Retrieval-based。** FAQ 风格系统。编码每一对（utterance, response）。运行时编码用户消息，检索最相近的已存回复。可以理解为 Zendesk 经典“similar articles”功能。比规则更能处理改写。没有生成，所以没有 hallucination。

**Neural（seq2seq）。** 在对话日志上训练 encoder-decoder。从零生成回复。流畅，但容易产出泛泛回复（“I don't know”）和事实漂移。长期不可靠地保持主题。这就是 Google、Facebook、Microsoft 在 2016-2019 年的 chatbot 都令人失望的原因。

**LLM agents。** 把 language model 包在一个循环里，让它规划、调用工具、验证结果。它不是一个长 prompt 的 chatbot，而是 agent loop：plan → call tool → observe result → decide next step。Retrieval-first grounding（RAG）防止它胡编。Tool calls 让它真的能做事。这是 2026 年的架构。

这四种范式不是简单的顺序替代。2026 年的生产 chatbot 会路由经过四者：认证和破坏性操作用 rule-based，FAQ 用 retrieval，自然措辞用 neural generation，模糊开放问题用 LLM agent。

## 构建它

### 第 1 步：基于规则的 pattern matching

```python
import re


class RulePattern:
    def __init__(self, pattern, response_template):
        self.regex = re.compile(pattern, re.IGNORECASE)
        self.template = response_template


PATTERNS = [
    RulePattern(r"my name is (\w+)", "Nice to meet you, {0}."),
    RulePattern(r"i (need|want) (.+)", "Why do you {0} {1}?"),
    RulePattern(r"i feel (.+)", "Why do you feel {0}?"),
    RulePattern(r"(.*)", "Tell me more about that."),
]


def rule_based_respond(user_input):
    for pattern in PATTERNS:
        m = pattern.regex.match(user_input.strip())
        if m:
            return pattern.template.format(*m.groups())
    return "I don't understand."
```

20 行里的 ELIZA。反射技巧（“I feel sad” → “Why do you feel sad”）是 Weizenbaum 1966 年经典心理治疗师 demo。到今天仍然有教学价值。

### 第 2 步：retrieval-based（FAQ）

这个示例片段需要 `pip install sentence-transformers`（会拉入 torch）。本课可运行的 `code/main.py` 改用 stdlib Jaccard similarity，因此不需要外部依赖也能运行。

```python
from sentence_transformers import SentenceTransformer
import numpy as np


FAQ = [
    ("how do i reset my password", "Go to Settings > Security > Reset Password."),
    ("how do i cancel my order", "Go to Orders, find the order, click Cancel."),
    ("what is your return policy", "30-day returns on unused items, original packaging."),
]


encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
faq_questions = [q for q, _ in FAQ]
faq_embeddings = encoder.encode(faq_questions, normalize_embeddings=True)


def faq_respond(user_input, threshold=0.5):
    q_emb = encoder.encode([user_input], normalize_embeddings=True)[0]
    sims = faq_embeddings @ q_emb
    best = int(np.argmax(sims))
    if sims[best] < threshold:
        return None
    return FAQ[best][1]
```

基于阈值的拒答是关键设计。如果最佳匹配还不够近，就返回 `None`，让系统升级处理。

### 第 3 步：neural generation（baseline）

使用一个小型 instruction-tuned encoder-decoder（FLAN-T5）或 fine-tuned conversational model。2026 年单独用于生产不可行（矛盾、跑题、事实胡说），但在 hybrid systems 里常用于自然措辞。DialoGPT 风格 decoder-only 模型需要显式 turn separators 和 EOS 处理才能生成连贯回复；FLAN-T5 text2text pipeline 对教学示例开箱即用。

```python
from transformers import pipeline

chatbot = pipeline("text2text-generation", model="google/flan-t5-small")

response = chatbot("Respond politely to: Hi there!", max_new_tokens=40)
print(response[0]["generated_text"])
```

### 第 4 步：LLM agent loop

2026 年生产形态：

```python
def agent_loop(user_message, tools, llm, max_steps=5):
    history = [{"role": "user", "content": user_message}]
    for _ in range(max_steps):
        response = llm(history, tools=tools)
        tool_call = response.get("tool_call")
        if tool_call:
            tool_name = tool_call.get("name")
            args = tool_call.get("arguments")
            if not isinstance(tool_name, str) or tool_name not in tools:
                history.append({"role": "assistant", "tool_call": tool_call})
                history.append({"role": "tool", "name": str(tool_name), "content": f"error: unknown tool {tool_name!r}"})
                continue
            if not isinstance(args, dict):
                history.append({"role": "assistant", "tool_call": tool_call})
                history.append({"role": "tool", "name": tool_name, "content": f"error: arguments must be a dict, got {type(args).__name__}"})
                continue
            fn = tools[tool_name]
            result = fn(**args)
            history.append({"role": "assistant", "tool_call": tool_call})
            history.append({"role": "tool", "name": tool_name, "content": result})
        else:
            return response["content"]
    return "I could not complete the task in the step budget."
```

要点有三个。Tools 是 LLM 可以调用的函数。当 LLM 返回最终答案而不是 tool call 时，循环结束。Step budget 防止在模糊任务上无限循环。

真实生产还会加：retrieval-first grounding（每次 LLM 调用前注入相关 docs）、guardrails（破坏性操作必须确认）、observability（记录每一步）、evaluations（自动检查 agent 行为是否符合 spec）。

### 第 5 步：hybrid routing

```python
def hybrid_chat(user_input):
    if is_destructive_action(user_input):
        return structured_flow(user_input)

    faq_answer = faq_respond(user_input, threshold=0.6)
    if faq_answer:
        return faq_answer

    return agent_loop(user_input, tools, llm)


def is_destructive_action(text):
    danger_words = ["delete", "cancel", "charge", "refund", "transfer"]
    return any(w in text.lower() for w in danger_words)
```

模式是：任何破坏性操作用 deterministic rules，固定 FAQ 用 retrieval，其他都交给 LLM agents。这就是 2026 年客户支持系统实际交付的样子。

## 使用它

2026 年技术栈：

| 用例 | 架构 |
|---------|---------------|
| 预订、支付、认证 | Rule-based state machines + slot filling |
| 客户支持 FAQ | 在 curated answers 上做 retrieval |
| 开放式帮助聊天 | 带 RAG + tool calls 的 LLM agent |
| 内部工具 / IDE assistants | 带 tool calls 的 LLM agent（search、read、write） |
| Companion / character chatbots | 调过的 LLM + persona system prompt + knowledge retrieval |

生产中始终使用 hybrid routing。没有单一架构能妥善处理每类请求。Routing layer 本身通常是一个小型 intent classifier。

## 仍然会被交付的失败模式

- **自信编造。** LLM agent 声称完成了一个实际上没完成的动作。缓解：验证结果、记录 tool calls、没有成功 tool return 时不允许 LLM 声称已完成。
- **Prompt injection。** 用户插入文本来覆盖 system prompt。在 OWASP Top 10 for LLM Applications 2025 中排名 LLM01。两种形式：direct injection（直接贴进聊天）和 indirect injection（隐藏在 agent 读取的文档、邮件或工具输出里）。

  攻击成功率随场景变化。一般 tool-use 和 coding benchmarks 上，frontier models 的测得成功率约 0.5-8.5%。特定高风险设置（针对 AI coding agents 的 adaptive attacks、脆弱 orchestration）可达到约 84%。生产 CVE 包括 EchoLeak（CVE-2025-32711，CVSS 9.3）：Microsoft 365 Copilot 中由攻击者控制邮件触发的 zero-click 数据外泄漏洞。

  缓解：在整个循环中都把用户输入当成不可信；tool calls 前做 sanitize；把工具输出和主 prompt 隔离；使用 Plan-Verify-Execute（PVE）模式，让 agent 先计划，再把每个动作与计划核对后执行（这能阻止 tool results 注入新的未计划动作）；破坏性操作要求用户确认；工具权限最小化。

  任何 prompt engineering 都无法完全消除这个风险。需要外部运行时防御层（LLM Guard、allowlist validation、semantic anomaly detection）。
- **Scope creep。** Agent 因为 tool call 返回了旁支信息而跑题。缓解：收窄工具契约；保持 system prompt 聚焦；加入 off-task rate 评估。
- **无限循环。** Agent 反复调用同一个工具。缓解：step budget、tool-call deduplication、LLM judge 判断“是否在取得进展”。
- **Context window exhaustion。** 长对话把最早的 turns 挤出上下文。缓解：总结旧 turns，按 similarity 检索相关历史 turns，或使用 long-context model。

## 交付它

保存为 `outputs/skill-chatbot-architect.md`：

```markdown
---
name: chatbot-architect
description: 为给定用例设计 chatbot stack。
version: 1.0.0
phase: 5
lesson: 17
tags: [nlp, agents, chatbot]
---

给定产品上下文（用户需求、合规约束、可用工具、数据量），输出：

1. Architecture。Rule-based、retrieval、neural、LLM agent 或 hybrid（说明哪些路径走哪里）。
2. LLM choice（如果适用）。命名模型家族（Claude、GPT-4、Llama-3.1、Mixtral）。匹配 tool-use 质量和成本。
3. Grounding strategy。RAG sources、retrieval method（见第 14 课）、tool contracts。
4. Evaluation plan。Task success rate、tool-call correctness、off-task rate、held-out dialogs 上的 hallucination rate。

拒绝为任何破坏性操作（支付、账号删除、数据修改）推荐纯 LLM agent，除非有结构化确认流程。若 agent 对任何东西有写权限，拒绝跳过 prompt-injection audit。
```

## 练习

1. **简单。** 用上面的 rule-based respond 实现一个咖啡店点单 bot，包含 10 个 patterns。测试边界情况：重复点单、修改、取消、不清楚的 intent。
2. **中等。** 构建 hybrid FAQ + LLM fallback。为一个 SaaS 产品准备 50 条固定 FAQ，LLM fallback 在 docs site 上做 retrieval。用 100 个真实支持问题测量 refusal rate 和 accuracy。
3. **困难。** 用三个工具（search、read-user-data、send-email）实现上面的 agent loop。用包含 prompt injection 尝试的 50 个测试场景做评估。报告 off-task rate、failed task rate 和 injection success。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|-----------------|-----------------------|
| Intent | 用户想要什么 | 分类标签（book_flight、reset_password），路由到 handler。 |
| Slot | 一条信息 | Bot 需要的参数（date、destination）。Slot filling 是逐步询问的过程。 |
| RAG | Retrieval plus generation | 检索相关 docs，再让 LLM 基于它回答。 |
| Tool call | 函数调用 | LLM 发出带 name + args 的结构化调用。Runtime 执行并返回结果。 |
| Agent loop | Plan、act、verify | 控制器交替运行 LLM calls 和 tool calls，直到任务完成。 |
| Prompt injection | 用户攻击 prompt | 恶意输入试图覆盖 system prompt。 |

## 延伸阅读

- [Weizenbaum (1966). ELIZA — A Computer Program For the Study of Natural Language Communication](https://web.stanford.edu/class/cs124/p36-weizenabaum.pdf) — 最早的 rule-based chatbot 论文。
- [Thoppilan et al. (2022). LaMDA: Language Models for Dialog Applications](https://arxiv.org/abs/2201.08239) — LLM agents 接管前 Google 晚期 neural-chatbot 论文。
- [Yao et al. (2022). ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629) — 命名 agent loop pattern 的论文。
- [Anthropic's guide on building effective agents](https://www.anthropic.com/research/building-effective-agents) — 2024 年生产指导，2026 年仍然成立。
- [Greshake et al. (2023). Not what you've signed up for: Compromising Real-World LLM-Integrated Applications with Indirect Prompt Injection](https://arxiv.org/abs/2302.12173) — prompt-injection 论文。
- [OWASP Top 10 for LLM Applications 2025 — LLM01 Prompt Injection](https://genai.owasp.org/llmrisk/llm01-prompt-injection/) — 让 prompt injection 成为头号安全问题的排名。
- [AWS — Securing Amazon Bedrock Agents against Indirect Prompt Injections](https://aws.amazon.com/blogs/machine-learning/securing-amazon-bedrock-agents-a-guide-to-safeguarding-against-indirect-prompt-injections/) — 包括 Plan-Verify-Execute 和用户确认流程的实用 orchestration-layer 防御。
- [EchoLeak (CVE-2025-32711)](https://www.vectra.ai/topics/prompt-injection) — indirect prompt injection 导致 zero-click 数据外泄的标志性 CVE。它说明为什么有写权限的 agents 需要 runtime defenses。
