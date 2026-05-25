# Prompt Caching 与 Context Caching

> 你的 system prompt 有 4,000 tokens。你的 RAG context 有 20,000 tokens。每次 request 都发送两者。你也为两者付费，而且每次都付。Prompt caching 让 provider 在他们那边保持这个 prefix warm，并在复用时按正常价格的 10% 收费。用对了，它能把 inference cost 降低 50-90%，把 first-token latency 降低 40-85%。

**类型：** 构建
**语言：** Python
**前置要求：** 阶段 11 · 01（Prompt Engineering），阶段 11 · 05（Context Engineering），阶段 11 · 11（Caching and Cost）
**时间：** ~60 分钟

## 问题

一个 coding agent 在 conversation 的每个 turn 都向 Claude 发送同一个 15,000-token system prompt。20 turns，按 $3/M input tokens 计，仅 input cost 就是 $0.90，还没算用户实际消息。每天 10,000 个 conversations，永不变化文本的账单就是 $9,000/day。

你不能缩短 prompt，否则质量会下降。你也不能不发送它，模型每个 turn 都需要它。唯一动作是：不要再为 provider 已经见过的 prefix 支付全价。

这个动作就是 prompt caching。Anthropic 在 2024 年 8 月发布它（2025 年加入 1-hour extended-TTL 变体），OpenAI 同年晚些时候自动化它，Google 随 Gemini 1.5 发布 explicit context caching，现在三者都在 frontier models 上把它作为 first-class feature。

## 概念

![Prompt caching: write once, read cheap](../assets/prompt-caching.svg)

**机制。** 当一个 request 的 prefix 与近期 request 的 prefix 匹配时，provider 会直接从前一次运行的 KV-cache 服务，而不是重新编码 tokens。第一次有小额 write premium，之后每次都有大额 read discount。

**2026 年三种 provider flavors。**

| Provider | API style | Hit discount | Write premium | Default TTL | Min cacheable |
|---------|-----------|--------------|---------------|-------------|---------------|
| Anthropic | Explicit `cache_control` markers on content blocks | 90% off input | 25% surcharge | 5 min (extendable to 1 hour) | 1,024 tokens (Sonnet/Opus), 2,048 (Haiku) |
| OpenAI | Automatic prefix detection | 50% off input | none | Up to 1 hour (best-effort) | 1,024 tokens |
| Google (Gemini) | Explicit `CachedContent` API | Storage-billed; read at ~25% of normal | Storage fee per token·hour | User-set (default 1 hour) | 4,096 tokens (Flash), 32,768 (Pro) |

**不变量。** 三者都只 cache prefixes。如果 requests 之间任意 token 不同，第一个不同 token 之后全是 miss。把*稳定*部分放顶部，把*可变*部分放底部。

### Cache-friendly layout

```
[system prompt]          <-- cache this
[tool definitions]       <-- cache this
[few-shot examples]      <-- cache this
[retrieved documents]    <-- cache if reused, else don't
[conversation history]   <-- cache up to last turn
[current user message]   <-- never cache (different every time)
```

违反顺序，例如把 user message 放在 system prompt 之上，或把 dynamic retrievals 插在 few-shots 中间，cache 就永远不会命中。

### Break-even calculation

Anthropic 的 25% write premium 意味着 cached block 至少要被 read 两次才净省钱。1 write + 1 read 的平均 request cost 为 0.675x（节省 32%）；1 write + 10 reads 平均为 0.205x（节省 80%）。经验法则：凡是你期望在 TTL 内复用至少 3 次的内容，都应该 cache。

## 构建它

### 第 1 步：Anthropic prompt caching with explicit markers

```python
import anthropic

client = anthropic.Anthropic()

SYSTEM = [
    {
        "type": "text",
        "text": "You are a senior Python reviewer. Follow the rubric exactly.\n\n" + RUBRIC_15K_TOKENS,
        "cache_control": {"type": "ephemeral"},
    }
]

def review(code: str):
    return client.messages.create(
        model="claude-opus-4-7",
        max_tokens=1024,
        system=SYSTEM,
        messages=[{"role": "user", "content": code}],
    )
```

`cache_control` marker 告诉 Anthropic 把这个 block 存 5 分钟。窗口内复用会 hit；过期后再次写入。

**Response usage fields:**

```python
response = review(code_a)
response.usage
# InputTokensUsage(
#     input_tokens=120,
#     cache_creation_input_tokens=15023,   # paid at 1.25x
#     cache_read_input_tokens=0,
#     output_tokens=340,
# )

response_b = review(code_b)
response_b.usage
# cache_creation_input_tokens=0
# cache_read_input_tokens=15023           # paid at 0.1x
```

在 CI 中检查两个字段。如果多次请求后 `cache_read_input_tokens` 仍为零，你的 cache keys 正在 drift。

### 第 2 步：one-hour extended TTL

长时间 batch jobs 中，5-minute default 会在 jobs 之间过期。设置 `ttl`：

```python
{"type": "text", "text": RUBRIC, "cache_control": {"type": "ephemeral", "ttl": "1h"}}
```

1-hour TTL 的 write premium 成本是 2x（相对 baseline +50%，而不是 +25%），但任何复用 prefix 超过 5 次的 batch 都会很快回本。

### 第 3 步：OpenAI automatic caching

OpenAI 不需要配置。任何超过 1,024 tokens 且与近期 request 匹配的 prefix，都会自动获得 50% discount。

```python
from openai import OpenAI
client = OpenAI()

resp = client.chat.completions.create(
    model="gpt-5",
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},   # long and stable
        {"role": "user", "content": user_msg},
    ],
)
resp.usage.prompt_tokens_details.cached_tokens  # the discounted portion
```

同样适用 cache-friendly layout rule。有两件事会杀死 OpenAI cache，但不会杀死 Anthropic cache：改变 `user` 字段（用作 cache key 组件）和重新排序 tools。

### 第 4 步：Gemini explicit context caching

Gemini 把 cache 当成你创建并命名的 first-class object：

```python
from google import genai
from google.genai import types

client = genai.Client()

cache = client.caches.create(
    model="gemini-3-pro",
    config=types.CreateCachedContentConfig(
        display_name="rubric-v3",
        system_instruction=RUBRIC,
        contents=[FEW_SHOT_EXAMPLES],
        ttl="3600s",
    ),
)

resp = client.models.generate_content(
    model="gemini-3-pro",
    contents=["Review this code:\n" + code],
    config=types.GenerateContentConfig(cached_content=cache.name),
)
```

Gemini 会按 cache 存活期间的 token·hour 收取 storage，并以约正常 input rate 的 25% 读取。当你要在多天内跨许多 sessions 复用同一个巨型 prompt 时，它的形状最合适。

### 第 5 步：在生产中测量 hit rate

`code/main.py` 有一个模拟三 provider accountant，跟踪 write/read/miss counts，并计算每 1K requests 的 blended cost。用 target hit rate gate deploys。多数 production Anthropic setups 在 warmup 后应看到 >80% read fraction。

## 2026 年仍会发布的坑

- **Dynamic timestamps at the top。** `"Current time: 2026-04-22 15:30:02"` 放在 system prompt 顶部。每次 request miss。把 timestamps 移到 cache breakpoint 下面。
- **Tool reordering。** 用稳定顺序 serialize tools；部署之间 dict reshuffle 会打破所有 hits。
- **Free-text near-duplicates。** "You are helpful." vs "You are a helpful assistant."，一个 byte difference = full miss。
- **Too-small blocks。** Anthropic 强制 1,024-token floor（Haiku 为 2,048）。更小 blocks 会 silently 不 cache。
- **Blind cost dashboards。** 把 “input tokens” 拆成 cached vs uncached。否则 traffic drop 看起来像 cache win。

## 使用它

2026 年 caching stack：

| Situation | Pick |
|-----------|------|
| Agent with stable 10k+ system prompt, many turns | Anthropic `cache_control` with 5-min TTL |
| Batch job reusing a prefix for 30+ minutes | Anthropic with `ttl: "1h"` |
| Serverless endpoints on GPT-5, no custom infra | OpenAI automatic (just make your prefix stable and long) |
| Multi-day reuse of a giant code/doc corpus | Gemini explicit `CachedContent` |
| Cross-provider fallback | Keep the cacheable prefix layout identical across providers so any hit works |

与 semantic caching（阶段 11 · 11）一起使用，用于 user-message layer：prompt caching 处理*token-identical*复用，semantic caching 处理*meaning-identical*复用。

## 交付它

保存 `outputs/skill-prompt-caching-planner.md`：

```markdown
---
name: prompt-caching-planner
description: Design a cache-friendly prompt layout and pick the right provider caching mode.
version: 1.0.0
phase: 11
lesson: 15
tags: [llm-engineering, caching, cost]
---

Given a prompt (system + tools + few-shot + retrieval + history + user) and a usage profile (requests per hour, TTL needed, provider), output:

1. Layout. Reordered sections with a single cache breakpoint marked; explain which sections are stable, which are volatile.
2. Provider mode. Anthropic cache_control, OpenAI automatic, or Gemini CachedContent. Justify from TTL and reuse pattern.
3. Break-even. Expected reads per write within TTL; net cost vs no-cache with math.
4. Verification plan. CI assertion that cache_read_input_tokens > 0 on the second identical request; dashboard split by cached vs uncached tokens.
5. Failure modes. List the three most likely reasons the cache will miss in this setup (dynamic timestamp, tool reorder, near-duplicate text) and how you will prevent each.

Refuse to ship a cache plan that places a dynamic field above the breakpoint. Refuse to enable 1h TTL without a reuse count that makes the 2x write premium pay back.
```

## 练习

1. **Easy.** 对 Claude 运行一个 10-turn conversation，system prompt 5,000 tokens。先不使用 `cache_control`，再使用它。报告两者的 input-token bill。
2. **Medium.** 写一个 test harness，给定 prompt template 和 request log，计算每个 provider（Anthropic 5m、Anthropic 1h、OpenAI automatic、Gemini explicit）的 expected hit rate 和 dollar savings。
3. **Hard.** 构建 layout optimizer：给定 prompt 和一组标记为 `stable=True/False` 的 fields，重写 prompt，把单个 cache breakpoint 放在最大 cache-friendly position，同时不丢信息。在真实 Anthropic endpoint 上验证。

## 关键术语

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Prompt caching | “Makes long prompts cheap” | 为匹配 prefixes 复用 provider-side KV-cache；重复 input tokens 打 50-90% 折扣。 |
| `cache_control` | “The Anthropic marker” | Content-block attribute，声明“一直到这里可 cache”；`{"type": "ephemeral"}`。 |
| Cache write | “Paying the premium” | 填充 cache 的第一个 request；Anthropic 按约 1.25x input rate 计费，OpenAI 免费。 |
| Cache read | “The discount” | 匹配 prefix 的后续 requests；Anthropic 10%、OpenAI 50%、Gemini 约 25%。 |
| TTL | “How long it lives” | Cache 保持 warm 的秒数；Anthropic 默认 5m（可扩展 1h），OpenAI best-effort up to 1h，Gemini user-set。 |
| Extended TTL | “1-hour Anthropic cache” | `{"type": "ephemeral", "ttl": "1h"}`；2x write premium，但 batch reuse 时值得。 |
| Prefix match | “Why my cache missed” | 只有从开头到 breakpoint 的每个 token 都 byte-identical，cache 才会 hit。 |
| Context caching (Gemini) | “The explicit one” | Google 的 named、storage-billed cache object；适合 multi-day reuse of large corpora。 |

## 延伸阅读

- [Anthropic — Prompt caching](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching) — `cache_control`、1h TTL、break-even tables。
- [OpenAI — Prompt caching](https://platform.openai.com/docs/guides/prompt-caching) — automatic prefix matching。
- [Google — Context caching](https://ai.google.dev/gemini-api/docs/caching) — `CachedContent` API 与 storage pricing。
- [Anthropic engineering — Prompt caching for long-context workloads](https://www.anthropic.com/news/prompt-caching) — 原始 launch post，含 latency numbers。
- Phase 11 · 05（Context Engineering）— 在哪里切 prompt 才能让 cache 落地。
- Phase 11 · 11（Caching and Cost）— 将 prompt caching 与 user messages 上的 semantic cache 配对。
- [Pope et al., "Efficiently Scaling Transformer Inference" (2022)](https://arxiv.org/abs/2211.05102) — prompt caching 暴露给用户的 KV-cache memory model；解释为什么 cached prefix reread 比 recompute 便宜约 10×。
- [Agrawal et al., "SARATHI: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills" (2023)](https://arxiv.org/abs/2308.16369) — prefill 是 prompt caching shortcut 的阶段；解释为什么 cache hit 会显著降低 TTFT，而 TPOT 不受影响。
- [Leviathan et al., "Fast Inference from Transformers via Speculative Decoding" (2023)](https://arxiv.org/abs/2211.17192) — prompt caching 与 speculative decoding、Flash Attention、MQA/GQA 一起作为弯曲 inference cost curve 的杠杆；读它了解另外三者。
