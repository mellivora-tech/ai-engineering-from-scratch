# Prompt Caching 和 Semantic Caching Economics

> **Pricing snapshot dated 2026-04.** 下面的数值声明反映本课发布时捕获的 vendor rate cards；在下游引用前，请对照链接文档重新验证。

> Caching 发生在两层。L2（provider-level）prompt/prefix caching 复用 repeated prefixes 的 attention KV：Anthropic prompt-caching docs 宣称长 prompt 上最多 90% cost reduction 和 85% latency reduction；对 Claude 3.5 Sonnet，cache reads 为 $0.30/M，而 fresh 为 $3.00/M，5-minute TTL，1-hour TTL option 有 2x write premium（docs.anthropic.com，2026-04）。OpenAI prompt caching 对 ≥1024 tokens 的 prompts 自动应用，cached input 价格大约比 fresh 低 90%（platform.openai.com，2026-04）；具体 per-model cached rate 取决于 live rate card。L1（app-level）semantic caching 在 embedding similarity hit 时完全跳过 LLM。Vendor “95% accuracy” 指 match correctness，不是 hit rate；production hit rates 从 10%（open-ended chat）到 70%（structured FAQ）不等；provider 都没有发布官方 baseline，所以应把它们视为 community telemetry，而不是 guarantee。Production pitfalls：parallelization 会杀死 caching（第一次 cache write 完成前发出的 N 个 parallel requests 会让 spend 膨胀数倍），prefix 中的 dynamic content 会完全阻止 cache hits。ProjectDiscovery 报告通过把 dynamic text 移出 cacheable prefix，hit rate 从 7% 提升到 74%（2025-11）。

**类型：** 学习
**语言：** Python（stdlib，玩具版 two-layer cache simulator）
**前置要求：** 阶段 17 · 04（vLLM Serving Internals），阶段 17 · 06（SGLang RadixAttention）
**时间：** ~60 分钟

## 学习目标

- 区分 L2 prompt/prefix caching（provider 侧 KV reuse）和 L1 semantic caching（相似 prompts 命中时绕过 LLM）。
- 解释 Anthropic 的 `cache_control` explicit marking 以及两个 TTL options（5-min vs 1-hour）和它们的 price multipliers。
- 给定 hit rate、prompt/response mix 和 token prices，计算 expected monthly savings。
- 说出让账单膨胀 5-10x 的 parallelization anti-pattern，以及让 hit rate 崩掉的 dynamic-content anti-pattern。

## 问题

你给 RAG service 添加了 prompt caching。账单没有变化。你测 hit rate：7%。你的 prompts 看起来是静态的，但其实不是：system prompt 中包含精确到分钟的当前日期、request ID，以及为了多样性随机重排的 examples。每个请求都写入新 cache entry，读取为零。

另外，你的 agent 对每个用户问题运行十个 parallel tool calls。十个都在第一次 cache write 完成前到达 provider。十次写入，零次读取。你的账单是“启用 caching 后本应花费”的 5-10x。

Caching 是协议，不是 flag。两层，两个不同 failure modes。

## 概念

### L2 — provider prompt/prefix caching

Provider 存储 cacheable prefix 的 attention KV，并在下一个匹配 prefix 的请求上复用。你支付一次 write cost，reads 几乎免费。

**Anthropic（Claude 3.5 / 3.7 / 4 series）**：request 中显式 `cache_control` marker。你标记哪些 blocks 可缓存。TTL：5-minute（write costs 1.25x base）或 1-hour（write costs 2x base）。Cache reads：Claude 3.5 Sonnet 上 $0.30/M vs $3.00/M fresh，便宜 10x（docs.anthropic.com，截至 2026-04）。Rates 按模型不同（Opus/Haiku 另行发布）；永远 cross-check live pricing page。

**OpenAI**：对 prompts ≥1024 tokens 自动 caching（platform.openai.com，2026-04）。没有显式 flag。Cached input 在当前 gpt-4o/gpt-5 rate cards 上大约比 fresh 便宜 10x。Docs 和 release notes 都没有发布官方 hit-rate baseline；community reports 在精心 prompt design 下集中在 30–60%。监控 `usage.cached_tokens` 来测你自己的。

**Google（Gemini）**：通过 explicit API 做 context caching；1M-token context 让 caching 更划算。

**Self-hosted（vLLM、SGLang）**：阶段 17 · 06 讲 RadixAttention，这是你自己 compute 上的同一模式。

### L1 — app-level semantic caching

在调用 LLM 之前，先 hash prompt、embed 它，并查找相似的 cached request（cosine similarity 高于 threshold，通常 0.95+）。命中时返回 cached response。miss 时调用 LLM 并缓存结果。

Open-source：Redis Vector Similarity、GPTCache、Qdrant。Commercial：Portkey Cache、Helicone Cache。

Vendor accuracy claims 指返回的 cached response 在语义上合适的频率，而不是命中频率。Production hit rates：

- Open-ended chat：10-15%。
- Structured FAQ / support：40-70%。
- Code questions：20-30%（小变体会杀死 hits）。
- Voice agents repeating prompts：50-80%（voice normalization fixed set）。

### Parallelization anti-pattern

你的 agent 并行发出 10 个 tool calls。它们都有同一个 4K-token system prompt。Anthropic cache writes 是 per-request；第一次 cache-write 在 provider 看到 prompt 后约 300 ms 完成。请求 2-10 在同一个毫秒窗口到达，各自看到 cache miss。你支付 10 次 write premiums，0 次 read discounts。

修复：batch with sequential-first。先单独发 request 1，然后等 1 的 cache populated 后再发 2-10。给第一个 tool call 增加 300 ms；节省 5-10x 账单。

### Dynamic content anti-pattern

你的 system prompt 看起来像：

```
You are a helpful assistant. The current time is 14:32:17.
User ID: abc123. Today is Tuesday...
```

每个请求都是唯一的。每个请求都写。零命中。

修复：把真正静态的内容移到 cacheable prefix；把 dynamic content 追加到 cache boundary 后面：

```
[cacheable]
You are a helpful assistant. [rules, examples, instructions]
[/cacheable]
[dynamic, not cached]
Current time: 14:32:17. User: abc123.
```

ProjectDiscovery 通过这种方式将 cache hit rate 从 7% 提升到 74%，并发布了 anatomy。

### 为 overnight workloads 叠加 batch + cache

Batch APIs（阶段 17 · 15）提供 24-hour turnaround 下的 50% discount。在此基础上叠加 cached input，再获得约 10x。Overnight classification、labeling 和 report generation workloads 可以通过 stacking 降到同步未缓存成本的约 10%。

### 你应该记住的数字

Pricing points 来自链接 vendor docs 中 2026-04 的捕获，每几个月都会 drift；依赖前重新检查。

- Anthropic cached read：Claude 3.5 Sonnet 上 $0.30/M，大约比 fresh input 便宜 10x（docs.anthropic.com）。
- Anthropic cache write premium：1.25x（5-min TTL）或 2x（1-hour TTL）。
- OpenAI auto-cache：适用于 prompts ≥1024 tokens；cached input 在当前 rate cards 上约为 fresh input 的 10%（platform.openai.com）。
- Semantic cache hit rate（community-reported）：open chat ~10%；structured FAQ 最高 ~70%。不是 vendor-documented baseline。
- ProjectDiscovery：通过把 dynamic 移出 prefix，hit rate 7% → 74%（project blog，2025-11）。
- Parallelization anti-pattern：N parallel requests miss 第一次 cache write 时，常见报告为 5–10x bill inflation。

## 使用它

`code/main.py` 在 mixed workloads 上模拟 L1 + L2 caching。报告 hit rates、bill，并展示 parallelization penalty。

## 交付它

本课会产出 `outputs/skill-cache-auditor.md`。给定 prompt template 和 traffic，它会审计 cacheability 并推荐 restructure。

## 练习

1. 运行 `code/main.py`。切换 parallelization flag。账单变化多少？
2. 你的 system prompt 中有日期。把它移出去。展示 before/after hit rate math。
3. 给定 request arrival rate，计算 1-hour TTL（2x write）vs 5-minute TTL（1.25x write）的 break-even。
4. Semantic cache 在 0.95 threshold 下命中 20%。在 0.85 下命中 50%，但你看到 incorrect cached responses。选择正确 threshold 并说明理由。
5. 每个用户问题会 batch 10 个 parallel sub-queries。改写为 cache-friendly，同时不增加 end-to-end latency。

## 关键词汇

| 术语 | 大家常说 | 实际含义 |
|------|----------|----------|
| L2 prompt cache | “prefix cache” | Provider 存储 repeated prefix 的 KV |
| `cache_control` | “Anthropic cache marker” | 标记 cacheable blocks 的 explicit attribute |
| Cache write premium | “write tax” | 首次 miss-to-cache 的额外成本（1.25x 或 2x） |
| L1 semantic cache | “embedding cache” | 调用 LLM 前的 app-level hash-and-embed |
| GPTCache | “LLM caching lib” | 流行的 OSS L1 cache library |
| Cache hit rate | “hits / total” | 由 cache 服务的请求比例 |
| Parallelization anti-pattern | “the N-write trap” | N 个 parallel requests 对 cache miss N 次 |
| Dynamic content trap | “the time-in-prompt trap” | Prefix 中的 dynamic bytes 杀死 hit rate |
| RadixAttention | “intra-replica cache” | SGLang 的 prefix-cache implementation |

## 延伸阅读

- [Anthropic Prompt Caching](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching) — 官方 `cache_control` semantics 和 TTLs。
- [OpenAI Prompt Caching](https://platform.openai.com/docs/guides/prompt-caching) — automatic caching behavior 和 eligibility。
- [TianPan — Semantic Caching for LLMs Production](https://tianpan.co/blog/2026-04-10-semantic-caching-llm-production)
- [ProjectDiscovery — Cut LLM Costs 59% With Prompt Caching](https://projectdiscovery.io/blog/how-we-cut-llm-cost-with-prompt-caching)
- [DigitalOcean / Anthropic — Prompt Caching](https://www.digitalocean.com/blog/prompt-caching-with-digital-ocean)
