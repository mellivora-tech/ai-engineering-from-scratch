# Supervisor / Orchestrator-Worker Pattern

> 一个 lead agent 负责规划和委派；specialized workers 在并行 contexts 中执行并回报。这是 Anthropic Research system 背后的 pattern（Claude Opus 4 作为 lead，Sonnet 4 作为 subagents），在 internal research evals 上比 single-agent Opus 4 高 +90.2%。Anthropic 的 engineering post 报告说，BrowseComp 上 80% 的方差仅由 token usage 解释 — multi-agent 很大程度上赢在每个 subagent 都获得了 fresh context window。本课从 primitives 构建 supervisor pattern，并覆盖 production deployments 中的 2026 engineering lessons。

**类型：** 学习 + 构建
**语言：** Python（stdlib，`threading`）
**前置要求：** 阶段 16 · 04（Primitive Model）
**时间：** ~75 分钟

## 问题

Research 是 single-agent systems 的典型失败任务。你问“2023 到 2026 年 multi-agent systems 发生了什么变化？”单 agent 按顺序读五篇论文，把一半 context 填成论文文本，然后还要对它们一起推理。读到第五篇时，它已经忘了第一篇。它无法并行。

supervisor pattern 修复了这一点：一个 lead agent 规划搜索，把每个 sub-question 委派给 worker，然后综合。每个 worker 都有自己的 200k-token window，用于一个狭窄问题。lead 不看原始论文，只看 worker summaries。

Anthropic 的 production Research system 报告称，在 internal research evals 上相较 single Opus 4 提升 +90.2%。同一篇文章指出，BrowseComp 方差的 80% 由 *token usage alone* 解释。每个 subagent 的 fresh context 是主要机制。

## 概念

### 这个 pattern

```
                 ┌──────────────┐
                 │   Lead       │  plans, decomposes,
                 │  (Opus 4)    │  synthesizes
                 └──┬────┬───┬──┘
                    │    │   │
            ┌───────┘    │   └───────┐
            ▼            ▼           ▼
      ┌─────────┐  ┌─────────┐  ┌─────────┐
      │ Worker1 │  │ Worker2 │  │ Worker3 │
      │(Sonnet) │  │(Sonnet) │  │(Sonnet) │
      └─────────┘  └─────────┘  └─────────┘
         fresh       fresh        fresh
         context     context      context
```

lead 从不读取原始材料。workers 在 lead synthesis 前也不看彼此工作。每条箭头都是带有窄 artifact 的 handoff。

### 为什么它有效

三种机制：

1. **每个 subagent 都有 fresh context。** 一个探索 “FIPA-ACL heritage” 的 worker 不会携带 lead 用于 planning 的 40k tokens。它为一个问题获得 200k window。
2. **通过 prompt specialization。** lead 的 prompt 是 “decompose and synthesize”，不是 “research”。每个 worker 的 prompt 很窄：“find what changed in X”。聚焦 prompts 产生聚焦 outputs。
3. **并行性。** workers 并发运行。wall-clock time 大约是 `max(worker_times) + plan + synthesis`，不是 `sum(worker_times)`。

### Engineering lessons（Anthropic 2025）

Anthropic 文章列出了一些到 2026 仍然相关的 production lessons：

- **按 query complexity 缩放 effort。** 简单 queries：一个 agent，3-10 次 tool calls。复杂 queries：10+ agents。lead 必须估计这一点，而不是 caller。
- **先广后窄。** 先分解为宽 sub-questions，然后如果答案值得深入，再为每个 sub-question 启动更多 workers。
- **Rainbow deployments。** Agents 是 long-running 且 stateful 的。传统 blue-green 不适用。Anthropic 使用 rainbow：新版本逐步 rollout，同时让旧版本 drain。
- **Token usage dominates。** Multi-agent 大约是 single-agent 的 15× tokens。只有当任务价值证明成本合理时才运行。

### LangGraph 的转向

LangGraph 最初发布了 `langgraph-supervisor` library，带高层 `create_supervisor` helper。2025 年，LangChain 把推荐做法改成通过 tool-calling 直接实现 supervisor pattern，因为 tool calls 能更好控制 *supervisor 看到什么*（context engineering）。library 仍可用；docs 现在推荐 tool-calling 形式。

### Failure modes

- **Lead hallucinate plan。** 如果 lead 生成的 sub-questions 没有真正分解问题，workers 会精准研究错误目标。
- **Workers over-explore。** 没有显式 scope boundaries，workers 会漂移到 assigned sub-question 之外，并污染 synthesis step。
- **Synthesis conflicts。** 两个 workers 返回矛盾 facts。lead 要么 re-ask（增加一轮），要么明确记录 disagreement。悄悄选一边是最坏失败：用户永远不知道发生过分歧。

### 什么时候 supervisor 是错的

- **Sequential tasks。** 如果步骤 2 必须依赖步骤 1 的输出，并行没有收益。使用 pipeline（CrewAI Sequential、LangGraph linear graph）。
- **Simple queries。** Single-agent 更快更便宜。启动 workers 前先用 lead 的 “scale effort” check。
- **Strict determinism。** Supervisor 使用 LLM-selected delegation。当 audit/replay 比 adaptability 更重要时，static graphs 更好。

## 构建它

`code/main.py` 用 `threading` 实现了一个包含三个并行 workers 的 supervisor。lead 把 query 分解为 sub-questions，workers 在每个 sub-question 上并发运行，然后 lead 综合。没有真实 LLMs — workers 是 scripted，用来模拟 fetch-and-summarize。

关键结构：

- `Lead.plan(query)` 把 query 拆成 3 个 sub-questions。
- `Worker.run(sub_q)` 返回 fake summary（production 中可以是任何 tool-using agent）。
- `Lead.run(query)` 启动 worker threads、join，并综合。

运行：

```
python3 code/main.py
```

输出展示 plan、带 start/end timestamps 的并行 worker traces，以及最终 synthesis。你能看到 wall-clock 的收益：三个 0.3 秒 workers 在 ~0.35 秒内完成，而不是 0.9 秒。

## 使用它

`outputs/skill-supervisor-designer.md` 接收 user query，并生成 supervisor-pattern design：lead system prompt、worker roles、sub-question decomposition rules 和 synthesis template。在构建新的 research-style agent system 前使用。

## 发布它

部署 supervisor pattern 前的 checklist：

- **Model pairing。** Lead 使用 reasoning-tier model（Opus class、`o3` class）。Workers 使用更快、更便宜的 model（Sonnet、`o4-mini`）。
- **Worker timeout。** 任何超过 2× median runtime 的 worker 都会被 kill；lead 要么用更窄 scope 重新 spawn，要么在缺少它的情况下继续。
- **Token cap per worker。** 硬限制（比如 expected synthesis input 的 10×）防止 runaway worker 烧穿预算。
- **Observability。** Trace lead 的 plan、每个 worker 的 tool calls 和 synthesis。这是任何事后调试的基础。
- **Rainbow rollout。** Stateful long-running agents 需要逐步版本过渡，而不是 hot swap。

## 练习

1. 运行 `code/main.py`，然后把 lead 改成启动 5 个 workers 而不是 3 个。观察 wall-clock effect。在这个 demo 中，worker count 到多少时 spawn overhead 超过 parallel savings？
2. 实现 worker timeout：kill 任意运行超过 0.5 秒的 worker，并让 lead 综合剩余结果。你需要什么 observability 才知道 worker 被 cut 了？
3. 给 lead synthesis 添加 conflict-detection step：如果两个 workers 返回矛盾答案，lead 记录 disagreement，而不是选一个。没有 LLM call 时如何检测 contradiction？
4. 阅读 Anthropic Research-system engineering post。列出这个 toy demo 若要跑在 production 中必须采用的三项 practices。
5. 比较 LangGraph 的 `create_supervisor`（legacy）与新的 tool-calling recommendation。哪个让你更好控制 supervisor 看到什么？为什么 Anthropic 明确只把 sub-answers，而不是 raw worker context，传入 synthesis？

## 关键词汇

| 术语 | 人们常说 | 实际含义 |
|------|----------------|------------------------|
| Supervisor | “Lead agent” | 一个规划、委派并综合的 orchestrator agent。自己不做实际工作。 |
| Worker | “Subagent” | supervisor 以窄 scope 调用的 focused agent，拥有自己的 context window。 |
| Orchestrator-worker | “Supervisor pattern” | 同一件事，不同名字。2026 literature 两者都用。 |
| Fresh context | “Clean window” | worker 的 context 从 system prompt 和 assigned question 开始，而不是 lead 的 history。 |
| Rainbow deployment | “Gradual rollout” | Long-running stateful agents 需要 versioned drain-and-replace，而不是 blue-green。 |
| Token dominance | “Context is the variable” | 根据 Anthropic，research-eval 方差的 80% 来自 total tokens used，而不是 model choice。 |
| Scale effort | “让 agent count 匹配 complexity” | lead 估计 query 难度，并相应启动 1 个或 10+ workers。 |
| Synthesis conflict | “Workers disagree” | 两个 workers 返回矛盾 facts；lead 必须暴露 disagreement，而不是静默选择一边。 |

## 延伸阅读

- [Anthropic engineering — How we built our multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system) — supervisor pattern 的 production reference
- [LangGraph workflows and agents](https://docs.langchain.com/oss/python/langgraph/workflows-agents) — tool-calling supervisor 现在是推荐形式
- [LangGraph supervisor reference](https://reference.langchain.com/python/langgraph-supervisor) — legacy helper，2026 production 仍在使用
- [OpenAI cookbook — Orchestrating Agents: Routines and Handoffs](https://developers.openai.com/cookbook/examples/orchestrating_agents) — handoff-based supervisor variant
