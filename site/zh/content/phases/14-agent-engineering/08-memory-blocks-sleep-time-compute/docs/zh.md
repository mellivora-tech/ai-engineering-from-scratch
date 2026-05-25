# Memory Blocks 和 Sleep-Time Compute（Letta）

> MemGPT 在 2024 年变成 Letta。2026 年的演进加入了两个想法：模型可以直接编辑的离散 functional memory blocks，以及在 primary agent 空闲时异步 consolidates memory 的 sleep-time agent。这就是把 memory 扩展到单次对话之外的方法。

**类型：** 构建
**语言：** Python (stdlib)
**前置要求：** 阶段 14 · 07（MemGPT）
**时间：** ~75 分钟

## 学习目标

- 说出 Letta 使用的三层 memory tiers（core、recall、archival），以及每层的作用。
- 解释 memory-block pattern：Human block、Persona block 和 user-defined blocks 作为一等 typed objects。
- 描述 sleep-time compute 是什么，为什么它位于 critical path 之外，以及为什么它可以运行比 primary agent 更强的模型。
- 实现一个 scripted two-agent loop，其中 primary agent 服务响应，sleep-time agent 在 turns 之间 consolidates blocks。

## 问题

MemGPT（第 07 课）解决了 virtual-memory control flow。随后出现了三个生产问题：

1. **Latency。** 每个 memory operation 都在 critical path 上。如果 agent 必须在用户等待时 prune、summarize 或 reconcile，tail latency 会爆炸。
2. **Memory rot。** Writes 积累。被矛盾推翻的 facts 仍然存在。Retrieval 淹没在 stale content 里。
3. **Structure loss。** Flat archival store 无法表达“Human block 始终在 prompt 中；Persona block 始终在 prompt 中；Task block 每个 session 切换”。

Letta（letta.com）是 2026 年的重写。Memory blocks 让结构显式化；sleep-time compute 把 consolidation 移出 critical path。

## 概念

### 三层

| Tier | Scope | Where it lives | Written by |
|------|-------|----------------|------------|
| Core | Always visible | Inside the main prompt | Agent tool call + sleep-time rewrites |
| Recall | Conversation history | Retrievable | Automatic turn logging |
| Archival | Arbitrary facts | Vector + KV + graph | Agent tool call + sleep-time ingest |

Core 是 MemGPT core。Recall 是带 evicted tail 的 conversation buffer。Archival 是 external store。这个拆分清理了 MemGPT two-tier 设计中的职责复用。

### Memory blocks

Block 是 core tier 中一个 typed、persistent、editable section。原始 MemGPT 论文定义了两个：

- **Human block**：关于用户的事实（name、role、preferences、goals）。
- **Persona block**：agent 的 self-concept（identity、tone、constraints）。

Letta 把它泛化为任意 user-defined blocks：表示当前目标的 `Task` block、表示 codebase facts 的 `Project` block、表示 hard constraints 的 `Safety` block。每个 block 都有 `id`、`label`、`value`、`limit`（character cap）、`description`（让模型知道什么时候编辑它）。

Blocks 可以通过 tool surface 编辑：

- `block_append(label, text)`
- `block_replace(label, old, new)`
- `block_read(label)`
- `block_summarize(label)`：压缩接近 limit 的 block。

### Sleep-time compute

Letta 在 2025 年加入的能力：在 background 中运行第二个 agent，位于 critical path 之外。Sleep-time agents 处理 conversation transcripts 和 codebase context，把 `learned_context` 写入 shared blocks，并 consolidate 或 invalidate archival records。

随之得到的性质：

- **无 latency cost。** Primary responses 不等待 memory ops。
- **允许更强模型。** Sleep-time agent 可以是更昂贵、更慢的模型，因为它不受 latency 约束。
- **天然 consolidation window。** 用户不等待时进行 dedup、summarize、invalidate contradicted facts。

这个形状符合人类工作方式：你完成任务，睡一觉，长期记忆在夜里沉淀下来。

### Letta V1 和 native reasoning

Letta V1（`letta_v1_agent`, 2026）弃用 `send_message`/heartbeat 和 inline `Thought:` tokens，改用 native reasoning。Responses API（OpenAI）和带 extended thinking 的 Messages API（Anthropic）会在独立 channel 上发出 reasoning，并跨 turns 传递（生产中跨 providers 加密）。Control loop 仍然是 ReAct。Thought trace 是结构性的，而不是 prompt-shaped。

### 这个模式会在哪里出错

- **Block bloat。** 无限 `block_append` 很快就会触碰 limit。在会越过 cap 的写入之前接一个 block summarizer。
- **Silent drift。** Sleep-time agent 重写了 block，primary agent 却没注意到。给 blocks version，并在 trace 里展示 diffs。
- **Poisoned consolidation。** Sleep-time agent 把攻击者可达内容处理进 core。第 27 课同样适用于 sleep-time surface。

## 构建它

`code/main.py` 实现：

- `Block`：id、label、value、limit、description。
- `BlockStore`：CRUD + `near_limit(label)` helper。
- 两个 scripted agents：`PrimaryAgent` 服务一次 turn，`SleepTimeAgent` 在 turns 之间 consolidate。
- 一条 trace，展示三轮对话中的 block writes，以及一次 sleep-time pass 如何 summarize block 并 invalidate stale fact。

运行它：

```
python3 code/main.py
```

Transcript 会显示拆分：primary turns 很快并产生 raw writes；sleep pass 负责 compact 和 cleanup。

## 使用它

- **Letta**（letta.com）：参考实现。可 self-host，也可使用 managed cloud。
- **Claude Agent SDK skills** 作为 block-shaped knowledge：skill 是 agent 按需加载的一块 named、versioned、retrievable instructions。
- **Custom builds**：适合想控制 storage backend 的团队。使用 Letta API contract，方便以后迁移。

## 发布它

`outputs/skill-memory-blocks.md` 会为任意 runtime 生成一个 Letta-shaped block system，包含 sleep-time hooks、safety rules 和 citation wiring。

## 练习

1. 添加一个 `block_summarize` tool，当 `near_limit` 返回 true 时，用 model-generated summary 替换 block value。哪个 trigger threshold 能同时最小化 summarization calls 和 block overflow？
2. 对 archival 实现 sleep-time dedup：两条 records 的 text 有 >90% token overlap 时合并为一条。只在 sleep pass 做，永远不要在 critical path 上做。
3. 给 blocks version。每次写入都记录 old value 和 diff。暴露 `block_history(label)`，让 operators 能调试“why did the agent forget X”。
4. 把 sleep-time agents 当作 untrusted writers。当它们触碰 Persona 或 Safety block 时，commit 前要求 second-agent review。
5. 把示例移植到 Letta API（`letta_v1_agent`）。Block schema 有什么变化？native reasoning 如何改变 trace shape？

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|----------------|------------------------|
| Memory block | “Editable prompt section” | Core memory 中 typed、persistent、LLM-editable 的片段 |
| Human block | “User memory” | 关于用户的 facts，固定在 core 中 |
| Persona block | “Agent identity” | Self-concept、tone、constraints，固定在 core 中 |
| Sleep-time compute | “Async memory work” | 第二个 agent 在 critical path 外做 consolidation |
| Core / Recall / Archival | “Tiers” | 三层 memory split：always-visible / conversation / external |
| Block limit | “Cap” | 每个 block 的 character limit；迫使 summarization |
| Native reasoning | “Thinking channel” | Provider-level reasoning output，不是 prompt-level `Thought:` |
| Learned context | “Sleep output” | Sleep-time agent 写入 shared blocks 的 facts |

## 延伸阅读

- [Letta, Memory Blocks blog](https://www.letta.com/blog/memory-blocks)：block pattern
- [Letta, Sleep-time Compute blog](https://www.letta.com/blog/sleep-time-compute)：async consolidation
- [Letta, Rearchitecting the Agent Loop](https://www.letta.com/blog/letta-v1-agent)：native reasoning rewrite
- [Packer et al., MemGPT (arXiv:2310.08560)](https://arxiv.org/abs/2310.08560)：起源
