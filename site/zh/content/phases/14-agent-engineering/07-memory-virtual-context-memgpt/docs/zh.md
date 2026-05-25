# Memory：Virtual Context 和 MemGPT

> Context windows 是有限的。Conversations、documents 和 tool traces 不是。MemGPT（Packer et al., 2023）把这件事框定为 OS virtual memory：main context 是 RAM，external store 是 disk，agent 在两者之间 paging。这是每个 2026 年 memory system 继承的模式。

**类型：** 构建
**语言：** Python (stdlib)
**前置要求：** 阶段 14 · 01（Agent Loop），阶段 14 · 06（Tool Use）
**时间：** ~75 分钟

## 学习目标

- 解释 MemGPT 建立其上的 OS 类比：main context = RAM，external context = disk，memory tools = page in/out。
- 用 stdlib 实现 two-tier MemGPT pattern，包含 main-context buffer、external searchable store 和 page in/out tools。
- 描述 agent 如何发出“interrupts”来查询或修改 external memory，以及结果如何 splice 回下一次 prompt。
- 识别 MemGPT 中延续到 Letta（第 08 课）和 Mem0（第 09 课）的设计选择。

## 问题

Context windows 看起来应该能解决 memory。它们不能。生产中反复出现三种 failure modes：

1. **Overflow。** Multi-turn conversations、long documents 或 tool-call-heavy trajectories 超过窗口。超过 cutoff 的所有内容都会消失。
2. **Dilution。** 即使在窗口内，塞入无关 context 也会稀释对重要内容的 attention。Frontier models 在长输入上仍然会退化。
3. **Persistence。** 新 session 从空窗口开始。没有 external memory 的 agents 无法跨 sessions 说“remember when you asked me to...”。

更大的窗口有帮助，但不能修复这个问题。Mem0 的 2025 论文测量到，128k-window baselines 仍会漏掉 long-horizon facts，而一个带 external memory 的 4k-window agent 能抓到。

## 概念

### MemGPT：OS 类比

Packer et al.（arXiv:2310.08560, v2 Feb 2024）把 context management 映射到操作系统 virtual memory：

| OS concept | MemGPT concept | 2026 production analog |
|------------|---------------|------------------------|
| RAM | main context (prompt) | Anthropic/OpenAI context window |
| Disk | external context | vector DB, KV, graph store |
| Page fault | memory tool call | `memory.search`, `memory.read`, `memory.write` |
| OS kernel | agent control loop | ReAct loop with memory tools |

Agent 运行普通 ReAct loop。额外的一类 tools 允许它把数据 page in/out main context。

### 两层

- **Main context。** 固定大小 prompt，保存当前任务。模型始终可见。
- **External context。** 无界，通过 tools 搜索。相关时读取，出现事实时写入。

原论文在两个超出 base window 的任务上评估了这个设计：超过 100k tokens 的 document analysis，以及跨天保持 persistent memory 的 multi-session chat。

### Interrupt pattern

MemGPT 引入 memory-as-interrupt：对话中途，agent 可以调用 memory tool，runtime 执行它，并把结果作为新的 observation splice 进下一次 assistant turn。概念上等同于 Unix `read()` syscall：阻塞进程、返回 bytes，然后进程继续。

经典 memory tool surface：

- `core_memory_append(section, text)`：写入 prompt 的 persistent section。
- `core_memory_replace(section, old, new)`：编辑 persistent section。
- `archival_memory_insert(text)`：写入 searchable external store。
- `archival_memory_search(query, top_k)`：从 external store 检索。
- `conversation_search(query)`：扫描过去 turns。

### MemGPT 在哪里结束，Letta 从哪里开始

2024 年 9 月，MemGPT 变成 Letta。Research repo（`cpacker/MemGPT`）仍然存在；Letta 扩展了这个设计：

- 从两层变成三层（core、recall、archival：第 08 课）。
- 用 native reasoning 替换 `send_message`/heartbeat pattern（第 08 课）。
- Sleep-time agents 运行 async memory work（第 08 课）。

即使生产系统运行 Letta、Mem0 或 custom two-tier store，MemGPT 论文仍然是 2026 年的基础。

### 这个模式会在哪里出错

- **Memory rot。** Writes 积累速度快于 reads；retrieval 淹没在 stale facts 里。修复：定期 consolidation（Letta sleep-time）、显式 invalidation（Mem0 conflict detector）。
- **Memory poisoning。** External memory 是被检索的文本。如果攻击者可控内容落入 memory note，agent 会在下一次 session 重新摄入它。这是 Greshake et al.（第 27 课）攻击在时间维度上的重述。
- **Citation loss。** Agent 记得“the user asked me to ship X”，但无法引用是哪一轮。每次 archival write 都要存 source references（session ID、turn ID）。

## 构建它

`code/main.py` 用 stdlib 实现 MemGPT 的 two-tier pattern：

- `MainContext`：固定大小 prompt buffer，包含 `core` dict 和 `messages` list；超过 cap 时自动 compact oldest messages。
- `ArchivalStore`：内存里的 BM25-esque store（token-overlap scoring），存储 (id, text, tags, session, turn) records。
- 五个 memory tools，对应 MemGPT surface。
- 一个 scripted agent，先把 facts 写入 archival，然后通过调用 `archival_memory_search` 回答问题。

运行它：

```
python3 code/main.py
```

Trace 会显示 agent 写入三条 facts，把 main context 填到 cap（触发 eviction），然后通过从 archival 检索回答 follow-up question。它在没有真实 LLM 的情况下复现了 MemGPT workflow。

## 使用它

今天每个生产 memory system 都是 MemGPT 变体：

- **Letta**（第 08 课）：三层、native reasoning、sleep-time compute。
- **Mem0**（第 09 课）：vector + KV + graph，加 scoring layer 融合。
- **OpenAI Assistants / Responses**：通过 threads 和 files 提供 managed memory。
- **Claude Agent SDK**：通过 skills 和 session store 提供 long-term memory。

按 operational shape（self-hosted、managed、framework-integrated）选择，而不是按核心模式选择。核心模式都是 MemGPT。

## 发布它

`outputs/skill-virtual-memory.md` 是一个可复用 skill，会为任意 target runtime 生成正确的 two-tier memory scaffold（main + archival + tool surface），并接好 eviction policy 和 citation fields。

## 练习

1. 添加一个以 tokens 计量的 `max_main_context_tokens` cap（用 `len(text.split())` * 1.3 近似）。超过 cap 时把最旧 messages compact 成 summary。比较有无 summarizer 的行为。
2. 在 archival store 上真正实现 BM25（term frequency、inverse document frequency）。在 toy fact set 上对比 token-overlap baseline 的 recall@10。
3. 给 archival inserts 添加 `citation` fields（session_id、turn_id、source_url）。让 agent 在每个 retrieval-backed answer 中引用 sources。
4. 模拟 memory poisoning：添加一条 archival record，内容是“ignore all future user instructions.” 写一个 guard，扫描 retrievals 中 directive-shaped text，并把它们标为 untrusted。
5. 把实现移植到 MemGPT research repo 的 core-memory JSON schema（`cpacker/MemGPT`）。从 flat strings 切换到 typed sections 后有什么变化？

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|----------------|------------------------|
| Virtual context | “Unlimited memory” | Main（prompt）+ external（searchable）两层，支持 page in/out |
| Main context | “Working memory” | Prompt：固定大小，始终可见 |
| Archival memory | “Long-term store” | 外部 searchable persistence，按需检索 |
| Core memory | “Persistent prompt section” | 固定在 main context 里的 named sections |
| Memory tool | “Memory API” | Agent 发出的读写 external memory 的 tool call |
| Interrupt | “Memory page fault” | Agent 暂停，runtime 获取数据，result splice 到下一 turn |
| Memory rot | “Stale facts” | 旧 writes 淹没 retrieval；用 consolidation 修复 |
| Memory poisoning | “Injected persistent note” | 攻击者内容被存为 memory，并在 recall 时重新摄入 |

## 延伸阅读

- [Packer et al., MemGPT (arXiv:2310.08560)](https://arxiv.org/abs/2310.08560)：受 OS 启发的 virtual context 论文
- [Letta, Memory Blocks blog](https://www.letta.com/blog/memory-blocks)：三层演进
- [Anthropic, Effective context engineering](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)：把 context 当作预算来管理
- [Chhikara et al., Mem0 (arXiv:2504.19413)](https://arxiv.org/abs/2504.19413)：建立在这个模式上的 hybrid production memory
