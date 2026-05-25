# Parallel / Swarm / Networked Architectures

> 与 supervisor 相反：没有 central decider。Agents 读取 shared event bus，异步领取工作，把结果写回。LangGraph 明确支持 “Swarm Architecture”，用于 decentralized、dynamic environments。Matrix（arXiv:2511.21686）把 control flow 和 data flow 都表示为通过 distributed queues 传递的 serialized messages，从而消除 orchestrator bottleneck。取舍很明确：用 determinism 和 traceability 换 scalability。Swarm 适合有大量 independent sub-problems 的任务；不适合需要单一 coherent plan 的任务。

**类型：** 学习 + 构建
**语言：** Python（stdlib，`threading`，`queue`）
**前置要求：** 阶段 16 · 05（Supervisor Pattern），阶段 16 · 04（Primitive Model）
**时间：** ~75 分钟

## 问题

Supervisor 能扩展到几个 workers。那几百个呢？supervisor 本身会变成 bottleneck：每个“谁做什么”的决定都通过一个 agent。一个缓慢的 plan step 会拖住整个系统。

Swarm architectures 反转设计。不是 central planner 派发工作，而是 workers 从 shared queue 领取工作。“coordination” 被烘进 event bus semantics。没有 orchestrator；系统扩展到 queue 能承受的程度。

## 概念

### 形状

```
                ┌──── shared queue ────┐
                │                      │
       ┌────────┼────────┐  ◄──────┬───┘
       ▼        ▼        ▼         │
     Worker  Worker  Worker   Worker
      A       B       C        D
       │        │        │         │
       └────────┴────────┴─────────┘
                 │
                 ▼
            results pool
```

没有 orchestrator。每个 worker 重复：pull a task、process、write result（并可选地 enqueue follow-ups）。

### Swarm 适合什么时候

- **大量 independent tasks。** scraping、transforming、classifying。任务之间不互相依赖。
- **可变时长工作。** 如果一些 tasks 花 100ms，另一些花 10s，swarm 会自动 balance load — 快 workers 拉下一个 jobs。supervisor 必须预估 duration。
- **吞吐优先于 determinism。** 你关心 total completion time，而不是 strict ordering。

### Swarm 什么时候失败

- **Ordered workflows。** 如果 step 3 需要 step 2 的输出，swarm 可能让 step 3 在 step 2 完成前触发。
- **Global-plan tasks。** 复杂 research questions 受益于 planner。一群 researchers 会产出独立 facts，而不是 coherent report。
- **Debugging。** 没有 central log 且 work 异步，复现 bug 很贵。

### Matrix（arXiv:2511.21686）

Matrix 是 2025 年把 swarm 推到自然终点的论文：control flow 和 data flow 都是 distributed queues 上的 serialized messages。没有 central coordinator。fault tolerance 来自 message durability。scalability 是 message broker 的问题，不是系统的问题。

贡献：一种 programming model，其中 multi-agent coordination 是“这个 agent 订阅哪个 message topic？”而不是“supervisor 选择哪个 agent 下一个？”这让系统看起来像 pub/sub event mesh。

### LangGraph 的 Swarm Architecture

LangGraph 2025 docs 明确把 “Swarm Architecture” 描述为 multi-agent patterns 之一：agents 是 nodes，但 edges 构成带 cycles 的 directed graph，任何 node 都可从 pool 中被激活。worker 根据 condition 从 available work 中选择，而不是由 supervisor assignment。

### Failure mode：starvation 和 hot-spotting

如果所有 workers 都拉取最快可用 task，long-running tasks 可能一直没人选，直到只剩它们。经典 queue starvation。

缓解：
- 带 explicit aging 的 priority queues（随等待时间提高 priority）。
- Worker specialization：一些 workers 只接 “long” tasks。
- Back-pressure：限制进入 queue 的 fast tasks 数量。

### 与 content-based routing 的联系

Swarm 天然适合 content-based routing（第 22 课）。不是一个 generic queue，而是每种 message type 一个 queue。Specialist workers 只订阅自己的 type。这是能扩展到数千 agents 的 message-bus architectures 的基础。

## 构建它

`code/main.py` 实现了 4 个 worker threads 从 shared `queue.Queue` 中拉取工作的 swarm。Tasks 有可变 durations（有些快，有些慢）。demo 对比：

- **Sequential baseline：** 一个 worker 串行处理所有 tasks。
- **Fixed assignment：** 每个 task 预先分配给特定 worker（supervisor-style）。
- **Swarm：** workers 从 shared queue 拉取。

Swarm 会自动 balance load；fixed assignment 会让快 workers 在自己的 assigned task 很慢时闲置。

运行：

```
python3 code/main.py
```

输出展示每个 worker 的 task counts（swarm 分布不均但最优）和 wall-clock times。

## 使用它

`outputs/skill-swarm-fit.md` 评估任务应该使用 swarm 还是 supervisor。输入：task independence、duration variance、ordering requirements、debuggability needs。

## 发布它

Checklist：

- **Priority queue with aging。** 防止 long-task starvation。
- **Worker idempotency。** 如果 worker mid-run 崩溃，一个 task 可能被拉取多次。Workers 必须 idempotent。
- **Durable queue。** Production 使用 Kafka、Redis Streams 或 database-backed queue。`queue.Queue` 只是 in-memory。
- **Observability per task。** 每个 task 有 trace ID；每个 worker 记录 start/end。
- **Back-pressure。** 如果 queue 增长快于 workers 消耗，就减慢 producer。

## 练习

1. 运行 `code/main.py`。在 variable-duration workload 上，swarm 比 sequential 快多少？比 fixed assignment 快多少？
2. 添加 priority queue variant（使用 `queue.PriorityQueue`）。按 task “importance” 字段分配 priority。观察 continuous load 下 low-priority tasks 是否会 starve。
3. 实现 hot-spot detector：当任意 worker 处理 task 数是最慢 worker 的 3× 时记录。它说明了 task-duration distribution 的什么？
4. 阅读 Matrix paper（arXiv:2511.21686）abstract 和 Section 3。识别 Matrix 接受的一个具体 tradeoff（scalability gain）和放弃的一个东西（traceability、determinism）。
5. 把 swarm demo 改成使用 `(task_type, payload)` tuples 的 `queue.Queue`，workers 只订阅特定 types。任务 heterogeneous 时什么 routing rules 合理？

## 关键词汇

| 术语 | 人们常说 | 实际含义 |
|------|----------------|------------------------|
| Swarm architecture | “Decentralized agents” | Workers 从 shared queue 拉取；没有 central orchestrator。 |
| Event bus | “Agents subscribe to topics” | 按 type 或 content 把 tasks 路由到 workers 的 message broker。 |
| Starvation | “Task never runs” | 低 priority task 因更高 priority work 持续到来而永远不被选中。 |
| Hot-spotting | “一个 worker 被淹没” | load imbalance：一个 worker 拿到大多数 tasks。 |
| Back-pressure | “减慢 producer” | 当 queue 填满时通知 upstream 停止生产的机制。 |
| Idempotent worker | “安全重跑” | task 处理两次也产出相同结果。worker 可能 mid-run 崩溃，因此必需。 |
| Durable queue | “crash 后仍存在” | 由 disk 或 replicated storage 支撑的 queue；worker 崩溃时 tasks 不丢失。 |
| Matrix framework | “Full message-passing swarm” | data 和 control flow 都是 distributed queues 上的 serialized messages。 |

## 延伸阅读

- [LangGraph workflows and agents — Swarm Architecture](https://docs.langchain.com/oss/python/langgraph/workflows-agents) — 显式 swarm support
- [Matrix — A Decentralized Framework for Multi-Agent Systems](https://arxiv.org/abs/2511.21686) — full message-passing swarm
- [Anthropic engineering — why supervisor not swarm in Research](https://www.anthropic.com/engineering/multi-agent-research-system) — 一个具体 production system 为什么明确选择 supervisor 而不是 swarm
- [AutoGen v0.4 actor-model docs](https://microsoft.github.io/autogen/stable/) — event-driven actor rewrite，比 v0.2 GroupChat 更接近 swarm
