# DualPipe Parallelism

> DeepSeek-V3 在 2,048 块 H800 GPUs 上训练，MoE experts 分散在各个节点。跨节点 expert all-to-all communication 的成本是每 1 GPU-hour compute 对应 1 GPU-hour comm。GPU 有一半时间闲着。DualPipe（DeepSeek，2024 年 12 月）是一种 bidirectional pipeline，把 forward 和 backward computation 与它们触发的 all-to-all comms 重叠起来。Bubbles 下降，throughput 上升，而保留两份 model-parameter copies（名字中的 “dual”）在 Expert Parallelism 已经把 experts 分散到各 ranks 之后代价并不高。本课是 Learn 类型的 walkthrough，解释 DualPipe 实际做了什么，以及为什么 Sea AI Lab 的 DualPipeV refinement 会以稍微更紧的 bubble 为代价，去掉 2x parameter cost。

**类型：** 学习
**语言：** Python（stdlib，schedule simulator）
**先修：** Phase 10 · 05（distributed training、FSDP、DeepSpeed）、Phase 10 · 14（open-model architectures and MoE）
**时间：** 约 60 分钟

## 学习目标

- 说出 DualPipe forward-backward chunk 的四个组成部分，以及为什么每个组成部分都有自己的 overlap window。
- 解释 scale 下的 pipeline bubble problem，以及“bubble-free”在实践中与营销话术中的区别。
- 手动追踪 8 个 PP ranks 和 16 个 micro-batches 的 DualPipe schedule，并确认 forward 与 reverse streams 会填满彼此的 idle slots。
- 说明 DualPipeV（Sea AI Lab, 2025）的 tradeoff：在 Expert Parallelism 不活跃时，以稍大的 bubble 为代价去掉 2x parameter replication。

## 问题

在 2k H800 GPUs 上训练 671B MoE model 会遇到三个叠加瓶颈：

1. **Memory pressure。** 每块 GPU 持有模型的一片。序列 8k、61 层、128 heads 时 activation memory 非常巨大。
2. **Pipeline bubbles。** 传统 pipeline parallelism（GPipe、1F1B）会让 GPU 在等待本 stage 的 input 或 gradient 时闲置。8 stages 下，即使使用 1F1B scheduling，约 12% GPU 时间也可能是 bubble。
3. **Cross-node all-to-all。** 带 expert parallelism 的 MoE 会把 experts 分散到节点之间。每次 forward pass 都触发一次 all-to-all，把 tokens dispatch 到它们的 experts，再触发另一次 combine。在 2k GPUs 下，这很容易变成 1:1 compute-to-comm ratio。

这些问题各有解决方案：gradient checkpointing 解决 memory，Zero Bubble（Sea AI Lab, 2023）解决 pipeline bubbles，expert-parallel comm kernels 解决 all-to-all。DualPipe 做的是让它们协同工作。它在单个 forward-backward chunk 内重叠 compute 和 comm，从 pipeline 两端同时注入 micro-batches，并用得到的 schedule 把 all-to-all 隐藏在 compute windows 里。

报告结果：在 DeepSeek-V3 的 14.8T-token training run 中，pipeline bubbles 几乎被消除，GPU utilization 超过 95%。

## 概念

### Pipeline parallelism refresher

把一个 N-layer model 分到 P 个设备上。设备 `i` 持有层 `i * N/P .. (i+1) * N/P - 1`。一个 micro-batch 从设备 0 到 P-1 做 forward，然后从 P-1 到 0 做 backward。每个设备只有在前一个设备发来 output 后才能开始自己的 forward stage，也只有在下游设备发来 upstream gradient 后才能开始 backward。

GPipe（Huang et al., 2019）一次调度一个 micro-batch，会浪费大部分 GPU 时间。1F1B（Narayanan et al., 2021）为多个 micro-batches 交错 forward 和 backward passes。Zero Bubble（Qi et al., 2023）把 backward pass 拆成两部分：backward-for-input（B）和 backward-for-weights（W），并调度它们来填充 bubble。Zero Bubble 之后，pipeline 已经接近紧凑。

DualPipe 是下一步。它在此基础上加了两个想法：

### Idea 1：chunk decomposition

每个 forward chunk 被拆成四个组成部分：

- **Attention。** Q/K/V projections、attention、output projection。
- **All-to-all dispatch。** 把 tokens 发送给它们 experts 的跨节点通信。
- **MLP。** MoE expert computation。
- **All-to-all combine。** 把 expert outputs 带回来的跨节点通信。

Backward chunk 会包含每个部分的 gradient 版本。DualPipe 调度它们，使 all-to-all dispatch 与下一个 chunk 的 attention compute 并行，all-to-all combine 与后续 chunk 的 MLP compute 并行。

### Idea 2：bidirectional scheduling

大多数 pipeline schedules 从 stage 0 注入 micro-batches，并流向 stage P-1。DualPipe 从两端都注入 micro-batches。Stage 0 看到从那里发起的 forward micro-batches；stage P-1 也看到从那里发起的 forward micro-batches。两股 streams 在中间相遇。

要做到这一点，设备 `i` 必须同时持有 early-pipeline layer `i` 和 late-pipeline layer `P - 1 - i`。这就是 DualPipe 中 “dual” 的部分：每个设备保留两份它需要服务的 model layers（每个方向一份）。在 DeepSeek-V3 的规模下，这是 2x parameter replication cost。它之所以负担得起，是因为 Expert Parallelism 已经把 MoE experts 分散得很薄，复制两份 non-expert layers 只是小菜一碟。

关键点是，一个方向的 forward stream 和另一个方向的 backward stream 正好在单向 schedule 产生 bubble 的地方重叠。Bubbles 消失了。

### 手动追踪一个 schedule

考虑 P = 4 ranks、8 micro-batches，分成 4 个 forward / 4 个 reverse。时间从左到右移动；行是 device ranks。

```
           Time →
rank 0:  F1 F2 F3 F4  F5R F6R F7R F8R  B1 B2 B3 B4  ...
rank 1:     F1 F2 F3  F4/F5R F6R F7R   B1 B2 ...
rank 2:        F1 F2  F3/F5R F4/F6R    B1 ...
rank 3:           F1  F2/F5R F3/F6R    ...
```

读 “F4/F5R” 这个记号：rank 1 在同一个 time slot 中同时运行 micro-batch 4 的 forward（在 pipeline 中从左到右）以及 micro-batch 5 的 forward（从右到左）。这就是 “bidirectional” 在操作层面的含义。

在 rank 2 上，两股 streams 更早重叠；在 rank 0 和 P-1 上，它们最晚重叠。在 schedule 的稳定中间阶段，每个 rank 都运行一个方向的 forward，并与另一个方向的 backward 重叠。Compute 很忙。Forward pass 的 all-to-all dispatches 被隐藏在 backward compute 里。All-to-all combines 被隐藏在 forward compute 里。Bubbles 被挤出去。

### Bubble accounting

标准 1F1B pipeline bubble（每个 rank 浪费的时间）：

```
bubble_1F1B = (P - 1) * forward_chunk_time
```

Zero Bubble refinement 会把它降下来，但不是降到零。DualPipe 在稳定阶段，如果 micro-batch count 能被 pipeline depth 的 2 倍整除，就有 zero bubble。在稳定阶段之外（warmup 和 cooldown），仍然有一些 bubble，但它不会随 micro-batches 数量增长。这是论文强调的关键性质。

营销术语：“bubble-free”。技术术语：bubbles 不会随 micro-batch count 增长。Sea AI Lab 的后续分析（DualPipeV / Cut-in-half）显示，只有在 Expert Parallelism 不是瓶颈时才有完整 zero-bubble；当 EP-driven all-to-all 存在时，总会有一些 scheduling compromise。

### DualPipeV：refinement

Sea AI Lab（2025）观察到，当 EP comm overlap 不是重点时，2x parameter replication 是浪费。他们的 DualPipeV schedule 把 bidirectional injection 折叠成一种 “V-shape” schedule，运行在单份 parameter copy 上。Bubble 比 DualPipe 略大，但 memory savings 很可观。DeepSeek 在它们的开源 DualPipe implementation 中采用 DualPipeV 作为 EP-off mode。

Tradeoff：

| 特性 | DualPipe | DualPipeV | 1F1B | Zero Bubble |
|---------|---------|-----------|------|------------|
| 每设备 parameter copies | 2 | 1 | 1 | 1 |
| Bubble vs micro-batches | constant | small growth | grows | grows |
| Compute-comm overlap | full | partial | minimal | partial |
| 使用场景 | EP-heavy MoE | dense 或 EP-light | baseline | 任意 pipeline |

### 对一个 14.8T-token run 的意义

DeepSeek-V3 的 pre-training 在 2,048 H800 GPUs 上消耗了 14.8T tokens，约 2.8M GPU-hours。使用 naive 1F1B 时，他们本会把其中 12-15% 浪费在 pipeline bubbles 上，也就是 340-420K GPU-hours，足够训练一个完整 70B model。DualPipe 拿回了其中大部分。没有内部 logs 很难直接量化贡献，但论文中的 claim 是训练平均 GPU utilization 超过 95%。

对更小的 runs（小于 1k GPUs）来说，DualPipe 过重：pipeline bubbles 相对总成本更小，dense-model training 也很少撞上 all-to-all bottleneck。对 multi-thousand GPU scale 的 frontier MoE training 来说，它基本是必需的。

### 它在 stack 中的位置

- 与 **FSDP**（Phase 10 · 05）互补。FSDP 在 ranks 之间 shard model parameters；DualPipe 在 ranks 之间调度 compute。它们可以组合。
- 兼容 **ZeRO-3** gradient sharding。两份 copy replication 的 bookkeeping 需要与 ZeRO 的 sharded gradients 协作。
- 需要针对具体 cluster topology 调优的 **custom all-to-all kernels**。DeepSeek 的 open-source kernels 是 reference implementation。

## 使用

`code/main.py` 是一个 pipeline schedule simulator。它接收 `(P, n_micro_batches, schedule)`，并打印 1F1B、Zero Bubble、DualPipe 和 DualPipeV 的 stable-phase utilization。它是教学工具：数字匹配论文中的定性 claims，但不是 production measured speedup claim。

Simulator 的价值：用不同 P 和 micro-batch counts 跑它，观察 1F1B 的 bubble fraction 如何增长，而 DualPipe 不会。

真实 training run 的集成考虑：

- 选择一个能整除 micro-batch count 的 pipeline-parallel depth。
- 确保你的 expert-parallel mesh 支持 bidirectional all-to-all。DeepSeek 的 kernels 是 reference。
- 第一次调 schedule 时，预计花一周 debugging。Bookkeeping 很琐碎。
- 监控每个 rank 的 GPU utilization，而不只是 aggregate。DualPipe 的收益来自拉紧 stragglers。

## 交付

本课会产出 `outputs/skill-dualpipe-planner.md`。给定一个 training cluster specification（GPU count、topology、interconnect、model shape），它会推荐 pipeline parallelism strategy、要用的 scheduling algorithm，以及目标规模下的 expected bubble fraction。

## 练习

1. 在 `(P=8, micro_batches=16, schedule=dualpipe)` 和 `(P=8, micro_batches=16, schedule=1f1b)` 上运行 `code/main.py`。计算 GPU utilization difference，并把它表达为每百万 training tokens 能回收的 GPU-hours。

2. 手动画出 `(P=4, micro_batches=8, schedule=dualpipe)` 的 schedule table。用 micro-batch ID 和 direction 标记每个 time slot。找出第一个没有 bubbles 的 time slot。

3. 阅读 DeepSeek-V3 technical report（arXiv:2412.19437）的 Figure 5。找出 DualPipe forward chunk 中 all-to-all dispatch 的 overlap window。解释 compute schedule 如何隐藏它。

4. 计算 DualPipe 对一个 P=8 pipeline stages 的 70B dense model 和一个 P=16 pipeline stages 的 671B MoE model 的 2x parameter overhead。说明为什么 MoE case 的 overhead 比例更小（大多数参数是 experts，并在大 EP group 中被 sharded）。

5. 比较 DualPipe 与 Chimera（2021 年的 competing bidirectional scheduler）。以论文 Section 3.4 为 reference，找出 DualPipe 增加而 Chimera 没有的两个具体属性。

## 关键术语

| 术语 | 人们怎么说 | 它真正的意思 |
|------|----------------|------------------------|
| Pipeline bubble | “Idle time per rank” | Pipeline stage 等待 input 或 gradient 时浪费的 GPU cycles |
| 1F1B | “Default pipeline schedule” | One forward / one backward interleaved scheduling；DualPipe 击败的 baseline |
| Zero Bubble | “Sea AI Lab 2023” | 把 backward 拆成 B（input gradient）和 W（weight gradient）；几乎完全拉紧 pipeline |
| DualPipe | “DeepSeek-V3 schedule” | Bidirectional pipeline + compute-comm overlap；bubbles 不随 micro-batch count 增长 |
| DualPipeV | “Cut-in-half” | V-shape refinement，去掉 2x parameter replication，代价是略大的 bubbles |
| Chunk | “Unit of pipeline work” | 一个 micro-batch 通过一个 pipeline stage 的 forward 或 backward pass |
| All-to-all dispatch | “Send tokens to experts” | 把 tokens 路由到指定 MoE experts 的跨节点通信 |
| All-to-all combine | “Bring expert outputs back” | MLP 后收集 expert outputs 的跨节点通信 |
| Expert Parallelism（EP） | “Experts across GPUs” | 在 ranks 之间 shard MoE experts，让不同 GPUs 持有不同 experts |
| Pipeline Parallelism（PP） | “Layers across GPUs” | 在 ranks 之间 shard model layers；DualPipe 调度的维度 |
| Bubble fraction | “Wasted GPU time” | (bubble_time / total_time)；DualPipe 试图推近零的比例 |

## 延伸阅读

- [DeepSeek-AI — DeepSeek-V3 Technical Report (arXiv:2412.19437), Section 3.3.2 and Figure 5](https://arxiv.org/abs/2412.19437) — 主要 DualPipe reference
- [DeepSeek — DualPipe GitHub repository](https://github.com/deepseek-ai/DualPipe) — 开源 reference implementation，包含 DualPipeV（Cut-in-half）mode
- [Qi et al. — Zero Bubble Pipeline Parallelism (arXiv:2401.10241, Sea AI Lab 2023)](https://arxiv.org/abs/2401.10241) — Zero Bubble predecessor
- [Sea AI Lab — DualPipe could be better without the Dual](https://sail.sea.com/blog/articles/63) — 影响 DeepSeek EP-off mode 的 DualPipeV analysis
- [Narayanan et al. — PipeDream / 1F1B (arXiv:1806.03377, 2018-2021)](https://arxiv.org/abs/1806.03377) — DualPipe 对比的 1F1B schedule
- [Huang et al. — GPipe (arXiv:1811.06965, 2018)](https://arxiv.org/abs/1811.06965) — 原始 pipeline parallelism paper 和 bubble problem
