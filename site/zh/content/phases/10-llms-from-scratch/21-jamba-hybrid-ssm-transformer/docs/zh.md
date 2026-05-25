# Jamba：Hybrid SSM-Transformer

> State space models（SSMs）和 transformers 想要的东西不同。Transformers 通过 attention 购买质量，代价是二次方成本。SSMs 通过 recurrence 购买 linear-time inference 和 constant memory，但质量落后。AI21 的 Jamba（2024 年 3 月）和 Jamba 1.5（2024 年 8 月）把它们放进同一个模型：每 7 个 Mamba layers 配 1 个 Transformer layer，每隔一个 block 使用 MoE，并提供能放进单块 80GB GPU 的 256k context window。Mamba-3（ICLR 2026）用 complex-valued state spaces 和 MIMO projections 收紧了 SSM 侧。本课会端到端阅读这两个架构，并解释为什么 hybrid recipe 在 pure-SSM 和 pure-Transformer long-context 尝试都没能持续时，经历了三年 scaling 仍然存活下来。

**类型：** 学习
**语言：** Python（stdlib，layer-mix calculator）
**先修：** Phase 10 · 14（open-model architectures）、Phase 10 · 17（native sparse attention）
**时间：** 约 60 分钟

## 学习目标

- 解释 Jamba block 中的三个 primitives：Transformer layers、Mamba layers、MoE，以及 1:7:even interleaving recipe。
- 从高层说明 SSM 的 recurrence 长什么样，以及为什么它支持 constant-memory inference。
- 计算 Jamba model 在 256k context 下的 KV cache footprint，并与 pure-Transformer model 的需求比较。
- 命名三个 Mamba-3 innovations（exponential-trapezoidal discretization、complex-valued state update、MIMO）以及它们各自针对的问题。

## 问题

Attention 对序列长度是二次方。State space models 是线性的。这个差异会叠加放大：在 256k tokens 下，一个 Transformer attention map 每个 head 有 65B entries；而 SSM 的 recurrent state 不管序列长度如何都是固定大小。

Pure-SSM models（Mamba、Mamba-2）在小规模上能匹配 Transformer perplexity，但在 state-tracking tasks 上落后，并在某些 in-context retrieval 类别上失败。直觉是：SSMs 会把历史压缩进一个 fixed state；当历史很长时，信息会泄漏。Attention 精确记住一切，但支付二次方成本。

显而易见的修复：两个都用。在需要 exact recall 的地方放 Transformer layers。其他地方用 SSM layers。调 ratio。Jamba 是第一个在 scale 上发布这个 hybrid recipe 的 production-grade model（52B total、12B active、256k context、单块 80GB GPU）。Jamba 1.5 把这个家族扩展到 398B total / 94B active。Mamba-3（ICLR 2026）是当前最佳 pure-SSM baseline，可以围绕它重建 hybrids。

本课阅读三篇论文，并产出“如何选择正确 ratio”的 mental model。

## 概念

### 一页讲清 SSM

State space model 通过 fixed-size state `h` 处理序列 `x_1, ..., x_N`：

```
h_t = A h_{t-1} + B x_t
y_t = C h_t
```

每一步 state 通过 linear dynamics `A` 演化，接收输入 `B x_t`，并发出输出 `C h_t`。`A, B, C` 都可以学习。注意关键属性：计算 `y_t` 只需要 `h_{t-1}` 和 `x_t`，不需要任何更早的 `x`。Memory 是 constant。Inference 每个 token 是 O(1)。

建模质量的 trick 在于 `A` 的结构。S4（Gu 2021）使用了高度结构化的矩阵，可以在训练中高效地作为 long convolution 计算。Mamba（Gu, Dao 2023）把固定的 `A, B, C` 换成 data-dependent 版本（“selective” 部分）。Mamba-2（2024）进一步简化了结构。Mamba-3（2026）在特定位置重新加回复杂性。

关键属性：对 decoder LLM 来说，SSM layer 是 attention layer 的 drop-in replacement，只是用 fixed-size per-layer state 替代不断增长的 KV cache。

### Jamba block

Jamba block 根据两个数字交错 layers：

- `l`：attention-to-Mamba ratio。Jamba 使用 `l = 8`，表示每 7 个 Mamba layers 配 1 个 Transformer layer（7 Mamba + 1 Attention = 每组 8 layers）。
- `e`：MoE frequency。Jamba 使用 `e = 2`，表示每隔一层应用 MoE。

Block 内的 layer sequence：

```
M  M  M  M  M  M  M  A    (7 Mamba + 1 Attention)
|  M  |  M  |  M  |  M    (where | marks MoE applied)
```

每个 Jamba block 是 8 layers。4 blocks deep（总计 32 layers）时，你得到 28 个 Mamba 和 4 个 Attention layers。其中 16 个使用 MoE。

### 为什么是 1:7 ratio

AI21 做了 ablations：什么 attention-to-Mamba ratio 在他们的 long-context evals 上给出最佳 perplexity-per-parameter 和 in-context recall？

- Attention 太多（1:1）：质量上升，但 memory 和 speed 恶化。
- Attention 太少（1:15）：memory 很好，但 in-context retrieval 失败。
- Sweet spot：1:7 或 1:8。

直觉：Transformer layers 处理 exact recall 和 state tracking。Mamba layers 处理低成本的大部分 processing。

### Positional encoding

Mamba layers 通过 recurrence 自带 position-awareness。最初基于 Mamba 的 hybrids 中，attention layers 没有使用 RoPE：SSM layers 提供了 position info。Jamba 1.5 为 attention layers 增加 RoPE，以改善 longer-context generalization，这是基于经验 long-context evaluation 的事后 refinement。

### Memory budget

对 Jamba-1 形状（32 layers：28 Mamba + 4 Attention，hidden 4096，32 attention heads）：

- KV cache（只有 attention layers）：256k BF16 下 `2 * 4 * 32 * 128 * 256k * 2 = 8.4 GB`。只有 4 个 attention layers 贡献。
- SSM state：每个 token prefix 是 `28 * hidden * state_size`，但这是 fixed-size per layer，不随序列长度增长。典型 Mamba state 是每 feature 16，hidden 4096：总计 `28 * 4096 * 16 * 2 = 3.7 MB`。

与同样 hidden、32 layers、full MHA 32 heads 的 pure Transformer 比较：256k BF16 下 `2 * 32 * 32 * 128 * 256k * 2 = 128 GB`。KV cache 减少 8x。即使与大多数 2024 models 使用的 GQA(8) baseline 比较（`2 * 32 * 8 * 128 * 256k * 2 = 32 GB`），Jamba 的 1:7 hybrid 在 16 GB 处仍然小 2x。

这就是 AI21 所说“单块 80GB GPU 上 256k context”的含义。Full-MHA pure Transformer 的 KV cache 放不下；即使 GQA baseline 也几乎不给 weights 和 activations 留空间；Jamba 可以。

### Mamba-3：2026 年的 pure-SSM baseline

Mamba-3（ICLR 2026, arXiv:2603.15569）在 pure-SSM 侧引入三个 innovations：

1. **Exponential-trapezoidal discretization。** 用更 expressive recurrence 替代 Mamba-2 中的 Euler-method discretization。Convolution-like operation 被应用在 core recurrence 内部的 state-input 上，而不是作为 `x_t` 上的外部 convolution。

2. **Complex-valued state update。** 以前的 Mambas 把 state matrix 从 complex（S4）降为 real diagonal（Mamba），再降为 scaled identity（Mamba-2）。Mamba-3 重新加入 complex values，相当于在 state 上做 data-dependent rotary embedding。这恢复了此前 real-valued simplifications 损失的 state-tracking capabilities。

3. **Multi-input multi-output（MIMO）projections。** 不再使用 per-feature scalar projections，而是使用 matrix-valued projections。在不增加 decode latency 的情况下改善 modeling power 和 inference-time hardware utilization。

在 1.5B 参数上，Mamba-3 相比 Gated DeltaNet 平均 downstream accuracy 提升 0.6 points；MIMO variant 再增加 1.2，总计 1.8-point gain。在相同 state size 下，Mamba-3 用一半 state 匹配 Mamba-2。

Mamba-3 还没有在 production hybrid 中大规模发布，但它显然是下一代 Jamba-class model 中 SSM 侧的候选。

### 什么时候选择 hybrid

Hybrids 胜出于：

- Context 足够长，pure Transformer KV cache 开始痛苦（64k+）。
- 任务混合 short-range structure（适合 SSM）与 long-range recall（需要 Transformer）。
- 希望部署在 single-GPU memory budgets 中，而 Transformer KV cache 单独就放不下。

Hybrids 失败于：

- Context 很短（16k 以下）。SSM overhead 被浪费；pure Transformer 就很好。
- 任务需要 everywhere-to-everywhere attention（deep reasoning、multi-document cross-reference）。Hybrid 中稀疏的 attention layers 会伤害质量。
- 你正在 scale 到 trillion-parameter frontier models。Pure-Transformer + MLA + MoE（DeepSeek-V3 style）目前赢得 capability race。

### Competitive landscape

| Model | Family | Scale | Unique claim |
|-------|--------|------|-------------|
| Mamba-2 | pure SSM | 3B | linear time, constant memory |
| Jamba | hybrid | 52B/12B | 256k on 80GB |
| Jamba 1.5 Large | hybrid | 398B/94B | enterprise-grade long-context |
| Mamba-3 | pure SSM | 1.5B (paper) | state-tracking restored |
| DeepSeek-V3 | pure Transformer + MoE | 671B/37B | frontier capability |

2026 年的格局：pure-Transformer MoE 主导 frontier，但 hybrids 占据 256k-plus context niche。Mamba-3 的 state-tracking 胜利可能让下一代 hybrid ratios 更低（更多 SSM、更少 attention）。

## 使用

`code/main.py` 是 hybrid architectures 的 memory calculator。给定 SSM-Transformer ratio 和 hidden-size / layer-count config，它会计算：

- 目标 context 下的 KV cache。
- SSM state memory。
- 一系列 model shapes 在 context N 下的 total memory。

Calculator 支持：

- Pure-Transformer baseline（KV cache 随 N 增长）。
- Jamba-style 1:7 hybrid。
- Pure-SSM（完全没有 KV cache）。

数字来自 Jamba-1 和 Jamba-1.5 papers 的 published shapes，并对假设变体做 extrapolation。

真实 deployment 的集成考虑：

- 大多数 production inference servers（vLLM、SGLang）支持 Jamba 和 Mamba。检查具体版本。
- 在 256k context 下，Jamba 的 memory advantage 会体现在 concurrent-request throughput 上。相同 VRAM 中能放下更多 Jamba sequences，而不是 Transformer sequences。
- Mamba-3 作为 standalone model 还没有 production 发布：目前是 1.5B 的 research preview。

## 交付

本课会产出 `outputs/skill-hybrid-picker.md`。给定一个 workload specification（context length profile、task mix、memory budget），它会在 pure Transformer、Jamba-style hybrid 和 pure SSM 之间给出推荐，并明确说明 memory 与 quality tradeoffs。

## 练习

1. 运行 `code/main.py`，计算 32-layer pure Transformer（hidden 4096，32 heads）和同形状 Jamba-1 hybrid 在 256k context 下的 KV cache。验证 AI21 paper 声称的约 8x memory reduction。

2. 修改 calculator，建模 1:3 hybrid（4 Mamba : 1 Attention）和 1:15 hybrid（14 Mamba : 1 Attention）。绘制 KV cache vs ratio。在哪个 ratio 下 KV cache 等于 SSM state memory？

3. 阅读 Jamba paper（arXiv:2403.19887）的 Section 3。解释为什么 AI21 使用 Mamba-1 而不是更快的 Mamba-2。提示：hybrid ablation section 记录了这一点。

4. 计算 Jamba 1.5 Large（398B total，94B active）中 every-other-layer MoE 的 parameter overhead。把 active ratio 与 DeepSeek-V3（37B/671B）比较，并解释为什么 Jamba 架构会把 active ratio 推高。

5. 阅读 Mamba-3 paper（arXiv:2603.15569）的 Section 3。用三句话解释为什么 complex-valued state update 等价于 data-dependent rotary embedding。把答案和 Phase 7 · Lesson 04 的 RoPE derivation 联系起来。

## 关键术语

| 术语 | 人们怎么说 | 它真正的意思 |
|------|----------------|------------------------|
| State space model（SSM） | “Recurrence with a fixed state” | 带 learned recurrence `h_t = A h_{t-1} + B x_t` 的层；每个 token constant memory |
| Selective SSM | “Mamba's trick” | Data-dependent A、B、C parameters，让模型在 linear time 下拥有类似 gating 的 selectivity |
| Attention-to-Mamba ratio | “How many attention layers” | 在 Jamba 中，`l = 8` 表示每 7 个 Mamba layers 配 1 个 attention layer |
| Jamba block | “The 8-layer group” | 一个 attention + 七个 Mamba + alternate positions 上的 MoE |
| SSM state | “The hidden buffer” | 替代 Mamba layers 中 KV cache 的 fixed-size per-layer state |
| 256k context | “Jamba's flagship number” | Jamba-1 能装进单块 80GB GPU 的序列长度；pure Transformer 在这个大小下不行 |
| Mamba-3 | “2026 pure SSM” | 带 complex state + MIMO 的当前最佳 pure-SSM architecture；hybrids 重建时围绕的 baseline |
| MIMO | “Multi-input multi-output” | Mamba-3 innovation，用 matrix-valued projections 替代 per-feature scalar |
| Exponential-trapezoidal discretization | “Mamba-3's recurrence” | 更 expressive 的 recurrence，包含 Mamba-2 的 Euler-method discretization |
| Hybrid architecture | “Mix attention and SSM” | 任何交错 Transformer 和 SSM layers 的模型；Jamba 是 production archetype |

## 延伸阅读

- [Lieber et al. — Jamba: A Hybrid Transformer-Mamba Language Model (arXiv:2403.19887)](https://arxiv.org/abs/2403.19887) — 原始 Jamba paper，ratio ablations，256k context claim
- [AI21 — Jamba 1.5: Hybrid Transformer-Mamba at Scale (arXiv:2408.12570)](https://arxiv.org/abs/2408.12570) — scaled-up family，398B/94B 和 12B/52B public releases
- [Gu, Dao — Mamba: Linear-Time Sequence Modeling with Selective State Spaces (arXiv:2312.00752)](https://arxiv.org/abs/2312.00752) — Jamba 基于的 selective SSM paper
- [Dao, Gu — Mamba-2 (arXiv:2405.21060)](https://arxiv.org/abs/2405.21060) — simplified structured-state-space successor
- [Lahoti et al. — Mamba-3 (arXiv:2603.15569, ICLR 2026)](https://arxiv.org/abs/2603.15569) — complex-valued state、MIMO、2026 pure-SSM frontier
- [Gu et al. — Efficiently Modeling Long Sequences with Structured State Spaces (arXiv:2111.00396)](https://arxiv.org/abs/2111.00396) — S4 paper，LLM 中 SSM genealogy 的起点
