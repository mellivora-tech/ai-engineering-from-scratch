# Async 和 Hogwild! Inference

> Speculative decoding（Phase 10 · 15）在一个 sequence 内并行化 tokens。Multi-agent frameworks 在整个 sequences 之间并行化，但强制使用显式协调（voting、sub-task splitting）。Hogwild! Inference（Rodionov et al., arXiv:2504.06261）做的是另一件事：让同一个 LLM 的 N 个 instances 并行运行，并共享一个 key-value cache。每个 worker 都会即时看到其他 worker 生成的 tokens。现代 reasoning models：QwQ、DeepSeek-R1，可以通过这个 shared cache 自我协调，不需要任何 fine-tuning。这个方法仍是实验性的，但它打开了一个全新的 inference parallelism 轴线，与 spec decode 正交。本课会用 stdlib Python 实现一个 two-worker Hogwild! simulator，并解释为什么 shared-cache collaboration 会从已有模型的 reasoning abilities 中涌现。

**类型：** 构建
**语言：** Python（stdlib）
**先修：** Phase 10 · 12（inference optimization）、Phase 10 · 15（speculative decoding）
**时间：** 约 60 分钟

## 学习目标

- 描述三种常见 parallel-LLM topologies（voting、sub-task、Hogwild!），并说出每种针对什么问题。
- 说明 Hogwild! 的核心设置：multiple workers、one shared KV cache、通过 self-prompting 涌现 coordination。
- 把 Hogwild! 的 wall-time speedup 表达为 worker count `N`、task-level parallelism `p` 和 coordination overhead `c` 的函数。
- 在 toy problem 上实现 two-worker Hogwild! simulator，并观察 emergent task division。

## 问题

现代 LLM 通过生成很长的 reasoning chains 来解决难题：5000 tokens 的 step-by-step logic 很常见，深度数学问题上会有数万 tokens。在 70B model 上以 35 tokens/sec 解码，50k tokens 需要 24 分钟。交互性就没了。

Speculative decoding（Phase 10 · 15）通过在一个 sequence 内并行化，带来 3-5x speedup。再往上，自回归解码的 sequential dependency 就是硬天花板。每个新 token 都依赖之前所有 token。

显而易见的问题：能否跨 sequences 并行？在同一个问题上运行同一模型的多个 copies，让它们合作，并让它们分工？

已有工作：voting ensembles（运行 N 个模型，选 majority answer）、tree-of-thought（分叉 reasoning paths 并重组）、multi-agent frameworks（给每个 agent 分配 sub-task，使用 coordinator）。这些都在特定任务域有效。但它们也都会引入显式 coordination machinery：voting rules、branch-and-prune logic、agent-to-agent messaging protocols。

Hogwild! Inference 采用另一种方法。N 个 workers 共享单个 KV cache。每个 worker 会立即看到其他所有 worker 生成的 tokens，就像它们是自己的 context 一样。这些 workers 在没有任何 training 或 fine-tuning 的情况下，会自己想办法分工。现代 reasoning models（QwQ、DeepSeek-R1、Claude-family reasoning mode）可以阅读 shared cache，并说出类似“我看到 worker 2 已经处理了 base case，所以我来做 inductive step”的话。

截至 2026 年 4 月，speedup 依赖 workload 且仍处实验阶段。但这个想法值得知道，因为它打开了新的 inference parallelism 轴线。

## 概念

### 设置

初始化 N 个 worker processes，它们都运行同一个 LLM。不要 per-worker KV caches，而是维护一个 shared cache。当 worker `i` 生成 token `t_j` 时，这个 token 被写入 shared cache 的下一个位置。当 worker `k` 执行下一步时，它读取 cache 的当前状态（包含所有 N 个 workers 到目前为止生成的一切）。

Step time 上，workers 竞速写 tokens。没有 per-worker position index：cache 是单个不断增长的 sequence。顺序由 write arrival time 决定。

### 为什么 coordination 会涌现

Workers 共享一个 prompt。通常类似“你是 N 个一起解决此问题的 instances 之一。每个 instance 读取 shared memory，并能看到其他 instances 写了什么。避免重复工作。”Prompt 加 shared cache 就足够了。Reasoning models 会读取 cache，注意到问题的哪些部分已经被尝试过，并且（经常但不总是）转向未探索的部分。

Hogwild! paper（Rodionov et al., 2025）报告了如下观察：

- Workers 制定计划，并通过 cache 向其他 workers 通信。
- Workers 注意到其他 workers 推理中的错误，并指出它们。
- Workers 在计划失败时适应并提出 alternatives。
- 在 prompt 要求检查 redundancy 时，workers 能检测到并 pivot。

这些都不需要 fine-tuning。Emergent behavior 来自模型已经具备的 reasoning capabilities。

### 命名

论文名称借用了 Hogwild! SGD（Recht et al., 2011），一个 asynchronous-update optimizer。类比是：SGD 的异步 workers 都写入 shared parameter vector；Hogwild! Inference 的 workers 都写入 shared KV cache。二者都依赖 empirical convergence，而不是 synchronization guarantees。

### RoPE 让它可行

Rotary Position Embeddings（RoPE, Su et al. 2021）通过 Q 和 K vectors 中的 rotation 编码 position information。因为 positions 是 rotations，而不是 baked-in offsets，一个 token 的位置可以移动而不必重新计算 KV cache entry。当 worker `i` 写入 shared cache 的 position `p` 时，其他 workers 读取该位置可以直接使用 cached entry，不需要 re-rotation。

在 learned-position 或 absolute-position model 中，Hogwild! 每次 concurrent write 都需要 cache invalidation。RoPE 让 cache 保持稳定。

### Wall-time math

令 `T_serial` 为一个 worker 独自解决问题的时间。令 `p` 为 task-level parallelizable fraction。令 `c` 为 per-step coordination overhead（读取 extended cache，决定写什么）。

Single-worker time：`T_serial`。
N-worker Hogwild! time，如果 coordination 免费：`T_serial * ((1 - p) + p / N)`。经典 Amdahl。
加上 coordination overhead：`T_serial * ((1 - p) + p / N) + c * steps_per_worker`。

要让 worker 有生产力，`c` 必须相对 per-step decode time 很小。Reasoning models 生成 5k+ tokens 时，workers 可以承受数百 tokens 的 coordination overhead，仍然领先。短 chat tasks 中，coordination 占主导，Hogwild! 比 serial 更差。

### 具体例子

Reasoning problem：10k tokens 的 chain-of-thought。假设问题有 `p = 0.7` 的 parallelizable content（不同 proof strategies、不同 case analyses），并且每个 worker 的 `c = 200` tokens coordination overhead。使用 `N = 4` workers：

- Serial time：10000 decode steps。
- Hogwild! time：10000 * (0.3 + 0.7 / 4) + 200 * 4 = 10000 * 0.475 + 800 = 5550 decode steps。
- Speedup：10000 / 5550 = 1.8x。

这不算夸张。但在更长的 reasoning problems（50k tokens）上，coordination overhead 被摊薄，speedup 会推向 2.5-3x。Hogwild! 就像 inference 版的 thread-level parallelism，只是语言本身让你自然写 multi-threaded code。

### 什么时候使用 Hogwild!

- Long reasoning problems（数千 tokens），并且 task 可以跨 independent sub-goals 并行化。
- 已经被训练为 step-by-step 思考的 reasoning models。Non-reasoning models 自我协调不好。
- 有足够 VRAM 容纳 shared cache 加 N 个 worker processes 的 single-node deployments。Cache 是 shared，但每个 worker 有自己的 activation memory。

### 什么时候不要使用

- 短 interactive chat。Coordination overhead 占主导。
- 不能并行化的任务（单条 linear proof、单次 compilation）。N=1 是上限。
- Non-reasoning models。没有 coordination 涌现。
- Multi-node deployments。Shared cache 需要非常快的 cross-worker synchronization。Intra-node 可以；cross-node 是 latency disaster。

### 实验状态

截至 2026 年 4 月，Hogwild! 是一个带 open-source PyTorch implementation 的研究方法。Production adoption 尚未发生。三个 blockers：

1. 跨 concurrent processes 的 shared KV cache management 是不简单的工程。
2. Emergent coordination 依赖 task；benchmarks 仍在构建。
3. Speedups 相比 speculative decoding 已经提供的提升更温和；两者可以组合，但组合工程又多一层。

值得知道。值得实验。还不值得把产品押在上面。

## 构建

`code/main.py` 实现一个 toy Hogwild! simulator：

- 两个 worker processes，每个都是 deterministic “LLM”，按已知概率生成几类 tokens（work-token、observe-token、coordinate-token）。
- 一个 shared cache（只是 token list），两个 workers 都读取和写入。
- 简单 coordination logic：当一个 worker 看到另一个 worker 已经在某个 category 产生了足够 work tokens，它会选择不同 category。

Simulator 运行固定 step budget，并报告：

- Total work-tokens produced。
- Total wall time（worker steps 数量）。
- 相比 single worker 的 effective speedup。
- 哪个 worker 写了哪个 token 的 trace。

### Step 1：shared cache

两个 workers 都 append 的 list。真实实现中用简单 locking（Python `threading.Lock`）；这里用 counter 模拟。

### Step 2：worker loop

每个 worker 在每一步：

- 读取当前 shared cache。
- 根据已经存在的内容决定要写哪类 token。
- 写入一个 token。

### Step 3：coordination heuristic

如果 category X 在 cache 中已经有 K 个 tokens，而 worker 原本打算写 X，那么 worker 切换到 category Y。这是 reasoning-model 行为的 toy 替身：注意到“这里已经覆盖了，去做别的”。

### Step 4：measured speedup

用 N=1 worker 和 N=2 workers 运行 simulator，使用相同 total step budget。统计 work-tokens produced。N=2 应该产生大约 1.5-1.8x 的 work-tokens，因为 coordination-driven task division。

### Step 5：stress coordination

降低 coordination heuristic 的敏感度。再次运行。观察如果没有良好 coordination，N=2 会重复产生相同 tokens，speedup 降到 1 以下。这匹配论文观察：只有当 workers 有 reasoning capacity 进行 self-coordinate 时，这个 trick 才有效。

## 使用

截至 2026 年 4 月，Hogwild! production integration 仍是 research-grade。Yandex/HSE/IST 的 reference implementation 基于 PyTorch，并针对 DeepSeek-R1 和 QwQ models 的 single-node multi-process setups。

务实 adoption path：

1. Profile 你的 reasoning-task workload。衡量 exploratory tokens（multiple strategies、case analyses、search）与 linear tokens 的比例。
2. 如果 exploration 占主导，跑一个 two-worker Hogwild! experiment。衡量 wall-time improvement。
3. 如果 improvement 低于 1.3x，你处于 coordination-dominated regime。回到 single-worker。
4. 如果 improvement 超过 1.5x，推到 N=4 并再次测量。Diminishing returns 通常在 N=4-8 左右出现。

与 speculative decoding 组合：每个 Hogwild! worker 可以独立使用 spec decode。两个 speedups 会（大致）相乘，把 3x spec decode 和 1.8x Hogwild! 带到相对 naive single-worker decoding 的有效 5.4x。

## 交付

本课会产出 `outputs/skill-parallel-inference-router.md`。给定一个 reasoning workload profile（token budget、task parallelism profile、model family、deployment target），它会在 voting、tree-of-thought、multi-agent、Hogwild! 和 speculative decoding strategies 之间路由。

## 练习

1. 使用默认设置运行 `code/main.py`。确认在相同 wall time 内，N=2 Hogwild! configuration 比 N=1 baseline 产生更多 work-tokens。

2. 降低 coordination heuristic 的强度（设置 `coordination_weight=0.1`）。重新运行。展示 speedup collapses。解释原因：workers 在不能协调时会重复努力。

3. 计算一个 50k-token reasoning task 的 expected Hogwild! speedup，参数为 `p=0.8, c=500`，N=4 workers。再对一个 1k-token chat task 计算，参数为 `p=0.3, c=200`，N=4。为什么一个赢，一个亏？

4. 阅读 Hogwild! paper 的 Section 4（preliminary evaluation）。找出 authors 报告的两个 failure modes。描述更好的 coordination prompt 如何缓解每个问题。

5. 在 toy 中把 Hogwild! 与 speculative decoding 组合：每个 worker 内部使用 2-token spec-decode。报告 multiplicative speedup。当两个 workers 都想扩展同一个 shared-cache prefix 时，会出现什么 bookkeeping problem？

## 关键术语

| 术语 | 人们怎么说 | 它真正的意思 |
|------|----------------|------------------------|
| Hogwild! | “Parallel workers, shared cache” | 同一个 LLM 的 N 个 instances 并发运行，带一个 shared KV cache；通过 self-prompting 涌现 coordination |
| Shared KV cache | “The coordination medium” | 所有 workers 都读取和写入的单个增长 KV buffer；让 tokens 跨 workers 即时可见 |
| Emergent coordination | “No training needed” | 具备 reasoning 能力的 LLMs 可以读取 shared cache 并分工，不需要 fine-tuning 或显式 protocol |
| Coordination overhead（c） | “Tokens spent orienting” | 每个 worker 读取 extended cache 并决定做什么的成本；必须相对 total decode time 保持很小 |
| Parallelizable fraction（p） | “What can run in parallel” | Task-level parallelism：总工作中不是内在 sequential 的比例 |
| RoPE enables Hogwild! | “Rotary positions are shift-invariant” | 因为 positions 是 rotations，写入 shared cache 不需要重算 prior tokens |
| Voting ensemble | “Run N, pick the majority” | 最简单的 parallel inference topology；适合 classification，不太适合 long-form reasoning |
| Tree of thought | “Branch and prune” | 探索多个 branches 并 pruning 的 reasoning strategy；显式 coordination logic |
| Multi-agent framework | “Assign sub-tasks” | 每个 agent 得到一个 role；coordinator 进行编排；protocol overhead 很重 |

## 延伸阅读

- [Rodionov et al. — Hogwild! Inference: Parallel LLM Generation via Concurrent Attention (arXiv:2504.06261)](https://arxiv.org/abs/2504.06261) — Hogwild! paper，在 QwQ 和 DeepSeek-R1 上的 preliminary evaluation
- [Recht, Re, Wright, Niu — Hogwild!: A Lock-Free Approach to Parallelizing Stochastic Gradient Descent (arXiv:1106.5730, NeurIPS 2011)](https://arxiv.org/abs/1106.5730) — 原始 Hogwild!，命名来源
- [Su et al. — RoFormer: Enhanced Transformer with Rotary Position Embedding (arXiv:2104.09864)](https://arxiv.org/abs/2104.09864) — RoPE，让 shared-cache inference 可行的属性
- [Yao et al. — Tree of Thoughts: Deliberate Problem Solving with Large Language Models (arXiv:2305.10601)](https://arxiv.org/abs/2305.10601) — tree-of-thought reasoning strategy，Hogwild! 与它正交
- [Leviathan et al. — Fast Inference from Transformers via Speculative Decoding (arXiv:2211.17192)](https://arxiv.org/abs/2211.17192) — speculative decoding，Hogwild! 可以与之组合的 within-sequence parallelism
- [Hogwild! reference PyTorch implementation](https://github.com/eqimp/hogwild_llm) — 论文实验的 single source of truth
