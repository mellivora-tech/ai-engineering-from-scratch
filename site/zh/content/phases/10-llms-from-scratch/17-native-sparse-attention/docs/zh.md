# 原生稀疏注意力（DeepSeek NSA）

> 在 64k tokens 下，attention 会吞掉 70-80% 的解码延迟。每个 open-model 实验室都有方案想解决它。DeepSeek 的 NSA（ACL 2025 best paper）是站住脚的那个：三个并行 attention 分支：压缩后的粗粒度 tokens、选择性保留的细粒度 tokens，以及用于局部上下文的滑动窗口，再通过 learned gate 组合。它是 hardware-aligned（kernel-friendly）、natively trainable（用于 pre-training，而不是推理时硬接上去），并且在 64k 解码时比 FlashAttention 更快，同时质量匹配或超过 full attention。本课会端到端构建这三个分支，并展示为什么这种 sparsity 是端到端可微的。

**类型：** 构建
**语言：** Python（stdlib）
**先修：** Phase 7 · 12（KV cache、flash-attention）、Phase 7 · 15（attention variants）、Phase 10 · 16（differential attention）
**时间：** 约 60 分钟

## 学习目标

- 说出 NSA 的三个 attention 分支，以及每个分支捕获什么。
- 解释为什么 NSA 是“natively trainable”，而此前很多 sparse-attention 方法只能用于 inference。
- 以 compression block size 和 selection top-k 为变量，计算 NSA 相比 64k context full attention 的 attention compute savings。
- 在短合成序列上用 stdlib Python 实现三分支组合，并验证 gating weights 的行为。

## 问题

序列长度为 N 时，full attention 的时间成本是 `O(N^2)`，每层 KV cache 是 `O(N)`。在 64k tokens 下，compute 和 memory bandwidth 数字都非常灾难。NSA 论文中的理论估算显示：在 64k 下，attention 占总解码延迟的 70-80%。下游的一切：TTFT、tokens/sec、每百万 tokens 成本，都被 attention cost 主导。

Sparse attention 是显而易见的答案。以往尝试大致分成两类。固定模式 sparsity（sliding-window、strided、block-local）会丢掉信息，并在 long-range recall 任务上失败。Inference-time sparsity（KV cache pruning、H2O、StreamingLLM）被应用在一个用 dense attention 预训练的模型上，因此只能拿回一部分潜在加速，因为模型从未被要求通过 sparse pattern 路由信息。

Native Sparse Attention（Yuan et al., DeepSeek + PKU + UW, ACL 2025 best paper, arXiv:2502.11089）同时做到两件事：一个在 pre-training 中被模型学习到的 sparsity pattern，以及一个 kernel-aligned、在 inference 中真正兑现 compute savings 的算法。两年后，NSA 或它的直接后代会成为每个 frontier long-context model 的默认 attention。

## 概念

### 三个并行分支

对每个 query，NSA 会针对 KV cache 的三种不同视图运行三次 attention：

1. **压缩分支。** Tokens 被分组成大小为 `l` 的 blocks（通常 32 或 64）。每个 block 通过一个小的 learned MLP 压缩成一个 summary token。Query 对这些 compressed tokens 做 attention，得到整个序列的粗粒度视图。

2. **选择分支。** 使用 compressed branch 的 attention scores，找出当前 query 最相关的 top-k blocks。读取这些 blocks 中细粒度的原始 tokens，并让 query 对它们全部做 attention。可以把 compressed-branch attention 看成 selection 的 routing signal。

3. **滑动窗口分支。** Query 关注最近的 `W` 个 tokens（通常 512）作为 local context。这个分支捕获结构密集的短程模式（语法、局部共指），这些模式可能被另外两个分支漏掉。

三个分支的输出通过每个位置的 learned gate 组合：

```
out = g_cmp * out_cmp + g_sel * out_sel + g_win * out_win
```

`g_cmp, g_sel, g_win` 是 query 上一个小 MLP 输出的 gate weights。它们不一定要和为 1，可以独立地给各分支加权。

### 为什么这是“natively trainable”

Selection step（top-k blocks）是离散操作。离散操作会中断 gradient flow。此前的 sparse-attention 工作要么跳过 selection 的 backprop（限制训练），要么使用连续 relaxation，但 inference 时无法得到真正的 sparsity。

NSA 绕过了这个问题：compressed-branch attention 本身就是对整个序列的可微粗粒度 attention。Top-k 操作只是复用 compressed branch 中最高的 attention scores，决定要加载哪些细粒度 blocks。梯度会流过 compressed-branch scores（它们既影响 compressed output，也影响 selection logic），而 selected blocks 对最终输出的贡献同样可微。不可微的 `top_k` 操作在前向计算图上只是一个 no-op：它只控制从内存加载哪些 blocks。

这就是为什么 NSA 能端到端用于 pre-training。模型会联合学习如何通过三个分支路由信息，产生一个在 inference 时真正带来承诺中加速的 sparse pattern。

### Hardware-aligned kernel

NSA 的 kernel 是为现代 GPU memory hierarchy 设计的。Kernel 按 GQA groups 加载 queries（outer loop），为每个 group 取对应的 sparse KV blocks（inner loop），并在 SRAM 上运行 attention。因为每个 query group 看到同一组选中的 blocks（selection 是 per-query-group，而不是 per-query-head），KV loads 可以在 group 内摊销。Arithmetic intensity 保持很高。

论文报告 Triton kernels 在 64k 解码上比 FlashAttention 快 9x，并且 speedup ratio 会随序列长度增长。Forward 和 backward kernels 都有提供。

### Compute budget

令 `N` 为序列长度，`l` 为 compression block size，`k` 为 top-k selection count，`w` 为 sliding window，`b` 为 selected block size（通常等于 `l`）。

- 压缩分支：每个 query 有 `O(N/l)` 个 keys，所以总计 `O(N * N / l)`。
- 选择分支：每个 query 有 `O(k * b)` 个 keys，所以 `O(N * k * b)`。
- 滑动分支：每个 query 有 `O(w)` 个 keys，所以 `O(N * w)`。

总计：`O(N * (N/l + k*b + w))`。

当 `N = 64k, l = 64, k = 16, b = 64, w = 512`：每个 query 的成本是 `1000 + 1024 + 512 = 2536 keys`。Full attention 是 `64000 keys`。也就是 25x compute reduction。

当 `N = 128k, l = 64, k = 16, b = 64, w = 512`：每个 query 的成本是 `2000 + 1024 + 512 = 3536 keys`。Full attention 是 `128000 keys`。也就是 36x reduction。收益会随序列长度增长，这正是重点。

### 如何比较

| 方法 | 可微 | 真实 inference speedup | Long-range recall |
|--------|---------------|----------------------|-------------------|
| 仅 sliding window | 是 | 是 | 失败 |
| Strided / block-sparse | 是 | 是 | 部分 |
| KV pruning（H2O、StreamingLLM） | N/A（inference-time） | 是 | 部分 |
| MoBA（Moonshot） | 部分 | 是 | 好 |
| NSA | 是（native） | 是（64k 下 9x） | 匹配 full attention |

MoBA（Moonshot, arXiv:2502.13189）同期发表，采取了类似“三个比一个好”的路线，把 MoE principle 应用到 attention blocks。NSA 和 MoBA 是理解 2026 long-context pre-training 必须知道的两个架构。

## 构建

`code/main.py` 在短合成序列上实现三个分支，并展示：

- Compression MLP（为教学清晰使用 simple mean-pool baseline；真实 NSA 使用 learned MLP）。
- 由 compressed-branch scores 驱动的 top-k block selection。
- 对最后 `w` 个 tokens 的 sliding-window attention。
- Gated combination。
- 与 full attention 对比的 compute-count printout。

### Step 1：将 tokens 压缩成 blocks

```python
def compress(K, l):
    n = len(K)
    n_blocks = (n + l - 1) // l
    out = []
    for b in range(n_blocks):
        start, end = b * l, min((b + 1) * l, n)
        block = K[start:end]
        summary = [sum(row[d] for row in block) / len(block) for d in range(len(K[0]))]
        out.append(summary)
    return out
```

### Step 2：compressed-branch attention

让 query 对 compressed keys 运行 softmax attention。Compressed-branch scores 同时作为 top-k selection 的信号。

### Step 3：top-k block selection

选择 compressed blocks 中 score 最高的 `k` 个索引。加载这些 blocks 的原始未压缩 tokens，并对它们运行 attention。

### Step 4：sliding-window attention

取最后 `w` 个 tokens，对它们运行标准 attention。

### Step 5：gate + combine

Query 上的一个小 MLP 产生三个 gate weights。最终输出是三个分支输出的加权和。

### Step 6：compute counting

打印每个分支以及总计每个 query 关注的 keys 数量。与 `N`（full attention）比较。在一个 1024-token synthetic 中，`l = 32, k = 4, w = 128` 时，NSA 每个 query 看到 `32 + 128 + 128 = 288` 个 keys，而 full attention 是 1024，少了 3.5x。

## 使用

NSA 已经进入 DeepSeek 自己的 long-context pre-training pipeline。截至 2026 年 4 月，公共 inference stacks 的集成状态：

- **DeepSeek internal**：native，公开权重使用 NSA 或其后继 DSA（Deepseek Sparse Attention）。
- **vLLM**：面向 DeepSeek-V3.x 权重的实验性 NSA support 正在开发中。
- **SGLang**：NSA benchmarks 已发布；production path 跟随 vLLM。
- **llama.cpp / CPU**：不支持；kernel decomposition 的 overhead 对 CPU throughput 不划算。

什么时候使用 NSA：

- 目标是 64k-plus context，且有严肃 compute budget 的 pre-training 或 continued-training run。
- 对 DeepSeek 自己的 long-context checkpoints 做 inference。这些 weights 是 NSA-native。

什么时候不要使用：

- Serving 一个已经用 dense-attention pre-trained 的模型。没有 continued training，不能 retrofitting NSA。
- Context 小于 16k。三分支 overhead 会压过 savings。
- Batch-1 interactive chat。Latency-sensitive decode 会受益，但只在长上下文下明显。

## 交付

本课会产出 `outputs/skill-nsa-integrator.md`。给定一个 long-context pre-training run specification，它会生成 NSA integration plan：compression block size、top-k、sliding window、gate MLP width、kernel choice，以及能证明架构变化合理的具体 long-context evals。

## 练习

1. 在 1024-token synthetic 上运行 `code/main.py`。对三个 presets sweep `(l, k, w)` 并打印 compute counts。找出在 needle-in-haystack test 上保持 full attention 95% recall 的同时，每个 query key-count 最低的 preset。

2. 把 mean-pool compressor 替换成一个 tiny learned MLP（2-layer，hidden 32）。在一个 block signal 为平均值的 synthetic task 上训练它。衡量它在 held-out data 上相对 mean-pool baseline 的 perplexity gap。

3. 实现 gate MLP。它以 query 为输入，输出三个标量。展示 gate 的行为合理：随机 queries 上接近 uniform weighting；当 query 命中 far-back block 时，selected branch 权重明显更高。

4. 计算一个 NSA-enabled 70B model 在 128k context 下的 KV cache memory budget。KV heads 是 8，head dim 128，BF16。与 full attention 和 MLA（Phase 10 · 14 展示了 MLA 数字）对比。找出 NSA 的 fine-grained branch KV cache 等于 full attention 的序列长度。

5. 阅读 NSA 论文（arXiv:2502.11089）的 Section 4，用三句话解释为什么 compressed branch 的 attention scores 会被复用于 top-k selection，而不是计算单独的 routing score。把答案和 gradient flow 联系起来。

## 关键术语

| 术语 | 人们怎么说 | 它真正的意思 |
|------|----------------|------------------------|
| Compressed branch | “Coarse view” | 在 block-averaged keys 上做 attention，以每个 query `O(N/l)` keys 提供 global context |
| Selected branch | “Top-k blocks” | 对 compressed-branch scores 最高的 `k` 个 blocks 中的细粒度 tokens 做 attention |
| Sliding window | “Local context” | 对最后 `W` 个 tokens 做 attention，用于 short-range patterns |
| Native trainability | “Pre-train with the sparsity on” | Sparsity pattern 是在 pre-training 中学习出来的，不是在 inference 时硬接上去 |
| Compression block size l | “Group size for coarse view” | 多少 tokens 被合并成一个 summary；典型值 32-64 |
| Top-k | “Blocks to keep” | 要读取其 uncompressed tokens 的 compressed blocks 数量；典型值 16 |
| Sliding window W | “Local attention radius” | 通常 512；太短会伤害 local coherence，太长会浪费 compute |
| Branch gate | “How to mix the three” | 每个位置的 MLP 输出，用来加权三个分支的贡献 |
| Hardware alignment | “Kernel-friendly sparsity” | Sparse pattern 的选择让实际 GPU kernel 能实现理论 speedup |
| DSA | “NSA's successor” | Deepseek Sparse Attention，DeepSeek lineage 中接替 NSA 的架构 |

## 延伸阅读

- [Yuan et al. — Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention (arXiv:2502.11089, ACL 2025 Best Paper)](https://arxiv.org/abs/2502.11089) — 论文
- [DeepSeek-V3 Technical Report (arXiv:2412.19437)](https://arxiv.org/abs/2412.19437) — NSA 面向的 architecture family
- [Moonshot AI — MoBA: Mixture of Block Attention for Long-Context LLMs (arXiv:2502.13189)](https://arxiv.org/abs/2502.13189) — 同期工作，MoE-style attention over blocks
- [Beltagy et al. — Longformer: The Long-Document Transformer (arXiv:2004.05150)](https://arxiv.org/abs/2004.05150) — sliding-window 的源头
- [Xiao et al. — StreamingLLM: Efficient Streaming Language Models with Attention Sinks (arXiv:2309.17453)](https://arxiv.org/abs/2309.17453) — NSA 改进的 inference-time sparsity baseline
- [Dao et al. — FlashAttention-2 (arXiv:2307.08691)](https://arxiv.org/abs/2307.08691) — NSA kernels 在 64k 下击败的 full-attention baseline
