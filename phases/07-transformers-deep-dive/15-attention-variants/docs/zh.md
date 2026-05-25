# Attention Variants — Sliding Window、Sparse、Differential

> Full attention 是一个圆。每个 token 都看每个 token，而内存付出代价。四种变体弯曲了这个圆的形状，并收回一半成本。

**类型：** 构建
**语言：** Python
**前置要求：** 阶段 7 · 02（Self-Attention），阶段 7 · 03（Multi-Head），阶段 7 · 12（KV Cache / Flash Attention）
**时间：** ~60 分钟

## 问题

Full attention 在序列长度上需要 `O(N²)` 内存和 `O(N²)` 计算。对于 128K-context Llama 3 70B，这意味着每层 160 亿个 attention entries，再乘以 80 层。Flash Attention（第 12 课）隐藏了 `O(N²)` activation memory，但不改变算术成本 — 每个 token 仍然 attend 到每个其他 token。

三类变体会改变 attention 矩阵本身的拓扑：

1. **Sliding window attention（SWA）。** 每个 token 只 attend 到固定窗口内的邻居，而不是完整 prefix。内存和计算降到 `O(N · W)`，其中 `W` 是 window。Gemma 2/3、Mistral 7B 的前几层、Phi-3-Long。
2. **Sparse / block attention。** 只有选中的 `(i, j)` 对会被打分；其余强制为零权重。Longformer、BigBird、OpenAI sparse transformer。
3. **Differential attention。** 用独立 Q/K 投影计算两张 attention maps，并相减。清除把权重泄漏到前几个 tokens 的 “attention sink”。Microsoft 的 DIFF Transformer（2024）。

这些会共存。一个 2026 年前沿模型经常混合它们：大多数层是 SWA-1024，每五层一层 global full attention，还有少量 differential heads 清理 retrieval。Gemma 3 的 5:1 SWA-to-global 比例是当前教科书默认值。

## 概念

### Sliding Window Attention (SWA)

位置 `i` 的每个 query 只 attend 到 `[i - W, i]`（causal SWA）或 `[i - W/2, i + W/2]`（bidirectional）中的位置。窗口外 tokens 在 score matrix 中得到 `-inf`。

```
full causal:           sliding window (W=4):
positions 0-7          positions 0-7, W=4
    0 1 2 3 4 5 6 7        0 1 2 3 4 5 6 7
0 | x                0 |  x
1 | x x              1 |  x x
2 | x x x            2 |  x x x
3 | x x x x          3 |  x x x x
4 | x x x x x        4 |    x x x x
5 | x x x x x x      5 |      x x x x
6 | x x x x x x x    6 |        x x x x
7 | x x x x x x x x  7 |          x x x x
```

对于 `N = 8192` 和 `W = 1024`，score matrix 期望中只有 1024 × 8192 个非零行 — 减少 8×。

**KV cache 会随 SWA 缩小。** 每层只需要保留最近 `W` 个 tokens 的 K 和 V。对于 Gemma-3-ish 配置（1024 window、128K context），KV cache 降低 128×。

**质量成本。** 纯 SWA transformers 在长距离 retrieval 上会吃力。修复方式：把 SWA layers 与 full-attention layers 交错。Gemma 3 使用 5:1 SWA:global。Mistral 7B 使用 causal-SWA stack，信息通过重叠窗口“向前流动” — 每层把 effective receptive field 扩展 `W`，经过 `L` 层后模型可以 attend 到 `L × W` tokens 之前。

### Sparse / Block Attention

提前选择一个 `N × N` sparsity pattern。三个标准形状：

- **Local + strided（OpenAI sparse transformer）。** Attend 到最近 `W` 个 tokens，加上在那之前每隔 `stride` 个 token 的位置。以 `O(N · sqrt(N))` compute 同时捕捉局部和长距离。
- **Longformer / BigBird。** Local window + 少量 global tokens（例如 `[CLS]`），这些 tokens attend 到所有人，也被所有人 attend + random-sparse links。同等质量下经验上能把 context 扩大 2×。
- **Native Sparse Attention（DeepSeek, 2025）。** 学习哪些 `(Q, K)` blocks 重要；在 kernel 层跳过零 blocks。兼容 FlashAttention。

Sparse attention 是 kernel-engineering 故事。数学很简单（mask score matrix）；收益来自永远不把零 entries 加载进 SRAM。FlashAttention-3 和 2026 年 FlexAttention API 让自定义 sparse patterns 成为 PyTorch 的一等公民。

### Differential Attention（DIFF Transformer, 2024）

普通 attention 有 “attention sink” 问题：softmax 强制每一行和为 1，所以那些并不特别想 attend 到任何内容的 tokens 会把权重倒到第一个 token（或前几个）上。这偷走了本应给真实内容的容量。

Differential attention 通过计算 **两张** attention maps 并相减来修复：

```
A1 = softmax(Q1 K1^T / √d)
A2 = softmax(Q2 K2^T / √d)
DiffAttn = (A1 - λ · A2) V
```

其中 `λ` 是学习得到的标量（通常 0.5–0.8）。A1 捕捉真实内容权重；A2 捕捉 sink。相减会抵消 sink，把权重重新分配给相关 tokens。

报告结果（Microsoft 2024）：perplexity 低 5–10%，同训练长度下 effective context 长 1.5–2×，needle-in-haystack retrieval 更尖锐。

### Variant Comparison

| 变体 | Compute | KV cache | 相对 full 的质量 | 生产使用 |
|---------|---------|----------|-----------------|----------------|
| Full attention | O(N²) | O(N) per layer | baseline | 每个模型的默认层 |
| SWA (window 1024) | O(N·W) | O(W) per layer | -0.1 ppl，配合 global layers 很好 | Gemma 2/3, Phi-3-Long |
| Local + strided sparse | O(N·√N) | mixed | 类似 SWA | OpenAI sparse transformer, Longformer |
| BigBird (local + global + random) | O(N) approx | mixed | 2× context 下匹配 full | 早期 long-context BERT |
| Native Sparse (DeepSeek-V3.2) | O(N · active fraction) | O(N) | within 0.05 ppl | DeepSeek-V3.2, 2025 |
| Differential | O(2·N²) | O(2N) | -5 到 -10% ppl | DIFF Transformer, early 2026 models |

## 构建它

见 `code/main.py`。我们实现一个 causal mask comparator，在 toy sequence 上并排展示 full、SWA、local+strided 和 differential attention。

### 第 1 步：full causal mask（baseline）

```python
def causal_mask(n):
    return [[0.0 if j <= i else float("-inf") for j in range(n)] for i in range(n)]
```

来自第 07 课的 baseline。下三角；对角线上方权重为零。

### 第 2 步：sliding window causal mask

```python
def swa_mask(n, window):
    M = [[float("-inf")] * n for _ in range(n)]
    for i in range(n):
        lo = max(0, i - window + 1)
        for j in range(lo, i + 1):
            M[i][j] = 0.0
    return M
```

一个参数 — `window`。当 `window >= n` 时，恢复 full causal attention。当 `window = 1` 时，每个 token 只 attend 到自己。

### 第 3 步：local + strided sparse mask

```python
def strided_mask(n, window, stride):
    M = [[float("-inf")] * n for _ in range(n)]
    for i in range(n):
        lo = max(0, i - window + 1)
        for j in range(lo, i + 1):
            M[i][j] = 0.0
        for j in range(0, i + 1, stride):
            M[i][j] = 0.0
    return M
```

Dense local window 加上从当前位置回到序列开头每隔 `stride` 个 token。Receptive field 会随额外层以 log steps 增长。

### 第 4 步：differential attention

```python
def diff_attention(Q1, K1, Q2, K2, V, lam):
    A1 = softmax_causal(Q1 @ K1.T / sqrt_d)
    A2 = softmax_causal(Q2 @ K2.T / sqrt_d)
    return (A1 - lam * A2) @ V
```

两次 attention passes，用学习到的 mixing coefficient 相减。在代码中，我们比较 single vs differential 的 attention-sink heatmap，并观察 sink 崩塌。

### 第 5 步：KV cache sizes

在 `N = 131072` 下打印每个变体每层 cache size。SWA 和 sparse 变体下降 10–100×。Differential 翻倍。要有意识地支付内存账单。

## 使用它

2026 年生产模式：

```python
from transformers import AutoModelForCausalLM
# Gemma 3 mixes SWA (window=1024) and global layers at 5:1.
model = AutoModelForCausalLM.from_pretrained("google/gemma-3-27b-it")
# print(model.config.sliding_window, model.config.layer_types)
```

PyTorch 2.5+ 中的 FlexAttention 接受 mask function：

```python
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

def swa_pattern(b, h, q_idx, kv_idx):
    return (q_idx - kv_idx < 1024) & (q_idx >= kv_idx)

mask = create_block_mask(swa_pattern, B=batch, H=heads, Q_LEN=n, KV_LEN=n)
out = flex_attention(q, k, v, block_mask=mask)
```

这会编译成自定义 Triton kernel。对常见 patterns，速度在 FlashAttention-3 的 10% 以内，并且 mask function 是 Python callable。

**什么时候选哪种：**

- **Pure full attention** — context 到 ~16K 且每层都用，或 retrieval quality 最重要时。
- **SWA + global mix** — 长 context（>32K），训练和推理 memory-bound。32K 以上的 2026 默认。
- **Sparse block attention** — 自定义 kernel、自定义 pattern。保留给 specialized workloads（retrieval、audio）。
- **Differential attention** — 任何 attention-sink contamination 伤害任务的 workload（long-context RAG、needle-in-haystack）。

## 交付它

见 `outputs/skill-attention-variant-picker.md`。这个 skill 会根据 target context length、retrieval demands 和 training/inference compute profile，为新模型选择 attention topology。

## 练习

1. **简单。** 运行 `code/main.py`。验证 `window=4` 的 SWA 会把每行最近 4 个 tokens 外的所有内容置零。验证 `window=n` bit-identically 复现 full causal attention。
2. **中等。** 在第 07 课 capstone 上实现 `window=1024` 的 causal SWA。在 tinyshakespeare 上训练 1,000 步。相对 full attention，val loss 回退多少？peak memory 下降多少？
3. **困难。** 在 capstone 模型中实现 Gemma-3-style 5:1 layer mix（5 SWA，1 global）。在 matched parameters 下，对比 pure-SWA 和 pure-global baselines 的 loss、memory 和 generation quality。
4. **困难。** 实现每 head 一个学习 `λ` 的 differential attention。在合成 retrieval task（one needle、2,000 distractors）上训练。以 matched parameters 测量相对 single-attention baseline 的 retrieval accuracy。

## 关键词

| 术语 | 大家怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| Sliding window attention (SWA) | “Local attention” | 每个 query attend 到最近 `W` 个 tokens；KV cache 缩到 `O(W)`。 |
| Effective receptive field | “模型能往回看多远” | 在 window `W` 的 `L` 层 SWA stack 中，最多 `L × W` tokens。 |
| Longformer / BigBird | “Local + global + random” | Sparse patterns，带少量 always-attending global tokens；早期长上下文方法。 |
| Native Sparse Attention | “DeepSeek 的 kernel 技巧” | 学习 block-level sparsity；在 kernel 层跳过零 blocks，同时保持质量。 |
| Differential attention | “两张图，一张相减” | DIFF Transformer：从第一张 attention map 中减去学习到的 `λ` 倍第二张 map，以抵消 attention sinks。 |
| Attention sink | “权重流向 token 0” | Softmax normalization 强制每行和为 1；无信息 query 会把权重倒到位置 0。 |
| FlexAttention | “Mask-as-Python” | PyTorch 2.5+ API，可把任意 mask functions 编译成 FlashAttention 形状的 kernels。 |
| Layer type mix | “5:1 SWA-to-global” | 在 stack 中交错 sparse 和 full attention layers，以较低内存保持质量。 |

## 延伸阅读

- [Beltagy, Peters, Cohan (2020). Longformer: The Long-Document Transformer](https://arxiv.org/abs/2004.05150) — 标准 sliding-window + global-token 论文。
- [Zaheer et al. (2020). Big Bird: Transformers for Longer Sequences](https://arxiv.org/abs/2007.14062) — local + global + random。
- [Child et al. (2019). Generating Long Sequences with Sparse Transformers](https://arxiv.org/abs/1904.10509) — OpenAI 的 local+strided pattern。
- [Gemma Team (2024). Gemma 2: Improving Open Language Models at a Practical Size](https://arxiv.org/abs/2408.00118) — 1:1 SWA:global mix。
- [Gemma Team (2025). Gemma 3 technical report](https://arxiv.org/abs/2503.19786) — window=1024 的 5:1 mix，现在的教科书默认。
- [Ye et al. (2024). Differential Transformer](https://arxiv.org/abs/2410.05258) — DIFF Transformer 论文。
- [Yuan et al. (2025). Native Sparse Attention](https://arxiv.org/abs/2502.11089) — DeepSeek-V3.2 的 learned-sparsity attention。
- [PyTorch — FlexAttention blog and docs](https://pytorch.org/blog/flexattention/) — Use It 中 mask-as-callable pattern 的 API reference。
