# KV Cache、Flash Attention 与 Inference Optimization

> 训练是并行且 FLOP-bound。推理是串行且 memory-bound。瓶颈不同，技巧也不同。

**类型：** 构建
**语言：** Python
**前置要求：** 阶段 7 · 02（Self-Attention），阶段 7 · 05（Full Transformer），阶段 7 · 07（GPT）
**时间：** ~75 分钟

## 问题

一个 naive autoregressive decoder 生成 `N` 个 tokens 要做 `O(N²)` 工作：每一步都重新在整个 prefix 上计算 attention。对 4K-token response 来说，这是 16M 次 attention operations，其中大多数是重复的。Prefix token 的每个 hidden state 一旦算出就是确定的 — 你只需要让新 token 的 query 对之前所有 cached keys 和 values 运行一次。

除此之外，attention 本身会搬运大量数据。标准 attention 会物化 N×N score matrix、N×d softmax output、N×d final output — 对 HBM 读写太多。N≥2K 时，attention 在成为 FLOP-bound 之前先成为 memory-bound。经典 attention kernels 会让现代 GPU 利用率低 4–10×。

两个优化都来自 Dao et al.，把前沿推理从“慢”推到“快”：

1. **KV cache。** 存储每个 prefix token 的 K 和 V 向量。每个新 token 的 attention 是一个 query 对 cached keys。推理从每个 generation step 的 `O(N²)` 降到 `O(N)`。
2. **Flash Attention。** Tile attention 计算，让完整 N×N 矩阵永远不会进入 HBM。所有 softmax + matmul 都在 SRAM 中发生。在 A100 上 wall-clock 加速 2–4×；在 H100 + FP8 上 5–10×。

到 2026 年两者都是通用配置。每个生产推理栈（vLLM、TensorRT-LLM、SGLang、llama.cpp）都默认假设它们。每个前沿模型都启用 Flash Attention。

## 概念

![KV cache growth and Flash Attention tiling](../assets/kv-cache-flash-attn.svg)

### KV cache 数学

每个 decoder layer、每个 token、每个 head：

```
bytes_per_token_per_layer = 2 * d_head * dtype_size
                          ^
                          K and V
```

对于一个 7B 模型，32 layers、32 heads、d_head=128、fp16：

```
per token per layer = 2 * 128 * 2 = 512 bytes
per token (32 layers) = 16 KB
per 32K context = 512 MB
```

对于 Llama 3 70B（80 layers、d_head=128、GQA with 8 KV heads）：

```
per token per layer = 2 * 8 * 128 * 2 = 4096 bytes (4 KB)
per 32K context = 10.4 GB
```

这 10 GB 解释了为什么 Llama 3 70B 在 128K context、batch size 1 下，仅 KV cache 就需要占掉一张 40 GB A100 的大部分。

**GQA 是 KV-cache 胜利点。** 如果用 64 heads 的 MHA，会是 32 GB。MLA 还能进一步压缩。

### Flash Attention — tiling 技巧

标准 attention：

```
S = Q @ K^T          (HBM read, N×N, HBM write)
P = softmax(S)       (HBM read, HBM write)
O = P @ V            (HBM read, HBM write)
```

三次 HBM 往返。在 H100 上，HBM bandwidth 是 3 TB/s；SRAM 是 30 TB/s。每次 HBM 往返相对于 on-chip 保留数据，都是 10 倍级 slowdown。

Flash Attention：

```
for each block of Q (tile size ~128 × 128):
    load Q_tile into SRAM
    for each block of K, V:
        load K_tile, V_tile into SRAM
        compute S_tile = Q_tile @ K_tile^T     (SRAM)
        running softmax aggregation             (SRAM)
        accumulate into O_tile                  (SRAM)
    write O_tile to HBM
```

每个 tile 一次 HBM 往返。总内存 footprint 从 `O(N²)` 降到 `O(N)`。Backward pass 会重新计算 forward pass 的一些值，而不是存下来 — 又省一轮内存。

**数值技巧。** Running softmax 跨 tiles 维护 `(max, sum)`，所以最终归一化是精确的。这不是近似 — Flash Attention 计算的输出和标准 attention bit-identical（除了 fp16 非结合性）。

**版本演进：**

| 版本 | 年份 | 关键变化 | 参考硬件上的加速 |
|---------|------|-----------|-------------------------------|
| Flash 1 | 2022 | Tiled SRAM kernel | A100 上 2× |
| Flash 2 | 2023 | 更好的并行性，causal-first ordering | A100 上 3× |
| Flash 3 | 2024 | Hopper asynchrony，FP8 | H100 上 1.5–2×（~740 TFLOPs FP16） |
| Flash 4 | 2026 | Blackwell 5-stage pipeline，software exp2 | Inference-first（初始仅 forward） |

Flash 4 发布时只支持 forward-pass。训练仍使用 Flash 3。Flash 4 的 GQA 和 varlen support 仍在推进中（2026 年中）。

### Speculative decoding — 另一种 latency win

便宜模型提议 N 个 tokens。大模型并行验证所有 N 个。如果验证接受 k 个 tokens，你用 1 次大模型 forward pass 换来 k 次生成。代码和 prose 上典型 k=3–5。

2026 默认：
- **EAGLE 2 / Medusa。** 共享 verifier hidden states 的 integrated draft heads。无质量损失下加速 2–3×。
- **Speculative decoding with draft model。** 消费级硬件上加速 2–4×。
- **Lookahead decoding。** Jacobi iteration；不需要 draft model。小众但免费。

### Continuous batching

经典 batched inference：等待最慢序列结束，然后开始新 batch。短回复提前结束时 GPU 会被浪费。

Continuous batching（最早在 Orca 中发布，现在在 vLLM、TensorRT-LLM、SGLang 中）：旧请求一结束，新请求立刻换入 batch。典型聊天 workload 吞吐量提升 5–10×。

### PagedAttention — 把 KV cache 当虚拟内存

vLLM 的 headline feature。KV cache 以 16-token blocks 分配；page table 把 logical positions 映射到 physical blocks。让你能在 parallel samples（beam search、parallel sampling）之间共享 KV，为 prompt caching 热交换 prefixes，并整理内存碎片。相对 naive contiguous allocation，吞吐提升 4×。

## 构建它

见 `code/main.py`。我们实现：

1. 一个 naive `O(N²)` incremental decoder。
2. 一个 `O(N)` KV-cached decoder。
3. 一个模拟 Flash Attention running-max 算法的 tiled softmax。

### 第 1 步：KV cache

```python
class KVCache:
    def __init__(self, n_layers, n_heads, d_head):
        self.K = [[[] for _ in range(n_heads)] for _ in range(n_layers)]
        self.V = [[[] for _ in range(n_heads)] for _ in range(n_layers)]

    def append(self, layer, head, k, v):
        self.K[layer][head].append(k)
        self.V[layer][head].append(v)

    def read(self, layer, head):
        return self.K[layer][head], self.V[layer][head]
```

很简单：在 per-layer、per-head lists 中持续追加 per-token K、V vectors。

### 第 2 步：tiled softmax

```python
def tiled_softmax_dot(q, K, V, tile=4):
    """Flash-attention-style softmax(qK^T)V with running max/sum."""
    m = float("-inf")
    s = 0.0
    out = [0.0] * len(V[0])
    for start in range(0, len(K), tile):
        k_block = K[start:start + tile]
        v_block = V[start:start + tile]
        scores = [sum(qi * ki for qi, ki in zip(q, k)) for k in k_block]
        new_m = max(m, *scores)
        exp_old = math.exp(m - new_m) if m != float("-inf") else 0.0
        exp_new = [math.exp(sc - new_m) for sc in scores]
        s = s * exp_old + sum(exp_new)
        for j in range(len(out)):
            out[j] = out[j] * exp_old + sum(e * v[j] for e, v in zip(exp_new, v_block))
        m = new_m
    return [o / s for o in out]
```

输出与一次性 `softmax(qK) V` bit-identical，但任意时刻 working set 都是一个 `tile × d_head` block，而不是完整 `N × d_head`。

### 第 3 步：在 100-token generation 上比较 naive vs cached decoding

统计 attention operations。Naive：`O(N²)` = 5050。Cached：`O(N)` = 100。代码会打印两者。

## 使用它

```python
# HuggingFace transformers auto-enables KV cache on decoder-only generate().
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-3B",
    attn_implementation="flash_attention_2",  # use FA3 if Hopper
    torch_dtype="bfloat16",
)
# generate() uses KV cache automatically
```

vLLM 生产：

```bash
pip install vllm
vllm serve meta-llama/Llama-3.1-70B-Instruct \
    --tensor-parallel-size 4 \
    --max-model-len 32768 \
    --enable-prefix-caching \
    --kv-cache-dtype fp8
```

跨请求 prefix caching 是 2026 年的大胜利 — 相同 system prompt、few-shot examples 或 long context document 会在不同调用之间复用 KV。对带重复 tool prompts 的 agent workloads，prefix caching 经常带来 5× 吞吐提升。

## 交付它

见 `outputs/skill-inference-optimizer.md`。这个 skill 会为新的 inference deployment 选择 attention implementation、KV cache strategy、quantization 和 speculative decoding。

## 练习

1. **简单。** 运行 `code/main.py`。确认 naive 和 cached decoders 产生相同输出；注意 op-count 差异。
2. **中等。** 实现 prefix caching：给定 prompt P 和多个 completions，对 P 运行一次 forward pass 填充 KV cache，然后按 completion 分支。和每次重新编码 P 相比测量 speedup。
3. **困难。** 实现一个 toy PagedAttention：KV cache 放在固定 16-token blocks 中，并带 free-list。当一个序列结束，把 blocks 归还池中。模拟 1,000 个不同长度的 chat completions。和 contiguous allocation 比较内存碎片。

## 关键词

| 术语 | 大家怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| KV cache | “让 decoding 变快的技巧” | 存储每个 prefix token 的 K 和 V；新 queries attend 到它们，而不是重算。 |
| HBM | “GPU 主内存” | High Bandwidth Memory；H100 上 80 GB，B200 上 192 GB。带宽约 3 TB/s。 |
| SRAM | “On-chip memory” | 每个 SM 的快速内存，H100 上每 SM 约 256 KB。带宽约 30 TB/s。 |
| Flash Attention | “Tiled attention kernel” | 不在 HBM 中物化 N×N 的 attention 计算。 |
| Continuous batching | “No-wait batching” | 不清空 batch，就把完成的序列换出、把新的换入。 |
| PagedAttention | “vLLM 的 headline” | KV cache 以固定 blocks 分配并用 page table 管理；消除碎片。 |
| Prefix caching | “复用长 prompts” | 在请求间缓存 shared prefix 的 KV；对 agents 是重大成本削减。 |
| Speculative decoding | “Draft + verify” | 便宜 draft model 提议 tokens；大模型一次验证 k 个。 |

## 延伸阅读

- [Dao et al. (2022). FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135) — Flash 1。
- [Dao (2023). FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691) — Flash 2。
- [Shah et al. (2024). FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision](https://arxiv.org/abs/2407.08608) — Flash 3。
- [FlashAttention-4 release notes (Dao-AILab, 2026)](https://github.com/Dao-AILab/flash-attention) — Blackwell 5-stage pipeline 和 software-exp2 技巧；阅读 repo README 了解本课提到的 forward-only launch caveats。
- [Kwon et al. (2023). Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180) — vLLM 论文。
- [Leviathan et al. (2023). Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) — spec decoding。
- [Li et al. (2024). EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty](https://arxiv.org/abs/2401.15077) — 本课引用的 integrated-draft 方法 EAGLE-1/2 论文。
- [Cai et al. (2024). Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads](https://arxiv.org/abs/2401.10774) — 和 EAGLE 并列提到的 Medusa 方法。
- [vLLM docs — PagedAttention](https://docs.vllm.ai/en/latest/design/kernel/paged_attention.html) — 16-token block 与 page-table 设计的标准 deep dive。
