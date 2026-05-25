# Pre-Training 一个 Mini GPT（124M 参数）

> GPT-2 Small 有 1.24 亿参数。它是 12 个 transformer layers、12 个 attention heads，以及 768 维 embeddings。你可以在单张 GPU 上从零训练它，耗时几个小时。多数人从未这样做。他们使用 pre-trained checkpoints。但如果你没有亲手训练过一个，你其实并不了解自己正在构建产品所依赖的模型内部发生了什么。

**类型：** 构建
**语言：** Python（with numpy）
**前置要求：** 阶段 10，第 01-03 课（Tokenizers，Building a Tokenizer，Data Pipelines）
**时间：** ~120 分钟

## 学习目标

- 从零实现完整 GPT-2 架构（124M 参数）：token embeddings、positional embeddings、transformer blocks 和 language model head
- 使用 next-token prediction 和 cross-entropy loss 在文本语料上训练 GPT 模型
- 实现带 temperature sampling 和 top-k/top-p filtering 的 autoregressive 文本生成
- 监控 training loss curve，并验证模型学到了连贯的语言模式

## 问题

你知道 transformer 是什么。你看过图。你能背出 “attention is all you need”，也能在白板上画出标着 “Multi-Head Attention” 的方框。

这些都不意味着你理解模型生成文本时发生了什么。

GPT-2 Small（使用 weight tying）有 124,438,272 个参数。每一个参数都是通过训练循环设定的：forward pass、compute loss、backward pass、update weights。12 个 transformer blocks。每个 block 12 个 attention heads。768 维 embedding space。50,257 token 的词表。每当模型生成一个 token，全部 1.24 亿参数都会参与一串矩阵乘法，把 token ID 序列变成下一个 token 的概率分布。

如果你从未亲手构建过它，你面对的就是黑箱。你可以使用 API。你可以 fine-tune。但当出问题时，当模型 hallucinate，当它重复自己，当它拒绝遵循指令，你没有一个关于 *为什么* 的心智模型。

本课会从零构建 GPT-2 Small。不是用 PyTorch，而是用 numpy。每个矩阵乘法都可见。每个梯度都由你的代码计算。你会看到 1.24 亿个数字如何合谋预测下一个词。

## 概念

### GPT 架构

GPT 是 autoregressive language model。“Autoregressive” 意味着它一次生成一个 token，每个 token 都 conditioned on 所有 previous tokens。架构是一堆 transformer decoder blocks。

从 token ID 到 next-token probabilities 的完整 computation graph 如下：

1. 输入 Token IDs。Shape: (batch_size, seq_len)。
2. Token embedding lookup。每个 ID 映射到 768 维向量。Shape: (batch_size, seq_len, 768)。
3. Position embedding lookup。每个位置（0、1、2、...）映射到 768 维向量。shape 相同。
4. 相加 token embeddings + position embeddings。
5. 通过 12 个 transformer blocks。
6. Final layer normalization。
7. Linear projection 到词表大小。Shape: (batch_size, seq_len, vocab_size)。
8. Softmax 得到概率。

这就是整个模型。没有卷积。没有 recurrence。只有 embeddings、attention、feedforward networks 和 layer norms，堆叠 12 次。

```mermaid
graph TD
    A["Token IDs\n(batch, seq_len)"] --> B["Token Embeddings\n(batch, seq_len, 768)"]
    A --> C["Position Embeddings\n(batch, seq_len, 768)"]
    B --> D["Add"]
    C --> D
    D --> E["Transformer Block 1"]
    E --> F["Transformer Block 2"]
    F --> G["..."]
    G --> H["Transformer Block 12"]
    H --> I["Layer Norm"]
    I --> J["Linear Head\n(768 -> 50257)"]
    J --> K["Softmax\nNext-token probabilities"]

    style A fill:#1a1a2e,stroke:#e94560,color:#fff
    style B fill:#1a1a2e,stroke:#0f3460,color:#fff
    style C fill:#1a1a2e,stroke:#0f3460,color:#fff
    style D fill:#1a1a2e,stroke:#16213e,color:#fff
    style E fill:#1a1a2e,stroke:#e94560,color:#fff
    style F fill:#1a1a2e,stroke:#e94560,color:#fff
    style H fill:#1a1a2e,stroke:#e94560,color:#fff
    style I fill:#1a1a2e,stroke:#16213e,color:#fff
    style J fill:#1a1a2e,stroke:#0f3460,color:#fff
    style K fill:#1a1a2e,stroke:#51cf66,color:#fff
```

### Transformer Block

12 个 block 中每一个都遵循同样模式。Pre-norm architecture（GPT-2 使用 pre-norm，而不是原始 transformer 那种 post-norm）：

1. LayerNorm
2. Multi-Head Self-Attention
3. Residual connection（把 input 加回来）
4. LayerNorm
5. Feed-Forward Network（MLP）
6. Residual connection（把 input 加回来）

residual connections 很关键。没有它们，backpropagation 时梯度到达 block 1 之前就会消失。有了它们，梯度可以沿着 “skip” path 从 loss 直接流到任意层。这就是你能堆叠 12、32，甚至 96 个 block 的原因（传闻 GPT-4 使用 120 个）。

### Attention：核心机制

Self-attention 让每个 token 查看之前的每个 token，并决定要关注每个 token 多少。数学如下。

对每个 token 位置，从输入计算三个向量：

- **Query (Q)**：“我在找什么？”
- **Key (K)**：“我包含什么？”
- **Value (V)**：“我携带什么信息？”

```
Q = input @ W_q    (768 -> 768)
K = input @ W_k    (768 -> 768)
V = input @ W_v    (768 -> 768)

attention_scores = Q @ K^T / sqrt(d_k)
attention_scores = mask(attention_scores)   # causal mask: -inf for future positions
attention_weights = softmax(attention_scores)
output = attention_weights @ V
```

causal mask 让 GPT 成为 autoregressive。位置 5 可以 attend 到位置 0-5，但不能 attend 到 6、7、8 等未来位置。这防止模型在训练时“作弊”偷看未来 token。

**Multi-head attention** 把 768 维空间切成 12 个 head，每个 64 维。每个 head 学习不同 attention pattern。一个 head 可能跟踪语法关系（subject-verb agreement）。另一个可能跟踪语义相似性（synonyms）。还有一个可能跟踪位置接近性（nearby words）。12 个 head 的输出拼接起来，再投影回 768 维。

```mermaid
graph LR
    subgraph MultiHead["Multi-Head Attention (12 heads)"]
        direction TB
        I["Input (768)"] --> S1["Split into 12 heads"]
        S1 --> H1["Head 1\n(64 dims)"]
        S1 --> H2["Head 2\n(64 dims)"]
        S1 --> H3["..."]
        S1 --> H12["Head 12\n(64 dims)"]
        H1 --> C["Concat (768)"]
        H2 --> C
        H3 --> C
        H12 --> C
        C --> O["Output Projection\n(768 -> 768)"]
    end

    subgraph SingleHead["Each Head Computes"]
        direction TB
        Q["Q = X @ W_q"] --> A["scores = Q @ K^T / 8"]
        K["K = X @ W_k"] --> A
        A --> M["Apply causal mask"]
        M --> SM["Softmax"]
        SM --> MUL["weights @ V"]
        V["V = X @ W_v"] --> MUL
    end

    style I fill:#1a1a2e,stroke:#e94560,color:#fff
    style O fill:#1a1a2e,stroke:#e94560,color:#fff
    style Q fill:#1a1a2e,stroke:#0f3460,color:#fff
    style K fill:#1a1a2e,stroke:#0f3460,color:#fff
    style V fill:#1a1a2e,stroke:#0f3460,color:#fff
```

除以 `sqrt(d_k)`，也就是 `sqrt(64) = 8`，是 scaling。没有它，高维向量的点积会变大，把 softmax 推到梯度几乎为零的区域。这是原始 “Attention Is All You Need” 论文中的关键洞见之一。

### KV Cache：为什么 Inference 很快

训练时，你一次处理整个序列。inference 时，你一次生成一个 token。没有优化时，生成第 N 个 token 需要重新计算前 N-1 个 token 的 attention。这是每个生成 token O(N)，整段长度 N 则 O(N^2) 的额外代价。

KV Cache 解决了这个问题。计算出每个 token 的 K 和 V 后，把它们存起来。生成第 N+1 个 token 时，你只需要为新 token 计算 Q，并查找所有 previous tokens 的缓存 K 和 V。这把 K 和 V 计算的 per-token cost 从 O(N) 降到 O(1)。attention score 计算仍然是 O(N)，因为你仍要 attend 到所有 previous positions，但你避免了对输入进行重复矩阵乘法。

对于 12 层、12 head 的 GPT-2，KV cache 每个 token 存储 2（K + V）x 12 layers x 12 heads x 64 dims = 18,432 个值。1024-token 序列在 FP32 中约 75MB。对于 128 层的 Llama 3 405B，单个序列的 KV cache 可能超过 10GB。这就是 long-context inference 受内存限制的原因。

### Prefill vs Decode：Inference 的两个阶段

当你向 LLM 发送 prompt 时，inference 分两个不同阶段发生。

**Prefill** 并行处理整个 prompt。所有 token 已知，因此模型可以同时计算所有位置的 attention。这个阶段是 compute-bound：GPU 正在以满吞吐做矩阵乘法。在 A100 上，一个 1000-token prompt 的 prefill 大约需要 20-50ms。

**Decode** 一次生成一个 token。每个新 token 依赖所有 previous tokens。这个阶段是 memory-bound：瓶颈是从 GPU memory 读取模型权重和 KV cache，而不是矩阵数学本身。GPU compute cores 大部分时间在等内存读取。对于 GPT-2，每个 decode step 的耗时几乎与 matmul 需要多少 FLOPs 无关，因为 memory bandwidth 才是约束。

这个区别对生产系统很重要。Prefill throughput 随 GPU compute 增长（更多 FLOPS = 更快 prefill）。Decode throughput 随 memory bandwidth 增长（更快 memory = 更快 decode）。这就是 NVIDIA H100 相比 A100 重点提升 memory bandwidth 的原因，它直接加速 token generation。

```mermaid
graph LR
    subgraph Prefill["Phase 1: Prefill"]
        direction TB
        P1["Full prompt\n(all tokens known)"]
        P2["Parallel computation\n(compute-bound)"]
        P3["Builds KV Cache"]
        P1 --> P2 --> P3
    end

    subgraph Decode["Phase 2: Decode"]
        direction TB
        D1["Generate token N"]
        D2["Read KV Cache\n(memory-bound)"]
        D3["Append to KV Cache"]
        D4["Generate token N+1"]
        D1 --> D2 --> D3 --> D4
        D4 -.->|repeat| D1
    end

    Prefill --> Decode

    style P1 fill:#1a1a2e,stroke:#51cf66,color:#fff
    style P2 fill:#1a1a2e,stroke:#51cf66,color:#fff
    style P3 fill:#1a1a2e,stroke:#51cf66,color:#fff
    style D1 fill:#1a1a2e,stroke:#e94560,color:#fff
    style D2 fill:#1a1a2e,stroke:#e94560,color:#fff
    style D3 fill:#1a1a2e,stroke:#e94560,color:#fff
    style D4 fill:#1a1a2e,stroke:#e94560,color:#fff
```

### 训练循环

训练 LLM 就是 next-token prediction。给定 tokens `[0, 1, 2, ..., N-1]`，预测 tokens `[1, 2, 3, ..., N]`。loss function 是模型预测概率分布和真实 next token 之间的 cross-entropy。

一个 training step：

1. **Forward pass**：让 batch 通过全部 12 个 blocks。得到每个位置的 logits（pre-softmax scores）。
2. **Compute loss**：logits 与 target tokens（input 向后平移一位）之间的 cross-entropy。
3. **Backward pass**：用 backpropagation 计算全部 124M 参数的梯度。
4. **Optimizer step**：更新权重。GPT-2 使用 Adam，带 learning rate warmup 和 cosine decay。

learning rate schedule 比你想象中更重要。GPT-2 在前 2,000 步从 0 warm up 到 peak learning rate，然后按 cosine curve 衰减。一开始使用高 learning rate 会让模型 diverge。后期保持恒定高 rate 会导致振荡。warmup-then-decay 模式被每个主要 LLM 使用。

### GPT-2 Small：数字

| Component | Shape | Parameters |
|-----------|-------|------------|
| Token embeddings | (50257, 768) | 38,597,376 |
| Position embeddings | (1024, 768) | 786,432 |
| Per-block attention (W_q, W_k, W_v, W_out) | 4 x (768, 768) | 2,359,296 |
| Per-block FFN (up + down) | (768, 3072) + (3072, 768) | 4,718,592 |
| Per-block LayerNorms (2x) | 2 x 768 x 2 | 3,072 |
| Final LayerNorm | 768 x 2 | 1,536 |
| **Total per block** | | **7,080,960** |
| **Total (12 blocks)** | | **85,054,464 + 39,383,808 = 124,438,272** |

输出 projection（logits head）与 token embedding matrix 共享权重。这叫 weight tying，它减少了 38M 参数，并且提升性能，因为它迫使模型用同一个 representation space 来表示输入和输出。

## 构建它

### 第 1 步：Embedding Layer

Token embeddings 把 50,257 个可能 token 中的每个映射成一个 768 维向量。Position embeddings 添加每个 token 在序列中所处位置的信息。二者相加。

```python
import numpy as np

class Embedding:
    def __init__(self, vocab_size, embed_dim, max_seq_len):
        self.token_embed = np.random.randn(vocab_size, embed_dim) * 0.02
        self.pos_embed = np.random.randn(max_seq_len, embed_dim) * 0.02

    def forward(self, token_ids):
        seq_len = token_ids.shape[-1]
        tok_emb = self.token_embed[token_ids]
        pos_emb = self.pos_embed[:seq_len]
        return tok_emb + pos_emb
```

初始化使用 0.02 standard deviation，来自 GPT-2 paper。太大，初始 forward pass 会产生极端值，使训练不稳定。太小，初始输出几乎对所有输入都一样，让早期梯度信号无用。

### 第 2 步：带 Causal Mask 的 Self-Attention

先实现 single-head attention。causal mask 会在 softmax 前把未来位置设为 negative infinity，确保每个位置只能 attend 到自己和更早的位置。

```python
def attention(Q, K, V, mask=None):
    d_k = Q.shape[-1]
    scores = Q @ K.transpose(0, -1, -2 if Q.ndim == 4 else 1) / np.sqrt(d_k)
    if mask is not None:
        scores = scores + mask
    weights = np.exp(scores - scores.max(axis=-1, keepdims=True))
    weights = weights / weights.sum(axis=-1, keepdims=True)
    return weights @ V
```

softmax 实现会在 exponentiating 前减去最大值。没有这一步，`exp(large_number)` 会 overflow 到 infinity。这是一个数值稳定技巧，不改变输出，因为对任意常数 c，都有 `softmax(x - c) = softmax(x)`。

### 第 3 步：Multi-Head Attention

把 768 维输入切成 12 个 64 维 head。每个 head 独立计算 attention。拼接结果，再投影回 768 维。

```python
class MultiHeadAttention:
    def __init__(self, embed_dim, num_heads):
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.W_q = np.random.randn(embed_dim, embed_dim) * 0.02
        self.W_k = np.random.randn(embed_dim, embed_dim) * 0.02
        self.W_v = np.random.randn(embed_dim, embed_dim) * 0.02
        self.W_out = np.random.randn(embed_dim, embed_dim) * 0.02

    def forward(self, x, mask=None):
        batch, seq_len, d = x.shape
        Q = (x @ self.W_q).reshape(batch, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        K = (x @ self.W_k).reshape(batch, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        V = (x @ self.W_v).reshape(batch, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        scores = Q @ K.transpose(0, 1, 3, 2) / np.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask
        weights = np.exp(scores - scores.max(axis=-1, keepdims=True))
        weights = weights / weights.sum(axis=-1, keepdims=True)
        attn_out = weights @ V

        attn_out = attn_out.transpose(0, 2, 1, 3).reshape(batch, seq_len, d)
        return attn_out @ self.W_out
```

reshape-transpose-reshape 是 multi-head attention 最容易让人困惑的部分。发生的事情是：(batch, seq_len, 768) tensor 变成 (batch, seq_len, 12, 64)，然后变成 (batch, 12, seq_len, 64)。现在 12 个 head 中每一个都有自己的 (seq_len, 64) matrix 可以运行 attention。attention 后我们反向操作：(batch, 12, seq_len, 64) 变成 (batch, seq_len, 12, 64)，再变成 (batch, seq_len, 768)。

### 第 4 步：Transformer Block

一个完整 transformer block：LayerNorm、带 residual 的 multi-head attention、LayerNorm、带 residual 的 feedforward。

```python
class LayerNorm:
    def __init__(self, dim, eps=1e-5):
        self.gamma = np.ones(dim)
        self.beta = np.zeros(dim)
        self.eps = eps

    def forward(self, x):
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        return self.gamma * (x - mean) / np.sqrt(var + self.eps) + self.beta


class FeedForward:
    def __init__(self, embed_dim, ff_dim):
        self.W1 = np.random.randn(embed_dim, ff_dim) * 0.02
        self.b1 = np.zeros(ff_dim)
        self.W2 = np.random.randn(ff_dim, embed_dim) * 0.02
        self.b2 = np.zeros(embed_dim)

    def forward(self, x):
        h = x @ self.W1 + self.b1
        h = np.maximum(0, h)  # GELU approximation: ReLU for simplicity
        return h @ self.W2 + self.b2


class TransformerBlock:
    def __init__(self, embed_dim, num_heads, ff_dim):
        self.ln1 = LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads)
        self.ln2 = LayerNorm(embed_dim)
        self.ffn = FeedForward(embed_dim, ff_dim)

    def forward(self, x, mask=None):
        x = x + self.attn.forward(self.ln1.forward(x), mask)
        x = x + self.ffn.forward(self.ln2.forward(x))
        return x
```

feedforward network 把 768 维输入扩展到 3,072 维（4x），应用非线性，再投影回 768。这种 expansion-contraction 模式让模型在每个位置上拥有一个“更宽”的内部表示。GPT-2 使用 GELU activation，但这里为了简单使用 ReLU；对于理解架构来说，差异不大。

### 第 5 步：完整 GPT Model

堆叠 12 个 transformer blocks。前面加 embedding layer，后面加 output projection。

```python
class MiniGPT:
    def __init__(self, vocab_size=50257, embed_dim=768, num_heads=12,
                 num_layers=12, max_seq_len=1024, ff_dim=3072):
        self.embedding = Embedding(vocab_size, embed_dim, max_seq_len)
        self.blocks = [
            TransformerBlock(embed_dim, num_heads, ff_dim)
            for _ in range(num_layers)
        ]
        self.ln_f = LayerNorm(embed_dim)
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

    def forward(self, token_ids):
        seq_len = token_ids.shape[-1]
        mask = np.triu(np.full((seq_len, seq_len), -1e9), k=1)

        x = self.embedding.forward(token_ids)
        for block in self.blocks:
            x = block.forward(x, mask)
        x = self.ln_f.forward(x)

        logits = x @ self.embedding.token_embed.T
        return logits

    def count_parameters(self):
        total = 0
        total += self.embedding.token_embed.size
        total += self.embedding.pos_embed.size
        for block in self.blocks:
            total += block.attn.W_q.size + block.attn.W_k.size
            total += block.attn.W_v.size + block.attn.W_out.size
            total += block.ffn.W1.size + block.ffn.b1.size
            total += block.ffn.W2.size + block.ffn.b2.size
            total += block.ln1.gamma.size + block.ln1.beta.size
            total += block.ln2.gamma.size + block.ln2.beta.size
        total += self.ln_f.gamma.size + self.ln_f.beta.size
        return total
```

注意 weight tying：`logits = x @ self.embedding.token_embed.T`。output projection 重用 token embedding matrix（转置）。这不只是节省参数的技巧。它意味着模型使用同一个向量空间来理解 token（embeddings）和预测 token（output）。

### 第 6 步：训练循环

真正训练 124M 参数需要 GPU 和 PyTorch。这个训练循环用一个能在纯 numpy 中运行的小模型演示机制。我们使用 tiny model（4 layers、4 heads、128 dims），让它可运行。

```python
def cross_entropy_loss(logits, targets):
    batch, seq_len, vocab_size = logits.shape
    logits_flat = logits.reshape(-1, vocab_size)
    targets_flat = targets.reshape(-1)

    max_logits = logits_flat.max(axis=-1, keepdims=True)
    log_softmax = logits_flat - max_logits - np.log(
        np.exp(logits_flat - max_logits).sum(axis=-1, keepdims=True)
    )

    loss = -log_softmax[np.arange(len(targets_flat)), targets_flat].mean()
    return loss


def train_mini_gpt(text, vocab_size=256, embed_dim=128, num_heads=4,
                   num_layers=4, seq_len=64, num_steps=200, lr=3e-4):
    tokens = np.array(list(text.encode("utf-8")[:2048]))
    model = MiniGPT(
        vocab_size=vocab_size, embed_dim=embed_dim, num_heads=num_heads,
        num_layers=num_layers, max_seq_len=seq_len, ff_dim=embed_dim * 4
    )

    print(f"Model parameters: {model.count_parameters():,}")
    print(f"Training tokens: {len(tokens):,}")
    print(f"Config: {num_layers} layers, {num_heads} heads, {embed_dim} dims")
    print()

    for step in range(num_steps):
        start_idx = np.random.randint(0, max(1, len(tokens) - seq_len - 1))
        batch_tokens = tokens[start_idx:start_idx + seq_len + 1]

        input_ids = batch_tokens[:-1].reshape(1, -1)
        target_ids = batch_tokens[1:].reshape(1, -1)

        logits = model.forward(input_ids)
        loss = cross_entropy_loss(logits, target_ids)

        if step % 20 == 0:
            print(f"Step {step:4d} | Loss: {loss:.4f}")

    return model
```

loss 一开始接近 `ln(vocab_size)`。对于 256-token 的 byte-level 词表，也就是 `ln(256) = 5.55`。随机模型给每个 token 分配相等概率。随着训练推进，loss 会下降，因为模型学会预测常见模式：`"t"` 后面的 `"h"`、句号后的空格，等等。

生产中你会使用 Adam optimizer，配合 gradient accumulation、learning rate warmup 和 gradient clipping。forward-pass-loss-backward-update 循环完全相同。只是 optimizer 更复杂。

### 第 7 步：文本生成

Generation 使用训练好的模型一次预测一个 token。每次预测都从输出分布中 sample（或贪心取 argmax）。

```python
def generate(model, prompt_tokens, max_new_tokens=100, temperature=0.8):
    tokens = list(prompt_tokens)
    seq_len = model.embedding.pos_embed.shape[0]

    for _ in range(max_new_tokens):
        context = np.array(tokens[-seq_len:]).reshape(1, -1)
        logits = model.forward(context)
        next_logits = logits[0, -1, :]

        next_logits = next_logits / temperature
        probs = np.exp(next_logits - next_logits.max())
        probs = probs / probs.sum()

        next_token = np.random.choice(len(probs), p=probs)
        tokens.append(next_token)

    return tokens
```

Temperature 控制随机性。Temperature 1.0 使用原始分布。Temperature 0.5 会锐化分布（更确定，模型更常选择 top choices）。Temperature 1.5 会展平分布（更随机，低概率 token 得到更大机会）。Temperature 0.0 是 greedy decoding（永远选最高概率 token）。

`tokens[-seq_len:]` 窗口是必要的，因为模型有最大 context length（GPT-2 是 1024）。超过之后，你必须丢掉最旧的 token。这就是大家说的 “context window”。

## 使用它

### 完整训练和生成 Demo

```python
corpus = """The transformer architecture has revolutionized natural language processing.
Attention mechanisms allow the model to focus on relevant parts of the input.
Self-attention computes relationships between all pairs of positions in a sequence.
Multi-head attention splits the representation into multiple subspaces.
Each attention head can learn different types of relationships.
The feedforward network provides nonlinear transformations at each position.
Residual connections enable gradient flow through deep networks.
Layer normalization stabilizes training by normalizing activations.
Position embeddings give the model information about token ordering.
The causal mask ensures autoregressive generation during training.
Pre-training on large text corpora teaches the model general language understanding.
Fine-tuning adapts the pre-trained model to specific downstream tasks."""

model = train_mini_gpt(corpus, num_steps=200)

prompt = list("The transformer".encode("utf-8"))
output_tokens = generate(model, prompt, max_new_tokens=100, temperature=0.8)
generated_text = bytes(output_tokens).decode("utf-8", errors="replace")
print(f"\nGenerated: {generated_text}")
```

在小语料和小模型上，生成文本最多也只是半连贯。它会从训练文本学到一些 byte-level pattern，但无法像 GPT-2 那样用 40GB 训练数据和完整 124M 参数架构进行泛化。重点不是输出质量。重点是你能追踪每一步：embedding lookup、attention computation、feedforward transformation、logit projection、softmax 和 sampling。每个操作都可见。

## 交付它

本课会产出 `outputs/prompt-gpt-architecture-analyzer.md`，这是一个分析任意 GPT-style model 架构选择的 prompt。给它一个 model card 或 technical report，它会拆解 parameter allocation、attention design 和 scaling decisions。

## 练习

1. 把模型改成 24 层、16 heads，而不是 12/12。数参数。把 depth 翻倍与把 width（embedding dimension）翻倍相比，参数变化如何？

2. 实现 GELU activation function（`GELU(x) = x * 0.5 * (1 + erf(x / sqrt(2)))`），并替换 feedforward network 中的 ReLU。每种 activation 都训练 500 步，比较最终 loss。

3. 给 generation function 添加 KV cache。第一次 forward pass 后，为每层存储 K 和 V tensors，并在后续 token 中重用。测量 speedup：分别用有 cache 和无 cache 生成 200 个 token，比较 wall-clock time。

4. 实现 top-k sampling（只考虑概率最高的 k 个 token）和 top-p sampling（nucleus sampling：考虑累计概率超过 p 的最小 token 集合）。在 temperature 0.8 下比较 top-k=50 与 top-p=0.95 的输出质量。

5. 构建 training loss curve plotter。训练模型 1000 步，绘制 loss vs step。识别三个阶段：快速初始下降（学习常见 bytes）、较慢的中段（学习 byte patterns）和平台期（在小语料上 overfitting）。无论你训练的是 128-dim 模型还是 GPT-4，这条曲线的形状都类似。

## 关键词

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| Autoregressive | “一次生成一个词” | 每个输出 token conditioned on 所有 previous tokens；模型预测 `P(token_n \| token_0, ..., token_{n-1})` |
| Causal mask | “不能看未来” | 一个由 negative infinity 组成的上三角矩阵，训练时阻止 attention 到未来位置 |
| Multi-head attention | “多种 attention pattern” | 把 Q、K、V 切成并行 heads（例如 GPT-2 的 12 heads，每个 64 dims），让每个 head 学习不同关系类型 |
| KV Cache | “加速用缓存” | 存储 previous tokens 已计算好的 Key 和 Value tensors，避免 autoregressive generation 中重复计算 |
| Prefill | “处理 prompt” | inference 的第一阶段，所有 prompt tokens 并行处理；受 GPU FLOPS 约束 |
| Decode | “生成 token” | inference 的第二阶段，一次生成一个 token；受 GPU bandwidth 约束 |
| Weight tying | “共享 embeddings” | 输入 token embeddings 和输出 projection head 使用同一矩阵；GPT-2 中节省 38M 参数 |
| Residual connection | “Skip connection” | 把输入直接加到 sublayer 输出上（`x + sublayer(x)`），让深层网络中的梯度可以流动 |
| Layer normalization | “规范化 activations” | 沿 feature dimension 规范化到均值 0、方差 1，并带可学习 scale 和 bias 参数 |
| Cross-entropy loss | “预测错得多严重” | `-log(分配给正确 next token 的概率)`，在所有位置上取平均；标准 LLM 训练目标 |

## 延伸阅读

- [Radford et al., 2019 -- "Language Models are Unsupervised Multitask Learners" (GPT-2)](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) -- GPT-2 论文，引入 124M 到 1.5B 参数模型家族
- [Vaswani et al., 2017 -- "Attention Is All You Need"](https://arxiv.org/abs/1706.03762) -- 原始 transformer 论文，包含 scaled dot-product attention 和 multi-head attention
- [Llama 3 Technical Report](https://arxiv.org/abs/2407.21783) -- Meta 如何用 16K GPUs 把 GPT 架构扩展到 405B 参数
- [Pope et al., 2022 -- "Efficiently Scaling Transformer Inference"](https://arxiv.org/abs/2211.05102) -- 形式化 prefill vs decode 和 KV cache 分析的论文
