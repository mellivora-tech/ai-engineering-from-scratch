# Multi-Head Attention

> 一个 attention head 一次学一种关系。八个 heads 学八种。Heads 很便宜，多用一些。

**类型：** 构建
**语言：** Python
**前置要求：** 阶段 7 · 02（从零实现 Self-Attention）
**时间：** ~75 分钟

## 问题

单个 self-attention head 会计算一个 attention 矩阵。这个矩阵捕捉一种关系 — 通常是能在当前训练信号上最小化 loss 的那一种。如果你的数据里主谓一致、共指、长距离 discourse 和句法 chunking 全都纠缠在一起，单个 head 会把它们涂抹进一个 softmax 分布里，并丢掉一半信号。

2017 年 Vaswani 论文给出的修复方式是：并行运行多个 attention 函数，每个函数有自己的 Q、K、V 投影，然后拼接输出。每个 head 在维度为 `d_model / n_heads` 的较小子空间里工作。总参数量保持不变。表达能力上升。

Multi-head attention 是 2026 年每个 transformer 都默认携带的组件。唯一争论是 *多少个* heads，以及 key 和 value 是否共享投影（Grouped-Query Attention、Multi-Query Attention、Multi-head Latent Attention）。

## 概念

![Multi-head attention splits, attends, concatenates](../assets/multi-head-attention.svg)

**Split。** 取形状为 `(N, d_model)` 的 `X`。投影成 Q、K、V，每个形状都是 `(N, d_model)`。reshape 成 `(N, n_heads, d_head)`，其中 `d_head = d_model / n_heads`。再转置为 `(n_heads, N, d_head)`。

**并行 Attend。** 在每个 head 内运行 scaled dot-product attention。每个 head 产生 `(N, d_head)`。这些 heads 在 embedding 的不同子空间上工作，在 attention 计算本身期间彼此不会交流。

**拼接并投影。** 把 heads 叠回 `(N, d_model)`，再乘以学习得到的输出矩阵 `W_o`，形状为 `(d_model, d_model)`。`W_o` 是 heads 得以混合的地方。

**为什么有效。** 每个 head 都可以专门化，而不必和其他 head 争抢表示预算。2019–2024 年的 probing studies 显示了不同的 head 角色：positional heads、关注前一个 token 的 head、copy heads、named-entity heads、induction heads（in-context learning 的基础）。

**2026 年的变体谱系：**

| 变体 | Q heads | K/V heads | 使用者 |
|---------|---------|-----------|---------|
| Multi-head (MHA) | N | N | GPT-2, BERT, T5 |
| Multi-query (MQA) | N | 1 | PaLM, Falcon |
| Grouped-query (GQA) | N | G（例如 N/8） | Llama 2 70B, Llama 3+, Qwen 2+, Mistral |
| Multi-head latent (MLA) | N | compressed to low-rank | DeepSeek-V2, V3 |

GQA 是现代默认选项，因为它把 KV-cache 内存减少 `N/G` 倍，同时几乎保留完整质量。MLA 更进一步，把 K/V 压缩进 latent space，然后在计算时投影回来 — 多花 FLOPs，但省下更多内存。

## 构建它

### 第 1 步：从已有 single-head attention 拆分 heads

取第 02 课的 `SelfAttention`，用一对 split/concat 包起来。`code/main.py` 里有 numpy 实现；逻辑如下：

```python
def split_heads(X, n_heads):
    n, d = X.shape
    d_head = d // n_heads
    return X.reshape(n, n_heads, d_head).transpose(1, 0, 2)  # (heads, n, d_head)

def combine_heads(H):
    h, n, d_head = H.shape
    return H.transpose(1, 0, 2).reshape(n, h * d_head)
```

一次 reshape 和一次 transpose。没有循环。这正是 PyTorch 在 `nn.MultiheadAttention` 下做的事情。

### 第 2 步：每个 head 运行 scaled-dot-product attention

每个 head 得到自己的 Q、K、V 切片。Attention 变成 batched matmul：

```python
def mha_forward(X, W_q, W_k, W_v, W_o, n_heads):
    Q = X @ W_q
    K = X @ W_k
    V = X @ W_v
    Qh = split_heads(Q, n_heads)         # (heads, n, d_head)
    Kh = split_heads(K, n_heads)
    Vh = split_heads(V, n_heads)
    scores = Qh @ Kh.transpose(0, 2, 1) / np.sqrt(Qh.shape[-1])
    weights = softmax(scores, axis=-1)
    out = weights @ Vh                    # (heads, n, d_head)
    concat = combine_heads(out)
    return concat @ W_o, weights
```

在真实硬件上，`Qh @ Kh.transpose(...)` 是一次 `bmm`。GPU 看到的是一个形状为 `(heads, N, d_head) × (heads, d_head, N) -> (heads, N, N)` 的单个 batched matmul。增加 heads 几乎是免费的。

### 第 3 步：Grouped-Query Attention 变体

只有 key 和 value 投影会改变。Q 有 `n_heads` 组；K 和 V 有 `n_kv_heads < n_heads` 组，然后重复以匹配：

```python
def gqa_project(X, W, n_kv_heads, n_heads):
    kv = split_heads(X @ W, n_kv_heads)       # (kv_heads, n, d_head)
    repeat = n_heads // n_kv_heads
    return np.repeat(kv, repeat, axis=0)      # (n_heads, n, d_head)
```

推理时这会节省内存，因为 KV cache 里只保存 `n_kv_heads` 份，而不是 `n_heads` 份。Llama 3 70B 使用 64 个 query heads 和 8 个 KV heads — cache 缩小 8×。

### 第 4 步：探测每个 head 学到了什么

在一个短句上用 4 个 heads 运行 MHA。对每个 head，打印 `(N, N)` attention 矩阵。即使是随机初始化，你也会看到不同 heads 选出不同结构 — 一部分是信号，一部分是子空间中的旋转对称性。

## 使用它

在 PyTorch 中，一行版本是：

```python
import torch.nn as nn

mha = nn.MultiheadAttention(embed_dim=512, num_heads=8, batch_first=True)
```

PyTorch 2.5+ 中的 GQA：

```python
from torch.nn.functional import scaled_dot_product_attention

# scaled_dot_product_attention auto-dispatches Flash Attention on CUDA.
# For GQA, pass Q of shape (B, n_heads, N, d_head) and K,V of shape
# (B, n_kv_heads, N, d_head). PyTorch handles the repeat.
out = scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
```

**应该用多少 heads？** 2026 年生产模型的经验法则：

| 模型大小 | d_model | n_heads | d_head |
|------------|---------|---------|--------|
| Small (~125M) | 768 | 12 | 64 |
| Base (~350M) | 1024 | 16 | 64 |
| Large (~1B) | 2048 | 16 | 128 |
| Frontier (~70B) | 8192 | 64 | 128 |

`d_head` 几乎总是落在 64 或 128。它代表一个 head 能“看见”多少信息。低于 32 时，heads 开始和缩放因子 `sqrt(d_head)` 打架；高于 256 时，你会失去“许多小专家”的优势。

## 交付它

见 `outputs/skill-mha-configurator.md`。这个 skill 会根据参数预算、序列长度和部署目标，为新的 transformer 推荐 head 数、kv-head 数和投影策略。

## 练习

1. **简单。** 取 `code/main.py` 中的 MHA，在固定 `d_model=64` 的情况下把 `n_heads` 从 1 改到 16。在一个合成 copy task 上画出 tiny one-layer model 的 loss。更多 heads 是有帮助、平台化，还是伤害？
2. **中等。** 实现 MQA（一个 KV head 被所有 query heads 共享）。测量相比完整 MHA 参数量下降多少。计算 N=2048 推理时 KV-cache 大小缩小多少。
3. **困难。** 实现一个 tiny 版本的 Multi-head Latent Attention：把 K,V 压缩到 rank-`r` latent，把 latent 存入 KV cache，在 attention 时解压。`r` 到多少时 cache 内存低于完整 MHA 的 1/8，同时质量仍在验证 ppl 的 1 bit 以内？

## 关键词

| 术语 | 大家怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| Head | “一个 attention circuit” | 一个维度为 `d_head = d_model / n_heads` 的 Q/K/V 投影，有自己的 attention 矩阵。 |
| d_head | “Head dimension” | 每个 head 的 hidden width；生产中几乎总是 64 或 128。 |
| Split / combine | “Reshape tricks” | attention 前后的 `(N, d_model) ↔ (n_heads, N, d_head)` reshape+transpose。 |
| W_o | “Output projection” | 拼接 heads 后应用的 `(d_model, d_model)` 矩阵；heads 在这里混合。 |
| MQA | “一个 KV head” | Multi-Query Attention：单个共享 K/V 投影。KV cache 最小，但有一些质量损失。 |
| GQA | “Llama 2 以来的默认选项” | `n_kv_heads < n_heads` 的 Grouped-Query Attention；重复 K/V 来匹配 Q。 |
| MLA | “DeepSeek 的技巧” | Multi-head Latent Attention：K,V 被压缩为低秩 latent，在 attend 时解压。 |
| Induction head | “in-context learning 背后的 circuit” | 一对 heads，用来检测之前出现过的模式并复制其后继 token。 |

## 延伸阅读

- [Vaswani et al. (2017). Attention Is All You Need §3.2.2](https://arxiv.org/abs/1706.03762) — 原始 multi-head 规格。
- [Shazeer (2019). Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/abs/1911.02150) — MQA 论文。
- [Ainslie et al. (2023). GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245) — 如何在训练后把 MHA 转成 GQA。
- [DeepSeek-AI (2024). DeepSeek-V2 Technical Report](https://arxiv.org/abs/2405.04434) — MLA，以及为什么它在 cache 内存上胜过 MHA/GQA。
- [Olsson et al. (2022). In-context Learning and Induction Heads](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html) — 从 mechanistic 角度看 heads 实际在做什么。
