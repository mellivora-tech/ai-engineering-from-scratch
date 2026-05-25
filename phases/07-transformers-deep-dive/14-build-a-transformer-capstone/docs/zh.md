# 从零构建 Transformer — Capstone

> 十三节课。一个模型。没有捷径。

**类型：** 构建
**语言：** Python
**前置要求：** 阶段 7 · 01 到 13。不要跳过。
**时间：** ~120 分钟

## 问题

你已经读过每篇论文。你已经实现过 attention、multi-head splits、positional encodings、encoder 和 decoder blocks、BERT 和 GPT losses、MoE、KV cache。现在让它们在真实任务上一起工作。

Capstone：在 character-level language modeling 任务上端到端训练一个小型 decoder-only transformer。它读莎士比亚。它生成新的莎士比亚。它小到可以在笔记本上 10 分钟内训练。它正确到只要换更大数据集和更长训练，就能得到真正的 LM。

这是本课程的 “nanoGPT”。它并不原创 — Karpathy 2023 年 nanoGPT 教程是每个学生至少写一次的参考实现。我们借用形状，并围绕前面课程覆盖的内容重新加工。

## 概念

![Transformer-from-scratch block diagram](../assets/capstone.svg)

带注释的架构：

```
input tokens (B, N)
   │
   ▼
token embedding + positional embedding  ◀── Lesson 04 (RoPE option)
   │
   ▼
┌──── block × L ────────────────────┐
│  RMSNorm                          │  ◀── Lesson 05
│  MultiHeadAttention (causal)      │  ◀── Lesson 03 + 07 (causal mask)
│  residual                         │
│  RMSNorm                          │
│  SwiGLU FFN                       │  ◀── Lesson 05
│  residual                         │
└────────────────────────────────── ┘
   │
   ▼
final RMSNorm
   │
   ▼
lm_head (tied to token embedding)
   │
   ▼
logits (B, N, V)
   │
   ▼
shift-by-one cross-entropy            ◀── Lesson 07
```

### 我们交付什么

- `GPTConfig` — 所有超参数的一处配置。
- `MultiHeadAttention` — causal、batched，并带可选 Flash-style pathway（PyTorch 的 `scaled_dot_product_attention`）。
- `SwiGLUFFN` — 现代 FFN。
- `Block` — pre-norm、residual-wrapped attention + FFN。
- `GPT` — embeddings、stacked blocks、LM head、generate()。
- 训练循环，包含 AdamW、cosine LR、gradient clipping。
- Shakespeare 文本上的 char-level tokenizer。

### 我们不交付什么

- RoPE — 第 04 课已概念性实现。这里为了简单使用 learned positional embeddings。练习会要求你换成 RoPE。
- 生成时的 KV cache — 每个 generation step 都会在完整 prefix 上重新计算 attention。更慢但更简单。练习会要求你添加 KV cache。
- Flash Attention — 如果输入匹配，PyTorch 2.0+ 会自动 dispatch；我们使用 `F.scaled_dot_product_attention`。
- MoE — 每个 block 一个 FFN。你在第 11 课见过 MoE。

### 目标指标

在 Mac M2 笔记本上，一个 4-layer、4-head、d_model=128 的 GPT，在 `tinyshakespeare.txt` 上训练 2,000 步：

- Training loss 从 ~4.2（随机）收敛到 ~1.5，约 6 分钟。
- 采样输出看起来像莎士比亚：古英语味词语、换行、像 “ROMEO:” 这样的 proper names 会出现。
- Val loss（文本最后 10% held-out）紧跟 training loss；在这个规模/预算下没有 overfitting。

## 构建它

本课使用 PyTorch。安装 `torch`（CPU build 就可以）。见 `code/main.py`。脚本会处理：

- 如果缺失则下载 `tinyshakespeare.txt`（或读取本地副本）。
- Byte-level char tokenizer。
- 90/10 train/val split。
- 在支持硬件上用 bf16 autocast 的训练循环。
- 训练完成后 sampling。

### 第 1 步：data

```python
text = open("tinyshakespeare.txt").read()
chars = sorted(set(text))
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}
encode = lambda s: [stoi[c] for c in s]
decode = lambda xs: "".join(itos[x] for x in xs)
```

65 个 unique characters。Tiny vocabulary。能放进 4-byte vocab_size。没有 BPE，没有 tokenizer 麻烦。

### 第 2 步：model

见 `code/main.py`。Block 是第 05 课的教科书版本 — pre-norm、RMSNorm、SwiGLU、causal MHA。4/4/128 的参数量约 800K。

### 第 3 步：training loop

取随机 batch，每个样本是长度 256 的 token window。Forward。Shift-by-one cross-entropy。Backward。AdamW step。Log。重复。

```python
for step in range(max_steps):
    x, y = get_batch("train")
    logits = model(x)
    loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    opt.zero_grad()
```

### 第 4 步：sample

给定 prompt，反复 forward，从 top-p logits 采样，append，然后继续。500 tokens 后停止。

### 第 5 步：读输出

2,000 步之后：

```
ROMEO:
Away and mild will not thy friend, that thou shalt wit:
The chief that well shame and hath been his friends,
...
```

不是莎士比亚。但形状像莎士比亚。对于 ~800K 参数和笔记本上 6 分钟训练，这是清晰胜利。

## 使用它

这个 capstone 是一个参考架构。要把它推进到真实可用，有三个扩展：

1. **替换 tokenizer。** 使用 BPE（例如 `tiktoken.get_encoding("cl100k_base")`）。Vocab size 从 65 跳到 ~50,000。模型容量需要相应放大。
2. **在更大 corpus 上训练。** 使用 `OpenWebText` 或 `fineweb-edu`（HuggingFace）。在单张 A100 上用 10B tokens 训练 125M-param GPT 大约需要 24 小时。
3. **添加 RoPE + KV cache + Flash Attention。** 下面练习会逐步带你完成。

最终你会得到一个能生成流畅英文的 125M-parameter GPT。不是前沿模型。但同一条代码路径 — 只是更大 — 就是 Karpathy、EleutherAI 和 Allen Institute 在 2026 年训练研究 checkpoints 所用的路径。

## 交付它

见 `outputs/skill-transformer-review.md`。这个 skill 会检查一个 from-scratch transformer 实现是否正确覆盖前 13 课中的所有要点。

## 练习

1. **简单。** 运行 `code/main.py`。验证训练后最后一步 validation loss 低于 2.0。把 `max_steps` 从 2,000 改成 5,000 — val loss 还会继续改善吗？
2. **中等。** 用 RoPE 替换 learned positional embeddings。在 `MultiHeadAttention` 内对 Q 和 K 应用 rotation。训练并验证 val loss 至少不差。
3. **中等。** 在 sampling loop 中实现 KV cache。分别用和不用 cache 生成 500 tokens。笔记本上 wall-clock 应该提升 5–20×。
4. **困难。** 给模型添加第二个 head，用来预测 next-plus-one token（MTP — DeepSeek-V3 的 Multi-Token Prediction）。联合训练。它有帮助吗？
5. **困难。** 用 4-expert MoE 替换每个 block 的单个 FFN。Router + top-2 routing。在 matched active parameters 下观察 val loss 如何变化。

## 关键词

| 术语 | 大家怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| nanoGPT | “Karpathy 的教程 repo” | 最小 decoder-only transformer 训练代码，约 300 LOC；标准参考。 |
| tinyshakespeare | “标准玩具语料” | 约 1.1 MB 文本；2015 年以来每个 character-LM 教程都用它。 |
| Tied embeddings | “共享输入/输出矩阵” | LM head weight = token embedding matrix 的转置；省参数并提升质量。 |
| bf16 autocast | “训练精度技巧” | forward/back 用 bf16，optimizer state 保持 fp32；2021 年以来的标准。 |
| Gradient clipping | “阻止尖峰” | 把 global grad norm 限制在 1.0；防止训练爆炸。 |
| Cosine LR schedule | “2020+ 默认” | LR 线性升高（warmup），然后按 cosine 形状衰减到 peak 的 10%。 |
| MFU | “Model FLOP Utilization” | 已实现 FLOPs / 理论峰值；2026 年 dense 40%、MoE 30% 就很强。 |
| Val loss | “Held-out loss” | 模型从未见过的数据上的 cross-entropy；overfit detector。 |

## 延伸阅读

- [The Annotated Transformer (Harvard NLP)](https://nlp.seas.harvard.edu/annotated-transformer/) — 经典 annotated implementation。
