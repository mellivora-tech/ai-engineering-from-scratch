# 完整 Transformer — Encoder + Decoder

> Attention 是主角。其他所有东西 — residuals、normalization、feed-forward、cross-attention — 都是让你能把它堆深的脚手架。

**类型：** 构建
**语言：** Python
**前置要求：** 阶段 7 · 02（Self-Attention），阶段 7 · 03（Multi-Head Attention），阶段 7 · 04（Positional Encoding）
**时间：** ~75 分钟

## 问题

单个 attention 层是 feature extractor，不是模型。每层一次 matmul 不足以承载语言所需的容量。你需要深度 — 而没有正确的管线，深度会崩。

2017 年 Vaswani 论文把六个设计决策打包在一起，把一个 attention 层变成了可堆叠的 block。从那以后每个 transformer — encoder-only（BERT）、decoder-only（GPT）、encoder-decoder（T5）— 都继承同一个骨架。到 2026 年，block 已经被细化（RMSNorm、SwiGLU、pre-norm、RoPE），但骨架完全相同。

本课讲的就是这个骨架。后面的课程会专门化它 — 第 06 课讲 encoder，第 07 课讲 decoder，第 08 课讲 encoder-decoder。

## 概念

![Encoder and decoder block internals, wired](../assets/full-transformer.svg)

### 六个部件

1. **Embedding + 位置信号。** Tokens → vectors。位置通过 RoPE（现代）或 sinusoidal（经典）注入。
2. **Self-attention。** 每个位置 attend 到每个其他位置。在 decoder 中会 mask。
3. **Feed-forward network (FFN)。** 按位置独立的两层 MLP：`W_2 · activation(W_1 · x)`。默认扩展比例 4×。
4. **Residual connection。** `x + sublayer(x)`。没有它，梯度在约 6 层之后就会消失。
5. **Layer normalization。** `LayerNorm` 或 `RMSNorm`（现代）。稳定 residual stream。
6. **Cross-attention（只在 decoder 中）。** Queries 来自 decoder，keys 和 values 来自 encoder output。

### Encoder block（BERT、T5 encoder 使用）

```
x → LN → MHA(self) → + → LN → FFN → + → out
                     ^              ^
                     |              |
                     └── residual ──┘
```

Encoder 是双向的。没有 masking。所有位置都能看到所有位置。

### Decoder block（GPT、T5 decoder 使用）

```
x → LN → MHA(masked self) → + → LN → MHA(cross to encoder) → + → LN → FFN → + → out
```

Decoder 每个 block 有三个 sublayers。中间那个 — cross-attention — 是信息从 encoder 流入 decoder 的唯一地方。在纯 decoder-only 架构（GPT）中，cross-attention 被省略，只保留 masked self-attention + FFN。

### Pre-norm vs post-norm

原论文：`x + sublayer(LN(x))` vs `LN(x + sublayer(x))`。Post-norm 在 2019 年左右失宠 — 如果没有谨慎 warmup，很难训得很深。Pre-norm（在 sublayer *之前* 做 `LN`）是 2026 年默认选择：Llama、Qwen、GPT-3+、Mistral 都用它。

### 2026 年的现代化 block

Vaswani 2017 使用 LayerNorm + ReLU。现代 stack 替换了两者。生产 block 实际看起来是这样：

| 组件 | 2017 | 2026 |
|-----------|------|------|
| Normalization | LayerNorm | RMSNorm |
| FFN activation | ReLU | SwiGLU |
| FFN expansion | 4× | 2.6×（SwiGLU 用三个矩阵，总参数量匹配） |
| Position | Sinusoidal absolute | RoPE |
| Attention | Full MHA | GQA（或 MLA） |
| Bias terms | Yes | No |

RMSNorm 去掉了 LayerNorm 的 mean-centering（少一次减法），节省计算，并且经验上至少同样稳定。SwiGLU（`Swish(W1 x) ⊙ W3 x`）在 Llama、PaLM、Qwen 论文中都持续优于 ReLU/GELU FFN，LM ppl 大约好 0.5 点。

### 参数量

对于一个 `d_model = d`、FFN 扩展率为 `r` 的 block：

- MHA：`4 · d²`（Q、K、V、O 投影）
- FFN（SwiGLU）：`3 · d · (r · d)` ≈ `3rd²`
- Norms：可忽略

当 `d = 4096, r = 2.6, layers = 32`（大致相当于 Llama 3 8B）时，总量是：`32 · (4·4096² + 3·2.6·4096²) ≈ 32 · (16 + 32) M = ~1.5B parameters per layer × 32 ≈ 7B`（再加 embeddings 和 head）。这与公开参数量相符。

## 构建它

### 第 1 步：building blocks

使用第 03 课的小型 `Matrix` 类（为了独立性，本文件里复制了一份）：

- `layer_norm(x, eps=1e-5)` — 减去均值，除以标准差。
- `rms_norm(x, eps=1e-6)` — 除以 RMS。不减均值。
- `gelu(x)` 和 `silu(x) * W3 x`（SwiGLU）。
- `ffn_swiglu(x, W1, W2, W3)`。
- `encoder_block(x, params)` 和 `decoder_block(x, enc_out, params)`。

完整接线见 `code/main.py`。

### 第 2 步：接一个 2 层 encoder 和 2 层 decoder

把它们堆起来。把 encoder output 传入每个 decoder cross-attention。在 output projection 前加一个最终 LN。

```python
def encode(tokens, params):
    x = embed(tokens, params.emb) + sinusoidal(len(tokens), params.d)
    for block in params.encoder_blocks:
        x = encoder_block(x, block)
    return x

def decode(target_tokens, encoder_out, params):
    x = embed(target_tokens, params.emb) + sinusoidal(len(target_tokens), params.d)
    for block in params.decoder_blocks:
        x = decoder_block(x, encoder_out, block)
    return x
```

### 第 3 步：在玩具示例上运行 forward

把 6-token source 和 5-token target 送进去。验证输出形状是 `(5, vocab)`。不训练 — 本课关注架构，不关注 loss。

### 第 4 步：换成 RMSNorm + SwiGLU

用 RMSNorm 和 SwiGLU 替换 LayerNorm 和 ReLU-FFN。确认形状仍然匹配。这就是 2026 年现代化版本，只需要一次函数替换。

## 使用它

PyTorch/TF 参考实现是：`nn.TransformerEncoderLayer`、`nn.TransformerDecoderLayer`。但大多数 2026 生产代码都会自己写 block，因为：

- Flash Attention 是在 attention 内部调用的，不通过 `nn.MultiheadAttention`。
- GQA / MLA 不在 stdlib reference 中。
- RoPE、RMSNorm、SwiGLU 不是 PyTorch 默认值。

HF `transformers` 有很干净的参考 block，值得阅读：`modeling_llama.py` 是 2026 年 decoder-only block 的标准参考。大约 500 行，值得完整走读一次。

**Encoder vs decoder vs encoder-decoder — 什么时候选：**

| 需求 | 选择 | 示例 |
|------|------|---------|
| 分类、embeddings、文本 QA | Encoder-only | BERT, DeBERTa, ModernBERT |
| 文本生成、聊天、代码、推理 | Decoder-only | GPT, Llama, Claude, Qwen |
| 结构化输入 → 结构化输出（翻译、摘要） | Encoder-decoder | T5, BART, Whisper |

Decoder-only 赢下语言任务，是因为它扩展最干净，同时能处理理解和生成。Encoder-decoder 在输入有清晰“source sequence”身份时仍然最好（翻译、语音识别、结构化任务）。

## 交付它

见 `outputs/skill-transformer-block-reviewer.md`。这个 skill 会用 2026 年默认实践检查新的 transformer block 实现，并标出缺失部分（pre-norm、RoPE、RMSNorm、GQA、FFN expansion ratio）。

## 练习

1. **简单。** 在 `d_model=512, n_heads=8, ffn_expansion=4, swiglu=True` 下统计你的 encoder_block 参数量。通过实现 block 并使用 `sum(p.numel() for p in block.parameters())` 验证。
2. **中等。** 从 post-norm 切换到 pre-norm。初始化二者，并在随机输入上测量堆叠 12 层后的 activation norm。Post-norm 的 activations 应该爆炸；pre-norm 应该保持有界。
3. **困难。** 在玩具 copy task（复制反转后的 `x`）上实现 4 层 encoder-decoder。训练 100 步。报告 loss。换成 RMSNorm + SwiGLU + RoPE — loss 会下降吗？

## 关键词

| 术语 | 大家怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| Block | “一层 transformer” | norm + attention + norm + FFN 的 stack，并包在 residual connections 中。 |
| Residual | “Skip connection” | `x + f(x)` 输出；让梯度能流过深层 stack。 |
| Pre-norm | “先 normalize，不是后 normalize” | 现代形式：`x + sublayer(LN(x))`。无需 warmup 花活也能训得更深。 |
| RMSNorm | “没有 mean 的 LayerNorm” | 除以 RMS；少一个操作，经验稳定性相同。 |
| SwiGLU | “大家都换过去的 FFN” | `Swish(W1 x) ⊙ W3 x → W2`。在 LM ppl 上胜过 ReLU/GELU。 |
| Cross-attention | “decoder 如何看 encoder” | Q 来自 decoder、K/V 来自 encoder outputs 的 MHA。 |
| FFN expansion | “中间 MLP 有多宽” | hidden-size 与 d_model 的比值，通常是 4（LayerNorm）或 2.6（SwiGLU）。 |
| Bias-free | “去掉 +b 项” | 现代 stack 省略 linear layers 中的 biases；ppl 稍好，模型更小。 |

## 延伸阅读

- [Vaswani et al. (2017). Attention Is All You Need](https://arxiv.org/abs/1706.03762) — 原始 block 规格。
- [Xiong et al. (2020). On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745) — 为什么 pre-norm 在深层模型中优于 post-norm。
- [Zhang, Sennrich (2019). Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467) — RMSNorm。
- [Shazeer (2020). GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202) — SwiGLU 论文。
- [HuggingFace `modeling_llama.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py) — 2026 年标准 decoder-only block。
