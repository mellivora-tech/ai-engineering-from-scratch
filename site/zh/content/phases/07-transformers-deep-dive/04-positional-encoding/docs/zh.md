# Positional Encoding — Sinusoidal、RoPE、ALiBi

> Attention 对排列不敏感。没有位置信号时，“The cat sat on the mat”和“mat the on sat cat the”会产生相同输出。三个算法修复了它 — 每个算法都对“位置”意味着什么做了不同下注。

**类型：** 构建
**语言：** Python
**前置要求：** 阶段 7 · 02（Self-Attention），阶段 7 · 03（Multi-Head Attention）
**时间：** ~45 分钟

## 问题

Scaled dot-product attention 是顺序盲的。Attention 矩阵 `softmax(Q K^T / √d) V` 由成对相似度计算。打乱 `X` 的行，输出的行也会被同样打乱。Attention 内部没有任何东西关心位置。

这对 bag-of-words 模型不是 bug。但对语言、代码、音频、视频 — 任何顺序携带意义的东西 — 都是致命问题。

修复方式是以某种方式把位置注入 embedding。答案经历了三个时代：

1. **Absolute sinusoidal**（Vaswani 2017）。把位置的 `sin/cos` 加到 embedding 上。简单、不需要学习参数，但很难外推到训练长度之外。
2. **RoPE — Rotary Position Embeddings**（Su 2021）。按与位置成比例的角度旋转 Q 和 K 向量。直接在点积中编码 *相对* 位置。2026 年的主流。
3. **ALiBi — Attention with Linear Biases**（Press 2022）。完全跳过 embedding；根据距离给 attention scores 加上每个 head 的线性惩罚。长度外推非常好。

到 2026 年，几乎每个前沿开源模型都使用 RoPE：Llama 2/3/4、Qwen 2/3、Mistral、Mixtral、DeepSeek-V3、Kimi。少数长上下文模型使用 ALiBi 或它的现代变体。Absolute sinusoidal 已经是历史。

## 概念

![Sinusoidal absolute vs RoPE rotations vs ALiBi distance bias](../assets/positional-encoding.svg)

### Absolute sinusoidal

预先计算一个形状为 `(max_len, d_model)` 的固定矩阵 `PE`：

```
PE[pos, 2i]   = sin(pos / 10000^(2i / d_model))
PE[pos, 2i+1] = cos(pos / 10000^(2i / d_model))
```

然后在 attention 之前做 `X' = X + PE[:N]`。每个维度是不同频率的正弦波。模型学习从相位模式中读取位置。超过 `max_len` 就会失败：如果模型只看过位置 0–2047，就没有任何东西告诉它位置 2048 会发生什么。

### RoPE

旋转 Q 和 K 向量（不是 embeddings）。对于一对维度 `(2i, 2i+1)`：

```
[q'_2i    ]   [ cos(pos·θ_i)  -sin(pos·θ_i) ] [q_2i   ]
[q'_2i+1  ] = [ sin(pos·θ_i)   cos(pos·θ_i) ] [q_2i+1 ]

θ_i = base^(-2i / d_head),  base = 10000 by default
```

对位置为 `pos_k` 的 keys 应用同样旋转。点积 `q'_m · k'_n` 会变成只依赖 `(m - n)` 的函数。也就是说：**attention score 只依赖相对距离**，尽管旋转是用绝对位置驱动的。漂亮的技巧。

扩展 RoPE：可以缩放 `base`（NTK-aware、YaRN、LongRoPE），无需重新训练就外推到更长 context。Llama 3 就是这样从 8K 扩展到 128K context 的。

### ALiBi

跳过 embedding 技巧。直接给 attention scores 加 bias：

```
attn_score[i, j] = (q_i · k_j) / √d  -  m_h · |i - j|
```

其中 `m_h` 是每个 head 的 slope（例如 `1 / 2^(8·h/H)`）。近处 token 被增强；远处 token 被惩罚。没有训练时成本。论文显示，长度外推优于 sinusoidal，并在原训练长度上匹配 RoPE。

### 2026 年该选什么

| 变体 | 外推能力 | 训练成本 | 使用者 |
|---------|---------------|---------------|---------|
| Absolute sinusoidal | 差 | 免费 | original transformer, early BERT |
| Learned absolute | 无 | 很小 | GPT-2, GPT-3 |
| RoPE | 配合 scaling 时好 | 免费 | Llama 2/3/4, Qwen 2/3, Mistral, DeepSeek-V3, Kimi |
| RoPE + YaRN | 极好 | fine-tune 阶段 | Qwen2-1M, Llama 3.1 128K |
| ALiBi | 极好 | 免费 | BLOOM, MPT, Baichuan |

RoPE 胜出是因为它能嵌入 attention 而不改变架构，编码相对位置，并且它的 `base` 超参数提供了一个干净的长上下文 fine-tuning 旋钮。

## 构建它

### 第 1 步：sinusoidal encoding

见 `code/main.py`。4 行计算：

```python
def sinusoidal(N, d):
    pe = [[0.0] * d for _ in range(N)]
    for pos in range(N):
        for i in range(d // 2):
            theta = pos / (10000 ** (2 * i / d))
            pe[pos][2 * i]     = math.sin(theta)
            pe[pos][2 * i + 1] = math.cos(theta)
    return pe
```

在第一个 attention 层之前，把它加到 embedding 矩阵上。

### 第 2 步：把 RoPE 应用到 Q、K

RoPE 在 Q 和 K 上原地工作。对每一对维度：

```python
def apply_rope(x, pos, base=10000):
    d = len(x)
    out = list(x)
    for i in range(d // 2):
        theta = pos / (base ** (2 * i / d))
        c, s = math.cos(theta), math.sin(theta)
        a, b = x[2 * i], x[2 * i + 1]
        out[2 * i]     = a * c - b * s
        out[2 * i + 1] = a * s + b * c
    return out
```

关键点：对位置 `m` 的 Q 和位置 `n` 的 K 应用同一个函数。它们的点积会在每个坐标对上获得一个 `cos((m-n)·θ_i)` 因子。Attention 免费学到相对位置。

### 第 3 步：ALiBi slopes 和 bias

```python
def alibi_bias(n_heads, seq_len):
    # slope_h = 2 ** (-8 * h / n_heads) for h = 1..n_heads
    slopes = [2 ** (-8 * (h + 1) / n_heads) for h in range(n_heads)]
    bias = []
    for m in slopes:
        row = [[-m * abs(i - j) for j in range(seq_len)] for i in range(seq_len)]
        bias.append(row)
    return bias  # add to attention scores before softmax
```

把 `bias[h]` 加到 head `h` 的 `(seq_len, seq_len)` attention score 矩阵上，然后 softmax。

### 第 4 步：验证 RoPE 的相对距离性质

选两个随机向量 `a, b`。先按 `(pos_a, pos_b)` 旋转。再按 `(pos_a + k, pos_b + k)` 旋转。两个点积必须在浮点误差内相等。这个性质就是 RoPE 的全部意义 — 它对绝对偏移不变，只关心相对距离。

## 使用它

PyTorch 2.5+ 在 `torch.nn.functional` 中提供 RoPE utilities。大多数生产代码使用 `flash_attn` 或 `xformers`，RoPE 会在 attention kernel 内部应用。

```python
from transformers import AutoModel
model = AutoModel.from_pretrained("meta-llama/Llama-3.2-3B")
# model.config.rope_scaling → {"type": "yarn", "factor": 32.0, "original_max_position_embeddings": 8192}
```

**2026 年的长上下文技巧：**

- **NTK-aware interpolation。** 从 4K 扩展到 16K+ 时，把 `base` 重新缩放为 `base * (scale_factor)^(d/(d-2))`。
- **YaRN。** 更聪明的 interpolation，在长 context 上保持 attention entropy。Llama 3.1 128K 使用它。
- **LongRoPE。** Microsoft 2024 年的方法，用 evolutionary search 选择每个维度的 scale factors。Phi-3-Long 使用它。
- **Position interpolation + fine-tuning。** 只要按扩展因子缩小位置，然后 fine-tune 1–5B tokens。效果出奇地好。

## 交付它

见 `outputs/skill-positional-encoding-picker.md`。这个 skill 会根据目标 context 长度、外推需求和训练预算，为新模型选择 encoding 策略。

## 练习

1. **简单。** 把 sinusoidal `PE` 矩阵画成 heatmap，设置 `max_len=512, d=128`。确认“维度索引越大，条纹越宽”的模式。
2. **中等。** 实现 NTK-aware RoPE scaling。在长度 256 的序列上训练 tiny LM，然后在长度 1024 上分别用和不用 scaling 测试。测量 perplexity。
3. **困难。** 在同一个 attention module 中实现 ALiBi 和 RoPE。在长度 512 的 copy task 上训练 4 层 transformer。测试时外推到 2048。比较退化程度。

## 关键词

| 术语 | 大家怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| Positional encoding | “告诉 attention 顺序” | 添加到 embeddings 或 attention 中、用于编码位置的任何信号。 |
| Sinusoidal | “最早那个” | 按几何频率取 `sin/cos` 并加到 embeddings 上；不能外推。 |
| RoPE | “Rotary embeddings” | 按位置相关角度旋转 Q、K；点积编码相对距离。 |
| ALiBi | “线性 bias 技巧” | 给 attention scores 加 `-m·|i-j|`；不需要 embedding，外推很好。 |
| base | “RoPE 的旋钮” | RoPE 中的频率缩放器；增大它可以在推理时扩展 context。 |
| NTK-aware | “RoPE scaling 技巧” | 重新缩放 `base`，让 context 扩大时高频维度不会被挤压。 |
| YaRN | “花哨版本” | 保持 attention entropy 的逐维 interpolation+extrapolation。 |
| Extrapolation | “超过训练长度仍能工作” | position scheme 能否在训练见过的 `max_len` 之外产生正确输出？ |

## 延伸阅读

- [Vaswani et al. (2017). Attention Is All You Need §3.5](https://arxiv.org/abs/1706.03762) — 原始 sinusoidal。
- [Su et al. (2021). RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864) — RoPE 论文。
- [Press, Smith, Lewis (2021). Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation](https://arxiv.org/abs/2108.12409) — ALiBi。
- [Peng et al. (2023). YaRN: Efficient Context Window Extension of Large Language Models](https://arxiv.org/abs/2309.00071) — 先进的 RoPE scaling。
- [Chen et al. (2023). Extending Context Window of Large Language Models via Positional Interpolation](https://arxiv.org/abs/2306.15595) — Meta 的 Llama 2 长上下文论文。
- [Ding et al. (2024). LongRoPE: Extending LLM Context Window Beyond 2 Million Tokens](https://arxiv.org/abs/2402.13753) — Microsoft 方法，Phi-3-Long 使用，且在 Use It 中被引用。
- [HuggingFace Transformers — `modeling_rope_utils.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_rope_utils.py) — 每一种 RoPE scaling scheme（default、linear、dynamic、YaRN、LongRoPE、Llama-3）的生产级实现。
