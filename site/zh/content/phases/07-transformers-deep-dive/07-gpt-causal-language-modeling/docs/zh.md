# GPT — Causal Language Modeling

> BERT 看两边。GPT 只看过去。三角 mask 是现代 AI 中影响最大的单行代码。

**类型：** 构建
**语言：** Python
**前置要求：** 阶段 7 · 02（Self-Attention），阶段 7 · 05（Full Transformer），阶段 7 · 06（BERT）
**时间：** ~75 分钟

## 问题

语言模型回答一个问题：给定前 `t-1` 个 tokens，token `t` 上的概率分布是什么？用这个信号训练 — next-token prediction — 你会得到一个可以一次生成一个 token 的任意文本生成模型。

为了在整个序列上并行端到端训练，你需要每个位置的预测只依赖更早的位置。否则模型会通过看答案来轻松作弊。

Causal mask 做的就是这件事。它是一个上三角矩阵，里面是 `-inf`，在 softmax 之前加到 attention scores 上。Softmax 之后，这些位置变成 0。每个位置只能 attend 到自己和更早的位置。而因为你一次把它应用到整个序列，所以一次 forward pass 就能得到 N 个并行的 next-token predictions。

GPT-1（2018）、GPT-2（2019）、GPT-3（2020）、GPT-4（2023）、GPT-5（2024）、Claude、Llama、Qwen、Mistral、DeepSeek、Kimi — 它们全都是 decoder-only causal transformers，核心循环相同。只是更大、更好的数据，以及更好的 RLHF。

## 概念

![Causal mask creates a triangular attention matrix](../assets/causal-attention.svg)

### Mask

给定长度为 `N` 的序列，构建一个 `N × N` 矩阵：

```
M[i, j] = 0       if j <= i
M[i, j] = -inf    if j > i
```

在 softmax 之前把 `M` 加到原始 attention scores 上。`exp(-inf) = 0`，所以 masked positions 贡献零权重。Attention 矩阵的每一行都是只覆盖过去位置的概率分布。

实现成本：一次 `torch.tril()` 调用。计算时间：纳秒。对整个领域的影响：一切。

### 并行训练，串行推理

训练：对整个 `(N, d_model)` 序列做一次 forward-pass，计算 N 个 cross-entropy losses（每个位置一个），求和，backprop。沿序列并行。这就是 GPT 训练可扩展的原因 — 你可以在一次 GPU pass 中处理一个 batch 的 1M tokens。

推理：你逐 token 生成。输入 `[t1, t2, t3]`，得到 `t4`。输入 `[t1, t2, t3, t4]`，得到 `t5`。输入 `[t1, t2, t3, t4, t5]`，得到 `t6`。KV cache（第 12 课）会保存 `t1…tn` 的 hidden states，这样每步不必重新计算它们。但推理时的串行深度 = 输出长度。这就是 autoregressive tax，也是每个 LLM 的解码延迟瓶颈。

### Loss — shift-by-one

给定 tokens `[t1, t2, t3, t4]`：

- 输入：`[t1, t2, t3]`
- Targets：`[t2, t3, t4]`

对每个位置 `i`，计算 `-log P(target_i | inputs[:i+1])`。求和。这就是整个序列的 cross-entropy。

你听说过的每个 transformer LM 都用这个 loss 训练。Pre-training、fine-tuning、SFT — 同一个 loss，不同数据。

### Decoding 策略

训练之后，sampling 选择比许多人以为的更重要。

| 方法 | 做什么 | 什么时候用 |
|--------|--------------|-------------|
| Greedy | 每步 argmax | 确定性任务、代码补全 |
| Temperature | logits 除以 T，再采样 | 创意任务，T 越高多样性越大 |
| Top-k | 只从 top-k tokens 采样 | 去掉低概率长尾 |
| Top-p (nucleus) | 从 cumulative prob ≥ p 的最小集合采样 | 2020+ 默认；适应分布形状 |
| Min-p | 保留 `p > min_p * max_p` 的 tokens | 2024+；比 top-p 更擅长拒绝长尾 |
| Speculative decoding | Draft model 提议 N 个 tokens，大模型验证 | 同质量下降低 2–3× 延迟 |

2026 年，对于 open-weights models，min-p + temperature 0.7 是一个合理默认值。Speculative decoding 是任何生产推理栈的基本配置。

### “GPT recipe” 为什么有效

1. **Decoder-only。** 没有 encoder 开销。每层一次 attention + FFN。
2. **Scaling。** 124M → 1.5B → 175B → 万亿级。Chinchilla scaling laws（第 13 课）告诉你如何花 compute。
3. **In-context learning。** 大约在 6B–13B 出现。模型无需 fine-tuning 就能跟随 few-shot examples。
4. **RLHF。** 基于人类偏好做 post-training，把原始预训练文本模型转成聊天助手。
5. **Pre-norm + RoPE + SwiGLU。** 在规模上稳定训练。

核心架构自 GPT-2 以来变化不大。真正有趣的事都发生在数据、规模和 post-training 中。

## 构建它

### 第 1 步：causal mask

见 `code/main.py`。一行：

```python
def causal_mask(n):
    return [[0.0 if j <= i else float("-inf") for j in range(n)] for i in range(n)]
```

把它加到 softmax 之前的 attention scores 上。这就是整个机制。

### 第 2 步：一个 2 层 GPT-ish 模型

堆叠两个 decoder blocks（masked self-attention + FFN，没有 cross-attention）。加上 token embedding、positional encoding 和 unembedding（与 token embedding 矩阵 tied — 自 GPT-2 以来的标准技巧）。

### 第 3 步：端到端 next-token prediction

在 20-token toy vocab 上，每个位置产生 logits。对 shift-by-one target 计算 cross-entropy loss。没有 gradient — 这是 forward-pass sanity check。

### 第 4 步：sampling

实现 greedy、temperature、top-k、top-p、min-p。在固定 prompt 上运行每种方法并比较输出。一个 sampling 函数只有 10 行。

## 使用它

PyTorch，2026 惯用写法：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")

prompt = "Attention is all you need because"
inputs = tok(prompt, return_tensors="pt")
out = model.generate(
    **inputs,
    max_new_tokens=64,
    temperature=0.7,
    top_p=0.9,
    do_sample=True,
)
print(tok.decode(out[0]))
```

在底层，`generate()` 会运行 forward pass，取 final-position logits，采样下一个 token，append，然后重复。每个生产 LLM 推理栈（vLLM、TensorRT-LLM、llama.cpp、Ollama、MLX）都实现了同样循环，并做重度优化 — batched prefill、continuous batching、KV cache paging、speculative decoding。

**GPT vs BERT，各用一句话：** GPT 预测 `P(x_t | x_{<t})`。BERT 预测 `P(x_masked | x_unmasked)`。Loss 决定模型是否能生成。

## 交付它

见 `outputs/skill-sampling-tuner.md`。这个 skill 会为新的生成任务选择 sampling 参数，并标出什么时候必须使用确定性 decoding。

## 练习

1. **简单。** 运行 `code/main.py`，验证 softmax 后的 causal attention 矩阵是下三角。抽查：第 3 行应该只在第 0–3 列有权重。
2. **中等。** 实现 width 4 的 beam search。在 10 个短 prompts 上比较 beam-4 和 greedy 的 perplexity。Beam 总是赢吗？（提示：通常在翻译中是，在开放式聊天中不是。）
3. **困难。** 实现 speculative decoding：用 tiny 2-layer 模型作为 draft，用 6-layer 模型作为 verifier。测量 100 个长度 64 completions 的 wall-clock speedup。确认输出匹配 verifier 的 greedy。

## 关键词

| 术语 | 大家怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| Causal mask | “三角形” | 加到 attention scores 上的上三角 `-inf` 矩阵，使位置 `i` 只能看到位置 `≤ i`。 |
| Next-token prediction | “那个 loss” | 模型分布和真实下一个 token 在每个位置上的 cross-entropy。 |
| Autoregressive | “一次生成一个” | 把输出再喂回输入；并行性只存在于训练中，不存在于生成中。 |
| Logits | “Softmax 前分数” | LM head 在 softmax 前的原始输出；sampling 发生在这里。 |
| Temperature | “创造力旋钮” | logits 除以 T；T→0 = greedy，T→∞ = uniform。 |
| Top-p | “Nucleus sampling” | 把分布截断到累积和 ≥p 的最小集合；从剩余部分采样。 |
| Min-p | “比 top-p 更好” | 保留 `p ≥ min_p × max_p` 的 tokens；根据分布尖锐程度自适应 cutoff。 |
| Speculative decoding | “Draft + verify” | 便宜模型提议 N 个 tokens；大模型并行验证。 |
| Teacher forcing | “训练技巧” | 训练时喂入真实前一个 token，而不是模型预测。每个 seq2seq LM 的标准做法。 |

## 延伸阅读

- [Radford et al. (2018). Improving Language Understanding by Generative Pre-Training](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) — GPT-1。
- [Radford et al. (2019). Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) — GPT-2。
- [Brown et al. (2020). Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165) — GPT-3 和 in-context learning。
- [Leviathan, Kalman, Matias (2023). Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) — speculative decoding 论文。
- [HuggingFace `modeling_llama.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py) — 标准 causal-LM 参考代码。
