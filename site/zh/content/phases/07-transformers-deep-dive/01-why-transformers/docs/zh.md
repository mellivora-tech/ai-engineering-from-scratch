# 为什么是 Transformer — RNN 的问题

> RNN 一次处理一个 token。Transformer 一次处理所有 token。这个单一的架构下注，改变了 2017 年之后深度学习里的每一条 scaling curve。

**类型：** 学习
**语言：** Python
**前置要求：** 阶段 3（深度学习核心），阶段 5 · 09（Sequence-to-Sequence），阶段 5 · 10（Attention Mechanism）
**时间：** ~45 分钟

## 问题

2017 年之前，地球上每一个最先进的序列模型 — 语言、翻译、语音 — 都是 recurrent neural network。LSTM 和 GRU 在相当于 ImageNet 地位的翻译 benchmark 上赢了半个十年。那时大家手里只有这一个工具。

它们有三个致命弱点。顺序计算意味着你不能沿着时间轴并行：token `t+1` 需要 token `t` 的 hidden state。一个 1,024-token 序列意味着在 GPU 上走 1,024 个串行步骤，而这个 GPU 每个周期可以做 1,000,000 次浮点运算。训练 wall-clock time 会随序列长度线性增长，而硬件本来是为并行设计的。

梯度消失意味着 50 个 token 之前的信息已经被压过 50 层非线性。带门控的 recurrent unit（LSTM、GRU）缓解了这种挤压，但从未消除它。长距离依赖 — “the book I read last summer on a plane to Kyoto was…” — 经常失败。

固定宽度 hidden state 意味着 encoder 必须在 decoder 看到任何东西之前，把整个源序列压进一个向量里。源序列是 5 个 token 还是 500 个 token 都没关系；瓶颈的形状一样。

2017 年论文 “Attention Is All You Need” 提出了一个激进想法：完全丢掉 recurrence。让每个位置并行 attend 到每个其他位置。用一次大型矩阵乘法训练，而不是 1,024 次顺序计算。

到 2026 年，结果已经主导了每一种模态。语言（GPT-5、Claude 4、Llama 4），视觉（ViT、DINOv2、SAM 3），音频（Whisper），生物学（AlphaFold 3），机器人（RT-2）。同一个 block，不同的输入。

## 概念

![RNN sequential compute vs Transformer parallel attention](../assets/rnn-vs-transformer.svg)

**Recurrence 是瓶颈。** RNN 计算 `h_t = f(h_{t-1}, x_t)`。每一步依赖上一步。你不能在 `h_4` 之前计算 `h_5`。在拥有 10,000+ 并行核心的现代 GPU 上，长序列会浪费 99% 的硅片。

**Attention 是广播。** Self-attention 会为每一对 `(i, j)` 同时计算 `output_i = sum_j(a_ij * v_j)`。整个 N×N attention 矩阵在一次 batched matmul 中填满。没有一步依赖另一步。GPU 很喜欢这种形状。

**加速不是一个常数。** 它是 `O(N)` 串行深度和 `O(1)` 串行深度之间的差异。实践中，在 N=512 且硬件匹配时，transformer 每个 epoch 训练速度快 5–10×；随着序列长度增加，这个差距会继续扩大，直到撞上 attention 的 `O(N²)` 内存墙（后来 Flash Attention 修复了这个常数问题 — 见第 12 课）。

**Transformer 的代价。** Attention 内存按 `O(N²)` 增长。2K context 没问题。128K context 时，你需要 sliding window、RoPE extrapolation、Flash Attention tiling，或者 linear attention 变体。Recurrence 在时间和内存上都是 `O(N)`；transformer 用内存换时间，然后又通过并行把时间赢回来。

**Inductive bias 的转移。** RNN 假设局部性和近因性。Transformer 不做这种假设 — 任意一对 token 都可能 attention。也正因为如此，transformer 需要更多数据才能训好，但一旦有足够数据就能扩展得更远。Chinchilla（2022）形式化了这一点：只要 token 足够多，同等参数量的 transformer 总会击败 RNN。

## 构建它

这里没有神经网络 — 我们用数值方式模拟核心瓶颈，让你在自己的笔记本上感受到差距。

### 第 1 步：测量串行深度

见 `code/main.py`。我们构建两个函数。一个把序列编码成加法链（串行，像 RNN）。另一个把它编码成并行 reduction（广播，像 attention）。数学相同，依赖图不同。

```python
def rnn_style(xs):
    h = 0.0
    for x in xs:
        h = 0.9 * h + x   # can't parallelize: h depends on previous h
    return h

def attention_style(xs):
    return sum(xs) / len(xs)  # every x is independent
```

我们在最长 100,000 个元素的序列上计时。RNN 版本是 O(N)，并且只能走单条 CPU pipeline。即使在纯 Python 中，attention-style reduction 在长度 ≥ 1,000 时也会胜出，因为 Python 的 `sum()` 是用 C 实现的，每一步不需要解释器开销。

### 第 2 步：统计理论操作数

两个算法都做 N 次加法。差异在于 *dependency depth*：有多少操作必须按顺序完成，下一步才能开始。RNN 深度 = N。Attention 用树形 reduction 时深度 = log(N)，用 parallel scan 时深度 = 1。决定 GPU 时间的是深度，不是操作数。

### 第 3 步：长序列上的经验 scaling

我们打印一张计时表，让 O(N) 差距可见。在一台 2026 年的 Mac 笔记本上，1,000 个元素以下的序列太快，很难测量。100,000 个元素的序列会显示出清晰的线性扫描。把它放大到一个 16,384-token transformer，并和 12 层 LSTM 等价模型相比，你就能看到为什么训练 wall-clock 在 2016 年是阻碍。

## 使用它

2026 年什么时候仍然选择 RNN：

| 场景 | 选择 |
|-----------|------|
| 流式推理，一次一个 token，常量内存 | RNN 或 state-space model（Mamba、RWKV） |
| 极长序列（>1M tokens），attention 内存爆炸 | Linear attention、Mamba 2、Hyena |
| 没有 matmul 加速器的边缘设备 | Depthwise-separable RNN 在 FLOPs/watt 上仍然胜出 |
| 其他任何情况（训练、批量推理、最高 128K context） | Transformer |

像 Mamba 这样的 state-space model（SSM）本质上是带结构化参数化的 RNN，拿到了两边的好处：`O(N)` scan 内存，以及通过 selective scan 实现的并行训练。它们能恢复 transformer 90% 的质量，同时有更好的长上下文 scaling。到 2026 年，大多数前沿实验室都会训练混合 SSM+transformer 模型（例如 Jamba、Samba）— recurrence 没死，它成了一个组件。

## 交付它

见 `outputs/skill-architecture-picker.md`。这个 skill 会根据长度、吞吐量和训练预算约束，为新的序列问题选择架构。对于超过 1B tokens 的训练运行，它应该总是拒绝推荐纯 RNN，除非同时说明 trade-off。

## 练习

1. **简单。** 从 `code/main.py` 取出 `rnn_style`，把标量 hidden state 替换成长度为 64 的 hidden state 向量。重新测量。串行开销会如何随 hidden-state 维度增长？
2. **中等。** 用纯 Python 实现 parallel prefix-sum（Hillis-Steele scan）。验证它在长度 1024 上产生和串行 scan 相同的数值输出。统计深度。
3. **困难。** 把 attention-style reduction 移植到 GPU 上的 PyTorch。扫过从 64 到 65,536 的序列长度并给两者计时。画图并解释曲线形状。

## 关键词

| 术语 | 大家怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| Recurrence | “RNN 是顺序的” | 第 `t` 步依赖第 `t-1` 步的计算，迫使时间轴上串行执行。 |
| Serial depth | “图有多深” | 最长的依赖操作链；即使有无限硬件，也会限制 wall-clock。 |
| Attention | “让 token 互相看” | 加权和 `sum_j a_ij v_j`，其中 `a_ij` 来自位置 i 和 j 之间的相似度分数。 |
| Context window | “模型能看到多少” | 一个 attention 层可接收的 position 数；二次内存成本在这里增长。 |
| Inductive bias | “写进架构的假设” | 关于数据长什么样的先验；CNN 假设平移不变，RNN 假设近因性。 |
| State-space model | “背后有代数的 RNN” | 通过结构化 state-space 矩阵参数化 recurrence，使其可以并行训练。 |
| Quadratic bottleneck | “为什么 context 这么贵” | Attention 内存 = 序列长度上的 `O(N²)`；Flash Attention 隐藏常数，不改变 scaling。 |

## 延伸阅读

- [Vaswani et al. (2017). Attention Is All You Need](https://arxiv.org/abs/1706.03762) — 让 recurrence 退出主流 NLP 的论文。
- [Bahdanau, Cho, Bengio (2014). Neural MT by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473) — attention 诞生之处，当时还接在 RNN 上。
- [Hochreiter, Schmidhuber (1997). Long Short-Term Memory](https://www.bioinf.jku.at/publications/older/2604.pdf) — 原始 LSTM 论文，留作记录。
- [Gu, Dao (2023). Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752) — 对 transformer 的现代 recurrent 回答。
