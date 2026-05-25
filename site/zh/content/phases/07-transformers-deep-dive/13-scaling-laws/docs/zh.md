# Scaling Laws

> 2020 年 Kaplan 论文说：模型越大，loss 越低。2022 年 Hoffmann 论文说：你训练得不够。Compute 会进入两个桶 — 参数和 tokens — 而分配方式并不显然。

**类型：** 学习
**语言：** Python
**前置要求：** 阶段 7 · 05（Full Transformer），阶段 7 · 07（GPT）
**时间：** ~45 分钟

## 问题

当你有 C FLOPs 训练 compute，并想得到最好的模型时，你面对两个旋钮：

1. **多少参数（N）？** 更大的模型，更高容量。
2. **多少训练 tokens（D）？** 更多数据，更好利用容量。

FLOPs 近似按 `6 × N × D` 增长。你可以提高 N、降低 D，也可以提高 D、降低 N。哪个更好？

2022 年之前，答案是“大力推 N”。GPT-3（2020）有 175B 参数，在约 300B tokens 上训练。比例约为每参数 1.7 tokens。Kaplan scaling laws 支持这个做法。

Hoffmann et al.（2022）训练了一组小模型 Chinchilla，发现了不同结果：最优比例更接近 **每参数 20 tokens**。GPT-3 训练不足 10×。Chinchilla（70B params，1.4T tokens）在每个 benchmark 上击败 GPT-3（175B，300B tokens），推理成本还低 2.5×。

2026 年是 Chinchilla 的世界 — 但有一个重要转折。Llama 3 8B 在 15 trillion tokens 上训练，比例是每参数 1,875 tokens。超过 Chinchilla-optimal 94 倍。对于会被大规模使用的模型，推理成本比训练成本更重要，所以用更小可部署 footprint 做 over-training（超过 Chinchilla）是 2026 年默认选择。

## 概念

![Chinchilla curves: loss vs compute at various N/D ratios](../assets/scaling-laws.svg)

### Hoffmann law

来自 Chinchilla 论文，loss 遵循：

```
L(N, D) = A / N^α + B / D^β + E
```

- `N` = parameters（非 embedding）。
- `D` = training tokens。
- `α ≈ 0.34`，`β ≈ 0.28`（大致对称）。
- `E ≈ 1.69`，不可约 loss ceiling。
- `A ≈ 406`，`B ≈ 411`。

随着 scale 上升，两个项彼此 trade off。在固定 compute（C = 6ND）下对 `N` 求导并求解：

```
N_opt ≈ 0.6 × (C/6)^0.5
D_opt ≈ 0.6 × (C/6)^0.5
D_opt / N_opt ≈ 20
```

Compute-optimal：每参数 20 tokens。

### 为什么仍然 over-training

Chinchilla-optimal 最小化的是每训练 FLOP 的训练 loss。但训练成本只付一次；推理成本会永远付。

对于每月服务一万亿 tokens 的 chatbot，推理主导总成本。Llama 的方法是：更小模型，更长训练。8B at 15T tokens 深度面向推理优化：

- 能放进 consumer GPUs。
- 延迟只是 70B Chinchilla-optimal 的一小部分。
- 质量对大多数任务足够接近。

DeepMind 2024 年论文（“Over-training is the new optimal”）形式化了这一点。对于 inference-dominated workloads，正确比例会更接近每参数 100–500 tokens，具体取决于 serving volume。

### Emergence vs smoothness

说法：某些能力（算术、多步推理、chain-of-thought following）会在某个 scale 突然“涌现”。

Schaeffer et al.（2023）认为这是 measurement artifact：emergent metrics 使用不连续 scoring（exact match、accuracy at threshold），隐藏了底层 logits 的平滑改善。连续指标（cross-entropy）显示的是平滑曲线。

2026 年共识是：通过 continuous loss 做预测是可靠的。Benchmark jumps 经常是 scorer artifacts。预算规划应面向 continuous metrics。

### 2026 年图景

Scaling laws 仍然有效，但：

| 因素 | 如何变化 |
|--------|-------------|
| Data quality | 筛选“好” tokens（Phi-style）会让曲线移动 >2× effective compute |
| MoE | 总参数与 active FLOPs 解耦；scaling laws 按 per-active-FLOP 看 |
| Post-training | 一些能力（instruction following、code）受 SFT+RLHF 影响超过 pretraining |
| Multimodality | Image + text tokens 一起 scale；每种 modality 有独立曲线 |
| Synthetic data | 模型生成训练数据；effective compute 可以复利 |

Muon optimizer（Kimi Moonlight，2024）在同等数据下相对 AdamW 展示了约 2× effective-compute gain。一些 2026 训练运行默认使用 Muon。它改变 scaling law 的绝对常数，不改变形状。

## 构建它

见 `code/main.py`。我们实现 Chinchilla loss equation，并在多个 compute budgets 下求解 compute-optimal `(N, D)`。

### 第 1 步：Chinchilla loss

```python
def chinchilla_loss(N, D, A=406.4, B=410.7, alpha=0.34, beta=0.28, E=1.69):
    return A / N ** alpha + B / D ** beta + E
```

在固定 `C = 6ND` 下，把 `L` 画成 `(N, D)` contour。找到最小值。

### 第 2 步：compute-optimal frontier

对从 `1e17` 到 `1e25` FLOPs 的 compute budgets，找到满足 `6ND = C` 且使 loss 最小的 `(N, D)`。验证比例 `D/N ≈ 20`。

### 第 3 步：over-training cost

计算训练一个小 10× 的模型（1/10 optimal N、10× optimal D）会多付出的 extra loss。报告换来的 inference FLOP savings（与 N 成比例）。

### 第 4 步：和真实模型比较

放入 GPT-3、Chinchilla、Llama 3 8B、DeepSeek-V3（active params）的已知 `(N, D)` pairs，比较 predicted vs reported loss。

## 使用它

你大概率不会自己训练前沿模型。但 scaling laws 会告诉你：

1. **你的 fine-tune 是否有足够数据。** 如果任务特定数据低于 base model 每参数 20 tokens，预期会在某个 loss floor 饱和。
2. **是否该选择更大的 base model。** 如果预算主要花在推理上，优先选择更小、训练更久的模型。
3. **收益什么时候递减。** 超过 Chinchilla-optimal 1000× 后，log-loss 变化会变成噪声。

**2026 年研究轨迹：**

- **Data-constrained regime。** Web 上高质量 tokens 数量有限（过滤后英文约 5–10 trillion）。前沿 pretraining 正接近这个上限。Synthetic data、multilingual、multimodal 和 RLHF-scaled fine-tuning 是下一批杠杆。
- **Compute-multiplier tricks。** Muon optimizer、MoE、更好的 data curation — 每个都移动绝对常数，不移动渐近线。
- **Scaling laws for RL。** 开放问题。早期证据显示 RL samples 上也有 power-law，但 exponent 和 pretraining 很不同。

## 交付它

见 `outputs/skill-training-budget-estimator.md`。这个 skill 会根据 compute budget、deployment constraints 和 target loss，为新的训练运行选择 `(N, D, hours, GPU)`。

## 练习

1. **简单。** 运行 `code/main.py`。打印 compute budgets `1e20`、`1e22`、`1e24` 下的 Chinchilla-optimal `(N, D)`。和真实模型表比较。
2. **中等。** 实现 Hoffmann loss-as-function-of-compute curve。对 compute-optimal frontier 画出 loss vs `log10(C)`。找出何时该 law 预测下一次 cross-entropy 降低 0.1 需要 `>10^28` FLOPs。
3. **困难。** 在同一数据集上训练 5 个 tiny models（100K 到 10M params），拟合你自己的 scaling law。估计 `α` 和 `E`。你的 exponents 和公开结果匹配得如何？

## 关键词

| 术语 | 大家怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| Parameters (N) | “模型大小” | 非 embedding 权重数；决定容量。 |
| Tokens (D) | “训练数据” | 训练中看到的 tokens 数；决定参数被利用得有多好。 |
| Compute (C) | “花掉的 FLOPs” | 对标准 transformer，约为 `6 × N × D`。 |
| Chinchilla-optimal | “D/N ≈ 20” | 最小化 pretraining 每 FLOP loss 的比例。 |
| Over-training | “超过 Chinchilla” | 多花训练 FLOPs 来省推理 FLOPs；D/N >> 20。 |
| Irreducible loss | “下限” | Scaling law 中的 `E` 项；数据本身的熵。 |
| Emergent capability | “scale 上的突然跳跃” | 经常是 scorer artifact；continuous loss 是平滑的。 |
| Effective compute | “训练效率乘数” | 更好的数据 / optimizer / architecture 会成倍增加一个 FLOP 的效果。 |

## 延伸阅读

- [Kaplan et al. (2020). Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361) — 第一篇 scaling law 论文；训练不足。
- [Hoffmann et al. (2022). Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556) — Chinchilla。
- [Schaeffer et al. (2023). Are Emergent Abilities of Large Language Models a Mirage?](https://arxiv.org/abs/2304.15004) — emergence 是 measurement artifact。
- [Sardana, Frankle (2024). Beyond Chinchilla-Optimal: Accounting for Inference in Language Model Scaling Laws](https://arxiv.org/abs/2401.00448) — 为什么 Llama 的 over-training 对其 workload 是正确的。
- [Jordan et al. (2024). Muon: An optimizer for hidden layers in neural networks](https://kellerjordan.github.io/posts/muon/) — 2× compute multiplier。
