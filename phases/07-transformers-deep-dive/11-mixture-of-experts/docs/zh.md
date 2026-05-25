# Mixture of Experts (MoE)

> 一个 dense 70B transformer 会为每个 token 激活每个参数。一个 671B MoE 每个 token 只激活 37B 参数，并在每个 benchmark 上击败它。Sparsity 是这个十年最重要的 scaling idea。

**类型：** 构建
**语言：** Python
**前置要求：** 阶段 7 · 05（Full Transformer），阶段 7 · 07（GPT）
**时间：** ~45 分钟

## 问题

Dense transformer 的推理 FLOPs 等于参数量（forward pass 乘以 2）。放大 dense model，每个 token 都要付完整账单。到 2024 年，前沿模型撞上了 compute wall：想显著更聪明，就需要每 token 指数级更多 FLOPs。

Mixture of Experts 打断了这个连接。把每个 FFN 替换成 `E` 个独立 experts + 一个 router，router 为每个 token 选择 `k` 个 experts。总参数量 = `E × FFN_size`。每 token active parameters = `k × FFN_size`。典型 2026 配置：`E=256`、`k=8`。存储按 `E` 增长，计算按 `k` 增长。

2026 年前沿几乎全是 MoE：DeepSeek-V3（671B total / 37B active）、Mixtral 8×22B、Qwen2.5-MoE、Llama 4、Kimi K2、gpt-oss。在 Artificial Analysis 的独立 leaderboard 上，前 10 个开源模型全都是 MoE。

## 概念

![MoE layer: router selects k of E experts per token](../assets/moe.svg)

### FFN 替换

Dense transformer block：

```
h = x + attn(norm(x))
h = h + FFN(norm(h))
```

MoE block：

```
h = x + attn(norm(x))
scores = router(norm(h))              # (N_tokens, E)
top_k = argmax_k(scores)              # pick k of E per token
h = h + sum_{e in top_k}(
        gate(scores[e]) * Expert_e(norm(h))
    )
```

每个 expert 是独立 FFN（通常是 SwiGLU）。Router 是单个 linear layer。每个 token 选择自己的 `k` 个 experts，并获得它们输出的 gated mixture。

### Load-balancing 问题

如果 router 把 90% tokens 都送进 expert 3，其他 experts 会饿死。已经尝试过三种修复：

1. **Auxiliary load-balancing loss**（Switch Transformer、Mixtral）。加入一个与 expert usage 方差成比例的 penalty。有效，但增加一个超参数和第二个梯度信号。
2. **Expert capacity + token dropping**（早期 Switch）。每个 expert 最多处理 `C × N/E` tokens；溢出的 tokens 跳过该层。伤害质量。
3. **Auxiliary-loss-free balancing**（DeepSeek-V3）。添加一个每 expert 的学习 bias，移动 router 的 top-k selection。Bias 在训练 loss 外更新。主 objective 没有 penalty。2024 年的重要解锁。

DeepSeek-V3 的做法：每个训练步骤后，对每个 expert 检查 usage 是高于还是低于目标。用 `±γ` 轻推 bias。Selection 使用 `scores + bias`。用于 gating 的 expert probabilities 仍然使用未改动的 raw `scores`。把 routing 和 expression 解耦。

### Shared experts

DeepSeek-V2/V3 还把 experts 分成 *shared* 和 *routed*。每个 token 都会通过所有 shared experts。Routed experts 通过 top-k 选择。Shared experts 捕捉通用知识；routed experts 专门化。V3 运行 1 个 shared expert 加 top-8 of 256 routed。

### Fine-grained experts

经典 MoE（GShard、Switch）：每个 expert 和完整 FFN 一样宽。`E` 较小（8–64），`k` 较小（1–2）。

现代 fine-grained MoE（DeepSeek-V3、Qwen-MoE）：每个 expert 更窄（1/8 FFN size）。`E` 很大（256+），`k` 更大（8+）。总参数量相同，但组合数量扩展得快得多。`C(256, 8) = 400 trillion` 个可能的 token “experts”。质量上升，延迟保持平坦。

### 成本画像

每 token、每层：

| 配置 | Active params / token | Total params |
|--------|-----------------------|--------------|
| Mixtral 8×22B | ~39B | 141B |
| Llama 3 70B (dense) | 70B | 70B |
| DeepSeek-V3 | 37B | 671B |
| Kimi K2 (MoE) | ~32B | 1T |

DeepSeek-V3 在几乎每个 benchmark 上都击败 Llama 3 70B（dense），同时每 token **active FLOPs 更少**。更多参数 = 更多知识。更多 active FLOPs = 每 token 更多计算。MoE 把它们解耦。

### 代价：内存

无论哪些 experts 被激活，所有 experts 都必须驻留在 GPU 上。671B 模型需要约 1.3 TB VRAM 存放 fp16 权重。前沿 MoE 部署需要 expert parallelism — 把 experts 分片到多个 GPUs，跨网络 route tokens。延迟由 all-to-all communication 主导，而不是 matmul。

## 构建它

见 `code/main.py`。一个纯 stdlib 的紧凑 MoE layer，包含：

- `n_experts=8` SwiGLU-ish experts（为了说明，每个只是一个 linear）
- top-k=2 routing
- softmax-normalized gating weights
- 通过 per-expert bias 实现 auxiliary-loss-free balancing

### 第 1 步：router

```python
def route(hidden, W_router, top_k, bias):
    scores = [sum(h * w for h, w in zip(hidden, W_router[e])) for e in range(len(W_router))]
    biased = [s + b for s, b in zip(scores, bias)]
    top_idx = sorted(range(len(biased)), key=lambda i: -biased[i])[:top_k]
    # softmax over ORIGINAL scores of the chosen experts
    chosen = [scores[i] for i in top_idx]
    m = max(chosen)
    exps = [math.exp(c - m) for c in chosen]
    s = sum(exps)
    gates = [e / s for e in exps]
    return top_idx, gates
```

Bias 影响 selection，不影响 gate weight。这就是 DeepSeek-V3 技巧 — bias 修正 load imbalance，而不 steering 模型预测。

### 第 2 步：让 100 个 tokens 通过 router

追踪每个 expert 被触发多少次。没有 bias 时，usage 是倾斜的。加入 bias update loop（over-used experts 用 `-γ`，under-used 用 `+γ`）后，usage 会在几轮内收敛到均匀分布。

### 第 3 步：参数量比较

打印一个 MoE 配置的 “dense equivalent”。DeepSeek-V3 形状：256 routed + 1 shared，8 active，d_model=7168。总参数量惊人。Active count 是 dense Llama 3 70B 的七分之一。

## 使用它

HuggingFace loading：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("mistralai/Mixtral-8x22B-v0.1")
```

2026 年生产推理：vLLM 原生支持 MoE routing。SGLang 拥有最快的 expert-parallel path。两者都自动处理 top-k selection 和 expert parallelism。

**什么时候选择 MoE：**
- 你想用更低的每 token 推理成本获得前沿质量。
- 你有 VRAM / expert-parallel infrastructure。
- 你的 workload 是 token-heavy（chat、code），不是 context-heavy（long docs）。

**什么时候不要选择 MoE：**
- Edge deployment — 任何 active FLOP 都要付完整 storage。
- Latency-critical single-user serving — expert routing 增加开销。
- 小模型（<7B）— MoE 的质量优势只在超过某个 compute threshold（~6B active params）后出现。

## 交付它

见 `outputs/skill-moe-configurator.md`。这个 skill 会根据 parameter budget、training tokens 和 deployment target，为新的 MoE 选择 E、k 和 shared-expert layout。

## 练习

1. **简单。** 运行 `code/main.py`。观察 auxiliary-loss-free bias update 如何在 50 次迭代内拉平 expert usage。
2. **中等。** 用 hash-based router（确定性、无学习）替换 learned router。比较 quality 和 balance。为什么 learned router 更好？
3. **困难。** 实现 GRPO-style “rollout-matched routing”（DeepSeek-V3.2 技巧）：记录推理时哪些 experts 被触发，在 gradient computation 中强制使用同样 routing。在 toy policy-gradient setup 上测量影响。

## 关键词

| 术语 | 大家怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| Expert | “许多 FFN 中的一个” | 独立 feed-forward network；参数专门用于 FFN 计算的稀疏切片。 |
| Router | “Gate” | 一个很小的 linear layer，为每个 token 对每个 expert 打分；top-k selection。 |
| Top-k routing | “每 token k 个 active experts” | 每个 token 的 FFN 计算正好经过 k 个 experts，并由 gate 加权。 |
| Auxiliary loss | “Load-balance penalty” | 额外 loss term，用来惩罚 expert usage 倾斜。 |
| Auxiliary-loss-free | “DeepSeek-V3 的技巧” | 只通过 router selection 上的 per-expert bias 来平衡；没有额外梯度。 |
| Shared expert | “Always on” | 每个 token 都经过的额外 expert；捕捉通用知识。 |
| Expert parallelism | “按 expert 分片” | 把不同 experts 分发到不同 GPUs；跨网络 route tokens。 |
| Sparsity | “Active params < total params” | 比例 `k × expert_size / (E × expert_size)`；DeepSeek-V3 为 37/671 ≈ 5.5%。 |

## 延伸阅读

- [Shazeer et al. (2017). Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer](https://arxiv.org/abs/1701.06538) — 这个想法。
- [Fedus, Zoph, Shazeer (2022). Switch Transformer: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](https://arxiv.org/abs/2101.03961) — Switch，经典 MoE。
- [Jiang et al. (2024). Mixtral of Experts](https://arxiv.org/abs/2401.04088) — Mixtral 8×7B。
- [DeepSeek-AI (2024). DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437) — MLA + auxiliary-loss-free MoE + MTP。
- [Wang et al. (2024). Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts](https://arxiv.org/abs/2408.15664) — bias-based balancing 论文。
- [Dai et al. (2024). DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models](https://arxiv.org/abs/2401.06066) — 本课 router 使用的 fine-grained + shared-expert split。
- [Kim et al. (2022). DeepSpeed-MoE: Advancing Mixture-of-Experts Inference and Training](https://arxiv.org/abs/2201.05596) — 原始 shared-expert 论文。
