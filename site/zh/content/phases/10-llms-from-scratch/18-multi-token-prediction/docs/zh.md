# Multi-Token Prediction（MTP）

> 从 GPT-2 到 Llama 3，每个 autoregressive LLM 都在每个位置上训练一个 loss：预测下一个 token。DeepSeek-V3 在每个位置上增加了第二个 loss：预测再下一个 token。额外的 14B 参数（在 671B 模型上）通过 gradient flow 蒸馏回主模型，而训练好的 MTP heads 在 inference 时被复用为 speculative-decoding drafters，acceptance 超过 80%。1.8× generation throughput 几乎是白送的。本课会构建 DeepSeek 技术报告中的 sequential MTP module，计算 loss 和 shared-head parameter layout，并解释为什么 MTP 保留 causal chain，而 Gloeckle et al. 最初的 parallel MTP 会破坏它。

**类型：** 构建
**语言：** Python（stdlib）
**先修：** Phase 10 · 04（pre-training a mini GPT）、Phase 10 · 15（speculative decoding）
**时间：** 约 60 分钟

## 学习目标

- 说出 MTP training objective，并推导跨 prediction depths 的 joint loss。
- 解释 Gloeckle et al. 的 parallel MTP heads（2024）与 DeepSeek-V3 的 sequential MTP modules 的区别，以及为什么 sequential design 能保留 causal chain。
- 计算给 pre-training run 增加 MTP modules 的 parameter 和 memory overhead。
- 从零实现一个 MTP module：shared embedding、per-depth transformer block、projection 和 shared output head。

## 问题

Next-token prediction 是标准 LLM training objective。每个 hidden state 只被监督预测一件事：紧随其后的 token。这是一个惊人地弱的信号。序列中的大多数信息都延伸到不止一个 token 之外：结构、一致性、事实性、算术流。模型必须通过 trillions of tokens 上许多 one-token signals 的累积来学习这些。

MTP 提出的问题是：如果每个 hidden state 同时被监督预测多个 future tokens 会怎样？Gloeckle et al.（Meta, 2024）展示这会有帮助。他们的实现是在 backbone 上放几个独立 output heads，每个预测不同 offset。并行、简单，但这些 heads 看到的是同一个 hidden state，没有任何 hierarchical refinement，而且预测之间没有 causal chain，因此不能用于 speculative decoding。

DeepSeek-V3（2024 年 12 月）把 MTP 重新设计为 sequential modules，在每个 prediction depth 保持 causal chain。模型从 `h_i^(0)` 预测 `t+1`，然后从新的 hidden state `h_i^(1)` 预测 `t+2`，其中 `h_i^(1)` 结合了 `h_i^(0)` 与 `E(t+1)` embedding，依此类推。每个 depth 都有自己的小 transformer block。Shared embedding 和 shared output head 让 parameter overhead 保持适中。在 DeepSeek-V3 的规模上，在 671B main-model weights 之上，MTP modules 增加 14B 参数。这个 2% overhead 同时买到了更密集的训练信号，以及 inference 时现成的 speculative-decoding draft。

本课会从零构建单个 MTP module 和 D-depth loss。数学很整洁，实现大约 150 行。

## 概念

### Sequential MTP recipe

DeepSeek-V3 在主模型之上添加 `D` 个 MTP modules。每个 module `k`（`k = 1..D`）预测 depth `k` 的 token，也就是在给定截至位置 `i` 的 prefix 时预测 `t_{i+k}`。

Module `k` 包含：

- 一个 transformer block `T_k`，带有自己的 attention 和 MLP。
- 一个 projection matrix `M_k`，用于把 previous-depth hidden state 和 next-depth ground-truth token 的 embedding 组合起来。
- Shared embedding `E`（与主模型相同）。
- Shared output head `Out`（与主模型相同）。

训练时，对于截至位置 `i` 的 prefix，每个 depth 的 hidden state 是：

```
h_i^(0) = main model backbone at position i
h_i^(k) = T_k( M_k * concat(RMSNorm(h_i^(k-1)), RMSNorm(E(t_{i+k}))) )   for k >= 1
```

每个 depth 的 prediction 是：

```
logits_{i+k} = Out(h_i^(k-1))   for k = 1..D
```

每个 depth 的 loss 是针对 ground-truth `t_{i+k}` 的 cross-entropy：

```
L_k = CE(logits_{i+k}, t_{i+k})
```

跨 depths 的 joint loss：

```
L_MTP = (lambda / D) * sum_{k=1..D} L_k
```

`lambda` 是一个小的 weighting factor：DeepSeek-V3 在训练前 10% 使用 0.3，之后使用 0.1。总 training loss 是 `L_main + L_MTP`。

### 为什么是 sequential，而不是 parallel

Gloeckle 最初的 parallel MTP 有 D 个 output heads，每个都直接应用到 `h_i^(0)` 上。每个 head 都从同一个 backbone hidden state 预测 `t_{i+k}`。这样可以训练，但预测之间没有互相条件依赖。你不能用 `head_1` 的输出来帮助 `head_2`，因为这些 heads 是并行发射的。

DeepSeek-V3 的 sequential design 从 `h_i^(k-1)` 加上实际 next-token embedding `E(t_{i+k})` 构建 `h_i^(k)`。这保留了 causal chain：要预测 `t_{i+k+1}`，depth `k+1` 的 module 会看到 `t_{i+k}` 上是什么。这在结构上与 autoregressive decoder 消费自己输出的方式一致，因此 MTP modules 可以直接用作 speculative-decoding drafters。

Inference 时：把 `h_i^(k-1)` 和草拟出的 `t_{i+k}` 输入 module `k+1`，得到 `t_{i+k+1}` 的预测。重复。它本质上就是 EAGLE-style draft，只是使用训练好的 MTP module 作为 draft network。DeepSeek-V3 报告第一个 MTP module 的 acceptance 超过 80%，speedup 约 1.8×。

### Parameter accounting

对 hidden 为 `h`、vocabulary 为 `V` 的模型：

- 主模型：数十亿参数，加上一个大小为 `V * h` 的 output head。
- Shared output head：复用主模型的 head。没有额外参数。
- Shared embedding：复用主模型的 embedding。没有额外参数。
- 每个 MTP module：
  - Projection `M_k`：`(2h) * h = 2h^2`。
  - Transformer block `T_k`：attention（MHA 约 `4h^2`）加 MLP（SwiGLU ratio 8/3 时通常约 `8h^2`）。每个 block 约 `12h^2`。

每个 module 额外总计：`~14h^2`。对 DeepSeek-V3 的 `h = 7168`，D = 1 module：纸面上约 `~14 * 7168^2 = ~720M` 参数。DeepSeek-V3 报告 14B，差异主要来自 MTP module 中 expert layers 也采用 MoE。

### Speculative-decoding payoff

Pre-training 期间，MTP modules 会让训练慢约 10%（更多 forward compute、额外 loss）。回报有两点：

1. 更密集的训练信号。每个 hidden state 看到 D+1 个 supervision targets。DeepSeek-V3 的 ablations 中，在 MMLU、GSM8K、MATH、HumanEval 上都有稳定的几个百分点提升。

2. Inference 时免费得到 speculative decoding draft。MTP module 已经被训练来预测未来几个 tokens。复用为 draft network 后，它能达到 80%+ acceptance rates。在这个水平下，N=3 或 N=5 spec decoding 能带来 1.8× throughput。10% training-time cost 在第一次运行 inference 时就开始回本。

### 与 EAGLE 的关系

EAGLE 在 pre-training 之后单独训练一个小 draft model。MTP 把 draft 烘进 pre-training。两者在 accept rates 上趋近，但 pipeline 不同：

| 维度 | EAGLE-3 | MTP（DeepSeek-V3） |
|-----------|---------|------------------|
| 何时训练 | Post-pre-training | During pre-training |
| 是否兼容已有权重 | 是 | 否（需要重新训练） |
| Draft params | 1-2 transformer layers | 1 transformer block + projection |
| Acceptance rate | 0.88-0.92 | depth 1 时 0.80+ |
| 除速度外的收益 | 只有 speculative decoding | 更密集训练信号 + 加速 |

## 构建

`code/main.py` 端到端构建一个 MTP module：shared embedding、projection、transformer block、shared output head。然后在短合成序列上计算 per-depth cross-entropy loss，并按组件打印 parameter count。Toy vocabulary 只有 32 个 tokens，让数字可读。

### Step 1：shared embedding table

主模型和每个 depth 的每个 MTP module 都使用同一个 `vocab_size x hidden` table。不是第二份拷贝，而是字面意义上的同一个 tensor。

### Step 2：per-depth combination

```python
def combine(prev_hidden, next_token_embed, M_k):
    # concat along feature dim, then project down to hidden
    concat = rms_norm(prev_hidden) + rms_norm(next_token_embed)  # vector addition stand-in
    projected = matvec(M_k, concat)
    return projected
```

真实 DeepSeek-V3 会把两个 RMSNormed vectors concat 成 `[2h]`，再用一个 `h x 2h` matrix 投影。Toy 为了 stdlib 简洁，用 vector addition 代替。

### Step 3：depth k 的 transformer block

Self-attention 加 MLP。在 toy 中，用一个一层 linear attention block 和 SwiGLU MLP 保持结构可见，同时不引入 numpy。

### Step 4：shared output head

复用主模型的 output projection。输出 vocabulary 上的 logits。

### Step 5：per-depth loss

对 softmax(logits) 与 offset `k` 处的 ground-truth token 计算 cross-entropy。用 `lambda / D` scaling factor 跨 depths 聚合。

### Step 6：parameter accounting

打印总参数量、shared（embedding、head）参数量，以及 per-module extra count。展示 MTP extra 与 main-model size 的比例。

## 使用

MTP 已集成进 DeepSeek-V3（2024 年 12 月）和 DeepSeek-R1 系列。Inference 时：

- DeepSeek 自己的 serving stack 开箱即用地把 MTP modules 当作 speculative decoders。
- 截至 2026 年 4 月，vLLM 和 SGLang 已有面向 DeepSeek-V3 MTP 的 integration paths。
- AMD 的 ROCm SGLang tutorial 展示了 V3 checkpoint 上具体的 MTP speculative-decoding config，并测得 1.8× speedup。

什么时候在新的 pre-training run 中使用 MTP：

- 你控制完整 pre-training pipeline，并希望存下更密集的训练信号。
- 你知道模型会大规模 serving，并希望免费获得 speculative decoding。
- Hidden size 至少 4096。在 1B-scale，overhead 的伤害大于收益。

什么时候不要使用：

- Fine-tuning 一个已有 pre-trained dense model。MTP module 没有被训练过。
- 研究模型中你希望有一个干净 baseline 来比较。MTP 会改变架构。

## 交付

本课会产出 `outputs/skill-mtp-planner.md`。给定一个 pre-training run specification（model size、data、compute），它会返回集成 MTP 的计划：depths D 的数量、`lambda` schedule、memory overhead，以及 inference-time speculative-decoding wiring。

## 练习

1. 运行 `code/main.py`。展示当 synthetic signal 变强时，per-depth loss 会单调下降。把 synthetic 改为固定 pattern，并验证 depth-1 和 depth-2 losses 都会收敛。

2. 计算一个 dense 70B model（hidden 8192，80 layers）配 D=1 MTP module 的 parameter overhead。与 DeepSeek-V3 报告的 14B overhead 对比。解释为什么 DeepSeek 的数字更高：MTP transformer block 继承了同样的 MoE 结构，膨胀了 per-module parameter count。

3. 在 toy 中实现 D=2：添加第二个 MTP module，它接收 h^(1) 并预测 `t_{i+2}`。验证 joint loss 和 parameter accounting 匹配 DeepSeek 论文中的 equations 19-21。

4. 把 toy 切换为 parallel MTP（Gloeckle-style）：在 main hidden state 上添加 D 个 output heads，每个预测不同 offset。在同一个 synthetic signal 上衡量每个 depth 的 losses 与 sequential version 的差异。Sequential version 在 k > 1 时应当有更低的 depth-k loss，因为它会 condition on intermediate predictions。

5. 把训练好的 MTP module 当作 EAGLE-style draft 使用：inference 时调用 module k 来提出 `t_{i+k}`。在 held-out sequence 上衡量这些 draft tokens 相对 main model predictions 的 acceptance rate。如果 toy 上达到 50%+，你就复现了 MTP-as-draft 的经验性质。

## 关键术语

| 术语 | 人们怎么说 | 它真正的意思 |
|------|----------------|------------------------|
| MTP module | “Extra loss block” | 一个小 transformer block 加 projection，用于预测主模型前方 `k` 个位置的 token |
| Prediction depth | “Which offset” | 整数 `k`，表示 module `k` 从截至位置 `i` 的 prefix 预测 `t_{i+k}` |
| Parallel MTP | “Gloeckle-style” | 同一个 backbone hidden state 上的 D 个独立 heads，没有 conditional chain |
| Sequential MTP | “DeepSeek-V3 style” | 每个 module 都 condition on 前一个 depth 的 hidden state 加 next token embedding；保留 causal chain |
| Shared output head | “Reuse the main head” | MTP modules 调用主模型的 LM head，而不是单独的 output projection |
| Shared embedding | “Reuse the main table” | 同一个 vocabulary embedding table 到处使用；没有重复参数 |
| Projection matrix M_k | “Combine hidden + next-token” | 一个 `h x 2h` linear layer，把 previous hidden state 和 target-token embedding 折叠成下一 depth 的 input |
| Joint loss L_MTP | “Averaged extra losses” | Per-depth cross-entropy losses 的算术平均，再乘以 `lambda` |
| Acceptance rate at depth 1 | “How often MTP draft is right” | D=1 MTP module 的 top-1 prediction 等于 main model top-1 prediction 的比例；DeepSeek-V3 上超过 80% |
| Lambda weighting | “Extra-loss importance” | Per-depth scaling factor；DeepSeek-V3 训练早期 0.3，之后 0.1 |

## 延伸阅读

- [DeepSeek-AI — DeepSeek-V3 Technical Report (arXiv:2412.19437)](https://arxiv.org/abs/2412.19437) — 完整的 sequential MTP 描述（Section 2.2），包含 joint-loss equations 和 inference 1.8× speedup
- [Gloeckle et al. — Better & Faster Large Language Models via Multi-token Prediction (arXiv:2404.19737)](https://arxiv.org/abs/2404.19737) — DeepSeek 设计改进的 parallel MTP baseline
- [DeepSeek-V3 model card on Hugging Face](https://huggingface.co/deepseek-ai/DeepSeek-V3) — 685B total（671B main + 14B MTP）、deployment notes
- [Leviathan et al. — Fast Inference from Transformers via Speculative Decoding (arXiv:2211.17192)](https://arxiv.org/abs/2211.17192) — MTP 所属的 speculative-decoding framework
- [Li et al. — EAGLE-3 (arXiv:2503.01840)](https://arxiv.org/abs/2503.01840) — EAGLE 的 2025 draft architecture，也是 MTP 的对应竞品
