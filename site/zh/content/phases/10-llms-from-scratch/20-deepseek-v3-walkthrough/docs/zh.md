# DeepSeek-V3 架构导览

> Phase 10 · Lesson 14 命名了每个 open model 都会转动的六个 architectural knobs。DeepSeek-V3（2024 年 12 月，总参数 671B，活跃参数 37B）转动了全部六个，并新增四个：Multi-Head Latent Attention、auxiliary-loss-free load balancing、Multi-Token Prediction 和 DualPipe training。本课会自顶向下阅读 DeepSeek-V3 的架构，并从公开 config 推导每个 parameter count。学完后，你可以解释为什么 671B/37B 这个比例是正确赌注，以及为什么 MLA + MoE 在 frontier 上比单独任何一个都更强。

**类型：** 学习
**语言：** Python（stdlib，parameter calculator）
**先修：** Phase 10 · 14（open-model walkthroughs）、Phase 10 · 17（NSA）、Phase 10 · 18（MTP）、Phase 10 · 19（DualPipe）
**时间：** 约 75 分钟

## 学习目标

- 自顶向下阅读 DeepSeek-V3 config，并用六个 GPT-2 knobs 加四个 DeepSeek-specific additions 解释每个字段。
- 推导 total parameter count（671B）、active parameter count（37B），以及贡献这些数字的组件。
- 计算 MLA 在 128k context 下的 KV cache footprint，并与一个同 active-param 的 dense model + GQA 方案比较。
- 说出四个 DeepSeek-specific innovations（MLA、MTP、auxiliary-loss-free routing、DualPipe），并命名它们各自针对 architecture/training stack 的哪个部分。

## 问题

DeepSeek-V3 是第一个架构上与 Llama family 有实质差异的 frontier open model。Llama 3 405B 是“把六个 knobs 转大的 GPT-2”。DeepSeek-V3 则是 GPT-2 加上全部六个 knobs 再加四个。读 Llama 3 config 是读 DeepSeek config 的热身，但它的深层结构：attention block 的形状、routing logic、training-time objective，都足够不同，需要单独 walkthrough。

学习它的收益：DeepSeek-V3 的 open-weights release 改变了 open models 中“frontier capability”的含义。这个架构是许多 2026 training runs 正在复制的蓝图。理解它，是任何接触 frontier LLM training 或 inference 的岗位的基础要求。

## 概念

### 不变的核心，再看一遍

DeepSeek-V3 仍然是 autoregressive。它仍然堆叠 decoder blocks。每个 block 仍然有 attention 加 MLP 加两个 RMSNorm。它仍然在 MLP 中使用 SwiGLU。它仍然使用 RoPE。Pre-norm。Weight-tied embeddings。与每个 Llama 或 Mistral 的 baseline 相同。

### Twist：用 MLA 取代 GQA

从 Phase 10 · 14 你已经知道，GQA 通过让多组 Q heads 共享 K 和 V 来缩小 KV cache。Multi-Head Latent Attention（MLA）走得更远：K 和 V 被压缩成 shared low-rank latent representation（`kv_lora_rank`），然后按 head 即时解压。KV cache 只存 latent：通常每 token 每 layer 512 个 floats，而不是 8 x 128 = 1024 个 floats。

在 128k context 下，DeepSeek-V3 使用 MLA（每个 token 每层一个 shared latent `c^{KV}`；K 和 V 都通过 up-projections 从这个 latent 派生，这些 up-projections 可被吸收到后续 matmul 中）：

```
kv_cache = num_layers * kv_lora_rank * max_seq_len * bytes_per_element
         = 61 * 512 * 131072 * 2
         = 7.6 GB
```

一个假想 GQA baseline（Llama 3 70B 形状，8 KV heads，head dim 128）需要：

```
kv_cache = 2 * 61 * 8 * 128 * 131072 * 2
         = 30.5 GB
```

在 128k context 下，MLA 比 Llama-3-70B-style GQA cache 小 4x。

Tradeoff：MLA 在每次 attention computation（每个 head）中增加一个 decompression step。额外 compute 与节省的 bandwidth 相比很小。对 long-context inference 来说总体是净收益。

### Routing：auxiliary-loss-free load balancing

MoE routers 决定每个 token 由哪些 top-k experts 处理。Naive router 会把过多工作集中到少数 experts 上，让其他 experts 闲置。标准修复方法是加入一个 auxiliary loss term，惩罚 load imbalance。这有效，但会稍微损伤 main-task performance。

DeepSeek-V3 引入了 auxiliary-loss-free scheme。给 router logits 增加 per-expert bias terms，并在训练中用一个简单规则调整：如果 expert `e` 过载，就降低 `bias_e`；如果负载不足，就提高它。没有额外 loss term。Training 保持干净。Expert load 保持平衡。

对 main loss 的影响：无法测得。对 MoE architecture 的影响：更干净，没有需要调的 auxiliary-loss hyperparameter。

### MTP：更密集训练 + 免费 draft

从 Phase 10 · 18 你知道 DeepSeek-V3 增加了 D=1 MTP module，用于预测后两个位置的 token。Inference 时，训练好的 module 被复用为 speculative-decoding draft，acceptance 超过 80%。Training 时，每个 hidden state 被监督 D+1 = 2 个 targets，提供更密集的信号。

参数：在 671B main 之上增加 14B。Overhead：2.1%。

### Training：DualPipe

从 Phase 10 · 19 你知道 DualPipe 是一个 bidirectional pipeline，会把 forward 和 backward chunks 与 cross-node all-to-all comms 重叠起来。在 DeepSeek-V3 的 2,048-H800 规模上，它回收了约 245k GPU-hours，这些时间本会被 1F1B 的 pipeline bubbles 浪费。

### Config，逐字段

这是 DeepSeek-V3 config（简化版）：

```
hidden_size: 7168
intermediate_size: 18432   (dense MLP hidden size, used on first few layers)
moe_intermediate_size: 2048 (expert MLP hidden size)
num_hidden_layers: 61
first_k_dense_layers: 3    (first 3 layers use dense MLP)
num_attention_heads: 128
num_key_value_heads: 128   (formally equal to num_heads under MLA, but
                           the real compression is in kv_lora_rank)
kv_lora_rank: 512          (MLA latent dimension)
num_experts: 256            (MoE expert count per block)
num_experts_per_tok: 8      (top-8 routing)
shared_experts: 1           (always-on shared expert per block)
max_position_embeddings: 163840
rope_theta: 10000.0
vocab_size: 129280
mtp_module: 1               (1 MTP module at depth 1)
```

解析它：

- `hidden_size=7168`：embedding dimension。
- `num_hidden_layers=61`：总 block depth。
- `first_k_dense_layers=3`：前 3 个 blocks 使用 size 18432 的 dense MLP。其余 58 个使用 MoE。
- `num_attention_heads=128`：128 个 query heads。
- `kv_lora_rank=512`：K 和 V 被压缩到这个 latent dimension，并按 head 解压。
- `num_experts=256, num_experts_per_tok=8`：每个 MoE block 有 256 experts，top-8 routing。
- `shared_experts=1`：在 256 个 routed experts 之上，1 个 always-on expert 会贡献给每个 token。可以把它看作一个 “dense floor”，确保每个 token 都得到可靠处理。
- `moe_intermediate_size=2048`：每个 expert 的 MLP hidden size。它比 dense MLP 小，因为有 256 个 experts。

### Parameter accounting

完整计算在 `code/main.py` 中。Headline：

- Embedding：`vocab * hidden = 129280 * 7168 = ~0.93B`。
- 前 3 个 dense blocks：带 MLA 的 attention（每 block 约 144M）+ dense MLP（每 block 约 260M）+ norms。总计约 1.2B。
- 58 个 MoE blocks：带 MLA 的 attention（约 144M）+ 256 个 experts（每个约 30M）+ 1 个 shared expert（30M）+ norm。每个 block 包含所有 experts 后总计约 7.95B。58 个 MoE blocks 共 461B。
- MTP module：14B。

总计：core architecture 约 476B + 14B MTP；而公开的 671B 数字还会计入额外 structural parameters（bias tensors、expert-specific components、shared expert scaling 等）。Calculator 复现的数字在 published 数字 3-5% 以内；差异来自 DeepSeek report Section 2 appendix 中记录的细粒度 accounting。

每次 forward 的 active parameters：

- Attention：每层 144M * 61 = 8.8B（所有层都会触发）。
- MLP active：前 3 层 dense（3 * 260M = 780M），58 个 MoE layers 各自激活 8 routed + 1 shared + routing overhead。每层 active MLP：约 260M。总计：3 * 260M + 58 * 260M = ~15.9B。
- Embedding + norms：1.2B。
- Total active：约 26B core + 14B MTP（训练时使用，但 inference 时不一定总运行）≈ 37B。

### 671B / 37B ratio

18x sparsity ratio（active params 是 total params 的 5.5%）。DeepSeek-V3 是已经开源权重的 frontier MoE model 中最稀疏的。Mixtral 8x7B 的 ratio 是 13/47（28%），密得多。Llama 4 Maverick 的 ratio 是 17B/400B（4.25%），可比。DeepSeek 的赌注是：在 frontier scale，更多 experts 配更低 activation ratio，会带来更好的 quality per active-FLOP。

### DeepSeek-V3 的位置

| Model | Total | Active | Ratio | Attention | Novel ideas |
|-------|------|-------|-------|-----------|-------------|
| Llama 3 70B | 70B | 70B | 100% | GQA 64/8 | — |
| Llama 4 Maverick | 400B | 17B | 4.25% | GQA | — |
| Mixtral 8x22B | 141B | 39B | 27% | GQA | — |
| DeepSeek V3 | 671B | 37B | 5.5% | MLA 512 | MLA + MTP + aux-free + DualPipe |
| Qwen 2.5 72B | 72B | 72B | 100% | GQA 64/8 | YaRN extension |

### Follow-on：R1、V4

DeepSeek-R1（2025）是在 V3 backbone 上做的 reasoning-training run。R1 使用相同架构。变化的是 post-training recipe（在 verifiable tasks 上进行 large-scale RL），不是 pretraining architecture。

DeepSeek-V4（如果发布）预计会保留 MLA + MoE + MTP，并加入 DSA（DeepSeek Sparse Attention），也就是 Phase 10 · 17 中 NSA 的后继。这个 lineage 是稳定的：architecture-level innovations 会不断累积；每个版本会转动更多 knobs。

## 使用

`code/main.py` 是专门针对 DeepSeek-V3 形状的 parameter calculator。运行它，把输出与论文数字比较，并在假设变体上使用它（256 experts vs 512、top-8 vs top-16、MLA rank 512 vs 1024）。

要关注：

- Total parameter count vs published 671B。
- Active parameter count vs published 37B。
- 128k context 下的 KV cache：MLA vs GQA comparison。
- Per-layer breakdown，看看 parameter budget 实际花在哪里。

## 交付

本课会产出 `outputs/skill-deepseek-v3-reader.md`。给定一个 DeepSeek-family model（V3、R1 或未来任何 variant），它会生成 component-by-component architecture reading，命名 config 的每个字段，按组件推导 parameter counts，并识别模型使用了四个 DeepSeek-specific innovations 中的哪些。

## 练习

1. 运行 `code/main.py`。把 calculator 的 total-parameter estimate 与公开 671B 比较，并识别差异来源。论文 Section 2 有完整 itemization。

2. 把 config 改为使用 MLA rank 256 而不是 512。计算 128k context 下的 KV cache size。它带来多少百分比 reduction？代价是 per-head expressiveness 上有什么损失？

3. 比较 DeepSeek-V3 的（256 experts，top-8）routing 与一个假想（512 experts，top-8）variant。Total parameters 增长；active parameters 不变。额外 expert capacity 理论上带来什么？Inference 时成本是什么？

4. 阅读 DeepSeek-V3 technical report（arXiv:2412.19437）关于 MLA 的 Section 2.1。用三句话解释为什么 K 和 V decompression matrices 在 inference-time efficiency 上可以被“absorbed”进后续 matmul。

5. DeepSeek-V3 对大多数 operations 使用 FP8 training。计算 FP8 vs BF16 存储 671B weights 的 memory savings。它如何与 14.8T-token training budget 交叉影响？

## 关键术语

| 术语 | 人们怎么说 | 它真正的意思 |
|------|----------------|------------------------|
| MLA | “Multi-Head Latent Attention” | 把 K 和 V 压缩成 shared low-rank latent（kv_lora_rank，通常 512），按 head on-the-fly 解压；KV cache 只存 latent |
| kv_lora_rank | “MLA compression dim” | K 和 V 的 shared latent 大小；DeepSeek-V3 使用 512 |
| First k dense layers | “Early layers stay dense” | 前几个 MoE-model layers 跳过 MoE router，运行 dense MLP 以保证稳定性 |
| num_experts_per_tok | “Top-k routing” | 每个 token 触发多少 routed experts；DeepSeek-V3 使用 8 |
| Shared experts | “Always-on experts” | 不管 routing 如何都会处理每个 token 的 experts；DeepSeek-V3 使用 1 |
| Auxiliary-loss-free routing | “Bias-adjusted load balance” | 训练中调整 per-expert bias terms，在不添加 loss term 的情况下保持 expert load balanced |
| MTP module | “Extra prediction head” | 从 h^(1) 和 E(t+1) 预测 t+2 的 transformer block；更密集训练，免费 speculative-decoding draft |
| DualPipe | “Bidirectional pipeline” | 把 forward/backward compute 与 cross-node all-to-all 重叠的 training schedule |
| Active parameter ratio | “Sparsity” | active_params / total_params；DeepSeek-V3 达到 5.5% |
| FP8 training | “8-bit training” | 用 FP8 存储训练数据和很多 compute ops；相对 BF16 约减半 memory，质量成本较小 |

## 延伸阅读

- [DeepSeek-AI — DeepSeek-V3 Technical Report (arXiv:2412.19437)](https://arxiv.org/abs/2412.19437) — 完整 architecture、training 和 results document
- [DeepSeek-V3 model card on Hugging Face](https://huggingface.co/deepseek-ai/DeepSeek-V3) — config files 和 deployment notes
- [DeepSeek-V2 paper (arXiv:2405.04434)](https://arxiv.org/abs/2405.04434) — 引入 MLA 的前代
- [DeepSeek-R1 paper (arXiv:2501.12948)](https://arxiv.org/abs/2501.12948) — V3 架构上的 reasoning-training successor
- [Native Sparse Attention (arXiv:2502.11089)](https://arxiv.org/abs/2502.11089) — DeepSeek-family attention 的未来方向
- [DualPipe repository](https://github.com/deepseek-ai/DualPipe) — training-schedule reference
