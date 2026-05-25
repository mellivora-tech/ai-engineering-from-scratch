# Differential Attention（V2）

> Softmax attention 会在每个不匹配 token 上分配少量概率。超过 100k tokens 时，这些噪声会累加并淹没信号。Differential Transformer（Ye et al., ICLR 2025）通过把 attention 计算成两个 softmax 的差来修复它，减去 shared noise floor。DIFF V2（Microsoft，2026 年 1 月）是 production-stack rewrite：decode latency 匹配 baseline Transformer，无 custom kernels，兼容 FlashAttention。本课从 V1 到 V2 端到端讲解，并提供一个可用 stdlib Python 运行的 difference operation toy implementation。

**类型：** 构建
**语言：** Python（stdlib）
**前置要求：** 阶段 7 · 02（self-attention），阶段 7 · 15（attention variants），阶段 10 · 14（architecture walkthrough）
**时间：** ~60 分钟

## 学习目标

- 精确说明为什么 softmax attention 有 noise floor，以及为什么它随 context length 增长。
- 推导 differential attention formula，并解释 subtraction 为什么会抵消 shared noise component，同时保留 signal。
- 梳理 V1-to-V2 diff：什么更快、什么更简单、什么更稳定，以及为什么每个变化对 production pre-training 必要。
- 用纯 Python 从零实现 differential attention，并在 synthetic signal-plus-noise query 上经验验证 noise-cancellation property。

## 问题

标准 softmax attention 有一个数学性质，在规模上会变成运维麻烦。对 query `q`，attention weights 是 `softmax(qK^T / sqrt(d))`。Softmax 永远不会产生精确零；每个 non-matching token 都会得到一些 positive mass。这个 residual mass 是噪声，而且会随 context length 扩大。128k tokens 下，即使每个不匹配 token 只得到 0.001% 概率，127,999 个加起来也贡献约 12% 总量。模型必须学会绕过一个随 context 增长的 noise floor。

经验上，这表现为 attention-head interference：long-context RAG 中 hallucinated citations，100k-token retrieval tasks 上 lost-in-the-middle failures，以及 needle-in-haystack benchmarks 超过 32k 后的细微 accuracy degradation。Differential Transformer paper（arXiv:2410.05258, ICLR 2025）测量了差距：相同大小下，DIFF Transformers 比 baseline 有更低 perplexity、更高 long-context accuracy、更少 hallucinations。

DIFF V1 有三个问题，使它无法进入 frontier pre-training pipelines。decode step 中 value cache 必须加载两次；它需要 custom CUDA kernels，破坏 FlashAttention compatibility；per-head RMSNorm 在 70B+ 规模长训练中不稳定。DIFF V2（Microsoft unilm blog，2026 年 1 月 20 日）修复了三者。本课讲解两个版本，构建 difference operator，并在 toy query 上 benchmark noise cancellation。

## 概念

### softmax 的 noise floor

对 query `q` 和 keys `K = [k_1, ..., k_N]`，attention weights 是：

```
w_i = exp(q . k_i / sqrt(d)) / sum_j exp(q . k_j / sqrt(d))
```

没有任何 `w_i` 会是零。如果 `k_i` 与 `q` 完全无关，score `q . k_i` 也不是 0，而是以 variance `||q||^2 / d` 围绕零波动。softmax normalization 后，每个 unrelated token 仍贡献 `O(1/N)` 到 weighted sum。unrelated tokens 的总贡献是 `O((N-1)/N) = O(1)`，不是小量。

模型想要的更像 hard top-k：匹配 tokens 上高权重，其他地方近零。Softmax 太平滑，无法直接做到。

### differential idea

把每个 head 的 Q 和 K projections 切成两份：Q = (Q_1, Q_2)，K = (K_1, K_2)。计算两个 attention maps：

```
A_1 = softmax(Q_1 K_1^T / sqrt(d))
A_2 = softmax(Q_2 K_2^T / sqrt(d))
```

输出：

```
DiffAttn = (A_1 - lambda * A_2) V
```

subtraction 会抵消两个 maps 共享的任何 noise distribution。如果两个 maps 在 127k unrelated tokens 上都有大致 uniform weight（随机初始化时确实如此），这些会 cancel。信号，也就是少数真正 relevant tokens 上的 peaked weight，只有在两个 maps 上以相同幅度出现时才会 cancel，而模型训练后不会这样。

`lambda` 是每个 head 的 learnable scalar，parameterized as `lambda = exp(lambda_q1 dot lambda_k1) - exp(lambda_q2 dot lambda_k2) + lambda_init`。它可以为负。`lambda_init` 默认为类似 0.8 的小正数。

### 为什么这像 head 内 noise-canceling

想象两个 noisy microphones 录同一个声音。二者都拾取 speaker 加 correlated background noise。相减后，shared noise 降低。voice 会保留，因为两个信号在 phase 或 amplitude 上足够不同，不会完全抵消。per-head `lambda` 学的就是这个平衡。

### V1 vs V2：diff

V1 保持与 baseline Transformer 相同的 parameter count。为了每个 head 得到两个 queries，它把 head dimension 减半。这损失 head expressiveness，更痛的是每个 head 的 value cache 也减半。decode 每步必须加载 value cache 两次（每个 softmax branch 一次）。结果：尽管 parameter count 匹配，decode 比 baseline 慢。

V2 把 query heads 数量翻倍，并保持 KV heads 不变（从 up-projection 借参数）。head dimension 保持与 baseline 相同。subtraction 后，extra dimension 再投影回 baseline Transformer 的 O_W projection 尺寸。三件事同时发生：

1. Decode speed 匹配 baseline（KV cache 只加载一次）。
2. FlashAttention 不变即可运行（无 custom kernel）。
3. Decode 的 arithmetic intensity 提升（每从 HBM 读取一个 byte，有更多 compute）。

V2 也移除了 V1 用于稳定 subtraction 的 per-head RMSNorm。在 70B-class pre-training scales 上，这个 RMSNorm 会 destabilize late training。V2 用更简单的 initialization scheme 替代它，保持训练稳定而不增加模块。

### 何时使用它

| Workload | Benefit |
|----------|---------|
| Long-context RAG (64k+) | 更干净 attention maps，更少 hallucinated citations |
| Needle-in-haystack benchmarks | 32k 以后 accuracy 明显提升 |
| Multi-document QA | 更少 cross-document interference |
| Code completion at 8k | 边际收益，不值得改 architecture |
| Short chat (< 4k) | 与 baseline 基本无差别 |

价值随 context length 增长。4k tokens 时 noise floor 小，standard attention 足够。128k 时它会伤害你。

### 与其他 2026 knobs 如何叠加

| Feature | Compatible with DIFF V2? |
|---------|------------------------|
| GQA | Yes (V2 increases Q heads, not KV heads) |
| MLA (DeepSeek) | Yes in principle, no published paper combining them |
| MoE | Yes (attention is independent of MLP block) |
| RoPE | Yes (unchanged) |
| YaRN / long-context scaling | Yes (exactly where DIFF helps most) |
| FlashAttention | Yes in V2 (was no in V1) |
| Speculative decoding | Yes (attention change is invisible to the spec-decode loop) |

## 构建它

`code/main.py` 用纯 Python 实现 differential attention。一个带已知 signal-plus-noise 结构的 toy query 让你可以直接测量 noise-cancellation ratio。

### 第 1 步：standard softmax attention

stdlib matrix ops：lists of lists、manual matmul、带 max subtraction 数值稳定的 softmax。

```python
def softmax(row):
    m = max(row)
    exps = [math.exp(x - m) for x in row]
    s = sum(exps)
    return [e / s for e in exps]
```

### 第 2 步：把 Q、K 切成两半

V1 风格：减半 head dimension。V2 风格：保持 head dimension，并把 heads 数量翻倍。toy implementation 为教学清晰使用 V1；数学完全相同，只是 bookkeeping 不同。

### 第 3 步：两个 softmax branches + subtraction

```python
A1 = [softmax([dot(q1, k) / scale for k in K1]) for q1 in Q1]
A2 = [softmax([dot(q2, k) / scale for k in K2]) for q2 in Q2]
diff_weights = [[a1 - lam * a2 for a1, a2 in zip(r1, r2)] for r1, r2 in zip(A1, A2)]
out = [[sum(w * v[j] for w, v in zip(row, V)) for j in range(d_v)] for row in diff_weights]
```

注意：output weights 可以为负。这没问题，value cache 仍能处理 signed contributions。后续 V projection 会吸收符号。

### 第 4 步：noise cancellation measurement

构建长度 1024 的 synthetic sequence。把 signal token 放在已知位置，其余填充 noise。计算 (a) signal position 上 standard softmax attention weight，(b) differential attention weight。测量二者 signal-to-noise ratio。DIFF attention 通常产生更高 signal-to-noise ratio，因两个 branches 被训练得有多不同，提升约 3x-10x。

### 第 5 步：V1 vs V2 parameter accounting

给定 config（hidden=4096、heads=32、d_head=128），打印：

- Baseline Transformer：Q、K、V 都是 `hidden * hidden`，MLP 为 4 * hidden。
- DIFF V1：Q、K 都是 `hidden * hidden`，V 是 `hidden * hidden`（不变），内部 head dim 减半。添加 per-head `lambda` parameters（O(heads * d_head)）。
- DIFF V2：Q 是 `2 * hidden * hidden`，K 是 `hidden * hidden`，V 是 `hidden * hidden`。extra dim 在 O_W 前投影回去。添加相同 `lambda` parameters。

toy 会测量 V2 的额外 parameter cost（每个 attention block 大约额外 `hidden * hidden`）并打印。

## 使用它

截至 2026 年 4 月，DIFF V2 尚未在每个 production inference server 中 ship，但 vLLM 和 SGLang 正在集成。同时，这个 pattern 已出现在：

- Microsoft 内部 long-context production models。
- 多个面向 256k+ context 的 open model training runs 的研究复现。
- 把 DIFF attention 与 sliding-window attention 在 alternate layers 结合的 hybrid architectures。

2026 年你会在这些情况下使用它：

- 从零训练一个目标为 64k+ effective context 的新模型。从一开始添加 differential attention；之后 retrain 很贵。
- fine-tune 一个 long-context model，而 lost-in-the-middle failures 主导 eval。Q projections 上的 LoRA 可以近似 DIFF 结构。

不适用情况：

- 你正在 serving 一个已有 pre-trained dense model，且 long-context performance 稳定。对现有 weights 来说，retraining cost 很少能回本。
- context 总是低于 16k。noise floor 可忽略。

## 交付它

本课会产出 `outputs/skill-diff-attention-integrator.md`。给定 model architecture、target context length、hallucination profile 和 training budget，它会给出把 differential attention 加入新 pre-training run 或 LoRA fine-tune 的 integration plan。

## 练习

1. 运行 `code/main.py`。验证 synthetic query 上 differential attention 报告的 signal-to-noise ratio 高于 standard softmax attention。改变 noise amplitude，并展示 standard attention 变得不可用的 crossover point。

2. 对 7B-class 模型（hidden=4096、heads=32、d_head=128、32 layers），计算 baseline 到 DIFF V1，以及 baseline 到 DIFF V2 的 parameter-count delta。展示哪些 components 增加参数，哪些保持不变。

3. 阅读 DIFF V1 paper（arXiv:2410.05258）第 3 节和 DIFF V2 Hugging Face blog 第 2 节。用两句话解释为什么 V1 per-head RMSNorm 是必要的，以及为什么 V2 可以移除它而不造成 training divergence。

4. 实现 ablation：用 `lambda = 0`（纯第一个 softmax）和 `lambda = 1`（完整 subtraction）计算 differential attention。在 synthetic query 上测量 sweep 中 signal-to-noise 如何变化。识别最大化 signal-to-noise 的 `lambda`。

5. 把 toy 扩展到 GQA + DIFF V2。选择 8 KV heads 和 32 Q heads。展示 KV cache size 与相同 (8, 32) 配置的 baseline GQA model 匹配。

## 关键词

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Differential attention | “两个 softmax 相减” | 把 Q、K 切成两半，计算两个 softmax maps，把第二个乘 lambda 后从第一个中减去，再乘 V |
| Noise floor | “softmax 的非零尾巴” | softmax 给每个 unrelated token 的 O(1/N) 权重，在 long contexts 中总和为 O(1) |
| lambda | “subtraction scale” | per-head learnable scalar，parameterized as `exp(lq1.lk1) - exp(lq2.lk2) + lambda_init`；可为负 |
| DIFF V1 | “ICLR 2025 version” | 原始 Differential Transformer；减半 head dim 保持 parameter count，需要 custom kernel，decode 更慢 |
| DIFF V2 | “2026 年 1 月修复版” | 双倍 Q heads、保持 KV heads；匹配 baseline decode speed，并兼容 FlashAttention |
| Per-head RMSNorm | “V1 stabilizer” | V1 在 difference 后应用的额外 norm；V2 移除它以防 late-training instability |
| Signal-to-noise ratio | “attention 浪费多少” | true signal position 上的权重与 unrelated positions 平均权重之比 |
| Lost in the middle | “long-context failure mode” | 长 context 中间位置 documents 的 retrieval accuracy 下滑；DIFF attention 减轻它 |
| Arithmetic intensity | “每加载 byte 的 FLOPs” | V2 通过每次 KV load 翻倍 queries 提高 decode arithmetic intensity；对 memory-bound decode 重要 |

## 延伸阅读

- [Ye et al. — Differential Transformer (arXiv:2410.05258, ICLR 2025)](https://arxiv.org/abs/2410.05258) — 原始论文，包含 noise-cancellation theory 和 long-context ablations
- [Microsoft unilm — Differential Transformer V2 (Hugging Face blog, January 2026)](https://huggingface.co/blog/microsoft/diff-attn-v2) — production-stack rewrite，匹配 baseline decode，兼容 FlashAttention
- [Understanding Differential Transformer Unchains Pretrained Self-Attentions (arXiv:2505.16333)](https://arxiv.org/abs/2505.16333) — 关于 subtraction 为什么恢复 pretrained attention structure 的理论分析
- [Shared DIFF Transformer (arXiv:2501.17900)](https://arxiv.org/html/2501.17900) — parameter-sharing variant
- [Vaswani et al. — Attention Is All You Need (arXiv:1706.03762)](https://arxiv.org/abs/1706.03762) — DIFF 所相减的 baseline Transformer
- [Liu et al. — Lost in the Middle (arXiv:2307.03172)](https://arxiv.org/abs/2307.03172) — DIFF attention 针对的 long-context benchmark
