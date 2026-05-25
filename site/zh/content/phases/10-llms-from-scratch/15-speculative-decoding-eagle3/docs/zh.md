# Speculative Decoding 与 EAGLE-3

> 阶段 7 第 16 课证明了数学：Leviathan rejection rule 会精确保留 verifier 的分布。本课是 2026 年 production speculative decoding 的 training-stack 视角。EAGLE-3 把 draft model 从廉价近似变成专门训练的小网络，它在 verifier 自己的 hidden states 上训练，然后加入 training-time test loop，让 train 和 inference distributions 对齐。结果是端到端 3× 到 6.5× speedup，chat 上 accepted per-token rates 超过 0.9，且没有 distributional tradeoff。2026 年每个 production inference stack 都默认 ship 它。

**类型：** 构建
**语言：** Python（stdlib）
**前置要求：** 阶段 7 · 16（speculative decoding math），阶段 10 · 12（inference optimization）
**时间：** ~75 分钟

## 学习目标

- 用一句话陈述 Leviathan theorem，并证明 speculative loop 产生的 samples 与 verifier 分布完全相同。
- 梳理 vanilla spec-decoding（Leviathan 2023）到 EAGLE、EAGLE-2、EAGLE-3 的两年演进，并说出每一步移除了什么具体限制。
- 根据 acceptance rate `α` 和 draft-to-verifier cost ratio `c` 计算 expected speedup，并为每个 regime 选择最优 draft length `N`。
- 从零实现完整 speculative loop：draft、verify、从 residual reject-sample、rejection 时回滚 KV cache、full acceptance 时 emit bonus token。

## 问题

70B 模型上的 autoregressive decoding 在 H100 上大概 35 tokens per second。GPU 远未饱和。memory bandwidth 是天花板：每个 token 都从 HBM 加载 70B weights，做一步 arithmetic，然后产出一个 float。compute units 大部分时间闲置。

Speculative decoding 把它变成一个可以解决的 throughput problem。廉价 draft 用 `N` 次小 forward passes 提议 `N` 个 tokens。verifier 在 prefix 加全部 `N` 个 drafts 上运行一次。如果 verifier 在位置 `i` 的分布以我们会精确定义的统计意义同意 draft，就 accept；否则 reject，并从 residual distribution 采样 correction。一次大模型 forward 产生最多 `N+1` 个 accepted tokens，而不是一个。

关键 theorem 是 Leviathan、Kalman、Matias（ICML 2023）：输出分布与直接从 verifier sampling 完全相同。不是近似。完全相同。这是 speculative decoding 能被生产接受的根本原因，它是纯 latency optimization，没有质量 tradeoff。

阶段 7 · 第 16 课给你数学。本课给你 training stack。好的 draft 比廉价 draft 多带来 2× speedup。EAGLE、EAGLE-2 和 EAGLE-3（Li et al., 2024-2025）把 “draft = 同家族小模型” 变成了精确工程学科。2026 年 production inference servers 默认使用 EAGLE-3。

## 概念

### 不变量：Leviathan rejection sampling

令 `p(t)` 是给定 prefix 下 draft 对 next token 的分布，`q(t)` 是 verifier 的分布。采样 draft token `d ~ p`。以 `min(1, q(d) / p(d))` 概率接受。若 reject，则从 residual distribution `(q - p)_+ / ||(q - p)_+||_1` 采样。所得 samples 分布为 `q`。无论 `p` 多差，这都成立；越差只是 reject 越多，但输出仍然精确。

把 `N` 次这种调用接起来，用一次 verifier forward pass 处理 `prefix + d_1 + ... + d_N`。verifier 同时返回 `q_1, q_2, ..., q_{N+1}`。从左到右走。第一次在位置 `j` reject 时，从 `residual(q_j, p_j)` 采样并停止。全部 accept 时，从 `q_{N+1}` 采样一个 bonus token。

### 什么决定 speedup

令 `α` 为每个 drafted token 的 expected acceptance rate。令 `c = cost(draft) / cost(verifier)`。每次 verifier forward 的 expected accepted tokens 是：

```
E[accepted] = (1 - α^(N+1)) / (1 - α)
```

每个 accepted token 的 expected total wall time 是 `(N * c + 1) / E[accepted]`。对 `N` 最小化得到 sweet spot。`α = 0.8, c = 0.05` 时：最优 `N` 约 5-7，speedup 3.2×。`α = 0.95, c = 0.02` 时：最优 `N` 约 8-10，speedup 接近 5×。

单个最大杠杆是 `α`。固定 `N = 5` 时，从 `α = 0.6`（vanilla draft）到 `α = 0.9`（EAGLE-3），每次 verifier forward 的 expected accepted tokens 从 2.2 提升到 4.1。同一个 verifier 获得近 2× more throughput。

### 两年演进

**Vanilla speculative（Leviathan，2023）。** Draft model 是同家族 independently trained smaller LLM。容易接入，`α ≈ 0.6`，speedup 最多约 2×。

**EAGLE-1（Li et al., 2024）。** Draft 是 tiny transformer，通常一两层，输入 verifier 的 last-layer hidden state 并直接预测 next token。因为 draft 看到 verifier 的 feature representation，它的分布更接近 verifier。`α` 提升到 0.7-0.8。

**EAGLE-2（Li et al., 2024）。** 添加 dynamic draft tree：不再提议单条 `N` tokens sequence，而是提议一个小 candidate tree，用一次 verifier forward（tree attention）给每个 candidate 打分，然后走最高概率路径。draft length 变成每步 adaptive。accepted-path token 的 `α` 超过 0.85。

**EAGLE-3（Li et al., 2025, NeurIPS）。** 又做两处变化。第一，完全去掉 feature-prediction loss；EAGLE-1/2 训练 draft 匹配 verifier hidden states，这会限制更多数据的收益。EAGLE-3 直接训练 token prediction。第二，training-time test（TTT）：draft training 期间，把 draft 自己之前的 predictions 像 inference 时一样反馈为 inputs，持续多步。这让 train 和 test distributions 对齐，并阻止 error accumulation。测得 speedup：chat 上最高 6.5×，SGLang 在 H100 batch 64 下 throughput 提升 38%。

### KV cache rollback

verification 会用一次 pass 把 verifier 的 KV cache 扩展 `N` 个 entries。如果 rejection 发生在位置 `j`，位置 `j-1` 之后的 cache contents 都错了。两个常见实现：写入 scratch buffer 并在 acceptance 时 commit（vLLM、TensorRT-LLM），或维护 physical KV cache 加 logical length，reject 时 truncate。不管哪种，rollback cost 都是每层每 head 的 bytes，相比 forward-pass cost 可忽略。

对 EAGLE-2 tree search，verifier 使用尊重 tree topology 的 non-causal mask 运行 attention。工程上有点细，但计算就是带 custom mask 的标准 flash-attention call。

### 2026 年的 Draft architectures

| Strategy | Draft type | `α` | Speedup | Training cost |
|----------|-----------|-----|---------|---------------|
| Vanilla | Separate small LLM | 0.55-0.70 | 1.8-2.3× | None (reuse existing small model) |
| Medusa | Extra LM heads on verifier | 0.65-0.75 | 2-3× | ~1B SFT tokens |
| EAGLE-1 | 1-layer transformer on hidden states | 0.70-0.80 | 2.5-3× | ~60B tokens |
| EAGLE-2 | EAGLE-1 + dynamic draft tree | 0.80-0.88 | 3-4× | ~60B tokens |
| EAGLE-3 | Multi-layer feature fusion + TTT | 0.88-0.92 | 3.5-6.5× | ~60-200B tokens |
| Lookahead | No draft (Jacobi iteration) | N/A | 1.3-1.6× | None |

2026 年生产中：vLLM 和 SGLang 在可用时默认 EAGLE-3，否则 EAGLE-2。TensorRT-LLM 对 Meta 和 NVIDIA public models 有最快的 Medusa path。llama.cpp 为 CPU deployments ship vanilla draft。

## 构建它

见 `code/main.py`。这是完整 Leviathan speculative loop，包含所有部分：draft-of-N、verifier parallel pass、per-position rejection、residual sampling、bonus token、KV rollback，以及经验验证输出分布与直接从 `q` sampling 相同。

### 第 1 步：rejection rule

```python
def accept(q_prob, p_prob, u):
    if p_prob <= 0:
        return True
    return u < min(1.0, q_prob / p_prob)
```

### 第 2 步：residual distribution

```python
def residual(q, p):
    raw = [max(0.0, qi - pi) for qi, pi in zip(q, p)]
    s = sum(raw)
    if s == 0:
        return list(q)
    return [r / s for r in raw]
```

### 第 3 步：完整 speculative step

`spec_step` function 从 `p` draft `N` 个 tokens，然后用一次 parallel `q` evaluation 验证全部。对每个 drafted token 应用 rejection rule，第一次 rejection 时从 residual 采样 correction。如果全部 accept，它从 `q_{N+1}` emit bonus token。

### 第 4 步：KV rollback bookkeeping

simulator 为每个 worker 跟踪 logical `kv_length`。accept `k` 个 drafts 时，`kv_length += k`。在位置 `j` reject 时，cache 已经写过 `j`，但 logical length 被设置为 `prefix_length + j + 1`，也就是 correction token 后一位。后续读取会 truncate 到 logical length。

### 第 5 步：Leviathan check

运行 50,000 个 speculative steps。统计 accepted tokens 的 empirical distribution。与 50,000 个直接从 `q` 采样的 direct samples 比较。chi-square statistic 应显著低于 critical value。theorem 在实践中通过。

### 第 6 步：speedup vs. α

通过以不同 amplitude 扰动 `p` 让它偏离 `q`，扫描 draft quality。测量 `α`，然后打印 expected tokens per verifier call 随 `α` 和 `N` 的表。代码会展示 EAGLE-3-class draft quality（`α ≈ 0.9`）如何解锁每次 verifier call 4-5 个 tokens。

## 使用它

production-level `vllm serve` with EAGLE-3：

```bash
vllm serve meta-llama/Llama-3.3-70B-Instruct \
  --speculative-config '{
    "model": "yuhuili/EAGLE3-LLaMA3.3-Instruct-70B",
    "num_speculative_tokens": 5,
    "method": "eagle3"
  }'
```

SGLang 在 H100 batch 64 下使用 EAGLE-3：根据 EAGLE-3 paper，相比 batch-64 vanilla decoding，throughput 大约提升 1.38×。

何时使用 speculative decoding：

- 任何 p50 latency 比 peak throughput 更重要的 interactive chat workload。
- Code generation 和 structured output（JSON、SQL）。因为 target distribution 高度可预测，`α` 超过 0.9。
- Long-form generation（数千 tokens）。amortized speedup 会持续带来收益。

何时不用：

- 很小的模型（< 3B）。draft 不比 verifier 便宜多少。
- tiny batch-1 CPU deployments。draft model 的 memory overhead 可能不值得。
- very-high-temperature creative sampling，其中 `α` 会崩塌。

## 交付它

本课会产出 `outputs/skill-eagle3-tuner.md`。给定 inference workload（model、batch size、target latency、task profile），它会推荐 speculative-decoding strategy 和 tuning parameters（draft family、`N`、tree depth、temperature-aware switching）。

## 练习

1. 运行 `code/main.py`。确认 Leviathan distribution check 在 50,000 samples 上的 chi-square statistic 低于 95% critical value。

2. 在 `α` 固定 0.9、`c` 固定 0.04 时，把 `N` 从 1 扫到 10。绘制 expected tokens per verifier call 和 actual wall time per token。找到最小化 wall time 的 `N`。解释曲线形状。

3. 修改代码模拟 EAGLE-2 tree search：每步 draft 提议形状 `[2, 2, 2]` 的 tree（八条 candidate paths）。verifier 运行一次，最高概率 accepted path 获胜。计算每个 leaf 的 `α` 和每次 verifier call 的 total tokens。与等价 compute 下的 linear-chain spec-decoding 比较。

4. 为两个 concurrent sequences 实现 batched KV rollback simulator。Sequence A 的所有 drafts 被接受；Sequence B 在位置 2 reject。展示每个 sequence 的 `kv_length` 被正确更新，且没有浪费工作。

5. 阅读 EAGLE-3 paper 第 4 节（Training-Time Test）。用两句话解释为什么没有 TTT 的 naive draft training 会遭遇 exposure bias，以及为什么训练时把 draft 自己的 predictions 喂回去能修复它。把它与 seq2seq 中的 scheduled-sampling literature 联系起来。

## 关键词

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Leviathan rule | “min(1, q over p)” | 以 `min(1, q(d)/p(d))` 概率 Bernoulli accept/reject；rejection 时从 residual 采样，可精确保留 verifier distribution |
| Residual distribution | “(q minus p) plus, normalized” | `(q - p)_+` clamp 到零并重新 normalize，是 rejection 时正确的 sampling distribution |
| Acceptance rate α | “draft 对的频率” | rejection rule 下每个 token 的 expected Bernoulli-success probability，支配所有 speedup math |
| EAGLE-1 | “hidden-state draft” | conditioned on verifier last-layer hidden state 的 tiny transformer draft（Li et al., 2024） |
| EAGLE-2 | “dynamic draft tree” | EAGLE-1 加 candidate continuation tree，用一次 verifier pass 中的 tree attention 打分 |
| EAGLE-3 | “training-time test” | 去掉 feature-prediction loss，用 draft 在训练时喂入自己 outputs 的 direct token prediction |
| Training-time test (TTT) | “exposure bias fix” | 训练时 autoregressively 运行 draft，让 train/test input distributions 匹配；对应 scheduled sampling |
| KV rollback | “撤销 rejected drafts” | rejection 后把 verifier KV cache 重置到 accepted-prefix length 的 bookkeeping |
| Bonus token | “免费的那个” | 当全部 `N` drafts accept 时，从 `q_{N+1}` 额外采样一个 token，不需要额外 verifier cost |
| Tree attention | “一次验证多个 candidates” | 使用尊重 draft tree topology 的 non-causal mask 的 attention，在一次 forward pass 中计算 tree 中每个 node 的 `q_i` |

## 延伸阅读

- [Leviathan, Kalman, Matias — Fast Inference from Transformers via Speculative Decoding (arXiv:2211.17192, ICML 2023)](https://arxiv.org/abs/2211.17192) — foundational paper 和 equivalence theorem
- [Chen et al. — Accelerating Large Language Model Decoding with Speculative Sampling (arXiv:2302.01318)](https://arxiv.org/abs/2302.01318) — concurrent independent introduction，证明清晰
- [Li et al. — EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty (arXiv:2401.15077)](https://arxiv.org/abs/2401.15077) — EAGLE-1，hidden-state-conditioned draft
- [Li et al. — EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees (arXiv:2406.16858)](https://arxiv.org/abs/2406.16858) — dynamic tree search
- [Li et al. — EAGLE-3: Scaling up Inference Acceleration via Training-Time Test (arXiv:2503.01840, NeurIPS 2025)](https://arxiv.org/abs/2503.01840) — 2026 production default
- [Cai et al. — Medusa: Multiple Decoding Heads (arXiv:2401.10774)](https://arxiv.org/abs/2401.10774) — alternative draft-free approach
- [vLLM Speculative Decoding documentation](https://docs.vllm.ai/en/latest/features/spec_decode.html) — 所有 strategies 接好的 canonical production reference
