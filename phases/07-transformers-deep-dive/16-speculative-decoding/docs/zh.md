# Speculative Decoding — Draft, Verify, Repeat

> Autoregressive decoding 是串行的。每个 token 都要等前一个。Speculative decoding 打断这条链：便宜模型 draft N 个 tokens，昂贵模型在一次 forward pass 中验证全部 N 个。当 draft 正确时，你用一次大 forward 换来了 N 次生成。

**类型：** 构建
**语言：** Python
**前置要求：** 阶段 7 · 07（GPT Causal LM），阶段 7 · 12（KV Cache & Flash Attention）
**时间：** ~60 分钟

## 问题

一个 70B LLM 在 H100 上采样一个 token 约需 30 ms。一个 3B draft model 约需 3 ms。如果我们让 3B draft 向前生成 5 个 tokens，然后运行 70B *一次* 验证全部 5 个，总耗时是 `5×3 + 30 = 45 ms`，最多接受 5 个 tokens — 相比直线生成的 `5×30 = 150 ms`。这就是 speculative-decoding 的完整卖点：用少量额外 GPU 内存（draft model）换 2–4× 更低 decode latency。

技巧必须保持分布不变。Leviathan et al.（2023）和 Chen et al. 同期提出的 speculative sampling 保证输出序列与大模型单独产生时 **同分布**。没有质量 tradeoff。只是更快。

2026 推理中占主导的是四类 draft-verifier pairs：

1. **Vanilla speculative（Leviathan 2023）。** 独立 draft model（例如 Llama 3 1B）+ verifier（例如 Llama 3 70B）。
2. **Medusa（Cai 2024）。** Verifier 上多个 decoding heads 并行预测位置 `t+1..t+k`。没有独立 draft model。
3. **EAGLE family（Li 2024, 2025）。** 轻量 draft 复用 verifier 的 hidden states；acceptance rate 比 vanilla 更接近；典型 3–4×。
4. **Lookahead decoding（Fu 2024）。** Jacobi iteration；完全不需要 draft model。Self-speculation。小众但无依赖。

2026 年每个生产推理栈默认提供 speculative decoding。vLLM、TensorRT-LLM、SGLang 和 llama.cpp 至少都支持 vanilla + EAGLE-2。

## 概念

### 核心算法

给定 verifier `M_q` 和更便宜的 draft `M_p`：

1. 令 `x_1..x_k` 为已经 decoded 的 prefix。
2. **Draft**：使用 `M_p` autoregressively 提议 `d_{k+1}, d_{k+2}, ..., d_{k+N}`，draft probabilities 为 `p_1..p_N`。
3. **并行 Verify**：在 `x_1..x_k, d_{k+1}, ..., d_{k+N}` 上运行 `M_q` 一次，得到位置 `k+1..k+N+1` 的 verifier probabilities `q_1..q_{N+1}`。
4. **从左到右 accept/reject 每个 draft token**：对每个 `i`，以概率 `min(1, q_i(d_i) / p_i(d_i))` 接受。
5. 在位置 `j` 第一次 rejection 时：从归一化后的 “residual” 分布 `(q_j - p_j)_+` 采样 `t_j`。`j` 之后的所有 drafts 丢弃。
6. 如果全部 `N` 都接受：从 `q_{N+1}` 采样一个额外 token `t_{N+1}`（免费 bonus token）。

Residual distribution 技巧是保持输出精确等同于 `M_q` 从头采样的数学洞见。

### 什么决定 speedup

令 `α` = 每个 draft token 的期望 acceptance rate。令 `c` = draft-to-verifier cost ratio。每步：

- Naive generation 每个 token 做 1 次 big-model call。
- Speculative 在 `α` 高时，每 `(1 - α^{N+1}) / (1 - α) ≈ 1/(1-α)` 个 tokens 做 1 次 big-model call。

在 `α = 0.75`、`N = 5` 时的典型经验法则：big-model calls 少 3×。Draft cost 便宜 5×。总 wall-clock 降低约 2.5×。

**α 取决于：**

- Draft 对 verifier 的近似程度。同 family / 同 training data 会显著提高 α。
- Decoding strategy。Greedy draft 对 greedy verifier：α 高。Temperature sampling：更难匹配；acceptance 下降。
- 任务类型。Code 和 structured output 接受更多（可预测）；free-form creative writing 接受更少。

### Medusa — 不需要 draft model 的 drafts

Medusa 用 verifier 上的额外 output heads 替换 draft model。在位置 `t`：

```
shared trunk → hidden h_t
    ├── head_0: predict token at t+1  (standard LM head)
    ├── head_1: predict token at t+2
    ├── head_2: predict token at t+3
    ├── head_3: predict token at t+4
```

每个 head 输出自己的 logits。推理时从每个 head 采样得到 candidate sequence，然后用 tree-attention scheme 做一次 forward pass 验证，同时考虑所有 candidate continuations。

优点：没有第二个模型。缺点：增加 trainable parameters；需要一个 supervised fine-tuning 阶段（~1B tokens）；acceptance rate 比带好 draft 的 vanilla speculative 稍低。

### EAGLE — 复用 hidden states 的更好 draft

EAGLE-1/2/3（Li et al., 2024–2025）把 draft model 做成一个 tiny transformer（通常 1 层），输入 verifier 的最后一层 hidden states。因为 draft 看到了 verifier 的 feature representation，它的预测和 verifier 输出分布强相关。Acceptance rates 从 ~0.6（vanilla）上升到 0.85+。

EAGLE-3（2025）加入了对 candidate continuations 的 tree search。vLLM 和 SGLang 将 EAGLE-2/3 作为 Llama 3/4 和 Qwen 3 的默认 spec pathway。

### KV cache dance

Verification 会把 `N` 个 draft tokens 在一次 forward pass 中喂给 verifier。这会让 verifier 的 KV cache 增加 `N` 条。如果有些 drafts 被拒绝，你必须把 cache 回滚到 accepted prefix length。

生产实现（vLLM 的 `--speculative-model`、TensorRT-LLM 的 LookaheadDecoder）用 scratch KV buffers 处理这个问题。先写入，接受时 commit。概念上不难，但细节很多。

## 构建它

见 `code/main.py`。我们实现核心 speculative-sampling 算法（rejection step + residual distribution），包含：

- 一个“big model”：对手写分布做 deterministic-softmax（这样可以分析性验证 acceptance math）。
- 一个“draft model”：big model 的扰动版本。
- 一个 acceptance / rejection loop，产生与 direct sampling 相同的 marginal distribution。

### 第 1 步：rejection step

```python
def accept_or_reject(q_prob, p_prob, draft_token, u):
    ratio = q_prob / p_prob if p_prob > 0 else float("inf")
    return u < min(1.0, ratio)
```

`u` 是一个 uniform random number。`q_prob` 是 verifier 对 drafted token 的概率。`p_prob` 是 draft model 的概率。Leviathan theorem 说明，这个 Bernoulli decision 再加上 rejection 时从 residual 采样，会精确保持 verifier 的分布。

### 第 2 步：residual distribution

```python
def residual_dist(q, p):
    raw = [max(0.0, qi - pi) for qi, pi in zip(q, p)]
    s = sum(raw)
    return [r / s for r in raw]
```

逐元素从 `q` 中减去 `p`，把负值 clamp 到零，重新归一化。在任何 rejection 时从这里采样。

### 第 3 步：一个 speculative step

```python
def spec_step(prefix, q_model, p_model, N, rng):
    drafts = []
    p_probs = []
    ctx = list(prefix)
    for _ in range(N):
        p_dist = p_model(ctx)
        d = sample(p_dist, rng)
        drafts.append(d)
        p_probs.append(p_dist[d])
        ctx.append(d)

    q_dists = [q_model(prefix + drafts[:i]) for i in range(N + 1)]

    for i, d in enumerate(drafts):
        u = rng.random()
        q_prob = q_dists[i][d]
        p_prob = p_probs[i]
        if u < min(1.0, q_prob / p_prob if p_prob > 0 else float("inf")):
            prefix = prefix + [d]
        else:
            res = residual_dist(q_dists[i], p_model(prefix))
            prefix = prefix + [sample(res, rng)]
            return prefix
    prefix = prefix + [sample(q_dists[N], rng)]
    return prefix
```

五个接受 → 一个 bonus → 一次 verifier pass 产生六个 tokens。

### 第 4 步：测量 acceptance rate

在不同 draft-quality levels 下运行 10,000 个 speculative steps。画 acceptance rate vs draft 和 verifier 分布之间的 KL divergence。你应该看到干净的单调关系。

### 第 5 步：验证 distribution equivalence

经验上：speculative loop 产生的 token histogram 应该匹配直接从 verifier 采样的 histogram。这就是 Leviathan theorem 的实践版本。Chi-square test 会在 sampling error 内确认。

## 使用它

生产：

```bash
# vLLM with EAGLE
vllm serve meta-llama/Llama-3.1-70B-Instruct \
    --speculative-model /models/llama-3.1-eagle-70b \
    --speculative-draft-tensor-parallel-size 1 \
    --num-speculative-tokens 5

# vLLM with vanilla draft model
vllm serve meta-llama/Llama-3.1-70B-Instruct \
    --speculative-model meta-llama/Llama-3.2-1B-Instruct \
    --num-speculative-tokens 5
```

截至 2026 年中，TensorRT-LLM 有最快的 Medusa path。`faster-whisper` 用一个小 draft 为 Whisper-large 包装 speculative decoding。

**选择 draft：**

| Strategy | When to pick | Speedup |
|----------|--------------|---------|
| Vanilla draft (1B/3B Llama family) | 快速 prototype，无需训练 | 1.8–2.3× |
| Medusa heads | 你可以 fine-tune verifier | 2–3× |
| EAGLE-2 / 3 | 生产，最大速度 | 3–4× |
| Lookahead | 无 draft、无训练、无额外参数 | 1.3–1.6× |

**什么时候不要 spec-decode：**

- 单序列生成 1–5 个 tokens。Overhead 占主导。
- 非常 creative / high-temperature sampling（α 下降）。
- Memory-constrained deployments（draft model 增加 VRAM）。

## 交付它

见 `outputs/skill-spec-decode-picker.md`。这个 skill 会为新的 inference workload 选择 speculative decoding 策略（vanilla / Medusa / EAGLE / lookahead）和调参（N、draft temperature）。

## 练习

1. **简单。** 运行 `code/main.py`。确认 50,000 tokens 上 speculative token distribution 与 verifier direct-sample distribution 匹配，chi-square p > 0.05。
2. **中等。** 对 `α = 0.5, 0.7, 0.85`，画出 speedup（每次 big-model forward 的 tokens 数）随 `N` 的变化。为每个 α 找到最佳 `N`。（提示：每次 verify call 的期望 tokens = `(1 - α^{N+1}) / (1 - α)`。）
3. **困难。** 实现 tiny Medusa：取第 14 课 capstone GPT，添加 3 个额外 LM heads，预测位置 t+2、t+3、t+4。在 tinyshakespeare 上用 joint multi-head loss 训练。和通过截断同一模型得到的 vanilla draft 比较 acceptance rates。
4. **困难。** 实现 rollback：从一个 10-token prefix KV cache 开始，喂入 5 个 draft tokens，模拟位置 3 的 rejection。验证下一轮读取 cache 时正确匹配 “prefix + first 2 accepted drafts”。

## 关键词

| 术语 | 大家怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| Draft model | “便宜那个” | 提议 candidate tokens 的较小模型；通常比 verifier 便宜 10–50×。 |
| Verifier | “大的那个” | 我们要保留其分布的目标模型；每个 speculative step 运行一次。 |
| Acceptance rate (α) | “draft 对的频率” | Verifier 接受 draft 的 per-token 概率。典型 0.7–0.9。 |
| Residual distribution | “rejection fallback” | 归一化后的 `(q - p)_+`；rejection 时从这里采样可保持 verifier 分布。 |
| Bonus token | “免费的那个” | 当所有 N 个 drafts 都接受时，从 verifier 的 next-step distribution 再采样一个。 |
| Medusa | “无 draft 的 speculative” | Verifier 上多个 LM heads 并行预测位置 t+1..t+k。 |
| EAGLE | “Hidden-state draft” | 以 verifier 最后一层 hidden states 为条件的 tiny transformer draft。 |
| Lookahead decoding | “Jacobi iteration” | 使用 fixed-point iteration 的 self-speculation；没有 draft model。 |
| Tree attention | “一次验证许多 candidates” | 同时考虑多个 draft continuations 的 branching verification。 |
| KV rollback | “撤销 rejected drafts” | Scratch KV buffer；接受时 commit，拒绝时 discard。 |

## 延伸阅读

- [Leviathan, Kalman, Matias (2023). Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) — 核心算法和 equivalence theorem。
- [Chen et al. (2023). Accelerating Large Language Model Decoding with Speculative Sampling](https://arxiv.org/abs/2302.01318) — 同期提出；干净的 Bernoulli-rejection proof。
- [Cai et al. (2024). Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads](https://arxiv.org/abs/2401.10774) — Medusa 论文；tree-attention verification。
- [Li et al. (2024). EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty](https://arxiv.org/abs/2401.15077) — EAGLE-1；hidden-state-conditioned draft。
- [Li et al. (2024). EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees](https://arxiv.org/abs/2406.16858) — EAGLE-2；dynamic tree depth。
- [Li et al. (2025). EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test](https://arxiv.org/abs/2503.01840) — EAGLE-3。
- [Fu et al. (2024). Break the Sequential Dependency of LLM Inference Using Lookahead Decoding](https://arxiv.org/abs/2402.02057) — lookahead，无 draft 方法。
- [vLLM docs — Speculative Decoding](https://docs.vllm.ai/en/latest/features/spec_decode.html) — 接入全部四种策略的标准生产参考。
- [SafeAILab / EAGLE reference implementation](https://github.com/SafeAILab/EAGLE) — EAGLE-1/2/3 的参考代码。
