# Speculative Decoding 和 EAGLE

> 一个 frontier LLM 生成一个 token，需要对数十亿参数做一次完整 forward pass。这个 forward pass 被严重 over-provisioned：大多数时候，一个小得多的模型可以正确猜出接下来的 3-5 个 tokens，大模型只需要 *verify* 这个猜测。猜对时，你用一次的价格得到了 5 个 tokens。Speculative decoding（Leviathan et al. 2023）让这件事变得精确，而 EAGLE-3（2025）把 acceptance rates 推到每次 verify 约 4.5 tokens：在匹配输出分布的情况下带来 4-5x speedup。

**类型：** 构建
**语言：** Python（with numpy）
**先修：** Phase 10 Lesson 12（Inference Optimization）、Phase 10 Lesson 04（Pre-training Mini-GPT）
**时间：** 约 75 分钟

## 问题

70B-class model 在 H100 上的 decode throughput 通常是 40-80 tokens/second。每个 token 都需要一次完整 forward pass，从 HBM 读取所有 model weights。不能在不改变输出的情况下缩小模型。也不能无限增加 batch size，因为 memory 会卡住。你被困住了，除非能让模型每次 forward pass 输出不止一个 token。

Autoregressive generation 看起来天然是 serial：`x_{t+1} = sample(p(· | x_{1:t}))`。但这里有一个 concurrency opportunity。如果你有一个 cheap predictor 说“接下来 4 个 tokens 可能是 [a, b, c, d]”，就可以在 **大模型的一次 forward pass** 中验证所有 5 个位置，并接受最长匹配 prefix。

Leviathan、Kalai、Matias（2023，“Fast Inference from Transformers via Speculative Decoding”）通过一个巧妙的 accept/reject rule 做到了精确，并保持 target model 的 sampling distribution 不变。同样的输出分布，快 2-4×。

## 概念

### Two-Model Setup

- **Target model** `M_p`：你真正想从中采样的大而慢、高质量模型。Distribution：`p(x)`。
- **Draft model** `M_q`：小而快、质量较低的模型。Distribution：`q(x)`。小 5-30×。

每一步：

1. Draft model 自回归提出 `K` 个 tokens：`x_1, x_2, ..., x_K ~ q`。
2. Target model 对全部 `K+1` 个位置并行运行一次 forward pass，为每个 proposed token 产生 `p(x_k)`。
3. 通过下面的 modified rejection-sampling rule 从左到右 accept/reject 每个 token。接受最长匹配 prefix。
4. 如果任何 token 被拒绝，就从 corrected distribution 采样 replacement 并停止。否则从 `p(· | x_1...x_K)` 采样一个 bonus token。

如果 draft 完美匹配 target，你每次 target-forward 会得到 K+1 tokens。如果 draft 在第 1 个位置就错了，你只得到 1 个 token。

### Exactness Rule

Speculative decoding **在分布上可证明等价于从 p 采样**。Rejection rule：

```
For each drafted token x_t:
    r ~ Uniform(0, 1)
    if r < p(x_t) / q(x_t):
        accept x_t
    else:
        sample replacement from residual: (p - q)+ / ||(p - q)+||_1
        stop
```

其中 `(p - q)+` 表示 pointwise difference 的 positive part。当 draft 和 target 同意（`p ≈ q`）时，acceptance 接近 1。当它们不同意时，residual distribution 会被构造出来，确保 overall sample 仍然精确为 `p`。

**Greedy case。** 对 temperature=0 sampling，只需要检查 `argmax(p) == x_t`。如果是，就 accept；否则输出 `argmax(p)` 并停止。

### Expected Speedup

如果 draft model 的 token-level acceptance rate 是 `α`，每次 target-forward pass 产生的 expected tokens 是：

```
E[tokens] = (1 - α^{K+1}) / (1 - α)        # K = draft length, α in [0, 1]
```

当 `α = 0.8, K = 4`：`(1 - 0.8^5)/(1 - 0.8) = 3.36` tokens per forward。一次 target forward 的成本大约是 `cost_q * K + cost_p`（K 个 draft steps 加一次 target verify）。如果 `cost_p >> cost_q * K`，throughput speedup ratio 就是 `3.36× / 1 = 3.36×`。

唯一真正的参数是 `α`，它完全取决于 draft-target alignment。好的 draft 就是一切。

### 训练 Draft：Distillation

随机小模型是很差的 draft。标准 recipe 是从 target distill：

1. 选择一个小架构（70B target 配 ~1B，7B target 配 ~500M）。
2. 在大文本语料上运行 target model，存储它的 next-token distributions。
3. 用 KL divergence 训练 draft，使其匹配 target distribution（不是 ground-truth tokens）。

结果：coding 上 `α` 通常 0.6-0.8，natural-language chat 上 0.7-0.85。Production 中 speedups 2-3×。

### EAGLE：Tree Drafting + Feature Reuse

Li、Wei、Zhang、Zhang（2024，“EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty”）观察到标准 speculative decoding 中的两个低效点：

1. Draft 做 K 个 serial steps，每一步都是 full-stack。但 draft 可以复用最近一次 verify 时 target 的 features（hidden states）：target 已经计算了丰富表示，draft 正在从零重复推导它们。
2. Draft 输出一条 linear chain。如果 draft 可以输出一个 candidates 的 *tree*（每个 node 有多个 guesses），target 的一次 forward pass 就能通过 tree attention mask 并行验证多个 candidate paths，并选择最长 accepted branch。

EAGLE-1 的变化：
- Draft input = target 在位置 t 的 final hidden state，而不是 raw tokens。
- Draft architecture = 1 transformer decoder layer（不是单独小模型）。
- Output = 深度 4-6、每层 K = 4-8 candidates 的 tree。

EAGLE-2（2024）加入 dynamic tree topology：draft 不确定的位置 tree 更宽，自信的位置保持窄。在不增加 verify cost 的情况下提高 `α_effective`。

EAGLE-3（Li et al. 2025，“EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test”）去掉了固定 top-layer feature dependency，并用新的 “test-time simulation” loss 训练 draft：draft 训练时匹配 target 的 test-time distribution 输出，而不是 teacher-forced training distribution。Acceptance rate 从 0.75（EAGLE-2）升到 0.82（EAGLE-3），mean tokens/verify 从 3.0 升到 4.5。

### Tree Attention Verification

当 draft 输出 tree 时，target model 使用 **tree attention mask** 在一次 forward pass 中验证它：这是一个编码 tree topology 而不是纯线性的 causal mask。每个 token 只关注它在 tree 中的 ancestors。Verify pass 仍然是一次 forward、一次 matmul；topological mask 只多花少量 KV entries。

```
        root
       /    \
      a      b
     / \    / \
    c  d   e   f
```

如果 `a, b` 是竞争的 first-token candidates，`c, d, e, f` 是 second-token candidates，所有六个位置都能在一次 forward pass 中验证。输出是任意 accepted path 上最长的 prefix。

### 什么时候赢，什么时候不赢

**赢：**
- Chat / completion 中可预测的文本（code、common English、structured output）。`α` 高。
- Decode 阶段 GPU compute 未充分使用的设置（memory-bound phase）。Tree drafting 会使用可用 FLOPs。

**输 / 没收益：**
- 高随机输出（高 temperature creative writing）。`α` 会降向 `1/|vocab|`。
- 非常高 concurrency 的 batch serving：batching 已经填满 FLOPs，tree verification 的空间很小。
- 很小的 target models，此时 draft 没有小很多。

Production shops 通常报告 chat 上 2-3× wall-clock speedup，code generation 上 3-5×，creative writing 上接近零。

## 构建

`code/main.py`：

- 一个 reference `speculative_decode(target, draft, prompt, K, temperature)`，实现 exact rejection rule，并验证它保持 target distribution（与 plain target sampling 的 empirical KL < 0.01）。
- 一个 EAGLE-style tree drafter，用 top-p branching 构建 depth-K tree。
- 一个 tree attention mask builder，生成 verifier 需要的正确 causal pattern。
- 一个 acceptance-rate harness，在 tiny LM 上运行二者（从 GPT-2-medium target distill 一个 GPT-2-small）。

```python
def speculative_step(p_target, q_draft, K, temperature=1.0):
    """One round of speculative decoding. Returns list of accepted tokens."""
    # 1. Draft K tokens
    draft_tokens = []
    q_probs = []
    state = draft_state_init()
    for _ in range(K):
        probs = softmax(q_draft(state) / temperature)
        t = np.random.choice(len(probs), p=probs)
        draft_tokens.append(t)
        q_probs.append(probs[t])
        state = draft_step(state, t)

    # 2. Target computes p at every drafted position + 1 extra
    p_probs_all = target_forward_batched(p_target, draft_tokens, temperature)

    # 3. Accept/reject left-to-right
    accepted = []
    for k, tok in enumerate(draft_tokens):
        r = np.random.uniform()
        if r < p_probs_all[k][tok] / q_probs[k]:
            accepted.append(tok)
        else:
            residual = np.maximum(p_probs_all[k] - q_probs[k], 0)
            residual /= residual.sum()
            accepted.append(np.random.choice(len(residual), p=residual))
            return accepted
    # 4. All K accepted → sample bonus token from target
    accepted.append(np.random.choice(len(p_probs_all[-1]), p=p_probs_all[-1]))
    return accepted
```

## 使用

- **vLLM** 和 **SGLang** 提供 first-class speculative decoding。Flags：`--speculative_model`、`--num_speculative_tokens`。EAGLE-2/3 支持通过 `--spec_decoding_algorithm eagle` flag。
- **NVIDIA TensorRT-LLM** 原生支持 Medusa 和 EAGLE trees。
- **Reference draft models**：`Qwen/Qwen3-0.6B-spec`（用于 Qwen3-32B 的 draft）、`meta-llama/Llama-3.2-1B-Instruct-spec`（用于 70B 的 draft）。
- **Medusa heads**（Cai et al. 2024，“Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads”）：不使用 draft model，而是在 target 自身上添加 K 个 parallel prediction heads。部署更简单，acceptance 略低于 EAGLE。

## 交付

本课会产出 `outputs/skill-speculative-tuning.md`：一个 skill，用于 profile target model 的 workload，并选择 draft model、K（draft length）、tree width、temperature，以及何时 fallback 到 plain decode。

## 练习

1. 实现 exact rejection rule 并做 empirical verification。通过 `speculative_decode` 和 plain target sampling 各运行 10K samples；计算两个 output distributions 的 TV distance。应当 < 0.01。

2. 计算 speedup formula。给定固定 `α` 和 `K`，绘制每次 target-forward 的 expected tokens。找出 α ∈ {0.5, 0.7, 0.9} 时的 optimal K。

3. 训练一个 tiny draft。取 124M GPT-2 target，并在 100M tokens 上用 KL loss distill 一个 30M GPT-2 draft。测量 held-out text 上的 `α`。预期：0.6-0.7。

4. 实现 EAGLE-style tree drafting。不要让 draft 输出 chain，而是在每个 depth 输出 top-3 branches。构建 tree attention mask。验证 target 接受最长 correct branch。

5. 衡量 failure modes。在 temperature=1.5（高 stochasticity）下运行 speculative decode。展示 α collapses，并且由于 draft overhead，算法比 plain decode 更慢。

## 关键术语

| 术语 | 人们怎么说 | 它真正的意思 |
|------|-----------------|------------------------|
| Target model | “The big model” | 你想从中采样的慢而高质量模型（p distribution） |
| Draft model | “The speculator” | 小而快的 predictor（q distribution）；小 5-30x |
| K / draft length | “Look-ahead” | 每次 verify pass 的 speculated tokens 数量 |
| α / acceptance rate | “Hit rate” | Draft proposal 被接受的 per-token probability |
| Exact rejection rule | “The accept test” | 保持 target distribution 的 r < p/q 比较 |
| Residual distribution | “Corrected p-q” | (p - q)+ / ||(p - q)+||_1，rejection 时要采样的 distribution |
| Tree drafting | “Branching speculation” | Draft 输出 candidates tree，用 tree-structured attention mask 一次验证 |
| Tree attention mask | “Topological mask” | 编码 tree topology 的 causal mask，让每个 node 只关注 ancestors |
| Medusa heads | “Parallel heads” | Target 自身上的 K 个额外 prediction heads；不需要单独 draft model |
| EAGLE feature reuse | “Hidden-state draft” | Draft input 是 target 的 last hidden state，而不是 raw tokens，从而缩小 draft |
| Test-time simulation loss | “EAGLE-3 training” | 让 draft 训练时匹配 target 的 test-time distribution，而不是 teacher forcing |

## 延伸阅读

- [Leviathan, Kalai, Matias, 2023 — "Fast Inference from Transformers via Speculative Decoding"](https://arxiv.org/abs/2211.17192) — exact rejection rule 和 theoretical speedup analysis
- [Chen, Borgeaud, Irving et al., 2023 — "Accelerating Large Language Model Decoding with Speculative Sampling"](https://arxiv.org/abs/2302.01318) — DeepMind 的同期 speculative-sampling paper
- [Cai, Li, Geng, Wang, Wang, Zhu, Dao, 2024 — "Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads"](https://arxiv.org/abs/2401.10774) — draft model 的 parallel-heads alternative
- [Li, Wei, Zhang, Zhang, 2024 — "EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty"](https://arxiv.org/abs/2401.15077) — feature reuse 和 tree drafting
- [Li et al., 2024 — "EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees"](https://arxiv.org/abs/2406.16858) — dynamic tree topology
- [Li et al., 2025 — "EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test"](https://arxiv.org/abs/2503.01840) — train-time test-time matching
- [Fu, Haotian, Peng et al., 2024 — "Break the Sequential Dependency of LLM Inference Using Lookahead Decoding"](https://arxiv.org/abs/2402.02057) — Jacobi/lookahead decoding，一个 speculator-free alternative
