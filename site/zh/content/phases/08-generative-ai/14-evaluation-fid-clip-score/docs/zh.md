# Evaluation — FID、CLIP Score、Human Preference

> 每个生成模型 leaderboard 都会引用 FID、CLIP score 和 human-preference arena 的 win rate。每个数字都有一个坚定研究者可以 exploit 的 failure mode。如果你不知道这些 failure modes，就分不清真实改进和 gaming run。

**类型：** 构建
**语言：** Python
**前置要求：** 阶段 8 · 01（Taxonomy），阶段 2 · 04（Evaluation Metrics）
**时间：** ~45 分钟

## 问题

生成模型按 *sample quality* 和 *conditioning adherence* 评判。二者都没有 closed-form measure。你的模型必须渲染 10,000 张图；某种东西必须给它们打分；你还必须相信这些数字能跨 model families、resolutions、architectures 比较。2014–2026 的考验后，三种指标存活下来：

- **FID（Fréchet Inception Distance）。** 在 Inception network feature space 中，真实分布和生成分布之间的距离。越低越好。
- **CLIP score。** 生成图像的 CLIP-image embedding 与 prompt 的 CLIP-text embedding 之间的 cosine similarity。越高越好。衡量 prompt adherence。
- **Human preference。** 用同一 prompt 让两个模型 head-to-head，让人类（或 GPT-4-class model）选择更好的，聚合成 Elo score。

你还会看到：IS（inception score，基本退役）、KID、CMMD、ImageReward、PickScore、HPSv2、MJHQ-30k。每个都修正前一个指标的某个失败点。

## 概念

![FID, CLIP, and preference: three axes, different failure modes](../assets/evaluation.svg)

### FID — sample quality

Heusel et al.（2017）。步骤：

1. 为 N 张真实图像和 N 张生成图像提取 Inception-v3 features（2048-D）。
2. 对每个池拟合 Gaussian：计算 mean `μ_r, μ_g` 和 covariance `Σ_r, Σ_g`。
3. FID = `||μ_r - μ_g||² + Tr(Σ_r + Σ_g - 2 · (Σ_r · Σ_g)^0.5)`。

解释：feature space 中两个 multivariate Gaussians 的 Fréchet distance。越低 = 分布越相似。

Failure modes：
- **小 N 有偏。** FID 对 feature distribution 做 mean-squared — 小 N 会低估 covariance，给出虚假的低 FID。总是使用 N ≥ 10,000。
- **依赖 Inception。** Inception-v3 在 ImageNet 上训练。远离 ImageNet 的 domains（faces、art、text images）会产生无意义 FID。使用 domain-specific feature extractor。
- **Gaming。** 过拟合 Inception prior 会降低 FID，但不改善视觉质量。用 CMMD（见下）对抗。

### CLIP score — prompt adherence

Radford et al.（2021）。对 generated image + prompt：

```
clip_score = cos_sim( CLIP_image(x_gen), CLIP_text(prompt) )
```

对 30k generated images 求平均 → 一个可跨模型比较的 scalar。

Failure modes：
- **CLIP 自己的 blind spots。** CLIP compositional reasoning 很弱（“a red cube on a blue sphere” 经常失败）。模型可以在 CLIP score 上排名很好，却没有真正遵循复杂 prompt。
- **短 prompt 偏置。** 短 prompts 在真实世界中有更多 CLIP-image matches。长 prompts 机械地 CLIP scores 更低。
- **Prompt gaming。** 在 prompt 中加入 “high quality, 4k, masterpiece” 会抬高 CLIP score，却不改善 image-text binding。

CMMD（Jayasumana et al., 2024）修复部分问题：使用 CLIP features 而不是 Inception，用 maximum-mean discrepancy 而不是 Fréchet。更擅长检测细微质量差异。

### Human preference — ground truth

选择一组 prompts。用模型 A 和模型 B 生成。把 pairs 展示给人类（或强 LLM judge）。把 wins 聚合成 Elo 或 Bradley-Terry score。Benchmarks：

- **PartiPrompts（Google）**：1,600 个多样 prompts，12 个类别。
- **HPSv2**：107k human annotations，广泛用作 automated proxy。
- **ImageReward**：137k prompt-image preference pairs，MIT-licensed。
- **PickScore**：在 Pick-a-Pic 2.6M preferences 上训练。
- **Chatbot-Arena-style image arenas**：https://imagearena.ai/ 等。

Failure modes：
- **Judge variance。** 非专家与专家偏好不同。两者都用。
- **Prompt distribution。** 精挑 prompts 会偏向某个 family。必须记录。
- **LLM-judge reward hacking。** GPT-4-judge 会被漂亮但错误的输出欺骗。用人类交叉验证。

## 组合使用

生产 eval report 应包含：

1. 在 10-30k samples 上，对 held-out real distribution 做 FID（sample quality）。
2. 在同一批 samples 上，对 prompts 做 CLIP score / CMMD（adherence）。
3. 在 blinded arena 中与 previous model 比 win rate（overall preference）。
4. Failure mode analysis：随机抽 50 个 outputs，标注已知问题（hand anatomy、text rendering、consistent object count）。

任何单个 metric 都是谎言。三个互相印证的 metrics + qualitative review 才是 claim。

## 构建它

`code/main.py` 在合成 “feature vectors” 上实现 FID、CLIP-score-like 和 Elo aggregation（我们用 4-D vectors 代替 Inception features）。你会看到：

- 小 N 和大 N 下的 FID 计算 — bias。
- “CLIP score” 作为 feature pools 之间的 cosine similarity。
- 来自 synthetic preference stream 的 Elo update rule。

### 第 1 步：四行 FID

```python
def fid(real_features, gen_features):
    mu_r, cov_r = mean_and_cov(real_features)
    mu_g, cov_g = mean_and_cov(gen_features)
    mean_diff = sum((a - b) ** 2 for a, b in zip(mu_r, mu_g))
    trace_term = trace(cov_r) + trace(cov_g) - 2 * sqrt_cov_product(cov_r, cov_g)
    return mean_diff + trace_term
```

### 第 2 步：CLIP-style cosine-similarity

```python
def clip_like(image_feat, text_feat):
    dot = sum(a * b for a, b in zip(image_feat, text_feat))
    norm = math.sqrt(dot_self(image_feat) * dot_self(text_feat))
    return dot / max(norm, 1e-8)
```

### 第 3 步：Elo aggregation

```python
def elo_update(r_a, r_b, winner, k=32):
    expected_a = 1 / (1 + 10 ** ((r_b - r_a) / 400))
    actual_a = 1.0 if winner == "a" else 0.0
    r_a_new = r_a + k * (actual_a - expected_a)
    r_b_new = r_b - k * (actual_a - expected_a)
    return r_a_new, r_b_new
```

## 陷阱

- **N=1000 的 FID。** N<10k 时 heuristic 不可靠。报告低 N FID 的论文是在 gaming。
- **跨分辨率比较 FID。** Inception 的 299×299 resize 会改变 feature distribution。只在 matched resolution 下比较。
- **只报告一个 seed。** 至少运行 3 seeds。报告 std。
- **通过 negative prompts 抬高 CLIP score。** 有些 pipelines 通过过拟合 prompt 提升 CLIP。检查 visual saturation。
- **Prompt overlap 导致 Elo bias。** 如果两个模型都在训练中见过 benchmark prompt，Elo 无意义。使用 held-out prompt sets。
- **Human eval paid-crowd skew。** Prolific、MTurk annotators 偏年轻 / tech-friendly。混入招募的 art/design experts。

## 使用它

2026 年生产 eval protocol：

| 支柱 | 最低要求 | 推荐 |
|--------|---------|-------------|
| Sample quality | 10k vs held-out real 上的 FID | + 5k 上 CMMD + 每类别 subset FID |
| Prompt adherence | 30k 上 CLIP score | + HPSv2 + ImageReward + VQA-style question answering |
| Preference | 200 blinded pairs vs baseline | + 2000 paired human + LLM-judge + Chatbot Arena |
| Failure analysis | 50 个手工标注 | 500 个手工标注 + automated safety classifier |

四个支柱都在一份 report 中 = claim。任何单个支柱 = marketing。

## 交付它

保存 `outputs/skill-eval-report.md`。Skill 接收新模型 checkpoint + baseline，并输出完整 eval plan：sample sizes、metrics、failure-mode probes、sign-off criteria。

## 练习

1. **简单。** 运行 `code/main.py`。在同一 synthetic distributions 上比较 N=100 vs N=1000 的 FID。报告 bias magnitude。
2. **中等。** 从 synthetic CLIP-style features 实现 CMMD（公式见 Jayasumana et al., 2024）。比较它相对 FID 对质量差异的敏感度。
3. **困难。** 复现 HPSv2 setup：从 Pick-a-Pic 子集取 1000 个 image-prompt pairs，在 preferences 上 fine-tune 一个小 CLIP-based scorer，并测量它与 held-out set 的一致性。

## 关键词

| 术语 | 大家怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| FID | “Fréchet Inception Distance” | 对 real vs gen Inception features 的 Gaussian fits 计算 Fréchet distance。 |
| CLIP score | “Text-image similarity” | CLIP image 和 text embeddings 之间的 cosine similarity。 |
| CMMD | “FID 的替代者” | CLIP-feature MMD；偏差更小，无 Gaussian assumption。 |
| IS | “Inception score” | Exp KL(p(y|x) || p(y))；在现代模型上相关性差，已退役。 |
| HPSv2 / ImageReward / PickScore | “Learned preference proxies” | 在 human preferences 上训练的小模型，用作 automatic judges。 |
| Elo | “Chess rating” | Pairwise wins 的 Bradley-Terry aggregation。 |
| PartiPrompts | “Benchmark prompt set” | Google curated 的 1,600 prompts，跨 12 类。 |
| FD-DINO | “Self-sup replacement” | 使用 DINOv2 features 的 FD；更适合 out-of-ImageNet domains。 |

## 生产备注：evaluation 也是 inference workload

在 10k samples 上跑 FID 意味着生成 10k 张图。对于单张 L4 上 1024² 的 50-step SDXL base，这是约 11 小时的 single-request inference。Evaluation budgets 很真实，框架正是 offline-inference 场景（最大化 throughput，忽略 TTFT）：

- **用力 batch，忘掉 latency。** Offline eval = 使用内存能容纳的最大 static batch。80GB H100 上 `pipe(...).images` 配合 `num_images_per_prompt=8`，wall-clock 比 single-request 快 4-6×。
- **缓存 real features。** 对真实 reference set 做 Inception（FID）或 CLIP（CLIP-score、CMMD）feature extraction 只运行 *一次*，存成 `.npz`。不要每次 eval 重算。

对于 CI / regression gates：每个 PR 在 500-sample subset 上跑 FID + CLIP score（~30 min）；nightly 跑完整 10k FID + HPSv2 + Elo。

## 延伸阅读

- [Heusel et al. (2017). GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium (FID)](https://arxiv.org/abs/1706.08500) — FID 论文。
- [Jayasumana et al. (2024). Rethinking FID: Towards a Better Evaluation Metric for Image Generation (CMMD)](https://arxiv.org/abs/2401.09603) — CMMD。
- [Radford et al. (2021). Learning Transferable Visual Models from Natural Language Supervision (CLIP)](https://arxiv.org/abs/2103.00020) — CLIP。
- [Wu et al. (2023). HPSv2: A Comprehensive Human Preference Score](https://arxiv.org/abs/2306.09341) — HPSv2。
- [Xu et al. (2023). ImageReward: Learning and Evaluating Human Preferences for Text-to-Image Generation](https://arxiv.org/abs/2304.05977) — ImageReward。
- [Yu et al. (2023). Scaling Autoregressive Models for Content-Rich Text-to-Image Generation (Parti + PartiPrompts)](https://arxiv.org/abs/2206.10789) — PartiPrompts。
- [Stein et al. (2023). Exposing flaws of generative model evaluation metrics](https://arxiv.org/abs/2306.04675) — failure-mode survey。
