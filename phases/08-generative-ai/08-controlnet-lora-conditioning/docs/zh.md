# ControlNet、LoRA 与 Conditioning

> 只有文本是笨拙的控制信号。ControlNet 让你克隆一个 pretrained diffusion model，并用 depth map、pose skeleton、scribble 或 edge image 驱动它。LoRA 让你通过训练 1000 万参数来 fine-tune 一个 2B 参数模型。两者一起把 Stable Diffusion 从玩具变成了 2026 年每个 agency 都在上线的 image pipeline。

**类型：** 构建
**语言：** Python
**前置要求：** 阶段 8 · 07（Latent Diffusion），阶段 10（LLMs from Scratch — LoRA 基础）
**时间：** ~75 分钟

## 问题

像 “a woman in a red dress walking a dog on a busy street” 这样的 prompt，并没有告诉模型狗在 *哪里*、女人是 *什么姿势*、街道的 *视角* 如何。文本大约只能钉住你需要指定图像信息的 10%。剩下是视觉信息，无法用文字高效描述。

为每一种信号（pose、depth、canny、segmentation）从零训练新的 conditional model 成本太高。你想保持 2.6B-param SDXL backbone 冻结，接上一个读取 conditioning 的小 side-network，让它轻推 backbone 的中间特征。这就是 ControlNet。

你还想教会模型新概念（你的脸、你的产品、你的风格），但不重新训练完整模型。你想要小 100x 的 delta。这就是 LoRA — 插进已有 attention weights 的 low-rank adapters。

ControlNet + LoRA + text = 2026 年 practitioner 工具箱。大多数生产 image pipelines 会在 SDXL / SD3 / Flux base 上叠 2-5 个 LoRAs、1-3 个 ControlNets 和一个 IP-Adapter。

## 概念

![ControlNet clones the encoder; LoRA adds low-rank deltas](../assets/controlnet-lora.svg)

### ControlNet（Zhang et al., 2023）

取一个 pretrained SD。*克隆* U-Net 的 encoder half。冻结原模型。训练克隆，让它接收额外 conditioning input（edges、depth、pose）。用 *zero-convolution* skip connections（初始化为零的 1×1 convs — 一开始是 no-op，学习 delta）把克隆连回原模型的 decoder half。

```
SD U-Net decoder:   ... ← orig_enc_features + zero_conv(controlnet_enc(condition))
```

Zero-conv init 意味着 ControlNet 一开始是 identity — 即使还没训练也不会伤害。用标准 diffusion loss 在 1M（prompt, condition, image）triples 上训练。

每个 modality 的 ControlNet 都是小 side model（SDXL 约 360M，SD 1.5 约 70M）。推理时可以组合：

```
features += weight_a * control_a(depth) + weight_b * control_b(pose)
```

### LoRA（Hu et al., 2021）

对模型中的任意 linear layer `W ∈ R^{d×d}`，freeze `W` 并加入低秩 delta：

```
W' = W + ΔW,  ΔW = B @ A,  A ∈ R^{r×d},  B ∈ R^{d×r}
```

其中 `r << d`。Attention 标准 rank 是 4-16，重度 fine-tunes 用 64-128。新增参数数：`2 · d · r`，而不是 `d²`。对于 `d=640` 的 SDXL attention、`r=16`：每个 adapter 20k params，而不是 410k — 降低 20x。整个模型上，一个 LoRA 通常是 20-200MB，而 base 是 5GB。

推理时可以缩放 LoRA：`W' = W + α · B @ A`。`α = 0.5-1.5` 很常见。多个 LoRAs 可以相加 stack（常见 caveat：它们会以非线性方式互相作用）。

### IP-Adapter（Ye et al., 2023）

一个 tiny adapter，接收一张 *image* 作为 conditioning（与文本并行）。使用 CLIP image encoder 产生 image tokens，把它们和 text tokens 一起注入 cross-attention。每个 base model 约 20MB。让你无需 LoRA 就能“生成一张具有这个 reference 风格的图像”。

## Composability matrix

| 工具 | 控制什么 | 大小 | 什么时候用 |
|------|------------------|------|-------------|
| ControlNet | Spatial structure（pose、depth、edges） | 70-360MB | 精确 layout、composition |
| LoRA | Style、subject、concept | 20-200MB | Personalization、style |
| IP-Adapter | 来自 reference image 的 style 或 subject | 20MB | 文本无法描述外观 |
| Textual Inversion | 作为新 token 的单个 concept | 10KB | Legacy，大多被 LoRA 取代 |
| DreamBooth | 对 subject 做 full fine-tune | 2-5GB | 强 identity，高 compute |
| T2I-Adapter | 更轻的 ControlNet alternative | 70MB | Edge devices、inference budget |

ControlNet ≈ spatial。LoRA ≈ semantic。两者一起用。

## 构建它

`code/main.py` 在 1-D 上模拟两种机制：

1. **LoRA。** 一个 pretrained linear layer `W`。Freeze 它。训练低秩 `B @ A`，使 `W + BA` 匹配 target linear layer。展示 `r = 1` 足以完美学习 rank-1 correction。

2. **ControlNet-lite。** 一个 “frozen base” predictor 和一个读取额外 signal 的 “side network”。Side network 输出由初始化为零的 learnable scalar gate 控制（我们的 zero-conv 版本）。训练并观察 gate 增大。

### 第 1 步：LoRA math

```python
def lora(W, A, B, x, alpha=1.0):
    # W is frozen; A, B are the trainable low-rank factors.
    return [W[i][j] * x[j] for i, j in ...] + alpha * (B @ (A @ x))
```

### 第 2 步：zero-init side network

```python
side_out = control_net(x, condition)
gated = gate * side_out  # gate initialized to 0
h = base(x) + gated
```

Step 0 时输出和 base 完全相同。训练早期 `gate` 慢慢更新 — 不会 catastrophic drift。

## 陷阱

- **Over-scaling LoRAs。** `α = 2` 或 `α = 3` 是常见“增强它”的 hack，会产生过度 stylized / broken 输出。保持 `α ≤ 1.5`。
- **ControlNet weight conflict。** Pose ControlNet weight 1.0 + Depth ControlNet weight 1.0 通常会 overshoot。权重之和 ≈ 1.0 是安全默认值。
- **LoRA 用错 base。** SDXL LoRAs 在 SD 1.5 上会静默 no-op，因为 attention dimensions 不匹配。Diffusers 0.30+ 会警告。
- **Textual Inversion drift。** 在一个 checkpoint 上训练的 tokens 会在另一个 checkpoint 上严重漂移。LoRA 更可移植。
- **LoRA weight-merging and storage。** 你可以把 LoRA bake 进 base model weights 来加速推理（运行时不做 addition），但会失去运行时缩放 `α` 的能力。保留两个版本。

## 使用它

| 目标 | 2026 pipeline |
|------|---------------|
| 复现品牌 art style | 在约 30 张 curated images 上训练 rank 32 LoRA |
| 把我的脸放进生成图 | DreamBooth 或 LoRA + IP-Adapter-FaceID |
| 特定 pose + prompt | ControlNet-Openpose + SDXL + text |
| Depth-aware composition | ControlNet-Depth + SD3 |
| Reference + prompt | IP-Adapter + text |
| 精确 layout | ControlNet-Scribble 或 ControlNet-Canny |
| Background replace | ControlNet-Seg + Inpainting（第 09 课） |
| Fast 1-step style | SDXL-Turbo 上的 LCM-LoRA |

## 交付它

保存 `outputs/skill-sd-toolkit-composer.md`。Skill 接收一个任务（input assets：prompt、optional reference image、optional pose、optional depth、optional scribble），并输出 tool stack、weights 和 reproducible seed protocol。

## 练习

1. **简单。** 在 `code/main.py` 中把 LoRA rank `r` 从 1 变到 4。LoRA 到哪个 rank 才能精确匹配 rank-2 target delta？
2. **中等。** 在两个 target transforms 上分别训练两个 LoRAs。一起加载它们并展示 additive interaction。什么时候 interaction 会破坏线性？
3. **困难。** 用 diffusers stack：SDXL-base + Canny-ControlNet（weight 0.8）+ style LoRA（α 0.8）+ IP-Adapter（weight 0.6）。随着 stack weights 变化，测量 FID-vs-prompt-adherence trade-off。

## 关键词

| 术语 | 大家怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| ControlNet | “Spatial control” | 克隆 encoder + zero-conv skips；读取 conditioning image。 |
| Zero convolution | “Starts as identity” | 初始化为零的 1×1 conv；ControlNet 从 no-op 开始。 |
| LoRA | “Low-rank adapter” | `W + B @ A`，`r << d`；比 full fine-tune 少 100x 参数。 |
| rank r | “The knob” | LoRA 压缩度；典型 4-16，重度 personalization 用 64+。 |
| α | “LoRA strength” | LoRA delta 的运行时缩放。 |
| IP-Adapter | “Reference image” | 通过 CLIP-image tokens 做小型 image-conditioning adapter。 |
| DreamBooth | “Full subject fine-tune” | 在约 30 张 subject 图像上训练完整模型。 |
| Textual Inversion | “New token” | 只学习新的 word embedding；legacy，大多被取代。 |

## 生产备注：LoRA swaps、ControlNet lanes、multi-tenant serving

真实 text-to-image SaaS 会在同一个 base checkpoint 上服务数百个 LoRAs 和十几个 ControlNets。Serving 问题很像 LLM multi-tenancy（生产文献在 continuous batching 和 LoRAX / S-LoRA 下讨论 LLM case）：

- **Hot-swap LoRAs，不要 merge。** 把 `W' = W + α·B·A` merge 进 base 会让每步推理快约 3-5%，但冻结了 `α` 和 base。把 LoRAs 作为 rank-r deltas 热驻留 VRAM；diffusers 暴露 `pipe.load_lora_weights()` + `pipe.set_adapters([...], adapter_weights=[...])` 支持 per-request activation。Swap cost 是 `2 · d · r · num_layers` 权重 — MB 级，sub-second。
- **ControlNet 是第二条 attention lane。** 克隆 encoder 与 base 并行运行。两个 weight 1.0 的 ControlNets = 每步两个额外 forward passes，不是一次 merged pass。Batch-size headroom 会按二次方式下降。为每个 active ControlNet 预算约 1.5× step cost。
- **LoRAs 也能量化。** 如果 base 已量化（见第 07 课，8GB 上的 Flux），LoRA delta 也能干净量化到 8-bit 或 4-bit。QLoRA-style loading 让你能在 4-bit Flux base 上叠 5-10 个 LoRAs 而不爆内存。

Flux-specific：Niels 的 Flux-on-8GB notebook 量化 base 到 4-bit；在该量化 base 上 stack 一个 style LoRA（`pipe.load_lora_weights("user/style-lora")`），并指定 `weight_name="pytorch_lora_weights.safetensors"` 仍然有效。这是 2026 年大多数 SaaS agencies 实际上线的配方。

## 延伸阅读

- [Zhang, Rao, Agrawala (2023). Adding Conditional Control to Text-to-Image Diffusion Models](https://arxiv.org/abs/2302.05543) — ControlNet。
- [Hu et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) — LoRA（最初用于 LLMs；移植到 diffusion）。
- [Ye et al. (2023). IP-Adapter: Text Compatible Image Prompt Adapter](https://arxiv.org/abs/2308.06721) — IP-Adapter。
- [Mou et al. (2023). T2I-Adapter: Learning Adapters to Dig Out More Controllable Ability](https://arxiv.org/abs/2302.08453) — ControlNet 的轻量替代。
- [Ruiz et al. (2023). DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation](https://arxiv.org/abs/2208.12242) — DreamBooth。
- [HuggingFace Diffusers — ControlNet / LoRA / IP-Adapter docs](https://huggingface.co/docs/diffusers/training/controlnet) — 参考 pipelines。
