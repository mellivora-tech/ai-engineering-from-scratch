# Latent Diffusion 与 Stable Diffusion

> 在 512×512 图像上做 pixel-space diffusion 是计算上的战争罪。Rombach et al.（2022）注意到，生成图像并不需要全部 786k 维 — 你需要足够捕捉语义结构，再用单独 decoder 处理剩下部分。在 VAE 的 latent space 里运行 diffusion。这个想法就是 Stable Diffusion。

**类型：** 构建
**语言：** Python
**前置要求：** 阶段 8 · 02（VAE），阶段 8 · 06（DDPM），阶段 7 · 09（ViT）
**时间：** ~75 分钟

## 问题

512² 的 pixel-space diffusion 意味着 U-Net 运行在形状为 `[B, 3, 512, 512]` 的 tensors 上。对于 500M-param U-Net，每个 sampling step 约 100 GFLOPS。五十步就是每张图 5 TFLOPS。在十亿张图上训练，计算账单荒谬。

大多数 FLOPs 都花在把感知上不重要的细节推过网络 — 也就是有损 VAE 本可以压缩掉的高频纹理。Rombach 的想法：先训练一次 VAE（*first stage*），freeze 它，然后完全在 4-channel 64×64 latent space（*second stage*）中运行 diffusion。同一个 U-Net。像素数 1/16。以约 64x 更少 FLOPs 达到可比质量。

这就是 Stable Diffusion 配方。SD 1.x / 2.x 使用在 `64×64×4` latents 上的 860M U-Net，SDXL 使用在 `128×128×4` 上的 2.6B U-Net，SD3 把 U-Net 换成带 flow matching 的 Diffusion Transformer（DiT）。Flux.1-dev（Black Forest Labs, 2024）发布了一个 12B-param DiT-MMDiT。它们全都运行在同一个 two-stage substrate 上。

## 概念

![Latent diffusion: VAE compression + diffusion in latent space](../assets/latent-diffusion.svg)

**两个 stage，分开训练。**

1. **Stage 1 — VAE。** Encoder `E(x) → z`，decoder `D(z) → x`。目标压缩：每个空间轴下采样 8× + 调整 channels，使 total latent size 约为 pixel count 的 1/16。Loss = reconstruction（L1 + LPIPS perceptual）+ KL（小权重，让 `z` 不会被强迫得太 Gaussian，因为我们并不需要从 `z` 精确采样）。通常还带 adversarial loss，使 decoded images 更锐利。

2. **Stage 2 — 在 `z` 上做 diffusion。** 把 `z = E(x_real)` 当作数据。训练 U-Net（或 DiT）denoise `z_t`。推理时：通过 diffusion 采样 `z_0`，然后 `x = D(z_0)`。

**Text conditioning。** 两个额外组件。一个 frozen text encoder（SD 1.x 用 CLIP-L，SD 2/XL 用 CLIP-L+OpenCLIP-G，SD3 和 Flux 用 T5-XXL）。一个 cross-attention injection：每个 U-Net block 接收 `[Q = image features, K = V = text tokens]` 并混合。Tokens 是文本影响图像的唯一通道。

**Loss function 和第 06 课完全相同。** 同样的 DDPM / flow matching noise MSE。你只是替换了数据域。

## 架构变体

| 模型 | 年份 | Backbone | Latent shape | Text encoder | Params |
|-------|------|----------|--------------|--------------|--------|
| SD 1.5 | 2022 | U-Net | 64×64×4 | CLIP-L（77 tokens） | 860M |
| SD 2.1 | 2022 | U-Net | 64×64×4 | OpenCLIP-H | 865M |
| SDXL | 2023 | U-Net + refiner | 128×128×4 | CLIP-L + OpenCLIP-G | 2.6B + 6.6B |
| SDXL-Turbo | 2023 | Distilled | 128×128×4 | same | 1-4 step sampling |
| SD3 | 2024 | MMDiT (multimodal DiT) | 128×128×16 | T5-XXL + CLIP-L + CLIP-G | 2B / 8B |
| Flux.1-dev | 2024 | MMDiT | 128×128×16 | T5-XXL + CLIP-L | 12B |
| Flux.1-schnell | 2024 | MMDiT distilled | 128×128×16 | T5-XXL + CLIP-L | 12B, 1-4 step |

趋势：用 DiT（latent patches 上的 transformer）替换 U-Net，scale text encoder（T5 在 prompt adherence 上胜过 CLIP），增加 latent channels（4 → 16 给更多细节余量）。

## 构建它

`code/main.py` 把一个 toy 1-D “VAE”（为了演示，是 identity encoder + decoder；真实 VAE 会是 conv net）叠在第 06 课的 DDPM 上，并加入带 classifier-free guidance 的 class conditioning。它展示了同一个 diffusion loss 无论运行在原始 1-D 值还是 encoded values 上都有效 — 这是关键洞见。

### 第 1 步：encoder/decoder

```python
def encode(x):    return x * 0.5          # toy "compression" to smaller scale
def decode(z):    return z * 2.0
```

真实 VAE 有训练得到的权重。教学上，这个线性映射足以展示 diffusion 在 `z` 上运行，而无需关心原始数据空间。

### 第 2 步：在 `z`-space 中 diffusion

和第 06 课同一个 DDPM。Net 看到的数据是 `z = E(x)`。采样 `z_0` 后，用 `D(z_0)` decode。

### 第 3 步：classifier-free guidance

训练时，10% 的时间 drop 掉 class label（替换为 null token）。推理时同时计算 `ε_cond` 和 `ε_uncond`，然后：

```python
eps_cfg = (1 + w) * eps_cond - w * eps_uncond
```

`w = 0` = 无 guidance（完整多样性），`w = 3` = 默认，`w = 7+` = 饱和 / 过锐。

### 第 4 步：text conditioning（概念，不在代码中）

用 frozen text encoder output 替换 class label。通过 cross-attention 把 text embedding 喂给 U-Net：

```python
h = h + CrossAttention(Q=h, K=text_embed, V=text_embed)
```

这是 class-conditional diffusion model 和 Stable Diffusion 之间唯一实质差异。

## 陷阱

- **VAE-scale mismatch。** SD 1.x VAEs 在 encoding 后有一个 scaling constant（`scaling_factor ≈ 0.18215`）。忘记它会让 U-Net 在方差严重错误的 latents 上训练。每个 checkpoint 都带这个值。
- **Text encoder 静默错误。** SD3 需要 T5-XXL 且 >=128 tokens，fallback 到 CLIP-only 会损失很大。总是检查 `use_t5=True`，否则 prompt fidelity 会塌。
- **混用 latent spaces。** SDXL、SD3、Flux 都使用不同 VAEs。在 SDXL latents 上训练的 LoRA 不能用于 SD3。Hugging Face diffusers 0.30+ 会拒绝加载不匹配 checkpoints。
- **CFG 太高。** `w > 10` 会产生饱和、油腻的图像，并以多样性为代价过拟合 prompt。甜点区是 `w = 3-7`。
- **Negative prompts leaking。** 空 negative prompt 变成 null token；填了内容的 negative prompt 变成 `ε_uncond`。二者不同；有些 pipelines 会静默默认到 null。

## 使用它

2026 年生产 stack：

| 目标 | 推荐 backbone |
|--------|----------------------|
| 窄领域、paired data、从零训练模型 | SDXL fine-tune（LoRA / full）— 上线最快 |
| Open-domain text-to-image，open weights | Flux.1-dev（12B，Apache / non-commercial）或 SD3.5-Large |
| 最快推理，open weights | Flux.1-schnell（1-4 step，Apache）或 SDXL-Lightning |
| 最佳 prompt adherence，hosted | GPT-Image / DALL-E 3（仍然）、Midjourney v7、Imagen 4 |
| 编辑 workflows | Flux.1-Kontext（2024 年 12 月）— 原生接受 image + text |
| 研究 baseline | SD 1.5 — 古老但研究充分 |

## 交付它

保存 `outputs/skill-sd-prompter.md`。Skill 接收 text prompt + target style，并输出：model + checkpoint、CFG scale、sampler、negative prompt、resolution、可选 ControlNet/IP-Adapter combo，以及 per-step QA checklist。

## 练习

1. **简单。** 用 guidance `w ∈ {0, 1, 3, 7, 15}` 运行 `code/main.py`。记录每类的 mean sample。`w` 到多少时 class means 超过真实数据 means？
2. **中等。** 把 toy linear encoder 换成带 reconstruction loss 的 tanh-MLP encoder/decoder pair。在新 latents 上重新训练 diffusion。Sample quality 会变化吗？
3. **困难。** 用 diffusers 设置真实 Stable Diffusion inference：加载 `sdxl-base`，用 CFG=7 运行 30 Euler steps，计时。再切到 `sdxl-turbo`，4 steps 且 CFG=0。同一 subject，不同质量 — 描述变化及原因。

## 关键词

| 术语 | 大家怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| First stage | “The VAE” | 训练好的 encoder/decoder pair；把 512² 压缩到 64²。 |
| Second stage | “The U-Net” | Latent space 上的 diffusion model。 |
| CFG | “Guidance scale” | `(1+w)·ε_cond - w·ε_uncond`；调节 conditioning strength。 |
| Null token | “Empty prompt embed” | 用于 `ε_uncond` 的 unconditional embed。 |
| Cross-attention | “文本如何进入” | 每个 U-Net block 把 text tokens 作为 K 和 V 来 attend。 |
| DiT | “Diffusion Transformer” | 用 latent patches 上的 transformer 替换 U-Net；scale 更好。 |
| MMDiT | “Multi-modal DiT” | SD3 架构：text 和 image streams 使用 joint attention。 |
| VAE scaling factor | “Magic number” | 把 latents 除以约 5.4，让 diffusion 在 unit-variance 空间中运行。 |

## 生产备注：在 8GB 消费级 GPU 上运行 Flux-12B

reference Flux integration 是标准的“我有消费级 GPU，能上线这个吗？”配方。技巧和生产推理文献里的三旋钮配方相同，只是应用到 diffusion DiT：

1. **Staggered loading。** Flux 有三个不必同时驻留 VRAM 的网络：T5-XXL text encoder（fp32 约 10 GB）、CLIP-L（小）、12B MMDiT 和 VAE。先 encode prompt，*delete* encoders，加载 DiT，denoise，*delete* DiT，加载 VAE，decode。消费级 8GB GPU 一次只装一个 stage。
2. **通过 bitsandbytes 做 4-bit quantization。** 在 T5 encoder 和 DiT 上使用 `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)`。内存降 8×，按 Aritra 的 benchmarks（notebook 中链接），text-to-image 质量下降不可察觉。
3. **CPU offload。** `pipe.enable_model_cpu_offload()` 会随着 forward pass 推进，自动在 CPU 和 GPU 间交换 modules。增加 10-20% latency，但能让 pipeline 跑起来。

内存账：`10 GB T5 / 8 = 1.25 GB` 量化后，`12 B params × 0.5 bytes = ~6 GB` 量化 DiT，加上 activations。用 stas00 的话说，这是 TP=1 inference 的极端端点 — 没有 model parallelism，最大量化。生产中你会在 H100 上用 TP=2 或 TP=4；单台开发笔记本就用这个配方。

## 延伸阅读

- [Rombach et al. (2022). High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752) — Stable Diffusion。
- [Podell et al. (2023). SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis](https://arxiv.org/abs/2307.01952) — SDXL。
- [Peebles & Xie (2023). Scalable Diffusion Models with Transformers (DiT)](https://arxiv.org/abs/2212.09748) — DiT。
- [Esser et al. (2024). Scaling Rectified Flow Transformers for High-Resolution Image Synthesis](https://arxiv.org/abs/2403.03206) — SD3、MMDiT。
- [Ho & Salimans (2022). Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598) — CFG。
- [Labs (2024). Flux.1 — Black Forest Labs announcement](https://blackforestlabs.ai/announcing-black-forest-labs/) — Flux.1 family。
- [Hugging Face Diffusers docs](https://huggingface.co/docs/diffusers/index) — 上述每个 checkpoint 的参考实现。
