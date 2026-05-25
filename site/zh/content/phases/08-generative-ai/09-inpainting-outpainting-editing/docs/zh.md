# Inpainting、Outpainting 与 Image Editing

> Text-to-image 创造新东西。Inpainting 修复旧东西。生产中，70% 可计费图像工作都是编辑 — 换背景、去 logo、扩展画布、重画手。Inpainting 是 diffusion 证明自己价值的地方。

**类型：** 构建
**语言：** Python
**前置要求：** 阶段 8 · 07（Latent Diffusion），阶段 8 · 08（ControlNet & LoRA）
**时间：** ~75 分钟

## 问题

客户发来一张完美产品照，但背景里有一个分散注意力的标识。你想擦掉标识，并保持其他所有像素完全一致。你不能从头运行 text-to-image — 结果会有不同颜色、不同光照、不同产品角度。你只想重新生成 *masked region*，并希望生成结果尊重周围 context。

这就是 inpainting。变体：

- **Inpainting。** 在 mask 内重新生成，保留外部 pixels。
- **Outpainting。** 在 mask 外（或画布之外）重新生成，保留内部。
- **Image editing。** 重新生成整张图，但保持与原图的语义或结构一致（SDEdit、InstructPix2Pix）。

2026 年每个 diffusion pipeline 都有 inpainting mode。Flux.1-Fill、Stable Diffusion Inpaint、SDXL-Inpaint、DALL-E 3 Edit。它们使用同一个原则。

## 概念

![Inpainting: mask-aware denoising with context-preserving reinjection](../assets/inpainting.svg)

### Naive 方法（以及为什么错）

用 mask 运行标准 text-to-image。在每个 sampling step，把 unmasked region 的 noisy latent 替换成 clean image 的 forward-diffused 版本。它能工作……但很差。Boundary artifacts 会渗透，因为模型不知道 masked region 里有什么。

### 正确的 inpainting model

训练一个修改后的 U-Net，它接收 9 个 input channels，而不是 4 个：

```
input = concat([ noisy_latent (4ch), encoded_image (4ch), mask (1ch) ], dim=channel)
```

额外 channels 是 VAE-encoded source image 的副本，加一个单通道 mask。训练时，随机 mask 图像区域，训练模型只 denoise masked region，而 unmasked region 作为 clean conditioning signal 给出。推理时，模型可以“看见”masked region 周围内容，并产生一致补全。

SD-Inpaint、SDXL-Inpaint、Flux-Fill 都使用这种 9-channel（或类似）输入。Diffusers `StableDiffusionInpaintPipeline`、`FluxFillPipeline`。

### SDEdit（Meng et al., 2022）— 免费编辑

把 source image 加噪到某个中间 `t`，然后带新 prompt 从 `t` 反向跑到 0。不需要重新训练。起始 `t` 的选择在 fidelity 和 creative freedom 之间取舍：

- `t/T = 0.3` → 几乎和 source 相同，小风格变化
- `t/T = 0.6` → 中等编辑，保留粗结构
- `t/T = 0.9` → 从接近噪声生成，source preservation 很少

### InstructPix2Pix（Brooks et al., 2023）

在 `(input_image, instruction, output_image)` triples 上 fine-tune diffusion model。推理时同时 condition on input image 和 text instruction（“make it sunset”，“add a dragon”）。两个 CFG scales：image scale 和 text scale。

### RePaint（Lugmayr et al., 2022）

保持标准 unconditional diffusion model。在每个 reverse step，做 resample — 偶尔跳回更 noisy 的状态并重新生成。避免 boundary artifacts。用于没有训练好的 inpainting model 时。

## 构建它

`code/main.py` 在 5 维数据上实现 toy 1-D inpainting。我们在 5-D mixture data 上训练 DDPM，每个 sample 是来自两个 clusters 之一的 5 个 floats。推理时，我们 “mask” 5 维中的 2 维，在每一步注入 unmasked 三维的 noisy-forward 版本，并只重新生成 masked dimensions。

### 第 1 步：5-D DDPM data

```python
def sample_data(rng):
    cluster = rng.choice([0, 1])
    center = [-1.0] * 5 if cluster == 0 else [1.0] * 5
    return [c + rng.gauss(0, 0.2) for c in center], cluster
```

### 第 2 步：训练覆盖全部 5 dims 的 denoiser

标准 DDPM。Net 对 5-D noisy input 输出 5-D noise prediction。

### 第 3 步：推理时做 mask-aware reverse

```python
def inpaint_step(x_t, mask, clean_image, alpha_bars, t, rng):
    # replace unmasked dims with a freshly noised version of the clean source
    a_bar = alpha_bars[t]
    for i in range(len(x_t)):
        if not mask[i]:
            x_t[i] = math.sqrt(a_bar) * clean_image[i] + math.sqrt(1 - a_bar) * rng.gauss(0, 1)
    # ...then run the normal reverse step on x_t
```

这是 naive 方法，在 toy 1-D 数据上有效。真实图像 inpainting 使用 9-channel input，因为 texture coherence 更重要。

### 第 4 步：outpainting

Outpainting 是 mask 反过来的 inpainting：mask 新的（之前不存在的）canvas，其余部分用原图填充。训练目标完全相同。

## 陷阱

- **Seams。** Naive 方法会留下可见边界，因为 gradient info 不会跨 mask 流动。修复：把 mask dilate 8-16 pixels，或使用 proper inpainting model。
- **Mask leakage。** 如果 conditioning image 的 unmasked region 低质量或有噪声，它会污染 mask 内生成。先稍微 denoise 或 blur。
- **CFG 与 mask size 互动。** 小 mask 上高 CFG = saturated patch。小编辑降低 CFG。
- **SDEdit fidelity cliff。** 从 `t/T = 0.5` 到 `t/T = 0.6` 可能丢失主体 identity。Sweep 并 checkpoint。
- **Prompt mismatch。** Prompt 应描述 *整张* 图，而不只是新内容。用 “A cat sitting on a chair”，不是 “a cat”。

## 使用它

| 任务 | Pipeline |
|------|----------|
| 移除物体，小 mask | SD-Inpaint 或 Flux-Fill，标准 prompt |
| 替换天空 | SD-Inpaint + “blue sky at sunset” |
| 扩展画布 | SDXL outpaint mode（8px feather）或 Flux-Fill with outpaint mask |
| 重生成手 / 脸 | SD-Inpaint，prompt 重新描述主体 + ControlNet-Openpose |
| 改变某一区域风格 | Masked region 上 `t/T=0.5` 的 SDEdit |
| “Make it sunset” | InstructPix2Pix 或 Flux-Kontext |
| Background replacement | SAM mask → SD-Inpaint |
| Ultra-high-fidelity | Flux-Fill 或 GPT-Image（hosted）用于最难情况 |

SAM（Meta 的 Segment Anything，2023）+ diffusion inpaint 是 2026 年背景移除管线。SAM 2（2024）可用于视频。

## 交付它

保存 `outputs/skill-editing-pipeline.md`。Skill 接收 original image + edit description + optional mask（或 SAM prompt），并输出：mask-generation approach、base model、CFG scales（image + text）、SDEdit-t 或 inpainting mode，以及 QA checklist。

## 练习

1. **简单。** 在 `code/main.py` 中，把 masked dimensions 比例从 0.2 调到 0.8。到哪个比例时，inpaint quality（masked dims residual）等同于 unconditional generation？
2. **中等。** 实现 RePaint：每第 10 个 reverse step，跳回 5 步（加噪）并重新 denoise。测量它是否降低 mask edge 的 boundary residual。
3. **困难。** 用 Hugging Face diffusers 比较：SD 1.5 Inpaint + ControlNet-Openpose vs Flux.1-Fill，在 20 个 face-regeneration tasks 上测试。分别打分 pose adherence 和 identity preservation。

## 关键词

| 术语 | 大家怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| Inpainting | “Fill the hole” | 在 mask 内重新生成；保留外部 pixels。 |
| Outpainting | “Extend the canvas” | 在画布外重新生成；保留内部。 |
| 9-channel U-Net | “Proper inpainting model” | 以 `noisy | encoded-source | mask` 作为输入的 U-Net。 |
| SDEdit | “带 noise level 的 img2img” | 加噪到时间 `t`，再用新 prompt denoise。 |
| InstructPix2Pix | “Text-only edits” | 在（image, instruction, output）triples 上 fine-tuned diffusion。 |
| RePaint | “No retraining” | 在 reverse 中周期性 re-noise 来减少 seams。 |
| SAM | “Segment Anything” | 通过 clicks 或 boxes 生成 mask；搭配 inpaint。 |
| Flux-Kontext | “Edit with context” | Flux 变体，接受 reference image + instruction 做编辑。 |

## 生产备注：edit pipelines 对延迟敏感

用户编辑图像时，期望 sub-5-second round trips。1024² 的 30-step SDXL-Inpaint 在 L4 上约 3-4 s，再加 SAM mask generation（~200 ms）和 VAE encode/decode（合计 ~500 ms）。生产框架中，这是 TTFT-bound 而不是 throughput-bound — batch 1、低并发，最小化每个阶段：

- **SAM-H 是慢的。** 1024² 上 SAM-H 约 200 ms；SAM-ViT-B 约 40 ms，质量损失小。SAM 2（video）增加 temporal overhead；不要用于单图编辑。
- **能跳过 encode 就跳过。** `pipe.image_processor.preprocess(img)` 会 encode 到 latents。如果你有上一次 generation 的 latents（iterative-edit UI 中常见），直接通过 `latents=...` 传入，跳过一次 VAE encode。
- **Mask dilation 也影响 throughput。** 小 mask 意味着大部分 U-Net forward 被浪费（unmasked pixels 反正会 clamp）。`diffusers` 的 `StableDiffusionInpaintPipeline` 无论如何都跑完整 U-Net；只有 9-channel proper-inpaint 变体会利用 masked compute。
- **Flux-Kontext 是 2025 年答案。** 对 `(source_image, instruction)` 做单次 forward pass — 没有单独 mask，没有 SDEdit noise sweep。在 H100 上约 1.5 s 完成编辑。架构教训：折叠 stages。

## 延伸阅读

- [Lugmayr et al. (2022). RePaint: Inpainting using Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2201.09865) — training-free inpainting。
- [Meng et al. (2022). SDEdit: Guided Image Synthesis and Editing with Stochastic Differential Equations](https://arxiv.org/abs/2108.01073) — SDEdit。
- [Brooks, Holynski, Efros (2023). InstructPix2Pix](https://arxiv.org/abs/2211.09800) — text-instruction editing。
- [Kirillov et al. (2023). Segment Anything](https://arxiv.org/abs/2304.02643) — SAM，mask source。
- [Ravi et al. (2024). SAM 2: Segment Anything in Images and Videos](https://arxiv.org/abs/2408.00714) — video SAM。
- [Hertz et al. (2022). Prompt-to-Prompt Image Editing with Cross-Attention Control](https://arxiv.org/abs/2208.01626) — attention-level editing。
- [Black Forest Labs (2024). Flux.1-Fill and Flux.1-Kontext](https://blackforestlabs.ai/flux-1-tools/) — 2024 tooling。
