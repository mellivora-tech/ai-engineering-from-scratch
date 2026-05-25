# Video Generation

> 图像是 2-D tensor。视频是 3-D tensor。理论相同；计算难 10-100x。OpenAI 的 Sora（2024 年 2 月）证明了它可行。到 2026 年，Veo 2、Kling 1.5、Runway Gen-3、Pika 2.0 和 WAN 2.2 已经能从文本生成 1080p 生产级视频 — open-weights stack（CogVideoX、HunyuanVideo、Mochi-1、WAN 2.2）落后约 12 个月。

**类型：** 构建
**语言：** Python
**前置要求：** 阶段 8 · 07（Latent Diffusion），阶段 7 · 09（ViT），阶段 8 · 06（DDPM）
**时间：** ~45 分钟

## 问题

10 秒 1080p、24fps 视频是 240 帧 1920×1080×3 pixels。每个 clip 约 1.5 GB raw data。Pixel-space diffusion 不可行。你需要：

1. **Spatiotemporal compression。** 一个把视频而不是单帧编码为 spatial-temporal patches 序列的 VAE。
2. **Temporal coherence。** 帧之间需要在数秒内共享内容、光照和物体 identity。Net 必须建模运动。
3. **Compute budget。** 同等模型大小下，视频训练比图像贵 10-100x。
4. **Conditioning。** Text、image（first-frame）、audio 或另一个 video。多数生产模型四者都支持。

解决这件事的架构是应用在 spatiotemporal patches 上的 **Diffusion Transformer（DiT）**，在巨大（prompt, caption, video）数据集上训练。和第 06 课是同一个 diffusion loss。

## 概念

![Video diffusion: patchify, DiT, decode](../assets/video-generation.svg)

### Patchify

用 3D VAE（学习到的 spatiotemporal compression）编码视频。Latent 形状是 `[T_latent, H_latent, W_latent, C_latent]`。切成大小为 `[t_p, h_p, w_p]` 的 patches。对于 Sora-style models，`t_p = 1`（per-frame patches）或 `t_p = 2`（每两帧）。一个 10 秒 1080p 视频会压缩成约 20,000-100,000 patches。

### Spatiotemporal DiT

Transformer 处理扁平 patch 序列。每个 patch 有 3D positional embedding（time + y + x）。Attention 通常 factorized：

- **Spatial attention** 在每帧的 patches 内。
- **Temporal attention** 跨 frames，在相同 spatial location。
- **Full 3D attention** 贵 16-100x；只在低分辨率或研究中使用。

### Text conditioning

与大型 text encoder cross-attention（Sora 用 T5-XXL，CogVideoX-5B 用 T5-XXL）。长 prompt 很重要 — Sora 的训练集有 GPT 生成的 dense re-captions，平均每个 clip 约 200 tokens。

### Training

在 spatiotemporal latents 上用标准 diffusion loss（ε 或 v prediction）。数据：web video + ~100M curated clips + synthetic text captions。Compute：即使小研究运行也要 10,000+ GPU hours；Sora-scale 是 100,000+。

## 2026 年生产格局

| 模型 | 日期 | 最大时长 | 最大分辨率 | Open weights? | 亮点 |
|-------|------|--------------|---------|---------------|---------|
| Sora (OpenAI) | 2024-02 | 60s | 1080p | No | 第一个在规模上展示 world simulator properties 的模型 |
| Sora Turbo | 2024-12 | 20s | 1080p | No | 生产版 Sora，推理快 5x |
| Veo 2 (Google) | 2024-12 | 8s | 4K | No | 2025 年最高质量 + physics |
| Veo 3 | 2025 Q3 | 15s | 4K | No | 原生 audio 和更强 camera control |
| Kling 1.5 / 2.1 (Kuaishou) | 2024-2025 | 10s | 1080p | No | 2025 Q1 最好 human motion |
| Runway Gen-3 Alpha | 2024-06 | 10s | 768p | No | 上层有专业视频工具 |
| Pika 2.0 | 2024-10 | 5s | 1080p | No | 最强 character consistency |
| CogVideoX (THUDM) | 2024 | 10s | 720p | Yes (2B, 5B) | 第一个 open 5B-scale video |
| HunyuanVideo (Tencent) | 2024-12 | 5s | 720p | Yes (13B) | 2024 末 open SOTA |
| Mochi-1 (Genmo) | 2024-10 | 5.4s | 480p | Yes (10B) | 最宽松 license |
| WAN 2.2 (Alibaba) | 2025-07 | 5s | 720p | Yes | 2025 年中最强 open model |

Open weights 在视频领域追赶得比图像更快：到 2026 年中，HunyuanVideo + WAN 2.2 LoRAs 已经驱动大多数 open-source workflows。

## 构建它

`code/main.py` 模拟核心 spatiotemporal DiT 思路：patchify 一个小 synthetic video，添加 per-patch position embedding，并用 transformer-style attention over patches denoise 整个序列。无 numpy，纯 Python。我们展示即使在 1-D 中，当 adjacent-frame patches 共享 denoiser 和 position embeddings 时，也会出现 temporal coherence。

### 第 1 步：patchify 一个 synthetic 1-D “video”

```python
def make_video(T_frames=8, rng=None):
    # a "video" is a sequence of 1-D values following a smooth trajectory
    base = rng.gauss(0, 1)
    return [base + 0.3 * t + rng.gauss(0, 0.1) for t in range(T_frames)]
```

### 第 2 步：每帧 position embedding

```python
def pos_embed(t, dim):
    return sinusoidal(t, dim)
```

### 第 3 步：denoiser 看到整个序列

我们的 tiny net 不是独立 denoise 每帧，而是拼接所有 frame values + position embeddings，并预测所有 frames 的 noise。

### 第 4 步：temporal coherence test

训练后采样一个视频。测量 frame-to-frame delta。如果模型学到了 temporal structure，deltas 应该小于独立采样每一帧。

## 陷阱

- **Independent per-frame sampling = flicker。** 如果你对每帧独立运行 image diffusion，输出会 flicker，因为每帧 noise 独立。Video diffusion 通过 attention 或 shared noise 耦合 frames 来修复。
- **Naive 3D attention = OOM。** 在 10 秒 1080p latent 上做 full 3D attention 是数千亿 operations。拆成 spatial + temporal。
- **Data captioning 比规模更重要。** Sora 相比 prior work 的主要升级，是在约 10x 更详细 captions 上训练（GPT-4 重新标注 clips）。OpenAI technical report 明确说明了这一点。
- **First-frame conditioning。** 大多数生产模型也接受一张图作为 first frame。这是 “image-to-video” mode；训练包含这个变体。
- **Physics drift。** 长 clips（>10s）会累积细微不一致。Sliding-window generation + keyframe anchoring 有帮助。

## 使用它

| 用例 | 2026 选择 |
|----------|-----------|
| 最高质量 text-to-video，hosted | Veo 3 或 Sora |
| Camera-controlled cinematic | Runway Gen-3 with motion brushes |
| 跨 clips character consistency | Pika 2.0 或 Kling 2.1 |
| Open weights，快速 fine-tune | WAN 2.2 + LoRA |
| Image-to-video | WAN 2.2-I2V、Kling 2.1 I2V 或 Runway |
| Audio-to-video lip sync | Veo 3（原生 audio）或专用 lip-sync model |
| Video editing | Runway Act-Two、Kling Motion Brush、Flux-Kontext（still-frame） |

同等质量下，每秒视频成本在 2024 到 2026 年间下降了 20x。

## 交付它

保存 `outputs/skill-video-brief.md`。Skill 接收 video brief（duration、aspect ratio、style、camera plan、subject consistency、audio），并输出：model + hosting、prompt scaffolding（camera language、subject description、motion descriptors）、seed + reproducibility protocol，以及 frame-level QA checklist。

## 练习

1. **简单。** 在 `code/main.py` 中比较（a）independent per-frame sampling，（b）joint sequence sampling 的 frame-to-frame delta。报告 deltas 的 mean 和 variance。
2. **中等。** 添加 first-frame condition：把 frame 0 固定为给定值，采样剩余 frames。测量 pinned value 如何传播。
3. **困难。** 用 HuggingFace diffusers 在本地 GPU 上运行 CogVideoX-2B。对 720p、6 秒 clip 运行 20 inference steps 并计时。Profile spatiotemporal attention 找出瓶颈。

## 关键词

| 术语 | 大家怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| Video VAE | “3-D VAE” | 把 `(T, H, W, C)` 压缩为 spatiotemporal latent 的 encoder。 |
| Patches | “Tokens” | Latent 的固定大小 3-D blocks；DiT 的输入。 |
| Factorized attention | “Spatial + temporal” | 先对空间做 attention，再对时间做；跳过 full 3-D attention。 |
| Image-to-video (I2V) | “Animate this photo” | 模型接收 image + text，并输出从它开始的视频。 |
| Keyframe conditioning | “Anchor frames” | 固定特定帧以控制视频 arc。 |
| Motion brush | “Directional hint” | 用户在图像上绘制 motion vectors 的 UI 输入。 |
| Re-captioning | “Dense captions” | 用 LLM 对训练 clips 重新标注详细 prompts。 |
| Flicker | “Temporal artifact” | 帧间不一致；通过 coupled denoising 修复。 |

## 生产备注：video latents 是 memory-bandwidth 问题

10 秒 1080p clip、24 fps 是 240 frames × 1920 × 1080 × 3 ≈ 1.5 GB raw pixels。经过 4× video VAE compression（`2 × spatial × 2 × temporal`）后，每个请求 latent 约 100 MB。把它通过 spatiotemporal DiT 跑 30 steps，batch 1，每步要通过 HBM 搬运约 3 GB — 瓶颈是 memory bandwidth，不是 FLOPs。

三个生产旋钮，全部直接来自 production-inference literature 的 inference chapter：

- **跨 DiT 做 TP。** Text-to-video models 经常 ≥10B params。4 张 H100 上 TP=4 是标准；405B-class models 用 PP=2 × TP=2。每 step latency 随 TP 近似线性下降，直到撞上 all-reduce wall。
- **Frame batching = continuous batching。** 生成时，视频概念上是一批被 attention 连接的 frames。Continuous batching（in-flight scheduling）适用：如果模型架构允许 sliding-window generation，可在返回 frame `t-1` 时开始渲染 frame `t+1`。
- **Clip-level prefill cache。** 对 image-to-video，first-frame conditioning 类似 LLM 的 prompt prefill：算一次，在 temporal decoder passes 之间复用。这本质上是视频的 KV-cache。

## 延伸阅读

- [Brooks et al. (2024). Video generation models as world simulators](https://openai.com/index/video-generation-models-as-world-simulators/) — Sora technical report。
- [Yang et al. (2024). CogVideoX: Text-to-Video Diffusion Models with An Expert Transformer](https://arxiv.org/abs/2408.06072) — CogVideoX。
- [Kong et al. (2024). HunyuanVideo: A Systematic Framework for Large Video Generative Models](https://arxiv.org/abs/2412.03603) — HunyuanVideo。
- [Genmo (2024). Mochi-1 Technical Report](https://www.genmo.ai/blog/mochi) — Mochi-1。
- [Alibaba (2025). WAN 2.2](https://wanvideo.io/) — 2025 年中 open SOTA。
- [Ho, Salimans, Gritsenko et al. (2022). Video Diffusion Models](https://arxiv.org/abs/2204.03458) — seminal video diffusion paper。
- [Blattmann et al. (2023). Align your Latents (Video LDM)](https://arxiv.org/abs/2304.08818) — Stable Video Diffusion 的祖先。
