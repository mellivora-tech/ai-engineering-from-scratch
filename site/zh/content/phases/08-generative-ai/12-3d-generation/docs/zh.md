# 3D Generation

> 3D 是 2D-to-3D leverage 最强的模态。2023 年突破是 3D Gaussian Splatting。2024–2026 年的生成推进，是在上面叠 multi-view diffusion + 3D reconstruction，从单个 prompt 或 photo 产生 objects 和 scenes。

**类型：** 学习
**语言：** Python
**前置要求：** 阶段 4（Vision），阶段 8 · 07（Latent Diffusion）
**时间：** ~45 分钟

## 问题

3D 内容很痛苦：

- **Representation。** Meshes、point clouds、voxel grids、signed distance fields（SDFs）、neural radiance fields（NeRFs）、3D Gaussians。每种都有 trade-offs。
- **Data scarcity。** ImageNet 有 14M images。最大的干净 3D dataset（Objaverse-XL，2023）约 10M objects，大多质量不高。
- **Memory。** 512³ voxel grid 是 128M voxels；有用的 scene NeRF 每条 ray 需要 1M samples。Generation 比 reconstruction 更难。
- **Supervision。** 对 2D 图像，你有 pixels。对 3D，你通常只有少数 2D views，并且必须 lift 到 3D。

2026 年 stack 把两个问题分开。先用 diffusion model 生成 *2D multi-view images*。再把一个 *3D representation*（通常是 Gaussian splatting）fit 到这些图像上。

## 概念

![3D generation: multi-view diffusion + 3D reconstruction](../assets/3d-generation.svg)

### Representation：3D Gaussian Splatting（Kerbl et al., 2023）

把一个 scene 表示为约 1M 个 3D Gaussians 的 cloud。每个有 59 个参数：position（3）、covariance（6，或 quaternion 4 + scale 3）、opacity（1）、spherical-harmonics color（degree 3 时 48，degree 0 时 3）。

Rendering = projection + alpha-compositing。很快（4090 上 1080p 约 100 fps）。可微。通过对 ground-truth photos 做 gradient descent 来 fit。一个 scene 在消费级 GPU 上 5-30 分钟可 fit。

其上的两个 2023–2024 创新：
- **Generative Gaussian splats。** LGM、LRM、InstantMesh 这类模型直接从一张或少数图像预测 Gaussian cloud。
- **4D Gaussian Splatting。** 带 per-frame offsets 的 Gaussians，用于 dynamic scenes。

### Multi-view diffusion

Fine-tune 一个 pretrained image diffusion model，让它从 text prompt 或单张图像生成同一物体的多个一致视角。Zero123（Liu et al., 2023）、MVDream（Shi et al., 2023）、SV3D（Stability, 2024）、CAT3D（Google, 2024）。通常围绕物体输出 4-16 个 views，再通过 Gaussian splatting 或 NeRF lift 到 3D。

### Text-to-3D pipelines

| 模型 | 输入 | 输出 | 时间 |
|-------|-------|--------|------|
| DreamFusion (2022) | text | NeRF via SDS | ~1 hour per asset |
| Magic3D | text | mesh + texture | ~40 min |
| Shap-E (OpenAI, 2023) | text | implicit 3D | ~1 min |
| SJC / ProlificDreamer | text | NeRF / mesh | ~30 min |
| LRM (Meta, 2023) | image | triplane | ~5 s |
| InstantMesh (2024) | image | mesh | ~10 s |
| SV3D (Stability, 2024) | image | novel views | ~2 min |
| CAT3D (Google, 2024) | 1-64 images | 3D NeRF | ~1 min |
| TripoSR (2024) | image | mesh | ~1 s |
| Meshy 4 (2025) | text + image | PBR mesh | ~30 s |
| Rodin Gen-1.5 (2025) | text + image | PBR mesh | ~60 s |
| Tencent Hunyuan3D 2.0 (2025) | image | mesh | ~30 s |

2025–2026 方向：直接 text-to-mesh models，带适用于 game engines 的 PBR materials。Multi-view diffusion intermediate step 仍然是 general objects 上表现最好的配方。

### NeRF（背景）

Neural Radiance Field（Mildenhall et al., 2020）。一个 tiny MLP 接收 `(x, y, z, view direction)` 并输出 `(color, density)`。通过沿 rays 积分渲染。质量上胜过 mesh-based novel-view synthesis，但渲染慢 100-1000x。大多数实时用途中已经被 Gaussian splatting 取代，但研究中仍占主导。

## 构建它

`code/main.py` 实现一个 toy 2D “Gaussian splatting” fit：把 synthetic target image（平滑 gradient）表示为一组 2D Gaussian splats。通过 gradient descent 优化 positions、colors 和 covariances 来匹配 target。你会看到两个核心操作：forward render（splat + alpha-composite）和通过 gradient descent fit。

### 第 1 步：2D Gaussian splat

```python
def gaussian_at(x, y, gaussian):
    px, py = gaussian["pos"]
    sigma = gaussian["sigma"]
    d2 = (x - px) ** 2 + (y - py) ** 2
    return math.exp(-d2 / (2 * sigma * sigma))
```

### 第 2 步：通过 summing splats 渲染

```python
def render(image_size, gaussians):
    img = [[0.0] * image_size for _ in range(image_size)]
    for g in gaussians:
        for y in range(image_size):
            for x in range(image_size):
                img[y][x] += g["color"] * gaussian_at(x, y, g)
    return img
```

真实 3D Gaussian splatting 会按深度排序 Gaussians，并按顺序 alpha-composite。我们的 2D toy 只是求和。

### 第 3 步：通过 gradient descent fit

```python
for step in range(steps):
    pred = render(size, gaussians)
    loss = mse(pred, target)
    gradients = compute_grads(pred, target, gaussians)
    update(gaussians, gradients, lr)
```

## 陷阱

- **View inconsistency。** 如果你独立生成 4 个 views，而它们对物体结构意见不一致，3D fit 会模糊。修复：带 shared attention 的 multi-view diffusion。
- **Back-side hallucination。** Single-image → 3D 必须想象看不见的背面。质量波动很大。
- **Gaussian splat explosion。** 无约束训练会长到 10M splats 并过拟合。Densification + pruning heuristics（来自原始 3D-GS 论文）必不可少。
- **Topology issues。** 从 implicit fields（SDFs）得到的 meshes 经常有洞或自交。上线前运行 remesher（例如 blender 的 voxel remesh）。
- **License of training data。** Objaverse licenses 混杂；商业使用因模型而异。

## 使用它

| 任务 | 2026 选择 |
|------|-----------|
| 从照片做 scene reconstruction | Gaussian splatting（3DGS、Gsplat、Scaniverse） |
| 游戏用 text-to-3D object | Meshy 4 或 Rodin Gen-1.5（PBR output） |
| Image-to-3D | Hunyuan3D 2.0、TripoSR、InstantMesh |
| Few images 的 novel-view synthesis | CAT3D、SV3D |
| Dynamic scene reconstruction | 4D Gaussian Splatting |
| Avatar / clothed human | Gaussian Avatar、HUGS |
| Research / SOTA | 上周刚发的那个 |

对游戏或电商 pipeline 中的生产 3D：Meshy 4 或 Rodin Gen-1.5 输出可直接进 Unity / Unreal 的 PBR meshes。

## 交付它

保存 `outputs/skill-3d-pipeline.md`。Skill 接收 3D brief（input：text / one image / few images；output：mesh / splat / NeRF；usage：render / game / VR），并输出：pipeline（multi-view diffusion + fit，或 direct mesh model）、base model、iteration budget、topology post-processing、material channels needed。

## 练习

1. **简单。** 用 4、16、64 个 Gaussians 运行 `code/main.py`。报告相对 target 的 final MSE。
2. **中等。** 扩展到 color Gaussians（RGB）。确认 reconstruction 匹配 target color pattern。
3. **困难。** 使用 gsplat 或 Nerfstudio，从 50-photo capture 重建真实物体。报告 fit time 和 held-out views 上的 final SSIM。

## 关键词

| 术语 | 大家怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| 3D Gaussian Splatting | “3DGS” | 把 scene 表示为 3D Gaussians cloud；可微 alpha-composite render。 |
| NeRF | “Neural radiance field” | 在 3D point 输出 color + density 的 MLP；通过 ray integration 渲染。 |
| Triplane | “三张 2-D planes” | 把 3D factor 成三张 axis-aligned 2-D feature grids；比 volumetric 便宜。 |
| SDS | “Score distillation sampling” | 使用 2D-diffusion score 作为 pseudo-gradient 来训练 3D model。 |
| Multi-view diffusion | “一次多个 views” | 输出一批一致 camera views 的 diffusion model。 |
| PBR | “Physically-based rendering” | 带 albedo、roughness、metallic、normal channels 的 material。 |
| Densification | “Grow splats” | 3DGS 训练 heuristic：在高梯度区域 split / clone splats。 |

## 生产备注：3D 还没有共享底座

不同于图像（latent diffusion + DiT）和视频（spatiotemporal DiT），2026 年 3D 没有单一主导 runtime。生产决策树会按 representation 分叉：

- **NeRF / triplane。** 推理是 ray-marching + 每个 sample 做一次 MLP forward。512² render 需要数百万 MLP forwards。积极 batch ray samples；SDPA/xformers 适用。
- **Multi-view diffusion + LRM reconstruction。** Two-stage pipeline。Stage 1（multi-view DiT）就是类似第 07 课的 diffusion server。Stage 2（LRM transformer）是对 views 的一次 one-shot forward pass。总体延迟画像是 “diffusion + one-shot” — 需要按 stage 选择 serving primitives。
- **SDS / DreamFusion。** Per-asset optimization，不是 inference。构建 jobs，而不是 request handlers。

对大多数 2026 产品，正确答案是“按请求运行 multi-view diffusion model，异步 reconstruct 到 3DGS，再 serve 3DGS 做实时查看”。这把 workload 清晰分成 GPU-inference server（快）和 offline optimizer（慢）。

## 延伸阅读

- [Mildenhall et al. (2020). NeRF: Representing Scenes as Neural Radiance Fields](https://arxiv.org/abs/2003.08934) — NeRF。
- [Kerbl et al. (2023). 3D Gaussian Splatting for Real-Time Radiance Field Rendering](https://arxiv.org/abs/2308.04079) — 3DGS。
- [Poole et al. (2022). DreamFusion: Text-to-3D using 2D Diffusion](https://arxiv.org/abs/2209.14988) — SDS。
- [Liu et al. (2023). Zero-1-to-3: Zero-shot One Image to 3D Object](https://arxiv.org/abs/2303.11328) — Zero123。
- [Shi et al. (2023). MVDream](https://arxiv.org/abs/2308.16512) — multi-view diffusion。
- [Hong et al. (2023). LRM: Large Reconstruction Model for Single Image to 3D](https://arxiv.org/abs/2311.04400) — LRM。
- [Gao et al. (2024). CAT3D: Create Anything in 3D with Multi-View Diffusion Models](https://arxiv.org/abs/2405.10314) — CAT3D。
- [Stability AI (2024). Stable Video 3D (SV3D)](https://stability.ai/research/sv3d) — SV3D。
