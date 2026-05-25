# 3D 视觉：Point Clouds 与 NeRFs

> 3D 视觉有两种味道。Point clouds 是传感器的 raw output。NeRFs 是学到的 volumetric field。两者都回答“空间中哪里有什么”。

**类型：** 学习 + 构建
**语言：** Python
**前置要求：** 阶段 4 第 03 课（CNN），阶段 1 第 12 课（张量操作）
**时间：** ~45 分钟

## 学习目标

- 区分 explicit（point cloud、mesh、voxel）和 implicit（signed distance field、NeRF）3D representations，并说明各自何时使用
- 理解 PointNet 的 symmetric-function 技巧，它让神经网络对无序点集 permutation-invariant
- 追踪一次 NeRF forward pass：ray casting、volumetric rendering、positional encoding、MLP density+colour head
- 使用 `nerfstudio` 或 `instant-ngp`，从少量带 pose 图像中做预训练 3D reconstruction

## 问题

相机会产生 2D 图像。LIDAR 会产生一组没有顺序的 3D points。Structure-from-motion pipeline 会产生稀疏的 3D keypoint cloud。NeRF 会从少量带 pose 图像中重建完整 3D 场景。这些都是“视觉”，但没有一个看起来像 CNN 想要的 dense tensor。

3D 视觉重要，是因为几乎每个高价值机器人任务都在 3D 中运行：grasping、obstacle avoidance、navigation、AR occlusion、3D content capture。只理解 2D 图像的视觉工程师，会被锁在这个领域增长最快的一块之外（AR/VR content、robotics、autonomous driving stacks、基于 NeRF 的房地产或施工 3D reconstruction）。

两种 representation 因不同原因占主导。Point clouds 是传感器免费给你的东西。NeRFs 及其后继者（3D Gaussian splatting、neural SDFs）是你让神经网络学习一个场景时得到的东西。

## 概念

### Point clouds

Point cloud 是 R^3 中 N 个点的无序集合，每个点可选带 features（colour、intensity、normal）。

```
cloud = [
  (x1, y1, z1, r1, g1, b1),
  (x2, y2, z2, r2, g2, b2),
  ...
  (xN, yN, zN, rN, gN, bN),
]
```

没有 grid，没有 connectivity。两个性质让它对神经网络很难：

- **Permutation invariance**：输出不能依赖点的顺序。
- **Variable N**：一个模型必须处理不同大小的 clouds。

PointNet（Qi 等，2017）用一个想法解决了两者：对每个点应用共享 MLP，然后用 symmetric function（max pool）聚合。结果是一个不依赖顺序的固定大小向量。

```
f(P) = max_{p in P} MLP(p)
```

这就是 PointNet 的完整核心。更深的变体（PointNet++、Point Transformer）会添加 hierarchical sampling 和 local aggregation，但 symmetric-function 技巧不变。

### PointNet 架构

```mermaid
flowchart LR
    PTS["N points<br/>(x, y, z)"] --> MLP1["shared MLP<br/>(64, 64)"]
    MLP1 --> MLP2["shared MLP<br/>(64, 128, 1024)"]
    MLP2 --> MAX["max pool<br/>(symmetric)"]
    MAX --> FEAT["global feature<br/>(1024,)"]
    FEAT --> FC["MLP classifier"]
    FC --> CLS["class logits"]

    style MLP1 fill:#dbeafe,stroke:#2563eb
    style MAX fill:#fef3c7,stroke:#d97706
    style CLS fill:#dcfce7,stroke:#16a34a
```

“Shared MLP” 意味着同一个 MLP 独立运行在每个点上。为了效率，通常实现为沿 point dimension 的 1x1 conv。

### Neural Radiance Fields（NeRFs）

NeRFs（Mildenhall 等，2020）把“我们能否从 N 张照片重建 3D 场景？”这个问题回答为：一个神经网络就是这个场景。网络把 `(x, y, z, viewing_direction)` 映射到 `(density, colour)`。渲染一个新视角，就是在这个网络上做 ray-casting loop。

```
NeRF MLP:  (x, y, z, theta, phi) -> (sigma, r, g, b)

To render a pixel (u, v) of a new view:
  1. Cast a ray from the camera through pixel (u, v)
  2. Sample points along the ray at distances t_1, t_2, ..., t_N
  3. Query the MLP at each point
  4. Composite the colours weighted by (1 - exp(-sigma * dt))
  5. The sum is the rendered pixel colour
```

Loss 会把 rendered pixel 与训练照片中的 ground-truth pixel 比较。通过 rendering step 反向传播来更新 MLP。没有 3D ground truth，没有 explicit geometry；场景存储在 MLP weights 中。

### NeRF 中的 positional encoding

只在 `(x, y, z)` 上运行的普通 MLP 无法表示高频细节，因为 MLP 有偏向低频的 spectral bias。NeRF 通过在进入 MLP 前把每个坐标编码成 Fourier feature vector 来修复：

```
gamma(p) = (sin(2^0 pi p), cos(2^0 pi p), sin(2^1 pi p), cos(2^1 pi p), ...)
```

最高 L=10 个频率 level。这和 transformer 用于 position 的技巧相同，也会在 diffusion time conditioning 中再次出现（第 10 课）。没有它，NeRF 会显得模糊。

### Volumetric rendering

```
C(r) = sum_i T_i * (1 - exp(-sigma_i * delta_i)) * c_i

T_i  = exp(- sum_{j<i} sigma_j * delta_j)
delta_i = t_{i+1} - t_i
```

`T_i` 是 transmittance，也就是有多少光能到达点 i。`(1 - exp(-sigma_i * delta_i))` 是点 i 处的 opacity。`c_i` 是颜色。最终像素是沿 ray 的加权和。

### 什么替代了 NeRFs

纯 NeRF 训练慢（小时级），渲染也慢（每张图数秒）。后续谱系：

- **Instant-NGP**（2022）：hash-grid encoding 替换 MLP 的位置输入；数秒内训练。
- **Mip-NeRF 360**：处理 unbounded scenes 和 anti-aliasing。
- **3D Gaussian Splatting**（2023）：用数百万个 3D Gaussians 替代 volumetric field；分钟级训练，实时渲染。当前生产默认。

2026 年几乎每个真实 NeRF 产品其实都是 3D Gaussian splatting。心智模型仍然是 NeRF。

### 数据集和 benchmarks

- **ShapeNet**：以 point cloud 表示的 3D CAD models 分类和 segmentation。
- **ScanNet**：用于 segmentation 的真实室内 scans。
- **KITTI**：自动驾驶的户外 LIDAR point clouds。
- **NeRF Synthetic** / **Blended MVS**：用于 view synthesis 的带 pose 图像数据集。
- **Mip-NeRF 360** dataset：unbounded real scenes。

## 构建它

### 第 1 步：PointNet classifier

```python
import torch
import torch.nn as nn

class PointNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.mlp1 = nn.Sequential(
            nn.Conv1d(3, 64, 1),    nn.BatchNorm1d(64),   nn.ReLU(inplace=True),
            nn.Conv1d(64, 64, 1),   nn.BatchNorm1d(64),   nn.ReLU(inplace=True),
        )
        self.mlp2 = nn.Sequential(
            nn.Conv1d(64, 128, 1),  nn.BatchNorm1d(128),  nn.ReLU(inplace=True),
            nn.Conv1d(128, 1024, 1), nn.BatchNorm1d(1024), nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.Linear(1024, 512),   nn.BatchNorm1d(512),  nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),    nn.BatchNorm1d(256),  nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        # x: (N, 3, num_points) — transposed for Conv1d
        x = self.mlp1(x)
        x = self.mlp2(x)
        x = torch.max(x, dim=-1)[0]       # (N, 1024)
        return self.head(x)

pts = torch.randn(4, 3, 1024)
net = PointNet(num_classes=10)
print(f"output: {net(pts).shape}")
print(f"params: {sum(p.numel() for p in net.parameters()):,}")
```

约 160 万参数。每个 cloud 上运行 1,024 个点。

### 第 2 步：Positional encoding

```python
def positional_encoding(x, L=10):
    """
    x: (..., D) -> (..., D * 2 * L)
    """
    freqs = 2.0 ** torch.arange(L, dtype=x.dtype, device=x.device)
    args = x.unsqueeze(-1) * freqs * 3.141592653589793
    sinc = torch.cat([args.sin(), args.cos()], dim=-1)
    return sinc.reshape(*x.shape[:-1], -1)

x = torch.randn(5, 3)
y = positional_encoding(x, L=10)
print(f"input:  {x.shape}")
print(f"encoded: {y.shape}     # (5, 60)")
```

乘以 `2^l * pi` 会给出逐渐更高的频率。

### 第 3 步：Tiny NeRF MLP

```python
class TinyNeRF(nn.Module):
    def __init__(self, L_pos=10, L_dir=4, hidden=128):
        super().__init__()
        self.L_pos = L_pos
        self.L_dir = L_dir
        pos_dim = 3 * 2 * L_pos
        dir_dim = 3 * 2 * L_dir
        self.trunk = nn.Sequential(
            nn.Linear(pos_dim, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),  nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),  nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),  nn.ReLU(inplace=True),
        )
        self.sigma = nn.Linear(hidden, 1)
        self.color = nn.Sequential(
            nn.Linear(hidden + dir_dim, hidden // 2), nn.ReLU(inplace=True),
            nn.Linear(hidden // 2, 3), nn.Sigmoid(),
        )

    def forward(self, x, d):
        x_enc = positional_encoding(x, self.L_pos)
        d_enc = positional_encoding(d, self.L_dir)
        h = self.trunk(x_enc)
        sigma = torch.relu(self.sigma(h)).squeeze(-1)
        rgb = self.color(torch.cat([h, d_enc], dim=-1))
        return sigma, rgb

nerf = TinyNeRF()
x = torch.randn(128, 3)
d = torch.randn(128, 3)
s, c = nerf(x, d)
print(f"sigma: {s.shape}   rgb: {c.shape}")
```

相比原始 NeRF 很小（原始 NeRF 有两个深度为 8 的 MLP trunks）。足够展示架构。

### 第 4 步：沿 ray 做 volumetric rendering

```python
def volumetric_render(sigma, rgb, t_vals):
    """
    sigma: (..., N_samples)
    rgb:   (..., N_samples, 3)
    t_vals: (N_samples,) distances along the ray
    """
    delta = torch.cat([t_vals[1:] - t_vals[:-1], torch.full_like(t_vals[:1], 1e10)])
    alpha = 1.0 - torch.exp(-sigma * delta)
    trans = torch.cumprod(torch.cat([torch.ones_like(alpha[..., :1]), 1.0 - alpha + 1e-10], dim=-1), dim=-1)[..., :-1]
    weights = alpha * trans
    rendered = (weights.unsqueeze(-1) * rgb).sum(dim=-2)
    depth = (weights * t_vals).sum(dim=-1)
    return rendered, depth, weights


N = 64
t_vals = torch.linspace(2.0, 6.0, N)
sigma = torch.rand(N) * 0.5
rgb = torch.rand(N, 3)
rendered, depth, weights = volumetric_render(sigma, rgb, t_vals)
print(f"rendered colour: {rendered.tolist()}")
print(f"depth:           {depth.item():.2f}")
```

一条 ray，64 个 samples，composite 成单个 RGB pixel 和 depth。

## 使用它

真实工作中：

- `nerfstudio`（Tancik 等）：当前 NeRF / Instant-NGP / Gaussian Splatting 参考库。命令行加 web viewer。
- `pytorch3d`（Meta）：differentiable rendering、point-cloud utilities、mesh ops。
- `open3d`：point cloud processing、registration、visualisation。

部署时，3D Gaussian splatting 已经很大程度上替代纯 NeRF，因为它渲染快 100 倍。Reconstruction quality 相当。

## 交付它

本课会产出：

- `outputs/prompt-3d-task-router.md`：一个 prompt，会根据任务和输入数据路由到正确 3D representation（point cloud、mesh、voxel、NeRF、Gaussian splat）。
- `outputs/skill-point-cloud-loader.md`：一个 skill，会为 .ply / .pcd / .xyz 文件写 PyTorch `Dataset`，包含正确 normalisation、centring 和 point sampling。

## 练习

1. **（简单）** 展示 PointNet 是 permutation-invariant：同一个 cloud 运行两次，一次打乱点顺序。验证输出只在浮点噪声范围内有差异。
2. **（中等）** 实现一个最小 ray-generation 函数：给定 camera intrinsics 和 pose，为 H x W 图像的每个像素产生 ray origins 和 directions。
3. **（困难）** 在一个彩色立方体渲染视图的合成数据集上训练 TinyNeRF（通过 differentiable rendering 或简单 ray tracer 生成）。报告 epoch 1、10、100 的 rendering loss。到第几个 epoch，模型能产生可识别视图？

## 关键术语

| 术语 | 人们常说 | 它实际意味着 |
|------|----------------|----------------------|
| Point cloud | “LIDAR 的 3D 点” | 无序的 (x, y, z) + optional features 点集合 |
| PointNet | “第一个 point cloud 神经网络” | 每点共享 MLP + symmetric (max) pool；构造上就是 permutation-invariant |
| NeRF | “作为场景的 MLP” | 把 (x, y, z, dir) 映射到 (density, colour) 的网络；通过 ray casting 渲染 |
| Positional encoding | “Fourier features” | 把每个坐标编码成多个频率上的 sin/cos，以克服 MLP 低频偏置 |
| Volumetric rendering | “Ray integration” | 使用 transmittance 和 alpha，把 ray 上的 samples composite 成单个像素 |
| Instant-NGP | “Hash-grid NeRF” | 用 multi-resolution hash grid 替代 NeRF 的 coordinate MLP；快 100-1000 倍 |
| 3D Gaussian splatting | “数百万个 Gaussians” | 场景 = 一组 3D Gaussians；实时渲染，分钟级训练 |
| SDF | “Signed distance field” | 返回到最近表面有符号距离的函数；另一种 implicit representation |

## 延伸阅读

- [PointNet (Qi et al., 2017)](https://arxiv.org/abs/1612.00593)：permutation-invariant classifier
- [NeRF (Mildenhall et al., 2020)](https://arxiv.org/abs/2003.08934)：把从照片重建 3D 变成 neural-net 问题的论文
- [Instant-NGP (Müller et al., 2022)](https://arxiv.org/abs/2201.05989)：hash grids，1000 倍加速
- [3D Gaussian Splatting (Kerbl et al., 2023)](https://arxiv.org/abs/2308.04079)：在生产中替代 NeRF 的架构
