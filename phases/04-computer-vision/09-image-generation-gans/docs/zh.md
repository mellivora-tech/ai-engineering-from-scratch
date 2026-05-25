# 图像生成：GANs

> GAN 是两个神经网络组成的固定博弈。一个画图，一个评论。它们一起变好，直到画出来的东西骗过评论者。

**类型：** 构建
**语言：** Python
**前置要求：** 阶段 4 第 03 课（CNN），阶段 3 第 06 课（优化器），阶段 3 第 07 课（正则化）
**时间：** ~75 分钟

## 学习目标

- 解释 generator 和 discriminator 之间的 minimax game，以及为什么 equilibrium 对应 p_model = p_data
- 用 PyTorch 实现 DCGAN，并在 60 行以内让它生成连贯的 32x32 合成图像
- 用三个标准技巧稳定 GAN 训练：non-saturating loss、spectral norm、TTUR（two-timescale update rule）
- 阅读训练曲线，区分健康收敛与 mode collapse、oscillation、discriminator-wins-completely

## 问题

Classification 教网络把图像映射到标签。Generation 会反转问题：采样看起来像来自同一分布的新图像。没有一个“正确”输出可以让你做 diff；只有一个你想模仿的分布。

标准 loss function（MSE、cross-entropy）无法衡量“这个样本是否来自真实分布”。最小化 per-pixel error 会产生模糊平均值，而不是真实样本。突破点是学习 loss：训练第二个网络，让它负责区分真图和假图，再用它的判断推动 generator。

GAN（Goodfellow 等，2014）定义了这个框架。到 2018 年，StyleGAN 已经能生成与照片难以区分的 1024x1024 人脸。Diffusion model 后来在质量和可控性上登顶，但每个让 diffusion 实用的技巧：normalisation 选择、latent spaces、feature losses，都先在 GAN 上被理解。

## 概念

### 两个网络

```mermaid
flowchart LR
    Z["z ~ N(0, I)<br/>noise"] --> G["Generator<br/>transposed convs"]
    G --> FAKE["Fake image"]
    REAL["Real image"] --> D["Discriminator<br/>conv classifier"]
    FAKE --> D
    D --> OUT["P(real)"]

    style G fill:#dbeafe,stroke:#2563eb
    style D fill:#fef3c7,stroke:#d97706
    style OUT fill:#dcfce7,stroke:#16a34a
```

**Generator** G 接收噪声向量 `z` 并输出图像。**Discriminator** D 接收图像并输出一个标量：这张图像为真的概率。

### 这个博弈

G 希望 D 犯错。D 希望自己正确。形式化地说：

```
min_G max_D  E_x[log D(x)] + E_z[log(1 - D(G(z)))]
```

从右往左读：D 正在最大化它对真图（`log D(real)`）和假图（`log (1 - D(fake))`）的准确性。G 正在最小化 D 对假图的准确性，也就是希望 `D(G(z))` 很高。

Goodfellow 证明了这个 minimax 有一个全局 equilibrium，其中 `p_G = p_data`，D 处处输出 0.5，生成分布和真实分布之间的 Jensen-Shannon divergence 为零。难点是到达那里。

### Non-saturating loss

上面的形式数值不稳定。训练早期，所有假图上的 `D(G(z))` 都接近零，所以 `log(1 - D(G(z)))` 对 G 的梯度会消失。修复方法是翻转 G 的 loss。

```
L_D = -E_x[log D(x)] - E_z[log(1 - D(G(z)))]
L_G = -E_z[log D(G(z))]                          # non-saturating
```

现在当 `D(G(z))` 接近零时，G 的 loss 很大，梯度也有信息。每个现代 GAN 都用这个变体训练。

### DCGAN 架构规则

Radford、Metz、Chintala（2015）把多年失败实验浓缩成五条让 GAN 训练稳定的规则：

1. 用 strided convs 替代 pooling（两个网络都如此）。
2. 在 generator 和 discriminator 中使用 batch norm，但 G 的输出和 D 的输入除外。
3. 在更深架构中移除全连接层。
4. G 在所有层上使用 ReLU，输出层除外（输出用 tanh 到 [-1, 1]）。
5. D 在所有层上使用 LeakyReLU（negative_slope=0.2）。

每个现代 conv-based GAN（StyleGAN、BigGAN、GigaGAN）仍然从这些规则出发，然后一次替换一个部件。

### 失败模式及其信号

```mermaid
flowchart LR
    M1["Mode collapse<br/>G produces a narrow<br/>set of outputs"] --> S1["D loss low,<br/>G loss oscillating,<br/>sample variety drops"]
    M2["Vanishing gradients<br/>D wins completely"] --> S2["D accuracy ~100%,<br/>G loss huge and static"]
    M3["Oscillation<br/>G and D keep trading<br/>wins forever"] --> S3["Both losses swing<br/>wildly with no downward trend"]

    style M1 fill:#fecaca,stroke:#dc2626
    style M2 fill:#fecaca,stroke:#dc2626
    style M3 fill:#fecaca,stroke:#dc2626
```

- **Mode collapse**：G 找到一张能骗过 D 的图，并只生成那张。修复：加入 minibatch discrimination、spectral norm 或 label-conditioning。
- **Discriminator wins**：D 过快变得太强，G 的梯度消失。修复：更小的 D、更低的 D learning rate，或在真实标签上应用 label smoothing。
- **Oscillation**：两个网络轮流胜出，却永远不接近 equilibrium。修复：TTUR（D 学得比 G 快 2-4 倍），或切换到 Wasserstein loss。

### 评估

GAN 没有 ground truth，那你怎么知道它们在工作？

- **Sample inspection**：每个 epoch 末尾直接看 64 个样本。不可协商。
- **FID（Fréchet Inception Distance）**：真实集合与生成集合的 Inception-v3 feature distribution 之间的距离。越低越好。社区标准。
- **Inception Score**：更老、更脆弱；优先用 FID。
- **Precision/Recall for generative models**：分别衡量质量（precision）和覆盖度（recall）。比单独 FID 更有信息量。

对小型合成数据 run，sample inspection 就足够。

## 构建它

### 第 1 步：Generator

一个小 DCGAN generator，接收 64 维噪声并产生 32x32 图像。

```python
import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, z_dim=64, img_channels=3, feat=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(z_dim, feat * 4, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(feat * 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(feat * 4, feat * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feat * 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(feat * 2, feat, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feat),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(feat, img_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z):
        return self.net(z.view(z.size(0), -1, 1, 1))
```

四个 transposed conv，每个使用 `kernel_size=4, stride=2, padding=1`，所以空间尺寸会干净地翻倍。通过 tanh 输出 [-1, 1] 内的 activation。

### 第 2 步：Discriminator

Generator 的镜像。LeakyReLU、strided convs，最后得到一个标量 logit。

```python
class Discriminator(nn.Module):
    def __init__(self, img_channels=3, feat=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(img_channels, feat, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feat, feat * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feat * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feat * 2, feat * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feat * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feat * 4, 1, kernel_size=4, stride=1, padding=0),
        )

    def forward(self, x):
        return self.net(x).view(-1)
```

最后一个 conv 把 `4x4` feature map 降成 `1x1`。输出是每张图一个标量；只在 loss 计算期间应用 sigmoid。

### 第 3 步：Training step

交替进行：每个 batch 先更新一次 D，再更新一次 G。

```python
import torch.nn.functional as F

def train_step(G, D, real, z, opt_g, opt_d, device):
    real = real.to(device)
    bs = real.size(0)

    # D step
    opt_d.zero_grad()
    d_real = D(real)
    d_fake = D(G(z).detach())
    loss_d = (F.binary_cross_entropy_with_logits(d_real, torch.ones_like(d_real))
              + F.binary_cross_entropy_with_logits(d_fake, torch.zeros_like(d_fake)))
    loss_d.backward()
    opt_d.step()

    # G step
    opt_g.zero_grad()
    d_fake = D(G(z))
    loss_g = F.binary_cross_entropy_with_logits(d_fake, torch.ones_like(d_fake))
    loss_g.backward()
    opt_g.step()

    return loss_d.item(), loss_g.item()
```

D step 中的 `G(z).detach()` 至关重要：我们不希望在 D 更新期间梯度流入 G。忘记它是经典新手 bug。

### 第 4 步：在合成形状上完整训练

```python
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

def synthetic_images(num=2000, size=32, seed=0):
    rng = np.random.default_rng(seed)
    imgs = np.zeros((num, 3, size, size), dtype=np.float32) - 1.0
    for i in range(num):
        r = rng.uniform(6, 12)
        cx, cy = rng.uniform(r, size - r, size=2)
        yy, xx = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 < r ** 2
        color = rng.uniform(-0.5, 1.0, size=3)
        for c in range(3):
            imgs[i, c][mask] = color[c]
    return torch.from_numpy(imgs)

device = "cuda" if torch.cuda.is_available() else "cpu"
data = synthetic_images()
loader = DataLoader(TensorDataset(data), batch_size=64, shuffle=True)

G = Generator(z_dim=64, img_channels=3, feat=32).to(device)
D = Discriminator(img_channels=3, feat=32).to(device)
opt_g = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
opt_d = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))

for epoch in range(10):
    for (batch,) in loader:
        z = torch.randn(batch.size(0), 64, device=device)
        ld, lg = train_step(G, D, batch, z, opt_g, opt_d, device)
    print(f"epoch {epoch}  D {ld:.3f}  G {lg:.3f}")
```

`Adam(lr=2e-4, betas=(0.5, 0.999))` 是 DCGAN 默认值；较低的 beta1 会防止 momentum term 过度稳定这个 adversarial game。

### 第 5 步：Sampling

```python
@torch.no_grad()
def sample(G, n=16, z_dim=64, device="cpu"):
    G.eval()
    z = torch.randn(n, z_dim, device=device)
    imgs = G(z)
    imgs = (imgs + 1) / 2
    return imgs.clamp(0, 1)
```

Sampling 前始终切到 eval mode。对 DCGAN，这很重要，因为 batch norm 会使用 running stats，而不是当前 batch 的 stats。

### 第 6 步：Spectral normalisation

Discriminator 中 BN 的 drop-in 替代品，保证网络是 1-Lipschitz。能修复大多数 “D wins too hard” 失败。

```python
from torch.nn.utils import spectral_norm

def build_sn_discriminator(img_channels=3, feat=64):
    return nn.Sequential(
        spectral_norm(nn.Conv2d(img_channels, feat, 4, 2, 1)),
        nn.LeakyReLU(0.2, inplace=True),
        spectral_norm(nn.Conv2d(feat, feat * 2, 4, 2, 1)),
        nn.LeakyReLU(0.2, inplace=True),
        spectral_norm(nn.Conv2d(feat * 2, feat * 4, 4, 2, 1)),
        nn.LeakyReLU(0.2, inplace=True),
        spectral_norm(nn.Conv2d(feat * 4, 1, 4, 1, 0)),
    )
```

把 `Discriminator` 换成 `build_sn_discriminator()`，你常常就不需要 TTUR trick。Spectral norm 是你能应用的最简单单项 robustness 升级。

## 使用它

对严肃生成任务，使用预训练权重，或者切换到 diffusion。两个标准库：

- `torch_fidelity` 可以为你的 generator 计算 FID / IS，无需写自定义 eval 代码。
- `pytorch-gan-zoo`（legacy）和 `StudioGAN` 提供经过测试的 DCGAN、WGAN-GP、SN-GAN、StyleGAN 和 BigGAN 实现。

2026 年，GAN 仍然是这些任务的最佳选择：实时图像生成（latency <10 ms）、style transfer、带精确控制的 image-to-image translation（Pix2Pix、CycleGAN）。Diffusion 在 photorealism 和 text conditioning 上胜出。

## 交付它

本课会产出：

- `outputs/prompt-gan-training-triage.md`：一个 prompt，会读取训练曲线描述，选择失败模式（mode collapse、D-wins、oscillation），并给出一个推荐修复。
- `outputs/skill-dcgan-scaffold.md`：一个 skill，会根据 `z_dim`、目标 `image_size` 和 `num_channels` 写出 DCGAN scaffold，包括 training loop 和 sample saver。

## 练习

1. **（简单）** 在上面的合成圆形数据集上训练 DCGAN，并在每个 epoch 结束保存 16 个样本的 grid。到第几个 epoch，生成的圆开始明显像圆？
2. **（中等）** 用 spectral norm 替换 discriminator 的 batch norm。并排训练两个版本。哪个收敛更快？哪个在三个 seed 上方差更低？
3. **（困难）** 实现 conditional DCGAN：把 class label 喂给 G 和 D（在 G 中把 one-hot concat 到噪声，在 D 中 concat class embedding channel）。在第 7 课的合成 “circles vs squares” 数据集上训练，并通过按指定 label sampling 展示 class conditioning 有效。

## 关键术语

| 术语 | 人们常说 | 它实际意味着 |
|------|----------------|----------------------|
| Generator (G) | “画东西的网络” | 把噪声映射到图像；训练目标是骗过 discriminator |
| Discriminator (D) | “Critic” | 二分类器；训练目标是区分真实图像和生成图像 |
| Minimax | “博弈” | 对 adversarial loss 关于 G 求 min、关于 D 求 max；equilibrium 是 p_G = p_data |
| Non-saturating loss | “数值上靠谱的版本” | G 的 loss 是 -log(D(G(z)))，而不是 log(1 - D(G(z)))，避免训练早期梯度消失 |
| Mode collapse | “Generator 只做一种东西” | G 只产生数据分布的一小部分；用 SN、minibatch discrimination 或更大 batch 修复 |
| TTUR | “两个 learning rate” | D 比 G 学得更快，通常快 2-4 倍；稳定训练 |
| Spectral norm | “1-Lipschitz layer” | 一种 weight-normalisation，限制每层的 Lipschitz constant；阻止 D 变得任意陡峭 |
| FID | “Fréchet Inception Distance” | 真实集合和生成集合的 Inception-v3 feature distribution 之间的距离；标准评估指标 |

## 延伸阅读

- [Generative Adversarial Networks (Goodfellow et al., 2014)](https://arxiv.org/abs/1406.2661)：开启这一切的论文
- [DCGAN (Radford, Metz, Chintala, 2015)](https://arxiv.org/abs/1511.06434)：让 GAN 可训练的架构规则
- [Spectral Normalization for GANs (Miyato et al., 2018)](https://arxiv.org/abs/1802.05957)：最有用的单个稳定化技巧
- [StyleGAN3 (Karras et al., 2021)](https://arxiv.org/abs/2106.12423)：SOTA GAN；读起来像过去十年所有技巧的精选集
