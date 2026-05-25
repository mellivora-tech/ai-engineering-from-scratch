# 世界模型与视频扩散

> 一个能预测场景接下来几秒的视频模型，就是世界模拟器。把这个预测建立在动作条件上，你就得到了一个学出来的游戏引擎。

**类型：** 学习 + 构建
**语言：** Python
**前置要求：** 阶段 4 第 10 课（Diffusion），阶段 4 第 12 课（Video Understanding），阶段 4 第 23 课（DiT + Rectified Flow）
**时间：** ~75 分钟

## 学习目标

- 解释纯视频生成模型（Sora 2）和 action-conditioned world model（Genie 3、DreamerV3）的区别
- 描述 video DiT：spatio-temporal patches、3D position encoding、跨 `(T, H, W)` tokens 的 joint attention
- 追踪 world model 如何接入机器人：VLM plans → video model simulates → inverse dynamics emits actions
- 为给定用例（creative video、interactive sim、autonomous-driving synthesis）在 Sora 2、Genie 3、Runway GWM-1 Worlds、Wan-Video 和 HunyuanVideo 之间选择

## 问题

视频生成和世界建模在 2026 年收敛了。一个能生成连贯一分钟视频的模型，在某种意义上已经学会了世界如何运动：object permanence、gravity、causality、style。如果你用 actions（向左走、打开门）来条件化这个预测，视频模型就会变成一个可学习的模拟器，可以替代游戏引擎、驾驶模拟器或机器人环境。

影响是具体的。Genie 3 从单张图片生成可玩的环境。Runway GWM-1 Worlds 合成无限可探索场景。Sora 2 生成带同步音频和建模物理的一分钟视频。NVIDIA Cosmos-Drive、Wayve Gaia-2 和 Tesla DrivingWorld 生成真实驾驶视频，用作自动驾驶训练数据。World-model paradigm 正在悄悄接管机器人里的 sim-to-real。

本课是阶段 4 的“大图景”课。它把图像生成、视频理解和 agentic reasoning 连接到一个 dominant research 正在走向的架构模式。

## 概念

### 三类 world-modelling

```mermaid
flowchart LR
    subgraph GEN["Pure video generation"]
        G1["Text / image prompt"] --> G2["Video DiT"] --> G3["Video frames"]
    end
    subgraph ACTION["Action-conditioned world model"]
        A1["Past frames + action"] --> A2["Latent-action video DiT"] --> A3["Next frames"]
        A3 --> A1
    end
    subgraph RL["World models for RL (DreamerV3)"]
        R1["State + action"] --> R2["Latent transition model"] --> R3["Next latent + reward"]
        R3 --> R1
    end

    style GEN fill:#dbeafe,stroke:#2563eb
    style ACTION fill:#fef3c7,stroke:#d97706
    style RL fill:#dcfce7,stroke:#16a34a
```

- **Sora 2** 是由 prompts 条件化的纯视频生成。没有 action interface。你不能在 rollout 中途“转向”它。
- **Genie 3**、**GWM-1 Worlds**、**Mirage / Magica** 是 action-conditioned world models。它们从观测视频中推断 latent actions，再根据 actions 条件化未来帧预测。它们是交互式的：你按键或移动相机，场景会响应。
- **DreamerV3** 和经典 RL world-model 家族在 latent space 中显式用 actions 条件化预测，并基于 reward signal 训练。视觉更弱；对 sample-efficient RL 更有用。

### Video DiT 架构

```
Video latent:          (C, T, H, W)
Patchify (spatial):    grid of P_h x P_w patches per frame
Patchify (temporal):   group P_t frames into a temporal patch
Resulting tokens:      (T / P_t) * (H / P_h) * (W / P_w) tokens
```

Positional encoding 是 3D 的：每个 `(t, h, w)` 坐标都有 rotary 或 learned embedding。Attention 可以是：

- **Full joint**：所有 tokens attend 到所有 tokens。N tokens 时 O(N^2)。长视频中不可承受。
- **Divided**：交替做 temporal attention（同一空间位置，跨时间：`(H*W) * T^2`）和 spatial attention（同一时刻，跨空间：`T * (H*W)^2`）。TimeSformer 和多数 video DiTs 使用。
- **Window**：在 `(t, h, w)` 中做 local windows。Video Swin 使用。

每个 2026 video diffusion model 都使用这三种模式之一，加上 AdaLN conditioning（第 23 课）和 rectified flow。

### 用 actions 条件化：latent action models

Genie 通过判别式地预测相邻两帧之间的 action，为每一帧学习一个 **latent action**。模型的 decoder 随后根据推断出的 latent action 条件化，而不是根据显式键盘按键。推理时，用户可以指定一个 latent action（或从 fresh prior 中 sample 一个），模型会生成与该 action 一致的下一帧。

Sora 完全跳过 action interface。它的 decoder 从过去 spacetime tokens 预测下一个 spacetime tokens。Prompt 条件化开始；生成中途没有东西可以 steer 它。

### Physical plausibility

Sora 2 的 2026 发布明确宣传了 **physical plausibility**：weight、balance、object permanence、cause-and-effect。团队通过人工评分 plausibility scores 测量；相比 Sora 1，模型在掉落物体、角色碰撞，以及故意失败（跳跃失败）上明显改善。

Plausibility 仍是主导失败模式。2024-2025 年人们吃意面或用杯子喝水的视频暴露了模型缺乏持久 object representation。2026 模型（Sora 2、Runway Gen-5、HunyuanVideo）减少了这些问题，但没有消除。

### 自动驾驶 world models

Driving world models 根据 trajectories、bounding boxes 或 navigation maps 条件化生成真实道路场景。用途：

- **Cosmos-Drive-Dreams**（NVIDIA）：为 RL training 生成数分钟驾驶视频。
- **Gaia-2**（Wayve）：用于 policy evaluation 的 trajectory-conditioned scene synthesis。
- **DrivingWorld**（Tesla）：模拟多样天气、时间和交通条件。
- **Vista**（ByteDance）：reactive driving scene synthesis。

它们替代了昂贵真实数据采集，用来覆盖 corner cases：夜间行人乱穿马路、结冰路口、少见车辆类型。这些情况否则需要数百万英里驾驶。

### 机器人栈：VLM + video model + inverse dynamics

正在出现的三组件机器人循环：

1. **VLM** 解析目标（"pick up the red cup"），规划高层动作序列。
2. **Video generation model** 模拟执行每个动作会是什么样子，预测未来 N 帧 observation。
3. **Inverse dynamics model** 抽取能产生这些 observations 的具体 motor commands。

这替代了 reward shaping 和样本消耗巨大的 RL。World model 负责想象；inverse dynamics 闭合 actuation 回路。Genie Envisioner 是一个实例；很多研究组都在收敛到这个结构。

### 评估

- **Visual quality**：FVD（Fréchet Video Distance）、user studies。
- **Prompt alignment**：逐帧 CLIPScore、VQA-style evaluation。
- **Physical plausibility**：在 benchmark suite 上人工评分（Sora 2 内部 benchmark、VBench）。
- **Controllability**（交互式 world models）：action → observation consistency；你能否回到先前状态？

### 2026 年模型版图

| Model | Use | Parameters | Output | License |
|-------|-----|------------|--------|---------|
| Sora 2 | text-to-video, audio | — | 1-min 1080p + audio | API only |
| Runway Gen-5 | text/image-to-video | — | 10s clips | API |
| Runway GWM-1 Worlds | interactive world | — | infinite 3D rollout | API |
| Genie 3 | interactive world from image | 11B+ | playable frames | research preview |
| Wan-Video 2.1 | open text-to-video | 14B | high-quality clips | non-commercial |
| HunyuanVideo | open text-to-video | 13B | 10s clips | permissive |
| Cosmos / Cosmos-Drive | autonomous driving sim | 7-14B | driving scenes | NVIDIA open |
| Magica / Mirage 2 | AI-native game engine | — | modifiable worlds | product |

## 构建它

### 第 1 步：视频 3D patchify

```python
import torch
import torch.nn as nn


class VideoPatch3D(nn.Module):
    def __init__(self, in_channels=4, dim=64, patch_t=2, patch_h=2, patch_w=2):
        super().__init__()
        self.proj = nn.Conv3d(
            in_channels, dim,
            kernel_size=(patch_t, patch_h, patch_w),
            stride=(patch_t, patch_h, patch_w),
        )
        self.patch_t = patch_t
        self.patch_h = patch_h
        self.patch_w = patch_w

    def forward(self, x):
        # x: (N, C, T, H, W)
        x = self.proj(x)
        n, c, t, h, w = x.shape
        tokens = x.reshape(n, c, t * h * w).transpose(1, 2)
        return tokens, (t, h, w)
```

一个 kernel 和 stride 相同的 3D conv 就是 spatio-temporal patchifier。`(T, H, W) -> (T/2, H/2, W/2)` 的 token grid。

### 第 2 步：3D rotary position encoding

Rotary Position Embeddings（RoPE）分别沿 `t`、`h`、`w` axes 应用：

```python
def rope_3d(tokens, t_dim, h_dim, w_dim, grid):
    """
    tokens: (N, T*H*W, D)
    grid: (T, H, W) sizes
    t_dim + h_dim + w_dim == D
    """
    T, H, W = grid
    n, seq, d = tokens.shape
    if t_dim + h_dim + w_dim != d:
        raise ValueError(f"t_dim+h_dim+w_dim ({t_dim}+{h_dim}+{w_dim}) must equal D={d}")
    assert seq == T * H * W
    t_idx = torch.arange(T, device=tokens.device).repeat_interleave(H * W)
    h_idx = torch.arange(H, device=tokens.device).repeat_interleave(W).repeat(T)
    w_idx = torch.arange(W, device=tokens.device).repeat(T * H)
    # Simplified: just scale channels by frequencies. Real RoPE rotates pairs.
    freqs_t = torch.exp(-torch.log(torch.tensor(10000.0)) * torch.arange(t_dim // 2, device=tokens.device) / (t_dim // 2))
    freqs_h = torch.exp(-torch.log(torch.tensor(10000.0)) * torch.arange(h_dim // 2, device=tokens.device) / (h_dim // 2))
    freqs_w = torch.exp(-torch.log(torch.tensor(10000.0)) * torch.arange(w_dim // 2, device=tokens.device) / (w_dim // 2))
    emb_t = torch.cat([torch.sin(t_idx[:, None] * freqs_t), torch.cos(t_idx[:, None] * freqs_t)], dim=-1)
    emb_h = torch.cat([torch.sin(h_idx[:, None] * freqs_h), torch.cos(h_idx[:, None] * freqs_h)], dim=-1)
    emb_w = torch.cat([torch.sin(w_idx[:, None] * freqs_w), torch.cos(w_idx[:, None] * freqs_w)], dim=-1)
    return tokens + torch.cat([emb_t, emb_h, emb_w], dim=-1)
```

这是简化的 additive 形式。真实 RoPE 会按频率旋转成对 channels；位置信息相同。

### 第 3 步：Divided attention block

```python
class DividedAttentionBlock(nn.Module):
    def __init__(self, dim=64, heads=2):
        super().__init__()
        self.time_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.space_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        self.ln3 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))

    def forward(self, x, grid):
        T, H, W = grid
        n, seq, d = x.shape
        # time attention: same (h, w), across t
        xt = x.view(n, T, H * W, d).permute(0, 2, 1, 3).reshape(n * H * W, T, d)
        a, _ = self.time_attn(self.ln1(xt), self.ln1(xt), self.ln1(xt), need_weights=False)
        xt = (xt + a).reshape(n, H * W, T, d).permute(0, 2, 1, 3).reshape(n, seq, d)
        # space attention: same t, across (h, w)
        xs = xt.view(n, T, H * W, d).reshape(n * T, H * W, d)
        a, _ = self.space_attn(self.ln2(xs), self.ln2(xs), self.ln2(xs), need_weights=False)
        xs = (xs + a).reshape(n, T, H * W, d).reshape(n, seq, d)
        xs = xs + self.mlp(self.ln3(xs))
        return xs
```

Time attention 在每个 spatial position 上跨时间 attend；space attention 在每一帧内跨 positions attend。两个 O(T^2 + (HW)^2) 操作，而不是一个 O((THW)^2) 操作。这是 TimeSformer 和每个现代 video DiT 的核心。

### 第 4 步：组合一个 tiny video DiT

```python
class TinyVideoDiT(nn.Module):
    def __init__(self, in_channels=4, dim=64, depth=2, heads=2):
        super().__init__()
        self.patch = VideoPatch3D(in_channels=in_channels, dim=dim, patch_t=2, patch_h=2, patch_w=2)
        self.blocks = nn.ModuleList([DividedAttentionBlock(dim, heads) for _ in range(depth)])
        self.out = nn.Linear(dim, in_channels * 2 * 2 * 2)

    def forward(self, x):
        tokens, grid = self.patch(x)
        for blk in self.blocks:
            tokens = blk(tokens, grid)
        return self.out(tokens), grid
```

这不是一个能工作的 video generator，而是结构 demo，展示每个组件的 shape 都对。

### 第 5 步：检查 shapes

```python
vid = torch.randn(1, 4, 8, 16, 16)  # (N, C, T, H, W)
model = TinyVideoDiT()
out, grid = model(vid)
print(f"input  {tuple(vid.shape)}")
print(f"tokens grid {grid}")
print(f"output {tuple(out.shape)}")
```

期望 `grid = (4, 8, 8)`，patching 后 `out = (1, 256, 32)`；head 会投影到每个 token 的 spatio-temporal patches，准备 un-patchify 回视频。

## 使用它

2026 年生产访问模式：

- **Sora 2 API**（OpenAI）：text-to-video，同步音频。高端定价。
- **Runway Gen-5 / GWM-1**（Runway）：image-to-video、interactive worlds。
- **Wan-Video 2.1 / HunyuanVideo**：开源 self-host。
- **Cosmos / Cosmos-Drive**（NVIDIA）：driving simulation open weights。
- **Genie 3**：research preview，需要申请访问。

构建 interactive world-model demo：从 Wan-Video 获取质量，再叠加 latent-action adapter 实现交互。自动驾驶仿真：Cosmos-Drive 是 2026 年开放参考。

机器人里常见的栈：

1. Language goal -> VLM（Qwen3-VL）-> high-level plan。
2. Plan -> latent-action video model -> imagined rollout。
3. Rollout -> inverse dynamics model -> low-level actions。
4. Actions executed -> observation fed back into step 1。

## 交付它

本课产出：

- `outputs/prompt-video-model-picker.md`：根据 task、license 和 latency，在 Sora 2 / Runway / Wan / HunyuanVideo / Cosmos 之间选择。
- `outputs/skill-physical-plausibility-checks.md`：一个 skill，会定义自动检查（object permanence、gravity、continuity），用于在 shipping 前检查任何 generated video。

## 练习

1. **（简单）** 计算 5 秒 360p 视频在 patch-t=2、patch-h=8、patch-w=8 下的 token count。推理这个尺寸下 attention 的内存。
2. **（中等）** 把上面的 divided attention block 换成 full joint attention block，并测量 shape 和 parameter count。解释为什么真实 video models 必须使用 divided attention。
3. **（困难）** 构建一个最小 latent-action video model：取一个（frame_t, action_t, frame_{t+1}）triples 数据集（任意简单 2D game），训练一个 tiny video DiT，用 action embeddings 条件化，并展示不同 actions 会产生不同下一帧。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|----------------|----------------------|
| World model | “Learned simulator” | 给定 state 和 action，预测未来 observations 的模型 |
| Video DiT | “Spacetime transformer” | 带 3D patchification 和 divided attention 的 diffusion transformer |
| Latent action | “Inferred control” | 从 frame pairs 推断出的离散或连续 action latent；用于条件化 next-frame generation |
| Divided attention | “Time then space” | 每个 block 两次 attention：先跨时间再跨空间，让 O(N^2) 可管理 |
| Object permanence | “Things stay real” | 视频模型必须学习的场景属性；食物、玻璃器皿上的经典失败模式 |
| FVD | “Fréchet Video Distance” | FID 的视频等价物；主要 visual quality metric |
| Inverse dynamics model | “Observations to actions” | 给定（state, next state），输出连接二者的 action；闭合机器人回路 |
| Cosmos-Drive | “NVIDIA driving sim” | 用于 RL 和 evaluation 的 open-weights 自动驾驶 world model |

## 延伸阅读

- [Sora technical report (OpenAI)](https://openai.com/index/video-generation-models-as-world-simulators/)
- [Genie: Generative Interactive Environments (Bruce et al., 2024)](https://arxiv.org/abs/2402.15391) — latent action world models
- [TimeSformer (Bertasius et al., 2021)](https://arxiv.org/abs/2102.05095) — video transformers 的 divided attention
- [DreamerV3 (Hafner et al., 2023)](https://arxiv.org/abs/2301.04104) — world models for RL
- [Cosmos-Drive-Dreams (NVIDIA, 2025)](https://research.nvidia.com/labs/toronto-ai/cosmos-drive-dreams/) — driving world model
- [Top 10 Video Generation Models 2026 (DataCamp)](https://www.datacamp.com/blog/top-video-generation-models)
- [From Video Generation to World Model — survey repo](https://github.com/ziqihuangg/Awesome-From-Video-Generation-to-World-Model/)
