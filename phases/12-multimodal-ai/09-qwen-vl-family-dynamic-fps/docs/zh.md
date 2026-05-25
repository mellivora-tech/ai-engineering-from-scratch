# Qwen-VL Family 与 Dynamic-FPS Video

> Qwen-VL family，包括 Qwen-VL（2023）、Qwen2-VL（2024）、Qwen2.5-VL（2025）、Qwen3-VL（2025），是 2026 年最有影响力的开放 vision-language model lineage。每一代都做了一个决定性的架构下注，并在十二个月内被开放生态复制：通过 M-RoPE 实现 native dynamic resolution、带 absolute time alignment 的 dynamic-FPS sampling、ViT 中的 window attention、structured agent output formats。到 Qwen3-VL，recipe 已经稳定：带 2D-RoPE 的 ViT encoder，支持 native-aspect-ratio inputs；MLP projector 接入大型 Qwen3 language base；训练阶段把 OCR、grounding 和 agent behavior 作为 first-class targets。本课按时间顺序阅读这个 family，让你理解每个旋钮为什么在那里。

**类型：** 学习
**语言：** Python（stdlib，M-RoPE encoder + dynamic-FPS sampler）
**前置要求：** 阶段 12 · 06（patch-n'-pack）
**时间：** ~120 分钟

## 学习目标

- 计算 M-RoPE 的三轴旋转（temporal、height、width），并解释为什么三轴都需要。
- 为一个视频选择 dynamic-FPS sampling strategy，并推理 tokens-per-second 与 event-detection accuracy 的取舍。
- 按顺序说出四代 Qwen-VL upgrades，以及每一代启用了什么。
- 接入 Qwen2.5-VL-style JSON agent output format，并从 VLM response 中解析 structured tool calls。

## 问题

Qwen-VL 于 2023 年 8 月发布，是对 LLaVA-1.5 和 BLIP-2 的直接回应。Qwen 团队瞄准的差距有三类：resolution、video 和 structured output。

Resolution：LLaVA-1.5 运行在 336x336。对照片还行，对中文发票或密集电子表格截图没用。Qwen-VL 的第一个创新是 448x448 和 grounded bounding-box output，让模型能指向东西。

Video：Video-LLaMA 堆叠 per-frame encoders 并喂给 LLM。短 clips 有效，但对 multi-minute videos 不行，因为 temporal axis 才是信号所在。Qwen 团队想要一个理解时间的单一 encoder。

Structured output：LLaVA 输出 free-form text。Agent 需要 JSON。Qwen-VL 在显式 JSON output formats 上训练，包括把 bounding-box coordinates 作为文本输出。

每一代 Qwen-VL 都扩展这三个轴之一。

## 概念

### Qwen-VL（2023 年 8 月）

第一代：OpenCLIP ViT-bigG/14 作为 encoder（2.5B params），LLama-compatible Q-Former（1-step，256 queries），Qwen-7B base。贡献：

- 448x448 resolution（当时 open VLM 的 SOTA）。
- Grounding：在带有显式 coordinate-token output 的 image-text pairs 上训练。“The cat is at <box>(112, 204), (280, 344)</box>”。
- 从一开始就做中文 + 英文 multilingual training。

当时 benchmarks：英文上接近 GPT-4V，中文上占优。Grounding supervision 才是真正 headline。

### Qwen2-VL（2024 年 9 月）— M-RoPE 与 native resolution

Qwen2-VL 用 natively dynamic-resolution ViT encoder 替换了 fixed-resolution + Q-Former stack。关键变化：

- Native dynamic resolution。ViT 接受任意 HxW，只要能被 28 整除（patch 14 with 2x spatial merge）。1120x672 图像（40x24 merged patches）产生 960 visual tokens。没有 resize，没有 tiling，没有 thumbnail。
- M-RoPE（Multimodal RoPE）。每个 token 携带 3D position（t, h, w），而不是 1D。图像 t=0，视频 t = frame_index。RoPE 按每个轴的频率旋转 query/key vectors。没有 positional embedding table。
- MLP projector。丢掉 Q-Former；对 merged patch tokens 使用 2-layer MLP。
- Dynamic FPS video。默认 1-2 FPS 采样 video，但模型接受任意 frame count。

结果：Qwen2-VL-7B 在多个 multimodal benchmark 上匹配 GPT-4o，并在 DocVQA 上超过它（94.5 vs 88.4）。架构变化是决定性一步。

### Qwen2.5-VL（2025 年 2 月）— dynamic FPS + absolute time

Qwen2.5-VL 的大转变是 video。Dynamic FPS 不只是“需要时采更多帧”。论文形式化了：

- Absolute time tokens。不是用位置索引（frame 0、1、2...），而是用实际时间戳。“At 0:04, the cat jumps.” 模型看到与 frame tokens 交错的 `<time>0.04</time>` tokens。
- Dynamic FPS。慢镜头用 1 FPS，动作场景用 4+ FPS。用户或训练器选择；M-RoPE 自适应。
- ViT 中的 window attention。Spatial attention 被 windowed（局部 blocks 内）以提升吞吐，每隔几层使用 global attention。
- 显式 JSON output format。在 tool-call data 上训练："{\"tool\": \"click\", \"coords\": [380, 220]}"。开箱 agent-ready。
- MRoPE-v2 scaling。Positions 按最大输入尺寸缩放，避免 10 分钟视频耗尽 frequency range。

Benchmarks：Qwen2.5-VL-72B 在多数 video benchmarks 上超过 GPT-4o，在 documents 上匹配 Gemini 2.0，并在 GUI grounding 上达到 open-model SOTA（ScreenSpot：84% accuracy vs GPT-4o 的 38%）。

### Qwen3-VL（2025 年 11 月）

Qwen3-VL 是一次巩固而非重造：更大的 LLM backbone（Qwen3-72B）、扩展 training data、改进 OCR、通过 Qwen3 “thinking mode” 增强 reasoning。ViT 和 M-RoPE 保持不变。论文重点是数据和训练改进，而不是架构。

Lineage takeaway：到 2025 年，Qwen-VL 架构已经稳定。后续代际扩展的是 compute 和 data，而不是 primitives。

### M-RoPE 的数学

经典 RoPE 会按位置 `m` 旋转维度为 `d` 的 query `q`，使用成对坐标：

```
q_rot[2i]   = q[2i]   * cos(m * theta_i) - q[2i+1] * sin(m * theta_i)
q_rot[2i+1] = q[2i]   * sin(m * theta_i) + q[2i+1] * cos(m * theta_i)
theta_i     = 10000^(-2i/d)
```

M-RoPE 把 hidden dim 分成三个 band。假设 `d = 96`。给 temporal 32 dims、height 32 dims、width 32 dims。每个 band 按自己的轴位置旋转。位于（t=5, h=10, w=20）的 patch，会把 `R_t(5)`、`R_h(10)`、`R_w(20)` 分别应用到三个 band。

Text tokens 使用 `t = text_index, h = 0, w = 0`（或某种 normalized choice）以保持兼容。Video frames 使用 `t = frame_time, h = row, w = col`。Single images 使用 `t = 0`。

好处是：一种 position encoding 可以处理 text、image 和 video，不需要分支代码或不同 position tables。

### Dynamic-FPS sampling logic

给定时长为 `T` 秒的视频和目标 token budget `B`：

1. 计算能负担的最大 FPS：`fps_max = B / (T * tokens_per_frame)`。
2. 从 `{1, 2, 4, 8}` 中选择满足 `fps <= fps_max` 的 target FPS。
3. 如果 motion high（optical-flow heuristic 或显式用户请求），选择更高 FPS。如果 motion low，选择更低。
4. 按所选 FPS 均匀采样；在 frames 之间插入 `<time>t</time>` tokens。

Qwen2.5-VL 在训练中隐式学习这套逻辑；推理时用户通过 `fps` 参数控制。一个 60 秒 action sequence，如果 4 FPS、每帧 81 tokens，总计 19440 tokens，在 32k context 中可管理。

### Structured agent output

Qwen2.5-VL 的 agent training 明确以 structured tool calls 为目标：

```
{
  "tool": "mouse_click",
  "coords": [1024, 512],
  "button": "left",
  "modifier": null
}
```

解析是确定性的：对模型输出做 JSON.parse。对比 free-form “click at (1024, 512)”，后者需要 regex 和歧义处理。这个转变就是 Qwen2.5-VL 的 ScreenSpot 从 Qwen2-VL 的 55% 跳到 84% 的原因。

## 使用它

`code/main.py` 实现：

- 为混合 text、image patches 和 video frames 的 packed sequence 计算 M-RoPE positions。
- Dynamic-FPS sampler：给定（duration、budget、motion_level），选择 FPS 并输出 frame timestamps。
- 一个 toy Qwen2.5-VL JSON-output parser，处理带 coordinate fields 的 tool-call responses。

运行它，然后把固定 FPS 换成 dynamic-FPS，在 5 分钟视频上感受差异。

## 交付它

本课产出 `outputs/skill-qwen-vl-pipeline-designer.md`。给定 video task（monitoring、agent、action recognition、accessibility），它会输出 Qwen2.5-VL configuration（frame budget、FPS strategy、window-attention flag、agent-output mode）和 latency estimate。每次为 video product 部署 Qwen-VL-family model 时都使用它。

## 练习

1. 对 hidden 48（每个 band 16，base theta 10000）下位于（t=3, h=5, w=7）的 patch 计算 M-RoPE rotations。展示每个 band 中前三个 pair 的 rotation angles。

2. 一段 10 分钟 security-camera recording，以 1 FPS 会产生多少 frames？在 384 resolution + 3x pool 下，总 token 数是多少？Qwen2.5-VL 默认 32k context 能处理吗？

3. 为 30 秒网球回合、30 秒 recipe demo、30 秒 UI-agent recording 选择 FPS。用 dynamic-FPS logic 为每个选择辩护。

4. Qwen2.5-VL 完全丢掉 Q-Former。为什么一个简单 MLP 在 2025 年可行，但在 2023 年不可行？（提示：data scale 和 encoder quality。）

5. 把三个 Qwen2.5-VL JSON tool-call outputs 解析成 Python dict。Malformed JSON 会如何失败？Qwen cookbook 推荐什么 recovery strategy？

## 关键术语

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| M-RoPE | “Multimodal RoPE” | 在 hidden dim 中包含 temporal、height、width bands 的 3D rotary position embedding |
| Dynamic FPS | “Smart sampling” | 根据 motion、duration 和 token budget 为每个视频选择 frame sampling rate |
| Absolute time token | “Timestamp token” | 序列中交错的 `<time>t</time>`，让模型看到实际秒数而非 frame index |
| Window attention | “Local attention” | 为速度把 spatial self-attention 限制在小窗口内；周期性加入 global attention |
| Structured agent output | “JSON mode” | 训练数据监督 VLM 输出可解析 JSON，包含 coords 和 tool names |
| min_pixels / max_pixels | “Resolution bounds” | Qwen2.5-VL 的 per-request 控制项，用于限制总像素数，从而限制 token count |
| Grounding | “Point-at-it” | 以 text tokens 输出 bounding-box coordinates；自 Qwen-VL v1 起使用 |

## 延伸阅读

- [Bai et al. — Qwen-VL (arXiv:2308.12966)](https://arxiv.org/abs/2308.12966)
- [Wang et al. — Qwen2-VL (arXiv:2409.12191)](https://arxiv.org/abs/2409.12191)
- [Qwen Team — Qwen2.5-VL Technical Report (arXiv:2502.13923)](https://arxiv.org/abs/2502.13923)
- [Qwen Team — Qwen3-VL (arXiv:2511.21631)](https://arxiv.org/abs/2511.21631)
- [Zhu et al. — InternVL3 (arXiv:2504.10479)](https://arxiv.org/abs/2504.10479)
