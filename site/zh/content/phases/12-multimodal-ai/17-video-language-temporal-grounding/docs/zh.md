# Video-Language Models：Temporal Tokens 与 Grounding

> Video 不是一叠照片。5 秒 clip 有因果顺序、动作动词和事件时间，这些是 image model 无法表示的。Video-LLaMA（Zhang 等人，2023 年 6 月）发布了第一个带 audio-visual grounding 的 open video-LLM。VideoChat 和 Video-LLaVA 扩展了这个模式。到 2025 年，Qwen2.5-VL 的 TMRoPE 缩小了与 frontier proprietary models 的差距。每个系统都用不同方式解决 temporal tokens：per clip Q-former、per frame concat-pool、per token TMRoPE。本课阅读这些模式，构建 uniform-vs-dynamic frame sampler，并在 temporal grounding tasks 上评估。

**类型：** 构建
**语言：** Python（stdlib，frame sampler + temporal-grounding evaluator）
**前置要求：** 阶段 12 · 08（LLaVA-OneVision）
**时间：** ~180 分钟

## 学习目标

- 解释 temporal positional encoding 如何独立于 vision encoder 改变 video VLM 性能。
- 在 tokens-per-second 与 grounding accuracy 上比较 uniform、dynamic-FPS 和 event-driven frame sampling。
- 描述 Q-former-per-clip（Video-LLaMA）、pooled-per-frame（Video-LLaVA）和 M-RoPE-per-token（Qwen2.5-VL）设计。
- 说出四个 video benchmarks：VideoMME、TempCompass、EgoSchema、Video-MMMU。

## 问题

1 分钟 30 FPS 的视频有 1800 帧。如果每帧 196 visual tokens（ViT-B at 224），就是 352k tokens，已经大于任何 2024-era LLM context。

三种 reduction strategies：

1. Subsample frames（根据内容 1-8 FPS）。
2. Aggressively pool 每帧 patch tokens（3x3 或 4x4 bilinear pool）。
3. 通过 Q-former 压缩，把 16-frame clip 输出为 64 tokens。

每种 trade-off 不同。Subsampling 丢 temporal detail。Pooling 丢 spatial detail。Q-former 两者都丢一点，但省 token。

Temporal position encoding 是另一条轴：模型如何知道 frame 5 在 frame 6 之前？选项包括 simple 1D temporal RoPE（Video-LLaMA）、learned temporal embeddings（Video-LLaVA）、TMRoPE（Qwen2.5-VL，全 3D）。

## 概念

### Video-LLaMA：每个 clip 一个 Q-former + audio branch

Video-LLaMA（2023）是第一个 open video-LLM。架构：

- 16-frame clips at 2 FPS（即 8 秒）。
- Per-frame ViT features -> Video Q-former，对全部 16 帧做 cross-attend -> 32 learned queries -> LLM。
- 并行 audio branch：waveform -> ImageBind audio encoder -> Audio Q-former -> 32 queries -> LLM。

强项：audio-visual joint reasoning。弱点：固定 clip length，不能做任意 time grounding。

### VideoChat 与 Video-LLaVA

VideoChat 保留 Video-LLaMA 思路，但丢掉 audio 并简化。Video-LLaVA（Lin 等人，2023）在 images 和 video frames 上训练单个 visual encoder（“alignment before projection”），得到统一 representation。两者都是 frozen-CLIP-encoder + MLP + LLM。

两者都处理不了 long video。都是 8-16 frame systems。

### Qwen2.5-VL 与 TMRoPE

Qwen2.5-VL 引入 TMRoPE，即 Temporal-Modality Rotary Position Embedding。每个 patch token 携带（t, h, w）position，其中 t 是实际 timestamp（不是 frame index）。

与简单 temporal embedding 的关键差异：

- Absolute time，而不是 index。模型看到的是 “at 4.2 seconds”，不是 “at frame 15”。
- Per-token rotation，而不是 per-clip。每个 visual token 都按自己的 timestamp 独立旋转。
- 兼容 dynamic FPS。如果这里用 2 FPS、那里用 4 FPS 采样，TMRoPE 原生处理不均匀间隔。

TMRoPE 支持“猫在第几秒跳起来？”这样的查询。模型可以输出“在 4.2 秒”。Video-LLaMA 只能说“clip 前半段”。

### Frame sampling strategies

Uniform：在 duration 上均匀采样 N 帧。简单，但会丢 motion peaks。

Dynamic FPS：根据 motion intensity 自适应采样。Optical flow 或 frame differencing 为高运动片段选择更密采样。Qwen2.5-VL 在这类数据上训练。

Event-driven：运行轻量 detector，在动作发生处采更多。VideoAgent 使用这种方式。

Keyframe + context：在 shot boundaries 加少量相邻帧采样。用于 cinematic content。

### 每帧 pooling

1 FPS 且每帧 576 tokens 时，5 分钟 clip 是 172,800 tokens。Qwen2.5-VL-72B 的 128k context 勉强可行但昂贵。

3x3 bilinear pool 降到每帧 64 tokens -> 5 分钟 19,200 tokens。多数任务的甜点区。

对于 spatial detail 不那么重要的 agent workflows，可以更激进 pooling（6x6 -> 每帧 16 tokens）。

### 四个 video benchmarks

- VideoMME：comprehensive video understanding，短/中/长都有。
- TempCompass：fine-grained temporal reasoning，“before” / “after” questions。
- EgoSchema：long-horizon first-person video。
- Video-MMMU：multimodal multi-discipline video questions。

完整 video-VLM evaluation 会覆盖四者。它们强调不同轴：TempCompass 完全关注 ordering，EgoSchema 关注 3+ 分钟 reasoning，VideoMME 覆盖不同 duration。

### Grounding output formats

Temporal grounding 的输出格式：

- Free text：“The cat jumps around the 4-second mark.” 易解析但不精确。
- Structured JSON：`{"event": "jump", "start": 4.1, "end": 4.3}`。Qwen2.5-VL 训练这种格式。
- Token-based：答案中交错特殊 `<time>4.1</time>` tokens。Qwen2.5-VL 内部格式。

Token-based 对下游使用最准确。Qwen2.5-VL 的 JSON output format 可直接解析。

### 2026 best practice

2026 年的视频 VLM：

- Encoder：SigLIP 2 with M-RoPE or TMRoPE（Qwen2.5-VL）。
- Frame sampling：dynamic FPS（根据 motion 为 1-4）并带 max-frame cap。
- Per-frame pooling：3x3 bilinear。
- Output：structured JSON，含 time + event fields。
- Benchmarks：general 用 VideoMME + TempCompass；long-horizon 用 EgoSchema。

## 使用它

`code/main.py` 包含：

- Uniform 和 dynamic-FPS frame samplers。
- Toy temporal-grounding evaluator：给定 time T 的 “ground truth” event 和 model output，在 tolerance 下打分 accuracy。
- Video-LLaMA（16 frames，Q-former）、Video-LLaVA（8 frames，MLP）、Qwen2.5-VL（dynamic FPS + TMRoPE）的对比。

## 交付它

本课产出 `outputs/skill-video-vlm-frame-planner.md`。给定 video task（monitoring、action recognition、temporal grounding、summarization），它会选择 frame sampler、pooling factor、output format 和 expected accuracy tier。

## 练习

1. 对 3 分钟 cooking demo，选择 uniform 还是 dynamic FPS。用 token count 解释。

2. TMRoPE 具体增加了什么，是 simple temporal embedding table 做不到的？

3. 写一个 temporal grounding 的 JSON schema，让 VLM 能学习输出。包括 error cases。

4. 阅读 Video-LLaVA 第 3 节 “Alignment Before Projection”。为什么这优于训练 separate image and video encoders？

5. 给定 VideoMME leaderboard，截至 2026 年 top open model 与 top proprietary model 的差距是多少？这个差距有多少归因于 temporal encoding，有多少归因于 base LLM scale？

## 关键术语

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| Temporal grounding | “Time-localized answers” | VLM 输出事件发生的具体 timestamp range |
| TMRoPE | “Time-Multimodal RoPE” | 使用 absolute timestamps 的 3D rotary position，用于 Qwen2.5-VL |
| Dynamic FPS | “Motion-aware sampling” | 高运动片段采更多帧，静态片段采更少 |
| Frame pooling | “Spatial compress per frame” | 在送入 LLM 前用 bilinear interpolation 减少每帧 patches |
| Video Q-former | “Clip compressor” | Cross-attention bottleneck，把 N 帧映射成 K 个 learned queries |
| VideoMME | “Video bench” | 覆盖短/中/长视频的综合 benchmark，2500+ samples |

## 延伸阅读

- [Zhang et al. — Video-LLaMA (arXiv:2306.02858)](https://arxiv.org/abs/2306.02858)
- [Li et al. — VideoChat (arXiv:2305.06355)](https://arxiv.org/abs/2305.06355)
- [Lin et al. — Video-LLaVA (arXiv:2311.10122)](https://arxiv.org/abs/2311.10122)
- [Qwen Team — Qwen2.5-VL (arXiv:2502.13923)](https://arxiv.org/abs/2502.13923)
- [Lin et al. — VILA-1.5 (arXiv:2312.07533)](https://arxiv.org/abs/2312.07533)
