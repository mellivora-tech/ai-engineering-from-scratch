# Embodied VLAs：RT-2、OpenVLA、π0、GR00T

> 第一次让一个模型从网站上阅读食谱并在厨房机器人中执行，是 RT-2（Google DeepMind，2023 年 7 月）。RT-2 把 actions 离散化为 text tokens，在 web data 加 robot-action data 上 co-fine-tune VLM，并证明 web-scale vision-language knowledge 可以迁移到 robotic control。OpenVLA（2024 年 6 月）发布了 open 7B reference。Physical Intelligence 的 π0 series（2024-2025）加入 flow-matching action experts。NVIDIA 的 GR00T N1（2025 年 3 月）为 humanoid robots 大规模交付 dual-system（System 1 / System 2）control。VLA primitive，即 vision-language-action，一个能看、读、行动的单一模型，是本阶段 understanding models 与阶段 15 Autonomous Systems 之间的桥梁。

**类型：** 学习
**语言：** Python（stdlib，action tokenizer + VLA inference skeleton）
**前置要求：** 阶段 12 · 05（LLaVA），阶段 15（Autonomous Systems，引用）
**时间：** ~180 分钟

## 学习目标

- 描述 action tokenization：discrete bin encoding（RT-2）、FAST efficient action tokens、continuous flow-matching actions（π0）。
- 解释为什么 web + robot data 上的 co-fine-tuning 会保留 general-knowledge transfer 到 novel tasks。
- 在同一个 robot task 上比较 OpenVLA（open 7B Llama+VLM）、π0（flow-matching）和 GR00T N1（dual-system）。
- 说出 Open X-Embodiment dataset 及其作为 RT-X training corpus 的角色。

## 问题

能从 natural language instructions 做家务的机器人，自 1970 年代以来一直是研究目标。2020 年代的答案是 vision-language-action（VLA）model。它使用和 VQA 相同的 VLM 架构，但输出不是文本，而是 actions（joint torques、end-effector poses、discrete commands）。

VLA 特有挑战：

1. Action spaces 是 continuous（joint angles、forces）且 high-dimensional（7-DOF arm + 3-DOF gripper = 10 dims at 30 Hz）。
2. Robot-specific training data 稀缺。Open X-Embodiment 约 1M trajectories；web text-image 是 5B+。
3. Control frequency 很重要。30 Hz control loop 意味着每个 action 只有 33ms budget。
4. Safety。错误 action 会损坏硬件、伤到人或破坏财物。

## 概念

### Action tokenization（RT-2）

RT-2 的技巧：把每个 joint target 表示为 quantized text token。把 normalized [-1, 1] 范围离散成 256 bins，把每个 bin 映射到一个 vocabulary ID。10-DOF action 在每个 control step 变成 10 tokens。

在混合数据上 co-fine-tune PaLM-X VLM：

- Web image-text pairs（captioning、VQA）。
- Robot demonstrations，action 作为 tokens。

模型看到 “pick up the red cube”（language）-> image（vision）-> 10-token action sequence（discretized joint targets）。Web pretraining 保留 general-knowledge transfer：即使 “fast-moving” 不在训练数据中，RT-2 也能遵循 “move towards the fast-moving object”。

RT-2 论文中 inference 为 3-5 Hz，受限于 VLM autoregressive decode。

### OpenVLA：open 7B reference

OpenVLA（Kim 等人，2024 年 6 月）是 open-weights RT-2 等价物。7B Llama backbone，DINOv2 + SigLIP dual vision encoder，基于 256 bins 的 action tokenization。

在 Open X-Embodiment（970k trajectories，覆盖 22 robots）上训练。附带 LoRA fine-tuning support，用于适配新机器人。

Inference：带 quantization 的 A100 上 4-5 Hz。足够慢速 manipulation，不足以做 high-frequency control。

### FAST tokenizer：更快的 action decode

Pertsch 等人（2024）表明 discrete-bin tokenization 很低效，因为多数 actions 聚集在 bin-space 的小区域。FAST（Frequency-domain Action Sequence Tokenizer）通过 DCT 压缩 action sequences，并量化 coefficients。

30-step action trajectory 变成约 10 个 FAST tokens，而不是 300 个 discrete-bin tokens。Inference 加速 3-5 倍，且不损失质量。

### π0 与 flow-matching actions

Physical Intelligence 的 π0（Black 等人，2024 年 10 月）用 flow-matching action expert 替换 discrete action tokens：

- 小型 action transformer 读取 VLM hidden states，并通过 rectified flow 输出 continuous 50-step action sequence。
- Action head 使用 flow-matching loss 训练；VLM pretraining 保持不变。
- Inference：完整 action sequence 约 5 denoising steps 输出，实际上是 50 Hz control。

π0 的 claim：在广泛 manipulation tasks 上超过 OpenVLA 和 Octo。Continuous-action formulation 保留了 discretization 会破坏的 smoothness。

π0.5 和 π0-FAST 是增量升级。π0-FAST 结合 FAST tokenization 与 flow matching。

### GR00T N1：humanoids 的 dual-system

NVIDIA 的 GR00T N1（2025 年 3 月）面向 humanoid robots（>30 DOF，全身）：

- System 2：大型 VLM 读取 scene + instruction，以约 1 Hz 产生 high-level subgoals。
- System 1：小型 action-head transformer 基于 subgoals 产生 50-100 Hz low-level joint commands。

这个拆分映射到 Kahneman 的 fast-and-slow thinking：System 2 规划，System 1 行动。好处：慢速 VLM-sized planning 不阻塞快速 control；System 1 保持小型以保证 latency。

GR00T N1.7（2025 年末）提升 data scaling。GR00T 使用来自 Omniverse 的 sim-to-real data 做 fine-tuning。

### Open X-Embodiment

训练数据。RT-X（2023 年 10 月）组合了 22 个 datasets，覆盖 22 种 robots、1M trajectories。Open X-Embodiment 是所有人使用的 corpus：

- ALOHA / Bridge V2 / Droid / RT-2 Kitchen / Language Table。
- 每个 sample：（robot state、camera views、instruction、action sequence）。
- Training hygiene：统一 action space、normalize joint ranges、resize cameras。

OpenVLA 和 π0 都在 Open X-Embodiment 上训练。到任意特定机器人的 domain gap 通过 100-1000 条 task-specific demos 上的 LoRA fine-tuning 缩小。

### Co-fine-tuning vs robot-only

Co-fine-tuning 把 web VQA data 与 robot trajectories 混在一起。比例很重要：VQA 太多，模型忘记 actions；robot data 太多，模型失去 general knowledge。

RT-2 比例约为 1:1。OpenVLA 约 0.5:1 web-to-robot。π0 类似。精确比例是要按 dataset size 调的 hyperparameter。

Robot-only training 会产生 task-specific models，对 out-of-distribution instructions 失败。Co-fine-tuning 让“pick up the red cube（demo 中）”变成“pick up the third largest object from the left（新措辞）”。

### Safety 与 action limits

每个 production VLA 都带有：

- Hard joint limits（不能超过 torque spec）。
- Velocity limits（soft clipping）。
- Workspace bounds（end-effector 不能离开桌面）。
- Human-in-the-loop approval for novel tasks。

这些位于 VLA 外部，作为 control-layer checks。VLA 的输出是建议，不是命令。

## 使用它

`code/main.py`：

- 实现 256-bin action tokenization 和 de-tokenization。
- 基于 DCT + quantization 草拟 FAST tokenizer。
- 对比 discrete-bin、FAST、continuous-flow 在每个 action step 的 token-count。
- 打印 RT-2 -> OpenVLA -> π0 -> GR00T 的 lineage summary。

## 交付它

本课产出 `outputs/skill-vla-action-format-picker.md`。给定 robot task（manipulation、navigation、humanoid whole-body），它会在 discrete-bin + RT-2、FAST + OpenVLA、flow-matching + π0、dual-system + GR00T 之间选择。

## 练习

1. 一个 10-DOF arm，30 Hz control rate。256 bins 的 discrete-bin tokenization 每秒输出多少 tokens？7B VLM 能跟上吗？

2. FAST tokenization 把 30-step trajectories 压缩到约 10 tokens。如果 trajectory 有 high-frequency motion（例如 drumming），用户会损失什么？

3. π0 的 flow-matching head 约 5 步 denoise。与 OpenVLA 的 4-5 Hz autoregressive decode 比较 throughput。

4. GR00T 的 System 1 / System 2 split 映射到 Kahneman。提出一个不同 split（System 3？），可能帮助 bipedal walking。

5. 阅读 Open X-Embodiment 第 4 节关于 dataset curation。说出防止 domain leakage 的三条 curation rules。

## 关键术语

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| VLA | “Vision-language-action” | 接收 image + instruction 并输出 action commands 的模型 |
| Action tokenization | “Discrete bins” | 把 continuous joint targets 量化成每维 256 bins，每个 bin 是一个 vocab ID |
| FAST tokenizer | “Frequency action tokens” | DCT + quantize，把 30-step trajectories 压缩到约 10 tokens |
| Co-fine-tune | “Mix web + robot” | 同时在 web VQA data 和 robot demos 上训练，以保留 general knowledge |
| Flow-matching action head | “π0 continuous output” | 小型 transformer 通过 rectified flow 输出 50-step action sequence |
| System 1 / System 2 | “Dual-system control” | 大型 VLM 慢速规划，小型 action head 快速行动；GR00T pattern |
| Open X-Embodiment | “RT-X dataset” | 1M-trajectory cross-robot dataset；训练 corpus |

## 延伸阅读

- [Brohan et al. — RT-2 (arXiv:2307.15818)](https://arxiv.org/abs/2307.15818)
- [Kim et al. — OpenVLA (arXiv:2406.09246)](https://arxiv.org/abs/2406.09246)
- [Black et al. — π0 (arXiv:2410.24164)](https://arxiv.org/abs/2410.24164)
- [NVIDIA — GR00T N1 (arXiv:2503.14734)](https://arxiv.org/abs/2503.14734)
- [Open X-Embodiment Collab — RT-X (arXiv:2310.08864)](https://arxiv.org/abs/2310.08864)
