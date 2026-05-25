# Sim-to-Real Transfer

> 一个在模拟器里训练、到硬件上失败的 policy，只是记住了模拟器。Domain randomization、domain adaptation 和 system identification，是让 learned controllers 跨越 reality gap 的三件工具。

**类型：** 学习
**语言：** Python
**先修：** Phase 9 · 08（PPO），Phase 2 · 10（Bias/Variance）
**时间：** 约 45 分钟

## 问题

训练真实机器人缓慢、危险且昂贵。双足机器人需要数百万个训练 episode 才能学会行走；真实双足机器人哪怕只摔倒一次，也可能损坏硬件。Simulation 给你无限 reset、确定性可复现、并行环境，以及不会造成物理损坏。

但模拟器是错的。轴承的摩擦比 MuJoCo 模型更大。相机有镜头畸变，而模拟器没有包含。电机有延迟、backlash 和饱和，99% 的 sim model 都会跳过。风、灰尘、可变光照会破坏在无菌渲染上训练的 policy。**Reality gap**，也就是 sim distribution 和 real distribution 之间的系统性差异，是部署机器人 RL 的核心问题。

你需要一个对 *sim-to-real distribution shift* 鲁棒的 policy。三种历史方法：随机化模拟器（domain randomization）、用少量真实数据适配 policy（domain adaptation / fine-tuning），或者识别真实系统参数并匹配它们（system identification）。到 2026 年，主流配方把三者都和大规模并行模拟结合起来（Isaac Sim、Isaac Lab、GPU 上的 Mujoco MJX）。

## 核心概念

![Three sim-to-real regimes: domain randomization, adaptation, system identification](../assets/sim-to-real.svg)

**Domain Randomization（DR）。** Tobin et al. 2017，Peng et al. 2018。训练期间，随机化真实机器人上可能不同的每个 sim 参数：质量、摩擦系数、motor PD gains、sensor noise、camera position、lighting、textures、contact models。Policy 会学习一个关于“今天它处在哪个 sim 中”的条件分布，并在整个范围内泛化。如果真实机器人落在训练 envelope 内，policy 就能工作。

- **优点：** 不需要真实数据。一套配方，多种机器人。
- **缺点：** 过度随机化训练会产生一个“通用”但过度谨慎的 policy。噪声太多 ≈ 正则化太强。

**System Identification（SI）。** 训练前，把模拟器参数拟合到真实世界数据。如果你能测量真实机器人手臂关节摩擦，就把它填进 sim。然后训练一个期待这些值的 policy。它需要接触真实系统，但能直接缩小 reality gap。

- **优点：** 精确、低噪声训练目标。
- **缺点：** 残余 model error 对 policy 不可见；小的未识别效应（例如 motor deadband）仍会破坏部署。

**Domain Adaptation。** 在 sim 中训练，用少量真实数据 fine-tune。两种风味：

- **Real2Sim2Real：** 用真实 rollouts 学一个 residual simulator `f(s, a, z) - f_sim(s, a)`，然后在修正后的 sim 中训练。用很少真实数据就能缩小差距。
- **Observation adaptation：** 训练一个 policy，通过 learned feature extractor（例如 GAN pixel-to-pixel）把 real obs → sim-like obs。Controller 保持在 sim 中。

**Privileged learning / teacher-student。** Miki et al. 2022（ANYmal quadruped）。在模拟中训练一个可以访问 privileged information（ground truth friction、terrain height、IMU drift）的 *teacher*。再蒸馏一个只看真实传感器 observation 的 *student*。Student 学会从 history 中推断 privileged features，在不同物理参数下保持鲁棒。

**Massively parallel simulation。** 2024–2026。Isaac Lab、Mujoco MJX、Brax 都能在单张 GPU 上运行数千个并行机器人。PPO 配 4,096 个并行 humanoid，几小时内就能收集数年的经验。随着训练分布变宽，“reality gap” 会缩小；当 4,096 个 env 中每个都有不同随机参数时，DR 几乎是免费的。

**真实世界 2026 配方（四足行走示例）：**

1. 大规模并行 sim，domain-randomized gravity、friction、motor gains、payload。
2. 使用 privileged info（terrain map、body velocity ground truth）训练 teacher policy。
3. 只用 proprioception（腿部关节编码器）从 teacher 蒸馏 student policy。
4. 可选：通过真实 IMU 上的 autoencoder 做 observation adaptation。
5. 部署。Zero-shot 到 10+ 种环境。如果失败，用带 safety constraint 的 PPO 做几分钟真实世界 fine-tuning。

## 动手构建

本课代码是在带*噪声* transition 的 GridWorld 上演示 domain randomization。我们训练一个在“sim”中经历随机 slip probabilities 的 policy，然后在带有从未训练过的 slip level 的“real”环境上评估。这个形状直接映射到 MuJoCo-to-hardware transfer。

### 第 1 步：parameterized sim

```python
def step(state, action, slip):
    if rng.random() < slip:
        action = random_perpendicular(action)
    ...
```

`slip` 是模拟器暴露的参数。在真实机器人中，它可以是 friction、mass、motor gain，任何会在 sim 和 real 之间漂移的东西。

### 第 2 步：用 DR 训练

每个 episode 开始时，采样 `slip ~ Uniform[0.0, 0.4]`。训练 PPO / Q-learning / 任何方法。重复很多 episode。

### 第 3 步：在“real” slips 上 zero-shot 评估

在 `slip ∈ {0.0, 0.1, 0.2, 0.3, 0.5, 0.7}` 上评估。前四个在训练支持内；`0.5` 和 `0.7` 在范围外。DR-trained policy 应该在支持内保持近似最优，并在支持外优雅退化。Fixed-slip-trained policy 在训练 slip 之外会很脆。

### 第 4 步：与 narrow training 对比

训练第二个只使用 `slip = 0.0` 的 policy。在同一组 `slip` 上评估。你应该看到，只要真实 slip > 0，性能就会灾难性下降。

## 常见坑

- **随机化太多。** 在 `slip ∈ [0, 0.9]` 上训练，你的 policy 会过度规避风险，以至于永远不尝试最优路径。匹配*预期*真实世界分布，而不是“任何事都可能发生”。
- **随机化太少。** 在很薄的一片范围内训练，policy 完全无法泛化。使用 adaptive curriculum（Automatic Domain Randomization），随着 policy 改善逐步扩大分布。
- **参数空间识别错误。** 随机化错误的东西（真实差距是 motor delay，却随机化 camera hue），DR 不会有帮助。先 profile 真实机器人。
- **Privileged info leakage。** 如果 teacher 用 global state 做动作，而不只是 observation，可能产生 student 无法追上的行为。确保 teacher policy 在给定 observation history 时对 student 是可实现的。
- **Sim-to-sim transfer failure。** 如果你的 policy 对更难的 sim 变体都不鲁棒，它也不会对真实世界鲁棒。部署前始终在 hold-out sim variant 上测试。
- **没有真实世界 safety envelope。** 一个在 sim 中可行、在 real 中“可行”的 policy，如果没有低层 safety shield，仍可能损坏硬件。把 rate limits、torque limits、joint limits 加到非学习控制器中。

## 使用它

2026 年 sim-to-real stack：

| 领域 | Stack |
|------|-------|
| Legged locomotion（ANYmal, Spot, humanoid） | Isaac Lab + DR + privileged teacher / student |
| Manipulation（dexterous hands, pick-and-place） | Isaac Lab + DR + DR-GAN for vision |
| Autonomous driving | CARLA / NVIDIA DRIVE Sim + DR + real fine-tune |
| Drone racing | RotorS / Flightmare + DR + online adaptation |
| Finger/in-hand manipulation | OpenAI Dactyl（前所未有规模的 DR） |
| Industrial arms | MuJoCo-Warp + SI + small real fine-tune |

各种规模的控制任务，工作流都是一致的：尽力拟合 sim，随机化无法拟合的部分，训练大型 policy，蒸馏，带 safety shield 部署。

## 交付

保存为 `outputs/skill-sim2real-planner.md`：

```markdown
---
name: sim2real-planner
description: Plan a sim-to-real transfer pipeline for a given robot + task, covering DR, SI, and safety.
version: 1.0.0
phase: 9
lesson: 11
tags: [rl, sim2real, robotics, domain-randomization]
---

Given a robot platform, a task, and access to real hardware time, output:

1. Reality gap inventory. Suspected sources ranked by expected impact (contact, sensing, actuation delay, vision).
2. DR parameters. Exact list, ranges, distribution. Justify each range against real measurements.
3. SI steps. Which parameters to measure; measurement method.
4. Teacher/student split. What privileged info the teacher uses; what obs the student uses.
5. Safety envelope. Low-level limits, emergency stops, backup controller.

Refuse to deploy without (a) a zero-shot sim-variant test, (b) a safety shield, (c) a rollback plan. Flag any DR range wider than 3× measured real variability as likely over-randomized.
```

## 练习

1. **简单。** 在 fixed-slip GridWorld（slip=0.0）上训练 Q-learning agent。在 slip ∈ {0.0, 0.1, 0.3, 0.5} 上评估。绘制 return vs slip。
2. **中等。** 训练一个 DR Q-learning agent，采样 `slip ~ Uniform[0, 0.3]`。评估同一组 slip。DR 在 slip=0.5（out-of-distribution）上带来多少收益？
3. **困难。** 实现 curriculum：从 slip=0.0 开始，每当 policy 达到 90% 最优时扩大 DR range。测量到达 slip=0.3 zero-shot 所需的总 environment steps，并与 fixed DR baseline 比较。

## 关键术语

| 术语 | 人们通常怎么说 | 它实际的含义 |
|------|----------------|--------------|
| Reality gap | “Sim-to-real difference” | 训练与部署 physics/sensing 之间的 distribution shift。 |
| Domain randomization（DR） | “跨随机 sim 训练” | 训练时随机化 sim 参数，让 policy 泛化。 |
| System identification（SI） | “测量真实系统并拟合 sim” | 估计真实物理参数；设置 sim 与其匹配。 |
| Domain adaptation | “在真实数据上 fine-tune” | Sim 训练后进行少量真实世界 fine-tune；可能适配 obs 或 dynamics。 |
| Privileged info | “teacher 的 ground truth” | 只有 sim 拥有的信息；student 必须从 obs history 推断。 |
| Teacher/student | “把 privileged 蒸馏成 observable” | Teacher 带捷径训练；student 学会在没有捷径时模仿。 |
| ADR | “Automatic Domain Randomization” | 随着 policy 改善而扩大 DR ranges 的 curriculum。 |
| Real2Sim | “用真实数据缩小差距” | 学一个 residual，让 sim 模仿真实 rollouts。 |

## 延伸阅读

- [Tobin et al. (2017). Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World](https://arxiv.org/abs/1703.06907) — 原始 DR 论文（机器人视觉）。
- [Peng et al. (2018). Sim-to-Real Transfer of Robotic Control with Dynamics Randomization](https://arxiv.org/abs/1710.06537) — dynamics 的 DR，四足 locomotion。
- [OpenAI et al. (2019). Solving Rubik's Cube with a Robot Hand](https://arxiv.org/abs/1910.07113) — Dactyl，大规模 ADR。
- [Miki et al. (2022). Learning robust perceptive locomotion for quadrupedal robots in the wild](https://www.science.org/doi/10.1126/scirobotics.abk2822) — ANYmal 的 teacher-student。
- [Makoviychuk et al. (2021). Isaac Gym: High Performance GPU Based Physics Simulation for Robot Learning](https://arxiv.org/abs/2108.10470) — 驱动 2025–2026 部署的大规模并行 sim。
- [Akkaya et al. (2019). Automatic Domain Randomization](https://arxiv.org/abs/1910.07113) — ADR curriculum 方法。
- [Sutton & Barto (2018). Ch. 8 — Planning and Learning with Tabular Methods](http://incompleteideas.net/book/RLbook2020.pdf) — Dyna framing（用 model 做 planning + rollouts），支撑现代 sim-to-real pipelines。
- [Zhao, Queralta & Westerlund (2020). Sim-to-Real Transfer in Deep Reinforcement Learning for Robotics: a Survey](https://arxiv.org/abs/2009.13303) — sim-to-real 方法分类与 benchmark 结果综述。
