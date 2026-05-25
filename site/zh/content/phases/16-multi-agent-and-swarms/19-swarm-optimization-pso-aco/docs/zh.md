# 面向 LLMs 的 Swarm Optimization（PSO、ACO）

> Bio-inspired optimization 正在以 LLM 形式回归。**LMPSO**（arXiv:2504.09247）使用 PSO，其中每个 particle 的 velocity 是 prompt，LLM 生成下一个 candidate；在 structured-sequence outputs（math expressions、programs）上表现很好。**Model Swarms**（arXiv:2410.11163）把每个 LLM expert 当成 model-weight manifold 上的 PSO particle，在 9 个 datasets、12 个 baselines 上报告 **13.3% average gain**，每轮只用 200 instances。**SwarmPrompt**（ICAART 2025）混合 PSO + Grey Wolf 做 prompt optimization。**AMRO-S**（arXiv:2603.12933）是 ACO-inspired pheromone specialists，用于 multi-agent LLM routing — **4.7x speedup**、可解释 routing evidence、quality-gated asynchronous update，把 inference 与 learning 解耦。本课在 prompt parameter space 上实现 PSO，在 agent routing 上实现 ACO，并测量为什么这些 classical algorithms 适合 LLM era，以及什么时候不适合。

**类型：** 学习 + 构建
**语言：** Python（stdlib）
**前置要求：** 阶段 16 · 09（Parallel Swarm Networks），阶段 16 · 14（Consensus and BFT）
**时间：** ~75 分钟

## 问题

你有一个 prompt，在任务 eval 上得分 62%。你想改进它。naive 做法是 gradient-free manual tweaking，扩展很差。Reinforcement learning 需要 reward signals 和足够 rollouts 来训练。对 prompts 做 backprop 并不现实 — prompt 是 discrete string，不是 differentiable parameter。

Classical bio-inspired optimization — PSO 用于 continuous search spaces，ACO 用于 path selection — 正是为这种场景设计的：gradient-free、population-based、每次 evaluation 便宜。把它们与 LLMs 结合，用于 gradient-free search step，就得到出人意料实用的 optimizer。

同样 patterns 也适用于 multi-agent systems 中的 agent *routing*。ACO-style pheromone trail 记录哪个 agent 最适合哪个 task-type，让 router exploit trail，并让 pheromones 衰减以重新发现 routes。

## 概念

### PSO 复习（Kennedy & Eberhart 1995）

Particle Swarm Optimization：continuous search space 中的 particles population。每个 particle 有 position `x_i` 和 velocity `v_i`。每轮：

```
v_i <- w * v_i + c1 * r1 * (p_best_i - x_i) + c2 * r2 * (g_best - x_i)
x_i <- x_i + v_i
evaluate fitness(x_i)
update p_best_i if improved
update g_best if global best
```

其中 `p_best` 是 particle 自己的 best，`g_best` 是 swarm 的 best，`w, c1, c2` 是 inertia + cognitive + social weights，`r1, r2` 是 random factors。

### LLM outputs 上的 PSO — LMPSO

arXiv:2504.09247 为 LLM-generated structured outputs（math expressions、programs）适配 PSO。每个 particle 是一个 candidate output。Velocity 是一个 *prompt*，描述如何把 current output 修改得更接近 personal/global best。LLM 根据 velocity prompt 生成 new output。“inertia” 是类似 “make small incremental changes” 的 prompt。

它在这些情况下效果好：
- output 是 structured（parseable、evaluable）。
- fitness 是 automatic（test runs、arithmetic evaluation）。
- population 小（约 10-30 particles），让 total LLM calls 可控。

当 fitness 需要 human review 时效果不好 — per-iteration cost 会过高。

### Model Swarms

arXiv:2410.11163 把 PSO 从 output layer 搬到 *model* layer。每个 “particle” 是一个 expert LLM（parameters）。swarm 通过 gradient-free update 把 parameters 推向 collective best。报告：在 9 个 datasets、12 个 baselines 上平均提升 13.3%，每轮只需 200 instances。

关键 insight 是 LLM expert models 已经在 shared parameter manifold 上相近（adapter weights、LoRA deltas）。在这个 low-dimensional subspace 上做 PSO 便宜且有效。

### ACO 复习（Dorigo 1992）

Ant Colony Optimization：ants 遍历 graph；每条 path 有 pheromone trail。Ant move probabilities 按 pheromone strength 加权。完成任务的 ants 按 solution quality deposit pheromone。pheromone 随时间 decay。

### AMRO-S — agent routing 的 ACO

arXiv:2603.12933 将 ACO 用于 multi-agent routing。每个 task-type 是 “destination”；每个 agent 是 possible route。产出高质量 outputs 的 routes 获得更强 pheromones。关键贡献：

- **Interpretable routing evidence。** pheromone strength 是 human-readable signal。
- **Quality-gated asynchronous update。** 只有 quality checks 通过后才 update pheromones，从而 decouple inference from learning。
- **4.7x speedup** 在 multi-agent routing benchmark 上。

quality gate 很重要：没有它，fast-but-wrong agents 会积累 pheromone，系统会锁定坏 routes。

### 什么时候为 LLMs 使用 PSO / ACO

**使用 PSO 的情况：**
- search space 是 continuous，或映射到 continuous parameters（prompt embeddings、LoRA weights、numeric generation parameters）。
- fitness 便宜且 automatic。
- population 可以小（10-30）。

**使用 ACO 的情况：**
- 你有 routing 或 path-selection problem。
- decisions 会随时间 reinforce（相同 task types 会反复出现）。
- 你需要 routing decisions 的 interpretable evidence。

**不要使用二者的情况：**
- fitness 需要 human review（每轮太贵）。
- search space 是 PSO 不适合的 discrete combinatorial 结构（改用 genetic algorithms）。
- real-time decisions 需要严格 latency（PSO/ACO 相比 single-pass heuristics 收敛慢）。

### 为什么 bio-inspired 仍然能赢

Gradient-based methods 需要 differentiable signals。LLM outputs 和 routing decisions 不容易 differentiable。Pseudo-gradient methods（reinforcement-learned routers、DPO-style prompt tuners）有效，但需要昂贵 training。

PSO 和 ACO 只需要一个 *evaluator* function。如果你能给 candidate output 或 routing decision 打分，就能在这个 space 上 optimize。这让 applicability 门槛低得多。

### Practical limits

- **Population budget。** N particles × T iterations × per-eval cost。对于每 call ~$0.02 的 LLM eval，一个 20-particle PSO 跑 50 iterations 约 $20。提前规划。
- **Exploration vs exploitation。** pheromone decay rate 与 PSO inertia 是 tradeoff；decay 太快 → 忘记 solutions；太慢 → 卡在 early local optima。
- **Catastrophic drift。** fitness landscape shift（新 data distribution）时，两种算法都可能先 converge 后 diverge。监控 best-fitness stability。

## 构建它

`code/main.py` 实现：

- `LMPSO` — 在 numeric prompt parameters（temperature、top_k weights）上的 PSO。每个 particle 的 “LLM generation” 用 scripted fitness function 模拟。运行 30 iterations 并展示 g_best convergence。
- `AMRO_S` — ACO-style routing。3 个 agents、4 种 task types、pheromone matrix、100 个 routed tasks。打印 (task_type → agent choices) distribution over time，展示 trail formation。
- Comparison：random routing vs ACO routing 在同一 task stream 上。测量 quality 和 latency。

运行：

```
python3 code/main.py
```

预期输出：
- LMPSO：g_best fitness 在 30 iterations 内从 random 提升到 near-optimal。
- AMRO-S：pheromone table 稳定到每个 task-type 的正确 agent；ACO routing 在 quality 上比 random 高约 30-40%，并减少 latency（更少 retries）。

## 使用它

`outputs/skill-swarm-optimizer.md` 帮你在 LLM / agent optimization problems 中选择 PSO、ACO、genetic algorithms 和 gradient-based optimizers。

## 发布它

- **Start small。** 10-20 particles，20-50 iterations。只有 convergence curve 显示明确 gain 时再 scale up。
- **每轮记录 pheromones 或 g_best。** 没有 trail 的 swarm optimizer 很痛苦。
- **Quality-gate updates。** 尤其对 ACO routing：fast-and-wrong agents 不能积累 pheromone。
- **Distribution shift 时 reset decay。** eval distribution 变化时，老 pheromones 已 stale；reset 或临时加倍 decay rate。
- **限制 per-iteration cost。** 发出 cost-per-iteration metric。每轮 $500 只换 0.5% gain 的 PSO 不可发布。

## 练习

1. 运行 `code/main.py`。观察 LMPSO convergence。改变 population size 5、10、20、50。time-to-converge 在多大 size 饱和？
2. 实现 “catastrophic drift” experiment：iteration 30 后改变 fitness function。PSO 多快适应？reset `p_best` 有帮助吗？
3. 给 AMRO-S 添加 quality gate：只有 eval score > 0.7 的 runs 才 deposit pheromone。与未 gated 版本相比 convergence 如何变化？
4. 阅读 LMPSO（arXiv:2504.09247）。把论文中的 “velocity as a prompt” 映射回你的 numeric velocity。simulation 中丢失了什么，又保留了什么？
5. 阅读 AMRO-S（arXiv:2603.12933）。实现 decoupled “inference fast-path” 和 asynchronous pheromone update。这如何改变 sustained load 下的 system latency？

## 关键词汇

| 术语 | 人们常说 | 实际含义 |
|------|----------------|------------------------|
| PSO | “Particle Swarm Optimization” | Kennedy-Eberhart 1995。Population-based gradient-free optimizer。 |
| ACO | “Ant Colony Optimization” | Dorigo 1992。通过 pheromone trails 做 path/route optimization。 |
| LMPSO | “PSO with LLM generation” | arXiv:2504.09247。velocity 是 prompt；LLM 产生 candidates。 |
| Model Swarms | “expert weights 上的 PSO” | arXiv:2410.11163。在 model parameter subspace 上做 gradient-free update。 |
| AMRO-S | “agent routing 的 ACO” | arXiv:2603.12933。task-type × agent 上的 pheromone matrix。 |
| p_best / g_best | “Personal / global best” | per-particle 和 swarm-wide 找到的 best solutions。 |
| Pheromone | “Routing memory” | edge 上的 strength；随时间 decay；按 quality deposit。 |
| Quality-gated update | “只从好 runs 学习” | pheromone deposit 以 quality check 为条件。 |
| Catastrophic drift | “Distribution shift” | fitness landscape 改变；旧 p_best 和 pheromones 变 stale。 |

## 延伸阅读

- [Kennedy & Eberhart — Particle Swarm Optimization](https://ieeexplore.ieee.org/document/488968) — 1995 PSO paper
- [Dorigo — Ant Colony Optimization](https://www.aco-metaheuristic.org/about.html) — 1992 ACO foundations
- [LMPSO — Language Model Particle Swarm Optimization](https://arxiv.org/abs/2504.09247) — structured LLM outputs 的 PSO
- [Model Swarms — gradient-free LLM expert optimization](https://arxiv.org/abs/2410.11163) — model-weight subspace 上的 PSO
- [AMRO-S — ant-colony multi-agent routing](https://arxiv.org/abs/2603.12933) — 带 quality gate 的 pheromone-driven routing
