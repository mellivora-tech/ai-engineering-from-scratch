# Society of Mind 与 Multi-Agent Debate

> Minsky 1986 年的前提 — intelligence 是一个 specialists 的 society — 每十年都会被重新发现。2023 年，Du et al. 把它变成一个具体算法：多个 LLM instances 提出答案、阅读彼此答案、critique、update。经过 N 轮后，它们收敛到一个 consensus，在六个 reasoning 和 factuality tasks 上超过 zero-shot CoT 和 reflection。有两个发现很重要：**multiple agents** 和 **multiple rounds** 都独立贡献增益。society 胜过 single-agent monologue；multi-round exchange 胜过 one-shot voting。

**类型：** 学习 + 构建
**语言：** Python（stdlib）
**前置要求：** 阶段 16 · 04（Primitive Model）
**时间：** ~60 分钟

## 问题

Self-consistency — 对一个 model 采样多次然后取 majority answer — 是最便宜的 reasoning improvement。它有效，但很快饱和。你把 samples 翻倍，也可能看不到新的有意义提升。

Debate 打破了这种饱和。不是从一个 model 得到 N 个独立 samples，而是让 N 个 agents 阅读彼此 reasoning 并 revise。samples 之间的相关性下降（它们不再是 i.i.d.），而 convergence point 经常在 i.i.d. voting 自信地错时给出正确答案。

## 概念

### Du et al. 2023 算法

来自 arXiv:2305.14325（ICML 2024）：

1. N 个 agents 各自对问题产生 initial answer。
2. 对 round r = 2..R：每个 agent 看到其他 agents 的 round r-1 answers，并被要求“考虑这些，给出 updated answer”。
3. R 轮之后，对 final answers majority-vote。

论文在 MMLU、GSM8K、biographies、MATH 和 factuality benchmarks 上测试。Debate 一贯超过 CoT 和 Self-Reflection。

### 两个独立 knobs

同一篇论文的 ablations：

- **Agent count alone**（1 round，N 个 answers majority vote）在多数任务上超过 single-agent，但会 plateau。
- **Round count alone**（1 个 agent 看自己的 prior reasoning）帮助很小 — 这是 reflection 的已知弱点。
- **Both together** 产生大幅提升。多个 agents 之间的 multi-round exchange 驱动增益。

### 为什么它有效

两种机制：

1. **暴露给 disagreement。** 当一个 agent 看到另一个 agent 的 reasoning chain 得出不同 conclusion，它必须 justify 或 update。不管哪种，round r+1 的 context 都比 round r 更丰富。
2. **降低 correlated error。** 在 self-consistency 中，所有 samples 来自同一个 model，所以 errors 相关 — 你会平均出一个自信但错误的答案。不同 models 或不同 seeds 会 decorrelate。不同 *debated views* 会进一步 decorrelate。

### Heterogeneous debate

A-HMAD 和相关后续工作为不同 agents 使用 *不同 base models*。Llama + Claude + GPT debate 能降低 monoculture collapse（第 26 课），因为一个 model family 的 correlated errors 不会被其他 families 共享。

缺点：一个弱 model 参与 debate 可能把 consensus 拉向它的错误答案（见 “Should we be going MAD?”, arXiv:2311.17371）。

### NLSOM — 129-agent extension

Zhuge et al.（“Mindstorms in Natural Language-Based Societies of Mind,” arXiv:2305.17066）把这个 idea 扩展到 129-member societies。结果：specialization 和 self-organization 随规模涌现，并且系统在 visual question answering 等任务上超过 single-agent。

### Failure modes

- **Sycophancy cascade。** 所有 agents 都服从听起来最自信的 agent。debate 坍缩成最大声的声音。提示 adversarial roles（“one agent must argue the counter-position”）有帮助。
- **Topic drift。** 多轮 debates 会从原始问题漂移。缓解：每轮重新注入问题。
- **Compute blowup。** N agents × R rounds = N·R LLM calls，而且每次 context 都会增长。5-agent、5-round debate 是 25 次 calls，context 还越来越大。每个问题成本可能超过单次 CoT call 的 10×。

## 构建它

`code/main.py` 在一个 math question 上运行 3-agent × 3-round debate，每个 agent 以不同（可能错误）的答案开始。Agents 是 scripted — 每个 “updates” 会按 scripted confidence 对 neighbors' answers 加权平均。逐轮 log 中可以看到 convergence。

demo 展示两个关键 effects：

- 一轮 exchange 会让 agents 更接近正确答案。
- round 2 之后的额外 rounds 呈现 diminishing returns（匹配 Du et al. 的 plateau）。

运行：

```
python3 code/main.py
```

## 使用它

`outputs/skill-debate-configurator.md` 为新任务配置 debate：agent 数量、round 数量、heterogeneity（same model vs mixed）、role assignment（symmetric vs one-adversarial）。它也会在运行前估算 token cost。

## 发布它

如果你发布 debate：

- **把 rounds 限制在 3。** Du et al. 表明 3 rounds 捕获大多数增益。更多是成本，不是质量。
- **把 agents 限制在 5。** 超过 5，context bloat 和成本占主导。
- **默认 heterogeneous。** pool 中至少两个不同 base models。
- **Adversarial slot。** 一个 agent 被 prompt 成无论如何都要 disagree。打破 sycophancy。
- **记录每一轮。** 隐藏 intermediate rounds 的 debate systems 无法 debug 或 audit。

## 练习

1. 运行 `code/main.py`，然后把 round count 设为 5，观察 diminishing returns。哪一轮开始 additional convergence 停止？
2. 添加第四个 agent，角色是 adversarial：总是不同意当前 majority。这会破坏还是改善 convergence？
3. 绘制（print）每轮 agreement score（agents 在 majority answer 上的比例）。它何时达到 1.0？这等同于 “correct” 吗？
4. 阅读 Du et al. Section 4 ablations。用这段代码复现 “agents-only” vs “rounds-only” vs “both” 结果。
5. 阅读 “Should we be going MAD?”（arXiv:2311.17371），列出 round-robin 之外的两个 debate variants，例如 judge-led、chain-of-debate、adversarial。

## 关键词汇

| 术语 | 人们常说 | 实际含义 |
|------|----------------|------------------------|
| Society of Mind | “Minsky 的 idea” | intelligence 是 interacting specialists；1986 framing 现在通过 LLM debate operationalized。 |
| Multi-agent debate | “Agents argue” | N 个 agents 提出、互相 critique、R 轮 revise，然后 majority-vote。 |
| Consensus | “它们同意了” | 不是 epistemic truth，只是 majority-answer 上的比例。可能自信地错。 |
| Rounds | “Exchange steps” | 一轮 = 每个 agent 读其他 agents 并 update 一次。 |
| Heterogeneous debate | “混合 model families” | 使用不同 base models 来 decorrelate errors。 |
| Sycophancy cascade | “大家都同意最大声的人” | debate failure：agents 不管正确性，服从最自信 agent。 |
| NLSOM | “129-agent society” | Natural-language society of mind；Zhuge et al. 的规模化版本。 |
| Correlated error | “同 model，同 bug” | self-consistency 饱和的原因；跨不同 views 的 debate 能 decorrelate。 |

## 延伸阅读

- [Du et al. — Improving Factuality and Reasoning in Language Models through Multiagent Debate](https://arxiv.org/abs/2305.14325) — 参考论文，ICML 2024
- [Zhuge et al. — Mindstorms in Natural Language-Based Societies of Mind](https://arxiv.org/abs/2305.17066) — 129-agent NLSOM
- [Should we be going MAD? A Look at Multi-Agent Debate Strategies for LLMs](https://arxiv.org/abs/2311.17371) — benchmark debate variants
- [Debate project page](https://composable-models.github.io/llm_debate/) — Du et al. 的 code、demos 和 ablation details
