# Self-Refine 和 CRITIC：迭代式输出改进

> Self-Refine（Madaan et al., 2023）让一个 LLM 在 loop 中扮演三个角色：generate、feedback、refine。平均收益：7 个任务上 absolute +20。CRITIC（Gou et al., 2023）通过把 verification 路由到外部工具来强化 feedback 步骤。2026 年，这个模式在每个框架中都以“evaluator-optimizer”（Anthropic）或 guardrail loop（OpenAI Agents SDK）的形式出现。

**类型：** 构建
**语言：** Python (stdlib)
**前置要求：** 阶段 14 · 01（Agent Loop），阶段 14 · 03（Reflexion）
**时间：** ~60 分钟

## 学习目标

- 说出 Self-Refine 的三个 prompts（generate、feedback、refine），并解释为什么 history 对 refine prompt 很重要。
- 解释 CRITIC 的关键洞见：没有外部 grounding 时，LLM 不擅长 self-verification。
- 用 stdlib 实现一个 Self-Refine loop，包含 history 和可选 external verifier。
- 把这个模式映射到 Anthropic 的“evaluator-optimizer” workflow，以及 OpenAI Agents SDK 的 output guardrails。

## 问题

一个 agent 产出的答案几乎正确。也许一行代码有 syntax error。也许 summary 太长。也许 plan 漏掉一个 edge case。你想要的是：agent 先 critique 自己的输出，然后修复它。

Self-Refine 证明了单个模型也能做到这件事，不需要训练数据，不需要 RL。但有一个问题：LLM 在硬事实上不擅长 self-verification。CRITIC 命名了修复方法：把 verify 步骤路由到外部工具（search、code interpreter、calculator、test runner）。

这两篇论文共同定义了 2026 年 iterative improvement 的默认模式：generate、verify（能外部验证就外部验证）、refine，在 verifier 通过时停止。

## 概念

### Self-Refine（Madaan et al., NeurIPS 2023）

一个 LLM，三个角色：

```
generate(task)            -> output_0
feedback(task, output_0)  -> critique_0
refine(task, output_0, critique_0, history) -> output_1
feedback(task, output_1)  -> critique_1
refine(task, output_1, critique_1, history) -> output_2
...
stop when feedback says "no issues" or budget exhausted.
```

关键细节：`refine` 会看到完整 history，包括所有之前的 outputs 和 critiques，因此不会重复犯错。论文对此做了 ablation：去掉 history，quality 会大幅下降。

Headline：在 7 个任务（math、code、acronym、dialog）上平均 absolute improvement +20，包括 GPT-4。无训练、无外部工具、单模型。

### CRITIC（Gou et al., arXiv:2305.11738, v4 Feb 2024）

Self-Refine 的弱点：feedback step 是 LLM 给自己打分。对 factual claims 来说这不可靠（hallucination 对生成它的模型来说也常常很有说服力）。CRITIC 用 `verify(task, output, tools)` 替换 `feedback(task, output)`，其中 `tools` 包括：

- 用于 factual claims 的 search engine。
- 用于 code correctness 的 code interpreter。
- 用于 arithmetic 的 calculator。
- Domain-specific verifiers（unit tests、type checkers、linters）。

Verifier 会生成基于 tool results 的 structured critique。Refiner 再以这段 critique 为条件。

Headline：CRITIC 在 factual tasks 上优于 Self-Refine，因为 critique 有 grounding。在没有 external verifiers 的任务上（creative writing、formatting），CRITIC 退化为 Self-Refine。

### 停止条件

两种常见形状：

1. **Verifier passes。** 外部 test 返回 success。有条件时首选（unit tests、type checker、guardrail assertion）。
2. **No feedback issued。** 模型说“the output is fine”。更便宜但不可靠；要搭配 max-iteration cap。

2026 年默认：组合使用。“Stop if verifier passes OR model says fine AND iterations >= 2 OR iterations >= max_iterations。”

### Evaluator-Optimizer（Anthropic, 2024）

Anthropic 2024 年 12 月的文章把它命名为五种 workflow patterns 之一。两个角色：

- Evaluator：给 output 打分并产生 critique。
- Optimizer：根据 critique 修订 output。

Loop 直到 evaluator 通过。这就是 Anthropic 框架中的 Self-Refine/CRITIC。Anthropic 额外补充的关键工程细节：evaluator 和 optimizer prompts 应该有明显不同的结构，这样模型不会只是 rubber-stamp。

### OpenAI Agents SDK output guardrails

OpenAI Agents SDK 把这个模式作为“output guardrails”提供。Guardrail 是一个在 agent final output 上运行的 validator。如果 guardrail 触发（抛出 `OutputGuardrailTripwireTriggered`），输出会被拒绝，agent 可以重试。Guardrails 可以调用工具（CRITIC-style），也可以是 pure functions（Self-Refine-style）。

### 2026 年的坑

- **Rubber-stamp loops。** 同一个模型用同一种 prompt 风格做 generation 和 critique，会收敛到“looks good to me”。使用结构明显不同的 prompts，或者用更小更便宜的模型做 critique。
- **Over-refinement。** 每次 refine pass 都会增加 latency 和 tokens。预算 1-3 次；再之后升级到 human review。
- **CRITIC on trivial tasks。** 如果没有 external verifier，CRITIC 会退化为 Self-Refine；不要为 stub verifier 支付 latency。

## 构建它

`code/main.py` 在一个 toy task 上实现 Self-Refine 和 CRITIC：给定 topic 生成短 bullet list。Verifier 检查格式（3 个 bullets，每个少于 60 chars）。CRITIC 添加一个外部“fact verifier”，会惩罚已知 hallucinations。

组件：

- `generate`：scripted producer。
- `feedback`：LLM-style self-critique。
- `verify_external`：CRITIC-style grounded verifier。
- `refine`：根据 history 重写 output。
- Stop condition：verifier passes 或 max 4 iterations。

运行它：

```
python3 code/main.py
```

对比 Self-Refine 和 CRITIC runs。CRITIC 会抓住 Self-Refine 漏掉的 factual error，因为 external verifier 拥有 self-critic 没有的 grounding。

## 使用它

Anthropic 的 evaluator-optimizer 是 Claude-friendly 语言里的这个模式。OpenAI Agents SDK 的 output guardrails 是 CRITIC-shaped（guardrails 可以调用工具）。LangGraph 提供一个读起来像 Self-Refine 的 reflection node。Google 的 Gemini 2.5 Computer Use 添加了 per-step safety evaluator，这是 CRITIC 的一种变体：每个 action 在 commit 前都会被验证。

## 发布它

`outputs/skill-refine-loop.md` 会根据 task shape、verifier availability 和 iteration budget 配置 evaluator-optimizer loop。它会生成 generator、evaluator/verifier、optimizer 的 prompts，以及 stop policy。

## 练习

1. 用 max_iterations=1 运行 toy。CRITIC 仍然有帮助吗？
2. 把 external verifier 换成 noisy verifier（随机 30% false positives）。Loop 会做什么？这就是 2026 年大多数 guardrail stacks 的现实。
3. 实现“generator-critic on different models”变体：大模型生成，小模型 critique。它能胜过 same-model 吗？
4. 阅读 CRITIC 第 3 节（arXiv:2305.11738 v4）。说出三类 verification-tool categories，并各举一个例子。
5. 把 OpenAI Agents SDK 的 `output_guardrails` 映射到 CRITIC 的 verifier role。SDK 做错了什么？做对了什么？

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|----------------|------------------------|
| Self-Refine | “LLM that fixes itself” | 单模型中的 generate -> feedback -> refine loop，带 history |
| CRITIC | “Tool-grounded verification” | 用外部 verifier（search、code、calc、tests）替换 feedback |
| Evaluator-Optimizer | “Anthropic workflow pattern” | 两个角色：evaluator 打分，optimizer 修订；loop 到收敛 |
| Output guardrail | “Post-hoc check” | OpenAI Agents SDK 在 agent 产出 output 后运行的 validator |
| Verify step | “Critique phase” | 承重决策：grounded 还是 self-rated |
| Refine history | “What the model already tried” | 前置到 refine prompt 的 prior outputs + critiques；去掉后质量崩塌 |
| Rubber-stamp loop | “Self-agreement failure” | 同 prompt critique 返回“looks good”；用结构不同的 prompts 修复 |
| Stop condition | “Convergence test” | Verifier passes 或 no feedback 且达到 iteration cap；永远不要只用单一条件 |

## 延伸阅读

- [Madaan et al., Self-Refine (arXiv:2303.17651)](https://arxiv.org/abs/2303.17651)：经典论文
- [Gou et al., CRITIC (arXiv:2305.11738)](https://arxiv.org/abs/2305.11738)：tool-grounded verification
- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents)：evaluator-optimizer workflow pattern
- [OpenAI Agents SDK docs](https://openai.github.io/openai-agents-python/)：作为 CRITIC-shaped verifiers 的 output guardrails
