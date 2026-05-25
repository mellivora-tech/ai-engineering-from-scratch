# Tree of Thoughts 和 LATS：Deliberate Search

> 单条 chain-of-thought trajectory 没有回溯空间。ToT（Yao et al., 2023）把 reasoning 变成一棵树，并在每个 node 上做 self-evaluation。LATS（Zhou et al., 2024）在 Monte Carlo Tree Search 下统一了 ToT、ReAct 和 Reflexion。Game of 24 从 4%（CoT）提升到 74%（ToT）；LATS 在 HumanEval 上达到 92.7% pass@1。

**类型：** 构建
**语言：** Python (stdlib)
**前置要求：** 阶段 14 · 01（Agent Loop），阶段 14 · 03（Reflexion）
**时间：** ~75 分钟

## 学习目标

- 把 reasoning 表述为 search：nodes 是“thoughts”，edges 是“expansions”，value 是“有多 promising”。
- 用 stdlib 实现一个 ToT-style BFS tree search，带 self-evaluation scoring。
- 扩展到 toy LATS MCTS loop，包含 select / expand / simulate / backpropagate。
- 判断什么时候 search 值得付 token multiplier（Game of 24、code generation），什么时候单条 trajectory 足够（简单 Q&A）。

## 问题

Chain-of-thought 是线性行走。如果第一步错了，后续每一步都在坏前提上继续。Game of 24（用四个数字和 + - * / 得到 24）中，GPT-4 CoT accuracy 只有 4%。模型很早选错 subexpression，之后无法恢复。

Reasoning 需要的是提出多个候选、评估它们、选择 promising ones，并在出现 dead ends 时回溯的能力。这就是 search。Tree of Thoughts 和 LATS 是两个经典表述。

## 概念

### Tree of Thoughts（Yao et al., NeurIPS 2023）

每个 node 是一个连贯的中间步骤（“a thought”）。每个 node 可以扩展出 K 个 child thoughts。LLM 用 scoring prompt 对每个 node 做 self-evaluation。Search 探索这棵树，可以是 BFS、DFS 或 beam。

```
                     (root: "find 24 from 4 6 4 1")
                    /               |            \
           ("6 - 4 = 2")    ("4 + 1 = 5")    ("4 * 6 = 24")  <- Score: HIGH
              /   \              |                  |
          ...    ...          ...                finish
```

Self-evaluation 是承重部分。论文展示了三种变体：`sure / likely / impossible` 分类、`1..10` numeric score，以及候选之间投票。三者在 Game of 24 上都明显优于 CoT（GPT-4 上从 4% 到 74%）。

### LATS（Zhou et al., ICML 2024）

LATS 在 MCTS 下统一了 ToT、ReAct 和 Reflexion。LLM 扮演三个角色：

- **Policy**：提出候选 next actions（ReAct-style）。
- **Value function**：给 partial trajectory 打分（ToT-style self-eval）。
- **Self-reflector**：失败时写一段 natural-language reflection（Reflexion-style），并用它重新播种 future rollouts。

Environment feedback（observations）会混入 value function，让 search 由真实工具结果而不仅是模型意见来提供信息。论文发布时的结果：HumanEval pass@1 92.7%，使用 GPT-4（SOTA）；WebShop average 75.9，使用 GPT-3.5（接近 gradient-based fine-tuning）。

### 最小化理解 MCTS

每次 iteration 有四个阶段：

1. **Select**：使用 UCT（upper confidence bound for trees）从 root 走到 leaf。
2. **Expand**：通过 policy 生成 K 个 children。
3. **Simulate**：从一个 child 开始用 policy rollout，用 value function（或 environment reward）给 leaf 打分。
4. **Backpropagate**：沿路径向上更新 visit counts 和 value estimates。

UCT 公式：`Q(s, a) + c * sqrt(ln N(s) / N(s, a))`。第一项是 exploitation，第二项是 exploration。按任务调 `c`。

### 成本现实

Search 会让 tokens 爆炸。Game of 24 上的 ToT 使用了 CoT 的 100-1000 倍 tokens。LATS 类似。这不是免费的；把 search 留给：

- 单条 trajectory 明显不够的任务（Game of 24、complex code）。
- 正确性比 wall-clock 更重要的任务。
- 有便宜且可靠 value function 的任务（code 的 unit tests、math 的 explicit target）。

如果你的任务有唯一正确答案，但 evaluator 很嘈杂，search 经常会让情况更糟，因为它会找到一个“得分高”的错误答案。

### 2026 年定位

大多数生产 agents 不运行 LATS。它们运行 ReAct 加 tool-grounded verification（CRITIC，第 05 课）。Search 出现在专门场景：

- 把 tests 作为 value function 的 coding agents（HumanEval-style）。
- 探索多条 query paths 的 deep-research agents。
- LangGraph subgraphs 内部的 planning-heavy workflows。

AlphaEvolve（第 11 课）是 2025 年的极端形态：对代码做 evolutionary search、machine-checkable fitness、frontier gains（56 年来首次 4x4 matmul 改进）。

## 构建它

`code/main.py` 实现了：

- 一个 tiny ToT BFS，运行在 stylized “pick arithmetic ops” 任务上。
- 同一任务上的 toy LATS MCTS loop（Select / Expand / Simulate / Backpropagate），使用 UCT selection。
- 一个 value function，组合 symbolic score 和 self-eval score。

运行它：

```
python3 code/main.py
```

Trace 会显示 ToT 用 BFS 每个 node 扩展三个候选，并和 LATS 通过 MCTS 收敛到最佳 rollout 做对比。两者都会打印 token counts。

## 使用它

LangGraph 把 ToT-style exploration 作为 subgraph patterns 提供；LangChain 团队关于 LATS 的博客（2024 年 5 月）是参考教程。LlamaIndex 提供 `TreeOfThoughts` agent。对大多数 2026 年生产 agents 来说，这个模式藏在 `if task_complexity > threshold: use_search()` gate 后面，见第 05 课的 evaluator-optimizer pattern。

## 发布它

`outputs/skill-search-policy.md` 会根据 task shape、budget 和 evaluator fidelity，在 linear ReAct、ToT、LATS 和 evolutionary search 之间选择。

## 练习

1. 用 UCT c=0.1 和 c=2.0 分别运行 toy LATS。Trace 中会有什么变化？
2. 把 value function 换成更 noisy 的 scorer（加入 random jitter）。MCTS 还能找到最佳 leaf 吗？它能容忍的最小 signal-to-noise 是多少？
3. 实现 beam-search ToT（每层保留 top-k），并和 BFS 对比。在 token budget 紧张时哪个更好？
4. 阅读 LATS 第 5.1 节。复现 HumanEval trajectory count：需要多少 rollouts 才达到报告的 pass@1？
5. 阅读 LATS 论文关于“when LATS helps less”的讨论。写一段 decision rule，把 task shape 映射到 search strategy。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|----------------|------------------------|
| Tree of Thoughts | “Branching CoT” | Yao et al.：带 self-evaluation 的 thought nodes 树 |
| LATS | “MCTS for LLMs” | Zhou et al.：在 MCTS 下统一 ToT + ReAct + Reflexion |
| UCT | “Upper confidence bound” | 平衡 exploitation（Q）和 exploration（ln N / n）的 select 公式 |
| Value function | “How good is this state” | Prompted LLM score 或 environment reward；反馈进 backprop |
| Policy | “Action proposer” | ReAct-style generator；发出候选 next thoughts/actions |
| Rollout | “Simulated trajectory” | 从一个 node 用 policy 走到 leaf，并用 value 打分 |
| Backpropagate | “Update ancestors” | 把 leaf reward 向上推回路径，更新 visit counts 和 Q |
| Search cost | “Token explosion” | Game of 24 上是 CoT 的 100-1000 倍；采用前先预算 |

## 延伸阅读

- [Yao et al., Tree of Thoughts (arXiv:2305.10601)](https://arxiv.org/abs/2305.10601)：经典论文
- [Zhou et al., LATS (arXiv:2310.04406)](https://arxiv.org/abs/2310.04406)：带 Reflexion feedback 的 MCTS
- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview)：用于 search 的 subgraph patterns
- [AlphaEvolve (arXiv:2506.13131)](https://arxiv.org/abs/2506.13131)：使用 programmatic evaluators 的 evolutionary search
