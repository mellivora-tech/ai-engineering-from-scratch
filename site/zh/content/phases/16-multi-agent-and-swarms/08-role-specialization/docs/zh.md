# Role Specialization — Planner、Critic、Executor、Verifier

> 2026 年最常见的 multi-agent decomposition：一个 agent 负责 plan，一个 execute，一个 critique 或 verify。MetaGPT（arXiv:2308.00352）把它形式化为编码进 role prompts 的 SOPs — Product Manager、Architect、Project Manager、Engineer、QA Engineer — 遵循 `Code = SOP(Team)`。ChatDev（arXiv:2307.07924）通过 “chat chain” 串起 designer、programmer、reviewer、tester，并加入 “communicative dehallucination”（agents 显式请求缺失细节）。verifier 是承重角色：Cemri et al.（MAST, arXiv:2503.13657）显示，每个 multi-agent failure 都可以追溯到缺失或损坏的 verification。PwC 报告称，在 CrewAI 中加入 structured validation loops，使 accuracy 从 10% 到 70%，提升 7×。

**类型：** 学习 + 构建
**语言：** Python（stdlib）
**前置要求：** 阶段 16 · 04（Primitive Model），阶段 16 · 05（Supervisor）
**时间：** ~60 分钟

## 问题

Generic multi-agent systems 产生 generic output。三个 coders 在 group chat 中写出三种同样平庸的 code。你可以加更多 agents、更多 rounds，仍然达不到质量门槛。

修复不是更多 agents，而是 *不同* agents。分配不同角色。给 critic planner 没有的 tools。给 verifier objective test suite。现在系统拥有带 grounded correction 的 internal disagreement，而不只是 parallel guessing。

## 概念

### 四个 canonical roles

**Planner。** 读取 goal，产出 step list 或 spec。Tools：knowledge retrieval、docs。Output：structured plan。

**Executor。** 一次读取一个 plan step，产出 artifact。Tools：真正做事的 tools（code compiler、shell、API client）。Output：artifact。

**Critic。** 根据 planner intent 读取 executor output。Tools：对 artifact 的 read-only access、static analysis。Output：accept/reject with reasons。

**Verifier。** 读取 artifact 并运行 deterministic check。Tools：test runner、type checker、schema validator。Output：pass/fail with evidence。

Critic 是 subjective、opinionated，通常基于 LLM。Verifier 是 objective、deterministic，通常基于 code。它们不是同一个角色。

### MetaGPT 的 SOP pattern

MetaGPT（arXiv:2308.00352）把 software engineering SOPs 编码为 role prompts：

- **Product Manager** 写 PRD。
- **Architect** 产出 system design。
- **Project Manager** 拆分 tasks。
- **Engineer** 实现。
- **QA Engineer** 运行 tests。

每个 role 都有严格 input/output schema。role prompt 说明 role *是什么* 以及 *必须产出什么*。`Code = SOP(Team)` formulation — deterministic SOPs 把一队 LLMs 变成可预测 pipeline。

### ChatDev 的 communicative dehallucination

ChatDev 加入一个关键动作：当 executor 需要 plan 中没有给出的具体细节时，它会在继续前明确询问 designer。这防止了 LLM 经典失败：合理地编造细节。

实现方式：role prompt 中包含 “when you need specific information you were not given, ask the relevant role by name before producing output.”

### 为什么 verifier 最重要

Cemri et al.（MAST）追踪了 1642 个 multi-agent execution failures。21.3% 是 verification gaps — 系统发布了无人检查的答案。剩下 79% 往往也能追溯到“有个 check 失败后沉默了，或根本没有运行”。Verification 是承重角色。

PwC 报告称（CrewAI deployments, 2025），加入 structured validation loop 让 accuracy 从 10% 到 70%。一个角色带来 7× gain。

### Critic vs verifier

- critic 是 LLM 对 artifact 做质量 review。Subjective。会被 plausible prose 欺骗。
- verifier 是对 artifact 运行的 deterministic program。Objective。给出 pass/fail with evidence。

两个都用。Critic 捕捉 verifier 无法表达的 taste issues。Verifier 捕捉 critic 看不到的 bugs，因为它们只会在 runtime 显现。

### Anti-pattern

系统中每个 role 都是 LLM，且每个 role 的 output 都是 “looks good to me”。经典 MAST failure mode。至少添加一个由 code 而不是 LLM 决定 pass/fail 的 verifier。

### Framework mappings

- **CrewAI** — `Agent(role, goal, backstory)` 是教科书 specialization surface。
- **LangGraph** — nodes 可以有 specialized prompts；edges 强制 pipeline。
- **AutoGen** — GroupChat 中带 one-word names 的 role-specific ConversableAgents。
- **OpenAI Agents SDK** — role-specialized Agents 之间的 handoff tools。

## 构建它

`code/main.py` 实现一个构建简单 Python function 的 4-role pipeline：

- **Planner** 产出 spec。
- **Executor** 生成 code string。
- **Critic**（LLM-simulated）标记明显问题。
- **Verifier** 在 sandbox（`exec`）中运行生成 code，并使用 test case 检查。

Demo 运行两次：一次 executor 生成正确 code（critic + verifier 都通过），一次 executor 生成 off-spec code（critic 因为看起来合理而错过 bug，verifier 因 test fail 捕捉到它）。

运行：

```
python3 code/main.py
```

## 使用它

`outputs/skill-role-designer.md` 接收 task 并产出 role roster（3-5 roles）、每个 role 的 input/output schema，以及 verifier check。在把 agents 接入 framework 前使用。

## 发布它

Checklist：

- **至少一个 deterministic verifier。** 永远不要 all-LLM。
- **每个 role 都有显式 I/O schema。** planner 返回 spec，而不是 prose；executor 读取该 schema。
- **Communicative dehallucination。** 当信息缺失时 executor 必须问 planner；永远不要编造。
- **Critic/verifier ordering。** 先跑 critic（便宜，捕捉 design issues），再跑 verifier（慢，捕捉 bugs）。
- **Loop budget。** critic-executor revision rounds 最多 2 轮，然后 escalate to human。

## 练习

1. 运行 `code/main.py`，观察 verifier 如何捕捉 critic 错过的 bug。添加一个 static-analysis check（统计 `return` 出现次数）作为额外 verifier。它捕捉了 runtime test 错过的什么？
2. 添加第 5 个 role：“requirements analyst”，把 user wish 翻译成 planner-ready spec。哪些 communicative dehallucination requests 应该流向它？
3. 阅读 MetaGPT Section 3（“Agents”）。列出 MetaGPT 5 个 roles 的 input/output schema。
4. 阅读 ChatDev 的 chat-chain diagram（arXiv:2307.07924 Figure 3）。识别 communicative dehallucination 在哪里打破了一个本来会无限循环的 loop。
5. PwC 的 7× accuracy gain 来自 verification loops。假设三个添加 verifier 没有帮助的任务 — deterministic correctness checking 不可能或成本过高。

## 关键词汇

| 术语 | 人们常说 | 实际含义 |
|------|----------------|------------------------|
| Role specialization | “不同 agents，不同 jobs” | 针对 planner/executor/critic/verifier roles 调优的不同 system prompts。 |
| SOP pattern | “编码的 standard operating procedure” | MetaGPT 的 framing：严格 I/O schemas per role 把团队变成 pipeline。 |
| Communicative dehallucination | “编造前先问” | ChatDev pattern：executor 信息缺失时问 planner，而不是自己补。 |
| Critic | “LLM reviewer” | Subjective、opinionated reviewer。捕捉 taste issues。会被 plausible prose 欺骗。 |
| Verifier | “Deterministic check” | Code-based pass/fail。Test runner、type checker、schema validator。不会被说辞骗过。 |
| Verification gap | “无人检查” | MAST failures 的 21.3%。答案在未被 check 的情况下发布。 |
| Revision loop | “Critic 打回去” | critic rejection 触发 executor 带 feedback 重跑。需要 budget。 |
| All-LLM anti-pattern | “Looks good to me” | 每个 role 都是 LLM，没有 deterministic check。经典 MAST failure。 |

## 延伸阅读

- [Hong et al. — MetaGPT: Meta Programming for Multi-Agent Collaboration](https://arxiv.org/abs/2308.00352) — SOP-as-role-prompt 参考论文
- [Qian et al. — Communicative Agents for Software Development (ChatDev)](https://arxiv.org/abs/2307.07924) — chat chain + communicative dehallucination
- [Cemri et al. — Why Do Multi-Agent LLM Systems Fail?](https://arxiv.org/abs/2503.13657) — MAST taxonomy；verification gaps 是 21.3% failures
- [CrewAI docs — Agent roles](https://docs.crewai.com/en/introduction) — production role specification surface
