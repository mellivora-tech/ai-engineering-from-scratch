# Skill Libraries 和 Lifelong Learning（Voyager）

> Voyager（Wang et al., TMLR 2024）把 executable code 当作 skill。Skills 是 named、retrievable、composable 的，并由 environment feedback refine。这是 Claude Agent SDK skills、skillkit 和 2026 skill-library pattern 的参考架构。

**类型：** 构建
**语言：** Python (stdlib)
**前置要求：** 阶段 14 · 07（MemGPT），阶段 14 · 08（Letta Blocks）
**时间：** ~75 分钟

## 学习目标

- 说出 Voyager 的三个组件：automatic curriculum、skill library、iterative prompting，以及各自的作用。
- 解释为什么 Voyager 把 action space 设为 code，而不是 primitive commands。
- 用 stdlib 实现一个 skill library，包含 registration、retrieval、composition 和 failure-driven refinement。
- 把 Voyager 的模式映射到 2026 年 Claude Agent SDK skills 和 skillkit 生态。

## 问题

每个 session 都从头重建能力的 agents 会犯三类错误：

1. **Waste tokens。** 每个任务都重新 eliciting 同一套 reasoning。
2. **Lose progress。** Session A 学到的 correction 不能迁移到 session B。
3. **Fail on long-horizon composition。** 复杂任务需要 capability hierarchies；one-shot prompts 表达不了。

Voyager 的回答：把每个可复用能力当作一块 named code，存进 library，可按 similarity 检索，可和其他 skills 组合，并由 execution feedback refine。

## 概念

### 三个组件

Voyager（arXiv:2305.16291）围绕三个东西组织 agent：

1. **Automatic curriculum。** 一个 curiosity-driven proposer 根据 agent 当前 skill set 和 environment state 选择下一个任务。Exploration 是 bottom-up。
2. **Skill library。** 每个 skill 都是 executable code。任务成功时添加新 skill。Skills 通过 query-to-description similarity 检索。
3. **Iterative prompting mechanism。** 失败时，agent 接收 execution errors、environment feedback 和 self-verification output，然后 refine skill。

Minecraft 评估（Wang et al., 2024）：相较 baselines，unique items 多 3.3 倍，stone tools 快 8.5 倍，iron tools 快 6.4 倍，map traversal 长 2.3 倍。数字是 Minecraft-specific，但模式可以迁移。

### Action space = code

大多数 agents 发出 primitive commands。Voyager 发出 JavaScript functions。一个 skill 是：

```
async function craftIronPickaxe(bot) {
  await mineIron(bot, 3);
  await mineStick(bot, 2);
  await placeCraftingTable(bot);
  await craft(bot, 'iron_pickaxe');
}
```

它由 sub-skills 组合而成，按 description 和 embedding 存储，以 program 而不是 prompt 的形式检索。

这就是 2026 年 Claude Agent SDK skill：一块 named、retrievable code 加 instructions，agent 按需加载。

### Skill retrieval

新任务：“make a diamond pickaxe。”Agent：

1. Embeds task description。
2. 查询 skill library，找 top-k similar skills。
3. 检索 `craftIronPickaxe`、`mineDiamond`、`placeCraftingTable` 等。
4. 用检索到的 primitives + 新逻辑组合出 new skill。

这就是 MCP resources（阶段 13）和 Agent SDK skills 实现的模式：在 knowledge/code surface 上检索，并限定到当前任务。

### Iterative refinement

Voyager 的 feedback loop：

1. Agent 写一个 skill。
2. Skill 对 environment 运行。
3. 返回三种信号之一：`success`、`error`（带 stack trace）、`self-verification failure`。
4. Agent 用这个信号作为 context 重写 skill。
5. Loop 直到 success 或 max rounds。

这是 Self-Refine（第 05 课）应用于 code generation，并用 environment-grounded verification。CRITIC（第 05 课）是同样模式，只是 verifier 是 external tools。

### Curriculum 和 exploration

Voyager 的 curriculum module 会基于 agent 拥有什么、还没做过什么，提出“build a shelter near the lake”这类任务。Proposer 使用 environment state + skill inventory 选择略高于当前能力的任务，也就是 exploration sweet spot。

对生产 agents 来说，这会转化为“what's missing” operator：给定当前 skill library 和一个 domain，我们还没有覆盖哪些 skills？团队通常把它手动实现为 curriculum review。

### 这个模式会在哪里出错

- **Skill library rot。** 同一个 skill 以略有不同的 descriptions 添加 10 次。写入时 deduplication；retrieval 只返回一个。
- **Composed-skill drift。** Parent skill 依赖一个被 refined 的 child。给 skills version；pin 到 v1 的 parent 不会自动拿到 v3。
- **Retrieval quality。** Skill descriptions 上的 vector retrieval 在 library 超过几百条后会退化。补充 tag filters 和 hard constraints（“only skills with `category=tooling`”）。

## 构建它

`code/main.py` 实现了一个 stdlib skill library：

- `Skill`：name、description、code（字符串）、version、tags、dependencies。
- `SkillLibrary`：register、search（token overlap）、compose（deps 的 topological sort）和 refine（更新时 bump version）。
- 一个 scripted agent，注册三个 primitive skills，组合第四个，遇到 failure，然后 refine。

运行它：

```
python3 code/main.py
```

Trace 会显示 library writes、retrieval、composition、一次 failed execution 和 v2 refinement，也就是 Voyager loop 的端到端形态。

## 使用它

- **Claude Agent SDK skills**（Anthropic）：2026 年参考形态。每个 skill 都有 description、code 和 instructions；在 agent session 中按需加载。
- **skillkit**（npm: skillkit）：面向 32+ AI coding agents 的 cross-agent skill management。
- **Custom skill libraries**：domain-specific（data agents 的 SQL skills、infra agents 的 Terraform skills）。Voyager pattern 可以缩小使用。
- **OpenAI Agents SDK `tools`**：低端形态；每个 tool 都是一个轻量 skill。

## 发布它

`outputs/skill-skill-library.md` 会为任意 target runtime 生成 Voyager-shaped skill library，接好 registration、retrieval、versioning 和 refinement。

## 练习

1. 给 `compose()` 添加 dependency-cycle detector。当 skill A 依赖 B，而 B 又依赖 A 时会发生什么？Error 还是 warning？
2. 实现 per-skill version pinning。当 parent skill 组合 child `crafting@1` 时，`crafting@2` refinement 不能静默升级 parent。
3. 用 sentence-transformers embeddings（或 stdlib BM25 实现）替换 token-overlap retrieval。在 50-skill toy library 上测 retrieval@5。
4. 添加一个“curriculum” agent：给定当前 library 和 domain description，提出 5 个 missing skills。每周调用一次。
5. 阅读 Anthropic 的 Claude Agent SDK skill docs。把 toy library 移植到 SDK 的 skill schema。Discoverability 有什么变化？

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|----------------|------------------------|
| Skill | “Reusable capability” | named code + description，可按 similarity 检索 |
| Skill library | “Agent memory of how-to” | persistent store of skills，可搜索且可组合 |
| Curriculum | “Task proposer” | 由当前 capability gap 驱动的 bottom-up goal generator |
| Composition | “Skill DAG” | Skills invoking skills；执行时 topologically sorted |
| Iterative refinement | “Self-correcting loop” | Env feedback + errors + self-verification 反馈到下一版本 |
| Action-space-as-code | “Programmatic actions” | 发出 functions，而不是 primitive commands，用于 temporally extended behavior |
| Dedup on write | “Skill collapse” | 近重复 descriptions 合并为一个 canonical skill |

## 延伸阅读

- [Wang et al., Voyager (arXiv:2305.16291)](https://arxiv.org/abs/2305.16291)：原始 skill-library 论文
- [Claude Agent SDK overview](https://platform.claude.com/docs/en/agent-sdk/overview)：作为 2026 productization 的 skills
- [Anthropic, Building agents with the Claude Agent SDK](https://www.anthropic.com/engineering/building-agents-with-the-claude-agent-sdk)：实践中的 skills 和 subagents
- [Madaan et al., Self-Refine (arXiv:2303.17651)](https://arxiv.org/abs/2303.17651)：Voyager 下方的 refinement loop
