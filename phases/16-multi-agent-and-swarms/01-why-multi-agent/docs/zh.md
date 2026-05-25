# 为什么要 multi-agent？

> 一个 agent 会撞上墙。聪明的做法不是造一个更大的 agent，而是使用更多 agent。

**类型：** 学习
**语言：** TypeScript
**前置要求：** 阶段 14（Agent Engineering）
**时间：** ~60 分钟

## 学习目标

- 识别单 agent 上限（context 溢出、混合专业能力、顺序瓶颈），并说明什么时候把工作拆给多个 agents 是正确选择
- 比较 orchestration patterns（pipeline、parallel fan-out、supervisor、hierarchical），并为给定任务结构选择合适模式
- 设计一个 multi-agent system，具备清晰的角色边界、shared state 和 communication contract
- 分析 multi-agent 复杂性（延迟、成本、调试难度）与 single-agent 简洁性之间的取舍

## 问题

你在阶段 14 构建了一个单 agent。它能工作。它可以读文件、运行命令、调用 API，并对结果进行推理。然后你把它指向一个真实代码库：200 个文件、三种语言、依赖基础设施的测试，以及一个“写代码前先研究外部 API”的需求。

这个 agent 卡住了。不是因为 LLM 笨，而是因为这个任务超出了一个 agent loop 能承载的范围。context window 被文件内容填满。agent 忘了 40 次 tool call 前读过什么。它试图同时当 researcher、coder 和 reviewer，结果三个角色都做得很差。

这就是 single-agent ceiling。每当任务需要以下能力时，你都会撞上它：

- **超出单个 window 的 context** - 读 50 个文件会轻易超过 200k tokens
- **不同阶段需要不同专业能力** - research 所需的 prompting 与 code generation 不一样
- **可以并行发生的工作** - 既然可以同时读三个文件，为什么要按顺序读？

## 概念

### Single-Agent Ceiling

单 agent 是一个 loop、一个 context window、一个 system prompt。可以这样想：

```
┌─────────────────────────────────────────┐
│            SINGLE AGENT                 │
│                                         │
│  ┌───────────────────────────────────┐  │
│  │         Context Window            │  │
│  │                                   │  │
│  │  research notes                   │  │
│  │  + code files                     │  │
│  │  + test output                    │  │
│  │  + review feedback                │  │
│  │  + API docs                       │  │
│  │  + ...                            │  │
│  │                                   │  │
│  │  ██████████████████████ FULL ███  │  │
│  └───────────────────────────────────┘  │
│                                         │
│  One system prompt tries to cover       │
│  research + coding + review + testing   │
│                                         │
│  Result: mediocre at everything         │
└─────────────────────────────────────────┘
```

三件事会坏掉：

1. **Context 饱和** - tool results 不断堆积。到第 30 轮时，agent 已经消耗了 150k tokens 的文件内容、命令输出和先前推理。第 5 轮的关键细节会丢失。

2. **角色混淆** - 一个写着“你是 researcher、coder、reviewer 和 tester”的 system prompt，会产生一个半吊子 research、半吊子 coding、却从不真正完成 review 的 agent。

3. **顺序瓶颈** - agent 先读文件 A，再读文件 B，再读文件 C。三次串行 LLM calls。三次串行 tool executions。没有并行性。

### Multi-Agent 解法

拆分工作。给每个 agent 一个任务、一个 context window，以及一个为该任务调优过的 system prompt：

```
┌──────────────────────────────────────────────────────────┐
│                    ORCHESTRATOR                          │
│                                                          │
│  "Build a REST API for user management"                  │
│                                                          │
│         ┌──────────┬──────────┬──────────┐               │
│         │          │          │          │               │
│         ▼          ▼          ▼          ▼               │
│   ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐  │
│   │RESEARCHER│ │  CODER   │ │ REVIEWER │ │  TESTER  │  │
│   │          │ │          │ │          │ │          │  │
│   │ Reads    │ │ Writes   │ │ Checks   │ │ Runs     │  │
│   │ docs,    │ │ code     │ │ code     │ │ tests,   │  │
│   │ finds    │ │ based on │ │ quality, │ │ reports  │  │
│   │ patterns │ │ research │ │ finds    │ │ results  │  │
│   │          │ │ + spec   │ │ bugs     │ │          │  │
│   └─────┬────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘  │
│         │           │            │             │         │
│         └───────────┴────────────┴─────────────┘         │
│                          │                               │
│                     Merge results                        │
└──────────────────────────────────────────────────────────┘
```

每个 agent 都有：
- 一个聚焦的 system prompt（“你是 code reviewer。唯一任务是找 bug。”）
- 自己的 context window（不会被其他 agent 的工作污染）
- 清晰的 input/output contract（接收 research notes，输出 code）

### 真实系统中的例子

**Claude Code subagents** - 当 Claude Code 通过 `Task` 启动 subagent 时，会创建一个带有 scoped task 的 child agent。parent 保持 context 干净。child 专注完成工作，并返回 summary。

**Devin** - 运行 planner agent、coder agent 和 browser agent。planner 把工作拆成步骤。coder 写代码。browser 研究文档。每个都有独立 context。

**Multi-agent coding teams（SWE-bench）** - SWE-bench 上表现最好的系统使用 researcher 读取代码库，planner 设计修复方案，coder 实现。Single-agent 系统得分更低。

**ChatGPT Deep Research** - 并行启动多个 search agents，每个探索不同角度，然后综合结果。

### 光谱

Multi-agent 不是二元选择。它是一条光谱：

```
SIMPLE ──────────────────────────────────────────── COMPLEX

 Single        Sub-         Pipeline      Team         Swarm
 Agent         agents

 ┌───┐       ┌───┐        ┌───┐───┐    ┌───┐───┐    ┌─┐┌─┐┌─┐
 │ A │       │ A │        │ A │ B │    │ A │ B │    │ ││ ││ │
 └───┘       └─┬─┘        └───┘─┬─┘    └─┬─┘─┬─┘    └┬┘└┬┘└┬┘
               │                │        │   │       ┌┴──┴──┴┐
             ┌─┴─┐          ┌───┘───┐    │   │       │shared │
             │ a │          │ C │ D │  ┌─┴───┴─┐    │ state │
             └───┘          └───┘───┘  │  msg   │    └───────┘
                                       │  bus   │
 1 loop      Parent +      Stage by    │       │    N peers,
 1 context   child tasks   stage       └───────┘    emergent
                                       Explicit      behavior
                                       roles
```

**Single agent** - 一个 loop，一个 prompt。适合简单任务。

**Subagents** - parent 为聚焦子任务启动 children。parent 维护计划。children 回报结果。这就是 Claude Code 的做法。

**Pipeline** - agents 按顺序运行。Agent A 的输出成为 Agent B 的输入。适合分阶段 workflow：research -> code -> review -> test。

**Team** - agents 通过 shared message bus 并行运行。每个都有角色。orchestrator 负责协调。适合需要不同技能同时参与的任务。

**Swarm** - 大量相同或近似相同的 agents 共享 state。没有固定 orchestrator。agents 从 queue 中领取工作。适合高吞吐并行任务。

### 四种 Multi-Agent Patterns

#### Pattern 1: Pipeline

```
Input ──▶ Agent A ──▶ Agent B ──▶ Agent C ──▶ Output
          (research)  (code)      (review)
```

每个 agent 转换数据并向前传递。容易推理。一个阶段失败会阻塞后续所有阶段。

#### Pattern 2: Fan-out / Fan-in

```
                ┌──▶ Agent A ──┐
                │              │
Input ──▶ Split ├──▶ Agent B ──├──▶ Merge ──▶ Output
                │              │
                └──▶ Agent C ──┘
```

把工作拆给并行 agents，然后合并结果。适合可分解为独立子任务的工作。

#### Pattern 3: Orchestrator-Worker

```
                    ┌──────────┐
                    │  Orch.   │
                    └──┬───┬───┘
                  task │   │ task
                 ┌─────┘   └─────┐
                 ▼               ▼
           ┌──────────┐   ┌──────────┐
           │ Worker A │   │ Worker B │
           └──────────┘   └──────────┘
```

一个聪明的 orchestrator 决定要做什么，委派给 workers，并综合结果。orchestrator 本身也是 agent，带有启动 workers 的 tools。

#### Pattern 4: Peer Swarm

```
         ┌───┐ ◄──── msg ────▶ ┌───┐
         │ A │                  │ B │
         └─┬─┘                  └─┬─┘
           │                      │
      msg  │    ┌───────────┐     │ msg
           └───▶│  Shared   │◄────┘
                │  State    │
           ┌───▶│  / Queue  │◄────┐
           │    └───────────┘     │
      msg  │                      │ msg
         ┌─┴─┐                  ┌─┴─┐
         │ C │ ◄──── msg ────▶ │ D │
         └───┘                  └───┘
```

没有 central orchestrator。agents 点对点通信。决策从交互中涌现。更难调试，但能扩展到很多 agents。

### 什么时候不要使用 Multi-Agent

Multi-agent 会增加复杂性。agents 之间的每条消息都是潜在失败点。调试会从“读一个 conversation”变成“追踪五个 agents 之间的消息”。

**保持 single-agent 的情况：**
- 任务能放进一个 context window（working data 少于 ~100k tokens）
- 不需要为不同阶段使用不同 system prompts
- 顺序执行已经足够快
- 任务足够简单，拆分带来的开销大于收益

**复杂性成本：**
- 每个 agent boundary 都是有损压缩步骤：agent A 的完整 context 会被 summarization 成传给 agent B 的消息
- 协调逻辑（谁做什么、什么时候做、按什么顺序做）本身就是 bug 来源
- 延迟增加：N 个 agents 至少意味着 N 次串行 LLM calls，如果需要来回沟通会更多
- 成本倍增：每个 agent 都独立消耗 tokens

经验法则：如果一个任务少于 20 次 tool calls，并且能放进 100k tokens，就保持 single-agent。

## 构建它

### 第 1 步：过载的 Single Agent

下面是一个试图做所有事情的 single agent。它有一个巨大的 system prompt，以及一个同时保存 research、code 和 reviews 的 context window：

```typescript
type AgentResult = {
  content: string;
  tokensUsed: number;
  toolCalls: number;
};

async function singleAgentApproach(task: string): Promise<AgentResult> {
  const systemPrompt = `You are a full-stack developer. You must:
1. Research the requirements
2. Write the code
3. Review the code for bugs
4. Write tests
Do ALL of these in a single conversation.`;

  const contextWindow: string[] = [];
  let totalTokens = 0;
  let totalToolCalls = 0;

  const research = await fakeLLMCall(systemPrompt, `Research: ${task}`);
  contextWindow.push(research.output);
  totalTokens += research.tokens;
  totalToolCalls += research.calls;

  const code = await fakeLLMCall(
    systemPrompt,
    `Given this research:\n${contextWindow.join("\n")}\n\nNow write code for: ${task}`
  );
  contextWindow.push(code.output);
  totalTokens += code.tokens;
  totalToolCalls += code.calls;

  const review = await fakeLLMCall(
    systemPrompt,
    `Given all previous context:\n${contextWindow.join("\n")}\n\nReview the code.`
  );
  contextWindow.push(review.output);
  totalTokens += review.tokens;
  totalToolCalls += review.calls;

  return {
    content: contextWindow.join("\n---\n"),
    tokensUsed: totalTokens,
    toolCalls: totalToolCalls,
  };
}
```

这种做法的问题：
- context window 每个阶段都会增长。到 review 步骤时，它同时包含 research notes、code 和先前 reasoning。
- system prompt 很泛。它无法针对每个阶段调优。
- 没有任何东西能并行运行。

### 第 2 步：Specialist Agents

现在拆分它。每个 agent 只做一个任务：

```typescript
type SpecialistAgent = {
  name: string;
  systemPrompt: string;
  run: (input: string) => Promise<AgentResult>;
};

function createSpecialist(name: string, systemPrompt: string): SpecialistAgent {
  return {
    name,
    systemPrompt,
    run: async (input: string) => {
      const result = await fakeLLMCall(systemPrompt, input);
      return {
        content: result.output,
        tokensUsed: result.tokens,
        toolCalls: result.calls,
      };
    },
  };
}

const researcher = createSpecialist(
  "researcher",
  "You are a technical researcher. Read documentation, find patterns, and summarize findings. Output only the facts needed for implementation."
);

const coder = createSpecialist(
  "coder",
  "You are a senior TypeScript developer. Given requirements and research notes, write clean, tested code. Nothing else."
);

const reviewer = createSpecialist(
  "reviewer",
  "You are a code reviewer. Find bugs, security issues, and logic errors. Be specific. Cite line numbers."
);
```

每个 specialist 都有聚焦 prompt。每个都拿到干净的 context window，只包含它需要的输入。

### 第 3 步：通过消息协调

用显式 message passing 把 specialists 连接起来：

```typescript
type AgentMessage = {
  from: string;
  to: string;
  content: string;
  timestamp: number;
};

async function multiAgentApproach(task: string): Promise<AgentResult> {
  const messages: AgentMessage[] = [];
  let totalTokens = 0;
  let totalToolCalls = 0;

  const researchResult = await researcher.run(task);
  messages.push({
    from: "researcher",
    to: "coder",
    content: researchResult.content,
    timestamp: Date.now(),
  });
  totalTokens += researchResult.tokensUsed;
  totalToolCalls += researchResult.toolCalls;

  const coderInput = messages
    .filter((m) => m.to === "coder")
    .map((m) => `[From ${m.from}]: ${m.content}`)
    .join("\n");

  const codeResult = await coder.run(coderInput);
  messages.push({
    from: "coder",
    to: "reviewer",
    content: codeResult.content,
    timestamp: Date.now(),
  });
  totalTokens += codeResult.tokensUsed;
  totalToolCalls += codeResult.toolCalls;

  const reviewerInput = messages
    .filter((m) => m.to === "reviewer")
    .map((m) => `[From ${m.from}]: ${m.content}`)
    .join("\n");

  const reviewResult = await reviewer.run(reviewerInput);
  messages.push({
    from: "reviewer",
    to: "orchestrator",
    content: reviewResult.content,
    timestamp: Date.now(),
  });
  totalTokens += reviewResult.tokensUsed;
  totalToolCalls += reviewResult.toolCalls;

  return {
    content: messages.map((m) => `[${m.from} -> ${m.to}]: ${m.content}`).join("\n\n"),
    tokensUsed: totalTokens,
    toolCalls: totalToolCalls,
  };
}
```

每个 agent 只接收发给自己的消息。没有 context pollution。researcher 读文档产生的 50k tokens 不会进入 reviewer 的 context。

### 第 4 步：比较

```typescript
async function compare() {
  const task = "Build a rate limiter middleware for an Express.js API";

  console.log("=== Single Agent ===");
  const single = await singleAgentApproach(task);
  console.log(`Tokens: ${single.tokensUsed}`);
  console.log(`Tool calls: ${single.toolCalls}`);

  console.log("\n=== Multi-Agent ===");
  const multi = await multiAgentApproach(task);
  console.log(`Tokens: ${multi.tokensUsed}`);
  console.log(`Tool calls: ${multi.toolCalls}`);
}
```

multi-agent 版本使用更多总 tokens（三个 agents，三次独立 LLM calls），但每个 agent 的 context 保持干净。因为 system prompt 专门化，每个阶段的质量都会提升。

## 使用它

本课产出一个可复用 prompt，用来判断什么时候应该转向 multi-agent。见 `outputs/prompt-multi-agent-decision.md`。

## 练习

1. 添加第四个 specialist：一个 “tester” agent，从 coder 接收 code，从 reviewer 接收 review feedback，然后写 tests
2. 修改 pipeline，让 reviewer 能把 feedback 发回 coder 进入 revision loop（最多 2 轮）
3. 把顺序 pipeline 转成 fan-out：并行运行 researcher 和一个 “requirements analyzer” agent，然后合并输出再传给 coder

## 关键词汇

| 术语 | 人们常说 | 实际含义 |
|------|----------------|----------------------|
| Swarm | “AI agents 的蜂群智能” | 一组带 shared state、没有固定 leader 的 peer agents。行为从局部交互中涌现。 |
| Orchestrator | “老板 agent” | 一个 tools 包含启动和管理其他 agents 的 agent。它计划和委派，但不一定亲自做实际工作。 |
| Coordinator | “交通警察” | 一个非 agent 组件（通常只是代码，不是 LLM），根据规则在 agents 之间路由消息。 |
| Consensus | “agents 达成一致” | 多个 agents 必须在继续前达成 agreement 的 protocol。用于解决冲突输出。 |
| Emergent behavior | “agents 自己搞明白了” | 由 agent interactions 产生、但没有被显式编程的系统级模式。可能有益，也可能有害。 |
| Fan-out / fan-in | “agents 版 map-reduce” | 把任务拆给并行 agents（fan-out），再组合结果（fan-in）。 |
| Message passing | “agents 互相说话” | agents 之间的 communication mechanism：从一个 agent 发送到另一个 agent 的 structured data，用来替代 shared context windows。 |

## 延伸阅读

- [The Landscape of Emerging AI Agent Architectures](https://arxiv.org/abs/2409.02977) - multi-agent patterns 综述
- [AutoGen: Enabling Next-Gen LLM Applications](https://arxiv.org/abs/2308.08155) - Microsoft 的 multi-agent conversation framework
- [Claude Code subagents documentation](https://docs.anthropic.com/en/docs/claude-code) - Claude Code 如何通过 Task 委派
- [CrewAI documentation](https://docs.crewai.com/) - 基于角色的 multi-agent framework
