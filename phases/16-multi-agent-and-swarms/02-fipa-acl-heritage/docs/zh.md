# FIPA-ACL 与 Speech Acts 的传承

> 在 MCP 之前，在 A2A 之前，有 FIPA-ACL。2000 年，IEEE Foundation for Intelligent Physical Agents 批准了一种 agent communication language：二十种 performatives、两种 content languages，以及一组 interaction protocols，包括 contract net、subscribe/notify、request-when。它在工业界淡出，是因为 ontology overhead 对 Web 来说太重；但 multi-agent systems 的 LLM 复兴正在悄悄重新实现同一批思想，只是没有 formal semantics：JSON contracts 代替 performatives，natural language 代替 ontologies。本课认真阅读 FIPA-ACL，让你看清 2026 的 protocol decisions 哪些是重新发明，哪些是真正的新东西，以及当前浪潮将在哪里重新遇到 2000 年代已经解决过的问题。

**类型：** 学习
**语言：** Python（stdlib）
**前置要求：** 阶段 16 · 01（Why Multi-Agent）
**时间：** ~60 分钟

## 问题

2026 年的 agent-protocol landscape 很拥挤：MCP 面向 tools，A2A 面向 agents，ACP 面向 enterprise audit，ANP 面向 decentralized trust，NLIP 面向 natural-language content，还有 CA-MCP 和几十个 research proposals。每个 spec 都宣称自己是基础设施。

诚实地看，大多数都在重新发现一棵非常具体的、已有二十年历史的决策树。Austin（1962）和 Searle（1969）的 speech-act theory 给出了“utterances are actions”。KQML（1993）把它变成 wire protocol。FIPA-ACL（2000 年批准）形成了参考标准：二十种 performatives、SL0/SL1 content languages、contract-net 和 subscribe-notify 的 interaction protocols。JADE 和 JACK 是 Java reference platforms。这个努力在 2010 年左右淡出，因为 ontology overhead 太重，而 Web 正在赢得 stack。

当你看 MCP 的 `tools/call`、A2A 的 task lifecycle，或者 CA-MCP 的 shared context store 时，你看到的是 FIPA 决策的更柔软、JSON-native 的重写版。了解这段传承会告诉你两件事：哪些新的“创新”其实是重新发明，以及哪些旧 failure modes 新 specs 还会再发现一次。

## 概念

### 一段话理解 speech acts

Austin 注意到，有些句子不是在描述世界，而是在改变世界。“I promise.” “I request.” “I declare.” 他称这些为 performative utterances。Searle 将其形式化为五类：assertive、directive、commissive、expressive、declarative。KQML（Finin et al., 1993）把这个思想用于 software agents：一条 message 是 performative（动作）加 content（动作关于什么）。FIPA-ACL 清理了 KQML 的缺口，并围绕约二十种 performatives 标准化。

### 二十种 FIPA performatives（部分列表）

| Performative | Intent |
|---|---|
| `inform` | “我告诉你 P 为真” |
| `request` | “我请求你做 X” |
| `query-if` | “P 为真吗？” |
| `query-ref` | “X 的值是什么？” |
| `propose` | “我提议我们做 X” |
| `accept-proposal` | “我接受这个 proposal” |
| `reject-proposal` | “我拒绝这个 proposal” |
| `agree` | “我同意做 X” |
| `refuse` | “我拒绝做 X” |
| `confirm` | “我确认 P 为真” |
| `disconfirm` | “我否认 P” |
| `not-understood` | “你的 message 没有解析成功” |
| `cfp` | “针对 X 征集 proposals” |
| `subscribe` | “当 X 变化时通知我” |
| `cancel` | “取消正在进行的 X” |
| `failure` | “我尝试了 X 并失败了” |

完整列表在 `fipa00037.pdf`（FIPA ACL Message Structure）中。重点不是背下来，而是这些每一个都对应一个 LLM protocol 最终会重新加入的 primitive。

### 标准 FIPA-ACL message

```
(inform
  :sender       agent1@platform
  :receiver     agent2@platform
  :content      "((price IBM 83))"
  :language     SL0
  :ontology     finance
  :protocol     fipa-request
  :conversation-id   conv-42
  :reply-with   msg-17
)
```

七个字段承载 protocol envelope；一个字段（`content`）承载 payload。其余字段正是你每次给 JSON protocol 补 retries、threading 和 ontology 时都会重新发明的东西。

### 两个 legacy platforms

**JADE**（Java Agent DEvelopment framework，1999–2020s）是使用最广泛的 FIPA-compliant runtime。Agents 继承 base class，交换 ACL messages，在 containers 内运行，并用 “behaviors” 协调。interaction-protocol library 自带 contract-net、subscribe-notify、request-when 和 propose-accept。

**JACK**（Agent Oriented Software，商业）强调在 FIPA messages 之上的 BDI（Belief-Desire-Intention）reasoning。更 formal，采用更少。

一旦 Web stack 吞掉 multi-agent use cases，这两者都衰落了。MCP 和 A2A 是 2026 年的 runtime “containers”。

### 为什么 FIPA 淡出

- **Ontology overhead。** FIPA 要求共享 ontology 来解析 `content`。达成 ontologies 需要多年 standards process。Web 直接用了 HTTP + JSON。
- **没人使用的 formal semantics。** SL（Semantic Language）提供严格 truth conditions，但大多数 production systems 使用 free-form content 并忽略 formalism。
- **Tooling lock-in。** JADE 只支持 Java；JACK 是商业产品。Polyglot teams 绕开了两者。
- **互联网赢得了 stack。** REST，然后 JSON-RPC，然后 gRPC 替代了 ACL 的 transport。

### LLM 复兴是 FIPA-lite

比较 FIPA `request` 和 MCP `tools/call`：

```
(request                                {
  :sender  agent1                         "jsonrpc": "2.0",
  :receiver tool-server                   "method":  "tools/call",
  :content "(lookup stock IBM)"           "params":  {"name":"lookup_stock",
  :ontology finance                                   "arguments":{"symbol":"IBM"}},
  :conversation-id c42                    "id": 42
)                                        }
```

同样的 envelope，不同的 syntax。两者都承载：who、whom、intent、payload、correlation id。两者都不是相对另一方的革命，它们是在同一设计上的不同 trade-offs。

Liu et al. 2025 survey（“A Survey of Agent Interoperability Protocols: MCP, ACP, A2A, ANP”, arXiv:2505.02279）明确指出这条 lineage：MCP 对应 tool-use speech acts，A2A 对应 agent-peer speech acts，ACP 对应 audit-trail speech acts，ANP 对应 decentralized-identity extensions。新 specs 是带 JSON syntax 和更松散 semantics 的 ACL descendants。

### 取舍，直白地说

**FIPA 给了你但 modern specs 放弃的东西：**

- Formal semantics — 你可以证明 `inform` 表示 sender 相信 content。
- 一套 canonical performatives catalog — 不必重新争论“我们是否应该有 `cancel`？”。
- 数十年的 interaction-protocol patterns — contract-net、subscribe-notify、propose-accept — 并带有已知 correctness properties。

**Modern specs 给了你但 FIPA 没有的东西：**

- 与所有现代工具兼容的 JSON-native payloads。
- LLM 可以在没有 hand-coded ontology 的情况下解释 natural-language content。
- Web-stack transport（HTTP、SSE、WebSocket）。
- 通过 self-describing documents 做 capability discovery（MCP `listTools`、A2A Agent Card）。

更松散的 intent semantics 换来更容易的实现。这就是精确的 trade。

### 值得移植的 interaction protocols

FIPA 自带约 15 种 interaction protocols。有三种值得带进 LLM multi-agent systems：

1. **Contract Net Protocol（CNP）。** Manager 发出 `cfp`（call for proposals）；bidders 用 `propose` 响应；manager 接受/拒绝。这是 canonical task-market pattern（阶段 16 · 16 Negotiation）。
2. **Subscribe/Notify。** Subscriber 发送 `subscribe`；publisher 在 topic 变化时发送 `inform`。这就是 2026 年的每个 event-bus。
3. **Request-When。** “当条件 Y 成立时做 X。”带 pre-conditions 的 delayed-action。2026 的类比是 durable workflow engines 中的 deferred tasks（阶段 16 · 22 Production Scaling）。

每一种都能干净映射到 modern message queues、HTTP + polling 或 SSE streaming。

### 放弃 ontology 后会坏什么

没有 shared ontology，agents 会从 natural-language content 推断含义。2026 年有文档记录的 failure mode 是 **semantic drift**：两个 agents 对同一个词（`"customer"`）使用略有不同的概念，receiver agent 按错误解释行动，没有 schema validator 捕捉到它。FIPA 的 ontology requirement 会在 parse time 拒绝这条 message。

不走 full ontology 的缓解办法：

- `content` 上的 JSON Schema — 在 wire 上拒绝结构错误。
- Typed artifacts（A2A）— 拒绝错误 modality。
- envelope 中的显式 performative — 即使 content 是 natural language，也让 intent 不含糊。

### 2026 specs 与 speech-act heritage 的映射

| Modern spec | FIPA analog | 保留了什么 | 放弃了什么 |
|---|---|---|---|
| MCP `tools/call` | `request` | explicit intent, correlation id | formal semantics, ontology |
| MCP `resources/read` | `query-ref` | explicit intent, correlation id | formal semantics |
| A2A Task lifecycle | contract-net + request-when | async lifecycle, state transitions | formal completeness guarantees |
| A2A streaming events | subscribe/notify | async push | typed-predicate subscription |
| CA-MCP shared context | blackboard (Hayes-Roth 1985) | multi-writer shared memory | logical consistency model |
| NLIP | natural-language content | LLM-native | schema |

从上到下读这张表，模式是：保留 structural primitive，放弃 formalism，让 LLMs 掩盖 ambiguity。

## 构建它

`code/main.py` 实现了一个 pure-stdlib FIPA-ACL translator。它编码和解码 canonical ACL envelope，并展示每一种 MCP / A2A message shape 如何归约到同样的七个字段。demo：

- 将五条 MCP-style 和 A2A-style messages 编码成 FIPA-ACL。
- 将 FIPA-ACL 解码回现代等价形式。
- 用 `cfp`、`propose`、`accept-proposal`、`reject-proposal` 在一个 manager 和三个 bidders 之间运行一个 toy Contract Net negotiation。

运行：

```
python3 code/main.py
```

输出是一个 side-by-side trace，展示每条 modern message 的 2026 JSON 形式和 FIPA-ACL 形式，然后 round-trip 一个 contract-net bid。相同的 protocol primitives 在 round-trip 后仍然存在；只有 syntax 不同。

## 使用它

`outputs/skill-fipa-mapper.md` 是一个 skill：读取任意 agent-protocol spec 并生成 FIPA-ACL mapping。在采用新 protocol 前用它回答：“这真的新吗，还是只是带 JSON syntax 的 `inform`？”

## 发布它

不要把 FIPA-ACL 带回来。把它的 checklist 带回来：

- 每条 message 的 intent primitive（performative）是什么？
- request-response 和 cancellation 是否有 correlation id？
- 是否有显式 content language（JSON-RPC、plain text、structured typed artifact）？
- interaction protocols 是 first-class，还是你在从头重新实现 contract-net？
- 当两个 agents 对 content meaning 有分歧（semantic drift）时会发生什么？

在任何新 protocol 进入 production 之前，记录这五个问题。

## 练习

1. 运行 `code/main.py`。观察 round-trip encoding。识别哪个 FIPA performative 对应 `tools/call`、`resources/read` 和 A2A task creation。
2. 给 contract-net demo 扩展一个 `cancel` performative，让 manager 能在竞标中途撤回任务。`cancel` 解决了 retries 单独无法解决的什么 failure case？
3. 阅读 FIPA ACL Message Structure（http://www.fipa.org/specs/fipa00037/）4.1–4.3 节。选一个本课没有覆盖的 performative，并描述它的 modern JSON-RPC analog。
4. 阅读 Liu et al., arXiv:2505.02279。对 MCP、A2A、ACP、ANP 分别列出它们保留和放弃的 FIPA performative families。
5. 为你自己系统中的 `request` performative 的 `content` 字段设计一个最小 JSON-Schema。这个 schema 给了你什么 pure natural-language 没有的能力，又带来什么成本？

## 关键词汇

| 术语 | 人们常说 | 实际含义 |
|------|----------------|------------------------|
| Speech act | “会做事的话语” | Austin/Searle：utterances as actions。ACL 的理论父级。 |
| FIPA | “那个老 XML 东西” | IEEE Foundation for Intelligent Physical Agents。2000 年标准化 ACL。 |
| ACL | “Agent Communication Language” | FIPA 的 envelope format：performative + content + metadata。 |
| Performative | “动词” | message 的 intent class：`inform`、`request`、`propose`、`cfp` 等。 |
| KQML | “FIPA 的前身” | Knowledge Query and Manipulation Language（1993）。更简单、更窄。 |
| Ontology | “共享词汇表” | 对 content language 所谈概念的 formal definition。 |
| SL0 / SL1 | “FIPA content languages” | Semantic Language levels 0 和 1 — formal content language family。 |
| Contract Net | “任务市场” | Manager 发出 cfp；bidders propose；manager accept。canonical interaction protocol。 |
| Interaction protocol | “消息模式” | 一串带已知 correctness 的 performatives：request-when、subscribe-notify 等。 |

## 延伸阅读

- [Liu et al. — A Survey of Agent Interoperability Protocols: MCP, ACP, A2A, ANP](https://arxiv.org/html/2505.02279v1) — 连接 modern specs 与 FIPA heritage 的 canonical 2025 survey
- [FIPA ACL Message Structure Specification (fipa00037)](http://www.fipa.org/specs/fipa00037/) — 2000 年批准的 envelope format
- [FIPA Communicative Act Library Specification (fipa00037)](http://www.fipa.org/specs/fipa00037/) — 完整 performative catalog
- [MCP specification 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25) — `request`/`query-ref` 的 modern tool-use 等价物
- [A2A specification](https://a2a-protocol.org/latest/specification/) — contract-net 和 subscribe-notify 的 modern agent-peer 等价物
