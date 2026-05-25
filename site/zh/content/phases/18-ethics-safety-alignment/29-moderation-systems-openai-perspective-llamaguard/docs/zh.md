# Moderation Systems：OpenAI、Perspective、Llama Guard

> Production moderation system 把第 12-16 课定义的 safety policy operationalize。OpenAI Moderation API：`omni-moderation-latest`（2024）基于 GPT-4o，在一次 call 中分类 text + images；在 multilingual test set 上比前一版本好 42%；response schema 返回 13 个 category boolean：harassment、harassment/threatening、hate、hate/threatening、illicit、illicit/violent、self-harm、self-harm/intent、self-harm/instructions、sexual、sexual/minors、violence、violence/graphic；对多数 developer 免费。分层模式：Input moderation（pre-generation）、Output moderation（post-generation）、Custom moderation（domain rules）。Async parallel call 隐藏 latency；flag 时使用 placeholder response。Llama Guard 3/4（第 16 课）：14 个 MLCommons hazards、Code Interpreter Abuse、8 languages（v3）、multi-image（v4）。Perspective API（Google Jigsaw）：早于 LLM-as-moderator 浪潮的 toxicity scoring；主要是 single-dimension toxicity，带 severe-toxicity/insult/profanity 变体；content-moderation research 的 baseline。Deprecations：Azure Content Moderator 于 2024 年 2 月 deprecated，2027 年 2 月 retired，由 Azure AI Content Safety 替代。

**类型：** 构建
**语言：** Python（stdlib，three-layer moderation harness）
**前置要求：** 阶段 18 · 16（Llama Guard / Garak / PyRIT）
**时间：** ~60 分钟

## 学习目标

- 描述 OpenAI Moderation API 的 category taxonomy，以及它与 Llama Guard 3 的 MLCommons set 有何不同。
- 描述三层 moderation pattern（input、output、custom），并为每层说出一个 failure mode。
- 描述 Perspective API 作为 pre-LLM-era baseline 的位置，以及为什么它仍在研究中使用。
- 说明 Azure deprecation timeline。

## 问题

第 12-16 课描述 attack 和 defense tooling。第 29 课覆盖 deployed moderation system，这些系统在用户接触产品的表面 operationalize defense。三层模式是 2026 默认配置。

## 概念

### OpenAI Moderation API

`omni-moderation-latest`（2024）。基于 GPT-4o。一次 call 分类 text + image。对多数 developer 免费。

类别（response schema 中的 13 个 boolean）：
- harassment, harassment/threatening
- hate, hate/threatening
- self-harm, self-harm/intent, self-harm/instructions
- sexual, sexual/minors
- violence, violence/graphic
- illicit, illicit/violent

Multimodal support 适用于 `violence`、`self-harm` 和 `sexual`，但不适用于 `sexual/minors`；其余为 text-only。

为了 `code/main.py` 中的 code harness 教学简洁，我们把 `/threatening`、`/intent`、`/instructions` 和 `/graphic` sub-category 折叠进其 top-level parent。Production code 应使用完整 13-category schema。

在 multilingual test set 上，比上一代 moderation endpoint 好 42%。每类都有 score；应用自行设置 threshold。

### Llama Guard 3/4

第 16 课已覆盖。14 个 MLCommons hazard category（组织方式不同于 OpenAI 的 13 个 response-schema boolean）。支持 8 种语言（v3）。Llama Guard 4（2025 年 4 月）是 natively multimodal，12B。

OpenAI 与 Llama Guard taxonomy 有重叠但也有分歧。OpenAI 有宽泛的 “illicit” 类别；Llama Guard 把 “violent crimes” 和 “non-violent crimes” 分开。Deployment 会根据自身 policy-taxonomy fit 选择。

### Perspective API（Google Jigsaw）

早于 LLM-as-moderator 浪潮（pre-2020）的 toxicity scoring system。类别：TOXICITY、SEVERE_TOXICITY、INSULT、PROFANITY、THREAT、IDENTITY_ATTACK。Primary score 是 single-dimension（TOXICITY），带 sub-dimension 变体。

它仍被广泛用作 content-moderation research baseline，因为 API 稳定、有文档，并且有多年 calibration data。对现代 LLM-adjacent use case，Llama Guard 或 OpenAI Moderation 通常更合适。

### 三层模式

1. **Input moderation。** 生成前 classify 用户 prompt。如果 flagged，则 reject。Latency：一次 classifier call。
2. **Output moderation。** 交付前 classify 模型 output。如果 flagged，则替换为 refusal。Latency：generation 后一次 classifier call。
3. **Custom moderation。** Domain-specific rules（regex、allowlists、business policy）。在 input 或 output 上运行。

这三层按设计 sequential：input moderation 必须在 generation 前完成，output moderation 在 generation 后运行。Parallelism 适用于一层内部，例如在同一 text 上并发运行多个 classifier（OpenAI Moderation + Llama Guard + Perspective），以隐藏 per-classifier latency。作为可选 optimization，可以在 input moderation 完成且 token-1 streaming 延后时展示 placeholder response（“one moment, checking...”）。Flag behaviour 可配置：refuse、sanitize、escalate to human review。

### Failure modes

- **Input only。** 抓不到 output hallucination（第 12-14 课 encoding attack 会绕过 input classifier）。
- **Output only。** 允许任何 input 到达模型；增加成本；把 internal reasoning 暴露给 attacker。
- **Custom only。** 不跨类别 robust；regex 脆弱。

Layered 是默认。Belt-and-suspenders。

### Azure deprecation

Azure Content Moderator：2024 年 2 月 deprecated，2027 年 2 月 retired。由 Azure AI Content Safety 替代，后者基于 LLM，并与 Azure OpenAI 集成。对 Azure deployment 来说，migration 是 2024-2027 的 field-level project。

### 它在阶段 18 中的位置

第 16 课覆盖 red-team 语境中的 moderation tooling。第 29 课覆盖 deployed moderation。第 30 课以当前 dual-use capability evidence 收束。

## 使用它

`code/main.py` 构建一个三层 moderation harness：input moderator（keyword + category score）、output moderator（同一 classifier 作用于 output）、custom moderator（domain rules）。你可以运行 input，观察哪一层捕获了什么。

## 交付它

本课会生成 `outputs/skill-moderation-stack.md`。给定一个 deployment，它会推荐 moderation stack configuration：input 用哪个 classifier、output 用哪个、custom rule 是什么，以及 edge case 使用哪个 judge。

## 练习

1. 运行 `code/main.py`。让 benign、borderline 和 harmful input 经过三层。报告每个由哪一层触发。

2. 用某个 specific category 的 Perspective-API-style toxicity scoring 扩展 harness。比较它的 threshold behaviour 与 category score。

3. 阅读 OpenAI Moderation API docs 和 Llama Guard 3 category list。把每个 OpenAI category 映射到最接近的 Llama Guard category。识别三个无法干净映射的类别。

4. 为 code-assistant deployment（例如 GitHub Copilot）设计 moderation stack。识别最相关和最不相关的 category，并提出 custom rules。

5. Azure Content Moderator 于 2027 年 2 月 retired。规划迁移到 Azure AI Content Safety。识别 migration 中风险最高的元素。

## 关键词汇

| 术语 | 人们常说 | 实际含义 |
|------|-----------------|------------------------|
| OpenAI Moderation | “omni-moderation-latest” | 基于 GPT-4o 的 13-category（text）classifier，带部分 multimodal support |
| Perspective API | “Google Jigsaw toxicity” | pre-LLM-era toxicity scoring baseline |
| Llama Guard | “MLCommons 14-category” | Meta hazard classifier（v3：8B text、8 langs；v4：12B multimodal） |
| Input moderation | “pre-generation filter” | 模型调用前作用于 user prompt 的 classifier |
| Output moderation | “post-generation filter” | 交付前作用于 model output 的 classifier |
| Custom moderation | “domain rules” | Deployment-specific rules（regex、allowlist、policy） |
| Layered moderation | “all three layers” | 标准生产 deployment pattern |

## 延伸阅读

- [OpenAI Moderation API docs](https://platform.openai.com/docs/api-reference/moderations) — omni-moderation endpoint
- [Meta PurpleLlama + Llama Guard](https://github.com/meta-llama/PurpleLlama) — Llama Guard repo
- [Google Jigsaw Perspective API](https://perspectiveapi.com/) — toxicity scoring
- [Azure AI Content Safety](https://learn.microsoft.com/en-us/azure/ai-services/content-safety/) — Azure replacement
