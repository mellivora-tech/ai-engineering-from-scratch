# Red-Team Tooling：Garak、Llama Guard、PyRIT

> 三个生产工具定义了 2026 red-team stack。Llama Guard（Meta）：一个在 14 个 MLCommons hazard category 上 fine-tune 的 Llama-3.1-8B classifier；2025 年的 Llama Guard 4 是从 Llama 4 Scout 剪枝而来的 12B natively multimodal classifier。Garak（NVIDIA）：开源 LLM vulnerability scanner，带有针对 hallucination、data leakage、prompt injection、toxicity 和 jailbreak 的 static、dynamic、adaptive probe。PyRIT（Microsoft）：用 Crescendo、TAP 和自定义 converter chain 进行 deep exploitation 的 multi-turn red-team campaign。Llama Guard 3 记录在 Meta 的 “Llama 3 Herd of Models”（arXiv:2407.21783）；Llama Guard 3-1B-INT4 见 arXiv:2411.17713；Garak 的 probe architecture 见 github.com/NVIDIA/garak。这些工具是 2026 年 red-team research（第 12-15 课）与 deployment（第 17+ 课）之间的生产接口。

**类型：** 构建
**语言：** Python（stdlib，tool-architecture simulator 与 Llama Guard-style classifier mock）
**前置要求：** 阶段 18 · 12-15（jailbreaks and IPI）
**时间：** ~75 分钟

## 学习目标

- 描述 Llama Guard 3/4 在 safety stack 中的位置：input classifier、output classifier，还是两者都是。
- 说出 14 个 MLCommons hazard category，并指出一个不明显的类别（Code Interpreter Abuse）。
- 描述 Garak 的 probe architecture：probes、detectors、harnesses。
- 描述 PyRIT 的 multi-turn campaign structure，以及它如何与 Garak probe 组合。

## 问题

第 12-15 课展示 attack surface。生产 deployment 需要可重复、可 scale 的 evaluation。三个工具主导 2026：Llama Guard（defense classifier）、Garak（scanner）、PyRIT（campaign orchestrator）。每个工具针对 red-team lifecycle 的不同层。

## 概念

### Llama Guard（Meta）

Llama Guard 3 是一个 Llama-3.1-8B 模型，在 MLCommons AILuminate 14 个类别上 fine-tune，用于 input/output classification：
- Violent crimes, non-violent crimes, sex-related, CSAM, defamation
- Specialized advice, privacy, IP, indiscriminate weapons, hate
- Suicide/self-harm, sexual content, elections, code-interpreter abuse

支持 8 种语言。用法：放在 LLM 之前（input moderation）、LLM 之后（output moderation），或两边都放。这两种用法产生不同 training distribution，Llama Guard 3 作为单个模型同时处理两者。

Llama Guard 3-1B-INT4（arXiv:2411.17713，440MB，在 mobile CPU 上约 30 tokens/s）是量化边缘变体。

Llama Guard 4（2025 年 4 月）是 12B、natively multimodal，从 Llama 4 Scout 剪枝而来。它用一个接收 text + image 的 classifier，替代了 8B text 和 11B vision 前代。

### Garak（NVIDIA）

开源 vulnerability scanner。架构：
- **Probes。** 为 hallucination、data leakage、prompt injection、toxicity、jailbreak 生成 attack。Static（固定 prompt）、dynamic（生成 prompt）、adaptive（响应 target output）。
- **Detectors。** 根据预期 failure mode 给 output 打分，包括 toxic、leaked、jailbroken。
- **Harnesses。** 管理 probe-detector pair，运行 campaign，生成 report。

TrustyAI 把 Garak 与 Llama-Stack shields（Prompt-Guard-86M input classifier、Llama-Guard-3-8B output classifier）集成，用于 end-to-end shielded-target evaluation。Tier-based scoring（TBSA）替代 binary pass/fail，一个模型可能在同一 probe 上通过 severity tier 3，但在 tier 5 失败。

### PyRIT（Microsoft）

Python Risk Identification Toolkit。Multi-turn red-team campaign。围绕以下概念构建：
- **Converters。** 转换 seed prompt：paraphrase、encode、translate、roleplay。
- **Orchestrators。** 运行 campaign：Crescendo（escalation）、TAP（branching）、RedTeaming（custom loop）。
- **Scoring。** LLM-as-judge 或 classifier-as-judge。

PyRIT 是 Garak 更重的近亲。Garak 运行数千个 single-turn probe；PyRIT 运行 deep multi-turn campaign，专门打破特定 failure mode。

### Stack

把 Llama Guard 放在模型两侧。每晚运行 Garak 做 regression。Pre-release 运行 PyRIT campaign。这是大多数 production deployment 在 2026 年的默认配置。

### Evaluation pitfalls

- **Judge identity。** 三个工具都可以使用 LLM judge；judge calibration 决定报告的 ASR（第 12 课）。必须把 judge 与 tool 一起说明。
- **Probe staleness。** 随着模型被 patch，Garak probe 会老化。Adaptive probe（PAIR-shaped）比 static probe 老化更慢。
- **Llama Guard 在 benign content 上的 FPR。** 早期 Llama Guard 版本会过度标记政治和 LGBTQ+ 内容；Llama Guard 3/4 calibration 有改进，但并未按 deployment 校准。

### 它在阶段 18 中的位置

第 12-15 课是 attack family。第 16 课是生产工具。第 17 课（WMDP）是 dual-use capability 的 evaluation。第 18 课是把这些工具包进 policy structure 的 frontier safety framework。

## 使用它

`code/main.py` 构建一个玩具 Llama Guard-style classifier（14 个 category 上的 keyword + semantic feature）、一个玩具 Garak harness（probe-detector loop）和一个 PyRIT-style multi-turn converter chain。你可以把三种工具运行在 mock target 上，并观察它们不同的 coverage signature。

## 交付它

本课会生成 `outputs/skill-red-team-stack.md`。给定一个 deployment description，它会指出三种工具中哪些适用、每个需要配置什么，以及应该以什么 cadence 运行 regression。

## 练习

1. 运行 `code/main.py`。比较 Llama-Guard-style classifier 在 single-turn 与 multi-turn attack 上的 detection rate。

2. 实现一个新的 Garak probe：base64-encoded harmful request。测量 Llama-Guard-style classifier 对它的 detection。

3. 为 PyRIT-style converter chain 扩展一个 “translate to French, then paraphrase” converter。重新测量 attack success。

4. 阅读 Llama Guard 3 的 hazard-category list。识别两个类别，在这些类别中 training data 现实上会对合法 developer content 产生高 false-positive rate。

5. 比较 Garak 和 PyRIT 的设计原则。分别论证一个 deployment，其中每个工具是正确选择。

## 关键词汇

| 术语 | 人们常说 | 实际含义 |
|------|-----------------|------------------------|
| Llama Guard | “classifier” | fine-tuned Llama-3.1-8B/4-12B safety classifier，带 14 个 hazard category |
| Garak | “scanner” | NVIDIA 开源 vulnerability scanner；probes、detectors、harnesses |
| PyRIT | “campaign tool” | Microsoft multi-turn red-team orchestrator；converters、orchestrators、scoring |
| Prompt-Guard | “small classifier” | Meta 的 86M prompt-injection classifier，与 Llama Guard 配对 |
| TBSA | “tier-based scoring” | Garak 的 tier-based pass/fail，替代 binary outcome |
| Converter chain | “paraphrase + encode + ...” | PyRIT 的组合 primitive，用于构建 multi-step attack |
| MLCommons hazard categories | “14 taxonomies” | Llama Guard 目标的行业标准 taxonomy |

## 延伸阅读

- [Meta — Llama Guard 3 (in Llama 3 Herd paper, arXiv:2407.21783)](https://arxiv.org/abs/2407.21783) — 8B classifier
- [Meta — Llama Guard 3-1B-INT4 (arXiv:2411.17713)](https://arxiv.org/abs/2411.17713) — quantized mobile classifier
- [NVIDIA Garak — GitHub](https://github.com/NVIDIA/garak) — scanner repo 与文档
- [Microsoft PyRIT — GitHub](https://github.com/Azure/PyRIT) — campaign toolkit
