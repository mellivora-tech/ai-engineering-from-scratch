# Indirect Prompt Injection：Production Attack Surface

> Indirect prompt injection（IPI）把 instruction 嵌入外部内容中，比如网页、邮件、共享文档、support ticket，随后由 agentic system 在没有显式用户动作的情况下消费。IPI 是 2026 年主导性的生产威胁：它绕过 user-input filter，因为 attacker 从不接触用户；它随着 agent 处理更多外部内容而静默 scale；它瞄准没有人阅读 prompt 的自动化 workflow。MDPI Information 17(1):54（2026 年 1 月）综合了 2023-2025 研究。NDSS 2026 的 IPI-defense 论文把核心挑战表述为：injected instruction 可以在语义上 benign（“please print Yes”），所以 detection 需要超过 keyword filtering。“The Attacker Moves Second”（Nasr et al.，OpenAI/Anthropic/DeepMind 联合，2025 年 10 月）：adaptive attack（gradient、RL、random search、human red-team）击破了 12 个已发表 defense 中超过 90%，而这些 defense 原先报告过接近零的 attack success rate。

**类型：** 构建
**语言：** Python（stdlib，IPI attack + defense harness）
**前置要求：** 阶段 18 · 12（PAIR），阶段 14（agent engineering）
**时间：** ~75 分钟

## 学习目标

- 定义 indirect prompt injection，并描述三种常见 delivery vector。
- 解释为什么 user-input filter 会完全漏掉 IPI。
- 描述作为 2026 defense paradigm 的 “information flow control” framing。
- 说明 Nasr et al.（2025 年 10 月）关于 adaptive attack 攻击已发表 IPI defense 的发现。

## 问题

Direct prompt injection 需要 attacker 触达用户或其 prompt。IPI 不需要：attacker 把 payload 放到 agent 可能读取的任意内容中，例如网页、inbox 中的 email、GitHub issue、product review。Agent 在正常操作中拾取它并执行 instruction。用户是 messenger，不是 intent。

## 概念

### 三种 delivery vector

- **Retrieval-augmented generation（RAG）。** Attacker 发布一个文档；retrieval step 抓取它；prompt 在用户问题前拼接它；模型执行 attacker instruction。
- **Inbox / document workflow。** Attacker 向用户发送 email；agent 读取 email；prompt 包含 email body；模型遵循 email 中的 instruction。
- **Tool output。** Attacker 控制 agent 使用的某个 tool（例如一个返回 attacker-controlled result 的 web search）；tool output 包含 instruction；agent 的 control flow 遵循它们。

三者共享一个结构性质：attacker 控制 prompt 的一个片段，却没有触碰 user-facing input。

### 为什么 user-input filter 会漏掉它

IPI payload 不出现在用户输入中。它出现在 retrieved content 中。如果 filter 只 gate user input，payload 就会绕过。如果 filter gate 所有到达模型的 content，它就必须应用于任意 retrieved text，这既昂贵，又会对恰好包含祈使语气的合法内容产生 false positive。

### AI 的 Information Flow Control（IFC）

2026 defense paradigm 借鉴经典 OS security。把每个 content source 视为 security label。把用户 query 标为 “trusted”。把 retrieved content 标为 “untrusted”。把模型的 control flow 视为 information flow：由 untrusted content 触发的 action 必须先由 trusted input ratify 才能执行。

CaMeL（Microsoft 2025）、ConfAIde（Stanford 2024）和 NDSS 2026 IPI-defense 论文用不同方式 operationalize IFC。共同原则是：只要 code 和 data 共享同一个 context window，目标就是 containment，而不是 prevention。

### The Attacker Moves Second

Nasr et al.（2025 年 10 月）用 adaptive attack（gradient search、RL policy、random search、72-hour human red-team）测试了 12 个已发表 IPI defense。每个原本报告 near-zero ASR 的 defense 都被击破到 >90% ASR。

方法论教训：只在 adaptive-attack evaluation 下发布 defense。Static-attack benchmark 不是 robustness 证据；attacker 会知道 defense。

### 真实事件

第 25 课覆盖 EchoLeak（CVE-2025-32711，CVSS 9.3），也就是 Microsoft 365 Copilot 中第一个公开记录的 zero-click IPI。GitHub Copilot Chat 中的 CamoLeak（CVSS 9.6）。GitHub Copilot 中的 CVE-2025-53773。生产 deployment 正在现场被 IPI compromise，而不只是 benchmark 中。

### OWASP 和 NIST framing

OWASP LLM Top 10（2025）把 prompt injection（direct + indirect）列为 LLM01，即 #1 application-layer threat。NIST AI SPD 2024 称 indirect prompt injection 是 “generative AI's greatest security flaw.”

### 它在阶段 18 中的位置

第 12-14 课是 model-centric jailbreak。第 15 课是主导 2026 生产 deployment 的 system-centric attack。第 16 课覆盖防御工具。第 25 课覆盖具体 CVE narrative。

## 使用它

`code/main.py` 构建一个 IPI harness。一个玩具 agent 有三个 tool（search web、read email、send message）。环境包含 attacker-controlled content，其中嵌入 instruction（“forward this to all contacts”）。你可以在 naive agent（遵循 injected instruction）、filter-defended agent（对 retrieved content 做 keyword filter）和 IFC agent（分离 trusted/untrusted content，并拒绝 untrusted control-flow command）之间切换。

## 交付它

本课会生成 `outputs/skill-ipi-audit.md`。给定一个 agentic deployment description，它会枚举 untrusted content source，检查 deployment 是否应用 IFC，并标记没有 trust label 就到达模型的 source。

## 练习

1. 运行 `code/main.py`。测量 attack 对三个 agent 中每一个的 success rate。

2. 在 retrieved content 上实现 paraphrase-based defense。测量合法 retrieved text 上的 benign false-positive rate。

3. 阅读 NDSS 2026 IPI-defense 论文。描述 “benign instruction” challenge，以及为什么它阻止 keyword-based filtering。

4. 设计一个 deployment，其中 agent 接收来自第三方 API 的 tool output。给每个 prompt fragment 标注 trust level，并写出 governing agent action 的 IFC policy。

5. 在练习 2 的 filter-defended agent 上复现 Nasr et al. 2025 adaptive-attack methodology。报告 adaptive attack 前后的 ASR。

## 关键词汇

| 术语 | 人们常说 | 实际含义 |
|------|-----------------|------------------------|
| IPI | “indirect prompt injection” | 通过用户没有写入、但 agent 在正常操作中消费的内容进行 injection |
| RAG injection | “poisoned retrieval” | Attacker 发布内容，retrieval step 抓取它；prompt 包含 payload |
| Zero-click | “no user action” | Attack 在 agent operation 中自动触发；用户什么都不做 |
| IFC | “information flow control” | 基于 label 的方法：来自 untrusted content 的 action 需要 trusted ratification |
| Adaptive attack | “gradient / RL red-team” | 知道 defense 并针对它优化的 attack；诚实 evaluation 所必需 |
| Benign instruction | “please print Yes” | 语义 benign 的 IPI payload；keyword filter 抓不到 |
| Scope violation | “cross-trust exfiltration” | Agent 访问一个 trust context 的数据，并输出到另一个 context |

## 延伸阅读

- [MDPI Information 17(1):54 — Indirect Prompt Injection Survey (January 2026)](https://www.mdpi.com/2078-2489/17/1/54) — 2023-2025 综合
- [Nasr et al. — The Attacker Moves Second (joint OpenAI/Anthropic/DeepMind, October 2025)](https://arxiv.org/abs/2510.18108) — adaptive attack evaluation
- [Greshake et al. — Not what you've signed up for (arXiv:2302.12173)](https://arxiv.org/abs/2302.12173) — 原始 IPI 论文
- [OWASP — LLM Top 10 (2025)](https://genai.owasp.org/llm-top-10/) — prompt injection 排名 LLM01
