# Browser Agents 与 Long-Horizon Web Tasks

> ChatGPT agent（2025 年 7 月）把 Operator 和 deep research 合并为一个 browser/terminal agent，并在 BrowseComp 上创下 68.9% 的 SOTA。OpenAI 于 2025 年 8 月 31 日关闭 Operator——这是产品层的整合。Anthropic 收购 Vercept 后，让 Claude Sonnet 在 OSWorld 上从低于 15% 提升到 72.5%。WebArena-Verified（ServiceNow，ICLR 2026）修复了原始 WebArena 中 11.3 个百分点的 false-negative rate，并发布 258-task Hard subset。数字是真的。attack surface 也是真的：OpenAI 的 preparedness 负责人公开表示，针对 browser agents 的 indirect prompt injection “is not a bug that can be fully patched”。已记录的 2025–2026 攻击包括：Tainted Memories（Atlas CSRF）、HashJack（Cato Networks），以及 Perplexity Comet 中的一键 hijacks。

**类型：** 学习
**语言：** Python（stdlib，indirect prompt-injection attack surface model）
**前置要求：** 阶段 15 · 10（Permission modes），阶段 15 · 01（Long-horizon agents）
**时间：** ~45 分钟

## 问题

browser agent 是一种 long-horizon agent：它读取不可信内容，并采取有后果的行动。agent 访问的每个页面都是用户没有写过的输入。每个页面上的每个 form 都可能是命令通道。2025–2026 攻击语料表明这不是假设：Tainted Memories 让攻击者通过 crafted page 将恶意指令绑定到 agent memory；HashJack 把命令藏在 agent 访问的 URL fragments 中；Perplexity Comet hijacks 一次点击即可触发。

防御图景令人不适。OpenAI 的 preparedness 负责人把安静的部分说了出来：indirect prompt injection “is not a bug that can be fully patched”。这是因为攻击存在于 agent 的 reading-vs-acting boundary，而这个边界在架构上是模糊的——模型读取的每个 token，原则上都可能被读作一条指令。

本课会命名 attack surface，命名 benchmark landscape（BrowseComp、OSWorld、WebArena-Verified），并建模一个极简 indirect-prompt-injection 场景，帮助你为第 14 和第 18 课中的真实防御做推理。

## 概念

### 2026 版图，每个系统一段

**ChatGPT agent（OpenAI）。** 2025 年 7 月发布。统一 Operator（browsing）和 Deep Research（multi-hour research）。2025 年 8 月 31 日关闭独立 Operator。在 BrowseComp 上达到 68.9% SOTA；在 OSWorld 和 WebArena-Verified 上也有强数字。

**Claude Sonnet + Vercept（Anthropic）。** Anthropic 收购 Vercept 聚焦 computer-use capabilities。让 Claude Sonnet 在 OSWorld 上从 <15% 提升到 72.5%。Claude Computer Use 以 tool API 形式发布。

**Gemini 3 Pro with Browser Use（DeepMind）。** Browser Use integration 发布 computer-use controls；FSF v3（2026 年 4 月，第 20 课）专门追踪 ML R&D domain 中的 autonomy。

**WebArena-Verified（ServiceNow，ICLR 2026）。** 修复一个已充分记录的问题：原始 WebArena 有约 11.3% false-negative rate（实际已解决的 tasks 被标记为失败）。Verified release 使用 human-curated success criteria 重新评分，并添加 258-task Hard subset（ICLR 2026 paper，openreview.net/forum?id=94tlGxmqkN）。

### BrowseComp vs OSWorld vs WebArena

| Benchmark | What it measures | Horizon |
|---|---|---|
| BrowseComp | Finding specific facts on the open web under time pressure | minutes |
| OSWorld | Agent operating a full desktop (mouse, keyboard, shell) | tens of minutes |
| WebArena-Verified | Transactional web tasks in simulated sites | minutes |
| Hard subset | WebArena-Verified tasks with multi-page state transitions | tens of minutes |

它们测量不同轴。高 BrowseComp 分数说明 agent 能找事实；不说明它能订机票。OSWorld 分数更接近“它能在我的桌面上工作吗”。WebArena-Verified 更接近“它能完成一个 flow 吗”。任何生产决策都需要匹配 task distribution 的 benchmark。

### attack surface，逐项命名

1. **Indirect prompt injection。** 不可信页面内容包含指令。agent 读取它。agent 执行它。公开例子：2024 Kai Greshake et al.、2025 Tainted Memories paper、2026 HashJack（Cato Networks）。
2. **URL fragment / query injection。** 被抓取 URL 的 `#fragment` 或 query string 包含命令。它从不被可见渲染；但仍在 agent 的 context 中。
3. **Memory-binding attacks。** 页面指示 agent 写入 persistent memory（第 12 课覆盖 durable state）。下个 session，memory 在没有可见触发器的情况下发射 payload。
4. **Authenticated sessions 上的 CSRF-shaped attacks。** Tainted Memories 类：agent 已登录某处；攻击者页面发起 state-changing requests，agent 带着用户 cookies 执行。
5. **One-click hijack。** 一个视觉上无害的按钮携带 agent 会跟随的后续 payload。Comet 类。
6. **agent host surface 中的 Content-Security-Policy holes。** rendering 和 tool layers 本身也可以是 attack vectors；browser-in-a-browser-agent stack 很宽。

### 为什么“not fully patchable”

攻击与 agent 的能力同构。agent 必须读取不可信内容才能工作。agent 读取的任何内容都可能包含指令。agent 遵循的任何指令都可能与用户真实请求不一致。防御（trust boundaries、classifiers、tool allowlists、consequential actions 上的 HITL）会提高攻击成本并降低 blast radius。它们不能关闭这一类问题。

这与 Lob's theorem（第 8 课）是同一种推理模式：agent 无法证明下一个 token 是安全的；它只能搭建一个让 unsafe tokens 更可检测的系统。

### 真正上线的 defense posture

- **Read / write boundary。** 读取从不产生后果。写入（提交 form、发布内容、调用带副作用的 tool）如果由 trust boundary 外的内容发起，就需要 fresh human approval。
- **每个 task 的 tool allowlist。** agent 可以浏览；不能发起 wire transfer，除非该 tool 为此 task 明确启用。第 13 课覆盖 budgets。
- **Session isolation。** Browser agent sessions 只使用 scoped credentials。无 production auth，无 personal email。保留每个 HTTP request 的 logs 供 audit。
- **Content sanitizer。** Fetched HTML 在拼接进 model context 前剥离 known-bad patterns。（减少简单攻击；挡不住复杂 payloads。）
- **Consequential actions 上的 HITL。** propose-then-commit pattern（第 15 课）。
- **Memory 上的 canary tokens。** 如果 memory entry 触发，用户能看到它（第 14 课）。

## 使用它

`code/main.py` 会模拟一个 tiny browser-agent run，访问三个 synthetic pages。一个 page benign，一个在可见文本中有 direct prompt-injection blob，一个有 URL-fragment injection（不可见，但在 agent context 内）。脚本展示（a）naïve agent 会做什么，（b）read/write boundary 捕捉什么，（c）sanitizer 捕捉什么，（d）二者都捕捉不到什么。

## 交付它

`outputs/skill-browser-agent-trust-boundary.md` 会为拟议 browser-agent deployment 划定 scope：它触碰哪些 trust zones、被授权写入什么，以及首次运行前必须到位哪些 defenses。

## 练习

1. 运行 `code/main.py`。指出哪种 attack sanitizer 能捕捉但 read/write boundary 不能，以及哪种 attack 只有 read/write boundary 能捕捉。

2. 扩展 sanitizer，使其检测一类 HashJack-style URL-fragment injection。在带合法 fragments 的 benign URLs 上测量 false-positive rate。

3. 选一个你熟悉的真实 browser-agent workflow（例如“book a flight”）。列出每个 read 和每个 write。标记哪些 writes 需要 HITL，并说明原因。

4. 阅读 WebArena-Verified ICLR 2026 paper。找出原始 WebArena scoring 不可靠的一类 task，并解释 Verified subset 如何解决它。

5. 为 browser-agent 设置设计一个 memory canary。你会存什么，存在哪里，什么触发 alarm？

## 关键术语

| Term | What people say | What it actually means |
|---|---|---|
| Indirect prompt injection | “坏页面文本” | agent 读取的页面中有不可信内容，包含 agent 会执行的指令 |
| Tainted Memories | “Memory attack” | Agent 将攻击者提供的指令写入 durable memory；下个 session 触发 |
| HashJack | “URL fragment attack” | payload 藏在 URL fragment / query string 中，进入 agent context 但不可见渲染 |
| One-click hijack | “Bad button” | 可见 affordance 携带 agent 执行的后续 payload |
| BrowseComp | “Web search benchmark” | 在 open web 上找具体事实；minute-scale horizon |
| OSWorld | “Desktop benchmark” | 完整 OS control；multi-step GUI tasks |
| WebArena-Verified | “Fixed web-task benchmark” | ServiceNow 重新评分的 WebArena，带 Hard subset |
| Read/write boundary | “Side-effect gate” | 读取不产生后果；如果内容 out-of-trust，写入需要 fresh approval |

## 延伸阅读

- [OpenAI — Introducing ChatGPT agent](https://openai.com/index/introducing-chatgpt-agent/) — Operator 与 deep research 的合并；BrowseComp SOTA。
- [OpenAI — Computer-Using Agent](https://openai.com/index/computer-using-agent/) — Operator lineage 以及后来成为 ChatGPT agent 的 architecture。
- [Zhou et al. — WebArena](https://webarena.dev/) — 原始 benchmark。
- [WebArena-Verified (OpenReview)](https://openreview.net/forum?id=94tlGxmqkN) — ICLR 2026 fixed-subset paper。
- [Anthropic — Measuring agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) — 包含 computer-use agents 的 attack-surface discussion。
