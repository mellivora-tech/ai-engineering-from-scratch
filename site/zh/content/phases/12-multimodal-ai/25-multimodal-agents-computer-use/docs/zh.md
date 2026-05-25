# Multimodal Agents 与 Computer-Use（Capstone）

> 2026 年 frontier product 是一个 multimodal agent：它读取 screenshots、点击 buttons、导航 web UIs、填写 forms，并端到端完成 workflows。SeeClick 和 CogAgent（2024）证明了 GUI-grounding primitive。Ferret-UI 加入 mobile。ChartAgent 引入 charts 的 visual tool-use。VisualWebArena 和 AgentVista（2026）是 frontier 追赶的 benchmarks，就连 Gemini 3 Pro 和 Claude Opus 4.7 在 AgentVista hard tasks 上也只有约 30%。这个 capstone 汇总阶段 12 的所有线索：perception（high-res VLM）、reasoning（带 tool use 的 LLM）、grounding（coordinate output）、long-horizon memory 和 evaluation。

**类型：** Capstone
**语言：** Python（stdlib，action schema + agent loop skeleton）
**前置要求：** 阶段 12 · 05（LLaVA），阶段 12 · 09（Qwen-VL JSON），阶段 14（Agent Engineering）
**时间：** ~240 分钟

## 学习目标

- 设计 multimodal agent loop：perceive -> reason -> act -> observe -> repeat。
- 构建 GUI grounding output schema（click coordinates、type text、scroll、drag），让 VLM 以 JSON 输出。
- 比较 screenshot-only agents、accessibility-tree agents 与 hybrid agents。
- 在一个小型 VisualWebArena slice 上设置 multimodal agent benchmark evaluation。

## 问题

一个 booking-site workflow：“find me a flight to Tokyo for April 15, aisle seat under $800, book it.”

Multimodal agent 需要：

1. 截取 browser screenshot。
2. 把 screenshot + URL + goal 解析成 plan。
3. 输出 structured action：click（at x,y）、type “Tokyo”（at element E）、scroll down、select（radio button）。
4. 把 action 应用到 browser。
5. 观察新状态（下一张 screenshot）。
6. 重复直到任务完成。

每一步都是一次 multimodal VLM call。VLM output 必须是 parseable JSON。错误会跨步骤累积，因此 recovery 很重要。

## 概念

### GUI grounding：primitive

GUI grounding 是：给定 screenshot 和 natural language instruction，输出要点击的（x, y）coordinate（或其他 action）。

SeeClick（arXiv:2401.10935）是第一个大规模 open result：在 synthetic + real GUI data 上 fine-tune VLM，以 plain text tokens 输出 coordinates。有效。

CogAgent（arXiv:2312.08914）加入 1120x1120 high-resolution encoding，用于 dense UIs。分数：web navigation 上约 84%。

Ferret-UI（arXiv:2404.05719）关注 mobile UIs，并集成 iOS accessibility data。

Output format 通常是 JSON：

```json
{"action": "click", "x": 384, "y": 220, "element_desc": "Search button"}
```

`element_desc` 有助于 recovery：如果 screenshots 之间坐标 drift，semantic hint 可以让系统 re-ground。

### Action schemas

典型 action schema 有 6-10 种 action types：

- `click`: (x, y)
- `type`: (text, x?, y?)
- `scroll`: (direction, amount)
- `drag`: (x0, y0, x1, y1)
- `select`: (option_index)
- `hover`: (x, y)
- `navigate`: (url)
- `wait`: (ms)
- `done`: (success, explanation)

Agent 每步输出一个 action。Browser wrapper 执行并返回新状态。

### Screenshot-only vs accessibility-tree

两种 input modes：

- Screenshot-only：完整图像，无结构信息。最通用，适用于任何 app。
- Accessibility tree：结构化 DOM / iOS accessibility info。Grounding 可靠得多；在 tree 可用时工作。
- Hybrid：两者都用，tree 负责 atomic actions 的可靠 grounding，screenshot 负责 semantic context。

Production agents 尽可能用 hybrid。Browser automation（Selenium + accessibility）总是有 tree；desktop apps 有时有。

### Long-horizon memory

20-step workflow 会产生 20 张 screenshots。VLM context 很快被填满。三种 compression strategies：

- Summary-chain：每 5 步总结发生了什么，丢弃旧 screenshots。
- Skip-frame：保留第一张、最后一张和每第 3 张 screenshot。
- Tool-recorded log：执行 actions，保留文字 log；不重新看旧 screenshots。

Claude 的 computer-use API 使用 log pattern。更简单、更可靠。

### Visual tool use

ChartAgent（arXiv:2510.04514）为 chart understanding 引入 visual tool use：crop、zoom、OCR、调用 external detection。Agent 可以输出 “crop to region (100, 200, 300, 400) then call OCR” 作为 tool call。Tool 返回文本，VLM 继续 reasoning。

这个 pattern 可以泛化：set-of-mark prompting、region annotation 和 external detection tools 都适合同一个“输出 tool call，接收 structured response”的 schema。

### 2026 benchmarks

- ScreenSpot-Pro。约 1k web screenshots 上的 GUI grounding。Open SOTA Qwen2.5-VL-72B 约 85%。Frontier 约 90%。
- VisualWebArena。端到端 web tasks（shop、forum、classifieds）。Open SOTA 约 20%。Gemini 3 Pro 约 27%。
- AgentVista（arXiv:2602.23166）。最难的 2026 benchmark。12 个 domains 的真实 workflows。Frontier models 27-40%；open models 10-20%。
- WebArena / WebShop。旧 benchmarks；frontier 已经饱和。

### 为什么仍然困难

Agent performance bottlenecks：

1. Fine scale visual grounding。“Click the small X” 在 mobile resolution 下经常失败。
2. Long-horizon planning。10 个 actions 后，agent 偏离 goal。
3. Error recovery。当 click failed（错误按钮）时，检测并恢复很少出现在训练数据中。
4. Cross-page context。Tabs 或 long forms 之间跳转会丢 state。

研究方向：memory architectures、explicit replanning、multimodal verification（用 screenshot match 检查 action success）。

### Capstone build-it

Capstone task：构建 computer-use agent，它能：

1. 读取 booking-site mock page 的 HTML + screenshot。
2. 规划多步序列：search -> select -> fill form -> submit。
3. 输出匹配 action schema 的 JSON actions。
4. 在固定 10-task slice 上评估。

本课提供的 scaffold code 很容易扩展成真实 browser。

## 使用它

`code/main.py` 是 capstone scaffold：

- Action schema JSON definition（10 actions）。
- Mock browser state as dict。
- Agent loop skeleton：receive state，emit action，apply，loop。
- 10-task mini-benchmark（synthetic pages），测量 end-to-end success rate。
- Action failed 时的 error-recovery hook。

## 交付它

本课产出 `outputs/skill-multimodal-agent-designer.md`。给定 computer-use product（domain、action set、evaluation target），它会设计完整 agent loop、memory strategy、grounding mode 和 expected benchmark score。

## 练习

1. 用 `screenshot_region` tool（crop + zoom）扩展 action schema。哪些任务受益？

2. 阅读 AgentVista（arXiv:2602.23166）。描述最难的 task category，以及为什么 frontier models 仍会失败。

3. Long-horizon memory compression：设计一个 summary-chain，live 保留 ≤4 张 screenshots，log 数量不限。

4. 构建 error-recovery hook：action failure（button not found）后，agent 下一步做什么？

5. 比较 screenshot-only Claude 4.7 与 hybrid screenshot + accessibility-tree Qwen2.5-VL 在 10 个 web tasks 上的表现。哪些任务谁赢？

## 关键术语

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| GUI grounding | “Click coordinates” | 模型为 screenshot 上 instruction 的目标输出（x,y） |
| Action schema | “Tool definitions” | 有效 actions（click、type、scroll、drag）的 JSON 描述 |
| Accessibility tree | “Structured DOM” | 来自 browser/iOS APIs 的机器可读 UI hierarchy |
| Hybrid agent | “Screenshot + tree” | 同时使用 image 和 structured info；比单独任一更可靠 |
| Visual tool use | “Zoom/crop/detect” | Agent 在 plan 中调用 external vision tools（OCR、detection） |
| Summary-chain | “Memory compression” | 定期用 text summaries 替换长 screenshot history |
| VisualWebArena | “E2E web bench” | 2024 年端到端 web tasks benchmark |
| AgentVista | “2026 hard bench” | 12-domain realistic workflows；即使 Gemini 3 Pro 也约 30% |

## 延伸阅读

- [Cheng et al. — SeeClick (arXiv:2401.10935)](https://arxiv.org/abs/2401.10935)
- [Hong et al. — CogAgent (arXiv:2312.08914)](https://arxiv.org/abs/2312.08914)
- [You et al. — Ferret-UI (arXiv:2404.05719)](https://arxiv.org/abs/2404.05719)
- [ChartAgent (arXiv:2510.04514)](https://arxiv.org/abs/2510.04514)
- [Koh et al. — VisualWebArena (arXiv:2401.13649)](https://arxiv.org/abs/2401.13649)
- [AgentVista (arXiv:2602.23166)](https://arxiv.org/abs/2602.23166)
