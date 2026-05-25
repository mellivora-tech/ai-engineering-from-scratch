# Voice Agents：Pipecat 和 LiveKit

> 到 2026 年，voice agents 已经是一类一等生产系统。Pipecat 提供 Python frame-based pipeline（VAD → STT → LLM → TTS → transport）。LiveKit Agents 通过 WebRTC 把 AI models 接到用户。高级栈的生产延迟目标落在端到端 450-600ms。

**类型：** 学习
**语言：** Python (stdlib)
**前置要求：** 阶段 14 · 01（Agent Loop），阶段 14 · 12（Workflow Patterns）
**时间：** ~60 分钟

## 学习目标

- 描述 Pipecat 的 frame-based pipeline：DOWNSTREAM（source→sink）和 UPSTREAM（control）。
- 说出标准 voice pipeline 阶段，以及 Pipecat 支持哪些 transports。
- 解释 LiveKit Agents 的两类 voice agent（MultimodalAgent、VoicePipelineAgent），以及各自适合什么时候。
- 总结 2026 年生产延迟预期，以及它们如何驱动架构选择。

## 问题

Voice agent 不是给 text loop 贴上 TTS。延迟预算极其苛刻（~600ms），partial audio 是默认形态，turn detection 本身是一个 model，transports 从 telephony SIP 到 WebRTC 都有。你要么构建 frame-based pipeline（Pipecat），要么依赖平台（LiveKit）。

## 概念

### Pipecat (pipecat-ai/pipecat)

- Python frame-based pipeline framework。
- `Frame` → `FrameProcessor` chain。
- 两个流动方向：
  - **DOWNSTREAM** — source → sink（audio in，TTS out）。
  - **UPSTREAM** — feedback 和 control（cancellation、metrics、barge-in）。
- `PipelineTask` 通过 events（`on_pipeline_started`、`on_pipeline_finished`、`on_idle_timeout`）和用于 metrics/tracing/RTVI 的 observers 管理 lifecycle。

典型 pipeline：

```
VAD (Silero) → STT → LLM (context alternates user/assistant) → TTS → transport
```

Transports：Daily、LiveKit、SmallWebRTCTransport、FastAPI WebSocket、WhatsApp。

Pipecat Flows 增加 structured conversations（state machines）。Pipecat Cloud 是 managed runtime。

### LiveKit Agents (livekit/agents)

- 通过 WebRTC 把 AI models 接到用户。
- 核心概念：`Agent`、`AgentSession`、`entrypoint`、`AgentServer`。
- 两类 voice agent：
  - **MultimodalAgent** — 通过 OpenAI Realtime 或同类能力直接处理 audio。
  - **VoicePipelineAgent** — STT → LLM → TTS cascade；提供 text-level control。
- 通过 transformer model 做 semantic turn detection。
- 原生 MCP integration。
- 通过 SIP 做 telephony。
- 通过 LiveKit Inference 无需 API keys 使用 50+ models；通过 plugins 使用另外 200+。

### Commercial platforms

Vapi（优化 premium stack 上 ~450-600ms）和 Retell（180 次测试通话端到端 ~600ms）构建在这些能力之上。当你想要 managed voice stack、但没有 WebRTC 团队时，选择平台。

### 这个模式会在哪里出错

- **没有处理 barge-in。** 用户打断；agent 继续说话。Pipecat 需要 UPSTREAM cancel frames，LiveKit 中需要等价机制。
- **忽略 STT confidence。** 低置信转写被当成事实喂给 LLM。应该按 confidence gate，或请求确认。
- **TTS 在句子中途被切断。** Pipeline 在 utterance 中途 cancel 时，TTS 需要知道，否则音频会被硬切。
- **忽略 latency budget。** 每个组件都增加 50-200ms。发布前先把整条链路加总。

### 典型 2026 延迟

- VAD：20-60ms
- STT partial：100-250ms
- LLM first token：150-400ms
- TTS first audio：100-200ms
- Transport RTT：30-80ms

端到端 450-600ms 属于 premium。800-1200ms 很常见。任何 > 1500ms 都会感觉坏了。

## 构建它

`code/main.py` 是一个 frame-based toy pipeline，包含：

- `Frame` types（audio、transcript、text、tts_audio、control）。
- 带 `process(frame)` 的 `Processor` interface。
- 一个五阶段 pipeline（VAD → STT → LLM → TTS → transport），由 scripted processors 组成。
- 一个 UPSTREAM cancel frame，用来演示 barge-in。

运行它：

```
python3 code/main.py
```

Trace 会展示正常 flow，以及一次 barge-in cancel 如何让 TTS 在 utterance 中途停止。

## 使用它

- **Pipecat** 用于 full control — custom processors、Python-first、pluggable providers。
- **LiveKit Agents** 用于 WebRTC-first deployments 和 telephony。
- **Vapi / Retell** 用于 hosted voice agents，不需要 WebRTC 团队。
- **OpenAI Realtime / Gemini Live** 用于 direct audio-in/audio-out（MultimodalAgent）。

## 发布它

`outputs/skill-voice-pipeline.md` 会 scaffold 一个 Pipecat-shaped voice pipeline，包含 VAD + STT + LLM + TTS + transport，以及 barge-in handling。

## 练习

1. 给 toy pipeline 添加 metrics observer：统计每个 stage 每秒的 frames 数。延迟在哪里累积？
2. 实现 confidence-gated STT：低于阈值时请求 “could you repeat that?”
3. 添加 semantic turn detection：简单规则 — 如果 transcript 以 “?” 结尾，就认为 turn 结束。
4. 阅读 Pipecat 的 transport docs。把 stdlib transport 换成 SmallWebRTCTransport config（stub）。
5. 在同一个 query 上测量 OpenAI Realtime 与 STT+LLM+TTS cascade。Text-level control 带来多少延迟成本？

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|----------------|------------------------|
| Frame | "Event" | Pipeline 中的 typed data unit（audio、transcript、text、control） |
| Processor | "Pipeline stage" | 带 process(frame) 的 handler |
| DOWNSTREAM | "Forward flow" | Source 到 sink：audio in，speech out |
| UPSTREAM | "Feedback flow" | Control：cancel、metrics、barge-in |
| VAD | "Voice activity detection" | 检测用户什么时候在说话 |
| Semantic turn detection | "Smart end-of-turn" | 基于 model 判断用户是否说完 |
| MultimodalAgent | "Direct audio agent" | Audio in、audio out；中间没有 text |
| VoicePipelineAgent | "Cascade agent" | STT + LLM + TTS；text-level control |

## 延伸阅读

- [Pipecat docs](https://docs.pipecat.ai/getting-started/introduction) — frame-based pipeline、processors、transports
- [LiveKit Agents docs](https://docs.livekit.io/agents/) — WebRTC + voice primitives
- [Vapi](https://vapi.ai/) — managed voice platform
- [Retell AI](https://www.retellai.com/) — managed voice，latency-benchmarked
