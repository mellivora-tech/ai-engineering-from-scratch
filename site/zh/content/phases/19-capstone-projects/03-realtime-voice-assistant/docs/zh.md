# Capstone 03 — 实时语音助手（ASR 到 LLM 到 TTS）

> 一个感觉自然的 voice agent，end-to-end latency 要低于 800ms，要知道你什么时候停止说话，要能处理 barge-in，还要能调用工具而不让对话卡住。Retell、Vapi、LiveKit Agents 和 Pipecat 在 2026 年都达到了这个门槛。它们采用同一种形态：streaming ASR、turn-detector、streaming LLM 和 streaming TTS，通过 WebRTC 串在一起，并在每一跳都设置严格 latency budget。构建一个这样的系统，衡量 WER、MOS 和 false-cutoff rate，并在 packet loss 下运行它。

**类型：** Capstone
**语言：** Python（agent + pipeline）、TypeScript（web client）
**前置要求：** 阶段 6（speech and audio）、阶段 7（transformers）、阶段 11（LLM engineering）、阶段 13（tools）、阶段 14（agents）、阶段 17（infrastructure）
**覆盖阶段：** P6 · P7 · P11 · P13 · P14 · P17
**时间：** 30 小时

## 问题

语音是 2025-2026 年发展最快的 AI UX 类别。技术天花板每个季度都在下降。OpenAI Realtime API、Gemini 2.5 Live、Cartesia Sonic-2、ElevenLabs Flash v3、LiveKit Agents 1.0 和 Pipecat 0.0.70 都让 sub-800ms first-audio-out 变得可达。门槛不只是 latency，而是交互感觉：不抢断用户、不被用户抢断、能从半句话打断中恢复、能在对话中调用工具而不让音频停顿、能承受抖动的移动网络。

把三个 REST call 拼在一起无法达到这个目标。架构必须是 end-to-end pipelined streaming。构建之后，失败模式会变得清晰：针对电话音频调过的 VAD 被背景电视触发；turn-detector 等待永远不会出现的标点；TTS 在发出音频前先 buffer 400ms。这个 capstone 要求你在负载下逐一修掉这些问题，并发布 latency-and-quality report。

## 概念

pipeline 有五个 streaming stages：**audio in**（来自浏览器或 PSTN 的 WebRTC）、**ASR**（来自 Deepgram Nova-3 或 faster-whisper 的 streaming partial transcripts）、**turn detection**（VAD 加一个小型 turn-detector model，读取 partial transcripts 判断完成线索）、**LLM**（一旦判断 turn complete 就开始 streaming tokens）、**TTS**（在第一个 LLM token 后约 200ms 内 streaming audio out）。

三个横切关注点。**Barge-in**：当 agent 正在说话时用户开始说话，TTS 会取消，ASR 立即接管。**Tool use**：对话中的 function calls（weather、calendar）必须在 side channel 上运行，不能阻塞音频；如果 latency 超过 300ms，agent 会预先填充一个 acknowledgement token（“one second...”）。**Backpressure**：在 packet loss 下，partial transcripts 会被保留，VAD 会提高 speech-gate threshold，agent 会避免盖过尚未确认的消息。

衡量标准是定量的：在 15 dB SNR 的 Hamming VAD benchmark 上 WER 低于 8%；100 通已测通话的 first-audio-out p50 低于 800ms；false-cutoff rate 低于 3%；TTS MOS 高于 4.2；单台 g5.xlarge 支撑 50 个 concurrent calls。这些数字就是交付物。

## 架构

```
browser / Twilio PSTN
        |
        v
   WebRTC / SIP edge
        |
        v
  LiveKit Agents 1.0  (or Pipecat 0.0.70)
        |
   +----+--------------+--------------+-----------------+
   |                   |              |                 |
   v                   v              v                 v
  ASR              VAD v5         turn-detector     side-channel
(Deepgram         (Silero)          (LiveKit)        tools
 Nova-3 /         speech-gate    completion score    (weather,
 Whisper-v3)      per 20ms        on partials        calendar)
   |                   |              |
   +--------+----------+--------------+
            v
        LLM (streaming)
     GPT-4o-realtime / Gemini 2.5 Flash /
     cascaded Claude Haiku 4.5
            |
            v
        TTS streaming
     Cartesia Sonic-2 / ElevenLabs Flash v3
            |
            v
     audio back to caller
            |
            v
   OpenTelemetry voice traces -> Langfuse
```

## 技术栈

- Transport：LiveKit Agents 1.0（WebRTC）加 Twilio PSTN gateway；Pipecat 0.0.70 作为替代框架
- ASR：Deepgram Nova-3（streaming，sub-300ms first partial）或自托管 faster-whisper Whisper-v3-turbo
- VAD：Silero VAD v5 加 LiveKit turn-detector（读取 partial transcripts 的小型 transformer）
- LLM：OpenAI GPT-4o-realtime（紧密集成）、Gemini 2.5 Flash Live，或 cascaded Claude Haiku 4.5（streaming completions，独立 audio path）
- TTS：Cartesia Sonic-2（最低 first-byte）、ElevenLabs Flash v3，或用于 self-host 的开源 Orpheus
- Tools：用于 weather/calendar/booking 的 FastMCP side-channel；若工具耗时 >300ms，agent 会预先发出 filler
- Observability：OpenTelemetry voice spans，带 audio replay 的 Langfuse voice traces
- Deployment：单台 g5.xlarge（24GB VRAM）用于 self-hosted Whisper + Orpheus；hosted APIs 用于最低 latency

## 构建它

1. **WebRTC session。** 启动 LiveKit room 和一个传输 microphone audio 的 web client。在 server 上挂接一个会加入 room 的 agent worker。

2. **ASR streaming。** 把 20ms PCM frames 送入 Deepgram Nova-3（或 GPU 上的 faster-whisper）。订阅 partial 和 final transcripts。记录每个 partial 的 latency。

3. **VAD and turn detector。** 在 frame stream 上运行 Silero VAD v5。出现 speech-end event 时，用最新 partial transcript 调用 LiveKit turn-detector。只有当 VAD 认为 silence 持续 500ms 且 turn-detector 的 completion score > 0.6 时，才 commit “turn complete”。

4. **LLM stream。** turn complete 后，用 running conversation 加 final transcript 启动 LLM call。流式输出 tokens。拿到第一个 token 时交给 TTS。

5. **TTS stream。** Cartesia Sonic-2 流式返回 audio chunks。第一个 chunk 必须在第一个 LLM token 后 200ms 内离开 server。把 chunks 发到 LiveKit room；client 通过 WebRTC jitter buffer 播放。

6. **Barge-in。** 当 TTS 正在播放时，如果 VAD 检测到新的 user speech，立即取消 TTS stream，丢弃剩余 LLM output，并重新 arm ASR。发布一个 `tts_canceled` span。

7. **Tool side channel。** 注册 weather 和 calendar 作为 function-calling tools。调用时并发发起；如果 300ms 内没有返回，就让 LLM 发出 “one second, let me check” 作为 filler；工具返回后继续。

8. **Eval harness。** 录制 100 通通话。计算 WER（对照 held-out transcript）、false-cutoff rate（用户还在句中时 TTS 被取消）、first-audio-out p50、TTS MOS（人工或 NISQA），以及 jitter-loss test（丢弃 3% packets）。

9. **Load test。** 用 synthetic caller 在单台 g5.xlarge 上驱动 50 个 concurrent calls。衡量持续运行时的 first-audio-out p95。

## 使用它

```
caller: "what is the weather in tokyo tomorrow"
[asr  ] partial @280ms: "what is the"
[asr  ] partial @540ms: "what is the weather"
[turn ] completion score 0.82 at @820ms; commit
[llm  ] first token @960ms
[tool ] weather.tokyo tomorrow -> 68/52 partly cloudy @1140ms
[tts  ] first audio-out @1040ms: "Tokyo tomorrow will be partly cloudy..."
turn latency: 1040ms user-stop -> audio-out
```

## 交付它

`outputs/skill-voice-agent.md` 是交付物。给定一个 domain（customer support、scheduling 或 kiosk），它会启动一个 LiveKit agent，并把 ASR/VAD/LLM/TTS pipeline 调到满足 measurement bar。评分标准：

| 权重 | 标准 | 衡量方式 |
|:-:|---|---|
| 25 | End-to-end latency | 100 通 recorded calls 的 p50 first-audio-out 低于 800ms |
| 20 | Turn-taking quality | Hamming VAD benchmark 上 false-cutoff rate 低于 3% |
| 20 | Tool-use correctness | 对话中的 tool calls 返回正确数据，且不阻塞音频 |
| 20 | Reliability under packet loss | 注入 3% packet drop 后的 WER 和 turn-taking stability |
| 15 | Eval harness completeness | 带 public config 的可复现实验测量 |
| **100** | | |

## 练习

1. 把 Deepgram Nova-3 换成 g5.xlarge 上的 faster-whisper v3 turbo。衡量 latency 和 WER 差距。识别 CPU-vs-GPU 决策会在哪些地方产生影响。

2. 添加 interruption-arbitration policy：用户在 tool call 期间 barge in 时，agent 该怎么做？比较三种策略（hard cancel、finish-tool-then-stop、queue next turn）。

3. 运行 adversarial turn-detector test：让用户在句子中间长时间停顿。调 VAD silence threshold 和 turn-detector score threshold，在不超过 900ms 的前提下降低 false-cutoff。

4. 通过 Twilio 把同一个 agent 部署到 PSTN。比较 PSTN first-audio-out 和 WebRTC。解释 jitter-buffer 和 codec 的差异。

5. 为非英语语言（日语、西班牙语）添加 voice activity detection。衡量 Silero VAD v5 false-trigger rate，并与 language-specific fine-tunes 对比。

## 关键词汇

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|------------------------|
| Turn detection | “End of utterance” | 给定 VAD silence 和 partial transcript 后，判断用户是否说完的 classifier |
| Barge-in | “Interruption handling” | 当 VAD 检测到新的 user speech 时，取消正在播放的 TTS |
| First-audio-out | “Latency” | 从用户停止说话到第一个 audio packet 离开 server 的时间 |
| VAD | “Speech gate” | 把 audio frames 分类为 speech vs silence 的模型；Silero VAD v5 是 2026 默认选择 |
| Jitter buffer | “Audio smoothing” | client-side buffer，短暂持有 packets 以吸收网络波动 |
| Filler | “Acknowledgment token” | 工具较慢时，agent 为避免沉默而发出的短语 |
| MOS | “Mean opinion score” | 感知语音质量评分；NISQA 是自动 proxy |

## 延伸阅读

- [LiveKit Agents 1.0](https://github.com/livekit/agents) — reference WebRTC agent framework
- [Pipecat](https://github.com/pipecat-ai/pipecat) — alternate Python-first streaming agent framework
- [OpenAI Realtime API](https://platform.openai.com/docs/guides/realtime) — integrated speech models 参考
- [Deepgram Nova-3 documentation](https://developers.deepgram.com/docs) — streaming ASR reference
- [Silero VAD v5](https://github.com/snakers4/silero-vad) — VAD reference model
- [Cartesia Sonic-2](https://docs.cartesia.ai) — low-latency TTS reference
- [Retell AI architecture](https://docs.retellai.com) — production voice agent architecture
- [Vapi.ai production stack](https://docs.vapi.ai) — alternate production reference
