# 构建 Voice Assistant Pipeline：Phase 6 Capstone

> 把第 01-11 课全部缝起来。构建一个会听、会推理、会说话的 voice assistant。2026 年这已经是工程问题，不是研究问题，但集成细节决定它能否交付。

**类型：** 构建
**语言：** Python
**前置要求：** 阶段 6 · 04、05、06、07、11；阶段 11 · 09（Function Calling）；阶段 14 · 01（Agent Loop）
**时间：** ~120 分钟

## 问题

构建 end-to-end assistant：

1. 捕获 mic input（16 kHz mono）。
2. 检测用户语音的 start/end。
3. Streaming transcribe。
4. 把 transcript 传给能调用 tools（timer、weather、calendar）的 LLM。
5. 把 LLM text stream 到 TTS。
6. 播放音频给用户。
7. 如果用户在 mid-response 打断，则停止。

Latency target：用户说完后 800 ms 内，在 laptop CPU 上发出 first TTS audio byte。Quality target：不漏词、silence 上不 hallucinate subtitles、不泄漏 voice cloning、prompt injection 零成功。

## 概念

![Voice assistant pipeline: mic → VAD → STT → LLM+tools → TTS → speaker](../assets/voice-assistant.svg)

### 七个组件

1. **Audio capture。** Mic → 16 kHz mono → 20 ms chunks。Python 中通常用 `sounddevice`，生产中用 native AudioUnit/ALSA/WASAPI。
2. **VAD（第 11 课）。** Silero VAD @ threshold 0.5，min speech 250 ms，silence hang-over 500 ms。发出 “start” 和 “end”。
3. **Streaming STT（第 4-5 课）。** Whisper-streaming、Parakeet-TDT 或 Deepgram Nova-3（API）。Partial + final transcripts。
4. **带 tool calling 的 LLM。** GPT-4o / Claude 3.5 / Gemini 2.5 Flash。Tools 用 JSON schema。Stream tokens。
5. **Streaming TTS（第 7 课）。** Kokoro-82M（最快开源）或 Cartesia Sonic（商业）。LLM 产生 20 tokens 后开始 TTS。
6. **Playback。** Speaker out；低带宽网络用 opus-encode。
7. **Interruption handler。** 如果 TTS playback 时 VAD 触发，停止 playback，取消 LLM，重启 STT。

### 你会遇到的三个失败模式

1. **First-word clip。** VAD 起得太晚。用户的 “hey” 丢了。Start threshold 用 0.3，不要 0.5。
2. **Mid-response interrupt confusion。** 用户打断后 LLM 继续生成；assistant 和用户抢话。把 VAD 接到 cancel-LLM。
3. **Silence hallucination。** Whisper 在 silent warm-up frames 上输出 “Thanks for watching”。始终 VAD-gate。

### 2026 production reference stacks

| Stack | Latency | License | Notes |
|-------|---------|---------|-------|
| LiveKit + Deepgram + GPT-4o + Cartesia | 350-500 ms | commercial API | 2026 industry default |
| Pipecat + Whisper-streaming + GPT-4o + Kokoro | 500-800 ms | mostly open | DIY-friendly |
| Moshi (full-duplex) | 200-300 ms | CC-BY 4.0 | Single-model；不同架构，第 15 课 |
| Vapi / Retell (managed) | 300-500 ms | commercial | 最快 launch；customization 有限 |
| Whisper.cpp + llama.cpp + Kokoro-ONNX | offline | open | Privacy / edge |

## 构建它

### 第 1 步：mic capture with chunking（pseudocode）

```python
import sounddevice as sd

def mic_stream(chunk_ms=20, sr=16000):
    q = queue.Queue()
    def cb(indata, frames, time, status):
        q.put(indata.copy().flatten())
    with sd.InputStream(channels=1, samplerate=sr, blocksize=int(sr * chunk_ms/1000), callback=cb):
        while True:
            yield q.get()
```

### 第 2 步：VAD-gated turn capture

```python
def capture_turn(stream, vad, pre_roll_ms=300, silence_ms=500):
    buf, pre, triggered = [], collections.deque(maxlen=pre_roll_ms // 20), False
    silent = 0
    for chunk in stream:
        pre.append(chunk)
        if vad(chunk):
            if not triggered:
                buf = list(pre)
                triggered = True
            buf.append(chunk)
            silent = 0
        elif triggered:
            silent += 20
            buf.append(chunk)
            if silent >= silence_ms:
                return b"".join(buf)
```

### 第 3 步：streaming STT → LLM → TTS

```python
async def turn(audio_bytes):
    transcript = await stt.transcribe(audio_bytes)
    async for token in llm.stream(transcript):
        async for audio in tts.stream(token):
            await speaker.play(audio)
```

### 第 4 步：LLM loop 中的 tool calling

```python
tools = [
    {"name": "get_weather", "parameters": {"location": "string"}},
    {"name": "set_timer", "parameters": {"seconds": "int"}},
]

async for chunk in llm.stream(user_text, tools=tools):
    if chunk.type == "tool_call":
        result = dispatch(chunk.name, chunk.args)
        continue_streaming(result)
    if chunk.type == "text":
        await tts.stream(chunk.text)
```

### 第 5 步：interruption handling

```python
tts_task = asyncio.create_task(tts_loop())
while True:
    chunk = await mic.get()
    if vad(chunk):
        tts_task.cancel()
        await speaker.stop()
        await new_turn()
        break
```

## 使用它

`code/main.py` 提供一个 runnable simulation，用 stub models 串起所有七个组件，即使没有硬件也能看到 pipeline 形状。真实实现中，把 stubs 替换为：

- `silero-vad`（`pip install silero-vad`）
- `deepgram-sdk` 或 `openai-whisper`
- `openai`（`gpt-4o`）或 `anthropic`
- `kokoro` 或 `cartesia`
- `sounddevice` 做 I/O

## 坑

- **永久记录 PII。** Full-turn audio 在大多数司法辖区是 PII。30-day retention，加密 at rest。
- **No barge-in。** 用户一定会打断。Assistant 必须停止说话。
- **TTS that blocks。** Synchronous TTS 会阻塞 event loop。用 async 或 separate thread。
- **No tool-call error handling。** Tools 会失败。LLM 必须拿到 error + retry once，然后 graceful degrade。
- **Overzealous hallucination filters。** 过度过滤，assistant 一直说 “I can't help with that.” 过滤不足，它什么都敢说。用 held-out set calibration。
- **No wake-word option。** Always-listening 是 privacy liability。加 wake-word gate（Porcupine 或 openWakeWord）。

## 交付它

保存为 `outputs/skill-voice-assistant-architect.md`。给定 budget + scale + language + compliance constraints，产出完整 stack spec。

## 练习

1. **简单。** 运行 `code/main.py`。它用 stub modules 模拟一个 full turn end-to-end，并打印 per-stage latency。
2. **中等。** 把 STT stub 替换为在预录 `.wav` 上运行的真实 Whisper model。测量 WER 和 end-to-end latency。
3. **困难。** 加入 tool calling：实现 `get_weather`（任意 API）和 `set_timer`。让 LLM 通过 tools 路由，并验证用户说 “set a 5 minute timer” 时正确函数被调用，且 spoken reply 确认。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|-----------------|-----------------------|
| Turn | 用户 + assistant round-trip | 一个 VAD-bounded user speech + 一个 LLM-TTS response。 |
| Barge-in | Interruption | 用户在 assistant 说话时开口；assistant 停止。 |
| Wake word | “Hey assistant” | 短 keyword detector；Porcupine、Snowboy、openWakeWord。 |
| End-pointing | Turn ending | VAD + min-silence 判断用户已说完。 |
| Pre-roll | Pre-speech buffer | 在 VAD 触发前保留 200-400 ms 音频，避免 first-word clip。 |
| Tool call | Function invocation | LLM 发出 JSON；runtime dispatch；result 回流 loop。 |

## 延伸阅读

- [LiveKit — voice agent quickstart](https://docs.livekit.io/agents/) — production-grade reference。
- [Pipecat — voice agent examples](https://github.com/pipecat-ai/pipecat) — DIY-friendly framework。
- [OpenAI Realtime API](https://platform.openai.com/docs/guides/realtime) — managed voice-native path。
- [Kyutai Moshi](https://github.com/kyutai-labs/moshi) — full-duplex reference（第 15 课）。
- [Porcupine wake-word](https://picovoice.ai/products/porcupine/) — wake-word gating。
- [Anthropic — tool use guide](https://docs.anthropic.com/en/docs/build-with-claude/tool-use) — LLM function calling。
