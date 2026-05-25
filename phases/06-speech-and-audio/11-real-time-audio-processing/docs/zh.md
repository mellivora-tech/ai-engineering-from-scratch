# Real-Time Audio Processing

> Batch pipelines 处理一个文件。Real-time pipelines 必须在下一个 20 ms 到来前处理当前 20 ms。每个 conversational AI、broadcast studio 和 telephony bot 都靠这个 latency budget 生存。

**类型：** 构建
**语言：** Python, Rust
**前置要求：** 阶段 6 · 02（Spectrograms），阶段 6 · 04（ASR），阶段 6 · 07（TTS）
**时间：** ~75 分钟

## 问题

你想要一个感觉活着的 voice assistant。人类 conversation turn-taking latency 大约是 230 ms（silence-to-response）。超过 500 ms 会显得机械；超过 1500 ms 会显得坏掉。2026 年完整 **hear → understand → respond → speak** loop 的预算是：

| Stage | Budget |
|-------|--------|
| Mic → buffer | 20 ms |
| VAD | 10 ms |
| ASR (streaming) | 150 ms |
| LLM (first token) | 100 ms |
| TTS (first chunk) | 100 ms |
| Render → speaker | 20 ms |
| **Total** | **~400 ms** |

Moshi（Kyutai, 2024）达到 200 ms full-duplex。GPT-4o-realtime（2024）约 320 ms。2022 年 cascaded pipelines 以 2500 ms 交付。10 倍提升来自三项技术：（1）everything streaming，（2）用 partial results 做 asynchronous pipelining，（3）interruptible generation。

## 概念

![Streaming audio pipeline with ring buffer, VAD gate, interruption](../assets/real-time.svg)

**Frame / chunk / window。** Real-time audio 以固定大小 blocks 流动。常见选择：20 ms（16 kHz 下 320 samples）。下游所有部分都必须跟上这个 cadence。

**Ring buffer。** 固定大小 circular buffer。Producer thread 写新 frames，consumer thread 读。防止 hot path 中 allocation。大小 ≈ maximum-latency × sample-rate；2 秒 16 kHz ring = 32,000 samples。

**VAD（Voice Activity Detection）。** 无人说话时 gate downstream work。Silero VAD 4.0（2024）在 CPU 上每 30 ms frame <1 ms。`webrtcvad` 是旧替代。

**Streaming ASR。** 音频到达时发出 partial transcripts 的模型。Streaming mode 下 Parakeet-CTC-0.6B（NeMo, 2024）在 320 ms latency 达到 2-5% WER。Whisper-Streaming（Macháček et al., 2023）把 Whisper chunk 成 near-streaming，延迟约 2 s。

**Interruption。** 用户在 assistant 说话时开口，你必须：（a）检测 barge-in，（b）停止 TTS，（c）丢弃剩余 LLM output。整个过程要在 100 ms 内完成，否则用户会觉得 assistant 听不见。

**WebRTC Opus transport。** 20 ms frames、48 kHz、adaptive bitrate 8-128 kbps。Browser 和 mobile 标准。LiveKit、Daily.co、Pion 是 2026 构建 voice apps 的 stacks。

**Jitter buffer。** 网络 packets 会乱序或延迟到达。Jitter buffer 负责 reorder 和平滑；太小会有 audible gaps，太大会增加 latency。典型 60-80 ms。

### 常见 gotchas

- **Thread contention。** Python GIL + heavy models 可能饿死 audio thread。用 C-callback audio library（sounddevice、PortAudio），并让 Python 离开 hot path。
- **Sample-rate conversion latency。** Pipeline 内 resampling 增加 5-20 ms。要么 upfront resample，要么用 zero-latency resampler（PolyPhase、`soxr_hq`）。
- **TTS priming。** 即使 Kokoro 这样的快 TTS，首次请求也有 100-200 ms warm-up。Cache model，并在第一次真实 turn 前 dummy run。
- **Echo cancellation。** 没有 AEC，TTS output 会重新进 mic，并触发 ASR 识别 bot 自己的声音。WebRTC AEC3 是开源默认。

## 构建它

### 第 1 步：ring buffer

```python
import collections

class RingBuffer:
    def __init__(self, capacity):
        self.buf = collections.deque(maxlen=capacity)
    def write(self, frame):
        self.buf.extend(frame)
    def read(self, n):
        return [self.buf.popleft() for _ in range(min(n, len(self.buf)))]
    def level(self):
        return len(self.buf)
```

Capacity 决定最大 buffering latency。16 kHz 下 32,000 samples = 2 s。

### 第 2 步：VAD gate

```python
def simple_energy_vad(frame, threshold=0.01):
    return sum(x * x for x in frame) / len(frame) > threshold ** 2
```

生产中替换为 Silero VAD：

```python
import torch
vad, _ = torch.hub.load("snakers4/silero-vad", "silero_vad")
is_speech = vad(torch.tensor(frame), 16000).item() > 0.5
```

### 第 3 步：streaming ASR

```python
# Parakeet-CTC-0.6B streaming via NeMo
from nemo.collections.asr.models import EncDecCTCModelBPE
asr = EncDecCTCModelBPE.from_pretrained("nvidia/parakeet-ctc-0.6b")
# chunk_ms=320 ms, look_ahead_ms=80 ms
for chunk in audio_stream():
    partial_text = asr.transcribe_streaming(chunk)
    print(partial_text, end="\r")
```

### 第 4 步：interruption handler

```python
class Dialog:
    def __init__(self):
        self.tts_task = None

    def on_user_speech(self, frame):
        if self.tts_task and not self.tts_task.done():
            self.tts_task.cancel()   # barge-in
        # then feed to streaming ASR

    def on_final_user_utterance(self, text):
        self.tts_task = asyncio.create_task(self.reply(text))

    async def reply(self, text):
        async for tts_chunk in llm_then_tts(text):
            speaker.write(tts_chunk)
```

关键在 async I/O 和 cancellable TTS streaming。WebRTC 中停止 audio track 的 canonical 方法是 peerconnection.stop()。

## 使用它

2026 年技术栈：

| Layer | 选择 |
|-------|------|
| Transport | LiveKit（WebRTC）或 Pion（Go） |
| VAD | Silero VAD 4.0 |
| Streaming ASR | Parakeet-CTC-0.6B 或 Whisper-Streaming |
| LLM first-token | Groq、Cerebras、vLLM-streaming |
| Streaming TTS | Kokoro 或 ElevenLabs Turbo v2.5 |
| Echo cancel | WebRTC AEC3 |
| End-to-end native | OpenAI Realtime API 或 Moshi |

## 坑

- **为安全 buffer 500 ms。** Buffer 本身就是 latency floor。缩小它。
- **不 pin threads。** Audio callback 在低于 UI 的 priority thread 上，会在负载下 glitch。
- **TTS chunks 太小。** 低于 200 ms 的 chunks 会让 vocoder artifacts 可闻。320 ms chunks 是 sweet spot。
- **No jitter buffer。** 真实网络有 jitter；没有 smoothing 会有 pops。
- **Single-shot error handling。** Audio pipelines 必须 crash-proof。一个 exception 会杀死 session。

## 交付它

保存为 `outputs/skill-realtime-designer.md`。设计一个 real-time audio pipeline，并为每个 stage 给出 concrete latency budgets。

## 练习

1. **简单。** 运行 `code/main.py`。模拟 ring buffer + energy VAD；为 fake 10-second stream 打印 stage latencies。
2. **中等。** 用 `sounddevice` 构建 passthrough loop，以 20 ms frames 处理你的 mic，并打印每帧 VAD state。
3. **困难。** 用 `aiortc` 构建 full duplex echo test：browser → WebRTC → Python → WebRTC → browser。用 1 kHz pulse 测 glass-to-glass latency。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|-----------------|-----------------------|
| Ring buffer | Circular queue | 固定大小、lock-free（或 SPSC-locked）的 audio frame FIFO。 |
| VAD | Silence gate | 标记 speech vs non-speech 的模型或 heuristic。 |
| Streaming ASR | Real-time STT | 音频到达时发 partial text；bounded lookahead。 |
| Jitter buffer | Network smoother | Queue 重排乱序 packets；典型 60-80 ms。 |
| AEC | Echo cancellation | 抵消 speaker-to-mic feedback path。 |
| Barge-in | User interrupt | 系统在 TTS 中检测用户说话；必须取消 playback。 |
| Full duplex | 双向同时 | 用户和 bot 可同时说话；Moshi 是 full duplex。 |

## 延伸阅读

- [Macháček et al. (2023). Whisper-Streaming](https://arxiv.org/abs/2307.14743) — chunked near-streaming Whisper。
- [Kyutai (2024). Moshi](https://kyutai.org/Moshi.pdf) — full-duplex 200 ms latency。
- [LiveKit Agents framework (2024)](https://docs.livekit.io/agents/) — production audio agent orchestration。
- [Silero VAD repo](https://github.com/snakers4/silero-vad) — sub-1 ms VAD，Apache 2.0。
- [WebRTC AEC3 paper](https://webrtc.googlesource.com/src/+/main/modules/audio_processing/aec3/) — 开源 echo cancellation。
