# Voice Activity Detection 与 Turn-Taking：Silero、Cobra 和 Flush Trick

> 每个 voice agent 都靠两个决策生存：用户现在是否在说话，以及他们是否说完了。VAD 回答第一个。Turn-detection（VAD + silence-hangover + semantic endpoint model）回答第二个。任何一个错了，assistant 要么打断用户，要么永远不闭嘴。

**类型：** 构建
**语言：** Python
**前置要求：** 阶段 6 · 11（Real-Time Audio），阶段 6 · 12（Voice Assistant）
**时间：** ~45 分钟

## 问题

Voice agent 在每个 20 ms chunk 上做三个不同决策：

1. **这一帧是 speech 吗？**——VAD。Binary，per-frame。
2. **用户开始了新 utterance 吗？**——onset detection。
3. **用户说完了吗？**——end-pointing（turn-end）。

朴素答案（energy threshold）在任何噪声中都会失败：traffic、keyboards、crowd babble。2026 年答案是：Silero VAD（open、deep-learned）+ turn-detection model（semantic endpointing）+ VAD-calibrated silence hangover。

## 概念

![VAD cascade: energy → Silero → turn-detector → flush trick](../assets/vad-turn-taking.svg)

### 三层 VAD cascade

**Tier 1：energy gate。** 最便宜。在 -40 dBFS 阈值上 threshold RMS。能过滤明显 silence，但任何超过 threshold 的 noise 都会触发。

**Tier 2：Silero VAD**（2020-2026，MIT）。1M 参数。6000+ 语言训练。单 CPU thread 每 30 ms chunk 约 1 ms。TPR 87.7% @ 5% FPR。开源默认。

**Tier 3：semantic turn detector。** LiveKit turn-detection model（2024-2026）或你自己的小 classifier。区分“句中停顿”和“说完了”。使用 linguistic context（intonation + recent words），不只是 silence。

### 关键参数及默认值

- **Threshold。** Silero 输出 probability；> 0.5（默认）或 > 0.3（更敏感）判为 speech。阈值越低，first-word clips 越少，false positives 越多。
- **Minimum speech duration。** 拒绝短于 250 ms 的 speech，通常是咳嗽或椅子噪声。
- **Silence hangover（end-pointing）。** VAD 回到 0 后，等 500-800 ms 再宣布 turn end。太短会打断用户；太长会显得慢。
- **Pre-roll buffer。** VAD 触发前保留 300-500 ms 音频。防止 “hey” 被截掉。

### Flush trick（Kyutai 2025）

Streaming STT models 有 look-ahead delay（Kyutai STT-1B 为 500 ms，STT-2.6B 为 2.5 s）。通常你要在 end-of-speech 后等待那么久才能拿到 transcript。Flush trick：当 VAD 触发 end-of-speech，**给 STT 发送 flush signal**，强制立即输出。STT 以约 4× realtime 处理，所以 500 ms buffer 会在约 125 ms 完成。

End-to-end：125 ms VAD + flush STT = conversational latency。

### 2026 VAD comparison

| VAD | TPR @ 5% FPR | Latency | License |
|-----|--------------|---------|---------|
| WebRTC VAD (Google, 2013) | 50.0% | 30 ms | BSD |
| Silero VAD (2020-2026) | 87.7% | ~1 ms | MIT |
| Cobra VAD (Picovoice) | 98.9% | ~1 ms | commercial |
| pyannote segmentation | 95% | ~10 ms | MIT-ish |

Silero 是正确默认。Cobra 是 compliance / accuracy upgrade。Energy-only VAD 在 2026 生产中没有位置。

## 构建它

### 第 1 步：energy gate

```python
def energy_vad(chunk, threshold_dbfs=-40.0):
    rms = (sum(x * x for x in chunk) / len(chunk)) ** 0.5
    dbfs = 20.0 * math.log10(max(rms, 1e-10))
    return dbfs > threshold_dbfs
```

### 第 2 步：Python 中的 Silero VAD

```python
from silero_vad import load_silero_vad, get_speech_timestamps

vad = load_silero_vad()
audio = torch.tensor(waveform_16k, dtype=torch.float32)
segments = get_speech_timestamps(
    audio, vad, sampling_rate=16000,
    threshold=0.5,
    min_speech_duration_ms=250,
    min_silence_duration_ms=500,
    speech_pad_ms=300,
)
for s in segments:
    print(f"{s['start']/16000:.2f}s - {s['end']/16000:.2f}s")
```

### 第 3 步：turn-end state machine

```python
class TurnDetector:
    def __init__(self, silence_hangover_ms=500, min_speech_ms=250):
        self.state = "idle"
        self.speech_ms = 0
        self.silence_ms = 0
        self.silence_hangover_ms = silence_hangover_ms
        self.min_speech_ms = min_speech_ms

    def update(self, is_speech, chunk_ms=20):
        if is_speech:
            self.speech_ms += chunk_ms
            self.silence_ms = 0
            if self.state == "idle" and self.speech_ms >= self.min_speech_ms:
                self.state = "speaking"
                return "START"
        else:
            self.silence_ms += chunk_ms
            if self.state == "speaking" and self.silence_ms >= self.silence_hangover_ms:
                self.state = "idle"
                self.speech_ms = 0
                return "END"
        return None
```

### 第 4 步：flush trick skeleton

```python
def flush_on_end(stt_client, audio_buffer):
    stt_client.send_audio(audio_buffer)
    stt_client.send_flush()
    return stt_client.recv_transcript(timeout_ms=150)
```

STT（Kyutai、Deepgram、AssemblyAI）必须支持 flush。Whisper streaming 不支持；它是 block-based，总是等待 chunks。

## 使用它

| 场景 | VAD choice |
|-----------|-----------|
| Open, fast, general | Silero VAD |
| Commercial call center | Cobra VAD |
| On-device（phone） | Silero VAD ONNX |
| Research / diarization | pyannote segmentation |
| Zero-dependency fallback | WebRTC VAD（legacy） |
| 需要 turn-ending quality | Silero + LiveKit turn-detector layered |

经验法则：除非真的没有选择，永远不要交付 energy-only VAD。

## 坑

- **Fixed threshold。** 安静环境好用，噪声中失败。要么 on-device calibrate，要么换 Silero。
- **Too-short silence hangover。** Agent 会在句中打断用户。500-800 ms 是 conversational speech 的 sweet spot。
- **Too-long hangover。** 感觉慢。对目标用户 A/B test。
- **No pre-roll buffer。** 用户音频前 200-300 ms 丢失。始终保留 rolling pre-roll。
- **Ignoring semantic endpointing。** “Hmm, let me think...” 包含长停顿。用户讨厌被打断思路。使用 LiveKit turn-detector 或类似模型。

## 交付它

保存为 `outputs/skill-vad-tuner.md`。为 workload 选择 VAD model、threshold、hangover、pre-roll 和 turn-detection strategy。

## 练习

1. **简单。** 运行 `code/main.py`。它模拟 speech + silence + speech + coughs sequence，并测试三层 VAD。
2. **中等。** 安装 `silero-vad`，处理一段 5 分钟录音，调 threshold 以同时最小化 first-word clips 和 false triggers。报告 precision/recall。
3. **困难。** 构建 mini turn-detector：Silero VAD + last 10 words embeddings 上的 3-layer MLP（用 sentence-transformers）。在手工标注 turn-end dataset 上训练。比 Silero-only 提升 10% F1。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|-----------------|-----------------------|
| VAD | Voice detector | Binary per-frame：这是 speech 吗？ |
| Turn detection | End-pointing | VAD + silence-hangover + semantic endpoint。 |
| Silence hangover | Wait-after-speech | 宣布 turn end 前等待时间；500-800 ms。 |
| Pre-roll | Pre-speech buffer | VAD 触发前保留 300-500 ms 音频。 |
| Flush trick | Kyutai hack | VAD → flush-STT → 125 ms，而不是 500 ms delay。 |
| Semantic endpoint | “他们是想停吗？” | 看 words 而不只看 silence 的 ML classifier。 |
| TPR @ FPR 5% | ROC point | 标准 VAD benchmark；Silero 87.7%，WebRTC 50%。 |

## 延伸阅读

- [Silero VAD](https://github.com/snakers4/silero-vad) — open VAD 参考。
- [Picovoice Cobra VAD](https://picovoice.ai/products/cobra/) — 商业 accuracy leader。
- [Kyutai — Unmute + flush trick](https://kyutai.org/stt) — sub-200 ms 工程技巧。
- [LiveKit — turn detection](https://docs.livekit.io/agents/logic/turns/) — 生产中的 semantic endpointing。
- [WebRTC VAD](https://webrtc.googlesource.com/src/) — legacy baseline。
- [pyannote segmentation](https://github.com/pyannote/pyannote-audio) — diarization-grade segmentation。
