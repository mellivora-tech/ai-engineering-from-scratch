# Streaming Speech-to-Speech：Moshi、Hibiki 与 Full-Duplex Dialogue

> 2024-2026 重新定义了 voice AI。Moshi 用单个模型同时听和说，延迟 200 ms。Hibiki chunk-by-chunk 做 speech-to-speech translation。两者都放弃 ASR → LLM → TTS pipeline，转向基于 Mimi codec tokens 的统一 full-duplex architecture。这是新的 reference design。

**类型：** 学习
**语言：** Python
**前置要求：** 阶段 6 · 13（Neural Audio Codecs），阶段 6 · 11（Real-Time Audio），阶段 7 · 05（Full Transformer）
**时间：** ~75 分钟

## 问题

每个由第 11 + 12 课构建的 voice agent，都有一个约 300-500 ms 的基础 latency floor：VAD 触发，STT 处理，LLM reasoning，TTS 生成。每个阶段都有自己的最低延迟。你可以调优和并行，但 pipeline 形状限制了上限。

Moshi（Kyutai, 2024-2026）提出另一个问题：如果没有 pipeline 呢？如果一个模型直接拿 audio in，并连续发出 audio out，而 text 只是中间的 “inner monologue”，不是必经阶段呢？

答案是 **full-duplex speech-to-speech**。理论延迟 160 ms（80 ms Mimi frame + 80 ms acoustic delay）。单张 L4 GPU 上 practical latency 200 ms。这是最佳 pipelined voice agent 的一半。

## 概念

![Moshi architecture: two parallel Mimi streams + inner-monologue text](../assets/moshi-hibiki.svg)

### Moshi 架构

**Inputs。** 两条 Mimi codec streams，都是 12.5 Hz × 8 codebooks：

- Stream 1：user audio（Mimi-encoded，持续到达）
- Stream 2：Moshi 自己的 audio（Moshi 生成的）

**Transformer。** 7B-parameter Temporal Transformer 处理两条 streams 和一条 text “inner monologue” stream。在每个 80 ms step，它：

1. 消耗最新 user Mimi tokens（8 codebooks）。
2. 消耗最近的 Moshi Mimi tokens（8 codebooks，由 Moshi 自己产生）。
3. 生成下一个 Moshi text token（inner monologue）。
4. 通过小型 Depth Transformer 生成下一个 Moshi Mimi tokens（8 codebooks）。

三条 streams：user audio、Moshi audio、Moshi text 并行运行。Moshi 能在说话时听用户；能在用户打断时自我打断；能 back-channel（“mhm”）而不破坏主 utterance。

**Depth transformer。** 在一个 frame 内，8 个 codebooks 不是并行预测的，它们有 inter-codebook dependencies。一个小型 2-layer “depth transformer” 在 80 ms 内 sequentially 预测它们。这是 AR codec LMs 的标准 factorization（VALL-E、VibeVoice 也使用）。

### 为什么 inner-monologue text 有帮助

没有显式 text，模型必须在 acoustic stream 中隐式建模 language。Moshi 的洞见：强制它在 audio 旁边发出 text tokens。Text stream 本质上是 Moshi 正在说的 transcript。这提升 semantic coherence，让替换 language model head 更容易，并免费给你 transcripts。

### Hibiki：streaming speech-to-speech translation

同样架构，在 translation pairs 上训练。Source audio in，target-language audio out，连续进行。Hibiki-Zero（2026 年 2 月）消除了对 word-level aligned training data 的需求，使用 sentence-level data + GRPO reinforcement learning 做 latency optimization。

最初支持四个 language pairs；用约 1000 小时可适配新语言。

### 更广的 Kyutai stack（2026）

- **Moshi**：full-duplex dialogue（法语优先，英语支持好）
- **Hibiki / Hibiki-Zero**：simultaneous speech translation
- **Kyutai STT**：streaming ASR（500 ms 或 2.5 s look-ahead）
- **Kyutai Pocket TTS**：100M-param TTS，在 CPU 上运行（Jan 2026）
- **Unmute**：在 public servers 上组合这些的 full pipeline

L40S GPU throughput：64 concurrent sessions at 3× real-time。

### Sesame CSM：近亲

Sesame CSM（2025）使用类似思想：Llama-3 backbone + Mimi codec head。但 CSM 是单向的（拿 context + text，产 speech），不是 full-duplex。它是市场上最好的 “voice presence” TTS，但不等同于 Moshi 的 full-duplex capability。

### 2026 performance numbers

| Model | Latency | Use case | License |
|-------|---------|----------|---------|
| Moshi | 200 ms (L4) | full-duplex English / French dialogue | CC-BY 4.0 |
| Hibiki | 12.5 Hz framerate | French ↔ English streaming translation | CC-BY 4.0 |
| Hibiki-Zero | same | 5 language-pairs, no aligned data | CC-BY 4.0 |
| Sesame CSM-1B | 200 ms TTFA | context-conditioned TTS | Apache-2.0 |
| GPT-4o Realtime | ~300 ms | closed, OpenAI API | commercial |
| Gemini 2.5 Live | ~350 ms | closed, Google API | commercial |

## 构建它

### 第 1 步：interface

Moshi 暴露一个 WebSocket server，接收 80 ms chunks 的 Mimi-encoded audio，并返回 80 ms chunks 的 Mimi-encoded audio。双向。持续不断。

```python
import asyncio
import websockets
from moshi.client_utils import encode_audio_mimi, decode_audio_mimi

async def moshi_chat():
    async with websockets.connect("ws://localhost:8998/api/chat") as ws:
        mic_task = asyncio.create_task(stream_mic_to(ws))
        spk_task = asyncio.create_task(stream_from_to_speaker(ws))
        await asyncio.gather(mic_task, spk_task)
```

### 第 2 步：full-duplex loop

```python
async def stream_mic_to(ws):
    async for chunk_80ms in mic_stream_at_12_5_hz():
        mimi_tokens = encode_audio_mimi(chunk_80ms)
        await ws.send(serialize(mimi_tokens))

async def stream_from_to_speaker(ws):
    async for msg in ws:
        mimi_tokens, text_token = deserialize(msg)
        audio = decode_audio_mimi(mimi_tokens)
        await play(audio)
```

两个方向同时运行。Python asyncio 或 Rust futures 是标准 transport。

### 第 3 步：training objective（概念）

对每个 80 ms frame `t`：

- Input：`user_mimi[0..t]`、`moshi_mimi[0..t-1]`、`moshi_text[0..t-1]`
- Predict：`moshi_text[t]`，再预测 `moshi_mimi[t, codebook_0..7]`

Text 在 audio 前预测（inner monologue）；audio 在 depth transformer 内按 codebook 顺序预测。

### 第 4 步：Moshi 赢在哪里，不赢在哪里

Moshi wins：

- 廉价硬件上 sub-250 ms end-to-end。
- 自然 back-channels 和 interruptions。
- 没有 pipeline glue code。

Moshi does not win：

- Tool calling（不是为它训练的；你需要 separate LLM path）。
- Long reasoning（Moshi 是约 8B 的 dialogue model，不是 Claude/GPT-4）。
- Niche topics 上的 factual accuracy。
- 大多数 production enterprise use cases（2026 年仍用 pipelines）。

## 使用它

| 场景 | 选择 |
|-----------|------|
| Lowest-latency voice companion | Moshi |
| Live translation call | Hibiki |
| Voice demo / research | Moshi、CSM |
| Enterprise agent with tools | Pipeline（第 12 课），不是 Moshi |
| Custom-voice TTS in context | Sesame CSM |
| Speech-to-speech，任意 languages | GPT-4o Realtime 或 Gemini 2.5 Live（commercial） |

## 坑

- **Limited tool calling。** Moshi 是 dialogue model，不是 agent framework。工具需求要和 pipeline 结合。
- **Specific-voice conditioning。** Moshi 使用一个 trained persona；cloning 是单独训练。
- **Language coverage。** 法语 + 英语优秀，其他有限。Hibiki-Zero 有帮助，但仍需要训练数据。
- **Resource cost。** 一个完整 Moshi session 占一个 GPU slot，不是便宜的 shared-tenant deploy pattern。

## 交付它

保存为 `outputs/skill-duplex-pipeline.md`。为 voice-agent workload 选择 pipeline vs full-duplex architecture，并给出理由。

## 练习

1. **简单。** 运行 `code/main.py`。它符号化模拟 two-stream + inner-monologue architecture。
2. **中等。** 从 HuggingFace 拉取 Moshi，运行 server，测试一次对话。测量从 end-of-user-speech 到 start-of-Moshi-response 的 wall-clock latency。
3. **困难。** 拿你的第 12 课 pipeline agent，和 Moshi 在 20 个 matched test utterances 上比较 P50 latency。写清楚 pipeline 在哪些情况下架构上仍然胜出。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|-----------------|-----------------------|
| Full-duplex | 边听边说 | 同一个模型上两个 audio streams 同时 active。 |
| Inner monologue | 模型的 text stream | Moshi 在 audio output 旁边发 text tokens。 |
| Depth transformer | Inter-codebook predictor | 小 transformer，在一个 80 ms frame 内预测 8 个 codebooks。 |
| Mimi | Kyutai codec | 12.5 Hz × 8 codebooks；semantic+acoustic；驱动 Moshi。 |
| Streaming S2S | Audio → audio live | Chunk-by-chunk translation/dialogue，无 pipeline stages。 |
| Back-channeling | “Mhm” reactions | Moshi 可发出小 acknowledgment 而不破坏自己的 turn。 |

## 延伸阅读

- [Défossez et al. (2024). Moshi — speech-text foundation model](https://arxiv.org/html/2410.00037v2) — 论文。
- [Kyutai Labs (2026). Hibiki-Zero](https://arxiv.org/abs/2602.12345) — 无 aligned data 的 streaming translation。
- [Sesame (2025). Crossing the uncanny valley of voice](https://www.sesame.com/research/crossing_the_uncanny_valley_of_voice) — CSM spec。
- [Kyutai — Moshi repo](https://github.com/kyutai-labs/moshi) — install + server。
- [OpenAI — Realtime API](https://platform.openai.com/docs/guides/realtime) — closed commercial peer。
- [Kyutai — Delayed Streams Modeling](https://github.com/kyutai-labs/delayed-streams-modeling) — 底层 STT/TTS framework。
