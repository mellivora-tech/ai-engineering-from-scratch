# Omni Models：Qwen2.5-Omni 与 Thinker-Talker Split

> GPT-4o 在 2024 年 5 月的产品 demo 具有冲击力，不是因为底层模型，而是因为产品形态：一个语音界面，你说话，模型看到摄像头看到的东西，并在 250ms 内说话回应。开放生态在 2024 和 2025 年剩余时间里都在追赶这个 product surface。Qwen2.5-Omni（2025 年 3 月）是参考开放设计：Thinker（大型 text-generating transformer）加 Talker（parallel speech-generating transformer），由 streaming speech tokens 连接。Mini-Omni 简化了它，Moshi 匹配了 latency，GLM-4-Voice 扩展到中文。本课阅读 Thinker-Talker 架构，以及让 streaming real-time dialogue 成立的 latency budget。

**类型：** 构建
**语言：** Python（stdlib，streaming pipeline latency simulator + VAD loop）
**前置要求：** 阶段 12 · 19（audio-LLMs），阶段 12 · 16（any-to-any）
**时间：** ~180 分钟

## 学习目标

- 把 inference pipeline 拆成 Thinker（text reasoning）和 Talker（speech synthesis），并解释为什么 parallel streaming 有效。
- 逐组件计算 conversational interaction 的 time-to-first-audio-byte（TTFAB）budget。
- 描述 TMRoPE 在 Thinker 内部跨 vision、audio 和 text 的 time-aligned position encoding。
- 说出三种 real-time conversational patterns：half-duplex、turn-taking、full-duplex。

## 问题

实时 voice assistant 必须快速完成很多事：

1. 听用户。Real-time speech tokenization，voice activity detection（VAD）判断用户是否说完。
2. 可选地看。Camera input at 2-4 FPS，与 audio 一起 stream 到 Thinker。
3. 思考。基于 conversation history 组合 response。
4. 说话。合成 audio tokens，decode 到 waveform，stream 到用户扬声器。

每一步都增加 latency。Conversational-feel 需要 total round-trip < 500ms；低于这个阈值，用户基本不再注意到延迟。GPT-4o 声称约 250ms。Moshi 约 160ms。Qwen2.5-Omni 约 350-500ms。

每个组件都需要 stream。不能“batch everything then decode”。

## 概念

### Thinker 与 Talker

Qwen2.5-Omni 的分解：

- Thinker：7B-80B text-generating transformer。消费 interleaved text + image + audio tokens。输出代表要说什么的 text tokens。
- Talker：更小的 speech-generating transformer（200M-1B）。消费 Thinker 的 text output tokens 加最近 speech-context tokens。输出 discrete speech tokens（residual-VQ indices）。
- Speech decoder：streaming waveform decoder（SNAC、MoVQGAN family），实时把 speech tokens 转为 audio samples。

分离很重要。Thinker 必须很大才能有良好 reasoning。Talker 可以小，因为它的工作是局部的：把文本转换成 speech tokens。更大的 Talker 不会更有表达力，只会更慢。

并行运行：

1. Thinker 发出 text token t_i。
2. Talker 通过 streaming 消费 t_i，并发出 speech tokens s_i, s_{i+1}, ..., s_{i+k}。
3. Speech decoder 边收到 speech tokens 边输出 audio samples。
4. 当 Thinker 到达 text token t_{i+3} 时，Talker 已经 stream 了 t_0..t_{i+2} 的 audio。

### TMRoPE：time-aligned multimodal positions

Thinker 需要整合 image frames（比如 4 FPS 到达）、audio frames（50 frames/second 到达）和 conversation history 中的 text。朴素 sequence order（先所有 images，再所有 audio，再 text）会丢失 temporal alignment。

TMRoPE 给每个 token 分配 absolute timestamp。Vision token at t=2.3s。Audio token at t=2.32s。用户说 “stop” 的 text token at t=2.35s。RoPE 按 timestamp 旋转 attention；模型把它们看作时间上并发。

这是让“他说 hello 的同时挥手”成立的基础设施：模型看到同一个概念时刻的视频帧和音频。

### Streaming speech synthesis

Speech tokens 必须 stream。Mini-Omni（Xie & Wu，2024）提出 “language models can hear, talk while thinking in streaming”：Thinker output tokens 和 Talker output tokens 在同一序列中交错。Talker 一旦 Thinker commit 下一个 text token 就触发。没有 batch boundaries。

Moshi（Défossez 等人，2024 年 10 月）是最快的 open implementation。单 A100 上 160ms TTFAB。架构是一个 7B transformer，在交替位置发出 text 和 speech tokens，并使用 “inner monologue” 分离 thinking stream 与 speaking stream。这本质上是把 Thinker + Talker 融合进一个模型并谨慎训练。

### VAD 与 turn-taking

Voice activity detection 在 input side 运行。两种模式：

- Half-duplex：用户说话，模型听。模型说话，用户听。通过 VAD silence detection（约 200ms）实现清晰 handoff。
- Full-duplex：双方可同时说话。模型可 backchannel（“uh-huh”）或打断。难得多。Moshi 支持这个。

Qwen2.5-Omni 默认支持 half-duplex，通过 silence threshold 做 turn-taking。Full-duplex 需要 application-layer handling。

### Qwen3-Omni（2025 年 11 月）

后继版本。Qwen3-80B Thinker、更大的 Talker、改进的 TMRoPE-v2。Latency 接近 GPT-4o 的 250ms。Open weights。在 OmniBench 上与 Gemini 2.0 Live 有竞争力。

### Production latency budget

典型 streaming interaction：

- Mic -> audio tokens：40-80ms。
- Prefill（prompt + history）：7B 下 100-200ms，70B 下更多。
- First Thinker text token：40ms。
- Talker 处理第一个 text token：20ms。
- First speech tokens commit：40ms。
- Residual-VQ decode：30ms。
- Speech waveform decode：50-80ms。

总 TTFAB：7B 下 320-510ms，70B 下 600-900ms。Frontier quality 通常需要 70B+，这就是 frontier latency gap 的来源。

### Token-rate math

16kHz speech、50 Hz base speech tokens 下，每秒 output 需要 50 个 speech tokens。Talker 必须发出 ≥50 tok/s 才能跟上。在 H100 上典型 LLM throughput 为 30-80 tok/s，所以小型（200-300M）Talker 足够快；7B Talker 会落后。

这就是为什么存在小型专用 Talker models，而不是“直接用主模型”。

## 使用它

`code/main.py`：

- 用 mock token-emission rates 模拟 Thinker-Talker pipeline。
- 为可配置 model sizes 和 mic sample rates 计算 TTFAB。
- 用 VAD silence threshold 演示 half-duplex turn-taking。

## 交付它

本课产出 `outputs/skill-omni-streaming-budget.md`。给定 real-time voice product 的 target TTFAB 和 feature set（vision-in、bilingual、full-duplex），它会选择 Qwen2.5-Omni、Qwen3-Omni、Moshi 或 Mini-Omni，并确定 Thinker/Talker sizing。

## 练习

1. 你的 target TTFAB 是 300ms。在 7B Thinker 和 300M Talker 上写出每个组件的 latency。

2. Qwen2.5-Omni 使用 TMRoPE。描述当用户在 t=1s 开始说话、摄像头在 t=1.2s 捕捉到手势时，模型看到了什么。

3. Full-duplex support 要求模型一边听一边发出 audio。提出一种训练数据格式来教会它。

4. 阅读 Moshi 论文第 4 节。描述 “inner monologue” separation，以及它为什么避免 Thinker-Talker split。

5. 计算 throughput budget：为了跟上 16kHz speech at 50 base-layer tokens/sec，Talker 必须多快 emit tokens？

## 关键术语

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| Thinker | “Reasoning brain” | 产生“要说什么”的大型 text-generating transformer |
| Talker | “Speech-generating mouth” | 从 Thinker 文本产生 discrete speech tokens 的小型 transformer |
| TTFAB | “Latency budget” | Time-to-first-audio-byte：从用户语音结束到首个 audio sample 输出 |
| TMRoPE | “Time-aligned RoPE” | 在 vision、audio、text 间使用 absolute timestamps 的 position encoding |
| Half-duplex | “Turn-taking” | 用户和模型轮流说话；VAD silence 判断用户说完 |
| Full-duplex | “Simultaneous” | 模型可以同时说和听；支持 backchannel |
| Inner monologue | “Moshi separation” | Single-model design，thinking-stream 与 speaking-stream 交错 |

## 延伸阅读

- [Xu et al. — Qwen2.5-Omni (arXiv:2503.20215)](https://arxiv.org/abs/2503.20215)
- [Qwen Team — Qwen3-Omni (arXiv:2509.17765)](https://arxiv.org/html/2509.17765v1)
- [Xie & Wu — Mini-Omni (arXiv:2408.16725)](https://arxiv.org/abs/2408.16725)
- [Défossez et al. — Moshi (arXiv:2410.00037)](https://arxiv.org/abs/2410.00037)
- [Zeng et al. — GLM-4-Voice (arXiv:2412.02612)](https://arxiv.org/abs/2412.02612)
