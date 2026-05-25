# Audio-Language Models：Qwen2.5-Omni、Audio Flamingo、GPT-4o Audio

> 2026 年 audio-language models 能对 speech + environmental sound + music 做 reasoning。Qwen2.5-Omni-7B 在 MMAU-Pro 上匹配 GPT-4o Audio。Audio Flamingo Next 在 LongAudioBench 上击败 Gemini 2.5 Pro。开源和闭源之间的差距基本关闭了，除了 multi-audio tasks，所有人都接近随机。

**类型：** 学习
**语言：** Python
**前置要求：** 阶段 6 · 04（ASR），阶段 12 · 03（Vision-Language Models），阶段 7 · 10（Audio Transformers）
**时间：** ~45 分钟

## 问题

你有 5 秒 audio：狗叫，有人喊 “stop!”，然后 silence。有用的问题跨多个轴：

- **Transcription。** “说了什么？”——ASR 范畴。
- **Semantic reasoning。** “这个人有危险吗？”——需要联合理解狗叫 + 叫喊 + silence。
- **Music reasoning。** “哪些乐器在演奏 melody？”
- **Long-audio retrieval。** “在这 90 分钟 lecture 中，老师在哪里解释 gradient descent？”

一个能用同一个 prompt 回答这些问题的模型就是 **audio-language model**（LALM / ALM）。它不同于纯 ASR：LALMs 生成自由形式自然语言答案，而不只是 transcripts。

## 概念

![Audio-language model: audio encoder + projector + LLM decoder](../assets/alm-architecture.svg)

### 三组件模板

每个 2026 LALM 都有相同骨架：

1. **Audio encoder。** Whisper encoder、BEATs、CLAP、WavLM，或模型自定义 encoder。
2. **Projector。** Linear 或 MLP，把 audio-encoder features bridge 到 LLM token embedding space。
3. **LLM。** Llama / Qwen / Gemma-based decoder。接受 interleaved text + audio tokens；生成 text。

训练：

- **Stage 1。** Freeze encoder + LLM；只在 ASR / captioning data 上训练 projector。
- **Stage 2。** 在 instruction-following audio tasks（QA、reasoning、music understanding）上 full / LoRA fine-tune。
- **Stage 3（可选）。** Voice-in / voice-out 加 speech decoder。Qwen2.5-Omni 和 AF3-Chat 做这个。

### 2026 model map

| Model | Backbone | Audio encoder | Output modality | Access |
|-------|----------|---------------|-----------------|--------|
| Qwen2.5-Omni-7B | Qwen2.5-7B | Custom + Whisper | text + speech | Apache-2.0 |
| Qwen3-Omni | Qwen3 | Custom | text + speech | Apache-2.0 |
| Audio Flamingo 3 | Qwen2 | AF-CLAP | text | NVIDIA non-commercial |
| Audio Flamingo Next | Qwen2 | AF-CLAP v2 | text | NVIDIA non-commercial |
| SALMONN | Vicuna | Whisper + BEATs | text | Apache-2.0 |
| LTU / LTU-AS | Llama | CAV-MAE | text | Apache-2.0 |
| GAMA | Llama | AST + Q-Former | text | Apache-2.0 |
| Gemini 2.5 Flash/Pro (closed) | Gemini | proprietary | text + speech | API |
| GPT-4o Audio (closed) | GPT-4o | proprietary | text + speech | API |

### Benchmark reality check（2026）

**MMAU-Pro。** 1800 QA pairs，覆盖 speech / sound / music / mixed。包含 multi-audio subset。

| Model | Overall | Speech | Sound | Music | Multi-audio |
|-------|---------|--------|-------|-------|-------------|
| Gemini 2.5 Pro | ~60% | 73.4% | 51.9% | 64.9% | ~22% |
| Gemini 2.5 Flash | ~57% | 73.4% | 50.5% | 64.9% | 21.2% |
| GPT-4o Audio | 52.5% | — | — | — | 26.5% |
| Qwen2.5-Omni-7B | 52.2% | 57.4% | 47.6% | 61.5% | ~20% |
| Audio Flamingo 3 | ~54% | — | — | — | — |
| Audio Flamingo Next | SOTA on LongAudioBench | — | — | — | — |

**Multi-audio column 对所有人都很难看。** 4-choice multiple choice 的 random chance = 25%；多数模型就在附近。LALMs 仍然很难比较两个 clips。

### 2026 年 LALMs 有用的地方

- **Call-center recordings 合规 audit。** “Agent 有没有提 required disclosure？”
- **Accessibility。** 给 deaf users 描述 sound events（不只是 transcription）。
- **Content moderation。** 检测 violent language + threatening tone + background context。
- **Podcast / meeting chaptering。** Semantic summary，而不只是 speaker turns。
- **Music catalog analysis。** “找出所有有 B-section key change 的 tracks。”

### 它们还不适合的地方

- 细粒度 music theory（低于 chord-level）。
- 长对话中的 speaker-attributed reasoning（超过 10 分钟会退化）。
- Multi-audio comparison（22-26% 几乎随机）。
- Real-time streaming reasoning（多数是 offline batch inference）。

## 构建它

### 第 1 步：query Qwen2.5-Omni

```python
from transformers import AutoModelForCausalLM, AutoProcessor

processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-Omni-7B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-Omni-7B", torch_dtype="auto")

audio, sr = load_wav("clip.wav", sr=16000)
messages = [{
    "role": "user",
    "content": [
        {"type": "audio", "audio": audio},
        {"type": "text", "text": "What sounds do you hear, and what's happening?"},
    ],
}]
inputs = processor.apply_chat_template(messages, tokenize=True, return_tensors="pt")
output = model.generate(**inputs, max_new_tokens=200)
print(processor.decode(output[0], skip_special_tokens=True))
```

### 第 2 步：projector pattern

```python
import torch.nn as nn

class AudioProjector(nn.Module):
    def __init__(self, audio_dim=1280, llm_dim=4096):
        super().__init__()
        self.down = nn.Linear(audio_dim, llm_dim)
        self.act = nn.GELU()
        self.up = nn.Linear(llm_dim, llm_dim)

    def forward(self, audio_features):
        return self.up(self.act(self.down(audio_features)))
```

就是这样。Projector 通常是 1-3 个 linear layers。用 ASR pairs（audio → transcript）训练它是 Stage-1 pretext task。

### 第 3 步：benchmarking MMAU / LongAudioBench

```python
from datasets import load_dataset
mmau = load_dataset("MMAU/MMAU-Pro")

correct = 0
for item in mmau["test"]:
    answer = call_model(item["audio"], item["question"], item["choices"])
    if answer == item["correct_choice"]:
        correct += 1
print(f"Accuracy: {correct / len(mmau['test']):.3f}")
```

分别报告 per-category（speech / sound / music / multi-audio）。Aggregate numbers 会隐藏模型失败的位置。

## 使用它

| 任务 | 2026 选择 |
|------|-----------|
| Free-form audio QA（open） | Qwen2.5-Omni-7B |
| Best open on long audio | Audio Flamingo Next |
| Best closed | Gemini 2.5 Pro |
| Voice-in / voice-out agent | Qwen2.5-Omni 或 GPT-4o Audio |
| Music reasoning | Audio Flamingo 3 或 2（music-specialized AF-CLAP） |
| Call-center audit | Gemini 2.5 Pro via API，结合你的 policy docs 做 RAG |

## 坑

- **过度相信 multi-audio。** 如果任务需要“which clip has X”，随机水平表现是真实的。
- **Long-audio degradation。** 超过 10 分钟，多数模型 speaker attribution 会坏。先 diarize（第 6 课），再 summarize。
- **Silence 上 hallucinations。** 使用 Whisper encoder 的 LALMs 继承同类问题。VAD-gate。
- **Benchmark cherry-picking。** Vendor blog posts 会突出 best-case categories。自己跑 MMAU-Pro multi-audio subset。

## 交付它

保存为 `outputs/skill-alm-picker.md`。为给定 audio-understanding task 选择 LALM + benchmark subset + output-modality（text vs speech）。

## 练习

1. **简单。** 运行 `code/main.py`，看 toy projector pattern + fake LALM 如何路由（audio-embedding, text-tokens）→ output tokens。
2. **中等。** 在 100 个 MMAU-Pro speech items 上评估 Qwen2.5-Omni-7B。和论文报告数字比较。
3. **困难。** 构建一个 minimal audio-captioning baseline：BEATs encoder + 2-layer projector + frozen Llama-3.2-1B。只在 AudioCaps 上 fine-tune projector。和 SALMONN 在 Clotho-AQA 上比较。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|-----------------|-----------------------|
| LALM | Audio ChatGPT | Audio encoder + projector + LLM decoder。 |
| Projector | Adapter | 把 audio features 映射到 LLM embedding space 的小 MLP。 |
| MMAU | benchmark | 10k audio-QA pairs，覆盖 speech、sound、music。 |
| MMAU-Pro | 更难的 MMAU | 1800 multi-audio / reasoning-heavy questions。 |
| LongAudioBench | Long-form eval | 带 semantic queries 的多分钟 clips。 |
| Voice-in / voice-out | Speech-native | 模型摄入 speech 并输出 speech，不走 text detour。 |

## 延伸阅读

- [Chu et al. (2024). Qwen2-Audio](https://arxiv.org/abs/2407.10759) — reference architecture。
- [Alibaba (2025). Qwen2.5-Omni](https://huggingface.co/Qwen/Qwen2.5-Omni-7B) — speech-in-speech-out。
- [NVIDIA (2025). Audio Flamingo 3](https://arxiv.org/abs/2507.08128) — open long-audio leader。
- [NVIDIA (2026). Audio Flamingo Next](https://arxiv.org/abs/2604.10905) — LongAudioBench SOTA。
- [Tang et al. (2023). SALMONN](https://arxiv.org/abs/2310.13289) — dual-encoder pioneer。
- [MMAU-Pro leaderboard](https://mmaubenchmark.github.io/) — 2026 live rankings。
