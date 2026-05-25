# Whisper：架构与 Fine-Tuning

> Whisper 是 30 秒窗口的 transformer encoder-decoder，在 680k 小时 multilingual weakly-supervised audio-text pairs 上训练。一个架构，多种任务，覆盖 99 种语言且鲁棒。2026 年 reference ASR。

**类型：** 构建
**语言：** Python
**前置要求：** 阶段 6 · 04（ASR），阶段 5 · 10（Attention），阶段 7 · 05（Full Transformer）
**时间：** ~75 分钟

## 问题

OpenAI 在 2022 年 9 月发布 Whisper。它是第一个作为 commodity 交付的 ASR model：贴上音频，得到文本，99 种语言，抗噪，笔记本可跑。到 2024 年，OpenAI 发布了 Large-v3 和 Turbo variants；到 2026 年，Whisper 是从 podcast transcription 到 voice assistants 再到 YouTube subtitles 的默认 baseline。

但 Whisper 不是一个你可以永远当黑盒的 pipeline。Domain shift 会杀死它：技术术语、speaker accents、proper nouns、短 clips、silence。你需要知道：

1. 它内部到底是什么。
2. 如何正确给它 chunked、streaming 或 long-form audio。
3. 什么时候 fine-tune，以及怎么做。

## 概念

![Whisper encoder-decoder, tasks, chunked inference, fine-tune](../assets/whisper.svg)

**架构。** 标准 transformer encoder-decoder。

- Input：30 秒 log-mel spectrogram，80 mels，10 ms hop → 3000 frames。短 clips zero-padded，长 clips chunked。
- Encoder：conv-downsample（stride 2）+ `N` transformer blocks。Large-v3：32 layers，1280-dim，20 heads。
- Decoder：`N` transformer blocks，带 causal self-attn + 对 encoder output 的 cross-attn。大小与 encoder 相同。
- Output：51,865-token vocab 上的 BPE tokens。

Large-v3 有 1.55B params。Turbo 使用 4-layer decoder（从 32 层减到 4 层），延迟降 8 倍，WER 损失 <1%。

**Prompt format。** Whisper 是由 decoder prompt 中的 special tokens 驱动的 multitask model：

```
<|startoftranscript|><|en|><|transcribe|><|notimestamps|> Hello world.<|endoftext|>
```

- `<|en|>`：language tag；强制 translation-vs-transcription 行为。
- `<|transcribe|>` 或 `<|translate|>`：从任意语言输入翻译成英语输出，或逐字转写。
- `<|notimestamps|>`：跳过 word-level timestamps（更快）。

Prompt 让一个模型完成多种任务。把 `<|en|>` 改成 `<|fr|>`，它就会转写法语。

**30 秒窗口。** 一切都固定在 30 秒。更长 clips 需要 chunking；更短 clips 会 padding。Window 不是原生 streaming，这就是 WhisperX、Whisper-Streaming 和 faster-whisper 存在的原因。

**Log-mel normalization。** `(log_mel - mean) / std`，stats 来自 Whisper 自己的训练 corpus。你 *必须* 使用 Whisper preprocessing（`whisper.audio.log_mel_spectrogram`），不要用 `librosa.feature.melspectrogram`。

### 2026 variants

| Variant | Params | Latency (A100) | WER (LibriSpeech-clean) |
|---------|--------|----------------|------------------------|
| Tiny | 39M | 1× realtime | 5.4% |
| Base | 74M | 1× | 4.1% |
| Small | 244M | 1× | 3.0% |
| Medium | 769M | 1× | 2.7% |
| Large-v3 | 1.55B | 2× | 1.8% |
| Large-v3-turbo | 809M | 8× | 1.58% |
| Whisper-Streaming (2024) | 1.55B | streaming | 2.0% |

### Fine-tuning

2026 年 canonical workflow：

1. 收集 10-100 小时目标领域 audio + aligned transcripts。
2. 用 `transformers.Seq2SeqTrainer`，带 `generate_with_loss` callback。
3. Parameter-efficient：在 attention layers 的 `q_proj`、`k_proj`、`v_proj` 上做 LoRA，GPU memory 降 4 倍，WER 代价 <0.3。
4. 如果少于 10 小时，freeze encoder。只调 decoder。
5. 使用 Whisper 自己的 tokenizer 和 prompt format；永远不要换 tokenizer。

社区结果：在 20 小时 medical dictation 上 fine-tune Medium，medical vocabulary 的 WER 从 12% 降到 4.5%。在 4 小时 Icelandic 上 fine-tune Turbo，WER 从 18% 降到 6%。

## 构建它

### 第 1 步：开箱运行 Whisper

```python
import whisper
model = whisper.load_model("large-v3-turbo")
result = model.transcribe(
    "clip.wav",
    language="en",
    task="transcribe",
    temperature=0.0,
    condition_on_previous_text=False,  # prevents runaway repetition
)
print(result["text"])
for seg in result["segments"]:
    print(f"[{seg['start']:.2f}–{seg['end']:.2f}] {seg['text']}")
```

你应该始终 override 的关键 defaults：`temperature=0.0`（sampling 默认有 0.0 → 0.2 → 0.4 … fallback chain）、`condition_on_previous_text=False`（防止 cascading hallucination）、`no_speech_threshold=0.6`（silence detection）。

### 第 2 步：chunked long-form

```python
# whisperx is the 2026 reference for long-form with word-level timestamps
import whisperx
model = whisperx.load_model("large-v3-turbo", device="cuda", compute_type="float16")
segments = model.transcribe("1hour.mp3", batch_size=16, chunk_size=30)
```

WhisperX 增加：（1）Silero VAD gating，（2）通过 wav2vec 2.0 做 word-level alignment，（3）通过 `pyannote.audio` 做 diarization。它是 2026 年 production transcription 的主力。

### 第 3 步：用 LoRA fine-tune

```python
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from peft import LoraConfig, get_peft_model

model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3-turbo")
lora = LoraConfig(
    r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1, bias="none", task_type="SEQ_2_SEQ_LM",
)
model = get_peft_model(model, lora)
# model.print_trainable_parameters()  -> ~3M trainable / 809M total
```

然后是标准 Trainer loop。每 1000 steps checkpoint。用 held-out WER 评估。

### 第 4 步：检查每层学到了什么

```python
# Grab cross-attention weights during decode to see what the decoder attends to.
with torch.inference_mode():
    out = model.generate(
        input_features=features,
        return_dict_in_generate=True,
        output_attentions=True,
    )
# out.cross_attentions: layer × head × step × src_len
```

用 heatmap 可视化。你会看到 decoder steps 扫过 encoder frames 的 diagonal alignment。这条对角线就是 Whisper 对 word timestamps 的理解。

## 使用它

2026 年技术栈：

| 场景 | 选择 |
|-----------|------|
| General English、offline | Large-v3-turbo via `whisperx` |
| Mobile / edge | Quantized Whisper-Tiny（int8）或 Moonshine |
| Multilingual long-form | Large-v3 via `whisperx` + diarization |
| Low-resource language | 用 LoRA fine-tune Medium 或 Turbo |
| Streaming（2 s latency） | Whisper-Streaming 或 Parakeet-TDT |
| Word-level timestamps | WhisperX（通过 wav2vec 2.0 forced alignment） |

`faster-whisper`（CTranslate2 backend）是 2026 年最快 CPU+GPU inference runtime，比 vanilla 快 4 倍，输出相同。

## 2026 年仍会交付的坑

- **Silence 上 hallucinated text。** Whisper 在 captions 上训练，包含 “Thanks for watching!”、“Subscribe!”、歌词。调用前始终 VAD-gate。
- **`condition_on_previous_text` cascade。** 一个 hallucination 会污染后续 windows。除非需要跨 chunks fluency，否则设为 `False`。
- **Short-clip padding。** 2 秒 clip padding 到 30 秒，会在 trailing silence hallucinate。用 `pad=False` 或 VAD-gate。
- **Wrong mel stats。** 使用 librosa mels 而不是 Whisper mels 会产生近似随机输出。用 `whisper.audio.log_mel_spectrogram`。

## 交付它

保存为 `outputs/skill-whisper-tuner.md`。为给定领域设计 Whisper fine-tune 或 inference pipeline。

## 练习

1. **简单。** 运行 `code/main.py`。它 tokenizes 一个 Whisper-style prompt，计算 decode shape budgets，并打印 10 分钟 clip 的 chunk schedule。
2. **中等。** 安装 `faster-whisper`，转写 10 分钟 podcast，和 human transcript 比较 WER。尝试 `language="auto"` vs forced `language="en"`。
3. **困难。** 用 HF `datasets`，选择一种 Whisper 表现差的语言（例如 Urdu），在 2 小时数据上 fine-tune Medium with LoRA 2 epochs，并报告 WER delta。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|-----------------|-----------------------|
| 30-sec window | Whisper 的限制 | 硬输入上限；长音频要 chunk。 |
| SOT | Start-of-transcript | `<|startoftranscript|>` 启动 decoder prompt。 |
| Timestamps token | Temporal alignment | 51k vocab 中每 0.02 s offset 都是一个 special token。 |
| Turbo | 快速 variant | 4-decoder layers，8× faster，<1% WER regression。 |
| WhisperX | Long-form wrapper | VAD + Whisper + wav2vec alignment + diarization。 |
| LoRA fine-tune | 高效调优 | 给 attention 加 low-rank adapters；训练约 0.3% params。 |
| Hallucination | 静默失败 | Whisper 从 noise/silence 中生成流畅英语。 |

## 延伸阅读

- [Radford et al. (2022). Whisper paper](https://arxiv.org/abs/2212.04356) — 原始架构和训练 recipe。
- [OpenAI (2024). Whisper Large-v3-turbo release](https://github.com/openai/whisper/discussions/2363) — 4-layer decoder，8× speedup。
- [Bain et al. (2023). WhisperX](https://arxiv.org/abs/2303.00747) — long-form、word-aligned、diarized。
- [Systran — faster-whisper repo](https://github.com/SYSTRAN/faster-whisper) — CTranslate2-backed，4× faster。
- [HuggingFace — Whisper fine-tune tutorial](https://huggingface.co/blog/fine-tune-whisper) — canonical LoRA / full-FT walkthrough。
