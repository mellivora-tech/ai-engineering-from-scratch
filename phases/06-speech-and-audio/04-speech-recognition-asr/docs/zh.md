# 语音识别（ASR）：CTC、RNN-T、Attention

> Speech recognition 是每个 timestep 的 audio classification，再由懂英语和沉默的 sequence model 粘起来。CTC、RNN-T 和 attention 是三种做法。选一个，并理解原因。

**类型：** 构建
**语言：** Python
**前置要求：** 阶段 6 · 02（Spectrograms & Mel），阶段 5 · 08（CNNs & RNNs for Text），阶段 5 · 10（Attention）
**时间：** ~45 分钟

## 问题

你有一个 10 秒、16 kHz clip。你想得到字符串：“turn on the kitchen lights”。挑战是结构性的：audio frames 不会和 characters 一一对齐。单词 “okay” 可能占 200 ms，也可能占 1200 ms。Silence 会给 utterance 加标点。一些 phonemes 比另一些更长。输出 tokens 数量事先未知。

三种 formulation 解决这个问题：

1. **CTC（Connectionist Temporal Classification）。** 每帧发出 token probabilities，包括特殊 *blank*。Decode 时 collapse repeats 和 blanks。Non-autoregressive、快。wav2vec 2.0、MMS 使用。
2. **RNN-T（Recurrent Neural Network Transducer）。** Joint network 基于 encoder frame 和 previous tokens 预测 next token。Streamable。Google on-device ASR、NVIDIA Parakeet 使用。
3. **Attention encoder-decoder。** Encoder 把 audio 压缩成 hidden states，decoder cross-attends 并 autoregressively 生成 tokens。Whisper、SeamlessM4T 使用。

2026 年 LibriSpeech test-clean 上 SOTA WER 是 1.4%（Parakeet-TDT-1.1B，NVIDIA）和 1.58%（Whisper-Large-v3-turbo）。质量差异很小，部署差异很大。

## 概念

![Three ASR formulations: CTC, RNN-T, attention-encoder-decoder](../assets/asr-formulations.svg)

**CTC 直觉。** 让 encoder 输出 `T` 个 frame-level distributions，覆盖 `V+1` tokens（V chars + blank）。对目标字符串 `y`，长度 `U < T`，任何 collapse 后得到 `y` 的 frame alignment 都计入。CTC loss 对所有这类 alignments 求和。推理：每帧 argmax、collapse repeats、remove blanks。

优点：non-autoregressive、streamable、zero lookahead。缺点：*conditional independence assumption*，每帧预测彼此独立，所以没有内部 language model。用 external LM 做 beam search 或 shallow fusion 修复。

**RNN-T 直觉。** 增加一个嵌入 token history 的 *predictor* network，以及一个把 predictor state 和 encoder frame 合并成 `V+1` joint distribution 的 *joiner*（`+1` 是 null / no-emit）。显式建模 CTC 忽略的 conditional dependence。因为每一步只依赖 past frames 和 past tokens，所以 streamable。

优点：streamable + internal LM。缺点：训练更复杂且吃内存（3D loss lattice）；RNN-T loss kernels 本身就是一个库类别。

**Attention encoder-decoder。** Encoder（6-32 transformer layers）处理 log-mel frames。Decoder（6-32 transformer layers）cross-attends encoder outputs，并 autoregressively 生成 tokens。没有 alignment constraint；attention 可以看 audio 任意位置。除非限制 attention（chunked Whisper-Streaming, 2024），否则不可 streaming。

优点：offline ASR 质量最高，能用标准 seq2seq tooling 训练。缺点：autoregressive latency 与 output length 成正比；没有工程改造无法 stream。

### WER：唯一数字

**Word Error Rate** = `(S + D + I) / N`，其中 S=substitutions、D=deletions、I=insertions、N=reference word count。等价于 word level 的 Levenshtein edit distance。越低越好。WER 超过 20% 通常不可用；低于 5% 对 read speech 是 human-parity。标准 benchmark 上 2026 数字：

| Model | LibriSpeech test-clean | LibriSpeech test-other | Size |
|-------|------------------------|------------------------|------|
| Parakeet-TDT-1.1B | 1.40% | 2.78% | 1.1B params |
| Whisper-Large-v3-turbo | 1.58% | 3.03% | 809M |
| Canary-1B Flash | 1.48% | 2.87% | 1B |
| Seamless M4T v2 | 1.7% | 3.5% | 2.3B |

这些都是 encoder-decoder 或 RNN-T。纯 CTC systems（wav2vec 2.0）在 test-clean 上约 1.8-2.1%。

## 构建它

### 第 1 步：greedy CTC decode

```python
def ctc_greedy(frame_logits, blank=0, vocab=None):
    # frame_logits: list of per-frame probability vectors
    preds = [max(range(len(p)), key=lambda i: p[i]) for p in frame_logits]
    out = []
    prev = -1
    for p in preds:
        if p != prev and p != blank:
            out.append(p)
        prev = p
    return "".join(vocab[i] for i in out) if vocab else out
```

两条规则：collapse consecutive repeats，drop blanks。例如：`a a _ _ a b b _ c` → `a a b c`。

### 第 2 步：beam-search CTC

```python
def ctc_beam(frame_logits, beam=8, blank=0):
    import math
    beams = [([], 0.0)]  # (tokens, log_prob)
    for p in frame_logits:
        log_p = [math.log(max(pi, 1e-10)) for pi in p]
        candidates = []
        for seq, lp in beams:
            for t, lpt in enumerate(log_p):
                new = seq[:] if t == blank else (seq + [t] if not seq or seq[-1] != t else seq)
                candidates.append((new, lp + lpt))
        candidates.sort(key=lambda x: -x[1])
        beams = candidates[:beam]
    return beams[0][0]
```

生产用 prefix tree beam search + LM fusion；这是概念骨架。

### 第 3 步：WER

```python
def wer(ref, hyp):
    r, h = ref.split(), hyp.split()
    dp = [[0] * (len(h) + 1) for _ in range(len(r) + 1)]
    for i in range(len(r) + 1):
        dp[i][0] = i
    for j in range(len(h) + 1):
        dp[0][j] = j
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            cost = 0 if r[i - 1] == h[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )
    return dp[len(r)][len(h)] / max(1, len(r))
```

### 第 4 步：用 Whisper 推理

```python
import whisper
model = whisper.load_model("large-v3-turbo")
result = model.transcribe("clip.wav")
print(result["text"])
```

2026 年最强通用 ASR 的一行调用。在 24 GB GPU 上以 ~20× realtime 运行。

### 第 5 步：用 Parakeet 或 wav2vec 2.0 streaming

```python
from transformers import pipeline
asr = pipeline("automatic-speech-recognition", model="nvidia/parakeet-tdt-1.1b")
for chunk in streaming_audio():
    print(asr(chunk, return_timestamps=True))
```

Streaming ASR 需要 chunked encoder attention 和 carryover state；使用支持它的库（Parakeet 用 NeMo，`transformers` pipeline 用 `chunk_length_s`）。

## 使用它

2026 年技术栈：

| 场景 | 选择 |
|-----------|------|
| English、offline、max quality | Whisper-large-v3-turbo |
| Multilingual、robust | SeamlessM4T v2 |
| Streaming、low latency | Parakeet-TDT-1.1B 或 Riva |
| Edge、mobile、<500 ms latency | Quantized Whisper-Tiny 或 Moonshine（2024） |
| Long-form | Whisper + VAD-based chunking（WhisperX） |
| Domain-specific（medical、legal） | Fine-tune wav2vec 2.0 + domain LM fusion |

## 2026 年仍会交付的坑

- **No VAD。** 在 silence 上运行 Whisper 会 hallucinate（“Thanks for watching!”）。始终用 VAD gate。
- **Character vs word vs subword WER。** 报告 word-level WER，且在 normalization（lowercase、去标点）之后。
- **Language ID drift。** Whisper 的 auto LID 会把噪声 clips 误路由到日语或威尔士语；已知语言时强制 `language="en"`。
- **Long clips without chunking。** Whisper 有 30 秒 window。超过的内容用 `chunk_length_s=30, stride=5`。

## 交付它

保存为 `outputs/skill-asr-picker.md`。为给定部署目标选择 model、decoding strategy、chunking 和 LM fusion。

## 练习

1. **简单。** 运行 `code/main.py`。它 greedy decode 一个手写 CTC output，并对 reference 计算 WER。
2. **中等。** 正确实现第 2 步的 prefix-tree beam search（处理 blank merge rule）。在 10-example synthetic dataset 上和 greedy 比较。
3. **困难。** 在 [LibriSpeech test-clean](https://www.openslr.org/12) 上使用 `whisper-large-v3-turbo`。计算前 100 个 utterances 的 WER。和发布数字比较。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|-----------------|-----------------------|
| CTC | blank-token loss | 对所有 frame-to-token alignments 求边缘化；non-AR。 |
| RNN-T | streaming loss | CTC + next-token predictor；处理 word-order。 |
| Attention enc-dec | Whisper-style | Encoder + cross-attending decoder；最佳 offline quality。 |
| WER | 你报告的数字 | Word level 的 `(S+D+I)/N`。 |
| Blank | 空白 | CTC 中表示“这一帧无 emission”的特殊 token。 |
| LM fusion | External language model | Beam search 时加入加权 LM log-probs。 |
| VAD | Silence gate | Voice activity detector；修剪 non-speech。 |

## 延伸阅读

- [Graves et al. (2006). Connectionist Temporal Classification](https://www.cs.toronto.edu/~graves/icml_2006.pdf) — CTC 论文。
- [Graves (2012). Sequence Transduction with RNNs](https://arxiv.org/abs/1211.3711) — RNN-T 论文。
- [Radford et al. / OpenAI (2022). Whisper: Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356) — 2022 经典论文；v3-turbo extension 在 2024。
- [NVIDIA NeMo — Parakeet-TDT card](https://huggingface.co/nvidia/parakeet-tdt-1.1b) — 2026 Open ASR Leaderboard leader。
- [Hugging Face — Open ASR Leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard) — 25+ models 的 live benchmark。
