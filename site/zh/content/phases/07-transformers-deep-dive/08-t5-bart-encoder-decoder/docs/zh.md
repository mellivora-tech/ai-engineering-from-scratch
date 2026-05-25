# T5、BART — Encoder-Decoder Models

> Encoders 负责理解。Decoders 负责生成。把它们重新放在一起，你得到的就是为 input → output 任务而生的模型：翻译、摘要、改写、转录。

**类型：** 学习
**语言：** Python
**前置要求：** 阶段 7 · 05（Full Transformer），阶段 7 · 06（BERT），阶段 7 · 07（GPT）
**时间：** ~45 分钟

## 问题

Decoder-only GPT 和 encoder-only BERT 分别为了不同目标精简了 2017 年架构。但许多任务天生就是 input-output：

- Translation: English → French.
- Summarization: 5,000-token article → 200-token summary.
- Speech recognition: audio tokens → text tokens.
- Structured extraction: prose → JSON.

对于这些任务，encoder-decoder 是最干净的匹配。Encoder 产生 source 的 dense representation。Decoder 生成输出，并在每一步 cross-attend 到该 representation。训练是在输出侧做 shift-by-one。和 GPT 是同一个 loss，只是额外 conditioned on encoder output。

两篇论文定义了现代玩法：

1. **T5**（Raffel et al. 2019）。“Text-to-Text Transfer Transformer。” 每个 NLP 任务都改写为 text-in、text-out。单一架构、单一 vocabulary、单一 loss。用 masked span prediction 预训练（破坏输入中的 spans，在输出中 decode 它们）。
2. **BART**（Lewis et al. 2019）。“Bidirectional and Auto-Regressive Transformer。” Denoising autoencoder：用多种方式破坏输入（shuffle、mask、delete、rotate），让 decoder 重建原文。

2026 年，encoder-decoder 格式仍然活在输入结构重要的地方：

- Whisper（speech → text）。
- Google 的翻译栈。
- 一些具有清晰 context-and-edit 结构的代码补全 / 修复模型。
- 用于结构化推理任务的 Flan-T5 和变体。

Decoder-only 赢得了聚光灯，但 encoder-decoder 从未消失。

## 概念

![Encoder-decoder with cross-attention](../assets/encoder-decoder.svg)

### Forward loop

```
source tokens ─▶ encoder ─▶ (N_src, d_model)  ──┐
                                                 │
target tokens ─▶ decoder block                   │
                 ├─▶ masked self-attention       │
                 ├─▶ cross-attention ◀───────────┘
                 └─▶ FFN
                ↓
              next-token logits
```

关键是，encoder 对每个输入只运行一次。Decoder 会 autoregressively 运行，但每一步都 cross-attend 到 *同一个* encoder output。缓存 encoder output 对长输入来说是免费的加速。

### T5 预训练 — span corruption

随机选择输入中的 spans（平均长度 3 tokens，总计 15%）。把每个 span 替换成一个唯一 sentinel：`<extra_id_0>`、`<extra_id_1>` 等。Decoder 只输出被破坏的 spans，并带上对应 sentinel prefix：

```
source: The quick <extra_id_0> fox jumps <extra_id_1> dog
target: <extra_id_0> brown <extra_id_1> over the lazy
```

这个信号比预测整个序列更便宜。在 T5 论文的 ablation 中，它和 MLM（BERT）以及 prefix-LM（UniLM）都有竞争力。

### BART 预训练 — multi-noise denoising

BART 尝试五种 noising functions：

1. Token masking.
2. Token deletion.
3. Text infilling（mask 一个 span，decoder 插入正确长度）。
4. Sentence permutation.
5. Document rotation.

Text infilling + sentence permutation 的组合产生了最好的下游结果。Decoder 总是重建原文。BART 的输出是完整序列，而不只是被破坏的 spans — 所以预训练 compute 比 T5 更高。

### 推理

和 GPT 一样是 autoregressive generation。Greedy / beam / top-p sampling 都适用。Beam search（width 4–5）是翻译和摘要的标准选择，因为输出分布比聊天更窄。

### 2026 年什么时候选哪种变体

| 任务 | Encoder-decoder? | 原因 |
|------|------------------|-----|
| Translation | Yes, usually | 清晰 source sequence；固定输出分布；beam search 有效 |
| Speech-to-text | Yes（Whisper） | 输入模态不同于输出；encoder 负责塑造 audio features |
| Chat / reasoning | No, decoder-only | 没有持久“input” — 对话本身就是序列 |
| Code completion | 通常 no | 带长 context 的 decoder-only 胜出；Qwen 2.5 Coder 这类代码模型是 decoder-only |
| Summarization | 两者都可 | BART、PEGASUS 胜过早期 decoder-only baselines；现代 decoder-only LLMs 可匹配 |
| Structured extraction | 两者都可 | T5 很干净，因为“text → text”能吸收任何输出格式 |

约 2022 年以来的趋势：decoder-only 接管了 encoder-decoder 曾经拥有的任务，因为（a）instruction-tuned decoder-only LLMs 可以通过 prompting 泛化到几乎任何事，（b）单一架构比两个 stack 更容易扩展，（c）RLHF 默认假设 decoder。Encoder-decoder 则保住输入模态不同（speech、images）或 beam search 质量重要的场景。

## 构建它

见 `code/main.py`。我们为 toy corpus 实现 T5-style span corruption — 这是本课最有用的单个部件，因为它出现在此后每个 encoder-decoder 预训练配方里。

### 第 1 步：span corruption

```python
def corrupt_spans(tokens, mask_rate=0.15, mean_span=3.0, rng=None):
    """Pick spans summing to ~mask_rate of tokens. Return (corrupted_input, target)."""
    n = len(tokens)
    n_mask = max(1, int(n * mask_rate))
    n_spans = max(1, int(round(n_mask / mean_span)))
    ...
```

Target format 是 T5 约定：`<sent0> span0 <sent1> span1 ...`。Corrupted input 会把未改变 tokens 和 span 位置上的 sentinel tokens 交错起来。

### 第 2 步：验证 round-trip

给定 corrupted input 和 target，重建原句。如果你的 corruption 可逆，forward pass 就定义良好。这是 sanity check — 真实训练从不这么做，但测试很便宜，并能抓出 span bookkeeping 中的 off-by-one bugs。

### 第 3 步：BART noising

五个函数：`token_mask`、`token_delete`、`text_infill`、`sentence_permute`、`document_rotate`。组合其中两个并展示结果。

## 使用它

HuggingFace 参考：

```python
from transformers import T5ForConditionalGeneration, T5Tokenizer
tok = T5Tokenizer.from_pretrained("google/flan-t5-base")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")

inputs = tok("translate English to French: Attention is all you need.", return_tensors="pt")
out = model.generate(**inputs, max_new_tokens=32)
print(tok.decode(out[0], skip_special_tokens=True))
```

T5 的技巧：任务名进入输入文本。同一个模型能处理几十种任务，因为每个任务都是 text-in、text-out。2026 年，这个模式已经被 instruction-tuned decoder-only models 泛化，但 T5 是先把它规范化的模型。

## 交付它

见 `outputs/skill-seq2seq-picker.md`。这个 skill 会根据 input-output 结构、latency 和 quality targets，为新任务选择 encoder-decoder 或 decoder-only。

## 练习

1. **简单。** 运行 `code/main.py`，对一个 30-token 句子应用 span corruption，验证把 non-sentinel source tokens 和 decoded target spans 拼起来可以重建原文。
2. **中等。** 实现 BART 的 `text_infill` noise：用单个 `<mask>` token 替换随机 spans，decoder 必须推断正确 span 长度和内容。展示一个例子。
3. **困难。** 在 tiny English → pig-Latin corpus（200 pairs）上 fine-tune `flan-t5-small`。在留出的 50-pair set 上测量 BLEU。和用相同数据、相同 compute fine-tune `Llama-3.2-1B` 比较。

## 关键词

| 术语 | 大家怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| Encoder-decoder | “Seq2seq transformer” | 两个 stacks：输入用 bidirectional encoder，输出用带 cross-attention 的 causal decoder。 |
| Cross-attention | “source 与 target 交流的地方” | Decoder 的 Q × encoder 的 K/V。Encoder 信息进入 decoder 的唯一位置。 |
| Span corruption | “T5 的预训练技巧” | 用 sentinel tokens 替换随机 spans；decoder 输出这些 spans。 |
| Denoising objective | “BART 的游戏” | 对输入应用 noise function，训练 decoder 重建干净序列。 |
| Sentinel token | “`<extra_id_N>` 占位符” | 在 source 中标记 corrupted spans，并在 target 中重新标记的特殊 tokens。 |
| Flan | “Instruction-tuned T5” | 在 >1,800 个任务上 fine-tuned 的 T5；让 encoder-decoder 在 instruction-following 上有竞争力。 |
| Beam search | “Decoding strategy” | 每一步保留 top-k partial sequences；翻译/摘要的标准做法。 |
| Teacher forcing | “训练时输入” | 训练时给 decoder 喂真实前一个输出 token，而不是采样出来的 token。 |

## 延伸阅读

- [Raffel et al. (2019). Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/abs/1910.10683) — T5。
- [Lewis et al. (2019). BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension](https://arxiv.org/abs/1910.13461) — BART。
- [Chung et al. (2022). Scaling Instruction-Finetuned Language Models](https://arxiv.org/abs/2210.11416) — Flan-T5。
- [Radford et al. (2022). Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356) — Whisper，2026 年标准 encoder-decoder。
- [HuggingFace `modeling_t5.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/t5/modeling_t5.py) — 参考实现。
