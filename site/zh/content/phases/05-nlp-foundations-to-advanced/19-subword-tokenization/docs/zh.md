# Subword Tokenization：BPE、WordPiece、Unigram、SentencePiece

> Word tokenizer 遇到没见过的词会卡住。Character tokenizer 会让序列长度爆炸。Subword tokenizer 取中间道路。每个现代 LLM 都靠它交付。

**类型：** 学习
**语言：** Python
**前置要求：** 阶段 5 · 01（文本处理），阶段 5 · 04（GloVe / FastText / Subword）
**时间：** ~60 分钟

## 问题

你的 vocabulary 有 50,000 个词。用户输入 `"untokenizable"`。Tokenizer 返回 `[UNK]`。模型现在对这个词没有任何信号。更糟的是：你 corpus 中第 90 百分位的文档有 40 个稀有词，这意味着每篇文档丢掉 40 bits 信息。

Subword tokenization 解决了这个问题。常见词保持单个 token。稀有词拆成有意义的片段：`untokenizable` → `un`, `token`, `izable`。训练数据能覆盖一切，因为任何字符串最终都是 bytes 的序列。

2026 年每个 frontier LLM 都使用三类算法之一（BPE、Unigram、WordPiece），并由三类库之一封装（tiktoken、SentencePiece、HF Tokenizers）。不选一个 tokenizer，就无法交付 language model。

## 概念

![BPE vs Unigram vs WordPiece, character-by-character](../assets/subword-tokenization.svg)

**BPE（Byte-Pair Encoding）。** 从 character-level vocabulary 开始。统计每个相邻 pair。把最高频 pair 合并成新 token。重复直到达到目标 vocabulary size。主流算法：GPT-2/3/4、Llama、Gemma、Qwen2、Mistral。

**Byte-level BPE。** 算法相同，但在原始 bytes（256 个 base tokens）上运行，而不是 Unicode characters。保证零 `[UNK]` tokens：任何 byte sequence 都能编码。GPT-2 使用 50,257 tokens（256 bytes + 50,000 merges + 1 special）。

**Unigram。** 从巨大 vocabulary 开始。给每个 token 一个 unigram probability。迭代删除那些移除后对 corpus log-likelihood 增加最小的 tokens。推理时是概率式的：可以 sample tokenizations（对 subword regularization 数据增强有用）。T5、mBART、ALBERT、XLNet、Gemma 使用。

**WordPiece。** 合并那些最大化训练 corpus likelihood 的 pairs，而不是原始频率。BERT、DistilBERT、ELECTRA 使用。

**SentencePiece vs tiktoken。** SentencePiece 是直接在原始 Unicode 文本上训练 vocabularies（BPE 或 Unigram）的库，用 `▁` 编码空白。tiktoken 是 OpenAI 针对预构建 vocabularies 的快速 encoder；它不负责训练。

经验法则：

- **训练新 vocabulary：** SentencePiece（multilingual，无需 pre-tokenization）或 HF Tokenizers。
- **针对 GPT vocab 做快速推理：** tiktoken（cl100k_base、o200k_base）。
- **两者都要：** HF Tokenizers，一个库同时覆盖训练和 serving。

## 构建它

### 第 1 步：从零实现 BPE

见 `code/main.py`。循环如下：

```python
def train_bpe(corpus, num_merges):
    vocab = {tuple(word) + ("</w>",): count for word, count in corpus.items()}
    merges = []
    for _ in range(num_merges):
        pairs = Counter()
        for symbols, freq in vocab.items():
            for a, b in zip(symbols, symbols[1:]):
                pairs[(a, b)] += freq
        if not pairs:
            break
        best = pairs.most_common(1)[0][0]
        merges.append(best)
        vocab = apply_merge(vocab, best)
    return merges
```

算法编码了三个事实。`</w>` 标记词尾，让 `"low"`（后缀）和 `"lower"`（前缀）保持区分。Frequency weighting 让高频 pairs 更早胜出。Merge list 是有序的：推理按训练顺序应用 merges。

### 第 2 步：用学到的 merges 编码

```python
def encode_bpe(word, merges):
    symbols = list(word) + ["</w>"]
    for a, b in merges:
        i = 0
        while i < len(symbols) - 1:
            if symbols[i] == a and symbols[i + 1] == b:
                symbols = symbols[:i] + [a + b] + symbols[i + 2:]
            else:
                i += 1
    return symbols
```

朴素复杂度是 O(n·|merges|)。生产实现（tiktoken、HF Tokenizers）使用 merge-rank lookup 和 priority queues，接近线性时间。

### 第 3 步：实践中的 SentencePiece

```python
import sentencepiece as spm

spm.SentencePieceTrainer.train(
    input="corpus.txt",
    model_prefix="my_tokenizer",
    vocab_size=8000,
    model_type="bpe",          # or "unigram"
    character_coverage=0.9995, # lower for CJK (e.g. 0.9995 for English, 0.995 for Japanese)
    normalization_rule_name="nmt_nfkc",
)

sp = spm.SentencePieceProcessor(model_file="my_tokenizer.model")
print(sp.encode("untokenizable", out_type=str))
# ['▁un', 'token', 'izable']
```

注意：不需要 pre-tokenization；空格编码为 `▁`；`character_coverage` 控制稀有字符是被保留还是映射到 `<unk>`。

### 第 4 步：用 tiktoken 处理 OpenAI-compatible vocabs

```python
import tiktoken
enc = tiktoken.get_encoding("o200k_base")
print(enc.encode("untokenizable"))        # [127340, 101028]
print(len(enc.encode("Hello, world!")))   # 4
```

仅编码。快（Rust backend）。在 byte-counting、cost estimation、context-window budgeting 上与 GPT-4/5 tokenization 精确一致。

## 2026 年仍然会交付的坑

- **Tokenizer drift。** 用 vocab A 训练，却用 vocab B 部署。Token IDs 不同，模型输出垃圾。在 CI 中检查 `tokenizer.json` hash。
- **Whitespace ambiguity。** BPE 中 `"hello"` 和 `" hello"` 会产生不同 tokens。始终显式指定 `add_special_tokens` 和 `add_prefix_space`。
- **Multilingual undertraining。** 英语占主导的 corpora 会产生把非拉丁脚本拆成 5-10 倍 tokens 的 vocabularies。同一 prompt 在 GPT-3.5 的日语/阿拉伯语上可能贵 5-10 倍。o200k_base 部分修复了这一点。
- **Emoji splits。** 一个 emoji 可能占 5 个 tokens。做 context budget 时要专门检查 emoji handling。

## 使用它

2026 年技术栈：

| 场景 | 选择 |
|-----------|------|
| 从零训练 monolingual model | HF Tokenizers（BPE） |
| 训练 multilingual model | SentencePiece（Unigram，`character_coverage=0.9995`） |
| 服务 OpenAI-compatible API | tiktoken（GPT-4+ 用 `o200k_base`） |
| Domain-specific vocab（code、math、protein） | 在 domain corpus 上训练 custom BPE，并与 base vocab 合并 |
| Edge inference，小模型 | Unigram（较小 vocabularies 表现更好） |

Vocabulary size 是 scaling 决策，不是常数。粗略启发：<1B 参数用 32k，1-10B 用 50-100k，multilingual/frontier 用 200k+。

## 交付它

保存为 `outputs/skill-bpe-vs-wordpiece.md`：

```markdown
---
name: tokenizer-picker
description: 为给定 corpus 和部署目标选择 tokenizer algorithm、vocab size、library。
version: 1.0.0
phase: 5
lesson: 19
tags: [nlp, tokenization]
---

给定 corpus（size、languages、domain）和部署目标（training from scratch / fine-tuning / API-compatible inference），输出：

1. Algorithm。BPE、Unigram 或 WordPiece。一句话说明理由。
2. Library。SentencePiece、HF Tokenizers 或 tiktoken。说明理由。
3. Vocab size。四舍五入到最接近的 1k。理由要关联 model size 和 language coverage。
4. Coverage settings。`character_coverage`、`byte_fallback`、special-token list。
5. Validation plan。Held-out set 的 average tokens-per-word、OOV rate、compression ratio、round-trip decode equality。

拒绝在包含 rare-script content 的 corpora 上训练 character-coverage <0.995 的 tokenizer。拒绝交付没有冻结 `tokenizer.json` hash CI 检查的 vocab。标记任何低于 16k vocab 的 monolingual tokenizer 可能 under-spec。
```

## 练习

1. **简单。** 在 `code/main.py` 的 tiny corpus 上训练 500-merge BPE。编码三个 held-out words。多少个正好产生 1 个 token，多少个产生 >1 token？
2. **中等。** 比较 100 个英语 Wikipedia 句子在 `cl100k_base`、`o200k_base` 和你用 vocab=32k 训练的 SentencePiece BPE 上的 token counts。报告每个的 compression ratio。
3. **困难。** 用 BPE、Unigram 和 WordPiece 训练同一个 corpus。测量小型 sentiment classifier 使用每个 tokenizer 后的 downstream accuracy。这个选择让 F1 变化超过 1 个点了吗？

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|-----------------|-----------------------|
| BPE | Byte-Pair Encoding | 贪心合并最高频 character pairs，直到达到目标 vocab size。 |
| Byte-level BPE | 永远没有 unknown tokens | 在原始 256 bytes 上做 BPE；GPT-2 / Llama 使用。 |
| Unigram | 概率式 tokenizer | 从大候选集合按 log-likelihood 剪枝；T5、Gemma 使用。 |
| SentencePiece | 处理空白的那个 | 在原始文本上训练 BPE/Unigram 的库；空格编码为 `▁`。 |
| tiktoken | 快的那个 | OpenAI 基于 Rust 的 BPE encoder，用于预构建 vocabs。不训练。 |
| Merge list | 魔法数字 | 有序 `(a, b) → ab` merges；推理按顺序应用。 |
| Character coverage | 多稀有算太稀有？ | Tokenizer 必须覆盖训练 corpus 中的字符比例；典型约 0.9995。 |

## 延伸阅读

- [Sennrich, Haddow, Birch (2015). Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909) — BPE 论文。
- [Kudo (2018). Subword Regularization with Unigram Language Model](https://arxiv.org/abs/1804.10959) — Unigram 论文。
- [Kudo, Richardson (2018). SentencePiece: A simple and language independent subword tokenizer](https://arxiv.org/abs/1808.06226) — 这个库。
- [Hugging Face — Summary of the tokenizers](https://huggingface.co/docs/transformers/tokenizer_summary) — 简明参考。
- [OpenAI tiktoken repo](https://github.com/openai/tiktoken) — cookbook + encoding list。
