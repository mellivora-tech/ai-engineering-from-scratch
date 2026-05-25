# BERT — Masked Language Modeling

> GPT 预测下一个词。BERT 预测缺失的词。一句话的差异 — 带来了半个十年的 embedding 形态的一切。

**类型：** 构建
**语言：** Python
**前置要求：** 阶段 7 · 05（Full Transformer），阶段 5 · 02（Text Representation）
**时间：** ~45 分钟

## 问题

2018 年，每个 NLP 任务 — 情感分析、NER、QA、entailment — 都在自己的标注数据上从零训练自己的模型。没有一个预训练好的“理解英语”checkpoint 可以拿来 fine-tune。ELMo（2018）展示了你可以用双向 LSTM 预训练 contextual embeddings；它有帮助，但泛化有限。

BERT（Devlin et al. 2018）提出：如果我们拿一个 transformer encoder，用互联网上的每个句子训练它，并强迫它从两侧 context 中预测缺失词，会怎样？然后你只需要在下游任务上 fine-tune 一个 head。参数效率令人震惊。

结果是：18 个月内，BERT 及其变体（RoBERTa、ALBERT、ELECTRA）统治了当时存在的所有 NLP leaderboard。到 2020 年，地球上的每个搜索引擎、内容审核管线和 semantic-search 系统里都有一个 BERT。

2026 年，encoder-only 模型仍然是分类、检索和结构化抽取的正确工具 — 它们每 token 运行速度比 decoder 快 5–10×，并且它们的 embeddings 是每个现代 retrieval stack 的骨干。ModernBERT（2024 年 12 月）用 Flash Attention + RoPE + GeGLU 把架构推进到 8K context。

## 概念

![Masked language modeling: pick tokens, mask them, predict originals](../assets/bert-mlm.svg)

### 训练信号

取一个句子：`the quick brown fox jumps over the lazy dog`。

随机 mask 15% 的 tokens：

```
input:  the [MASK] brown fox jumps [MASK] the lazy dog
target: the  quick brown fox jumps  over  the lazy dog
```

训练模型预测 masked positions 上的原始 tokens。因为 encoder 是双向的，预测位置 1 的 `[MASK]` 可以使用位置 2+ 的 `brown fox jumps`。这是 GPT 做不到的事情。

### BERT 的 mask 规则

在被选中用于预测的 15% tokens 中：

- 80% 被替换成 `[MASK]`。
- 10% 被替换成随机 token。
- 10% 保持不变。

为什么不总是 `[MASK]`？因为 `[MASK]` 在推理时从不出现。如果训练模型期望 100% masked positions 都是 `[MASK]`，就会在 pretraining 和 fine-tuning 之间产生 distribution shift。10% random + 10% unchanged 会让模型保持诚实。

### Next Sentence Prediction（NSP）— 以及为什么它被放弃

原始 BERT 还训练 NSP：给定句子 A 和 B，预测 B 是否跟在 A 后面。RoBERTa（2019）消融后显示 NSP 有害无益。现代 encoder 会跳过它。

### 2026 年变了什么：ModernBERT

2024 年 ModernBERT 论文用 2026 年原语重建了 block：

| 组件 | Original BERT (2018) | ModernBERT (2024) |
|-----------|----------------------|-------------------|
| Positional | Learned absolute | RoPE |
| Activation | GELU | GeGLU |
| Normalization | LayerNorm | Pre-norm RMSNorm |
| Attention | Full dense | Alternating local (128) + global |
| Context length | 512 | 8192 |
| Tokenizer | WordPiece | BPE |

并且不同于 2018 stack，它原生支持 Flash Attention。在序列长度 8K 时，推理比 DeBERTa-v3 快 2–3×，GLUE 分数还更好。

### 2026 年仍然选择 encoder 的用例

| 任务 | 为什么 encoder 胜过 decoder |
|------|---------------------------|
| Retrieval / semantic search embeddings | 双向 context = 每 token 更好的 embedding 质量 |
| Classification（sentiment、intent、toxicity） | 一次 forward pass；没有 generation 开销 |
| NER / token labeling | 每位置输出，天然双向 |
| Zero-shot entailment（NLI） | encoder 顶上的 classifier head |
| RAG reranker | Cross-encoder scoring，比 LLM rerankers 快 10x |

## 构建它

### 第 1 步：masking 逻辑

见 `code/main.py`。函数 `create_mlm_batch` 接收一组 token IDs、vocab size 和 mask probability。返回 input IDs（已应用 masks）和 labels（只有 masked positions 有标签，其他位置是 -100 — PyTorch 的 ignore index 约定）。

```python
def create_mlm_batch(tokens, vocab_size, mask_prob=0.15, rng=None):
    input_ids = list(tokens)
    labels = [-100] * len(tokens)
    for i, t in enumerate(tokens):
        if rng.random() < mask_prob:
            labels[i] = t
            r = rng.random()
            if r < 0.8:
                input_ids[i] = MASK_ID
            elif r < 0.9:
                input_ids[i] = rng.randrange(vocab_size)
            # else: keep original
    return input_ids, labels
```

### 第 2 步：在小语料上运行 MLM 预测

在 20 个词的 vocabulary、200 个句子上训练 2 层 encoder + MLM head。没有 gradient — 我们做 forward-pass sanity checks。完整训练需要 PyTorch。

### 第 3 步：比较 mask 类型

展示三路规则如何让模型在没有 `[MASK]` 时也可用。在 unmasked 句子和 masked 句子上预测。两者都应该产生合理的 token 分布，因为模型在训练中见过两种模式。

### 第 4 步：fine-tune head

在玩具 sentiment dataset 上，用 classification head 替换 MLM head。只训练 head；encoder 冻结。这是每个 BERT 应用遵循的模式。

## 使用它

```python
from transformers import AutoModel, AutoTokenizer

tok = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
model = AutoModel.from_pretrained("answerdotai/ModernBERT-base")

text = "Attention is all you need."
inputs = tok(text, return_tensors="pt")
out = model(**inputs).last_hidden_state   # (1, N, 768)
```

**Embedding models 是 fine-tuned BERT。** `sentence-transformers` 里的 `all-MiniLM-L6-v2` 这类模型，就是用 contrastive loss 训练的 BERT。Encoder 相同。Loss 变了。

**Cross-encoder rerankers 也是 fine-tuned BERT。** 在 `[CLS] query [SEP] doc [SEP]` 上做 pair-classification。Query 和 doc 之间的双向 attention，正是 cross-encoder 相比 biencoder 的质量优势来源。

**2026 年什么时候不要选 BERT。** 任何生成式任务。Encoder 没有合理方式 autoregressively 产生 tokens。另外：1B 参数以下，某些小 decoder 能以更高灵活性匹配质量（Phi-3-Mini、Qwen2-1.5B）。

## 交付它

见 `outputs/skill-bert-finetuner.md`。这个 skill 会为新的分类或抽取任务界定 BERT fine-tune（backbone 选择、head spec、数据、评估、停止条件）。

## 练习

1. **简单。** 运行 `code/main.py`，打印 10,000 个 tokens 的 mask 分布。确认约 15% 被选中，其中约 80% 变成 `[MASK]`。
2. **中等。** 实现 whole-word masking：如果一个词被 tokenized 成 subwords，要么一起 mask 所有 subwords，要么都不 mask。在 500 句语料上测量它是否提高 MLM accuracy。
3. **困难。** 在公开数据集的 10,000 个句子上训练一个 tiny（2 层，d=64）BERT。用 `[CLS]` token fine-tune SST-2 情感分析。和同参数量的 decoder-only baseline 比较 — 谁赢？

## 关键词

| 术语 | 大家怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| MLM | “Masked language modeling” | 训练信号：随机把 15% tokens 替换成 `[MASK]`，预测原始 tokens。 |
| Bidirectional | “两边都看” | Encoder attention 没有 causal mask — 每个位置都能看到每个其他位置。 |
| `[CLS]` | “Pooler token” | 加在每个序列开头的特殊 token；它的最终 embedding 用作句子级表示。 |
| `[SEP]` | “Segment separator” | 分隔成对序列（例如 query/doc、sentence A/B）。 |
| NSP | “Next sentence prediction” | BERT 的第二个预训练任务；RoBERTa 显示它无用，2019 年后被放弃。 |
| Fine-tuning | “适配任务” | 大体冻结 encoder；在其上训练小 head 做下游任务。 |
| Cross-encoder | “Reranker” | 同时输入 query 和 doc 并输出相关性分数的 BERT。 |
| ModernBERT | “2024 刷新版” | 用 RoPE、RMSNorm、GeGLU、交替 local/global attention、8K context 重建的 encoder。 |

## 延伸阅读

- [Devlin et al. (2018). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805) — 原始论文。
- [Liu et al. (2019). RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/abs/1907.11692) — 如何正确训练 BERT；终结 NSP。
- [Clark et al. (2020). ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators](https://arxiv.org/abs/2003.10555) — replaced-token detection 在同等 compute 下优于 MLM。
- [Warner et al. (2024). Smarter, Better, Faster, Longer: A Modern Bidirectional Encoder](https://arxiv.org/abs/2412.13663) — ModernBERT 论文。
- [HuggingFace `modeling_bert.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/bert/modeling_bert.py) — 标准 encoder 参考。
