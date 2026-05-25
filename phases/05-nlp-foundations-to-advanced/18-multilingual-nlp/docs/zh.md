# 多语言 NLP

> 一个模型，100 多种语言，而其中大多数语言没有训练数据。Cross-lingual transfer 是 2020 年代的实用奇迹。

**类型：** 学习
**语言：** Python
**前置要求：** 阶段 5 · 04（GloVe、FastText、Subword），阶段 5 · 11（机器翻译）
**时间：** ~45 分钟

## 问题

英语有数十亿标注样本。乌尔都语有几千个。迈蒂利语几乎没有。任何服务全球用户的实用 NLP 系统，都必须能在那些没有特定任务训练数据的长尾语言上工作。

Multilingual models 通过同时在多种语言上训练一个模型来解决这个问题。共享表示让模型把高资源语言里学到的能力迁移到低资源语言。只在英语情感分析上 fine-tune 模型，它就能开箱即用地对乌尔都语给出相当不错的情感预测。这就是 zero-shot cross-lingual transfer，它重塑了 NLP 面向世界交付的方式。

本课会命名这些 tradeoffs、经典模型，以及让新团队最容易踩坑的一个决策：为 transfer 选择 source language。

## 概念

![Cross-lingual transfer via shared multilingual embedding space](../assets/multilingual.svg)

**共享 vocabulary。** Multilingual models 使用 SentencePiece 或 WordPiece tokenizer，它在所有目标语言的文本上训练。Vocabulary 是共享的：同一个 subword unit 会在相关语言中表示相同词素。英语和意大利语里的 `anti-` 会得到同一个 token。

**共享 representation。** 在多种语言上用 masked language modeling 预训练的 transformer，会学到让不同语言中语义相近句子的 hidden states 彼此接近。mBERT、XLM-R 和 NLLB 都表现出这一点。英语 `"cat"` 的 embedding 会靠近法语 `"chat"` 和西班牙语 `"gato"`，整句 embedding 也类似。

**Zero-shot transfer。** 在一种语言（通常英语）的标注数据上 fine-tune 模型。推理时，在模型支持的任何其他语言上运行。不需要目标语言标签。类型学相近的语言效果强，距离远的语言效果弱。

**Few-shot fine-tuning。** 在目标语言中加入 100-500 个标注样本。分类任务的准确率会跳到英语 baseline 的 95-98%。这是 multilingual NLP 里性价比最高的单个杠杆。

## 模型

| 模型 | 年份 | 覆盖 | 说明 |
|-------|------|----------|-------|
| mBERT | 2018 | 104 种语言 | 在 Wikipedia 上训练。第一个实用 multilingual LM。低资源弱。 |
| XLM-R | 2019 | 100 种语言 | 在 CommonCrawl 上训练（远大于 Wikipedia）。定义 cross-lingual baseline。Base 270M，Large 550M。 |
| XLM-V | 2023 | 100 种语言 | XLM-R + 1M-token vocabulary（对比 250k）。低资源更好。 |
| mT5 | 2020 | 101 种语言 | 用于 multilingual generation 的 T5 架构。 |
| NLLB-200 | 2022 | 200 种语言 | Meta 翻译模型；包含 55 种低资源语言。 |
| BLOOM | 2022 | 46 种语言 + 13 种编程语言 | 开放的 176B multilingual LLM。 |
| Aya-23 | 2024 | 23 种语言 | Cohere 的 multilingual LLM。阿拉伯语、印地语、斯瓦希里语强。 |

按用例选择。分类任务用 XLM-R-base 是稳妥默认。生成任务取决于翻译还是开放生成，分别考虑 mT5 或 NLLB。LLM 风格工作可以用 Aya-23 或 Claude，并显式做 multilingual prompting。

## Source-language 决策（2026 研究）

多数团队默认用英语作为 fine-tuning source。最近的研究（2026）表明这经常是错的。

语言相似度比原始 corpus 大小更能预测 transfer 质量。对斯拉夫语族目标，德语或俄语常常胜过英语。对印度语族目标，印地语常常胜过英语。**qWALS** similarity metric（2026，基于 World Atlas of Language Structures 特征）量化了这一点。**LANGRANK**（Lin et al., ACL 2019）是另一个更早的方法，会综合语言相似度、corpus 大小和亲缘关系来给 candidate source languages 排名。

实用规则：如果目标语言有一个类型学相近的高资源亲属语言，先试着在它上面 fine-tune，再和英语 fine-tune 比较。

## 构建它

### 第 1 步：zero-shot cross-lingual classification

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

tok = AutoTokenizer.from_pretrained("joeddav/xlm-roberta-large-xnli")
model = AutoModelForSequenceClassification.from_pretrained("joeddav/xlm-roberta-large-xnli")


def classify(text, candidate_labels, hypothesis_template="This text is about {}."):
    scores = {}
    for label in candidate_labels:
        hypothesis = hypothesis_template.format(label)
        inputs = tok(text, hypothesis, return_tensors="pt", truncation=True)
        with torch.no_grad():
            logits = model(**inputs).logits[0]
        entail_score = torch.softmax(logits, dim=-1)[2].item()
        scores[label] = entail_score
    return dict(sorted(scores.items(), key=lambda x: -x[1]))


print(classify("I love this product!", ["positive", "negative", "neutral"]))
print(classify("मुझे यह उत्पाद पसंद है!", ["positive", "negative", "neutral"]))
print(classify("J'adore ce produit !", ["positive", "negative", "neutral"]))
```

一个模型，三种语言，同一个 API。XLM-R 在 NLI 数据上训练，通过 entailment trick 可以很好地迁移到分类。

### 第 2 步：multilingual embedding space

```python
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

pairs = [
    ("The cat is sleeping.", "Le chat dort."),
    ("The cat is sleeping.", "El gato está durmiendo."),
    ("The cat is sleeping.", "Die Katze schläft."),
    ("The cat is sleeping.", "The dog is barking."),
]

for eng, other in pairs:
    emb_eng = model.encode([eng], normalize_embeddings=True)[0]
    emb_other = model.encode([other], normalize_embeddings=True)[0]
    sim = float(np.dot(emb_eng, emb_other))
    print(f"  {eng!r} <-> {other!r}: cos={sim:.3f}")
```

翻译句会落在 embedding space 的近处。另一句英语会更远。这就是 cross-lingual retrieval、clustering 和 similarity 能工作的原因。

### 第 3 步：few-shot fine-tuning strategy

```python
from transformers import TrainingArguments, Trainer
from datasets import Dataset


def few_shot_finetune(base_model, base_tokenizer, examples):
    ds = Dataset.from_list(examples)

    def tokenize_fn(ex):
        out = base_tokenizer(ex["text"], truncation=True, max_length=128)
        out["labels"] = ex["label"]
        return out

    ds = ds.map(tokenize_fn)
    args = TrainingArguments(
        output_dir="out",
        per_device_train_batch_size=8,
        num_train_epochs=5,
        learning_rate=2e-5,
        save_strategy="no",
    )
    trainer = Trainer(model=base_model, args=args, train_dataset=ds)
    trainer.train()
    return base_model
```

对 100-500 个目标语言样本，`num_train_epochs=5` 和 `learning_rate=2e-5` 是安全默认值。更高学习率会让 multilingual alignment 坍塌，你会得到一个 English-only 模型。

## 真正有效的评估

- **每种语言各自的 held-out accuracy。** 不要聚合。聚合会隐藏长尾。
- **和 monolingual baseline 对比。** 对数据足够多的语言，从头训练的 monolingual model 有时会击败 multilingual model。要测试。
- **Entity-level tests。** 目标语言中的命名实体。对于远离拉丁文字的脚本，multilingual models 往往 tokenization 很弱。
- **Cross-lingual consistency。** 两种语言里相同含义应该产生相同预测。测量差距。

## 使用它

2026 年技术栈：

| 任务 | 推荐 |
|-----|-------------|
| 100 种语言分类 | Fine-tuned XLM-R-base（~270M） |
| Zero-shot text classification | `joeddav/xlm-roberta-large-xnli` |
| Multilingual sentence embeddings | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` |
| 200 种语言翻译 | `facebook/nllb-200-distilled-600M`（见第 11 课） |
| Generative multilingual | Claude、GPT-4、Aya-23、mT5-XXL |
| 低资源语言 NLP | XLM-V，或在相关高资源语言上做 domain-specific fine-tune |

如果性能重要，一定为目标语言留出 fine-tuning 预算。Zero-shot 是起点，不是终点。

### Tokenization tax（低资源语言会出什么问题）

Multilingual models 为所有语言共享一个 tokenizer。这个 vocabulary 在英语、法语、西班牙语、中文、德语占主导的 corpus 上训练。对任何不在主导集合内的语言，三种 tax 会悄悄叠加：

- **Fertility tax。** 低资源语言文本会比英语分成更多 tokens。一个印地语句子可能需要等价英语句子 3-5 倍 tokens。这 3-5 倍会吞掉 context window、训练效率和延迟。
- **Variant recovery tax。** 每个 typo、diacritic variant、Unicode normalization mismatch 或大小写变化，都会变成 embedding space 里的冷启动无关序列。模型学不到母语者觉得显然的正字法对应关系。
- **Capacity spillover tax。** 前两种 tax 消耗 context positions、layer depth 和 embedding dimensions。留给实际 reasoning 的容量系统性地小于高资源语言从同一模型得到的容量。

实际症状：模型在印地语上训练正常，loss curve 看着对，eval perplexity 合理，但生产输出细微错误。形态在句子中间崩塌。稀有变位无法恢复。**坏 tokenizer 不是靠更多数据就能补救的。**

缓解：选择对目标语言覆盖良好的 tokenizer（XLM-V 的 1M-token vocabulary 是直接修复）；训练前在 held-out 目标文本上验证 tokenization fertility；对真正长尾脚本使用 byte-level fallback（SentencePiece `byte_fallback=True`，GPT-2 风格 byte-level BPE），确保永远不会 OOV。

## 交付它

保存为 `outputs/skill-multilingual-picker.md`：

```markdown
---
name: multilingual-picker
description: 为 multilingual NLP 任务选择 source language、target model 和 evaluation plan。
version: 1.0.0
phase: 5
lesson: 18
tags: [nlp, multilingual, cross-lingual]
---

给定需求（target languages、task type、每种语言可用 labeled data），输出：

1. Fine-tuning source language。默认英语；如果 target language 有类型学接近的高资源语言，检查 LANGRANK 或 qWALS。
2. Base model。XLM-R（classification）、mT5（generation）、NLLB（translation）、Aya-23（generative LLM）。
3. Few-shot budget。如果可用，从 100-500 个目标语言样本开始。只有在无法标注时才用 zero-shot。
4. Evaluation plan。Per-language accuracy（不要 aggregate）、cross-lingual consistency、non-Latin scripts 上的 entity-level F1。

拒绝在没有 per-language evaluation 的情况下交付 multilingual model。Aggregate metrics 会隐藏长尾失败。标记 tokenization coverage 低的脚本（Amharic、Tigrinya、许多非洲语言），说明需要带 byte-fallback 的模型（SentencePiece with byte_fallback=True，或 GPT-2 这种 byte-level tokenizer）。
```

## 练习

1. **简单。** 在英语、法语、印地语、阿拉伯语中每种语言 10 个句子上运行 zero-shot classification pipeline。分别报告 accuracy。你应该看到法语强、印地语不错、阿拉伯语波动。
2. **中等。** 用 `paraphrase-multilingual-MiniLM-L12-v2` 在一个小型混合语言 corpus 上构建 cross-lingual retriever。用英语 query，检索任意语言的文档。测量 recall@5。
3. **困难。** 对一个印地语分类任务比较 English-source 和 Hindi-source fine-tuning。在两种方案下都使用 500 个目标语言样本做 few-shot fine-tuning。报告哪个 source 产生更高印地语 accuracy，以及高多少。这就是 LANGRANK thesis 的缩小版。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|-----------------|-----------------------|
| Multilingual model | 一个模型，多种语言 | 跨语言共享 vocabulary 和参数。 |
| Cross-lingual transfer | 在一种语言上训练，另一种语言上运行 | 在 source 上 fine-tune，在没有目标语言标签的 target 上评估。 |
| Zero-shot | 没有目标语言标签 | 不在目标语言上 fine-tune 的 transfer。 |
| Few-shot | 少量目标标签 | 用 100-500 个目标语言样本做 fine-tuning。 |
| mBERT | 第一个 multilingual LM | 在 Wikipedia 上预训练的 104 语言 BERT。 |
| XLM-R | 标准 cross-lingual baseline | 在 CommonCrawl 上预训练的 100 语言 RoBERTa。 |
| NLLB | Meta 的 200 语言 MT | No Language Left Behind。包含 55 种低资源语言。 |

## 延伸阅读

- [Conneau et al. (2019). Unsupervised Cross-lingual Representation Learning at Scale](https://arxiv.org/abs/1911.02116) — XLM-R 论文。
- [Pires, Schlinger, Garrette (2019). How Multilingual is Multilingual BERT?](https://arxiv.org/abs/1906.01502) — 开启 cross-lingual transfer 研究线的分析论文。
- [Costa-jussà et al. (2022). No Language Left Behind](https://arxiv.org/abs/2207.04672) — NLLB-200 论文。
- [Üstün et al. (2024). Aya Model: An Instruction Finetuned Open-Access Multilingual Language Model](https://arxiv.org/abs/2402.07827) — Aya，Cohere 的 multilingual LLM。
- [Language Similarity Predicts Cross-Lingual Transfer Learning Performance (2026)](https://www.mdpi.com/2504-4990/8/3/65) — qWALS / LANGRANK source-language 论文。
