# 自然语言推理：Textual Entailment

> “t entails h” 意味着人类读者看到 t 会得出 h 为真。NLI 是预测 entailment / contradiction / neutral 的任务。表面无聊，生产中却很承重。

**类型：** 学习
**语言：** Python
**前置要求：** 阶段 5 · 05（情感分析），阶段 5 · 13（问答）
**时间：** ~60 分钟

## 问题

你构建了一个 summarizer。它生成了摘要。你怎么知道摘要里没有 hallucination？

你构建了一个 chatbot。它回答了 “yes”。你怎么知道答案被检索段落支持？

你需要按主题分类 10,000 篇新闻文章。你没有训练标签。能复用一个模型吗？

这三个问题都能归约到 Natural Language Inference。NLI 问的是：给定 premise `t` 和 hypothesis `h`，`h` 是被 `t` 蕴含、被其矛盾，还是 neutral（无关）？

- **Hallucination check：** `t` = source document，`h` = summary claim。不是 entailment = hallucination。
- **Grounded QA：** `t` = retrieved passage，`h` = generated answer。不是 entailment = fabrication。
- **Zero-shot classification：** `t` = document，`h` = verbalized label（“This is about sports”）。Entailment = predicted label。

一个任务，三个生产用途。这就是为什么每个 RAG evaluation framework 底层都会带一个 NLI model。

## 概念

![NLI: three-way classification, premise vs hypothesis](../assets/nli.svg)

**三个标签。**

- **Entailment。** `t` → `h`。“The cat is on the mat” 蕴含 “There is a cat.”
- **Contradiction。** `t` → ¬`h`。“The cat is on the mat” 矛盾于 “There is no cat.”
- **Neutral。** 不能推出任一方向。“The cat is on the mat” 对 “The cat is hungry.” 是 neutral。

**不是逻辑蕴含。** NLI 是 *natural* language inference，关注典型人类读者会推断什么，而不是严格逻辑。“John walked his dog” 在 NLI 中蕴含 “John has a dog”，但严格一阶逻辑只有在你公理化 possession 时才承认。

**Datasets。**

- **SNLI**（2015）。570k 人工标注 pairs，premises 来自图像 captions。领域较窄。
- **MultiNLI**（2017）。433k pairs，覆盖 10 个 genres。2026 年标准训练 corpus。
- **ANLI**（2019）。Adversarial NLI。人类专门写来打破已有模型的样本。更难。
- **DocNLI, ConTRoL**（2020-21）。Document-length premises。测试 multi-hop 和 long-range inference。

**架构。** Transformer encoder（BERT、RoBERTa、DeBERTa）读取 `[CLS] premise [SEP] hypothesis [SEP]`。`[CLS]` representation 接 3-way softmax。在 MNLI 上训练，在 held-out benchmarks 上评估，in-distribution pairs 能达到 90%+ accuracy。

**通过 NLI 做 zero-shot。** 给定 document 和 candidate labels，把每个 label 写成 hypothesis（“This text is about sports”）。计算每个的 entailment probability。选择最大者。这就是 Hugging Face `zero-shot-classification` pipeline 背后的机制。

## 构建它

### 第 1 步：运行 pretrained NLI model

```python
from transformers import pipeline

nli = pipeline("text-classification",
               model="facebook/bart-large-mnli",
               top_k=None)  # return all labels; replaces deprecated return_all_scores=True

premise = "The cat is sleeping on the couch."
hypothesis = "There is a cat in the room."

result = nli({"text": premise, "text_pair": hypothesis})[0]
print(result)
# [{'label': 'entailment', 'score': 0.97},
#  {'label': 'neutral', 'score': 0.02},
#  {'label': 'contradiction', 'score': 0.01}]
```

生产 NLI 的开源默认是 `facebook/bart-large-mnli` 和 `microsoft/deberta-v3-large-mnli`。DeBERTa-v3 在 leaderboard 上领先。

### 第 2 步：zero-shot classification

```python
zs = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

text = "The stock market rallied after the central bank cut interest rates."
labels = ["finance", "sports", "politics", "technology"]

result = zs(text, candidate_labels=labels)
print(result)
# {'labels': ['finance', 'politics', 'technology', 'sports'],
#  'scores': [0.92, 0.05, 0.02, 0.01]}
```

默认 template 是 “This example is about {label}.”。可以用 `hypothesis_template` 自定义。不需要训练数据。不需要 fine-tuning。开箱即用。

### 第 3 步：RAG faithfulness check

```python
def is_faithful(answer, context, threshold=0.5):
    result = nli({"text": context, "text_pair": answer})[0]
    entail = next(s for s in result if s["label"] == "entailment")
    return entail["score"] > threshold
```

这是 RAGAS faithfulness 的核心。把 generated answer 拆成 atomic claims。逐条和 retrieved context 做 NLI check。报告 entail 的比例。

### 第 4 步：手写 NLI classifier（概念版）

`code/main.py` 里有 stdlib-only toy：通过 lexical overlap + negation detection 比较 premise 和 hypothesis。它无法和 transformer models 竞争，但展示了任务形状：两个文本输入，3-way label 输出，loss = `{entail, contradict, neutral}` 上的 cross-entropy。

## 坑

- **Hypothesis-only shortcuts。** 模型在 SNLI 上只看 hypothesis 就能用约 60% 预测标签，因为 “not”、“nobody”、“never” 和 contradiction 相关。它是检测 label leakage 的强 baseline。
- **Lexical overlap heuristic。** Subsequence heuristic（“每个子序列都被蕴含”）能过 SNLI，却会在 HANS/ANLI 上失败。使用 adversarial benchmarks。
- **Document-length degradation。** Sentence-level NLI models 在 document-length premises 上会掉 20+ F1。Long context 用 DocNLI-trained models。
- **Zero-shot template sensitivity。** “This example is about {label}” vs “{label}” vs “The topic is {label}” 可能让 accuracy 摆动 10+ points。要调 template。
- **Domain mismatch。** MNLI 在通用英语上训练。Legal、medical、scientific text 需要 domain-specific NLI models（如 SciNLI、MedNLI）。

## 使用它

2026 年技术栈：

| 用例 | 模型 |
|---------|-------|
| General-purpose NLI | `microsoft/deberta-v3-large-mnli` |
| Fast / edge | `cross-encoder/nli-deberta-v3-base` |
| Zero-shot classification（轻量） | `facebook/bart-large-mnli` |
| Document-level NLI | `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli` |
| Multilingual | `MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli` |
| RAG 中的 hallucination detection | RAGAS / DeepEval 内部的 NLI layer |

2026 年 meta-pattern：NLI 是文本理解的胶带。只要你需要“ A 是否支持 B？”或“A 是否矛盾 B？”先考虑 NLI，再考虑又调用一个 LLM。

## 交付它

保存为 `outputs/skill-nli-picker.md`：

```markdown
---
name: nli-picker
description: 为 classification / faithfulness / zero-shot 任务选择 NLI model、label template 和 evaluation setup。
version: 1.0.0
phase: 5
lesson: 21
tags: [nlp, nli, zero-shot]
---

给定 use case（faithfulness check、zero-shot classification、document-level inference），输出：

1. Model。命名 NLI checkpoint。理由关联 domain、length、language。
2. Template（如果是 zero-shot）。Verbalization pattern。给出例子。
3. Threshold。用于决策规则的 entailment cutoff。理由基于 calibration。
4. Evaluation。Held-out labeled set 上 accuracy、hypothesis-only baseline、adversarial subset。

拒绝没有 100-example labeled sanity check 的 zero-shot classification。拒绝在 document-length premises 上使用 sentence-level NLI model。标记任何“NLI 解决 hallucination”的说法；它只能降低，不能消除。
```

## 练习

1. **简单。** 在 20 个手写的（premise, hypothesis, label）三元组上运行 `facebook/bart-large-mnli`，覆盖三类标签。测量 accuracy。加入 adversarial “subsequence heuristic” 陷阱（“I did not eat the cake” vs “I ate the cake”），看看它是否会失败。
2. **中等。** 在 100 条 AG News headlines 上比较 zero-shot template `"This text is about {label}"`、`"The topic is {label}"` 和 `"{label}"`。报告 accuracy swing。
3. **困难。** 构建 RAG faithfulness checker：atomic-claim decomposition + 每条 claim 做 NLI。在 50 个带 gold context 的 RAG-generated answers 上评估。测量相对人工标签的 false-positive 和 false-negative rates。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|-----------------|-----------------------|
| NLI | Natural Language Inference | 对 premise-hypothesis 关系做 3-way classification。 |
| RTE | Recognizing Textual Entailment | NLI 的旧名字；同一任务。 |
| Entailment | “t implies h” | 典型读者基于 t 会认为 h 为真。 |
| Contradiction | “t rules out h” | 典型读者基于 t 会认为 h 为假。 |
| Neutral | “undecided” | 从 t 到 h 不能推出任一方向。 |
| Zero-shot classification | NLI as classifier | 把 labels verbalize 成 hypotheses，选最大 entailment。 |
| Faithfulness | 答案是否被支持？ | 对（retrieved context, generated answer）做 NLI。 |

## 延伸阅读

- [Bowman et al. (2015). A large annotated corpus for learning natural language inference](https://arxiv.org/abs/1508.05326) — SNLI。
- [Williams, Nangia, Bowman (2017). A Broad-Coverage Challenge Corpus for Sentence Understanding through Inference](https://arxiv.org/abs/1704.05426) — MultiNLI。
- [Nie et al. (2019). Adversarial NLI](https://arxiv.org/abs/1910.14599) — ANLI benchmark。
- [Yin, Hay, Roth (2019). Benchmarking Zero-shot Text Classification](https://arxiv.org/abs/1909.00161) — NLI-as-classifier。
- [He et al. (2021). DeBERTa: Decoding-enhanced BERT with Disentangled Attention](https://arxiv.org/abs/2006.03654) — 2026 年 NLI 主力。
