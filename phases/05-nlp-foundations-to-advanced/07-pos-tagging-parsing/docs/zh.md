# POS Tagging 与句法解析

> 语法曾经有一阵子不时髦。后来每条 LLM pipeline 都需要验证结构化抽取，它又回来了。

**类型：** 构建
**语言：** Python
**前置要求：** 阶段 5 第 01 课（Text Processing）、阶段 2 第 14 课（Naive Bayes）
**时间：** ~45 分钟

## 问题

第 01 课承诺过：lemmatization 需要 part-of-speech tag。如果不知道 `running` 是动词，lemmatizer 就不能把它还原成 `run`。如果不知道 `better` 是形容词，就不能把它还原成 `good`。

这个承诺背后藏着一个完整子领域。Part-of-speech tagging 会分配语法类别。Syntactic parsing 会恢复句子的树结构：哪个词修饰哪个词，哪个动词支配哪些论元。经典 NLP 花了二十年打磨二者。然后深度学习把它们压成 pretrained transformer 上的 token-classification 任务，研究社区就继续向前走了。

应用社区没有走开。每条结构化抽取 pipeline 仍然在底层使用 POS 和 dependency trees。LLM 生成的 JSON 会用语法约束验证。问答系统会用 dependency parses 拆解查询。机器翻译质量评估器会检查 parse trees 的对齐。

值得掌握。本课介绍 tagsets、baselines，以及什么时候该停止从零实现，直接调用 spaCy。

## 概念

**POS tagging** 给每个 token 标注语法类别。**Penn Treebank（PTB）** tagset 是英语默认选择。36 个标签，区分细到普通读者会觉得烦：`NN` 单数名词、`NNS` 复数名词、`NNP` 单数专有名词、`VBD` 动词过去式、`VBZ` 第三人称单数现在时，等等。**Universal Dependencies（UD）** tagset 更粗（17 个标签）且语言无关；它成了跨语言工作的默认选择。

```
The/DET cats/NOUN were/AUX running/VERB at/ADP 3pm/NOUN ./PUNCT
```

**Syntactic parsing** 产出一棵树。两大风格：

- **Constituency parsing。** 名词短语、动词短语、介词短语相互嵌套。输出是一棵 non-terminal categories（NP、VP、PP）为内部节点、词为叶子的树。
- **Dependency parsing。** 每个词都有一个它依附的 head word，并用语法关系标注。输出是一棵树，其中每条边都是一个 `(head, dependent, relation)` triple。

Dependency parsing 在 2010 年代胜出，因为它能自然泛化到各种语言，尤其是自由词序语言。

```
running is ROOT
cats is nsubj of running
were is aux of running
at is prep of running
3pm is pobj of at
```

## 构建它

### 第 1 步：most-frequent-tag baseline

能工作的最笨 POS tagger。对每个词，预测它在训练中最常出现的 tag。

```python
from collections import Counter, defaultdict


def train_mft(train_examples):
    word_tag_counts = defaultdict(Counter)
    all_tags = Counter()
    for tokens, tags in train_examples:
        for token, tag in zip(tokens, tags):
            word_tag_counts[token.lower()][tag] += 1
            all_tags[tag] += 1
    word_best = {w: c.most_common(1)[0][0] for w, c in word_tag_counts.items()}
    default_tag = all_tags.most_common(1)[0][0]
    return word_best, default_tag


def predict_mft(tokens, word_best, default_tag):
    return [word_best.get(t.lower(), default_tag) for t in tokens]
```

在 Brown corpus 上，这个 baseline 能达到约 85% accuracy。不算好，但任何严肃模型都不该低于这个地板。

### 第 2 步：bigram HMM tagger

建模整个序列的联合概率：

```
P(tags, words) = prod P(tag_i | tag_{i-1}) * P(word_i | tag_i)
```

两张表：transition probabilities（给定前一个 tag 的 tag 概率）和 emission probabilities（给定 tag 的词概率）。用带 Laplace smoothing 的计数估计二者。用 Viterbi 解码（在 tag lattice 上做 dynamic programming）。

```python
import math


def train_hmm(train_examples, alpha=0.01):
    transitions = defaultdict(Counter)
    emissions = defaultdict(Counter)
    tags = set()
    vocab = set()

    for tokens, ts in train_examples:
        prev = "<BOS>"
        for token, tag in zip(tokens, ts):
            transitions[prev][tag] += 1
            emissions[tag][token.lower()] += 1
            tags.add(tag)
            vocab.add(token.lower())
            prev = tag
        transitions[prev]["<EOS>"] += 1

    return transitions, emissions, tags, vocab


def log_prob(table, given, key, smooth_denom, alpha):
    return math.log((table[given].get(key, 0) + alpha) / smooth_denom)


def viterbi(tokens, transitions, emissions, tags, vocab, alpha=0.01):
    tags_list = list(tags)
    n = len(tokens)
    V = [[0.0] * len(tags_list) for _ in range(n)]
    back = [[0] * len(tags_list) for _ in range(n)]

    for j, tag in enumerate(tags_list):
        em_denom = sum(emissions[tag].values()) + alpha * (len(vocab) + 1)
        tr_denom = sum(transitions["<BOS>"].values()) + alpha * (len(tags_list) + 1)
        tr = log_prob(transitions, "<BOS>", tag, tr_denom, alpha)
        em = log_prob(emissions, tag, tokens[0].lower(), em_denom, alpha)
        V[0][j] = tr + em
        back[0][j] = 0

    for i in range(1, n):
        for j, tag in enumerate(tags_list):
            em_denom = sum(emissions[tag].values()) + alpha * (len(vocab) + 1)
            em = log_prob(emissions, tag, tokens[i].lower(), em_denom, alpha)
            best_prev = 0
            best_score = -1e30
            for k, prev_tag in enumerate(tags_list):
                tr_denom = sum(transitions[prev_tag].values()) + alpha * (len(tags_list) + 1)
                tr = log_prob(transitions, prev_tag, tag, tr_denom, alpha)
                score = V[i - 1][k] + tr + em
                if score > best_score:
                    best_score = score
                    best_prev = k
            V[i][j] = best_score
            back[i][j] = best_prev

    last_best = max(range(len(tags_list)), key=lambda j: V[n - 1][j])
    path = [last_best]
    for i in range(n - 1, 0, -1):
        path.append(back[i][path[-1]])
    return [tags_list[j] for j in reversed(path)]
```

Bigram HMM 在 Brown 上能达到约 93% accuracy。从 85% 到 93% 的跃迁主要来自 transition probabilities：模型学会了 `DET NOUN` 很常见，而 `NOUN DET` 很少见。

### 第 3 步：为什么现代 tagger 能打败它

Transition + emission probabilities 是局部的。它无法捕捉 `saw` 在 "I bought a saw" 里是名词，而在 "I saw the movie." 里是动词。带任意特征（suffix、word shape、前后词、词本身）的 CRF 能达到约 97%。BiLSTM-CRF 或 transformer 能达到约 98%+。

这个任务的天花板由标注者分歧决定。Penn Treebank 上，人类标注者大约 97% 情况下一致。超过 98% 的模型很可能在过拟合测试集。

### 第 4 步：dependency parsing 草图

从零完整实现 dependency parsing 超出本课范围；标准教材处理见 Jurafsky 和 Martin。需要知道两类经典方法：

- **Transition-based** parsers（arc-eager、arc-standard）像 shift-reduce parser 一样行动：读取 tokens，把它们 shift 到 stack 上，然后应用创建 arcs 的 reduce actions。Greedy decoding 很快。经典实现是 MaltParser。现代 neural 版本：Chen 和 Manning 的 transition-based parser。
- **Graph-based** parsers（Eisner's algorithm、Dozat-Manning biaffine）会给每条可能的 head-dependent edge 打分，然后选择 maximum spanning tree。更慢，但更准确。

对于大多数应用工作，调用 spaCy：

```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("The cats were running at 3pm.")
for token in doc:
    print(f"{token.text:10s} tag={token.tag_:5s} pos={token.pos_:6s} dep={token.dep_:10s} head={token.head.text}")
```

```
The        tag=DT    pos=DET    dep=det        head=cats
cats       tag=NNS   pos=NOUN   dep=nsubj      head=running
were       tag=VBD   pos=AUX    dep=aux        head=running
running    tag=VBG   pos=VERB   dep=ROOT       head=running
at         tag=IN    pos=ADP    dep=prep       head=running
3pm        tag=NN    pos=NOUN   dep=pobj       head=at
.          tag=.     pos=PUNCT  dep=punct      head=running
```

从下到上读 `dep` 列，句子的语法结构就出来了。

## 使用它

每个生产 NLP 库都会把 POS 和 dependency parsers 作为标准 pipeline 的一部分提供。

- **spaCy**（`en_core_web_sm` / `md` / `lg` / `trf`）。快、准，和 tokenization + NER + lemmatization 集成。`token.tag_`（Penn）、`token.pos_`（UD）、`token.dep_`（dependency relation）。
- **Stanford NLP（stanza）**。Stanford 对 CoreNLP 的继任者。60+ 语言上的 state-of-the-art。
- **trankit**。Transformer-based，UD accuracy 很好。
- **NLTK**。`pos_tag`。可用、慢、较老。适合教学。

### 为什么 2026 年仍然重要

- **Lemmatization。** 第 01 课需要 POS 才能正确 lemmatize。永远如此。
- **LLM 输出的结构化抽取。** 验证生成句子是否满足语法约束（例如 subject-verb agreement、required modifiers）。
- **Aspect-based sentiment。** Dependency parses 会告诉你哪个形容词修饰哪个名词。
- **Query understanding。** "movies directed by Wes Anderson starring Bill Murray" 可以通过 parse 拆成结构化约束。
- **Cross-lingual transfer。** UD tags 和 dependency relations 是语言无关的，可以对新语言做 zero-shot structured analysis。
- **Low-compute pipelines。** 如果你不能发布 transformer，POS + dependency parse + gazetteer 能把你带得相当远。

## 交付它

保存为 `outputs/skill-grammar-pipeline.md`：

```markdown
---
name: grammar-pipeline
description: 为下游 NLP 任务设计经典 POS + dependency pipeline。
version: 1.0.0
phase: 5
lesson: 07
tags: [nlp, pos, parsing]
---

给定下游任务（information extraction、rewrite validation、query decomposition、lemmatization），输出：

1. 使用的 tagset。英语单语 legacy pipelines 用 Penn Treebank，多语言或 cross-lingual 用 Universal Dependencies。
2. Library。大多数生产用 spaCy，学术级多语言用 stanza，最高 UD accuracy 用 trankit。写出具体 model ID。
3. Integration pattern。展示调用库并消费所需属性（`.pos_`、`.dep_`、`.head`）的 3-5 行代码。
4. 要测试的失败模式。Noun-verb ambiguity（`saw`、`book`、`can`）和 PP-attachment ambiguity 是经典陷阱。抽样 20 个输出并人工检查。

拒绝推荐自己写 parser。从零构建 parser 是研究项目，不是应用任务。指出任何消费 POS tags 但不处理 lowercase/uppercase 变体的 pipeline 都很脆弱。
```

## 练习

1. **简单。** 在小型 tagged corpus（例如 NLTK 的 Brown subset）上使用 most-frequent-tag baseline，在 held-out sentences 上衡量 accuracy。验证约 85% 的结果。
2. **中等。** 训练上面的 bigram HMM，并报告每个 tag 的 precision/recall。HMM 最容易混淆哪些 tag？
3. **困难。** 使用 spaCy 的 dependency parse，从 1000 个句子样本中抽取 subject-verb-object triples。在 50 个手工标注 triples 上评估。记录抽取失败的地方（通常是 passives、coordinations 和 elided subjects）。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|-----------------|-----------------------|
| POS tag | 词的类型 | 语法类别。PTB 有 36 个；UD 有 17 个。 |
| Penn Treebank | 标准 tagset | 英语专用。细分动词时态和名词数。 |
| Universal Dependencies | 多语言 tagset | 比 PTB 更粗；语言中立；跨语言工作的默认选择。 |
| Dependency parse | 句子树 | 每个词有一个 head，每条边有一个语法关系。 |
| Viterbi | Dynamic programming | 在给定 emissions 和 transitions 时，找到最高概率 tag sequence。 |

## 延伸阅读

- [Jurafsky and Martin — Speech and Language Processing, chapters 8 and 18](https://web.stanford.edu/~jurafsky/slp3/) — POS 与 parsing 的标准教材处理。
- [Universal Dependencies project](https://universaldependencies.org/) — 每个多语言 parser 都使用的跨语言 tagset 和 treebank collection。
- [spaCy linguistic features guide](https://spacy.io/usage/linguistic-features) — `Token` 暴露的每个属性的实用参考。
- [Chen and Manning (2014). A Fast and Accurate Dependency Parser using Neural Networks](https://nlp.stanford.edu/pubs/emnlp2014-depparser.pdf) — 把 neural parsers 带入主流的论文。
