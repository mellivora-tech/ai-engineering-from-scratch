# 文本处理：Tokenization、Stemming、Lemmatization

> 语言是连续的，模型是离散的。预处理就是二者之间的桥。

**类型：** 构建
**语言：** Python
**前置要求：** 阶段 2，第 14 课（Naive Bayes）
**时间：** ~45 分钟

## 问题

模型不能直接读懂 “The cats were running.”。它读到的是整数。

每个 NLP 系统一开始都会遇到同样三个问题：一个词从哪里开始？词根是什么？在有帮助的时候，怎样把 `run`、`running`、`ran` 当成同一个东西；而在不该合并的时候，又怎样保留它们的差异？

Tokenization 做错了，模型就会从垃圾里学习。如果 tokenizer 把 `don't` 当成一个 token，但另一个地方又把 `do n't` 当成两个 token，训练分布就会被拆开。如果 stemmer 把 `organization` 和 `organ` 压成同一个 stem，topic modeling 就会崩掉。如果 lemmatizer 需要词性上下文，而你没有传给它，动词就会被当作名词处理。

本课会从零构建三个预处理原语，然后展示 NLTK 和 spaCy 如何完成同样的工作，让你看清其中的取舍。

## 概念

三个操作。每个都有自己的职责，也都有自己的失败模式。

**Tokenization** 会把字符串切成 token。这里的 “token” 故意说得很宽，因为合适的粒度取决于任务。经典 NLP 常用词级别。Transformer 常用 subword。没有空格的语言有时会用字符级别。

**Stemming** 用规则砍掉后缀。快、激进、粗糙。`running -> run`。`organization -> organ`。第二个就是失败模式。

**Lemmatization** 利用语法知识把词还原为词典形式。更慢、更准确，需要查找表或形态分析器。`ran -> run`（需要知道 “ran” 是 “run” 的过去式）。`better -> good`（需要知道比较级形式）。

经验法则：当速度重要且你能忍受噪声时，用 stemming（搜索索引、粗分类）。当含义重要时，用 lemmatization（问答、语义搜索、任何用户会直接阅读的文本）。

## 构建它

### 第 1 步：一个 regex word tokenizer

最简单且有用的 tokenizer 会按非字母数字字符切分，同时把标点作为独立 token 保留下来。不完美，也不是终点，但一行就能跑。

```python
import re

def tokenize(text):
    return re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?|[0-9]+|[^\sA-Za-z0-9]", text)
```

按优先级有三个模式：带可选内部撇号的单词（`don't`、`it's`）、纯数字、任何单个非空白且非字母数字字符（标点）作为独立 token。

```python
>>> tokenize("The cats weren't running at 3pm.")
['The', 'cats', "weren't", 'running', 'at', '3', 'pm', '.']
```

需要注意的失败模式：`3pm` 会被切成 `['3', 'pm']`，因为我们把连续字母和连续数字拆成了不同分支。对多数任务已经够用。URL、email、hashtag 都会坏掉。生产环境里，要把这些模式加在通用模式之前。

### 第 2 步：一个 Porter stemmer（只做 step 1a）

完整 Porter 算法有五个阶段的规则。单独的 step 1a 覆盖了最常见的英语后缀，也足以教会你这种模式。

```python
def stem_step_1a(word):
    if word.endswith("sses"):
        return word[:-2]
    if word.endswith("ies"):
        return word[:-2]
    if word.endswith("ss"):
        return word
    if word.endswith("s") and len(word) > 1:
        return word[:-1]
    return word
```

```python
>>> [stem_step_1a(w) for w in ["caresses", "ponies", "caress", "cats"]]
['caress', 'poni', 'caress', 'cat']
```

按从上到下读规则。`ies -> i` 这条规则解释了为什么 `ponies -> poni`，而不是 `pony`。真正的 Porter 会在 step 1b 修正它。规则之间会竞争。更早的规则获胜。顺序比任何单条规则都更重要。

### 第 3 步：基于查找表的 lemmatizer

真正的 lemmatization 需要形态学。一个适合教学的版本可以用小型 lemma 表和 fallback。

```python
LEMMA_TABLE = {
    ("running", "VERB"): "run",
    ("ran", "VERB"): "run",
    ("runs", "VERB"): "run",
    ("better", "ADJ"): "good",
    ("best", "ADJ"): "good",
    ("cats", "NOUN"): "cat",
    ("cat", "NOUN"): "cat",
    ("were", "VERB"): "be",
    ("was", "VERB"): "be",
    ("is", "VERB"): "be",
}

def lemmatize(word, pos):
    key = (word.lower(), pos)
    if key in LEMMA_TABLE:
        return LEMMA_TABLE[key]
    if pos == "VERB" and word.endswith("ing"):
        return word[:-3]
    if pos == "NOUN" and word.endswith("s"):
        return word[:-1]
    return word.lower()
```

```python
>>> lemmatize("running", "VERB")
'run'
>>> lemmatize("cats", "NOUN")
'cat'
>>> lemmatize("better", "ADJ")
'good'
>>> lemmatize("watched", "VERB")
'watched'
```

最后一个例子是关键教学点。`watched` 不在表里，而我们的 fallback 只处理 `ing`。真正的 lemmatization 会覆盖 `ed`、不规则动词、比较级形容词、带音变的复数（`children -> child`）。这就是为什么生产系统会使用 WordNet、spaCy 的 morphologizer，或完整的形态分析器。

### 第 4 步：把它们串起来

```python
def preprocess(text, pos_tagger=None):
    tokens = tokenize(text)
    stems = [stem_step_1a(t.lower()) for t in tokens]
    tags = pos_tagger(tokens) if pos_tagger else [(t, "NOUN") for t in tokens]
    lemmas = [lemmatize(word, pos) for word, pos in tags]
    return {"tokens": tokens, "stems": stems, "lemmas": lemmas}
```

缺失的部分是 POS tagger。阶段 5 第 07 课会构建一个。现在先默认所有词都是 `NOUN`，并承认这个限制。

## 使用它

NLTK 和 spaCy 都提供了生产版实现。各自只需要几行。

### NLTK

```python
import nltk
nltk.download("punkt_tab")
nltk.download("wordnet")
nltk.download("averaged_perceptron_tagger_eng")

from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import pos_tag

text = "The cats were running."
tokens = word_tokenize(text)
stems = [PorterStemmer().stem(t) for t in tokens]
lemmatizer = WordNetLemmatizer()
tagged = pos_tag(tokens)


def nltk_pos_to_wordnet(tag):
    if tag.startswith("V"):
        return "v"
    if tag.startswith("J"):
        return "a"
    if tag.startswith("R"):
        return "r"
    return "n"


lemmas = [lemmatizer.lemmatize(t, nltk_pos_to_wordnet(tag)) for t, tag in tagged]
```

`word_tokenize` 会处理 contraction、Unicode，以及你的 regex 漏掉的边界情况。`PorterStemmer` 会运行全部五个阶段。`WordNetLemmatizer` 需要把 NLTK 的 Penn Treebank 词性标签翻译成 WordNet 的缩写集合。上面这段转换胶水代码，是大多数教程会跳过的部分。

### spaCy

```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("The cats were running.")

for token in doc:
    print(token.text, token.lemma_, token.pos_)
```

```
The      the     DET
cats     cat     NOUN
were     be      AUX
running  run     VERB
.        .       PUNCT
```

spaCy 把整条 pipeline 都藏在 `nlp(text)` 后面。Tokenization、POS tagging 和 lemmatization 都会运行。大规模下比 NLTK 更快，开箱也更准确。代价是你很难轻松替换单个组件。

### 什么时候选哪个

| 场景 | 选择 |
|-----------|------|
| 教学、研究、替换组件 | NLTK |
| 生产、多语言、速度重要 | spaCy |
| Transformer pipeline（反正会用模型自己的 tokenizer） | 使用 `tokenizers` / `transformers`，跳过经典预处理 |

### 没人提醒你的两个失败模式

大多数教程教完算法就停了。真实预处理 pipeline 里有两件事会咬你，而且几乎没人讲。

**可复现性漂移。** NLTK 和 spaCy 会在版本之间改变 tokenization 和 lemmatizer 行为。spaCy 2.x 里产出 `['do', "n't"]` 的东西，在 3.x 里可能产出 `["don't"]`。你的模型是在一个分布上训练的。现在推理跑在另一个分布上。准确率悄悄下降，没有人知道为什么。把库版本固定在 `requirements.txt`。写一个预处理回归测试，冻结 20 个样例句子的预期 tokenization。每次升级都跑它。

**训练 / 推理不匹配。** 训练时用了激进预处理（小写化、停用词删除、stemming），部署时却接原始用户输入，然后性能直接塌掉。这是生产 NLP 中最常见的失败。如果训练期间做了预处理，推理期间就必须运行完全相同的函数。把预处理作为函数随模型包一起发布，不要让它停留在 serving 团队会重写的 notebook cell 里。

## 交付它

一个可复用 prompt，帮助工程师在不读三本教材的情况下选择预处理策略。

保存为 `outputs/prompt-preprocessing-advisor.md`：

```markdown
---
name: preprocessing-advisor
description: 为 NLP 任务推荐 tokenization、stemming 和 lemmatization 设置。
phase: 5
lesson: 01
---

你负责为经典 NLP 预处理提供建议。给定任务描述后，输出：

1. Tokenization 选择（regex、NLTK word_tokenize、spaCy 或 transformer tokenizer）。解释原因。
2. 是否使用 stemming、lemmatization、二者都用或二者都不用。解释原因。
3. 具体库调用。写出函数名。如果涉及 NLTK，引用 POS-tag 转换方式。
4. 用户应该测试的一个失败模式。

拒绝为用户可见文本推荐 stemming。拒绝在没有 POS tags 的情况下推荐 lemmatization。把非英语输入标记为需要不同 pipeline。
```

## 练习

1. **简单。** 扩展 `tokenize`，让 URL 保持为单个 token。测试：`tokenize("Visit https://example.com today.")` 应该产出一个 URL token。
2. **中等。** 实现 Porter step 1b。如果一个词包含元音并以 `ed` 或 `ing` 结尾，就移除它。处理双辅音规则（`hopping -> hop`，而不是 `hopp`）。
3. **困难。** 构建一个 lemmatizer，用 WordNet 作为查找表，但当 WordNet 没有条目时回退到你的 Porter stemmer。在带标签语料上衡量准确率，并和纯 WordNet、纯 Porter 对比。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|-----------------|-----------------------|
| Token | 一个词 | 模型消费的任何单位。可以是 word、subword、character 或 byte。 |
| Stem | 词根 | 基于规则删除后缀的结果。不一定是真实单词。 |
| Lemma | 词典形式 | 你会去词典查的形式。需要语法上下文才能正确计算。 |
| POS tag | 词性 | NOUN、VERB、ADJ 等类别。准确 lemmatization 需要它。 |
| Morphology | 词形规则 | 一个词如何根据时态、数、格改变形式。Lemmatization 依赖它。 |

## 延伸阅读

- [Porter, M. F. (1980). An algorithm for suffix stripping](https://tartarus.org/martin/PorterStemmer/def.txt) — 原始论文，五页，至今仍是最清晰的解释。
- [spaCy 101 — linguistic features](https://spacy.io/usage/linguistic-features) — 真实 pipeline 如何接线。
- [NLTK book, chapter 3](https://www.nltk.org/book/ch03.html) — 你还没想到的 tokenization 边界情况。
