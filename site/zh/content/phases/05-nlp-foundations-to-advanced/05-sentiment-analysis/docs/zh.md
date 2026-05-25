# 情感分析

> 经典 NLP 任务。关于经典文本分类，你需要知道的大部分东西都会在这里出现。

**类型：** 构建
**语言：** Python
**前置要求：** 阶段 5 第 02 课（BoW + TF-IDF）、阶段 2 第 14 课（Naive Bayes）
**时间：** ~75 分钟

## 问题

"The food was not great." 是正面还是负面？

Sentiment 听起来很简单。评论者说他们喜欢或不喜欢某个东西。给句子打标签。它之所以成了经典 NLP 任务，是因为每个看起来简单的例子背后都藏着难题。否定会翻转含义。讽刺会反转它。"Not bad at all" 虽然有两个负面编码词，却是正面的。Emoji 携带的信号可能比周围文本更强。领域词汇很重要（音乐评论里的 `tight` 和时尚评论里的 `tight` 含义不同）。

Sentiment 是经典 NLP 的实验室。如果你理解为什么每个 naive baseline 都有具体失败模式，你就理解了为什么要发明更丰富的模型。本课会从零构建 Naive Bayes baseline，加上 logistic regression，并点名那些让生产 sentiment 变成合规级问题的陷阱。

## 概念

经典 sentiment 是一个两步配方。

1. **表示。** 把文本转换成 feature vector。BoW、TF-IDF 或 n-grams。
2. **分类。** 在标注样本上拟合线性模型（Naive Bayes、logistic regression、SVM）。

Naive Bayes 是能工作的最笨模型。假设在给定标签的情况下，每个特征彼此独立。从计数中估计 `P(word | positive)` 和 `P(word | negative)`。推理时，把概率相乘。“naive”的独立性假设错得可笑，但结果强得惊人。原因是：面对稀疏文本特征和中等数据量，分类器更关心每个词偏向哪一类，而不是偏向多少。

Logistic regression 修复了独立性假设。它为每个特征学习一个权重，包括负权重。`not good` 作为 bigram 特征会得到负权重。Naive Bayes 无法对从未标注过的 bigrams 做到这一点。

## 构建它

### 第 1 步：一个真实的小数据集

```python
POSITIVE = [
    "absolutely loved this movie",
    "beautiful cinematography and a great story",
    "one of the best films of the year",
    "brilliant acting from the lead",
    "heartwarming and funny",
]

NEGATIVE = [
    "boring and far too long",
    "not worth your time",
    "the plot made no sense",
    "terrible acting, awful script",
    "i want my two hours back",
]
```

故意做得很小。真实工作会使用数万样本（IMDb、SST-2、Yelp polarity）。数学完全相同。

### 第 2 步：从零实现 multinomial Naive Bayes

```python
import math
from collections import Counter


def train_nb(docs_by_class, vocab, alpha=1.0):
    class_priors = {}
    class_word_probs = {}
    total_docs = sum(len(d) for d in docs_by_class.values())

    for cls, docs in docs_by_class.items():
        class_priors[cls] = len(docs) / total_docs
        counts = Counter()
        for doc in docs:
            for token in doc:
                counts[token] += 1
        total = sum(counts.values()) + alpha * len(vocab)
        class_word_probs[cls] = {
            w: (counts[w] + alpha) / total for w in vocab
        }
    return class_priors, class_word_probs


def predict_nb(doc, class_priors, class_word_probs):
    scores = {}
    for cls in class_priors:
        s = math.log(class_priors[cls])
        for token in doc:
            if token in class_word_probs[cls]:
                s += math.log(class_word_probs[cls][token])
        scores[cls] = s
    return max(scores, key=scores.get)
```

Additive smoothing（alpha=1.0）就是 Laplace smoothing。没有它，一个在某类别中没见过的词概率会是零，log 会炸。实践里常用 `alpha=0.01`。`alpha=1.0` 是教学默认值。

### 第 3 步：从零实现 logistic regression

```python
import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


def train_lr(X, y, epochs=500, lr=0.05, l2=0.01):
    n_features = X.shape[1]
    w = np.zeros(n_features)
    b = 0.0
    for _ in range(epochs):
        logits = X @ w + b
        preds = sigmoid(logits)
        err = preds - y
        grad_w = X.T @ err / len(y) + l2 * w
        grad_b = err.mean()
        w -= lr * grad_w
        b -= lr * grad_b
    return w, b


def predict_lr(X, w, b):
    return (sigmoid(X @ w + b) >= 0.5).astype(int)
```

L2 regularization 在这里很重要。文本特征是稀疏的；没有 L2，模型会记住训练样本。从 `0.01` 开始调。

### 第 4 步：处理否定（失败模式）

想想 "not good" 和 "not bad"。BoW 分类器看到的是 `{not, good}` 和 `{not, bad}`，它会从训练里出现更多的情况学习。Bigram 分类器看到的是 `not_good` 和 `not_bad`，会把它们当成不同特征来学。这通常就够了。

当你没有 bigrams 时，一个更粗糙但有效的修复是：**negation scoping**。把否定词之后、直到下一个标点前的 token 都加上 `NOT_` 前缀。

```python
NEGATION_WORDS = {"not", "no", "never", "nor", "none", "nothing", "neither"}
NEGATION_TERMINATORS = {".", "!", "?", ",", ";"}


def apply_negation(tokens):
    out = []
    negate = False
    for token in tokens:
        if token in NEGATION_TERMINATORS:
            negate = False
            out.append(token)
            continue
        if token in NEGATION_WORDS:
            negate = True
            out.append(token)
            continue
        out.append(f"NOT_{token}" if negate else token)
    return out
```

```python
>>> apply_negation(["not", "good", "at", "all", ".", "but", "funny"])
['not', 'NOT_good', 'NOT_at', 'NOT_all', '.', 'but', 'funny']
```

现在 `good` 和 `NOT_good` 是不同特征。分类器可以给它们相反权重。三行预处理，在 sentiment benchmark 上通常能看到可测量的准确率提升。

### 第 5 步：真正重要的评估指标

如果类别不平衡，accuracy 会误导你。真实 sentiment 语料通常是 70-80% 正面或 70-80% 负面；一个永远预测多数类的分类器能拿到 80% accuracy，但毫无价值。以下每一项都要报告：

- **每类 precision 和 recall。** 每个类一对。用 macro-average 得到尊重类别平衡的单个数字。
- **Macro-F1（不平衡数据的主指标）。** 每类 F1 的均值，等权重。类别不平衡时用它代替 accuracy。
- **Weighted-F1（替代指标）。** 和 macro 类似，但按类别频率加权。当不平衡本身具有业务意义时，和 macro-F1 一起报告。
- **Confusion matrix。** 原始计数。信任任何标量指标前都要检查它；它会揭示模型混淆了哪一对类别。
- **每类错误样本。** 每类抽 5 个错误预测。读它们。没有东西能替代阅读真实错误。

对于严重不平衡数据（> 95-5 比例），报告 **AUROC** 和 **AUPRC**，而不是 accuracy。AUPRC 对少数类更敏感，而少数类通常才是你关心的东西（垃圾邮件、欺诈、稀有情感）。

**要避免的常见 bug。** 在不平衡数据上报告 micro-F1 而不是 macro-F1，会得到一个看起来很高的数字，因为它被多数类主导。Macro-F1 会强迫你看见少数类表现。

```python
def evaluate(y_true, y_pred):
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn, "precision": precision, "recall": recall, "f1": f1}
```

## 使用它

scikit-learn 用六行代码正确完成它。

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

pipe = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, stop_words=None)),
    ("clf", LogisticRegression(C=1.0, max_iter=1000)),
])
pipe.fit(X_train, y_train)
print(pipe.score(X_test, y_test))
```

注意三件事。`stop_words=None` 保留否定词。`ngram_range=(1, 2)` 加入 bigrams，让 `not_good` 成为特征。`sublinear_tf=True` 会压低重复词。这三个标志就是 SST-2 上 75% 准确 baseline 和 85% 准确 baseline 的差距。

### 什么时候转向 transformer

- 讽刺检测。经典模型在这里会失败。句号。
- 情感在长评论中途发生变化。
- Aspect-based sentiment。"Camera was great but battery was terrible." 你需要把情感归因到不同 aspect。只能用 transformers 或 structured output models。
- 非英语、低资源语言。Multilingual BERT 免费给你一个 zero-shot baseline。

如果你需要上面任何一点，跳到阶段 7（transformers deep dive）。否则，基于 TF-IDF + bigrams + negation handling 的 Naive Bayes 或 logistic regression，就是你在 2026 年的生产 baseline。

### 可复现性陷阱（再来一次）

重训 sentiment models 很常见。重新评估它们不常见。论文中的 accuracy 数字使用了特定 split、特定 preprocessing、特定 tokenizers。如果你没有使用完全相同的 pipeline 就把新模型和 baseline 比较，得到的 delta 会误导人。永远在你的 pipeline 上重新生成 baseline，不要直接拿论文数字。

## 交付它

保存为 `outputs/prompt-sentiment-baseline.md`：

```markdown
---
name: sentiment-baseline
description: 为新数据集设计 sentiment analysis baseline。
phase: 5
lesson: 05
---

给定数据集描述（领域、语言、大小、标签粒度、延迟预算），输出：

1. Feature extraction 配方。指定 tokenizer、n-gram range、stopword 策略（通常保留）、negation handling（scoped prefix 或 bigrams）。
2. Classifier。Baseline 用 Naive Bayes，生产用 logistic regression；只有当领域需要讽刺 / aspects / cross-lingual 时才用 transformer。
3. Evaluation plan。报告 precision、recall、F1、confusion matrix 和每类错误样本（不只报告标量）。
4. 部署后要监控的一个失败模式。Domain drift 和 sarcasm 是前两个。

拒绝为 sentiment tasks 推荐删除停用词。类别不平衡时（例如 90% positive），拒绝把 accuracy 作为唯一指标。指出 subword-rich languages 应该用 FastText 或 transformer embeddings，而不是 word-level TF-IDF。
```

## 练习

1. **简单。** 把 `apply_negation` 作为预处理步骤加进 scikit-learn pipeline，并在小型 sentiment 数据集上衡量 F1 delta。
2. **中等。** 实现 class-weighted logistic regression（给 scikit-learn 传 `class_weight="balanced"`，或自己推导梯度）。在合成的 90-10 类别不平衡上衡量效果。
3. **困难。** 通过在 sentiment model 的 residuals 上训练第二个分类器，构建 sarcasm detector。记录你的实验设置。当 accuracy 低于随机水平时提醒读者（2 类 sarcasm 的随机水平约 50%，大多数第一次尝试都会落在那里）。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|-----------------|-----------------------|
| Polarity | 正面或负面 | 二元标签；有时扩展到 neutral 或细粒度（5 星）。 |
| Aspect-based sentiment | 每个 aspect 的 polarity | 把情感归因到文本中提到的具体实体或属性。 |
| Negation scoping | 翻转附近 token | 在 "not" 后面的 token 前加 `NOT_`，直到标点为止。 |
| Laplace smoothing | 给计数加 1 | 防止 Naive Bayes 中出现零概率特征。 |
| L2 regularization | 收缩权重 | 在 loss 中加入 `lambda * sum(w^2)`。对稀疏文本特征很关键。 |

## 延伸阅读

- [Pang and Lee (2008). Opinion Mining and Sentiment Analysis](https://www.cs.cornell.edu/home/llee/opinion-mining-sentiment-analysis-survey.html) — 奠基性综述。很长，但前四节覆盖了经典部分的全部核心。
- [Wang and Manning (2012). Baselines and Bigrams: Simple, Good Sentiment and Topic Classification](https://aclanthology.org/P12-2018/) — 这篇论文说明了 bigrams + Naive Bayes 在短文本上有多难击败。
- [scikit-learn text feature extraction docs](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction) — `CountVectorizer`、`TfidfVectorizer` 以及你会调的每个旋钮的参考。
