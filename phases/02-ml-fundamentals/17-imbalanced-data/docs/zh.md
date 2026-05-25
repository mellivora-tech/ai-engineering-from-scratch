# 处理不平衡数据

> 当 99% 的数据都是“正常”时，accuracy 是谎言。

**类型：** 构建
**语言：** Python
**前置要求：** 阶段 2，第 01-09 课（尤其是 evaluation metrics）
**时间：** ~90 分钟

## 学习目标

- 从零实现 SMOTE，并解释 synthetic oversampling 与 random duplication 的区别
- 使用 F1、AUPRC 和 Matthews Correlation Coefficient 评估 imbalanced classifiers，而不是 accuracy
- 比较 class weighting、threshold tuning 和 resampling strategies，并为给定 imbalance ratio 选择正确方法
- 构建完整 imbalanced data pipeline，组合 SMOTE、class weights 和 threshold optimization

## 问题

你构建了一个欺诈检测模型。它达到 99.9% accuracy。你庆祝。然后你发现它对每笔交易都预测 “not fraud”。

这不是 bug。当只有 0.1% 交易是欺诈时，这是理性行为。模型学到总是猜 majority class 能最小化整体错误。它技术上正确，但完全没用。

这种情况出现在所有真正重要的分类问题中。疾病诊断：1% positive rate。网络入侵：0.01% attacks。制造缺陷：0.5% defective。垃圾邮件过滤：20% spam。流失预测：5% churners。Minority class 越关键，通常越稀有。

Accuracy 失败是因为它平等对待所有正确预测。正确标记合法交易和正确抓住欺诈都只算 accuracy 的一分。但抓住欺诈才是模型存在的全部理由。我们需要 metrics、techniques 和 training strategies，迫使模型关注罕见但重要的 class。

## 概念

### 为什么 Accuracy 失败

考虑一个 1000 个 samples 的数据集：990 个 negative，10 个 positive。总是预测 negative 的模型：

|  | Predicted Positive | Predicted Negative |
|--|---|---|
| Actually Positive | 0 (TP) | 10 (FN) |
| Actually Negative | 0 (FP) | 990 (TN) |

Accuracy = (0 + 990) / 1000 = 99.0%

模型抓住零个欺诈。零个疾病。零个缺陷。但 accuracy 说 99%。这就是 accuracy 对 imbalanced problems 危险的原因。

### 更好的 Metrics

**Precision** = TP / (TP + FP)。所有被 flag 为 positive 的样本中，有多少是真的？高 precision 表示 false alarms 少。

**Recall** = TP / (TP + FN)。所有实际 positive 的样本中，我们抓住多少？高 recall 表示 missed positives 少。

**F1 Score** = 2 * precision * recall / (precision + recall)。调和平均。相比算术平均，它更严厉惩罚 precision 和 recall 之间的极端不平衡。

**F-beta Score** = (1 + beta^2) * precision * recall / (beta^2 * precision + recall)。当 beta > 1，recall 更重要。当 beta < 1，precision 更重要。F2 常用于 fraud detection（漏掉欺诈比 false alarm 更糟）。

**AUPRC**（Area Under Precision-Recall Curve）。类似 AUC-ROC，但对 imbalanced data 更有信息量。Random classifier 的 AUPRC 等于 positive class rate（不像 ROC 是 0.5）。这让改进更容易被看见。

**Matthews Correlation Coefficient** = (TP * TN - FP * FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))。范围从 -1 到 +1。只有当模型在两个 classes 上都表现好时才给高分。即使 class sizes 差异很大也平衡。

对于上面的“总是预测 negative”模型：precision = 0/0（未定义，通常设为 0），recall = 0/10 = 0，F1 = 0，MCC = 0。这些 metrics 正确识别出模型毫无价值。

### Imbalanced Data Pipeline

```mermaid
flowchart TD
    A[Imbalanced Dataset] --> B{Imbalance Ratio?}
    B -->|Mild: 80/20| C[Class Weights]
    B -->|Moderate: 95/5| D[SMOTE + Threshold Tuning]
    B -->|Severe: 99/1| E[SMOTE + Class Weights + Threshold]
    C --> F[Train Model]
    D --> F
    E --> F
    F --> G[Evaluate with F1 / AUPRC / MCC]
    G --> H{Good Enough?}
    H -->|No| I[Try Different Strategy]
    H -->|Yes| J[Deploy with Monitoring]
    I --> B
```

### SMOTE：Synthetic Minority Oversampling Technique

Random oversampling 会复制现有 minority samples。它能工作，但有过拟合风险，因为模型会反复看到完全相同的点。

SMOTE 会创建新的 synthetic minority samples，这些样本合理但不是拷贝。算法：

1. 对每个 minority sample x，在其他 minority samples 中找到 k nearest neighbors
2. 随机选一个 neighbor
3. 在 x 和该 neighbor 之间的线段上创建新样本

公式：`new_sample = x + random(0, 1) * (neighbor - x)`

这会在真实 minority points 之间插值，在 feature space 的同一区域创建样本，而不是复制现有数据。

```mermaid
flowchart LR
    subgraph Original["Original Minority Points"]
        P1["x1 (1.0, 2.0)"]
        P2["x2 (1.5, 2.5)"]
        P3["x3 (2.0, 1.5)"]
    end
    subgraph SMOTE["SMOTE Generation"]
        direction TB
        S1["Pick x1, neighbor x2"]
        S2["random t = 0.4"]
        S3["new = x1 + 0.4*(x2-x1)"]
        S4["new = (1.2, 2.2)"]
        S1 --> S2 --> S3 --> S4
    end
    Original --> SMOTE
    subgraph Result["Augmented Set"]
        R1["x1 (1.0, 2.0)"]
        R2["x2 (1.5, 2.5)"]
        R3["x3 (2.0, 1.5)"]
        R4["synthetic (1.2, 2.2)"]
    end
    SMOTE --> Result
```

### Sampling Strategies 比较

**Random Oversampling**：复制 minority samples 直到匹配 majority count。
- 优点：简单，无信息损失
- 缺点：完全重复导致过拟合，增加训练时间

**Random Undersampling**：移除 majority samples 直到匹配 minority count。
- 优点：训练快，简单
- 缺点：丢弃可能有用的 majority data，variance 更高

**SMOTE**：通过插值创建 synthetic minority samples。
- 优点：生成新数据点，相比 random oversampling 减少过拟合
- 缺点：可能在 decision boundary 附近创建 noisy samples，不考虑 majority class distribution

| Strategy | Data Changed | Risk | When to Use |
|----------|-------------|------|-------------|
| Oversample | Minority duplicated | Overfitting | 小数据集，中等 imbalance |
| Undersample | Majority removed | Information loss | 大数据集，希望训练快 |
| SMOTE | Synthetic minority added | Boundary noise | 中等 imbalance，有足够 minority samples 做 k-NN |

### Class Weights

不改变数据，而是改变模型对错误的看法。给 minority class 的误分类更高权重。

对于一个有 950 个 negative 和 50 个 positive samples 的 binary problem：
- Negative class weight = n_samples / (2 * n_negative) = 1000 / (2 * 950) = 0.526
- Positive class weight = n_samples / (2 * n_positive) = 1000 / (2 * 50) = 10.0

Positive class 得到 19 倍权重。误分类一个 positive sample 的代价等于误分类 19 个 negative samples。模型被迫关注 minority class。

在 logistic regression 中，这会修改 loss function：

```
weighted_loss = -sum(w_i * [y_i * log(p_i) + (1-y_i) * log(1-p_i)])
```

其中 w_i 取决于 sample i 的 class。

Class weights 在期望上等价于 oversampling，但不创建新数据点。这让它们更快，并避免 duplicated samples 的过拟合风险。

### Threshold Tuning

多数 classifiers 输出概率。默认 threshold 是 0.5：如果 P(positive) >= 0.5，预测 positive。但 0.5 是任意的。当 classes 不平衡时，最优 threshold 通常低很多。

流程：
1. 训练模型
2. 在 validation set 上获取 predicted probabilities
3. 从 0.0 到 1.0 扫描 thresholds
4. 在每个 threshold 下计算 F1（或你选择的 metric）
5. 选择最大化 metric 的 threshold

```mermaid
flowchart LR
    A[Model] --> B[Predict Probabilities]
    B --> C[Sweep Thresholds 0.0 to 1.0]
    C --> D[Compute F1 at Each]
    D --> E[Pick Best Threshold]
    E --> F[Use in Production]
```

模型可能对一笔欺诈交易输出 P(fraud) = 0.15。在 threshold 0.5 下，它被分类为 not fraud。在 threshold 0.10 下，它被正确抓住。Probability calibration 不如 ranking 重要，只要 fraud 概率高于 non-fraud，就存在一个能分离它们的 threshold。

### Cost-Sensitive Learning

这是 class weights 的泛化。不要使用统一成本，而是指定具体 misclassification costs：

| | Predict Positive | Predict Negative |
|--|---|---|
| Actually Positive | 0 (correct) | C_FN = 100 |
| Actually Negative | C_FP = 1 | 0 (correct) |

漏掉一笔欺诈交易（FN）的成本是假警报（FP）的 100 倍。模型优化 total cost，而不是 total error count。

当你能估计真实世界成本时，这是最有原则的方法。漏诊癌症与导致额外活检的 false alarm 有非常不同的成本。显式设置这些成本会迫使模型做正确权衡。

### Decision Flowchart

```mermaid
flowchart TD
    A[Start: Imbalanced Dataset] --> B{How imbalanced?}
    B -->|"< 70/30"| C["Mild: try class weights first"]
    B -->|"70/30 to 95/5"| D["Moderate: SMOTE + class weights"]
    B -->|"> 95/5"| E["Severe: combine multiple strategies"]
    C --> F{Enough data?}
    D --> F
    E --> F
    F -->|"< 1000 samples"| G["Oversample or SMOTE, avoid undersampling"]
    F -->|"1000-10000"| H["SMOTE + threshold tuning"]
    F -->|"> 10000"| I["Undersampling OK, or class weights"]
    G --> J[Train + Evaluate with F1/AUPRC]
    H --> J
    I --> J
    J --> K{Recall high enough?}
    K -->|No| L[Lower threshold]
    K -->|Yes| M{Precision acceptable?}
    M -->|No| N[Raise threshold or add features]
    M -->|Yes| O[Ship it]
```

## 构建它

### 第 1 步：生成 imbalanced dataset

```python
import numpy as np


def make_imbalanced_data(n_majority=950, n_minority=50, seed=42):
    rng = np.random.RandomState(seed)

    X_maj = rng.randn(n_majority, 2) * 1.0 + np.array([0.0, 0.0])
    X_min = rng.randn(n_minority, 2) * 0.8 + np.array([2.5, 2.5])

    X = np.vstack([X_maj, X_min])
    y = np.concatenate([np.zeros(n_majority), np.ones(n_minority)])

    shuffle_idx = rng.permutation(len(y))
    return X[shuffle_idx], y[shuffle_idx]
```

### 第 2 步：从零实现 SMOTE

```python
def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))


def find_k_neighbors(X, idx, k):
    distances = []
    for i in range(len(X)):
        if i == idx:
            continue
        d = euclidean_distance(X[idx], X[i])
        distances.append((i, d))
    distances.sort(key=lambda x: x[1])
    return [d[0] for d in distances[:k]]


def smote(X_minority, k=5, n_synthetic=100, seed=42):
    rng = np.random.RandomState(seed)
    n_samples = len(X_minority)
    k = min(k, n_samples - 1)
    synthetic = []

    for _ in range(n_synthetic):
        idx = rng.randint(0, n_samples)
        neighbors = find_k_neighbors(X_minority, idx, k)
        neighbor_idx = neighbors[rng.randint(0, len(neighbors))]
        t = rng.random()
        new_point = X_minority[idx] + t * (X_minority[neighbor_idx] - X_minority[idx])
        synthetic.append(new_point)

    return np.array(synthetic)
```

### 第 3 步：Random oversampling 和 undersampling

```python
def random_oversample(X, y, seed=42):
    rng = np.random.RandomState(seed)
    classes, counts = np.unique(y, return_counts=True)
    max_count = counts.max()

    X_resampled = list(X)
    y_resampled = list(y)

    for cls, count in zip(classes, counts):
        if count < max_count:
            cls_indices = np.where(y == cls)[0]
            n_needed = max_count - count
            chosen = rng.choice(cls_indices, size=n_needed, replace=True)
            X_resampled.extend(X[chosen])
            y_resampled.extend(y[chosen])

    X_out = np.array(X_resampled)
    y_out = np.array(y_resampled)
    shuffle = rng.permutation(len(y_out))
    return X_out[shuffle], y_out[shuffle]


def random_undersample(X, y, seed=42):
    rng = np.random.RandomState(seed)
    classes, counts = np.unique(y, return_counts=True)
    min_count = counts.min()

    X_resampled = []
    y_resampled = []

    for cls in classes:
        cls_indices = np.where(y == cls)[0]
        chosen = rng.choice(cls_indices, size=min_count, replace=False)
        X_resampled.extend(X[chosen])
        y_resampled.extend(y[chosen])

    X_out = np.array(X_resampled)
    y_out = np.array(y_resampled)
    shuffle = rng.permutation(len(y_out))
    return X_out[shuffle], y_out[shuffle]
```

### 第 4 步：带 class weights 的 Logistic regression

```python
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def logistic_regression_weighted(X, y, weights, lr=0.01, epochs=200):
    n_samples, n_features = X.shape
    w = np.zeros(n_features)
    b = 0.0

    for _ in range(epochs):
        z = X @ w + b
        pred = sigmoid(z)
        error = pred - y
        weighted_error = error * weights

        gradient_w = (X.T @ weighted_error) / n_samples
        gradient_b = np.mean(weighted_error)

        w -= lr * gradient_w
        b -= lr * gradient_b

    return w, b


def compute_class_weights(y):
    classes, counts = np.unique(y, return_counts=True)
    n_samples = len(y)
    n_classes = len(classes)
    weight_map = {}
    for cls, count in zip(classes, counts):
        weight_map[cls] = n_samples / (n_classes * count)
    return np.array([weight_map[yi] for yi in y])
```

### 第 5 步：Threshold tuning

```python
def find_optimal_threshold(y_true, y_probs, metric="f1"):
    best_threshold = 0.5
    best_score = -1.0

    for threshold in np.arange(0.05, 0.96, 0.01):
        y_pred = (y_probs >= threshold).astype(int)
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))

        if metric == "f1":
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        elif metric == "recall":
            score = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        elif metric == "precision":
            score = tp / (tp + fp) if (tp + fp) > 0 else 0.0

        if score > best_score:
            best_score = score
            best_threshold = threshold

    return best_threshold, best_score
```

### 第 6 步：Evaluation functions

```python
def confusion_matrix_values(y_true, y_pred):
    tp = np.sum((y_pred == 1) & (y_true == 1))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    return tp, tn, fp, fn


def compute_metrics(y_true, y_pred):
    tp, tn, fp, fn = confusion_matrix_values(y_true, y_pred)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    denom = np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    mcc = (tp * tn - fp * fn) / denom if denom > 0 else 0.0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mcc": mcc,
    }
```

### 第 7 步：比较所有方法

```python
X, y = make_imbalanced_data(950, 50, seed=42)
split = int(0.8 * len(y))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Baseline: no treatment
w_base, b_base = logistic_regression_weighted(
    X_train, y_train, np.ones(len(y_train)), lr=0.1, epochs=300
)
probs_base = sigmoid(X_test @ w_base + b_base)
preds_base = (probs_base >= 0.5).astype(int)

# Oversampled
X_over, y_over = random_oversample(X_train, y_train)
w_over, b_over = logistic_regression_weighted(
    X_over, y_over, np.ones(len(y_over)), lr=0.1, epochs=300
)
preds_over = (sigmoid(X_test @ w_over + b_over) >= 0.5).astype(int)

# SMOTE
minority_mask = y_train == 1
X_minority = X_train[minority_mask]
synthetic = smote(X_minority, k=5, n_synthetic=len(y_train) - 2 * int(minority_mask.sum()))
X_smote = np.vstack([X_train, synthetic])
y_smote = np.concatenate([y_train, np.ones(len(synthetic))])
w_sm, b_sm = logistic_regression_weighted(
    X_smote, y_smote, np.ones(len(y_smote)), lr=0.1, epochs=300
)
preds_smote = (sigmoid(X_test @ w_sm + b_sm) >= 0.5).astype(int)

# Class weights
sample_weights = compute_class_weights(y_train)
w_cw, b_cw = logistic_regression_weighted(
    X_train, y_train, sample_weights, lr=0.1, epochs=300
)
probs_cw = sigmoid(X_test @ w_cw + b_cw)
preds_cw = (probs_cw >= 0.5).astype(int)

# Threshold tuning (tune on held-out validation set, not test set)
probs_val = sigmoid(X_val @ w_cw + b_cw)
best_thresh, best_f1 = find_optimal_threshold(y_val, probs_val, metric="f1")
preds_thresh = (probs_cw >= best_thresh).astype(int)
```

代码文件会在单个脚本中运行这些内容并打印结果。

## 使用它

使用 scikit-learn 和 imbalanced-learn，这些技术都是一行：

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)

model_weighted = LogisticRegression(class_weight="balanced")
model_weighted.fit(X_train, y_train)
print(classification_report(y_test, model_weighted.predict(X_test)))

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
model_smote = LogisticRegression()
model_smote.fit(X_resampled, y_resampled)
print(classification_report(y_test, model_smote.predict(X_test)))

pipeline = Pipeline([
    ("smote", SMOTE()),
    ("model", LogisticRegression(class_weight="balanced")),
])
pipeline.fit(X_train, y_train)
print(classification_report(y_test, pipeline.predict(X_test)))
```

从零实现展示了每项技术具体做什么。SMOTE 只是 minority class 上的 k-NN interpolation。Class weights 只是乘以 loss。Threshold tuning 只是遍历 cutoffs 的 for-loop。没有魔法。

## 交付它

本课会产出：
- `outputs/skill-imbalanced-data.md` -- 处理 imbalanced classification problems 的决策清单

## 练习

1. **Borderline-SMOTE**：修改 SMOTE 实现，只为靠近 decision boundary 的 minority points 生成 synthetic samples（这些点的 k-nearest neighbors 中包含 majority class samples）。在 classes 重叠的数据集上与标准 SMOTE 比较结果。

2. **Cost matrix optimization**：实现 cost-sensitive learning，其中 cost matrix 是参数。创建一个函数，接收 cost matrix 并返回最小化 expected cost 的最优预测。用不同 cost ratios（1:10、1:100、1:1000）测试，并绘制 precision-recall tradeoff 如何变化。

3. **Threshold calibration**：实现 Platt scaling（在模型 raw outputs 上拟合 logistic regression 以产生 calibrated probabilities）。比较校准前后的 precision-recall curve。展示 calibration 不会改变 ranking（AUC 不变），但会让概率更有意义。

4. **Ensemble with balanced bagging**：训练多个模型，每个模型都在 balanced bootstrap sample（所有 minority + majority 的随机子集）上训练。平均它们的预测。把这种方法与单个 SMOTE 模型比较。测量性能和跨 runs 的 variance。

5. **Imbalance ratio experiment**：取一个平衡数据集，逐渐增加 imbalance ratio（50/50、70/30、90/10、95/5、99/1）。对每个 ratio，分别用和不用 SMOTE 训练。绘制两种方法的 F1 vs imbalance ratio。SMOTE 从哪个 ratio 开始带来明显差异？

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|----------------|----------------------|
| Class imbalance | “一个 class 样本多得多” | 数据集中 classes 分布显著偏斜，导致模型偏向 majority class |
| SMOTE | “Synthetic oversampling” | 通过在现有 minority samples 和其 k-nearest minority neighbors 之间插值，创建新 minority samples |
| Class weights | “让 rare class 的错误更贵” | 用 class-specific weights 乘以 loss function，使模型更重罚 minority misclassification |
| Threshold tuning | “移动 decision boundary” | 把 classification probability cutoff 从默认 0.5 改成优化目标 metric 的值 |
| Precision-recall tradeoff | “两者无法兼得” | 降低 threshold 会抓住更多 positives（higher recall），但也 flag 更多 false positives（lower precision），反之亦然 |
| AUPRC | “PR curve 下面积” | 把 precision-recall curve 汇总成单一数字；classes 严重不平衡时比 AUC-ROC 更有信息量 |
| Matthews Correlation Coefficient | “平衡 metric” | predicted 和 actual labels 之间的 correlation；只有模型在两个 classes 上都表现好时才给高分 |
| Cost-sensitive learning | “不同错误成本不同” | 把真实世界 misclassification costs 纳入 training objective，使模型优化 total cost 而非 error count |
| Random oversampling | “复制 minority” | 重复 minority class samples 以平衡 class counts；简单但有过拟合到重复点的风险 |

## 延伸阅读

- [SMOTE: Synthetic Minority Over-sampling Technique (Chawla et al., 2002)](https://arxiv.org/abs/1106.1813) -- SMOTE 原始论文，仍是 imbalanced learning 中引用最多的工作
- [Learning from Imbalanced Data (He & Garcia, 2009)](https://ieeexplore.ieee.org/document/5128907) -- 覆盖 sampling、cost-sensitive 和 algorithmic approaches 的综合综述
- [imbalanced-learn documentation](https://imbalanced-learn.org/stable/) -- Python 库，包含 SMOTE variants、undersampling strategies 和 pipeline integration
- [The Precision-Recall Plot Is More Informative than the ROC Plot (Saito & Rehmsmeier, 2015)](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0118432) -- 不平衡问题中何时、为何偏好 PR curves 而不是 ROC curves
