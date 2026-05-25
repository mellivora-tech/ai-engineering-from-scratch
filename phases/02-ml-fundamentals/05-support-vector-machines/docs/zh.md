# 支持向量机

> 在两个类别之间找到最宽的街道。这就是全部思想。

**类型：** 构建
**语言：** Python
**前置要求：** 阶段 1（第 08 课优化、第 14 课范数与距离、第 18 课凸优化）
**时间：** ~90 分钟

## 学习目标

- 在 primal formulation 上使用 hinge loss 和 gradient descent，从零实现 linear SVM
- 解释 maximum margin principle，并从训练好的模型中识别 support vectors
- 比较 linear、polynomial 和 RBF kernels，并解释 kernel trick 如何避免显式高维映射
- 评估 C parameter 在 margin width 和 classification errors 之间控制的权衡

## 问题

你有两类数据点，需要画一条线（或 hyperplane）把它们分开。无穷多条线都可能可行。你应该选哪一条？

选 margin 最大的那条。Margin 是 decision boundary 和两侧最近数据点之间的距离。Margin 越宽，classifier 越有信心，对未见数据的泛化也越好。

这个直觉引出了 Support Vector Machines，这是 ML 中数学上最优雅的算法之一。SVMs 在 deep learning 之前曾是主导性的分类方法，并且在小数据集、高维数据，以及需要原则清晰、理解充分且有理论保证的模型时，仍然是最佳选择。

SVMs 直接连接阶段 1：优化是 convex 的（第 18 课），margin 用 norms 衡量（第 14 课），kernel trick 利用 dot products 处理非线性边界，却永远不需要在高维空间中显式计算。

## 概念

### Maximum margin classifier

给定 labels y_i in {-1, +1}、feature vectors x_i 的线性可分数据，我们想找一个 hyperplane w^T x + b = 0 来分隔类别。

点 x_i 到 hyperplane 的距离是：

```
distance = |w^T x_i + b| / ||w||
```

对于正确分类的点：y_i * (w^T x_i + b) > 0。Margin 是 hyperplane 到两侧最近点距离的两倍。

```mermaid
graph LR
    subgraph Margin
        direction TB
        A["w^T x + b = +1"] ~~~ B["w^T x + b = 0"] ~~~ C["w^T x + b = -1"]
    end
    D["+ class points"] --> A
    E["- class points"] --> C
    B --- F["Decision boundary"]
```

优化问题：

```
maximize    2 / ||w||     (the margin width)
subject to  y_i * (w^T x_i + b) >= 1  for all i
```

等价地（最小化 ||w||^2 更容易优化）：

```
minimize    (1/2) ||w||^2
subject to  y_i * (w^T x_i + b) >= 1  for all i
```

这是一个 convex quadratic program。它有唯一的 global solution。正好落在 margin boundaries 上的数据点（也就是 y_i * (w^T x_i + b) = 1 的点）就是 support vectors。它们是唯一决定 decision boundary 的点。移动或删除任何非 support-vector 点，边界都不会改变。

### Support vectors：关键少数

```mermaid
graph TD
    subgraph Classification
        SV1["Support Vector (+ class)<br>y(w'x+b) = 1"] --- DB["Decision Boundary<br>w'x+b = 0"]
        DB --- SV2["Support Vector (- class)<br>y(w'x+b) = 1"]
    end
    O1["Other + points<br>(do not affect boundary)"] -.-> SV1
    O2["Other - points<br>(do not affect boundary)"] -.-> SV2
```

大多数训练点都无关紧要。只有 support vectors 重要。这就是为什么 SVMs 在预测时很节省内存：你只需要存储 support vectors，不需要存储完整训练集。

Support vectors 的数量也给出了 generalization error 的一个界。相对于数据集大小，support vectors 越少，泛化越好。

### Soft margin：用 C parameter 处理噪声

真实数据很少完美可分。有些点可能在边界错误的一侧，或位于 margin 内部。Soft margin formulation 通过引入 slack variables 来允许违规。

```
minimize    (1/2) ||w||^2 + C * sum(xi_i)
subject to  y_i * (w^T x_i + b) >= 1 - xi_i
            xi_i >= 0  for all i
```

Slack variable xi_i 衡量点 i 违反 margin 的程度。C 控制权衡：

| C value | Behavior |
|---------|----------|
| Large C | 严重惩罚违规。Margin 窄，误分类更少。容易过拟合 |
| Small C | 允许更多违规。Margin 宽，误分类更多。容易欠拟合 |

C 是倒置的 regularization strength。Large C = less regularization。Small C = more regularization。

### Hinge loss：SVM 的 loss function

Soft margin SVM 可以改写成无约束优化：

```
minimize    (1/2) ||w||^2 + C * sum(max(0, 1 - y_i * (w^T x_i + b)))
```

项 max(0, 1 - y_i * f(x_i)) 就是 hinge loss。当点被正确分类且位于 margin 外时，它为零。当点在 margin 内或被误分类时，它是线性的。

```
Hinge loss for a single point:

loss
  |
  | \
  |  \
  |   \
  |    \
  |     \_______________
  |
  +-----|-----|-------->  y * f(x)
       0     1

Zero loss when y*f(x) >= 1 (correctly classified, outside margin).
Linear penalty when y*f(x) < 1.
```

与 logistic loss（logistic regression）比较：

```
Hinge:     max(0, 1 - y*f(x))          Hard cutoff at margin
Logistic:  log(1 + exp(-y*f(x)))        Smooth, never exactly zero
```

Hinge loss 会产生稀疏解（只有 support vectors 有非零贡献）。Logistic loss 会使用所有数据点。这让 SVMs 在预测时更节省内存。

### 使用 gradient descent 训练 linear SVM

你可以在 hinge loss 加 L2 regularization 上使用 gradient descent 训练 linear SVM，不需要求解带约束的 QP：

```
L(w, b) = (lambda/2) * ||w||^2 + (1/n) * sum(max(0, 1 - y_i * (w^T x_i + b)))

Gradient with respect to w:
  If y_i * (w^T x_i + b) >= 1:  dL/dw = lambda * w
  If y_i * (w^T x_i + b) < 1:   dL/dw = lambda * w - y_i * x_i

Gradient with respect to b:
  If y_i * (w^T x_i + b) >= 1:  dL/db = 0
  If y_i * (w^T x_i + b) < 1:   dL/db = -y_i
```

这叫 primal formulation。每个 epoch 的复杂度是 O(n * d)，其中 n 是样本数，d 是 features 数。对于大规模、稀疏、高维数据（text classification），这很快。

### Dual formulation 和 kernel trick

SVM 问题的 Lagrangian dual（来自阶段 1 第 18 课，KKT conditions）是：

```
maximize    sum(alpha_i) - (1/2) * sum_ij(alpha_i * alpha_j * y_i * y_j * (x_i . x_j))
subject to  0 <= alpha_i <= C
            sum(alpha_i * y_i) = 0
```

Dual 只涉及数据点之间的 dot products x_i . x_j。这是关键洞见。把每个 dot product 替换成 kernel function K(x_i, x_j)，SVM 就可以学习非线性边界，而不需要显式计算变换。

```
Linear kernel:      K(x, z) = x . z
Polynomial kernel:  K(x, z) = (x . z + c)^d
RBF (Gaussian):     K(x, z) = exp(-gamma * ||x - z||^2)
```

RBF kernel 会把数据映射到无限维空间。在输入空间中接近的点，kernel value 接近 1。相距很远的点，kernel value 接近 0。它可以学习任意平滑 decision boundary。

```mermaid
graph LR
    subgraph "Input Space (not separable)"
        A["Data points in 2D<br>circular boundary"]
    end
    subgraph "Feature Space (separable)"
        B["Data points in higher dim<br>linear boundary"]
    end
    A -->|"Kernel trick<br>K(x,z) = phi(x).phi(z)"| B
```

Kernel trick 在不真正进入高维空间的情况下，计算高维空间里的 dot product。对于 D 维中 degree 为 d 的 polynomial kernel，显式 feature space 有 O(D^d) 维。但 K(x, z) 只需 O(D) 时间计算。

### 用于回归的 SVM（SVR）

Support Vector Regression 会在数据周围拟合一条宽度为 epsilon 的管道。管道内的点 loss 为零。管道外的点受到线性惩罚。

```
minimize    (1/2) ||w||^2 + C * sum(xi_i + xi_i*)
subject to  y_i - (w^T x_i + b) <= epsilon + xi_i
            (w^T x_i + b) - y_i <= epsilon + xi_i*
            xi_i, xi_i* >= 0
```

Epsilon parameter 控制管道宽度。管道越宽 = support vectors 越少 = 拟合越平滑。管道越窄 = support vectors 越多 = 拟合越紧。

### SVMs 为什么输给 deep learning（以及何时仍然会赢）

SVMs 从 1990 年代末到 2010 年代初主导 ML。Deep learning 后来超越它们，原因有几个：

| Factor | SVMs | Deep learning |
|--------|------|---------------|
| Feature engineering | 需要 | 学习 features |
| Scalability | Kernel 下 O(n^2) 到 O(n^3) | 用 SGD 每个 epoch O(n) |
| Image/text/audio | 需要手工 features | 从原始数据学习 |
| Large datasets (>100k) | 慢 | 扩展性好 |
| GPU acceleration | 收益有限 | 大幅加速 |

SVMs 在这些情况下仍然会赢：
- 小数据集（几百到几千个样本）
- 高维稀疏数据（带 TF-IDF features 的文本）
- 需要数学保证时（margin bounds）
- 训练时间必须很短时（linear SVM 很快）
- Binary classification 且有清晰 margin structure
- Anomaly detection（one-class SVM）

## 构建它

### 第 1 步：Hinge loss 和 gradient

这是基础。计算一个 batch 的 hinge loss 和它的 gradient。

```python
def hinge_loss(X, y, w, b):
    n = len(X)
    total_loss = 0.0
    for i in range(n):
        margin = y[i] * (dot(w, X[i]) + b)
        total_loss += max(0.0, 1.0 - margin)
    return total_loss / n
```

### 第 2 步：通过 gradient descent 实现 Linear SVM

通过最小化 regularized hinge loss 来训练。不需要 QP solver。

```python
class LinearSVM:
    def __init__(self, lr=0.001, lambda_param=0.01, n_epochs=1000):
        self.lr = lr
        self.lambda_param = lambda_param
        self.n_epochs = n_epochs
        self.w = None
        self.b = 0.0

    def fit(self, X, y):
        n_features = len(X[0])
        self.w = [0.0] * n_features
        self.b = 0.0

        for epoch in range(self.n_epochs):
            for i in range(len(X)):
                margin = y[i] * (dot(self.w, X[i]) + self.b)
                if margin >= 1:
                    self.w = [wj - self.lr * self.lambda_param * wj
                              for wj in self.w]
                else:
                    self.w = [wj - self.lr * (self.lambda_param * wj - y[i] * X[i][j])
                              for j, wj in enumerate(self.w)]
                    self.b -= self.lr * (-y[i])

    def predict(self, X):
        return [1 if dot(self.w, x) + self.b >= 0 else -1 for x in X]
```

### 第 3 步：Kernel functions

实现 linear、polynomial 和 RBF kernels。

```python
def linear_kernel(x, z):
    return dot(x, z)

def polynomial_kernel(x, z, degree=3, c=1.0):
    return (dot(x, z) + c) ** degree

def rbf_kernel(x, z, gamma=0.5):
    diff = [xi - zi for xi, zi in zip(x, z)]
    return math.exp(-gamma * dot(diff, diff))
```

### 第 4 步：Margin 和 support vector 识别

训练后，识别哪些点是 support vectors，并计算 margin width。

```python
def find_support_vectors(X, y, w, b, tol=1e-3):
    support_vectors = []
    for i in range(len(X)):
        margin = y[i] * (dot(w, X[i]) + b)
        if abs(margin - 1.0) < tol:
            support_vectors.append(i)
    return support_vectors
```

完整实现和所有 demos 见 `code/svm.py`。

## 使用它

使用 scikit-learn：

```python
from sklearn.svm import SVC, LinearSVC, SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

clf = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(kernel="rbf", C=1.0, gamma="scale")),
])
clf.fit(X_train, y_train)
print(f"Accuracy: {clf.score(X_test, y_test):.4f}")
print(f"Support vectors: {clf['svm'].n_support_}")
```

重要：训练 SVM 前一定要 scale features。SVMs 对 feature magnitudes 很敏感，因为 margin 依赖 ||w||，未缩放 features 会扭曲几何结构。

对于大数据集，使用 `LinearSVC`（primal formulation，每个 epoch O(n)）而不是 `SVC`（dual formulation，O(n^2) 到 O(n^3)）：

```python
from sklearn.svm import LinearSVC

clf = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", LinearSVC(C=1.0, max_iter=10000)),
])
```

## 练习

1. 生成一个二维线性可分数据集。训练你的 LinearSVM 并识别 support vectors。验证 support vectors 就是离 decision boundary 最近的点。

2. 在 noisy dataset 上让 C 从 0.001 变化到 1000。为每个 C 值绘制 decision boundary。观察从宽 margin（欠拟合）到窄 margin（过拟合）的变化。

3. 创建一个类别边界是圆形（非线性）的数据集。展示 linear SVM 会失败。计算 RBF kernel matrix，并展示类别在 kernel-induced feature space 中变得可分。

4. 在同一数据集上比较 hinge loss 和 logistic loss。训练 linear SVM 和 logistic regression。统计有多少训练点参与每个模型的 decision boundary（support vectors vs all points）。

5. 实现 SVR（epsilon-insensitive loss）。拟合 y = sin(x) + noise。绘制预测周围的 epsilon tube，并高亮 support vectors（管道外的点）。

## 关键术语

| 术语 | 实际含义 |
|------|----------------------|
| Support vectors | 离 decision boundary 最近的训练点。唯一决定 hyperplane 的点 |
| Margin | Decision boundary 与最近 support vectors 之间的距离。SVMs 会最大化它 |
| Hinge loss | max(0, 1 - y*f(x))。正确分类且在 margin 外时为零，否则线性惩罚 |
| C parameter | Margin width 与 classification errors 之间的权衡。Large C = 窄 margin，small C = 宽 margin |
| Soft margin | 通过 slack variables 允许 margin violations 的 SVM formulation。处理不可分数据 |
| Kernel trick | 不显式映射到高维空间，却计算该空间中的 dot products |
| Linear kernel | K(x, z) = x . z。等价于标准 dot product。用于线性可分数据 |
| RBF kernel | K(x, z) = exp(-gamma * \|\|x-z\|\|^2)。映射到无限维。学习任意平滑边界 |
| Polynomial kernel | K(x, z) = (x . z + c)^d。映射到多项式组合构成的 feature space |
| Dual formulation | SVM 问题的改写形式，只依赖数据点之间的 dot products。使 kernels 成为可能 |
| SVR | Support Vector Regression。在数据周围拟合 epsilon-tube。管道内的点 loss 为零 |
| Slack variables | xi_i：衡量一个点违反 margin 的程度。正确分类且在 margin 外的点为零 |
| Maximum margin | 选择让各类别最近点距离最大的 hyperplane 的原则 |

## 延伸阅读

- [Vapnik: The Nature of Statistical Learning Theory (1995)](https://link.springer.com/book/10.1007/978-1-4757-3264-1) - SVMs 和统计学习的奠基文本
- [Cortes & Vapnik: Support-vector networks (1995)](https://link.springer.com/article/10.1007/BF00994018) - SVM 原始论文
- [Platt: Sequential Minimal Optimization (1998)](https://www.microsoft.com/en-us/research/publication/sequential-minimal-optimization-a-fast-algorithm-for-training-support-vector-machines/) - 让 SVM training 变得实用的 SMO 算法
- [scikit-learn SVM documentation](https://scikit-learn.org/stable/modules/svm.html) - 实用指南，包含实现细节
- [LIBSVM: A Library for Support Vector Machines](https://www.csie.ntu.edu.tw/~cjlin/libsvm/) - 大多数 SVM 实现背后的 C++ 库
