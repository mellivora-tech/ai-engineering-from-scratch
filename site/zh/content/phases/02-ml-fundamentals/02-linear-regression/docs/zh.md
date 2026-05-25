# 线性回归

> 线性回归是在数据中画出最合适的直线。它是机器学习里的 “hello world”。

**类型：** 构建
**语言：** Python
**前置要求：** 阶段 1（线性代数、微积分、优化），阶段 2 第 1 课
**时间：** ~90 分钟

## 学习目标

- 推导 mean squared error 的 gradient descent 更新规则，并从零实现线性回归
- 从计算复杂度和适用场景角度比较 gradient descent 与 normal equation
- 构建带 feature standardization 的多元线性回归模型，并解释学到的 weights
- 解释 Ridge regression（L2 regularization）如何通过惩罚大 weights 来防止过拟合

## 问题

你有一组数据：房屋面积和成交价格。你想根据新房子的面积预测它的价格。你可以在散点图上目测，但你需要的是一个公式。你需要一条最贴合数据的直线，这样输入任意面积，就能得到价格预测。

线性回归会给你这条线。更重要的是，它引入了完整的 ML 训练循环：定义模型，定义 cost function，优化 parameters。每个 ML 算法都遵循这个模式。先在最简单的案例里掌握它，你之后会在各处认出它。

这并不只适用于简单问题。线性回归被用于生产系统中的需求预测、A/B test 分析、金融建模，也常作为每个回归任务的 baseline。

## 概念

### 模型

线性回归假设输入（x）和输出（y）之间存在线性关系：

```
y = wx + b
```

- `w`（weight/slope）：x 增加 1 时，y 会变化多少
- `b`（bias/intercept）：当 x = 0 时，y 的值

对于多个输入（features），它扩展为：

```
y = w1*x1 + w2*x2 + ... + wn*xn + b
```

或者写成向量形式：`y = w^T * x + b`

目标：找到 w 和 b 的取值，让所有训练样本上的预测 y 尽可能接近真实 y。

### Cost Function（Mean Squared Error）

如何衡量“尽可能接近”？你需要一个单一数字来表示预测错得有多厉害。最常见选择是 Mean Squared Error（MSE）：

```
MSE = (1/n) * sum((y_predicted - y_actual)^2)
```

为什么平方？两个原因。第一，它对大错误的惩罚比小错误更重（误差 10 比误差 1 糟糕 100 倍，而不是 10 倍）。第二，平方函数处处平滑且可微，让优化更直接。

Cost function 会形成一个曲面。对于单个 weight w 和 bias b，MSE 曲面看起来像一个碗（凸抛物面）。碗底就是 MSE 最小的位置。训练就是找到这个碗底。

### Gradient Descent

Gradient descent 通过一步步向下走来找到碗底。

```mermaid
flowchart TD
    A[随机初始化 w 和 b] --> B[计算预测: y_hat = wx + b]
    B --> C[计算 cost: MSE]
    C --> D[计算 gradients: dMSE/dw, dMSE/db]
    D --> E[更新 parameters]
    E --> F{Cost 足够低了吗？}
    F -->|否| B
    F -->|是| G[完成：找到最优 w 和 b]
```

Gradients 告诉你两件事：每个 parameter 应该往哪个方向移动，以及移动多少。

对于 y_hat = wx + b 的 MSE：

```
dMSE/dw = (2/n) * sum((y_hat - y) * x)
dMSE/db = (2/n) * sum(y_hat - y)
```

更新规则：

```
w = w - learning_rate * dMSE/dw
b = b - learning_rate * dMSE/db
```

Learning rate 控制步长。太大：你会越过最小值并发散。太小：训练会耗时很久。常见初始值：0.01、0.001 或 0.0001。

### Normal Equation（闭式解）

对于线性回归，存在一个不需要任何迭代、可直接得到最优 weights 的公式：

```
w = (X^T * X)^(-1) * X^T * y
```

它通过矩阵求逆一步解出 w。它非常适合小数据集。对于大数据集（数百万行或数千个 features），gradient descent 更常用，因为矩阵求逆在 feature 数量上是 O(n^3)。

### 多元线性回归

有多个 features 时，模型变成：

```
y = w1*x1 + w2*x2 + ... + wn*xn + b
```

一切仍然相同：MSE 是 cost function，gradient descent 同时更新所有 weights。唯一的区别是你在拟合一个超平面，而不是一条直线。

Feature scaling 在这里很重要。如果一个 feature 范围是 0 到 1，另一个范围是 0 到 1,000,000，gradient descent 会很吃力，因为 cost surface 会被拉长。训练前要标准化 features（减去均值，再除以标准差）。

### 多项式回归

如果关系不是线性的怎么办？你仍然可以通过创建多项式 features 来使用线性回归：

```
y = w1*x + w2*x^2 + w3*x^3 + b
```

这仍然是“线性”回归，因为模型对 weights（w1、w2、w3）是线性的。你只是使用了 x 的非线性 features。

更高阶的多项式能拟合更复杂的曲线，但有过拟合风险。10 次多项式会穿过 10 个点数据集中的每个点，却可能在新数据上预测很差。

### R-Squared Score

MSE 告诉你错了多少，但这个数字依赖 y 的尺度。R-squared（R^2）给出一个与尺度无关的度量：

```
R^2 = 1 - (sum of squared residuals) / (sum of squared deviations from mean)
    = 1 - SS_res / SS_tot
```

- R^2 = 1.0：完美预测
- R^2 = 0.0：模型不比每次都预测均值更好
- R^2 < 0.0：模型比预测均值还差

### Regularization 预览（Ridge Regression）

当你有很多 features 时，模型可能通过分配很大的 weights 来过拟合。Ridge regression（L2 regularization）会加入惩罚项：

```
Cost = MSE + lambda * sum(w_i^2)
```

惩罚项会抑制大 weights。Hyperparameter lambda 控制权衡：lambda 越高，weights 越小，regularization 越强。后续课程会深入讲解。现在只需要知道它存在，以及它为什么有帮助。

## 构建它

### 第 1 步：生成样本数据

```python
import random
import math

random.seed(42)

TRUE_W = 3.0
TRUE_B = 7.0
N_SAMPLES = 100

X = [random.uniform(0, 10) for _ in range(N_SAMPLES)]
y = [TRUE_W * x + TRUE_B + random.gauss(0, 2.0) for x in X]

print(f"Generated {N_SAMPLES} samples")
print(f"True relationship: y = {TRUE_W}x + {TRUE_B} (+ noise)")
print(f"First 5 points: {[(round(X[i], 2), round(y[i], 2)) for i in range(5)]}")
```

### 第 2 步：用 gradient descent 从零实现线性回归

```python
class LinearRegression:
    def __init__(self, learning_rate=0.01):
        self.w = 0.0
        self.b = 0.0
        self.lr = learning_rate
        self.cost_history = []

    def predict(self, X):
        return [self.w * x + self.b for x in X]

    def compute_cost(self, X, y):
        predictions = self.predict(X)
        n = len(y)
        cost = sum((pred - actual) ** 2 for pred, actual in zip(predictions, y)) / n
        return cost

    def compute_gradients(self, X, y):
        predictions = self.predict(X)
        n = len(y)
        dw = (2 / n) * sum((pred - actual) * x for pred, actual, x in zip(predictions, y, X))
        db = (2 / n) * sum(pred - actual for pred, actual in zip(predictions, y))
        return dw, db

    def fit(self, X, y, epochs=1000, print_every=200):
        for epoch in range(epochs):
            dw, db = self.compute_gradients(X, y)
            self.w -= self.lr * dw
            self.b -= self.lr * db
            cost = self.compute_cost(X, y)
            self.cost_history.append(cost)
            if epoch % print_every == 0:
                print(f"  Epoch {epoch:4d} | Cost: {cost:.4f} | w: {self.w:.4f} | b: {self.b:.4f}")
        return self

    def r_squared(self, X, y):
        predictions = self.predict(X)
        y_mean = sum(y) / len(y)
        ss_res = sum((actual - pred) ** 2 for actual, pred in zip(y, predictions))
        ss_tot = sum((actual - y_mean) ** 2 for actual in y)
        return 1 - (ss_res / ss_tot)


print("=== Training Linear Regression (Gradient Descent) ===")
model = LinearRegression(learning_rate=0.005)
model.fit(X, y, epochs=1000, print_every=200)
print(f"\nLearned: y = {model.w:.4f}x + {model.b:.4f}")
print(f"True:    y = {TRUE_W}x + {TRUE_B}")
print(f"R-squared: {model.r_squared(X, y):.4f}")
```

### 第 3 步：Normal equation（闭式解）

```python
class LinearRegressionNormal:
    def __init__(self):
        self.w = 0.0
        self.b = 0.0

    def fit(self, X, y):
        n = len(X)
        x_mean = sum(X) / n
        y_mean = sum(y) / n
        numerator = sum((X[i] - x_mean) * (y[i] - y_mean) for i in range(n))
        denominator = sum((X[i] - x_mean) ** 2 for i in range(n))
        self.w = numerator / denominator
        self.b = y_mean - self.w * x_mean
        return self

    def predict(self, X):
        return [self.w * x + self.b for x in X]

    def r_squared(self, X, y):
        predictions = self.predict(X)
        y_mean = sum(y) / len(y)
        ss_res = sum((actual - pred) ** 2 for actual, pred in zip(y, predictions))
        ss_tot = sum((actual - y_mean) ** 2 for actual in y)
        return 1 - (ss_res / ss_tot)


print("\n=== Normal Equation (Closed-Form) ===")
model_normal = LinearRegressionNormal()
model_normal.fit(X, y)
print(f"Learned: y = {model_normal.w:.4f}x + {model_normal.b:.4f}")
print(f"R-squared: {model_normal.r_squared(X, y):.4f}")
```

### 第 4 步：多元线性回归

```python
class MultipleLinearRegression:
    def __init__(self, n_features, learning_rate=0.01):
        self.weights = [0.0] * n_features
        self.bias = 0.0
        self.lr = learning_rate
        self.cost_history = []

    def predict_single(self, x):
        return sum(w * xi for w, xi in zip(self.weights, x)) + self.bias

    def predict(self, X):
        return [self.predict_single(x) for x in X]

    def compute_cost(self, X, y):
        predictions = self.predict(X)
        n = len(y)
        return sum((pred - actual) ** 2 for pred, actual in zip(predictions, y)) / n

    def fit(self, X, y, epochs=1000, print_every=200):
        n = len(y)
        n_features = len(X[0])
        for epoch in range(epochs):
            predictions = self.predict(X)
            errors = [pred - actual for pred, actual in zip(predictions, y)]
            for j in range(n_features):
                grad = (2 / n) * sum(errors[i] * X[i][j] for i in range(n))
                self.weights[j] -= self.lr * grad
            grad_b = (2 / n) * sum(errors)
            self.bias -= self.lr * grad_b
            cost = self.compute_cost(X, y)
            self.cost_history.append(cost)
            if epoch % print_every == 0:
                print(f"  Epoch {epoch:4d} | Cost: {cost:.4f}")
        return self

    def r_squared(self, X, y):
        predictions = self.predict(X)
        y_mean = sum(y) / len(y)
        ss_res = sum((actual - pred) ** 2 for actual, pred in zip(y, predictions))
        ss_tot = sum((actual - y_mean) ** 2 for actual in y)
        return 1 - (ss_res / ss_tot)


random.seed(42)
N = 100
X_multi = []
y_multi = []
for _ in range(N):
    size = random.uniform(500, 3000)
    bedrooms = random.randint(1, 5)
    age = random.uniform(0, 50)
    price = 50 * size + 10000 * bedrooms - 1000 * age + 50000 + random.gauss(0, 20000)
    X_multi.append([size, bedrooms, age])
    y_multi.append(price)


def standardize(X):
    n_features = len(X[0])
    means = [sum(X[i][j] for i in range(len(X))) / len(X) for j in range(n_features)]
    stds = []
    for j in range(n_features):
        variance = sum((X[i][j] - means[j]) ** 2 for i in range(len(X))) / len(X)
        stds.append(variance ** 0.5)
    X_scaled = []
    for i in range(len(X)):
        row = [(X[i][j] - means[j]) / stds[j] if stds[j] > 0 else 0 for j in range(n_features)]
        X_scaled.append(row)
    return X_scaled, means, stds


y_mean_val = sum(y_multi) / len(y_multi)
y_std_val = (sum((yi - y_mean_val) ** 2 for yi in y_multi) / len(y_multi)) ** 0.5
y_scaled = [(yi - y_mean_val) / y_std_val for yi in y_multi]

X_scaled, x_means, x_stds = standardize(X_multi)

print("\n=== Multiple Linear Regression (3 features) ===")
print("Features: house size, bedrooms, age")
multi_model = MultipleLinearRegression(n_features=3, learning_rate=0.01)
multi_model.fit(X_scaled, y_scaled, epochs=1000, print_every=200)

print(f"\nWeights (standardized): {[round(w, 4) for w in multi_model.weights]}")
print(f"Bias (standardized): {multi_model.bias:.4f}")
print(f"R-squared: {multi_model.r_squared(X_scaled, y_scaled):.4f}")
```

### 第 5 步：多项式回归

```python
class PolynomialRegression:
    def __init__(self, degree, learning_rate=0.01):
        self.degree = degree
        self.weights = [0.0] * degree
        self.bias = 0.0
        self.lr = learning_rate

    def make_features(self, X):
        return [[x ** (d + 1) for d in range(self.degree)] for x in X]

    def predict(self, X):
        features = self.make_features(X)
        return [sum(w * f for w, f in zip(self.weights, row)) + self.bias for row in features]

    def fit(self, X, y, epochs=1000, print_every=200):
        features = self.make_features(X)
        n = len(y)
        for epoch in range(epochs):
            predictions = [sum(w * f for w, f in zip(self.weights, row)) + self.bias for row in features]
            errors = [pred - actual for pred, actual in zip(predictions, y)]
            for j in range(self.degree):
                grad = (2 / n) * sum(errors[i] * features[i][j] for i in range(n))
                self.weights[j] -= self.lr * grad
            grad_b = (2 / n) * sum(errors)
            self.bias -= self.lr * grad_b
            if epoch % print_every == 0:
                cost = sum(e ** 2 for e in errors) / n
                print(f"  Epoch {epoch:4d} | Cost: {cost:.6f}")
        return self

    def r_squared(self, X, y):
        predictions = self.predict(X)
        y_mean = sum(y) / len(y)
        ss_res = sum((actual - pred) ** 2 for actual, pred in zip(y, predictions))
        ss_tot = sum((actual - y_mean) ** 2 for actual in y)
        return 1 - (ss_res / ss_tot)


random.seed(42)
X_poly = [x / 10.0 for x in range(0, 50)]
y_poly = [0.5 * x ** 2 - 2 * x + 3 + random.gauss(0, 1.0) for x in X_poly]

x_max = max(abs(x) for x in X_poly)
X_poly_norm = [x / x_max for x in X_poly]
y_poly_mean = sum(y_poly) / len(y_poly)
y_poly_std = (sum((yi - y_poly_mean) ** 2 for yi in y_poly) / len(y_poly)) ** 0.5
y_poly_norm = [(yi - y_poly_mean) / y_poly_std for yi in y_poly]

print("\n=== Polynomial Regression (degree 2 vs degree 5) ===")
print("True relationship: y = 0.5x^2 - 2x + 3")

print("\nDegree 2:")
poly2 = PolynomialRegression(degree=2, learning_rate=0.1)
poly2.fit(X_poly_norm, y_poly_norm, epochs=2000, print_every=500)
print(f"  R-squared: {poly2.r_squared(X_poly_norm, y_poly_norm):.4f}")

print("\nDegree 5:")
poly5 = PolynomialRegression(degree=5, learning_rate=0.1)
poly5.fit(X_poly_norm, y_poly_norm, epochs=2000, print_every=500)
print(f"  R-squared: {poly5.r_squared(X_poly_norm, y_poly_norm):.4f}")

print("\nDegree 2 fits the true curve well. Degree 5 fits training data slightly better")
print("but risks overfitting on new data.")
```

### 第 6 步：Ridge regression（L2 regularization）

```python
class RidgeRegression:
    def __init__(self, n_features, learning_rate=0.01, alpha=1.0):
        self.weights = [0.0] * n_features
        self.bias = 0.0
        self.lr = learning_rate
        self.alpha = alpha

    def predict_single(self, x):
        return sum(w * xi for w, xi in zip(self.weights, x)) + self.bias

    def predict(self, X):
        return [self.predict_single(x) for x in X]

    def fit(self, X, y, epochs=1000, print_every=200):
        n = len(y)
        n_features = len(X[0])
        for epoch in range(epochs):
            predictions = self.predict(X)
            errors = [pred - actual for pred, actual in zip(predictions, y)]
            mse = sum(e ** 2 for e in errors) / n
            reg_term = self.alpha * sum(w ** 2 for w in self.weights)
            cost = mse + reg_term
            for j in range(n_features):
                grad = (2 / n) * sum(errors[i] * X[i][j] for i in range(n))
                grad += 2 * self.alpha * self.weights[j]
                self.weights[j] -= self.lr * grad
            grad_b = (2 / n) * sum(errors)
            self.bias -= self.lr * grad_b
            if epoch % print_every == 0:
                print(f"  Epoch {epoch:4d} | Cost: {cost:.4f} | L2 penalty: {reg_term:.4f}")
        return self


print("\n=== Ridge Regression (L2 Regularization) ===")
print("Same data as multiple regression, with alpha=0.1")
ridge = RidgeRegression(n_features=3, learning_rate=0.01, alpha=0.1)
ridge.fit(X_scaled, y_scaled, epochs=1000, print_every=200)
print(f"\nRidge weights: {[round(w, 4) for w in ridge.weights]}")
print(f"Plain weights: {[round(w, 4) for w in multi_model.weights]}")
print("Ridge weights are smaller (shrunk toward zero) due to the L2 penalty.")
```

## 使用它

现在用 scikit-learn 做同样的事。生产中你真正会用的是它。

```python
from sklearn.linear_model import LinearRegression as SklearnLR
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

np.random.seed(42)
X_sk = np.random.uniform(0, 10, (100, 1))
y_sk = 3.0 * X_sk.squeeze() + 7.0 + np.random.normal(0, 2.0, 100)

X_train, X_test, y_train, y_test = train_test_split(X_sk, y_sk, test_size=0.2, random_state=42)

lr = SklearnLR()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)

print("=== Scikit-learn Linear Regression ===")
print(f"Coefficient (w): {lr.coef_[0]:.4f}")
print(f"Intercept (b): {lr.intercept_:.4f}")
print(f"R-squared (test): {r2_score(y_test, y_pred):.4f}")
print(f"MSE (test): {mean_squared_error(y_test, y_pred):.4f}")

poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly_sk = poly.fit_transform(X_train)
X_poly_test = poly.transform(X_test)

lr_poly = SklearnLR()
lr_poly.fit(X_poly_sk, y_train)
print(f"\nPolynomial degree 2 R-squared: {r2_score(y_test, lr_poly.predict(X_poly_test)):.4f}")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

ridge = Ridge(alpha=1.0)
ridge.fit(X_train_scaled, y_train)
print(f"Ridge R-squared: {r2_score(y_test, ridge.predict(X_test_scaled)):.4f}")
print(f"Ridge coefficient: {ridge.coef_[0]:.4f}")
```

你的从零实现和 scikit-learn 会产生相同结果。区别是：scikit-learn 会处理边界情况、数值稳定性和性能优化。生产中使用库。从零版本用于理解发生了什么。

## 交付它

本课会产出：
- `outputs/skill-regression.md` - 一个 skill，用来根据问题选择合适的回归方法

## 练习

1. 实现 batch gradient descent、stochastic gradient descent（SGD）和 mini-batch gradient descent。在同一数据集上比较收敛速度。哪个收敛最快？哪个 cost 曲线最平滑？
2. 从三次函数生成数据（y = ax^3 + bx^2 + cx + d + noise）。拟合 1、3 和 10 次多项式。比较训练 R^2 和测试 R^2。从哪个次数开始，过拟合变得明显？
3. 实现 Lasso regression（L1 regularization：penalty = alpha * sum(|w_i|）。在多 feature 房价数据上训练。比较哪些 weights 变成 0，以及它与 Ridge 的差异。为什么 L1 会产生稀疏解，而 L2 不会？

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|----------------|----------------------|
| Linear regression | “穿过数据画一条线” | 找到 weight w 和 bias b，使 wx+b 与真实 y 值之间的平方差之和最小 |
| Cost function | “模型有多糟” | 一个把模型 parameters 映射为单一数字的函数，用来衡量预测误差，并由优化过程最小化 |
| Mean squared error | “平方误差的平均值” | (1/n) * sum of (predicted - actual)^2，对大错误施加不成比例的惩罚 |
| Gradient descent | “往山下走” | 使用偏导数，沿着降低 cost function 的方向迭代调整 parameters |
| Learning rate | “步长” | 控制每一步 gradient descent 中 parameters 改变量的标量 |
| Normal equation | “直接解出来” | 闭式解 w = (X^T X)^-1 X^T y，不需要迭代即可得到最优 weights |
| R-squared | “拟合有多好” | y 中被模型解释的方差比例，范围从负无穷到 1.0 |
| Feature scaling | “让 features 可比较” | 把 features 转换到相近范围（例如零均值、单位方差），让 gradient descent 更快收敛 |
| Regularization | “惩罚复杂度” | 向 cost function 添加项来缩小 weights，防止过拟合 |
| Ridge regression | “L2 regularization” | 在线性回归的 MSE 中加入 lambda * sum(w_i^2) 惩罚 |
| Polynomial regression | “用线性数学拟合曲线” | 在多项式 features（x、x^2、x^3、...）上做线性回归，对 weights 仍然是线性的 |
| Overfitting | “记住训练数据” | 模型复杂到拟合了训练数据中的噪声，导致在新数据上失败 |

## 延伸阅读

- [An Introduction to Statistical Learning (ISLR)](https://www.statlearning.com/) -- 免费 PDF，第 3 章和第 6 章用实践 R 示例讲解线性回归与 regularization
- [The Elements of Statistical Learning (ESL)](https://hastie.su.domains/ElemStatLearn/) -- 免费 PDF，是 ISLR 更偏数学的配套读物，更深入处理 ridge 和 lasso
- [Stanford CS229 Lecture Notes on Linear Regression](https://cs229.stanford.edu/main_notes.pdf) -- Andrew Ng 的讲义，从第一性原理推导 normal equation 和 gradient descent
- [scikit-learn LinearRegression documentation](https://scikit-learn.org/stable/modules/linear_model.html) -- LinearRegression、Ridge、Lasso 和 ElasticNet 的实用参考，含代码示例
