# 机器学习中的微积分

> 导数告诉你哪边是下坡。神经网络学习所需的东西，也就这些。

**类型：** 学习
**语言：** Python
**前置要求：** 阶段 1，第 01-03 课
**时间：** ~60 分钟

## 学习目标

- 为常见 ML 函数（x^2、sigmoid、cross-entropy）计算数值导数和解析导数
- 从零实现 gradient descent，在 1D 和 2D 中最小化 loss function
- 推导线性回归模型的 gradient，并通过手动更新权重训练它
- 解释 Hessian 矩阵、Taylor series 近似，以及它们与优化方法的关系

## 问题

你有一个包含数百万权重的神经网络。每个权重都是一个旋钮。你需要弄清楚每一个旋钮应该往哪个方向转，才能让模型稍微少错一点。微积分给你的就是这个方向。

没有微积分，训练神经网络就只能随机尝试修改，然后祈祷结果变好。有了导数，你就能准确知道每个权重如何影响误差。每次都把每个旋钮朝正确方向转。

## 概念

### 什么是导数？

导数衡量变化率。对函数 y = f(x)，导数 f'(x) 告诉你：如果把 x 轻轻推一点，y 会变化多少？

从几何上看，导数就是某一点处切线的斜率。

**f(x) = x^2：**

| x | f(x) | f'(x)（斜率） |
|---|------|---------------|
| 0 | 0    | 0（平的，在底部） |
| 1 | 1    | 2 |
| 2 | 4    | 4（这一点处切线的斜率） |
| 3 | 9    | 6 |

在 x=2 时，斜率是 4。如果你把 x 稍微向右移动一点，y 大约会增加这个移动量的 4 倍。在 x=0 时，斜率是 0。你正处在碗底。

形式化定义：

```
f'(x) = lim   f(x + h) - f(x)
        h->0  -----------------
                     h
```

在代码里，你不真的取极限，只用一个非常小的 h。这就是数值导数。

### 偏导数：一次只看一个变量

真实函数有很多输入。神经网络的 loss 取决于成千上万个权重。偏导数会保持除一个变量之外的所有变量不变，然后对这个变量求导。

```
f(x, y) = x^2 + 3xy + y^2

df/dx = 2x + 3y     (treat y as a constant)
df/dy = 3x + 2y     (treat x as a constant)
```

每个偏导数都回答同一个问题：如果我只轻轻改变这个权重，loss 会如何变化？

### Gradient：所有偏导数组成的向量

Gradient 会把每个偏导数收集到一个向量中。对函数 f(x, y, z)，gradient 是：

```
grad f = [ df/dx, df/dy, df/dz ]
```

Gradient 指向最陡上升方向。要最小化一个函数，就朝相反方向走。

**f(x,y) = x^2 + y^2 的等高线图：**

这个函数形成一个碗形曲面，等高线是同心圆。最小值在 (0, 0)。

| 点 | grad f | -grad f（下降方向） |
|-------|--------|----------------------------|
| (1, 1) | [2, 2]（指向上坡，远离最小值） | [-2, -2]（指向下坡，靠近最小值） |
| (0, 0) | [0, 0]（平的，在最小值处） | [0, 0] |

这就是一张图里的 gradient descent。计算 gradient，取负，然后走一步。

### 与优化的关系

训练神经网络就是优化。你有一个 loss function L(w1, w2, ..., wn)，用来衡量模型错得多厉害。你想最小化它。

```
Gradient descent update rule:

  w_new = w_old - learning_rate * dL/dw

For every weight:
  1. Compute the partial derivative of loss with respect to that weight
  2. Subtract a small multiple of it from the weight
  3. Repeat
```

Learning rate 控制步长。太大就会越过目标。太小就只能慢慢爬。

**Loss landscape（1D 切片）：**

当权重 w 变化时，loss function L(w) 会形成一条有峰谷的曲线。

| 特征 | 描述 |
|---------|-------------|
| 全局最小值 | 整条曲线上的最低点：最佳解 |
| 局部最小值 | 比邻近点低、但不是整体最低的山谷 |
| 斜率 | Gradient descent 从任意起点沿斜率下坡 |

Gradient descent 沿斜率下坡。它可能卡在局部最小值，但在高维空间（数百万权重）中，这很少是实际问题。

### 数值导数 vs 解析导数

计算导数有两种方式。

解析：手动应用微积分规则。对 f(x) = x^2，导数是 f'(x) = 2x。精确。快速。

数值：用定义近似。计算 f(x+h) 和 f(x-h)，其中 h 很小，然后取差值。

```
Numerical (central difference):

f'(x) ~= f(x + h) - f(x - h)
          -----------------------
                  2h

h = 0.0001 works well in practice
```

数值导数较慢，但适用于任何函数。解析导数快，但需要你推导公式。神经网络框架使用第三种方法：automatic differentiation，它会机械地计算精确导数。你会在阶段 3 看到它。

### 手算简单函数的导数

这些导数会在 ML 中反复出现。

```
Function        Derivative       Used in
--------        ----------       -------
f(x) = x^2     f'(x) = 2x      Loss functions (MSE)
f(x) = wx + b  f'(w) = x        Linear layer (gradient w.r.t. weight)
                f'(b) = 1        Linear layer (gradient w.r.t. bias)
                f'(x) = w        Linear layer (gradient w.r.t. input)
f(x) = e^x     f'(x) = e^x     Softmax, attention
f(x) = ln(x)   f'(x) = 1/x     Cross-entropy loss
f(x) = 1/(1+e^-x)  f'(x) = f(x)(1-f(x))   Sigmoid activation
```

对 f(x) = x^2：

```
f(x) = x^2    f'(x) = 2x

  x    f(x)   f'(x)   meaning
  -2    4      -4      slope tilts left (decreasing)
  -1    1      -2      slope tilts left (decreasing)
   0    0       0      flat (minimum!)
   1    1       2      slope tilts right (increasing)
   2    4       4      slope tilts right (increasing)
```

对 f(w) = wx + b，且 x=3、b=1：

```
f(w) = 3w + 1    f'(w) = 3

The derivative with respect to w is just x.
If x is big, a small change in w causes a big change in output.
```

### 链式法则

当函数互相嵌套时，链式法则告诉你如何求导。

```
If y = f(g(x)), then dy/dx = f'(g(x)) * g'(x)

Example: y = (3x + 1)^2
  outer: f(u) = u^2       f'(u) = 2u
  inner: g(x) = 3x + 1    g'(x) = 3
  dy/dx = 2(3x + 1) * 3 = 6(3x + 1)
```

神经网络是一串函数：input -> linear -> activation -> linear -> activation -> loss。Backpropagation 就是从输出到输入反复应用链式法则。这就是整个算法。

### Hessian 矩阵

Gradient 告诉你斜率。Hessian 告诉你曲率。

Hessian 是二阶偏导数组成的矩阵。对函数 f(x1, x2, ..., xn)，Hessian 的 (i, j) 项是：

```
H[i][j] = d^2f / (dx_i * dx_j)
```

对二变量函数 f(x, y)：

```
H = | d^2f/dx^2    d^2f/dxdy |
    | d^2f/dydx    d^2f/dy^2 |
```

**Hessian 在临界点（gradient = 0）告诉你什么：**

| Hessian 性质 | 含义 | 示例曲面 |
|-----------------|---------|-----------------|
| 正定（所有特征值 > 0） | 局部最小值 | 向上开的碗 |
| 负定（所有特征值 < 0） | 局部最大值 | 向下开的碗 |
| 不定（特征值正负混合） | 鞍点 | 马鞍形 |

**例子：** f(x, y) = x^2 - y^2（鞍面函数）

```
df/dx = 2x       df/dy = -2y
d^2f/dx^2 = 2    d^2f/dy^2 = -2    d^2f/dxdy = 0

H = | 2   0 |
    | 0  -2 |

Eigenvalues: 2 and -2 (one positive, one negative)
--> Saddle point at (0, 0)
```

与 f(x, y) = x^2 + y^2（碗形）对比：

```
H = | 2  0 |
    | 0  2 |

Eigenvalues: 2 and 2 (both positive)
--> Local minimum at (0, 0)
```

**为什么 Hessian 在 ML 中重要：**

Newton's method 使用 Hessian 采取比 gradient descent 更好的优化步。它不只是跟随斜率，还会考虑曲率：

```
Newton's update:    w_new = w_old - H^(-1) * gradient
Gradient descent:   w_new = w_old - lr * gradient
```

Newton's method 收敛更快，因为 Hessian 会“重新缩放”gradient：陡峭方向步子更小，平坦方向步子更大。

问题是：对有 N 个参数的神经网络，Hessian 是 N x N。一个有 100 万参数的模型需要一个 1 万亿项的矩阵。所以我们使用近似。

| 方法 | 使用什么 | 成本 | 收敛 |
|--------|-------------|------|-------------|
| Gradient descent | 只有一阶导数 | 每步 O(N) | 慢（线性） |
| Newton's method | 完整 Hessian | 每步 O(N^3) | 快（二次） |
| L-BFGS | 从 gradient 历史近似 Hessian | 每步 O(N) | 中等（超线性） |
| Adam | 每参数自适应速率（对角 Hessian 近似） | 每步 O(N) | 中等 |
| Natural gradient | Fisher information matrix（统计 Hessian） | 每步 O(N^2) | 快 |

实践中，Adam 是深度学习的默认 optimizer。它通过跟踪每个参数 gradient 的运行均值和方差，廉价地近似二阶信息。

### Taylor Series 近似

任何光滑函数都可以在局部用多项式近似：

```
f(x + h) = f(x) + f'(x)*h + (1/2)*f''(x)*h^2 + (1/6)*f'''(x)*h^3 + ...
```

包含的项越多，近似越好；但只在点 x 附近成立。

**为什么 Taylor series 对 ML 重要：**

- **一阶 Taylor = gradient descent。** 当你使用 f(x + h) ~ f(x) + f'(x)*h 时，你在做线性近似。Gradient descent 最小化这个线性模型，从而选择 h = -lr * f'(x)。

- **二阶 Taylor = Newton's method。** 使用 f(x + h) ~ f(x) + f'(x)*h + (1/2)*f''(x)*h^2，你会得到一个二次模型。最小化它得到 h = -f'(x)/f''(x)：Newton's step。

- **Loss function 设计。** MSE 和 cross-entropy 是光滑的，这意味着它们的 Taylor 展开行为良好。这不是偶然。光滑的 loss 会让优化更可预测。

```
Approximation order    What it captures    Optimization method
-------------------    -----------------   -------------------
0th order (constant)   Just the value      Random search
1st order (linear)     Slope               Gradient descent
2nd order (quadratic)  Curvature           Newton's method
Higher orders          Finer structure     Rarely used in ML
```

关键洞见：所有基于 gradient 的优化，本质上都是在局部近似 loss function，然后走向这个近似函数的最小值。

### ML 中的积分

导数告诉你变化率。积分计算累积量：曲线下方的面积。

在 ML 中，你很少手算积分，但这个概念无处不在：

**概率。** 对一个密度为 p(x) 的连续随机变量：
```
P(a < X < b) = integral from a to b of p(x) dx
```
概率密度曲线在 a 和 b 之间的面积，就是落在这个范围内的概率。

**期望值。** 按概率加权的平均结果：
```
E[f(X)] = integral of f(x) * p(x) dx
```
数据分布上的 expected loss 是一个积分。训练最小化的是它的经验近似。

**KL divergence。** 衡量两个分布有多不同：
```
KL(p || q) = integral of p(x) * log(p(x) / q(x)) dx
```
用于 VAE、knowledge distillation 和 Bayesian inference。

**归一化常数。** 在 Bayesian inference 中：
```
p(w | data) = p(data | w) * p(w) / integral of p(data | w) * p(w) dw
```
分母是对所有可能参数值的积分。它通常不可解，所以我们使用 MCMC 和 variational inference 这样的近似。

| 积分概念 | 在 ML 中出现的位置 |
|-----------------|----------------------|
| 曲线下面积 | 由密度函数得到概率 |
| 期望值 | Loss function、risk minimization |
| KL divergence | VAE、policy optimization、distillation |
| 归一化 | Bayesian posterior、softmax denominator |
| 边际似然 | 模型比较、evidence lower bound（ELBO） |

### 计算图中的多变量链式法则

链式法则不只适用于一条线上的标量函数。在神经网络中，变量会分叉，也会合并。下面是一个简单 forward pass 中导数如何流动：

```mermaid
graph LR
    x["x (input)"] -->|"*w"| z1["z1 = w*x"]
    z1 -->|"+b"| z2["z2 = w*x + b"]
    z2 -->|"sigmoid"| a["a = sigmoid(z2)"]
    a -->|"loss fn"| L["L = -(y*log(a) + (1-y)*log(1-a))"]
```

Backward pass 从右到左计算 gradients：

```mermaid
graph RL
    dL["dL/dL = 1"] -->|"dL/da"| da["dL/da = -y/a + (1-y)/(1-a)"]
    da -->|"da/dz2 = a(1-a)"| dz2["dL/dz2 = dL/da * a(1-a)"]
    dz2 -->|"dz2/dw = x"| dw["dL/dw = dL/dz2 * x"]
    dz2 -->|"dz2/db = 1"| db["dL/db = dL/dz2 * 1"]
```

每条边都会乘以局部导数。任意参数的 gradient，就是从 loss 到该参数路径上所有局部导数的乘积。当路径分叉又合并时，你会把各条路径的贡献相加（多变量链式法则）。

Backpropagation 的全部内容就是：从输出到输入，系统地在计算图上应用链式法则。

### Jacobian 矩阵

当一个函数把向量映射到向量（比如神经网络层）时，它的导数是一个矩阵。Jacobian 包含每个输出对每个输入的所有偏导数。

对 f: R^n -> R^m，Jacobian J 是一个 m x n 矩阵：

| | x1 | x2 | ... | xn |
|---|---|---|---|---|
| f1 | df1/dx1 | df1/dx2 | ... | df1/dxn |
| f2 | df2/dx1 | df2/dx2 | ... | df2/dxn |
| ... | ... | ... | ... | ... |
| fm | dfm/dx1 | dfm/dx2 | ... | dfm/dxn |

你不会手算神经网络的 Jacobian。PyTorch 会处理它。但知道它存在，有助于你理解 backpropagation 中的 shape：如果一层把 R^n 映射到 R^m，它的 Jacobian 就是 m x n。Gradient 会通过这个矩阵的转置向后流动。

### 为什么这对神经网络重要

神经网络中的每个权重都会得到一个 gradient。Gradient 告诉你如何调整这个权重以降低 loss。

```mermaid
graph LR
    subgraph Forward["Forward Pass"]
        I["input"] --> W1["W1"] --> R["relu"] --> W2["W2"] --> S["softmax"] --> L["loss"]
    end
```

```mermaid
graph RL
    subgraph Backward["Backward Pass"]
        dL["dL/dloss"] --> dW2["dL/dW2"] --> d2["..."] --> dW1["dL/dW1"]
    end
```

每次权重更新：
- `W1 = W1 - lr * dL/dW1`
- `W2 = W2 - lr * dL/dW2`

Forward pass 计算预测和 loss。Backward pass 计算 loss 对每个权重的 gradient。然后每个权重都朝下坡方向走一小步。重复数百万步。这就是 deep learning。

## 构建它

### 第 1 步：从零实现数值导数

```python
def numerical_derivative(f, x, h=1e-7):
    return (f(x + h) - f(x - h)) / (2 * h)

def f(x):
    return x ** 2

for x in [-2, -1, 0, 1, 2]:
    numerical = numerical_derivative(f, x)
    analytical = 2 * x
    print(f"x={x:2d}  f'(x) numerical={numerical:.6f}  analytical={analytical:.1f}")
```

数值导数与解析导数匹配到很多位小数。

### 第 2 步：偏导数和 gradients

```python
def numerical_gradient(f, point, h=1e-7):
    gradient = []
    for i in range(len(point)):
        point_plus = list(point)
        point_minus = list(point)
        point_plus[i] += h
        point_minus[i] -= h
        partial = (f(point_plus) - f(point_minus)) / (2 * h)
        gradient.append(partial)
    return gradient

def f_multi(point):
    x, y = point
    return x**2 + 3*x*y + y**2

grad = numerical_gradient(f_multi, [1.0, 2.0])
print(f"Numerical gradient at (1,2): {[f'{g:.4f}' for g in grad]}")
print(f"Analytical gradient at (1,2): [2*1+3*2, 3*1+2*2] = [{2*1+3*2}, {3*1+2*2}]")
```

### 第 3 步：用 gradient descent 找到 f(x) = x^2 的最小值

```python
x = 5.0
lr = 0.1
for step in range(20):
    grad = 2 * x
    x = x - lr * grad
    print(f"step {step:2d}  x={x:8.4f}  f(x)={x**2:10.6f}")
```

从 x=5 开始，每一步都会更接近 x=0（最小值）。

### 第 4 步：在 2D 函数上做 gradient descent

```python
def f_2d(point):
    x, y = point
    return x**2 + y**2

point = [4.0, 3.0]
lr = 0.1
for step in range(30):
    grad = numerical_gradient(f_2d, point)
    point = [p - lr * g for p, g in zip(point, grad)]
    loss = f_2d(point)
    if step % 5 == 0 or step == 29:
        print(f"step {step:2d}  point=({point[0]:7.4f}, {point[1]:7.4f})  f={loss:.6f}")
```

### 第 5 步：比较数值导数和解析导数

```python
import math

test_functions = [
    ("x^2",      lambda x: x**2,          lambda x: 2*x),
    ("x^3",      lambda x: x**3,          lambda x: 3*x**2),
    ("sin(x)",   lambda x: math.sin(x),   lambda x: math.cos(x)),
    ("e^x",      lambda x: math.exp(x),   lambda x: math.exp(x)),
    ("1/x",      lambda x: 1/x,           lambda x: -1/x**2),
]

x = 2.0
print(f"{'Function':<12} {'Numerical':>12} {'Analytical':>12} {'Error':>12}")
print("-" * 50)
for name, f, df in test_functions:
    num = numerical_derivative(f, x)
    ana = df(x)
    err = abs(num - ana)
    print(f"{name:<12} {num:12.6f} {ana:12.6f} {err:12.2e}")
```

### 第 6 步：用数值方法计算 Hessian

```python
def hessian_2d(f, x, y, h=1e-5):
    fxx = (f(x + h, y) - 2 * f(x, y) + f(x - h, y)) / (h ** 2)
    fyy = (f(x, y + h) - 2 * f(x, y) + f(x, y - h)) / (h ** 2)
    fxy = (f(x + h, y + h) - f(x + h, y - h) - f(x - h, y + h) + f(x - h, y - h)) / (4 * h ** 2)
    return [[fxx, fxy], [fxy, fyy]]

def saddle(x, y):
    return x ** 2 - y ** 2

def bowl(x, y):
    return x ** 2 + y ** 2

H_saddle = hessian_2d(saddle, 0.0, 0.0)
H_bowl = hessian_2d(bowl, 0.0, 0.0)
print(f"Saddle Hessian: {H_saddle}")  # [[2, 0], [0, -2]] -- mixed signs
print(f"Bowl Hessian:   {H_bowl}")    # [[2, 0], [0, 2]]  -- both positive
```

鞍面函数的 Hessian 有特征值 2 和 -2（符号混合，确认是鞍点）。碗形函数的特征值是 2 和 2（都为正，确认是最小值）。

### 第 7 步：Taylor 近似实战

```python
import math

def taylor_approx(f, f_prime, f_double_prime, x0, h, order=2):
    result = f(x0)
    if order >= 1:
        result += f_prime(x0) * h
    if order >= 2:
        result += 0.5 * f_double_prime(x0) * h ** 2
    return result

x0 = 0.0
for h in [0.1, 0.5, 1.0, 2.0]:
    true_val = math.sin(h)
    t1 = taylor_approx(math.sin, math.cos, lambda x: -math.sin(x), x0, h, order=1)
    t2 = taylor_approx(math.sin, math.cos, lambda x: -math.sin(x), x0, h, order=2)
    print(f"h={h:.1f}  sin(h)={true_val:.4f}  order1={t1:.4f}  order2={t2:.4f}")
```

在 x0=0 附近，sin(x) ~ x（一阶 Taylor）。对小 h 来说近似非常好，但 h 大时就会失效。这就是为什么 gradient descent 最适合小 learning rate：每一步都假设线性近似是准确的。

### 第 8 步：为什么这对神经网络重要

```python
import random

random.seed(42)

w = random.gauss(0, 1)
b = random.gauss(0, 1)
lr = 0.01

xs = [1.0, 2.0, 3.0, 4.0, 5.0]
ys = [3.0, 5.0, 7.0, 9.0, 11.0]

for epoch in range(200):
    total_loss = 0
    dw = 0
    db = 0
    for x, y in zip(xs, ys):
        pred = w * x + b
        error = pred - y
        total_loss += error ** 2
        dw += 2 * error * x
        db += 2 * error
    dw /= len(xs)
    db /= len(xs)
    total_loss /= len(xs)
    w -= lr * dw
    b -= lr * db
    if epoch % 40 == 0 or epoch == 199:
        print(f"epoch {epoch:3d}  w={w:.4f}  b={b:.4f}  loss={total_loss:.6f}")

print(f"\nLearned: y = {w:.2f}x + {b:.2f}")
print(f"Actual:  y = 2x + 1")
```

每个基于 gradient 的训练循环都遵循这个模式：预测、计算 loss、计算 gradients、更新权重。

## 使用它

使用 NumPy，同样的操作会更快、更简洁：

```python
import numpy as np

x = np.array([1, 2, 3, 4, 5], dtype=float)
y = np.array([3, 5, 7, 9, 11], dtype=float)

w, b = np.random.randn(), np.random.randn()
lr = 0.01

for epoch in range(200):
    pred = w * x + b
    error = pred - y
    loss = np.mean(error ** 2)
    dw = np.mean(2 * error * x)
    db = np.mean(2 * error)
    w -= lr * dw
    b -= lr * db

print(f"Learned: y = {w:.2f}x + {b:.2f}")
```

你刚刚从零构建了 gradient descent。PyTorch 会自动化 gradient 计算，但更新循环是一样的。

## 练习

1. 用两次调用 `numerical_derivative` 实现 `numerical_second_derivative(f, x)`。验证 x^3 在 x=2 处的二阶导数为 12。
2. 使用 gradient descent 找到 f(x, y) = (x - 3)^2 + (y + 1)^2 的最小值。从 (0, 0) 开始。答案应该收敛到 (3, -1)。
3. 给 gradient descent 循环加上 momentum：维护一个累积过去 gradients 的 velocity 向量。在 f(x) = x^4 - 3x^2 上比较有无 momentum 的收敛速度。

## 关键术语

| 术语 | 人们常说 | 它实际意味着什么 |
|------|----------------|----------------------|
| 导数 | “斜率” | 函数在某一点的变化率。告诉你输入每变化一个单位，输出会变化多少。 |
| 偏导数 | “一个变量的导数” | 对一个变量求导，同时保持其他所有变量不变。 |
| Gradient | “最陡上升方向” | 所有偏导数组成的向量。指向函数增长最快的方向。 |
| Gradient descent | “往下坡走” | 从参数中减去 gradient（乘以 learning rate）以降低 loss。神经网络训练的核心。 |
| Learning rate | “步长” | 控制每一步 gradient descent 有多大的标量。太大：发散。太小：收敛很慢。 |
| 链式法则 | “导数相乘” | 对复合函数求导的规则：df/dx = df/dg * dg/dx。Backpropagation 的数学基础。 |
| Jacobian | “导数矩阵” | 当函数把向量映射到向量时，Jacobian 是所有输出对所有输入偏导数组成的矩阵。 |
| 数值导数 | “有限差分” | 在两个相近点计算函数值，并用它们之间的斜率近似导数。 |
| Backpropagation | “Reverse-mode autodiff” | 使用链式法则，从输出到输入逐层计算 gradients。神经网络学习的方式。 |
| Hessian | “二阶导数矩阵” | 所有二阶偏导数组成的矩阵。描述函数曲率。临界点处 Hessian 正定意味着局部最小值。 |
| Taylor series | “多项式近似” | 用函数在某点的导数近似附近函数值：f(x+h) ~ f(x) + f'(x)h + (1/2)f''(x)h^2 + ...。它是理解 gradient descent 和 Newton's method 工作原理的基础。 |
| 积分 | “曲线下面积” | 在某个范围上累积一个量。在 ML 中，积分定义概率、期望值和 KL divergence。 |

## 延伸阅读

- [3Blue1Brown: Essence of Calculus](https://www.3blue1brown.com/topics/calculus) - 关于导数、积分和链式法则的视觉直觉
- [Stanford CS231n: Backpropagation](https://cs231n.github.io/optimization-2/) - gradients 如何流过神经网络层
