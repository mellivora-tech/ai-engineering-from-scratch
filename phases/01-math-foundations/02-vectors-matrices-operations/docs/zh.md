# 向量、矩阵与运算

> 每个神经网络都只是矩阵乘法，只是多了一些步骤。

**类型：** 构建
**语言：** Python, Julia
**前置要求：** 阶段 1，第 01 课（线性代数直觉）
**时间：** ~60 分钟

## 学习目标

- 构建一个 Matrix 类，支持逐元素运算、矩阵乘法、转置、行列式和逆矩阵
- 区分逐元素乘法和矩阵乘法，并解释各自适用的场景
- 只使用从零实现的 Matrix 类，实现单个稠密神经网络层（`relu(W @ x + b)`）
- 解释 broadcasting 规则，以及神经网络框架中的 bias 加法如何工作

## 问题

你想构建一个神经网络。你读到代码里有这一行：

```
output = activation(weights @ input + bias)
```

这里的 `@` 是矩阵乘法。`weights` 是矩阵。`input` 是向量。如果你不知道这些操作在做什么，这一行就是魔法。如果你知道，它就是一个层的完整 forward pass：三个操作而已。

模型处理的每张图像都是像素值矩阵。每个词 embedding 都是向量。每一层神经网络都是矩阵变换。想构建 AI 系统，你必须像理解变量那样熟练理解矩阵运算。

本课会从零建立这种熟练度。

## 概念

### 向量：有顺序的数字列表

向量是一个数字列表，具有方向和长度。在 AI 中，向量表示数据点、特征或参数。

```
v = [3, 4]        -- a 2D vector
w = [1, 0, -2]    -- a 3D vector
```

2D 向量 `[3, 4]` 指向平面上的坐标 (3, 4)。它的长度（magnitude）是 5（3-4-5 三角形）。

### 矩阵：数字网格

矩阵是一个 2D 网格。它有行和列。一个 m x n 矩阵有 m 行、n 列。

```
A = | 1  2  3 |     -- 2x3 matrix (2 rows, 3 columns)
    | 4  5  6 |
```

在神经网络中，权重矩阵把输入向量变换成输出向量。一个有 784 个输入和 128 个输出的层，会使用一个 128x784 的权重矩阵。

### 为什么 shape 很重要

矩阵乘法有一条严格规则：`(m x n) @ (n x p) = (m x p)`。内部维度必须匹配。

```
(128 x 784) @ (784 x 1) = (128 x 1)
  weights       input       output

Inner dimensions: 784 = 784  -- valid
```

如果你在 PyTorch 里遇到 shape mismatch 错误，原因就在这里。

### 运算地图

| 运算 | 做什么 | 神经网络用途 |
|-----------|-------------|-------------------|
| 加法 | 逐元素组合 | 给输出加 bias |
| 标量乘法 | 缩放每个元素 | Learning rate * gradients |
| 矩阵乘法 | 变换向量 | 层的 forward pass |
| 转置 | 翻转行和列 | Backpropagation |
| 行列式 | 单个数字摘要 | 检查可逆性 |
| 逆矩阵 | 撤销一个变换 | 求解线性系统 |
| 单位矩阵 | 什么都不做的矩阵 | 初始化、residual connections |

### 逐元素乘法 vs 矩阵乘法

这个区别经常绊倒初学者。

逐元素：匹配位置相乘。两个矩阵必须有相同 shape。

```
| 1  2 |   | 5  6 |   | 5  12 |
| 3  4 | * | 7  8 | = | 21 32 |
```

矩阵乘法：行和列做点积。内部维度必须匹配。

```
| 1  2 |   | 5  6 |   | 1*5+2*7  1*6+2*8 |   | 19  22 |
| 3  4 | @ | 7  8 | = | 3*5+4*7  3*6+4*8 | = | 43  50 |
```

不同的运算，不同的结果，不同的规则。

### Broadcasting

当你把一个 bias 向量加到输出矩阵上时，shape 并不匹配。Broadcasting 会把较小的数组拉伸到能匹配的位置。

```
| 1  2  3 |   +   [10, 20, 30]
| 4  5  6 |

Broadcasting stretches the vector across rows:

| 1  2  3 |   | 10  20  30 |   | 11  22  33 |
| 4  5  6 | + | 10  20  30 | = | 14  25  36 |
```

每个现代框架都会自动做这件事。理解它能避免这样的困惑：shape 看起来不对，但代码却能运行。

## 构建它

### 第 1 步：Vector 类

```python
class Vector:
    def __init__(self, data):
        self.data = list(data)
        self.size = len(self.data)

    def __repr__(self):
        return f"Vector({self.data})"

    def __add__(self, other):
        return Vector([a + b for a, b in zip(self.data, other.data)])

    def __sub__(self, other):
        return Vector([a - b for a, b in zip(self.data, other.data)])

    def __mul__(self, scalar):
        return Vector([x * scalar for x in self.data])

    def dot(self, other):
        return sum(a * b for a, b in zip(self.data, other.data))

    def magnitude(self):
        return sum(x ** 2 for x in self.data) ** 0.5
```

### 第 2 步：带核心运算的 Matrix 类

```python
class Matrix:
    def __init__(self, data):
        self.data = [list(row) for row in data]
        self.rows = len(self.data)
        self.cols = len(self.data[0])
        self.shape = (self.rows, self.cols)

    def __repr__(self):
        rows_str = "\n  ".join(str(row) for row in self.data)
        return f"Matrix({self.shape}):\n  {rows_str}"

    def __add__(self, other):
        return Matrix([
            [self.data[i][j] + other.data[i][j] for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def __sub__(self, other):
        return Matrix([
            [self.data[i][j] - other.data[i][j] for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def scalar_multiply(self, scalar):
        return Matrix([
            [self.data[i][j] * scalar for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def element_wise_multiply(self, other):
        return Matrix([
            [self.data[i][j] * other.data[i][j] for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def matmul(self, other):
        return Matrix([
            [
                sum(self.data[i][k] * other.data[k][j] for k in range(self.cols))
                for j in range(other.cols)
            ]
            for i in range(self.rows)
        ])

    def transpose(self):
        return Matrix([
            [self.data[j][i] for j in range(self.rows)]
            for i in range(self.cols)
        ])

    def determinant(self):
        if self.shape == (1, 1):
            return self.data[0][0]
        if self.shape == (2, 2):
            return self.data[0][0] * self.data[1][1] - self.data[0][1] * self.data[1][0]
        det = 0
        for j in range(self.cols):
            minor = Matrix([
                [self.data[i][k] for k in range(self.cols) if k != j]
                for i in range(1, self.rows)
            ])
            det += ((-1) ** j) * self.data[0][j] * minor.determinant()
        return det

    def inverse_2x2(self):
        det = self.determinant()
        if det == 0:
            raise ValueError("Matrix is singular, no inverse exists")
        return Matrix([
            [self.data[1][1] / det, -self.data[0][1] / det],
            [-self.data[1][0] / det, self.data[0][0] / det]
        ])

    @staticmethod
    def identity(n):
        return Matrix([
            [1 if i == j else 0 for j in range(n)]
            for i in range(n)
        ])
```

### 第 3 步：看它运行

```python
A = Matrix([[1, 2], [3, 4]])
B = Matrix([[5, 6], [7, 8]])

print("A + B =", (A + B).data)
print("A @ B =", A.matmul(B).data)
print("A^T =", A.transpose().data)
print("det(A) =", A.determinant())
print("A^-1 =", A.inverse_2x2().data)

I = Matrix.identity(2)
print("A @ A^-1 =", A.matmul(A.inverse_2x2()).data)
```

### 第 4 步：连接到神经网络

```python
import random

inputs = Matrix([[0.5], [0.8], [0.2]])
weights = Matrix([
    [random.uniform(-1, 1) for _ in range(3)]
    for _ in range(2)
])
bias = Matrix([[0.1], [0.1]])

def relu_matrix(m):
    return Matrix([[max(0, val) for val in row] for row in m.data])

pre_activation = weights.matmul(inputs) + bias
output = relu_matrix(pre_activation)

print(f"Input shape: {inputs.shape}")
print(f"Weight shape: {weights.shape}")
print(f"Output shape: {output.shape}")
print(f"Output: {output.data}")
```

这就是一个稠密层：`output = relu(W @ x + b)`。每个神经网络中的每个 dense layer 都在做完全相同的事。

## 使用它

NumPy 会用更少的代码、更快几个数量级的速度完成上面的一切。

```python
import numpy as np

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

print("A + B =\n", A + B)
print("A * B (element-wise) =\n", A * B)
print("A @ B (matrix multiply) =\n", A @ B)
print("A^T =\n", A.T)
print("det(A) =", np.linalg.det(A))
print("A^-1 =\n", np.linalg.inv(A))
print("I =\n", np.eye(2))

inputs = np.random.randn(3, 1)
weights = np.random.randn(2, 3)
bias = np.array([[0.1], [0.1]])
output = np.maximum(0, weights @ inputs + bias)

print(f"\nNeural network layer: {weights.shape} @ {inputs.shape} = {output.shape}")
print(f"Output:\n{output}")
```

Python 中的 `@` 运算符会调用 `__matmul__`。NumPy 使用 C 和 Fortran 编写的优化 BLAS 例程实现它。同样的数学，快 100 倍。

NumPy 中的 broadcasting：

```python
matrix = np.array([[1, 2, 3], [4, 5, 6]])
bias = np.array([10, 20, 30])
print(matrix + bias)
```

NumPy 会自动把 1D bias broadcast 到两行上。每个神经网络框架里的 bias 加法都是这样工作的。

## 交付它

本课会产出一个 prompt，用于通过几何直觉教授矩阵运算。见 `outputs/prompt-matrix-operations.md`。

这里构建的 Matrix 类，是我们在阶段 3 第 10 课中构建迷你神经网络框架的基础。

## 练习

1. **验证逆矩阵。** 计算 `A @ A.inverse_2x2()`，确认你得到单位矩阵。用三个不同的 2x2 矩阵试一遍。当行列式为零时会发生什么？

2. **实现 3x3 逆矩阵。** 扩展 Matrix 类，用伴随矩阵方法计算 3x3 矩阵的逆。用 NumPy 的 `np.linalg.inv` 测试它。

3. **构建两层网络。** 只使用你的 Matrix 类（不用 NumPy），创建一个两层神经网络：input (3) -> hidden (4) -> output (2)。随机初始化权重，运行一次 forward pass，并验证所有 shape 都正确。

## 关键术语

| 术语 | 人们常说 | 它实际意味着什么 |
|------|----------------|----------------------|
| 向量 | “一个箭头” | 有顺序的数字列表。在 AI 中：高维空间里的一个点。 |
| 矩阵 | “一张数字表” | 一个线性变换。它把向量从一个空间映射到另一个空间。 |
| 矩阵乘法 | “就是把数字相乘” | 第一个矩阵的每一行与第二个矩阵的每一列做点积。顺序很重要。 |
| 转置 | “翻一下” | 交换行和列。把 m x n 矩阵变成 n x m。对 backpropagation 很关键。 |
| 行列式 | “矩阵里的某个数字” | 衡量矩阵把面积（2D）或体积（3D）缩放了多少。零表示变换压扁了一个维度。 |
| 逆矩阵 | “撤销这个矩阵” | 反转变换的矩阵。只有行列式不为零时才存在。 |
| 单位矩阵 | “无聊的矩阵” | 矩阵版的乘以 1。用于 residual connections（ResNets）。 |
| Broadcasting | “神奇修 shape” | 通过沿缺失维度重复，把较小数组拉伸到匹配较大数组。 |
| 逐元素 | “普通乘法” | 匹配位置相乘。两个数组必须有相同 shape（或可以 broadcast）。 |

## 延伸阅读

- [3Blue1Brown: Essence of Linear Algebra](https://www.3blue1brown.com/topics/linear-algebra) - 本课所有操作的视觉直觉
- [NumPy documentation on broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html) - NumPy 遵循的精确规则
- [Stanford CS229 Linear Algebra Review](http://cs229.stanford.edu/section/cs229-linalg.pdf) - 面向 ML 的线性代数简明参考
