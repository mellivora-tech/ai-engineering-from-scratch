# Activation Functions

> 没有非线性，你的 100 层网络只是一个花哨的矩阵乘法。Activations 是让神经网络能用曲线思考的门。

**类型：** 构建
**语言：** Python
**先修：** Lesson 03.03（Backpropagation）
**时间：** 约 75 分钟

## 学习目标

- 从零实现 sigmoid、tanh、ReLU、Leaky ReLU、GELU、Swish 和 softmax，以及它们的 derivatives
- 通过测量 10+ 层中不同 activation 的 activation magnitudes，诊断 vanishing gradient problem
- 检测 ReLU 网络中的 dead neurons，并解释为什么 GELU 能避免这种 failure mode
- 为给定架构（transformer、CNN、RNN、output layer）选择正确的 activation function

## 问题

堆叠两个线性变换：y = W2(W1x + b1) + b2。展开它：y = W2W1x + W2b1 + b2。也就是 y = Ax + c——一个单独的线性变换。无论你堆多少个 linear layers，结果都会塌缩成一次矩阵乘法。你的 100 层网络和单层网络拥有相同的表示能力。

这不是理论上的小趣味。它意味着深层线性网络真的学不会 XOR，无法分类螺旋数据集，也无法识别人脸。没有 activation functions，深度只是幻觉。

Activation functions 打破线性。它们通过非线性函数扭曲每一层的输出，让网络能够弯曲 decision boundaries、近似任意函数，并真正学习。但如果选错 activation，你的 gradients 会消失到 0（深层网络中的 sigmoid）、爆炸到无穷大（没有谨慎初始化的无界 activations），或者你的神经元会永久死亡（带大负 bias 的 ReLU）。activation function 的选择直接决定你的网络是否能学起来。

## 概念

### 为什么非线性是必要的

矩阵乘法是可组合的。先用矩阵 A 乘一个向量，再用矩阵 B 乘，等价于直接乘以 AB。这意味着堆叠十个 linear layers 在数学上等价于一个带大矩阵的 linear layer。所有那些参数、所有那些深度——都浪费了。你需要某个东西打断这条链。这正是 activation functions 的作用。

证明如下。一个 linear layer 计算 f(x) = Wx + b。堆叠两个：

```
Layer 1: h = W1 * x + b1
Layer 2: y = W2 * h + b2
```

代入：

```
y = W2 * (W1 * x + b1) + b2
y = (W2 * W1) * x + (W2 * b1 + b2)
y = A * x + c
```

还是一层。在线性层之间插入 nonlinear activation g()：

```
h = g(W1 * x + b1)
y = W2 * h + b2
```

现在代入就断了。W2 * g(W1 * x + b1) + b2 不能化简成一个单独的线性变换。网络可以表示非线性函数。每多一层带 activation 的层，都会增加表示能力。

### Sigmoid

神经网络最早使用的 activation function。

```
sigmoid(x) = 1 / (1 + e^(-x))
```

输出范围：(0, 1)。平滑、可微，会把任意实数映射成类似概率的值。

derivative：

```
sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))
```

这个 derivative 的最大值是 0.25，出现在 x = 0。backpropagation 中，gradients 会穿过多层相乘。十层 sigmoid 意味着 gradient 最多被乘以 0.25 十次：

```
0.25^10 = 0.000000953674
```

不到原始信号的百万分之一。这就是 vanishing gradient problem。早期层的 gradients 变得极小，weights 几乎不更新。网络看起来在学习——后面层的 loss 在下降——但前几层被冻结了。深层 sigmoid 网络根本训练不起来。

另一个问题：sigmoid 输出永远为正（0 到 1），这意味着 weights 上的 gradients 总是同号。这会导致 gradient descent 中的之字形震荡。

### Tanh

sigmoid 的居中版本。

```
tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
```

输出范围：(-1, 1)。以 0 为中心，消除了之字形问题。

derivative：

```
tanh'(x) = 1 - tanh(x)^2
```

最大 derivative 在 x = 0 时是 1.0——比 sigmoid 好四倍。但 vanishing gradient problem 依然存在。对很大的正输入或负输入，derivative 都会接近 0。十层之后 gradient 仍然会被压扁，只是没那么激烈。

### ReLU：突破

Rectified Linear Unit。Nair 和 Hinton 在 2010 年把它推广到 deep learning（这个函数本身可追溯到 Fukushima 1969 年的工作），它改变了一切。

```
relu(x) = max(0, x)
```

输出范围：[0, infinity)。derivative 简单到极致：

```
relu'(x) = 1  if x > 0
            0  if x <= 0
```

对正输入没有 vanishing gradient。gradient 正好是 1，会直接传过去。这就是深层网络变得可训练的原因——ReLU 能跨层保留 gradient magnitude。

但它有一个 failure mode：dead neuron problem。如果某个神经元的加权输入永远为负（因为很大的负 bias 或糟糕的 weight initialization），它的输出永远是 0，gradient 永远是 0，也就永远不会更新。它永久死亡。在实践中，ReLU 网络里 10-40% 的神经元可能在训练中死亡。

### Leaky ReLU

dead neurons 最简单的修复。

```
leaky_relu(x) = x        if x > 0
                alpha * x if x <= 0
```

其中 alpha 是一个小常数，通常是 0.01。负半轴不再是 0，而是有一个小斜率，因此 dead neurons 仍能得到 gradient signal 并恢复。

### GELU：现代默认选择

Gaussian Error Linear Unit。Hendrycks 和 Gimpel 于 2016 年提出。它是 BERT、GPT 和大多数现代 transformers 的默认 activation。

```
gelu(x) = x * Phi(x)
```

其中 Phi(x) 是标准正态分布的 cumulative distribution function。实践中使用的近似：

```
gelu(x) ~= 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
```

GELU 处处平滑，允许小的负值（不同于把负数硬截断为 0 的 ReLU），并有概率解释：它根据输入在高斯分布下为正的可能性来加权该输入。这种平滑 gating 在 transformer 架构中优于 ReLU，因为它提供更好的 gradient flow，并完全避免 dead neuron problem。

### Swish / SiLU

Ramachandran 等人在 2017 年通过自动搜索发现的 self-gated activation。

```
swish(x) = x * sigmoid(x)
```

Swish 的形式就是 x * sigmoid(x)。Google 通过在 activation function 空间中自动搜索发现了它——让一个神经网络去设计神经网络的一部分。

和 GELU 一样，它平滑、非单调，并允许小的负值。差别很细微：Swish 使用 sigmoid 做 gating，而 GELU 使用 Gaussian CDF。实践中性能几乎相同。Swish 用在 EfficientNet 和一些视觉模型中。GELU 则主导语言模型。

### Softmax：输出 Activation

不用于 hidden layers。Softmax 会把一组原始分数（logits）转换为 probability distribution。

```
softmax(x_i) = e^(x_i) / sum(e^(x_j) for all j)
```

每个输出都在 0 和 1 之间。所有输出之和为 1。这让它成为 multi-class classification 的标准最终 activation。最大的 logit 得到最高概率，但不同于 argmax，softmax 可微，并保留相对置信度的信息。

### Shape 对比

```mermaid
graph LR
    subgraph "Activation Functions"
        S["Sigmoid<br/>Range: (0,1)<br/>Saturates both ends"]
        T["Tanh<br/>Range: (-1,1)<br/>Zero-centered"]
        R["ReLU<br/>Range: [0,inf)<br/>Dead neurons"]
        G["GELU<br/>Range: ~(-0.17,inf)<br/>Smooth gating"]
    end
    S -->|"Vanishing gradient"| Problem["Deep networks<br/>don't train"]
    T -->|"Less severe but<br/>still vanishes"| Problem
    R -->|"Gradient = 1<br/>for x > 0"| Solution["Deep networks<br/>train fast"]
    G -->|"Smooth gradient<br/>everywhere"| Solution
```

### Gradient Flow 对比

```mermaid
graph TD
    Input["Input Signal"] --> L1["Layer 1"]
    L1 --> L5["Layer 5"]
    L5 --> L10["Layer 10"]
    L10 --> Output["Output"]

    subgraph "Gradient at Layer 1"
        SigGrad["Sigmoid: ~0.000001"]
        TanhGrad["Tanh: ~0.001"]
        ReluGrad["ReLU: ~1.0"]
        GeluGrad["GELU: ~0.8"]
    end
```

### 什么时候用哪个 Activation

```mermaid
flowchart TD
    Start["What are you building?"] --> Hidden{"Hidden layers<br/>or output?"}

    Hidden -->|"Hidden layers"| Arch{"Architecture?"}
    Hidden -->|"Output layer"| Task{"Task type?"}

    Arch -->|"Transformer / NLP"| GELU["Use GELU"]
    Arch -->|"CNN / Vision"| ReLU["Use ReLU or Swish"]
    Arch -->|"RNN / LSTM"| Tanh["Use Tanh"]
    Arch -->|"Simple MLP"| ReLU2["Use ReLU"]

    Task -->|"Binary classification"| Sigmoid["Use Sigmoid"]
    Task -->|"Multi-class classification"| Softmax["Use Softmax"]
    Task -->|"Regression"| Linear["Use Linear (no activation)"]
```

## 构建

### Step 1: Implement All Activation Functions with Derivatives

每个函数接收一个 float 并返回一个 float。每个 derivative function 接收同样的输入并返回 gradient。

```python
import math

def sigmoid(x):
    x = max(-500, min(500, x))
    return 1.0 / (1.0 + math.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

def tanh_act(x):
    return math.tanh(x)

def tanh_derivative(x):
    t = math.tanh(x)
    return 1 - t * t

def relu(x):
    return max(0.0, x)

def relu_derivative(x):
    return 1.0 if x > 0 else 0.0

def leaky_relu(x, alpha=0.01):
    return x if x > 0 else alpha * x

def leaky_relu_derivative(x, alpha=0.01):
    return 1.0 if x > 0 else alpha

def gelu(x):
    return 0.5 * x * (1 + math.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x ** 3)))

def gelu_derivative(x):
    phi = 0.5 * (1 + math.erf(x / math.sqrt(2)))
    pdf = math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)
    return phi + x * pdf

def swish(x):
    return x * sigmoid(x)

def swish_derivative(x):
    s = sigmoid(x)
    return s + x * s * (1 - s)

def softmax(xs):
    max_x = max(xs)
    exps = [math.exp(x - max_x) for x in xs]
    total = sum(exps)
    return [e / total for e in exps]
```

### Step 2: Visualize Where Gradients Die

在 -5 到 5 之间取 100 个均匀间隔的点，计算每个点的 gradient。打印一个文本直方图，展示每个 activation 的 gradient 在哪里接近 0。

```python
def gradient_scan(name, derivative_fn, start=-5, end=5, n=100):
    step = (end - start) / n
    near_zero = 0
    healthy = 0
    for i in range(n):
        x = start + i * step
        g = derivative_fn(x)
        if abs(g) < 0.01:
            near_zero += 1
        else:
            healthy += 1
    pct_dead = near_zero / n * 100
    print(f"{name:15s}: {healthy:3d} healthy, {near_zero:3d} near-zero ({pct_dead:.0f}% dead zone)")

gradient_scan("Sigmoid", sigmoid_derivative)
gradient_scan("Tanh", tanh_derivative)
gradient_scan("ReLU", relu_derivative)
gradient_scan("Leaky ReLU", leaky_relu_derivative)
gradient_scan("GELU", gelu_derivative)
gradient_scan("Swish", swish_derivative)
```

### Step 3: Vanishing Gradient Experiment

使用 sigmoid 与 ReLU，让一个信号 forward-pass 穿过 N 层。测量 activation magnitude 如何变化。

```python
import random

def vanishing_gradient_experiment(activation_fn, name, n_layers=10, n_inputs=5):
    random.seed(42)
    values = [random.gauss(0, 1) for _ in range(n_inputs)]

    print(f"\n{name} through {n_layers} layers:")
    for layer in range(n_layers):
        weights = [random.gauss(0, 1) for _ in range(n_inputs)]
        z = sum(w * v for w, v in zip(weights, values))
        activated = activation_fn(z)
        magnitude = abs(activated)
        bar = "#" * int(magnitude * 20)
        print(f"  Layer {layer+1:2d}: magnitude = {magnitude:.6f} {bar}")
        values = [activated] * n_inputs

vanishing_gradient_experiment(sigmoid, "Sigmoid")
vanishing_gradient_experiment(relu, "ReLU")
vanishing_gradient_experiment(gelu, "GELU")
```

### Step 4: Dead Neuron Detector

创建一个 ReLU 网络，让随机 inputs 通过它，统计有多少神经元从不触发。

```python
def dead_neuron_detector(n_inputs=5, hidden_size=20, n_samples=1000):
    random.seed(0)
    weights = [[random.gauss(0, 1) for _ in range(n_inputs)] for _ in range(hidden_size)]
    biases = [random.gauss(0, 1) for _ in range(hidden_size)]

    fire_counts = [0] * hidden_size

    for _ in range(n_samples):
        inputs = [random.gauss(0, 1) for _ in range(n_inputs)]
        for neuron_idx in range(hidden_size):
            z = sum(w * x for w, x in zip(weights[neuron_idx], inputs)) + biases[neuron_idx]
            if relu(z) > 0:
                fire_counts[neuron_idx] += 1

    dead = sum(1 for c in fire_counts if c == 0)
    rarely_fire = sum(1 for c in fire_counts if 0 < c < n_samples * 0.05)
    healthy = hidden_size - dead - rarely_fire

    print(f"\nDead Neuron Report ({hidden_size} neurons, {n_samples} samples):")
    print(f"  Dead (never fired):     {dead}")
    print(f"  Barely alive (<5%):     {rarely_fire}")
    print(f"  Healthy:                {healthy}")
    print(f"  Dead neuron rate:       {dead/hidden_size*100:.1f}%")

    for i, c in enumerate(fire_counts):
        status = "DEAD" if c == 0 else "WEAK" if c < n_samples * 0.05 else "OK"
        bar = "#" * (c * 40 // n_samples)
        print(f"  Neuron {i:2d}: {c:4d}/{n_samples} fires [{status:4s}] {bar}")

dead_neuron_detector()
```

### Step 5: Training Comparison -- Sigmoid vs ReLU vs GELU

在 circle dataset（圆内点 = class 1，圆外点 = class 0）上，用三种不同 activations 训练同一个两层网络。比较收敛速度。

```python
def make_circle_data(n=200, seed=42):
    random.seed(seed)
    data = []
    for _ in range(n):
        x = random.uniform(-2, 2)
        y = random.uniform(-2, 2)
        label = 1.0 if x * x + y * y < 1.5 else 0.0
        data.append(([x, y], label))
    return data


class ActivationNetwork:
    def __init__(self, activation_fn, activation_deriv, hidden_size=8, lr=0.1):
        random.seed(0)
        self.act = activation_fn
        self.act_d = activation_deriv
        self.lr = lr
        self.hidden_size = hidden_size

        self.w1 = [[random.gauss(0, 0.5) for _ in range(2)] for _ in range(hidden_size)]
        self.b1 = [0.0] * hidden_size
        self.w2 = [random.gauss(0, 0.5) for _ in range(hidden_size)]
        self.b2 = 0.0

    def forward(self, x):
        self.x = x
        self.z1 = []
        self.h = []
        for i in range(self.hidden_size):
            z = self.w1[i][0] * x[0] + self.w1[i][1] * x[1] + self.b1[i]
            self.z1.append(z)
            self.h.append(self.act(z))

        self.z2 = sum(self.w2[i] * self.h[i] for i in range(self.hidden_size)) + self.b2
        self.out = sigmoid(self.z2)
        return self.out

    def backward(self, target):
        error = self.out - target
        d_out = error * self.out * (1 - self.out)

        for i in range(self.hidden_size):
            d_h = d_out * self.w2[i] * self.act_d(self.z1[i])
            self.w2[i] -= self.lr * d_out * self.h[i]
            for j in range(2):
                self.w1[i][j] -= self.lr * d_h * self.x[j]
            self.b1[i] -= self.lr * d_h
        self.b2 -= self.lr * d_out

    def train(self, data, epochs=200):
        losses = []
        for epoch in range(epochs):
            total_loss = 0
            correct = 0
            for x, y in data:
                pred = self.forward(x)
                self.backward(y)
                total_loss += (pred - y) ** 2
                if (pred >= 0.5) == (y >= 0.5):
                    correct += 1
            avg_loss = total_loss / len(data)
            accuracy = correct / len(data) * 100
            losses.append(avg_loss)
            if epoch % 50 == 0 or epoch == epochs - 1:
                print(f"    Epoch {epoch:3d}: loss={avg_loss:.4f}, accuracy={accuracy:.1f}%")
        return losses


data = make_circle_data()

configs = [
    ("Sigmoid", sigmoid, sigmoid_derivative),
    ("ReLU", relu, relu_derivative),
    ("GELU", gelu, gelu_derivative),
]

results = {}
for name, act_fn, act_d_fn in configs:
    print(f"\n=== Training with {name} ===")
    net = ActivationNetwork(act_fn, act_d_fn, hidden_size=8, lr=0.1)
    losses = net.train(data, epochs=200)
    results[name] = losses

print("\n=== Final Loss Comparison ===")
for name, losses in results.items():
    print(f"  {name:10s}: start={losses[0]:.4f} -> end={losses[-1]:.4f} (improvement: {(1 - losses[-1]/losses[0])*100:.1f}%)")
```

## 使用

PyTorch 同时提供这些 activation 的 functional 形式和 module 形式：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

x = torch.randn(4, 10)

relu_out = F.relu(x)
gelu_out = F.gelu(x)
sigmoid_out = torch.sigmoid(x)
swish_out = F.silu(x)

logits = torch.randn(4, 5)
probs = F.softmax(logits, dim=1)

model = nn.Sequential(
    nn.Linear(10, 64),
    nn.GELU(),
    nn.Linear(64, 32),
    nn.GELU(),
    nn.Linear(32, 5),
)
```

Transformer 的 hidden layers：GELU。CNN 的 hidden layers：ReLU。分类的 output layer：softmax。回归的 output layer：不加 activation（linear）。概率输出：sigmoid。就是这些。先从这些默认值开始。只有在有证据时才改变它们。

RNN 和 LSTM 对 hidden state 使用 tanh，对 gates 使用 sigmoid，但如果今天从零构建，你大概率不会使用 RNN。如果你的 ReLU 网络里神经元在死亡，切换到 GELU。除非有具体理由，否则不要先去用 Leaky ReLU——GELU 能解决 dead neuron problem，并提供更好的 gradient flow。

## 交付

本课会产出：
- `outputs/prompt-activation-selector.md`——一个可复用 prompt，帮助你为任意架构选择正确的 activation function

## 练习

1. 实现 Parametric ReLU（PReLU），其中负半轴斜率 alpha 是可学习参数。在 circle dataset 上训练它，并与固定的 Leaky ReLU 比较。

2. 把 vanishing gradient experiment 从 10 层改成 50 层。绘制 sigmoid、tanh、ReLU 和 GELU 在每一层的 magnitude。每种 activation 的信号在哪一层实际上到达了 0？

3. 实现 ELU（Exponential Linear Unit）：elu(x) = x if x > 0, alpha * (e^x - 1) if x <= 0。在同一个网络上比较它和 ReLU 的 dead neuron rate。

4. 构建一个“gradient health monitor”，在训练期间运行：每个 epoch 计算每一层的平均 gradient magnitude。当任意层的 gradient 低于 0.001 或超过 100 时打印警告。

5. 修改 training comparison，使用 Lesson 01 的 XOR dataset，而不是 circles。哪种 activation 在 XOR 上收敛最快？为什么这和 circle 结果不同？

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|----------|----------|
| Activation function | “非线性部分” | 应用于每个神经元输出的函数，打破线性，使网络能够学习非线性映射 |
| Vanishing gradient | “深层网络里的 gradients 消失了” | 当 activation 的 derivative 小于 1 时，gradients 会穿过层指数级缩小，使早期层无法训练 |
| Exploding gradient | “Gradients 爆了” | 当有效乘数超过 1 时，gradients 会穿过层指数级增长，导致训练不稳定 |
| Dead neuron | “停止学习的神经元” | 输入永久为负的 ReLU 神经元，输出为 0，gradient 也为 0 |
| Sigmoid | “把值压到 0-1” | logistic function 1/(1+e^-x)，历史上重要，但会在深层网络中造成 vanishing gradients |
| ReLU | “把负数截成 0” | max(0, x)——通过保留 gradient magnitude，让 deep learning 变得可行的 activation |
| GELU | “Transformer activation” | Gaussian Error Linear Unit，一种平滑 activation，会按输入为正的概率给输入加权 |
| Swish/SiLU | “Self-gated ReLU” | x * sigmoid(x)，通过自动搜索发现，用于 EfficientNet |
| Softmax | “把分数变成概率” | 把 logits 向量归一化成 probability distribution，所有值在 (0,1)，且总和为 1 |
| Leaky ReLU | “不会死的 ReLU” | max(alpha*x, x)，其中 alpha 很小（0.01），通过允许小的负 gradients 防止 dead neurons |
| Saturation | “sigmoid 的平坦部分” | activation 的 derivative 接近 0 的区域，会阻断 gradient flow |
| Logit | “softmax 之前的原始分数” | 在应用 softmax 或 sigmoid 之前，final layer 的未归一化输出 |

## 延伸阅读

- Nair & Hinton, "Rectified Linear Units Improve Restricted Boltzmann Machines" (2010)——引入 ReLU 并让深层网络训练成为可能的论文
- Hendrycks & Gimpel, "Gaussian Error Linear Units (GELUs)" (2016)——提出后来成为 transformers 默认 activation 的函数
- Ramachandran et al., "Searching for Activation Functions" (2017)——用自动搜索发现 Swish，展示 activation design 可以自动化
- Glorot & Bengio, "Understanding the difficulty of training deep feedforward neural networks" (2010)——诊断 vanishing/exploding gradients 并提出 Xavier initialization 的论文
- Goodfellow, Bengio, Courville, "Deep Learning" Chapter 6.3 (https://www.deeplearningbook.org/)——关于 hidden units 和 activation functions 的严谨处理
