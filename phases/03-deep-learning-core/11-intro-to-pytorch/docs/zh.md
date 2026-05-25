# PyTorch 入门

> 你已经用活塞和曲轴造出了引擎。现在来学习大家真正开的那辆车。

**类型：** 构建
**语言：** Python
**先修：** Lesson 03.10（Build Your Own Mini Framework）
**时间：** 约 75 分钟

## 学习目标

- 使用 PyTorch 的 nn.Module、nn.Sequential 和 autograd 构建并训练 neural networks
- 使用 PyTorch tensors、GPU acceleration 和标准 training loop（zero_grad、forward、loss、backward、step）
- 将你的 from-scratch mini framework 组件转换成 PyTorch 等价物
- 在同一任务上 profile 并比较纯 Python framework 与 PyTorch 的训练速度

## 问题

你已经有一个可工作的 mini framework。Linear layers、ReLU、dropout、batch norm、Adam、DataLoader、training loop。它可以用纯 Python 在 circle classification 问题上训练 4 层网络。

但它在同一问题上也比 PyTorch 慢 500 倍。

你的 mini framework 用嵌套 Python loops 一次处理一个样本。PyTorch 把同样的操作分发到优化过的 C++/CUDA kernels，并在 GPU 上运行。在单张 NVIDIA A100 上，PyTorch 可以在约 6 小时内训练 ImageNet（1.28M images）上的 ResNet-50（25.6M parameters）。你的 framework 在同一任务上大约需要 3,000 小时——前提是它没有先耗尽内存。

速度不是唯一差距。你的 framework 没有 GPU support。没有 automatic differentiation——每个 module 的 backward() 都是你手写的。没有 serialization。没有 distributed training。没有 mixed precision。没有不用 print statements 就能 debug gradient flow 的方式。

PyTorch 填补了所有这些差距。而且它保持了你已经构建出的同一个心智模型：Module、forward()、parameters()、backward()、optimizer.step()。概念一一迁移。语法几乎相同。区别是 PyTorch 把十年的 systems engineering 包在你从零设计的同一个接口后面。

## 概念

### 为什么 PyTorch 赢了

2015 年，TensorFlow 要求你在运行任何东西之前先定义 static computation graph。你构建 graph，编译它，然后把数据喂进去。调试意味着盯着 graph visualizations。修改架构意味着从零重建 graph。

PyTorch 在 2017 年带着不同哲学发布：eager execution。你写 Python，它立刻运行。`y = model(x)` 现在就实际计算 y，而不是“向稍后才会计算 y 的 graph 添加一个 node”。这意味着标准 Python debugging tools 能用。print() 能用。pdb 能用。forward pass 里的 if/else 能用。

到 2020 年，市场已经给出答案。PyTorch 在 ML 研究论文中的占比从 7%（2017）升到超过 75%（2022）。Meta、Google DeepMind、OpenAI、Anthropic 和 Hugging Face 都把 PyTorch 作为主要 framework。TensorFlow 2.x 随后采用 eager execution——等于是默认承认 PyTorch 的设计是对的。

教训是：developer experience 会复利。一个慢 10% 但调试快 50% 的 framework，每次都会赢。

### Tensors

tensor 是一个 multi-dimensional array，带有三个关键属性：shape、dtype 和 device。

```python
import torch

x = torch.zeros(3, 4)           # shape: (3, 4), dtype: float32, device: cpu
x = torch.randn(2, 3, 224, 224) # batch of 2 RGB images, 224x224
x = torch.tensor([1, 2, 3])     # from a Python list
```

**Shape** 是维度。scalar 的 shape 是 ()，vector 是 (n,)，matrix 是 (m, n)，image batch 是 (batch, channels, height, width)。

**Dtype** 控制精度和内存。

| dtype | Bits | 范围 | 使用场景 |
|-------|------|------|----------|
| float32 | 32 | 约 7 位十进制数字 | 默认训练 |
| float16 | 16 | 约 3.3 位十进制数字 | Mixed precision |
| bfloat16 | 16 | 与 float32 相同范围，精度更低 | LLM training |
| int8 | 8 | -128 到 127 | Quantized inference |

**Device** 决定计算在哪里发生。

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x = torch.randn(3, 4, device=device)
x = x.to("cuda")
x = x.cpu()
```

每个操作要求所有 tensors 位于同一 device。这是初学者最常遇到的 PyTorch 错误：`RuntimeError: Expected all tensors to be on the same device`。修复方法是在计算前把所有东西移到同一 device。

**Reshaping** 是常数时间——它改变 metadata，而不是 data。

```python
x = torch.randn(2, 3, 4)
x.view(2, 12)      # reshape to (2, 12) -- must be contiguous
x.reshape(6, 4)    # reshape to (6, 4) -- works always
x.permute(2, 0, 1) # reorder dimensions
x.unsqueeze(0)     # add dimension: (1, 2, 3, 4)
x.squeeze()        # remove size-1 dimensions
```

### Autograd

你的 mini framework 要求你为每个 module 实现 backward()。PyTorch 不需要。它把 tensors 上的每个操作记录成 directed acyclic graph（computational graph），然后反向遍历这张图来自动计算 gradients。

```mermaid
graph LR
    x["x (leaf)"] --> mul["*"]
    w["w (leaf, requires_grad)"] --> mul
    mul --> add["+"]
    b["b (leaf, requires_grad)"] --> add
    add --> loss["loss"]
    loss --> |".backward()"| add
    add --> |"grad"| b
    add --> |"grad"| mul
    mul --> |"grad"| w
```

与你的 framework 的关键差异是：PyTorch 使用 tape-based autodiff。forward pass 期间，每个操作都会追加到一条“tape”上。调用 `.backward()` 会反向 replay 这条 tape。

```python
x = torch.randn(3, requires_grad=True)
y = x ** 2 + 3 * x
z = y.sum()
z.backward()
print(x.grad)  # dz/dx = 2x + 3
```

autograd 的三条规则：

1. 只有 `requires_grad=True` 的 leaf tensors 会累积 gradients
2. Gradients 默认会累积——每次 backward pass 前调用 `optimizer.zero_grad()`
3. `torch.no_grad()` 会禁用 gradient tracking（评估时使用）

### nn.Module

`nn.Module` 是 PyTorch 中每个 neural network component 的基类。你在 Lesson 10 已经构建过这个抽象。PyTorch 版本增加了 automatic parameter registration、recursive module discovery、device management 和 state dict serialization。

```python
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x
```

当你在 `__init__` 中把 `nn.Module` 或 `nn.Parameter` 赋值为属性时，PyTorch 会自动注册它。`model.parameters()` 会递归收集每个已注册参数。这就是为什么你不需要像 mini framework 中那样手动收集 weights。

关键构建块：

| Module | 作用 | 参数量 |
|--------|------|--------|
| nn.Linear(in, out) | Wx + b | in*out + out |
| nn.Conv2d(in_ch, out_ch, k) | 2D convolution | in_ch*out_ch*k*k + out_ch |
| nn.BatchNorm1d(features) | Normalize activations | 2 * features |
| nn.Dropout(p) | Random zeroing | 0 |
| nn.ReLU() | max(0, x) | 0 |
| nn.GELU() | Gaussian error linear | 0 |
| nn.Embedding(vocab, dim) | Lookup table | vocab * dim |
| nn.LayerNorm(dim) | Per-sample normalization | 2 * dim |

### Loss Functions and Optimizers

PyTorch 提供了生产级版本的你所构建的一切。

**Loss functions**（来自 `torch.nn`）：

| Loss | 任务 | 输入 |
|------|------|------|
| nn.MSELoss() | Regression | 任意 shape |
| nn.CrossEntropyLoss() | Multi-class classification | Logits（不是 softmax） |
| nn.BCEWithLogitsLoss() | Binary classification | Logits（不是 sigmoid） |
| nn.L1Loss() | Regression（robust） | 任意 shape |
| nn.CTCLoss() | Sequence alignment | Log probabilities |

注意：`CrossEntropyLoss` 内部组合了 `LogSoftmax` + `NLLLoss`。传入 raw logits，不要传 softmax outputs。这是常见错误，会悄悄产生错误 gradients。

**Optimizers**（来自 `torch.optim`）：

| Optimizer | 使用时机 | 典型 LR |
|-----------|----------|---------|
| SGD(params, lr, momentum) | CNNs、调得很好的 pipelines | 0.01--0.1 |
| Adam(params, lr) | 默认起点 | 1e-3 |
| AdamW(params, lr, weight_decay) | Transformers、fine-tuning | 1e-4--1e-3 |
| LBFGS(params) | 小规模、二阶方法 | 1.0 |

### Training Loop

每个 PyTorch training loop 都遵循相同的 5 步模式。你在 Lesson 10 已经知道了。

```mermaid
sequenceDiagram
    participant D as DataLoader
    participant M as Model
    participant L as Loss fn
    participant O as Optimizer

    loop Each Epoch
        D->>M: batch = next(dataloader)
        M->>L: predictions = model(batch)
        L->>L: loss = criterion(predictions, targets)
        L->>M: loss.backward()
        O->>M: optimizer.step()
        O->>O: optimizer.zero_grad()
    end
```

标准模式：

```python
for epoch in range(num_epochs):
    model.train()
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
```

batch loop 里的五行。训练 GPT-4、Stable Diffusion 和 LLaMA 的也是这五行。架构会变。数据会变。这五行不会。

### Dataset and DataLoader

PyTorch 的 `Dataset` 是一个抽象类，有两个方法：`__len__` 和 `__getitem__`。`DataLoader` 用 batching、shuffling 和 multi-process data loading 包装它。

```python
from torch.utils.data import Dataset, DataLoader

class MNISTDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4)
```

`num_workers=4` 会启动 4 个进程并行加载数据，同时 GPU 训练当前 batch。在 disk-bound workloads（大图像、音频）上，这一点就能让训练速度翻倍。

### GPU Training

把 model 移到 GPU：

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
```

这会递归地把每个 parameter 和 buffer 移到 GPU。然后在训练期间移动每个 batch：

```python
inputs, targets = inputs.to(device), targets.to(device)
```

**Mixed precision** 在现代 GPU（A100、H100、RTX 4090）上通过用 float16 运行 forward/backward、同时保留 float32 master weights，可以把内存减半并让吞吐翻倍：

```python
from torch.amp import autocast, GradScaler

scaler = GradScaler()
for inputs, targets in loader:
    with autocast(device_type="cuda"):
        outputs = model(inputs)
        loss = criterion(outputs, targets)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()
```

### 对比：Mini Framework vs PyTorch vs JAX

| Feature | Mini Framework (L10) | PyTorch | JAX |
|---------|---------------------|---------|-----|
| Autodiff | Manual backward() | Tape-based autograd | Functional transforms |
| Execution | Eager (Python loops) | Eager (C++ kernels) | Traced + JIT compiled |
| GPU support | No | Yes (CUDA, ROCm, MPS) | Yes (CUDA, TPU) |
| Speed (MNIST MLP) | ~300s/epoch | ~0.5s/epoch | ~0.3s/epoch |
| Module system | Custom Module class | nn.Module | Stateless functions (Flax/Equinox) |
| Debugging | print() | print(), pdb, breakpoint() | Harder (JIT tracing breaks print) |
| Ecosystem | None | Hugging Face, Lightning, timm | Flax, Optax, Orbax |
| Learning curve | You built it | Moderate | Steep (functional paradigm) |
| Production use | Toy problems | Meta, OpenAI, Anthropic, HF | Google DeepMind, Midjourney |

## 构建

使用 PyTorch primitives 在 MNIST 上训练一个 3 层 MLP。不用高层 wrappers。不用 `torchvision.datasets`。我们自己下载并解析原始数据。

### Step 1: Load MNIST From Raw Files

MNIST 以 4 个 gzip 文件发布：training images（60,000 x 28 x 28）、training labels、test images（10,000 x 28 x 28）、test labels。我们下载它们并解析 binary format。

```python
import torch
import torch.nn as nn
import struct
import gzip
import urllib.request
import os

def download_mnist(path="./mnist_data"):
    base_url = "https://storage.googleapis.com/cvdf-datasets/mnist/"
    files = [
        "train-images-idx3-ubyte.gz",
        "train-labels-idx1-ubyte.gz",
        "t10k-images-idx3-ubyte.gz",
        "t10k-labels-idx1-ubyte.gz",
    ]
    os.makedirs(path, exist_ok=True)
    for f in files:
        filepath = os.path.join(path, f)
        if not os.path.exists(filepath):
            urllib.request.urlretrieve(base_url + f, filepath)

def load_images(filepath):
    with gzip.open(filepath, "rb") as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        data = f.read()
        images = torch.frombuffer(bytearray(data), dtype=torch.uint8)
        images = images.reshape(num, rows * cols).float() / 255.0
    return images

def load_labels(filepath):
    with gzip.open(filepath, "rb") as f:
        magic, num = struct.unpack(">II", f.read(8))
        data = f.read()
        labels = torch.frombuffer(bytearray(data), dtype=torch.uint8).long()
    return labels
```

### Step 2: Define the Model

一个 3 层 MLP：784 -> 256 -> 128 -> 10。ReLU activations。Dropout 做 regularization。为了简单，不使用 batch norm。

```python
class MNISTModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.net(x)
```

output layer 产生 10 个 raw logits（每个 digit 一个）。不加 softmax——`CrossEntropyLoss` 内部会处理。

参数量：784*256 + 256 + 256*128 + 128 + 128*10 + 10 = 235,146。按现代标准非常小。GPT-2 small 有 124M。这个模型几秒就能训练。

### Step 3: Training Loop

标准 forward-loss-backward-step 模式。

```python
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
    return total_loss / total, correct / total
```

注意 evaluation 期间的 `torch.no_grad()`。它会禁用 autograd，减少内存使用并加速 inference。没有它，PyTorch 会构建一张你永远不会用到的 computational graph。

### Step 4: Wire Everything Together

```python
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    download_mnist()
    train_images = load_images("./mnist_data/train-images-idx3-ubyte.gz")
    train_labels = load_labels("./mnist_data/train-labels-idx1-ubyte.gz")
    test_images = load_images("./mnist_data/t10k-images-idx3-ubyte.gz")
    test_labels = load_labels("./mnist_data/t10k-labels-idx1-ubyte.gz")

    train_dataset = torch.utils.data.TensorDataset(train_images, train_labels)
    test_dataset = torch.utils.data.TensorDataset(test_images, test_labels)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=64, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=256, shuffle=False
    )

    model = MNISTModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Device: {device}")
    print(f"Parameters: {num_params:,}")
    print(f"Train samples: {len(train_dataset):,}")
    print(f"Test samples: {len(test_dataset):,}")
    print()

    for epoch in range(10):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        test_loss, test_acc = evaluate(
            model, test_loader, criterion, device
        )
        print(
            f"Epoch {epoch+1:2d} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}"
        )

    torch.save(model.state_dict(), "mnist_mlp.pt")
    print(f"\nModel saved to mnist_mlp.pt")
    print(f"Final test accuracy: {test_acc:.4f}")
```

10 epochs 后预期输出：约 97.8% test accuracy。CPU 训练时间：约 30 秒。GPU：约 5 秒。你的 mini framework 用相同架构：约 45 分钟。

## 使用

### 快速对比：Mini Framework vs PyTorch

| Mini Framework（Lesson 10） | PyTorch |
|----------------------------|---------|
| `model = Sequential(Linear(784, 256), ReLU(), ...)` | `model = nn.Sequential(nn.Linear(784, 256), nn.ReLU(), ...)` |
| `pred = model.forward(x)` | `pred = model(x)` |
| `optimizer.zero_grad()` | `optimizer.zero_grad()` |
| `grad = criterion.backward()` then `model.backward(grad)` | `loss.backward()` |
| `optimizer.step()` | `optimizer.step()` |
| No GPU | `model.to("cuda")` |
| Manual backward for every module | Autograd handles everything |

接口几乎相同。差异都在底层。

### Saving and Loading Models

```python
torch.save(model.state_dict(), "model.pt")

model = MNISTModel()
model.load_state_dict(torch.load("model.pt", weights_only=True))
model.eval()
```

始终保存 `state_dict()`（parameter dictionary），不要保存 model object。保存 model object 会使用 pickle，代码重构后容易坏。State dicts 更可移植。

### Learning Rate Scheduling

```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=10
)
for epoch in range(10):
    train_one_epoch(model, train_loader, criterion, optimizer, device)
    scheduler.step()
```

PyTorch 自带 15+ schedulers：StepLR、ExponentialLR、CosineAnnealingLR、OneCycleLR、ReduceLROnPlateau。它们都接入同一个 optimizer interface。

## 交付

本课会产出两个 artifacts：

- `outputs/prompt-pytorch-debugger.md`——用于诊断常见 PyTorch training failures 的 prompt
- `outputs/skill-pytorch-patterns.md`——PyTorch training patterns 的技能参考

## 练习

1. **添加 batch normalization。** 在每个 linear layer 之后（activation 之前）插入 `nn.BatchNorm1d`。比较它与只用 dropout 版本的 test accuracy 和 training speed。batch norm 应该能用更少 epochs 达到 98%+。

2. **实现 learning rate finder。** 用指数增长的 learning rate（从 1e-7 到 1.0）训练一个 epoch。绘制 loss vs LR。最佳 LR 位于 loss 开始攀升之前。用它为 MNIST model 选择更好的 LR。

3. **用 mixed precision 移植到 GPU。** 给 training loop 添加 `torch.amp.autocast` 和 `GradScaler`。在 GPU 上测量使用和不使用 mixed precision 的 throughput（samples/second）。在 A100 上预期约 2x 提速。

4. **构建 custom Dataset。** 下载 Fashion-MNIST（与 MNIST 格式相同，但内容是服装）。实现一个带 `__getitem__` 和 `__len__` 的 `FashionMNISTDataset(Dataset)` class。训练同一个 MLP 并比较 accuracy。Fashion-MNIST 更难——预期约 88%，而不是约 98%。

5. **用 SGD + momentum 替换 Adam。** 使用 `SGD(params, lr=0.01, momentum=0.9)` 训练。比较 convergence curves。然后添加 `CosineAnnealingLR` scheduler，看看 SGD 到 epoch 10 时是否追上 Adam。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|----------|----------|
| Tensor | “多维数组” | 一个带类型和 device 感知的数组，每个操作都内置 automatic differentiation support |
| Autograd | “自动 backprop” | 一个 tape-based 系统，在 forward pass 期间记录操作，再反向 replay 以计算精确 gradients |
| nn.Module | “一层” | 任意 differentiable computation block 的基类——注册 parameters、支持嵌套、处理 train/eval modes |
| state_dict | “模型 weights” | 将 parameter names 映射到 tensors 的 OrderedDict——训练后模型的可移植、可序列化表示 |
| .backward() | “计算 gradients” | 反向遍历 computational graph，为每个 requires_grad=True 的 leaf tensor 计算并累积 gradients |
| .to(device) | “移动到 GPU” | 将所有 parameters 和 buffers 递归传输到指定 device（CPU、CUDA、MPS） |
| DataLoader | “数据管道” | 一个从 Dataset 批处理、shuffle，并可选地并行加载数据的 iterator |
| Mixed precision | “使用 float16” | 用 float16 forward/backward 加速训练，同时保留 float32 master weights 以维持数值稳定 |
| Eager execution | “现在就运行” | 操作被调用时立即执行，而不是延迟到之后的 compilation step——这是 PyTorch 区别于 TF 1.x 的核心设计选择 |
| zero_grad | “重置 gradients” | 在下一次 backward pass 前把所有 parameter gradients 设为 0，因为 PyTorch 默认累积 gradients |

## 延伸阅读

- Paszke et al., "PyTorch: An Imperative Style, High-Performance Deep Learning Library" (2019)——解释 PyTorch 设计权衡的原始论文
- PyTorch Tutorials: "Learning PyTorch with Examples" (https://pytorch.org/tutorials/beginner/pytorch_with_examples.html)——从 tensors 到 nn.Module 的官方路径
- PyTorch Performance Tuning Guide (https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html)——mixed precision、DataLoader workers、pinned memory 和其他生产优化
- Horace He, "Making Deep Learning Go Brrrr" (https://horace.io/brrr_intro.html)——解释为什么 GPU training 很快，并包含 PyTorch-specific optimization strategies
