# 调试与性能分析

> 最糟糕的 AI bug 不会崩溃。它们会默默用垃圾数据训练，然后给你一条漂亮的 loss 曲线。

**类型：** 构建
**语言：** Python
**前置要求：** 第 1 课（开发环境）、基础 PyTorch 熟悉度
**时间：** ~60 分钟

## 学习目标

- 使用条件 `breakpoint()` 和 `debug_print` 在训练中检查张量 shape、dtype 和 NaN 值
- 使用 `cProfile`、`line_profiler` 和 `tracemalloc` 分析训练循环，找出瓶颈
- 检测常见 AI bug：shape mismatch、NaN loss、data leakage 和 wrong-device tensors
- 设置 TensorBoard，可视化 loss 曲线、权重直方图和梯度分布

## 问题

AI 代码的失败方式和普通代码不同。Web app 会带着 stack trace 崩溃。配置错误的训练循环会运行 8 小时，烧掉 200 美元 GPU 时间，然后产出一个对所有输入都预测均值的模型。代码从没报错。bug 是某个张量在错误设备上、忘记 `.detach()`，或标签泄漏进了特征。

你需要能在这些 silent failures 浪费时间和算力之前抓住它们的调试工具。

## 概念

AI 调试有三个层级：

```mermaid
graph TD
    L3["3. Training Dynamics<br/>Loss curves, gradient norms, activations"] --> L2
    L2["2. Tensor Operations<br/>Shapes, dtypes, devices, NaN/Inf values"] --> L1
    L1["1. Standard Python<br/>Breakpoints, logging, profiling, memory"]
```

多数人会直接跳到第 3 层（盯着 TensorBoard）。但 80% 的 AI bug 活在第 1 层和第 2 层。

## 构建它

### 第 1 部分：Print Debugging（是的，它有用）

Print debugging 经常被轻视。它不该如此。对张量代码来说，有针对性的 print 语句比单步 debugger 更强，因为你需要同时看到 shape、dtype 和取值范围。

```python
def debug_print(name, tensor):
    print(f"{name}: shape={tensor.shape}, dtype={tensor.dtype}, "
          f"device={tensor.device}, "
          f"min={tensor.min().item():.4f}, max={tensor.max().item():.4f}, "
          f"mean={tensor.mean().item():.4f}, "
          f"has_nan={tensor.isnan().any().item()}")
```

在每个可疑操作之后调用它。找到 bug 后，删掉这些 prints。简单有效。

### 第 2 部分：Python Debugger（pdb 和 breakpoint）

内置 debugger 在 AI 工作中被低估了。把 `breakpoint()` 放进训练循环，然后交互式检查张量。

```python
def training_step(model, batch, criterion, optimizer):
    inputs, labels = batch
    outputs = model(inputs)
    loss = criterion(outputs, labels)

    if loss.item() > 100 or torch.isnan(loss):
        breakpoint()

    loss.backward()
    optimizer.step()
```

当 debugger 停住时，常用命令：

- `p outputs.shape` 检查 shape
- `p loss.item()` 查看 loss 值
- `p torch.isnan(outputs).sum()` 统计 NaN 数量
- `p model.fc1.weight.grad` 检查梯度
- `c` 继续，`q` 退出

这就是条件调试。只有当某些东西看起来不对时才停止。对 10,000 step 的训练运行来说，这很重要。

### 第 3 部分：Python Logging

当调试不再只是快速检查时，用 logging 替代 print 语句。

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

logger.info("Starting training: lr=%.4f, batch_size=%d", lr, batch_size)
logger.warning("Loss spike detected: %.4f at step %d", loss.item(), step)
logger.error("NaN loss at step %d, stopping", step)
```

Logging 给你时间戳、严重级别和文件输出。当训练运行在凌晨 3 点失败时，你想要的是日志文件，而不是早就滚出屏幕的终端输出。

### 第 4 部分：为代码段计时

知道时间花在哪里，是优化的第一步。

```python
import time

class Timer:
    def __init__(self, name=""):
        self.name = name

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        elapsed = time.perf_counter() - self.start
        print(f"[{self.name}] {elapsed:.4f}s")

with Timer("data loading"):
    batch = next(dataloader_iter)

with Timer("forward pass"):
    outputs = model(batch)

with Timer("backward pass"):
    loss.backward()
```

常见发现：数据加载占了训练时间的 60%。修复方法是 DataLoader 中的 `num_workers > 0`，不是换更快的 GPU。

### 第 5 部分：cProfile 和 line_profiler

当你需要的不只是手动 timer：

```bash
python -m cProfile -s cumtime train.py
```

这会显示每个函数调用，并按累计时间排序。要做逐行 profiling：

```bash
pip install line_profiler
```

```python
@profile
def train_step(model, data, target):
    output = model(data)
    loss = F.cross_entropy(output, target)
    loss.backward()
    return loss

# Run with: kernprof -l -v train.py
```

### 第 6 部分：内存分析

#### 使用 tracemalloc 分析 CPU 内存

```python
import tracemalloc

tracemalloc.start()

# your code here
model = build_model()
data = load_dataset()

snapshot = tracemalloc.take_snapshot()
top_stats = snapshot.statistics("lineno")
for stat in top_stats[:10]:
    print(stat)
```

#### 使用 memory_profiler 分析 CPU 内存

```bash
pip install memory_profiler
```

```python
from memory_profiler import profile

@profile
def load_data():
    raw = read_csv("data.csv")       # watch memory jump here
    processed = preprocess(raw)       # and here
    return processed
```

用 `python -m memory_profiler your_script.py` 运行，查看逐行内存使用。

#### 使用 PyTorch 分析 GPU 内存

```python
import torch

if torch.cuda.is_available():
    print(torch.cuda.memory_summary())

    print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    print(f"Cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
```

当你遇到 OOM（Out of Memory）：

1. 减小 batch size（永远是第一件要试的事）
2. 使用 `torch.cuda.empty_cache()` 释放缓存内存
3. 对大型中间变量使用 `del tensor`，然后调用 `torch.cuda.empty_cache()`
4. 使用 mixed precision（`torch.cuda.amp`）将内存使用减半
5. 对很深的模型使用 gradient checkpointing

### 第 7 部分：常见 AI Bug 以及如何抓住它们

#### Shape Mismatch

最常见的 bug。一个张量的 shape 是 `[batch, features]`，但模型期待 `[batch, channels, height, width]`。

```python
def check_shapes(model, sample_input):
    print(f"Input: {sample_input.shape}")
    hooks = []

    def make_hook(name):
        def hook(module, inp, out):
            in_shape = inp[0].shape if isinstance(inp, tuple) else inp.shape
            out_shape = out.shape if hasattr(out, "shape") else type(out)
            print(f"  {name}: {in_shape} -> {out_shape}")
        return hook

    for name, module in model.named_modules():
        hooks.append(module.register_forward_hook(make_hook(name)))

    with torch.no_grad():
        model(sample_input)

    for h in hooks:
        h.remove()
```

用一个样本 batch 运行一次。它会映射模型里的每一次 shape 变换。

#### NaN Loss

NaN loss 表示某些东西爆炸了。常见原因：

- Learning rate 太高
- 自定义 loss 中除以零
- 对零或负数取 log
- RNN 中梯度爆炸

```python
def detect_nan(model, loss, step):
    if torch.isnan(loss):
        print(f"NaN loss at step {step}")
        for name, param in model.named_parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any():
                    print(f"  NaN gradient in {name}")
                if torch.isinf(param.grad).any():
                    print(f"  Inf gradient in {name}")
        return True
    return False
```

#### Data Leakage

你的模型在 test set 上达到 99% accuracy。听起来很棒。它是 bug。

```python
def check_data_leakage(train_set, test_set, id_column="id"):
    train_ids = set(train_set[id_column].tolist())
    test_ids = set(test_set[id_column].tolist())
    overlap = train_ids & test_ids
    if overlap:
        print(f"DATA LEAKAGE: {len(overlap)} samples in both train and test")
        return True
    return False
```

还要检查 temporal leakage：用未来数据预测过去。划分前先按时间戳排序。

#### Wrong Device

不同设备上的张量（CPU vs GPU）会导致运行时错误。但有时某个张量会默默留在 CPU 上，而其他东西都在 GPU 上，训练只是变慢。

```python
def check_devices(model, *tensors):
    model_device = next(model.parameters()).device
    print(f"Model device: {model_device}")
    for i, t in enumerate(tensors):
        if t.device != model_device:
            print(f"  WARNING: tensor {i} on {t.device}, model on {model_device}")
```

### 第 8 部分：TensorBoard 基础

TensorBoard 会显示训练随时间发生了什么。

```bash
pip install tensorboard
```

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("runs/experiment_1")

for step in range(num_steps):
    loss = train_step(model, batch)

    writer.add_scalar("loss/train", loss.item(), step)
    writer.add_scalar("lr", optimizer.param_groups[0]["lr"], step)

    if step % 100 == 0:
        for name, param in model.named_parameters():
            writer.add_histogram(f"weights/{name}", param, step)
            if param.grad is not None:
                writer.add_histogram(f"grads/{name}", param.grad, step)

writer.close()
```

启动它：

```bash
tensorboard --logdir=runs
```

需要观察什么：

- **Loss 不下降**：Learning rate 太低，或模型架构有问题
- **Loss 剧烈震荡**：Learning rate 太高
- **Loss 变成 NaN**：数值不稳定（见上面的 NaN 部分）
- **Train loss 下降，val loss 上升**：过拟合
- **Weight histograms 塌缩到零**：梯度消失
- **Gradient histograms 爆炸**：需要 gradient clipping

### 第 9 部分：VS Code Debugger

对于交互式调试，用 `launch.json` 配置 VS Code：

```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Debug Training",
            "type": "debugpy",
            "request": "launch",
            "program": "${file}",
            "console": "integratedTerminal",
            "justMyCode": false
        }
    ]
}
```

点击 gutter 设置断点。使用 Variables 面板检查张量属性。Debug Console 可以在执行中运行任意 Python 表达式。

这对单步查看数据预处理流水线很有用，因为你想看到每一步变换。

## 使用它

下面这个调试工作流可以抓住大多数 AI bug：

1. **训练前**：用样本 batch 运行 `check_shapes`。验证输入和输出维度符合预期。
2. **前 10 step**：对 loss、outputs 和 gradients 使用 `debug_print`。确认没有 NaN，并且数值处在合理范围。
3. **训练中**：记录 loss、learning rate 和 gradient norms。用 TensorBoard 可视化。
4. **出问题时**：在失败点放 `breakpoint()`。交互式检查张量。
5. **性能问题**：分别计时数据加载、forward 和 backward pass。如果接近 OOM，就分析内存。

## 交付它

运行调试工具脚本：

```bash
python phases/00-setup-and-tooling/12-debugging-and-profiling/code/debug_tools.py
```

查看 `outputs/prompt-debug-ai-code.md`，它提供了一个帮助诊断 AI 特定 bug 的提示词。

## 练习

1. 运行 `debug_tools.py`，阅读每个部分的输出。修改 dummy model 引入一个 NaN（提示：在 forward pass 中除以零），观察 detector 抓住它。
2. 使用 `cProfile` 分析一个训练循环，并找出最慢的函数。
3. 使用 `tracemalloc` 找出数据加载流水线中哪一行分配了最多内存。
4. 为一个简单训练运行设置 TensorBoard，并判断模型是否过拟合。
5. 在训练循环中使用 `breakpoint()`。练习从 debugger prompt 中检查张量 shape、device 和梯度值。
