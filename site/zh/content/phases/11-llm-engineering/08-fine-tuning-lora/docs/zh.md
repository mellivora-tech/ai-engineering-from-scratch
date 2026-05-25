# 使用 LoRA 与 QLoRA 进行 Fine-Tuning

> Full fine-tuning 一个 7B 模型需要 56GB VRAM。你没有。大多数公司也没有。LoRA 通过训练不到 1% 的参数，让你用 6GB 就能 fine-tune 同一个模型。这不是妥协，在多数任务上它能匹配 full fine-tuning 的质量。整个开源 fine-tuning 生态都建立在这个技巧之上。

**类型：** 构建
**语言：** Python
**前置要求：** Phase 10，Lesson 06（Instruction Tuning / SFT）
**时间：** 约 75 分钟
**相关：** Phase 10 从零覆盖 SFT/DPO loops。本课把它们接入 2026 年的 PEFT toolkits（PEFT、TRL、Unsloth、Axolotl、LLaMA-Factory）。

## 学习目标

- 通过向 pretrained model 的 attention layers 注入 low-rank adapter matrices（A 和 B）实现 LoRA
- 计算 LoRA 相对 full fine-tuning 的参数节省：rank r 与 d_model 维度下，训练 2*r*d 参数，而不是 d^2
- 使用 QLoRA（4-bit quantized base + LoRA adapters）fine-tune 模型，使其适配消费级 GPU 显存
- 将 LoRA weights 合并回 base model 用于部署，并比较有无 adapters 时的 inference speed

## 问题

你有一个 base model。Llama 3 8B。你想让它用公司的语气回复 customer support tickets。SFT 是答案。但 SFT 有成本问题。

Full fine-tuning 会更新模型中的每个参数。Llama 3 8B 有 80 亿参数。fp16 下，每个参数占 2 bytes。仅加载 weights 就需要 16GB。训练期间还需要 gradients（16GB）、Adam optimizer states（momentum + variance 需要 32GB）以及 activations。总计：单个 8B 模型大约需要 56GB VRAM。

A100 80GB 勉强能放下。两张 A100 在 cloud providers 上每小时 $3-4。用 50,000 个 examples 训练 3 epochs 需要 6 到 10 小时。每次实验 $30-40。为了调好 hyperparameters 跑 10 次实验，在部署任何东西之前就花了 $400。

扩展到 Llama 3 70B 后数字会荒谬起来。光 weights 就 140GB。你需要一个 cluster。每次实验 $100+。

还有一个更深的问题。Full fine-tuning 会修改模型中的每个 weight。如果你在 customer support data 上 fine-tune，可能会损伤模型的通用能力。这叫 catastrophic forgetting。模型在你的任务上变好，在其他事情上变差。

你需要一种方法：训练更少参数、使用更少内存，并且不会破坏模型已有知识。

## 概念

### LoRA：Low-Rank Adaptation

Microsoft 的 Edward Hu 和同事在 2021 年 6 月发表了 LoRA。论文的洞察是：fine-tuning 期间的 weight updates 具有低 intrinsic rank。你不需要更新一个 4096x4096 weight matrix 中全部 1670 万参数。更新中的有用信息可以由 rank 16 或 32 的矩阵捕获。

数学如下。标准 linear layer 计算：

```
y = Wx
```

其中 W 是 d_out x d_in 矩阵。对于 4096x4096 的 attention projection，它有 16,777,216 个参数。

LoRA 冻结 W，并添加一个 low-rank decomposition：

```
y = Wx + BAx
```

其中 B 是 (d_out x r)，A 是 (r x d_in)。rank r 远小于 d，通常是 8、16 或 32。

对于 4096x4096 layer 上的 r=16：
- 原始参数：4096 x 4096 = 16,777,216
- LoRA 参数：(4096 x 16) + (16 x 4096) = 65,536 + 65,536 = 131,072
- 降幅：131,072 / 16,777,216 = 0.78%

你只训练 0.78% 的参数，却获得 95% 到 100% 的质量。

```mermaid
graph LR
    X["Input x"] --> W["Frozen W (d x d)"]
    X --> A["A (r x d)"]
    A --> B["B (d x r)"]
    W --> Plus["+ (merge)"]
    B --> Plus
    Plus --> Y["Output y"]

    style W fill:#1a1a2e,stroke:#e94560,color:#fff
    style A fill:#0f3460,stroke:#16213e,color:#fff
    style B fill:#0f3460,stroke:#16213e,color:#fff
```

A 用随机 Gaussian 初始化。B 初始化为零。这意味着 LoRA contribution 从零开始，模型从原始行为开始训练，然后逐步学习 adaptation。

### Scaling Factor：Alpha

LoRA 引入 scaling factor alpha，用于控制 low-rank update 对输出的影响：

```
y = Wx + (alpha / r) * BAx
```

当 alpha = r 时，scaling 是 1x。当 alpha = 2r（常见默认值）时，scaling 是 2x。这个 hyperparameter 独立于 base learning rate，控制 LoRA path 的学习率。

实践建议：
- alpha = 2 * rank 是常见社区惯例（原论文在多数实验中使用 alpha = rank）
- alpha = rank 给出 1x scaling，保守但稳定
- 更高 alpha 意味着每步更新更大，可能加快收敛，也可能造成不稳定

### LoRA 应用到哪里

Transformer 有许多 linear layers。你不需要给所有层都加 LoRA。原论文测试了不同组合：

| Target Layers | Trainable Params (7B) | Quality |
|--------------|----------------------|---------|
| q_proj only | 4.7M | Good |
| q_proj + v_proj | 9.4M | Better |
| q_proj + k_proj + v_proj + o_proj | 18.9M | Best for attention |
| All linear (attention + MLP) | 37.7M | Marginal gain, 2x params |

多数任务的甜点区间：q_proj + v_proj。它针对 self-attention 中的 query 和 value projections，控制模型关注什么以及提取什么信息。加入 MLP layers 有助于代码生成等复杂任务，但参数量翻倍，在简单任务上收益递减。

### Rank 选择

rank r 控制 adaptation 的表达能力：

| Rank | Trainable Params (per layer) | Best For |
|------|---------------------------|----------|
| 4 | 32,768 | 简单分类、情感分析 |
| 8 | 65,536 | 单领域 Q&A、summarization |
| 16 | 131,072 | 多领域任务、instruction following |
| 32 | 262,144 | 复杂推理、代码生成 |
| 64 | 524,288 | 多数任务收益递减 |
| 128 | 1,048,576 | 很少值得 |

Hu 等人展示，r=4 已经能捕获简单任务中的大部分 adaptation。r=8 和 r=16 是实践中最常见的选择。超过 r=64 很少改善质量，并开始失去 LoRA 的内存优势。

### QLoRA：4-Bit Quantization + LoRA

University of Washington 的 Tim Dettmers 和同事在 2023 年 5 月发表了 QLoRA。想法是：把 frozen base model 量化到 4-bit precision，然后在上面接 fp16 的 LoRA adapters。

这会显著改变内存公式：

| Method | Weight Memory (7B) | Training Memory (7B) | GPU Required |
|--------|-------------------|---------------------|-------------|
| Full fine-tune (fp16) | 14GB | ~56GB | 1x A100 80GB |
| LoRA (fp16 base) | 14GB | ~18GB | 1x A100 40GB |
| QLoRA (4-bit base) | 3.5GB | ~6GB | 1x RTX 3090 24GB |

QLoRA 有三项技术贡献：

**NF4（Normal Float 4-bit）**：专为 neural network weights 设计的新数据类型。神经网络权重近似服从 normal distribution。NF4 把 16 个 quantization levels 放在标准正态分布的分位点上。对于正态分布数据，这是信息论最优的。它比 uniform 4-bit quantization（INT4）或标准 Float4 损失更少信息。

**Double quantization**：quantization constants 本身也占内存。每 64 个 weights 的 block 需要一个 fp32 scale factor（4 bytes）。对 7B 模型来说，这是额外 0.4GB。Double quantization 把这些 constants 量化为 fp8，将开销降到 0.1GB。很小，但会累积。

**Paged optimizers**：训练时，optimizer states（Adam 的 momentum 和 variance）在长序列上可能超过 GPU memory。Paged optimizers 使用 NVIDIA unified memory，在 GPU memory 耗尽时自动把 optimizer states page 到 CPU RAM，需要时再 page 回来。这能防止 OOM crashes，代价是一些 throughput。

### 质量问题

减少参数或量化 base 是否会损害质量？多篇论文的结果：

| Method | MMLU (5-shot) | MT-Bench | HumanEval |
|--------|--------------|----------|-----------|
| Full fine-tune (Llama 2 7B) | 48.3 | 6.72 | 14.6 |
| LoRA r=16 | 47.9 | 6.68 | 14.0 |
| QLoRA r=16 (NF4) | 47.5 | 6.61 | 13.4 |
| QLoRA r=64 (NF4) | 48.1 | 6.70 | 14.2 |

LoRA r=16 在多数 benchmark 上与 full fine-tuning 相差不到 1%。QLoRA r=16 又损失零点几个百分点。QLoRA r=64 基本匹配 full fine-tuning，同时少用 90% 内存。

### 真实世界成本

在 50,000 个 examples 上 fine-tune Llama 3 8B（3 epochs）：

| Method | GPU | Time | Cost |
|--------|-----|------|------|
| Full fine-tune | 2x A100 80GB | 8 hours | ~$32 |
| LoRA r=16 | 1x A100 40GB | 4 hours | ~$8 |
| QLoRA r=16 | 1x RTX 4090 24GB | 6 hours | ~$5 |
| QLoRA r=16 (Unsloth) | 1x RTX 4090 24GB | 2.5 hours | ~$2 |
| QLoRA r=16 | 1x T4 16GB | 12 hours | ~$4 |

在单张消费级 GPU 上使用 QLoRA，成本低于一顿午饭。这就是为什么 open-weight fine-tuning community 在 2023 年爆发，也解释了为什么下面每个 training framework 到 2026 年都默认支持 QLoRA。

### 2026 PEFT stack

| Framework | 它是什么 | 何时选择 |
|-----------|-----------|-----------|
| **Hugging Face PEFT** | 标准 LoRA/QLoRA/DoRA/IA3 library | 你想要原始控制，并且 training loop 已经基于 `transformers.Trainer` |
| **TRL** | HF 的 reinforcement-from-feedback trainers（SFT、DPO、GRPO、PPO、ORPO） | SFT 后还需要 DPO/GRPO；构建在 PEFT 之上 |
| **Unsloth** | forward/backward pass 的 Triton-kernel rewrite | 你想要 2-5x 加速 + 一半 VRAM 且无准确率损失；Llama/Mistral/Qwen family |
| **Axolotl** | PEFT + TRL + DeepSpeed + Unsloth 之上的 YAML-config wrapper | 你想要可复现、版本控制的 training runs |
| **LLaMA-Factory** | PEFT + TRL 之上的 GUI/CLI/API | 你想要 zero-code fine-tuning；支持 100+ model families |
| **torchtune** | 原生 PyTorch recipes，没有 `transformers` 依赖 | 你想要最少依赖，且组织已经标准化在 PyTorch 上 |

经验法则：研究或一次性实验 → PEFT。可重复生产 pipeline → 启用 Unsloth kernels 的 Axolotl。临时原型 → LLaMA-Factory。

### Merging Adapters

训练后，你有两样东西：frozen base model 和一个小 LoRA adapter（通常 10-100MB）。你可以：

1. **保持分离**：加载 base model，再在上面加载 adapter。为不同任务切换 adapters。这是从一个 base model 服务多个 fine-tuned variants 的方式。

2. **永久合并**：计算 W' = W + (alpha/r) * BA，并把结果保存为新的 full model。Merged model 与原模型同尺寸。没有 inference overhead。没有 adapter 需要管理。

服务多个任务（customer support adapter、code adapter、translation adapter）时，保持分离。部署单个专用模型时，合并。

用于组合多个 adapters 的高级 merging techniques：

- **TIES-Merging**（Yadav et al. 2023）：修剪小幅度参数，解决 sign conflicts，然后合并。减少 adapters 之间的 interference。
- **DARE**（Yu et al. 2023）：合并前随机丢弃 adapter 参数，并重新缩放剩余部分。组合能力时意外有效。
- **Task arithmetic**：简单加减 adapter weights。把 “code” adapter 和 “math” adapter 相加，通常会产生一个两者都擅长的模型。

### 何时不要 Fine-Tune

Fine-tuning 是第三选择，不是第一选择。

**第一：prompt engineering。** 写更好的 system prompt。加入 few-shot examples。使用 chain-of-thought。这不花钱，只需要几分钟。如果 prompting 已经帮你达到 80%，你可能不需要 fine-tune。

**第二：RAG。** 如果模型需要了解你的特定数据（documents、knowledge base、product catalog），retrieval 比把知识烘进 weights 更便宜、更可维护。见 Lesson 06。

**第三：fine-tuning。** 当你需要模型采用特定 style、format 或 reasoning pattern，而且无法通过 prompting 实现时使用它。当你需要稳定的 structured output 时使用它。当你需要把大模型 distill 成小模型时使用它。当 latency 很重要，无法承受 few-shot prompting 的额外 tokens 时使用它。

```mermaid
graph TD
    Start["Need better model behavior?"] --> PE["Try prompt engineering"]
    PE -->|"Works"| Done["Ship it"]
    PE -->|"Not enough"| RAG["Need external knowledge?"]
    RAG -->|"Yes"| RAGBuild["Build RAG pipeline"]
    RAG -->|"No, need style/format change"| FT["Fine-tune with LoRA/QLoRA"]
    RAGBuild -->|"Works"| Done
    RAGBuild -->|"Also need style change"| FT
    FT --> Done

    style Start fill:#1a1a2e,stroke:#e94560,color:#fff
    style Done fill:#0f3460,stroke:#16213e,color:#fff
```

## 构建

我们用纯 PyTorch 从零实现 LoRA。没有库。没有魔法。你会构建 LoRA layer，把它注入模型，训练它，然后把 weights 合并回去。

### Step 1：LoRA Layer

```python
import torch
import torch.nn as nn
import math

class LoRALayer(nn.Module):
    def __init__(self, in_features, out_features, rank=8, alpha=16):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        self.A = nn.Parameter(torch.randn(in_features, rank) * (1 / math.sqrt(rank)))
        self.B = nn.Parameter(torch.zeros(rank, out_features))

    def forward(self, x):
        return (x @ self.A @ self.B) * self.scaling
```

A 用 scaled random values 初始化。B 初始化为零。乘积 BA 从零开始，因此模型以原始行为开始。

### Step 2：LoRA-Wrapped Linear Layer

```python
class LinearWithLoRA(nn.Module):
    def __init__(self, linear, rank=8, alpha=16):
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(
            linear.in_features, linear.out_features, rank, alpha
        )

        for param in self.linear.parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.linear(x) + self.lora(x)
```

原始 linear layer 被冻结。只有 LoRA 参数（A 和 B）可训练。

### Step 3：Inject LoRA into a Model

```python
def inject_lora(model, target_modules, rank=8, alpha=16):
    for param in model.parameters():
        param.requires_grad = False

    lora_layers = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if any(t in name for t in target_modules):
                parent_name = ".".join(name.split(".")[:-1])
                child_name = name.split(".")[-1]
                parent = dict(model.named_modules())[parent_name]
                lora_linear = LinearWithLoRA(module, rank, alpha)
                setattr(parent, child_name, lora_linear)
                lora_layers[name] = lora_linear
    return lora_layers
```

首先冻结模型中的每个参数。然后遍历 model tree，找到匹配目标名称的 linear layers，并替换为 LoRA-wrapped versions。LoRA A 和 B 矩阵是整个模型中唯一可训练参数。

### Step 4：Count Parameters

```python
def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    return {
        "total": total,
        "trainable": trainable,
        "frozen": frozen,
        "trainable_pct": 100 * trainable / total if total > 0 else 0
    }
```

### Step 5：Merge Weights Back

```python
def merge_lora_weights(model):
    for name, module in model.named_modules():
        if isinstance(module, LinearWithLoRA):
            with torch.no_grad():
                merged = (
                    module.lora.A @ module.lora.B
                ) * module.lora.scaling
                module.linear.weight.data += merged.T
            parent_name = ".".join(name.split(".")[:-1])
            child_name = name.split(".")[-1]
            if parent_name:
                parent = dict(model.named_modules())[parent_name]
            else:
                parent = model
            setattr(parent, child_name, module.linear)
```

合并后，LoRA layers 消失。模型尺寸与原始模型相同，adaptation 被烘进 weights。没有 inference overhead。

### Step 6：Simulated QLoRA Quantization

```python
def quantize_to_nf4(tensor, block_size=64):
    blocks = tensor.reshape(-1, block_size)
    scales = blocks.abs().max(dim=1, keepdim=True).values / 7.0
    scales = torch.clamp(scales, min=1e-8)
    quantized = torch.round(blocks / scales).clamp(-8, 7).to(torch.int8)
    return quantized, scales

def dequantize_from_nf4(quantized, scales, original_shape):
    dequantized = quantized.float() * scales
    return dequantized.reshape(original_shape)
```

这通过把 weights 映射到每个 64 block 内的 16 个离散级别，模拟 4-bit quantization。生产 QLoRA 使用 bitsandbytes library 在 GPU 上做真正的 NF4。

### Step 7：Training Loop

```python
def train_lora(model, data, epochs=5, lr=1e-3, batch_size=4):
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr
    )
    criterion = nn.MSELoss()

    losses = []
    for epoch in range(epochs):
        epoch_loss = 0.0
        n_batches = 0
        indices = torch.randperm(len(data["inputs"]))

        for i in range(0, len(indices), batch_size):
            batch_idx = indices[i:i + batch_size]
            x = data["inputs"][batch_idx]
            y = data["targets"][batch_idx]

            output = model(x)
            loss = criterion(output, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / n_batches
        losses.append(avg_loss)

    return losses
```

### Step 8：完整 Demo

```python
def demo():
    torch.manual_seed(42)
    d_model = 256
    n_classes = 10

    model = nn.Sequential(
        nn.Linear(d_model, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, n_classes),
    )

    n_samples = 500
    x = torch.randn(n_samples, d_model)
    y = torch.randint(0, n_classes, (n_samples,))
    y_onehot = torch.zeros(n_samples, n_classes).scatter_(1, y.unsqueeze(1), 1.0)

    data = {"inputs": x, "targets": y_onehot}

    params_before = count_parameters(model)

    lora_layers = inject_lora(
        model, target_modules=["0", "2"], rank=8, alpha=16
    )

    params_after = count_parameters(model)

    losses = train_lora(model, data, epochs=20, lr=1e-3)

    merge_lora_weights(model)
    params_merged = count_parameters(model)

    return {
        "params_before": params_before,
        "params_after": params_after,
        "params_merged": params_merged,
        "losses": losses,
    }
```

Demo 创建一个小模型，向两层注入 LoRA，训练它，然后把 weights 合并回去。在 LoRA training 期间，参数数量从全部可训练降到约 1% 可训练，合并后回到原始架构。

## 使用

使用 Hugging Face ecosystem，在真实模型上启用 LoRA 大约只需要 20 行：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
```

对于 QLoRA，加入 bitsandbytes quantization：

```python
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B",
    quantization_config=bnb_config,
    device_map="auto",
)

model = get_peft_model(model, lora_config)
```

就这些。同一套 training loop。同一套 data pipeline。Base model 现在以 4-bit 存在，LoRA adapters 用 fp16 训练，整个系统可以放进 6GB。

使用 Hugging Face Trainer 训练：

```python
from transformers import TrainingArguments, Trainer
from datasets import load_dataset

dataset = load_dataset("tatsu-lab/alpaca", split="train[:5000]")

training_args = TrainingArguments(
    output_dir="./lora-llama",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    save_strategy="epoch",
    optim="paged_adamw_8bit",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)

trainer.train()

model.save_pretrained("./lora-adapter")
```

保存下来的 adapter 是 10-100MB。Base model 保持不变。你可以在 Hugging Face Hub 上分享 adapters，而不必重新分发整个模型。

## 交付

本课会产出：
- `outputs/prompt-lora-advisor.md`：一个 prompt，用于帮助你为特定任务决定 LoRA rank、target modules 和 hyperparameters
- `outputs/skill-fine-tuning-guide.md`：一个 skill，教 agent 何时以及如何 fine-tune 的决策树

## 练习

1. **Rank ablation study。** 用 ranks 2、4、8、16、32 和 64 运行 demo。绘制 final loss vs. rank。找到收益递减点，也就是 rank 翻倍不再让 loss 减半的位置。对 256-dim features 上的简单分类任务，这应该在 r=8-16 附近。

2. **Target module comparison。** 修改 `inject_lora`，分别只 target layer "0"、只 target layer "2"、只 target layer "4"，以及 target 全部三层。每个变体训练 20 epochs。比较 convergence speed 和 final loss。这对应真实场景中选择 q_proj、v_proj 或 all linear layers 的决策。

3. **Quantization error analysis。** 对训练后模型的 weight matrices，在 `quantize_to_nf4` / `dequantize_from_nf4` 前后计算 mean squared error、max absolute error，以及原始 weights 与重建 weights 之间的 correlation。尝试 block_size 为 32、64、128 和 256。

4. **Multi-adapter serving。** 在数据的不同子集上训练两个 LoRA adapters（even indices vs odd indices）。保存两个 adapters。只加载一次 base model，然后切换 adapters，验证同一输入会产生不同输出。这就是生产系统从一个 base 服务多个 fine-tuned models 的方式。

5. **Merge vs. unmerged inference。** 对同样的 100 个输入，比较 LoRA model 在 `merge_lora_weights` 前后的输出。验证输出相同（在 1e-5 floating-point tolerance 内）。然后 benchmark 两者的 inference speed，merged 应略快，因为它是单次 matrix multiply，而不是两次。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|----------------|----------------------|
| LoRA | “Efficient fine-tuning” | Low-Rank Adaptation：冻结 base weights，训练两个小矩阵 A 和 B，其乘积近似完整 weight update |
| QLoRA | “在笔记本上 fine-tune” | Quantized LoRA：以 4-bit NF4 加载 base model，在上面用 fp16 训练 LoRA adapters，让 7B fine-tuning 适配 6GB VRAM |
| Rank (r) | “模型能学多少” | A 和 B 矩阵的内部维度；控制表达能力与参数数量之间的权衡 |
| Alpha | “LoRA learning rate” | 应用于 LoRA output 的 scaling factor；alpha/r 会缩放 adaptation 对最终输出的贡献 |
| NF4 | “4-bit quantization” | Normal Float 4：一种 4-bit 数据类型，quantization levels 位于正态分布分位点，适合 neural network weights |
| Adapter | “训练出来的小部分” | LoRA A 和 B 矩阵作为单独文件保存（10-100MB），可加载到任何 base model 副本之上 |
| Target modules | “哪些层加 LoRA” | 注入 LoRA adapters 的具体 linear layers（q_proj、v_proj 等） |
| Merging | “烘进去” | 计算 W + (alpha/r) * BA 并替换原始 weight，消除 inference 时的 adapter overhead |
| Paged optimizers | “训练时别 OOM” | 当 GPU memory 耗尽时，把 optimizer states（Adam momentum、variance）offload 到 CPU |
| Catastrophic forgetting | “Fine-tuning 把其他能力弄坏了” | 更新所有 weights 导致模型丢失先前学到的能力 |

## 延伸阅读

- Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models" (2021)：引入 low-rank decomposition 方法的原始论文，在 GPT-3 175B 上测试，rank 低至 4。
- Dettmers et al., "QLoRA: Efficient Finetuning of Quantized Language Models" (2023)：引入 NF4、double quantization 和 paged optimizers，让单张 48GB GPU 能 fine-tune 65B。
- PEFT library documentation (huggingface.co/docs/peft)：Hugging Face ecosystem 中 LoRA、QLoRA 和其他 parameter-efficient methods 的标准库。
- Yadav et al., "TIES-Merging: Resolving Interference When Merging Models" (2023)：在不降低质量的情况下组合多个 LoRA adapters 的技术。
- [Rafailov et al., "Direct Preference Optimization: Your Language Model is Secretly a Reward Model" (NeurIPS 2023)](https://arxiv.org/abs/2305.18290)：DPO 推导；SFT 之后的 preference-tuning 阶段，不需要 reward model。
- [TRL documentation](https://huggingface.co/docs/trl/)：`SFTTrainer`、`DPOTrainer`、`KTOTrainer` 以及与 PEFT/bitsandbytes/Unsloth 集成面的官方参考。
- [Unsloth documentation](https://docs.unsloth.ai/)：融合 kernels，可让 fine-tuning throughput 翻倍并减半 memory；TRL 底下的性能层。
- [Axolotl documentation](https://axolotl-ai-cloud.github.io/axolotl/)：YAML-configured multi-GPU SFT/DPO/QLoRA trainer；手写脚本之外的 config-as-code 方案。
