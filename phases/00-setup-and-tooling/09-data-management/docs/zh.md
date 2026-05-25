# 数据管理

> 数据是燃料。你管理它的方式，决定了你能跑多快。

**类型：** 构建
**语言：** Python
**前置要求：** 阶段 0，第 01 课
**时间：** ~45 分钟

## 学习目标

- 使用 Hugging Face `datasets` 库加载、流式读取和缓存数据集
- 在 CSV、JSON、Parquet 和 Arrow 格式之间转换，并解释它们的取舍
- 使用固定随机种子创建可复现的 train/validation/test 划分
- 使用 `.gitignore`、Git LFS 或 DVC 管理大型模型和数据集文件

## 问题

每个 AI 项目都从数据开始。你需要找到数据集、下载它们、在格式之间转换、划分训练和评估集，并对它们做版本管理，让实验可复现。每次都手动做这些事很慢，也容易出错。你需要一个可重复的工作流。

## 概念

```mermaid
graph TD
    A["Hugging Face Hub"] --> B["datasets library"]
    B --> C["Load / Stream"]
    C --> D["Local Cache<br/>~/.cache/huggingface/"]
    B --> E["Format Conversion<br/>CSV, JSON, Parquet, Arrow"]
    E --> F["Data Splits<br/>train / val / test"]
    F --> G["Your Training Pipeline"]
```

Hugging Face `datasets` 库是 AI 工作中加载数据的标准方式。它开箱即用地处理下载、缓存、格式转换和流式读取。

## 构建它

### 第 1 步：安装 datasets 库

```bash
pip install datasets huggingface_hub
```

### 第 2 步：加载数据集

```python
from datasets import load_dataset

dataset = load_dataset("imdb")
print(dataset)
print(dataset["train"][0])
```

这会下载 IMDB 电影评论数据集。第一次下载后，它会从 `~/.cache/huggingface/datasets/` 缓存加载。

### 第 3 步：流式读取大型数据集

有些数据集太大，无法放进磁盘。流式读取会逐行加载，不会下载完整数据集。

```python
dataset = load_dataset("wikimedia/wikipedia", "20220301.en", split="train", streaming=True)

for i, example in enumerate(dataset):
    print(example["title"])
    if i >= 4:
        break
```

流式读取会给你一个 `IterableDataset`。你在数据到达时逐行处理。无论数据集多大，内存使用都保持不变。

### 第 4 步：数据集格式

`datasets` 库底层使用 Apache Arrow。你可以根据流水线需要转换成其他格式。

```python
dataset = load_dataset("imdb", split="train")

dataset.to_csv("imdb_train.csv")
dataset.to_json("imdb_train.json")
dataset.to_parquet("imdb_train.parquet")
```

格式对比：

| 格式 | 大小 | 读取速度 | 最适合 |
|--------|------|-----------|----------|
| CSV | 大 | 慢 | 人类可读、电子表格 |
| JSON | 大 | 慢 | API、嵌套数据 |
| Parquet | 小 | 快 | 分析、列式查询 |
| Arrow | 小 | 最快 | 内存内处理（`datasets` 内部使用的格式） |

对 AI 工作来说，Parquet 是最佳存储格式。Arrow 是你在内存中使用的格式。CSV 和 JSON 用于交换。

### 第 5 步：数据划分

每个 ML 项目都需要三种划分：

- **Train**：模型从这里学习（通常 80%）
- **Validation**：你在训练过程中检查进展（通常 10%）
- **Test**：训练完成后的最终评估（通常 10%）

有些数据集已经预先划分。没有时，自己划分：

```python
dataset = load_dataset("imdb", split="train")

split = dataset.train_test_split(test_size=0.2, seed=42)
train_val = split["train"].train_test_split(test_size=0.125, seed=42)

train_ds = train_val["train"]
val_ds = train_val["test"]
test_ds = split["test"]

print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
```

一定要设置 seed 来保证可复现。同一个 seed 每次都会产生同样的划分。

### 第 6 步：下载和缓存模型

模型是大文件。`huggingface_hub` 库负责下载和缓存。

```python
from huggingface_hub import hf_hub_download, snapshot_download

model_path = hf_hub_download(
    repo_id="sentence-transformers/all-MiniLM-L6-v2",
    filename="config.json"
)
print(f"Cached at: {model_path}")

model_dir = snapshot_download("sentence-transformers/all-MiniLM-L6-v2")
print(f"Full model at: {model_dir}")
```

模型会缓存到 `~/.cache/huggingface/hub/`。下载一次后，后续运行会立刻加载。

### 第 7 步：处理大文件

模型权重和大型数据集不应该放进 git。有三个选项：

**选项 A：.gitignore（最简单）**

```
*.bin
*.safetensors
*.pt
*.onnx
data/*.parquet
data/*.csv
models/
```

**选项 B：Git LFS（在 git 中追踪大文件）**

```bash
git lfs install
git lfs track "*.bin"
git lfs track "*.safetensors"
git add .gitattributes
```

Git LFS 在你的仓库里存指针，把实际文件存在单独服务器上。GitHub 免费提供 1 GB。

**选项 C：DVC（数据版本控制）**

```bash
pip install dvc
dvc init
dvc add data/training_set.parquet
git add data/training_set.parquet.dvc data/.gitignore
git commit -m "Track training data with DVC"
```

DVC 创建小型 `.dvc` 文件指向你的数据。数据本身存在 S3、GCS 或其他远程存储后端。

| 方案 | 复杂度 | 最适合 |
|----------|-----------|----------|
| .gitignore | 低 | 个人项目、可以重新获取的下载数据 |
| Git LFS | 中 | 团队通过 git 共享模型权重 |
| DVC | 高 | 可复现实验、大型数据集、团队协作 |

对本课程来说，`.gitignore` 足够了。当你需要跨机器复现精确实验时，再使用 DVC。

### 第 8 步：存储模式

**本地存储**适用于 10 GB 以下的数据集。HF 缓存会自动处理。

**云存储**适用于更大的数据，或需要跨机器共享的数据：

```python
import os

local_path = os.path.expanduser("~/.cache/huggingface/datasets/")

# s3_path = "s3://my-bucket/datasets/"
# gcs_path = "gs://my-bucket/datasets/"
```

DVC 可以直接和 S3、GCS 集成：

```bash
dvc remote add -d myremote s3://my-bucket/dvc-store
dvc push
```

对本课程来说，本地存储已经足够。当你在远程 GPU 实例上微调时，云存储才会变得重要。

## 本课程使用的数据集

| 数据集 | 课程 | 大小 | 教什么 |
|---------|---------|------|----------------|
| IMDB | Tokenization, classification | 84 MB | 文本分类基础 |
| WikiText | Language modeling | 181 MB | 下一 token 预测 |
| SQuAD | QA systems | 35 MB | 问答、span |
| Common Crawl (subset) | Embeddings | Varies | 大规模文本处理 |
| MNIST | Vision basics | 21 MB | 图像分类基础 |
| COCO (subset) | Multimodal | Varies | 图像-文本对 |

你现在不需要下载所有这些数据集。每节课都会说明它需要什么。

## 使用它

运行工具脚本，验证一切正常：

```bash
python code/data_utils.py
```

这会下载一个小数据集、转换它、划分它，并打印摘要。

## 交付它

本课产出：
- `code/data_utils.py` - 可复用的数据加载和缓存工具
- `outputs/prompt-data-helper.md` - 用于为任务寻找合适数据集的提示词

## 练习

1. 加载带 `mrpc` config 的 `glue` 数据集，并检查前 5 个样本
2. 流式读取 `c4` 数据集，统计 10 秒内能处理多少样本
3. 将一个数据集转换为 Parquet，并和 CSV 比较文件大小
4. 使用固定 seed 创建 70/15/15 的 train/val/test 划分，并验证大小

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|----------------|----------------------|
| Dataset split | “训练数据” | 在 ML 生命周期不同阶段使用的命名子集（train/val/test） |
| Streaming | “惰性加载” | 从远程源逐行处理数据，而不是下载完整数据集 |
| Parquet | “压缩 CSV” | 针对分析查询和存储效率优化的列式文件格式 |
| Arrow | “快速 dataframe” | `datasets` 库内部用于零拷贝读取的内存内列式格式 |
| Git LFS | “大文件用的 Git” | 一个扩展，把大文件存到 git 仓库之外，同时在版本控制中保留指针 |
| DVC | “数据用的 Git” | 面向数据集和模型的版本控制系统，可与云存储集成 |
| Cache | “已经下载过” | 之前获取的数据的本地副本，默认存储在 ~/.cache/huggingface/ |
