# Python Environments

> 依赖地狱是真实存在的。虚拟环境就是解药。

**类型：** 构建
**语言：** Python
**先修要求：** Phase 0, Lesson 01
**时间：** ~30 分钟

## 学习目标

- 使用 `uv`、`venv` 或 `conda` 创建隔离的虚拟环境
- 编写带可选依赖组的 `pyproject.toml`，并生成 lockfiles 以保证可复现性
- 诊断并修复常见问题：全局安装、混用 pip/conda、CUDA 版本不匹配
- 为存在依赖冲突的项目实现按 phase 划分的环境策略

## 问题

你为一个微调项目安装了 PyTorch 2.4。下周，另一个项目需要 PyTorch 2.1，因为它固定了 CUDA 构建版本。你全局升级后，第一个项目坏了。你降级后，第二个项目又坏了。

这就是依赖地狱。它在 AI/ML 工作中非常常见，因为：

- PyTorch、JAX 和 TensorFlow 都会附带自己的 CUDA 绑定
- 模型库会固定特定框架版本
- 全局 `pip install` 会覆盖之前已有的内容
- CUDA 11.8 构建不能和 CUDA 12.x 驱动一起工作（反之亦然）

解决办法：每个项目都拥有自己的隔离环境和自己的包。

## 概念

```mermaid
graph TD
    subgraph without["Without virtual environments"]
        SP[System Python] --> T24["torch 2.4.0 (CUDA 12.4)\nProject A needs this"]
        SP --> T21["torch 2.1.0 (CUDA 11.8)\nProject B needs this"]
        SP --> CONFLICT["CONFLICT: only one\ntorch version can exist"]
    end

    subgraph with["With virtual environments"]
        PA["Project A (.venv/)"] --> PA1["torch 2.4.0 (CUDA 12.4)"]
        PA --> PA2["transformers 4.44"]
        PB["Project B (.venv/)"] --> PB1["torch 2.1.0 (CUDA 11.8)"]
        PB --> PB2["diffusers 0.28"]
    end
```

## 构建它

### Option 1: uv venv（推荐）

`uv` 是最快的 Python 包管理器（比 pip 快 10-100 倍）。它用一个工具处理虚拟环境、Python 版本和依赖解析。

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh

uv python install 3.12

cd your-project
uv venv
source .venv/bin/activate
```

安装包：

```bash
uv pip install torch numpy
```

一步创建带 `pyproject.toml` 的项目：

```bash
uv init my-ai-project
cd my-ai-project
uv add torch numpy matplotlib
```

### Option 2: venv（内置）

如果你不能安装 `uv`，Python 自带 `venv`：

```bash
python3 -m venv .venv
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\activate     # Windows

pip install torch numpy
```

它比 `uv` 慢，但在任何已安装 Python 的地方都可用。

### Option 3: conda（需要时使用）

Conda 管理 CUDA toolkits、cuDNN 和 C libraries 等非 Python 依赖。在这些情况下使用它：

- 你需要特定 CUDA toolkit 版本，但不想系统级安装
- 你在共享集群上，无法安装系统包
- 某个库的安装说明写着“use conda”

```bash
# Install miniconda (not the full Anaconda)
curl -LsSf https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o miniconda.sh
bash miniconda.sh -b

conda create -n myproject python=3.12
conda activate myproject

conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
```

一条规则：如果你在某个环境中使用 conda，就在这个环境里对所有包都使用 conda。把 `pip install` 混进 conda env 会造成依赖冲突，而且调试起来很痛苦。

### 本课程：按 Phase 划分的策略

你可以为整门课程创建一个环境。不要这样做。不同 phase 需要不同的（有时互相冲突的）依赖。

策略：

```
ai-engineering-from-scratch/
├── .venv/                    <-- shared lightweight env for phases 0-3
├── phases/
│   ├── 04-neural-networks/
│   │   └── .venv/            <-- PyTorch env
│   ├── 05-cnns/
│   │   └── .venv/            <-- same PyTorch env (symlink or shared)
│   ├── 08-transformers/
│   │   └── .venv/            <-- might need different transformer versions
│   └── 11-llm-apis/
│       └── .venv/            <-- API SDKs, no torch needed
```

`code/env_setup.sh` 中的脚本会为本课程创建基础环境。

## pyproject.toml 基础

每个 Python 项目都应该有一个 `pyproject.toml`。它用一个文件取代 `setup.py`、`setup.cfg` 和 `requirements.txt`。

```toml
[project]
name = "ai-engineering-from-scratch"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "numpy>=1.26",
    "matplotlib>=3.8",
    "jupyter>=1.0",
    "scikit-learn>=1.4",
]

[project.optional-dependencies]
torch = ["torch>=2.3", "torchvision>=0.18"]
llm = ["anthropic>=0.39", "openai>=1.50"]
```

然后安装：

```bash
uv pip install -e ".[torch]"    # base + PyTorch
uv pip install -e ".[llm]"     # base + LLM SDKs
uv pip install -e ".[torch,llm]" # everything
```

## Lockfiles

lockfile 会把每个依赖（包括传递依赖）固定到精确版本。这保证了可复现性：任何从 lockfile 安装的人都会得到完全相同的包。

```bash
# uv generates uv.lock automatically when using uv add
uv add numpy

# pip-tools approach
uv pip compile pyproject.toml -o requirements.lock
uv pip install -r requirements.lock
```

把你的 lockfile 提交到 git。当别人 clone 仓库时，他们从 lockfile 安装，就能得到一致的版本。

## 常见错误

### 1. 全局安装

```bash
pip install torch  # BAD: installs to system Python

source .venv/bin/activate
pip install torch  # GOOD: installs to virtual environment
```

检查你的包会安装到哪里：

```bash
which python       # should show .venv/bin/python, not /usr/bin/python
which pip           # should show .venv/bin/pip
```

### 2. 混用 pip 和 conda

```bash
conda create -n myenv python=3.12
conda activate myenv
conda install pytorch -c pytorch
pip install some-other-package   # BAD: can break conda's dependency tracking
conda install some-other-package # GOOD: let conda manage everything
```

如果你必须在 conda 中使用 pip（有些包只有 pip 版本），先安装所有 conda 包，最后再安装 pip 包。

### 3. 忘记激活

```bash
python train.py           # uses system Python, missing packages
source .venv/bin/activate
python train.py           # uses project Python, packages found
```

你的 shell prompt 应该显示环境名称：

```
(.venv) $ python train.py
```

### 4. 把 .venv 提交到 git

```bash
echo ".venv/" >> .gitignore
```

虚拟环境通常有 200MB-2GB。它们是本地的，不能在机器之间移植。改为提交 `pyproject.toml` 和 lockfile。

### 5. CUDA 版本不匹配

```bash
nvidia-smi                # shows driver CUDA version (e.g., 12.4)
python -c "import torch; print(torch.version.cuda)"  # shows PyTorch CUDA version

# These must be compatible.
# PyTorch CUDA version must be <= driver CUDA version.
```

## 使用它

运行设置脚本来创建你的课程环境：

```bash
bash phases/00-setup-and-tooling/06-python-environments/code/env_setup.sh
```

这会在仓库根目录创建一个 `.venv`，安装并验证核心依赖。

## 练习

1. 运行 `env_setup.sh` 并确认所有检查都通过
2. 创建第二个虚拟环境，在其中安装不同版本的 numpy，并确认两个环境相互隔离
3. 为一个同时需要 PyTorch 和 Anthropic SDK 的项目编写 `pyproject.toml`
4. 故意在全局安装一个包（不激活 venv），观察它安装到哪里，然后卸载它

## 关键术语

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| Virtual environment | “一个 venv” | 一个隔离目录，包含 Python 解释器和包，并与系统 Python 分开 |
| Lockfile | “固定依赖” | 一个列出每个包及其精确版本的文件，保证不同机器上的安装一致 |
| pyproject.toml | “新的 setup.py” | 标准 Python 项目配置文件，取代 setup.py/setup.cfg/requirements.txt |
| Transitive dependency | “依赖的依赖” | Package B 依赖 C；如果你安装依赖 B 的 A，那么 C 就是 A 的传递依赖 |
| CUDA mismatch | “我的 GPU 不能用” | PyTorch 编译时使用的 CUDA 版本与你的 GPU 驱动支持的版本不同 |
