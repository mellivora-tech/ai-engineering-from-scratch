# 编辑器设置

> 你的编辑器是你的副驾驶。一次配好，让它不挡路，并开始真正出力。

**类型：** 构建
**语言：** --
**前置要求：** 阶段 0，第 01 课
**时间：** ~20 分钟

## 学习目标

- 安装 VS Code，并配置 Python、Jupyter、linting 和 Remote SSH 所需的核心扩展
- 为 AI 工作流配置保存时格式化、类型检查和 notebook 输出滚动
- 设置 Remote SSH，让你像编辑本地代码一样编辑和调试远程 GPU 机器上的代码
- 评估编辑器替代方案（Cursor、Windsurf、Neovim）及其在 AI 工作中的取舍

## 问题

你会在编辑器里花上数千小时：写 Python、运行 notebooks、调试训练循环、SSH 到 GPU 机器。配置不好的编辑器会让每次会话都变成摩擦：没有自动补全，没有类型提示，没有行内错误，手动格式化，终端工作流也笨重。

正确设置只需要 20 分钟。跳过它，你每天都会损失 20 分钟。

## 概念

AI 工程的编辑器设置需要五样东西：

```mermaid
graph TD
    L5["5. Remote Development<br/>SSH into GPU boxes, cloud VMs"] --> L4
    L4["4. Terminal Integration<br/>Run scripts, debug, monitor GPU"] --> L3
    L3["3. AI-Specific Settings<br/>Auto-format, type checking, rulers"] --> L2
    L2["2. Extensions<br/>Python, Jupyter, Pylance, GitLens"] --> L1
    L1["1. Base Editor<br/>VS Code — free, extensible, universal"]
```

## 构建它

### 第 1 步：安装 VS Code

推荐使用 VS Code。它免费、可在所有 OS 上运行、拥有一流的 Jupyter notebook 支持，并且扩展生态覆盖了 AI 工作需要的所有东西。

从 [code.visualstudio.com](https://code.visualstudio.com/) 下载。

从终端验证：

```bash
code --version
```

如果 macOS 上找不到 `code`，打开 VS Code，按 `Cmd+Shift+P`，输入 "Shell Command"，然后选择 "Install 'code' command in PATH"。

### 第 2 步：安装核心扩展

在 VS Code 中打开集成终端（`Ctrl+`` ` 或 `` Cmd+` ``），安装 AI 工作中真正重要的扩展：

```bash
code --install-extension ms-python.python
code --install-extension ms-python.vscode-pylance
code --install-extension ms-toolsai.jupyter
code --install-extension eamodio.gitlens
code --install-extension ms-vscode-remote.remote-ssh
code --install-extension ms-python.debugpy
code --install-extension ms-python.black-formatter
code --install-extension charliermarsh.ruff
```

每个扩展的作用：

| 扩展 | 为什么需要 |
|-----------|-----|
| Python | 语言支持、虚拟环境检测、运行/调试 |
| Pylance | 快速类型检查、自动补全、导入解析 |
| Jupyter | 在 VS Code 内运行 notebooks，使用变量浏览器 |
| GitLens | 查看谁改了什么，行内 git blame |
| Remote SSH | 像打开本地文件夹一样打开远程 GPU 机器上的文件夹 |
| Debugpy | Python 单步调试 |
| Black Formatter | 保存时自动格式化，保持风格一致 |
| Ruff | 快速 linting，捕捉常见错误 |

本课的 `code/.vscode/extensions.json` 文件包含完整推荐列表。当你打开项目文件夹时，VS Code 会提示你安装它们。

### 第 3 步：配置设置

复制本课 `code/.vscode/settings.json` 中的设置，或通过 `Settings > Open Settings (JSON)` 手动应用。

AI 工作的关键设置：

```jsonc
{
    "python.analysis.typeCheckingMode": "basic",
    "editor.formatOnSave": true,
    "editor.rulers": [88, 120],
    "notebook.output.scrolling": true,
    "files.autoSave": "afterDelay"
}
```

为什么这些设置重要：

- **Type checking on basic**：在运行前捕捉错误参数类型。能节省调试张量形状不匹配和 API 参数错误的时间。
- **Format on save**：再也不用思考格式化。Black 会处理。
- **Rulers at 88 and 120**：Black 在 88 列换行。120 标尺会显示 docstring 和注释什么时候太长。
- **Notebook output scrolling**：训练循环会打印成千上万行。没有滚动时，输出面板会爆炸。
- **Auto-save**：你会忘记保存。训练脚本会运行旧代码。自动保存能防住这件事。

### 第 4 步：终端集成

VS Code 的集成终端是你运行训练脚本、监控 GPU 和管理环境的地方。

正确设置它：

```jsonc
{
    "terminal.integrated.defaultProfile.osx": "zsh",
    "terminal.integrated.defaultProfile.linux": "bash",
    "terminal.integrated.fontSize": 13,
    "terminal.integrated.scrollback": 10000
}
```

常用快捷键：

| 操作 | macOS | Linux/Windows |
|--------|-------|---------------|
| 切换终端 | `` Ctrl+` `` | `` Ctrl+` `` |
| 新建终端 | `Ctrl+Shift+`` ` | `Ctrl+Shift+`` ` |
| 分割终端 | `Cmd+\` | `Ctrl+\` |

分割终端很有用：一个运行脚本，一个用 `nvidia-smi -l 1` 或 `watch -n 1 nvidia-smi` 监控 GPU。

### 第 5 步：远程开发（SSH 到 GPU 机器）

这是 AI 工作中最重要的扩展。你会在远程机器上训练（云 VM、实验室服务器、Lambda、Vast.ai）。Remote SSH 让你打开远程文件系统、编辑文件、运行终端、调试，就像一切都在本地。

设置：

1. 安装 Remote SSH 扩展（第 2 步已完成）。
2. 按 `Ctrl+Shift+P`（或 `Cmd+Shift+P`），输入 "Remote-SSH: Connect to Host"。
3. 输入 `user@your-gpu-box-ip`。
4. VS Code 会自动在远程机器上安装它的 server 组件。

如需免密码访问，设置 SSH keys：

```bash
ssh-keygen -t ed25519 -C "your-email@example.com"
ssh-copy-id user@your-gpu-box-ip
```

把主机加到 `~/.ssh/config`，使用起来更方便：

```
Host gpu-box
    HostName 203.0.113.50
    User ubuntu
    IdentityFile ~/.ssh/id_ed25519
    ForwardAgent yes
```

现在 `Remote-SSH: Connect to Host > gpu-box` 会立刻连接。

## 替代方案

### Cursor

[cursor.com](https://cursor.com) 是带内置 AI 代码生成的 VS Code fork。它使用相同的扩展生态和设置格式。如果你用 Cursor，本课所有内容仍然适用。导入同一份 `settings.json` 和 `extensions.json`。

### Windsurf

[windsurf.com](https://windsurf.com) 是另一个 AI-first 的 VS Code fork。同样：相同的扩展、相同的设置格式、相同的 Remote SSH 支持。

### Vim/Neovim

如果你已经在用 Vim 或 Neovim，并且效率很高，就继续用。AI Python 工作的最低配置：

- **pyright** 或 **pylsp** 用于类型检查（通过 Mason 或手动安装）
- **nvim-lspconfig** 用于语言服务器集成
- **jupyter-vim** 或 **molten-nvim** 用于 notebook 式执行
- **telescope.nvim** 用于文件/符号搜索
- **none-ls.nvim** 配合 black 和 ruff 做格式化/linting

如果你还不会 Vim，不要现在开始学。学习曲线会和学习 AI 工程抢精力。用 VS Code。

## 使用它

有了这套设置，你的日常工作流会像这样：

1. 在 VS Code 中打开项目文件夹（或通过 Remote SSH 连接到 GPU 机器）。
2. 在编辑器里写 Python，拥有自动补全、类型提示和行内错误。
3. 使用 Jupyter 扩展在编辑器内运行 Jupyter notebooks。
4. 使用集成终端运行训练脚本、`uv pip install` 和 GPU 监控。
5. 提交前用 GitLens 检查改动。

## 练习

1. 安装 VS Code 和第 2 步列出的所有扩展
2. 把本课的 `settings.json` 复制到你的 VS Code 配置中
3. 打开一个 Python 文件，验证 Pylance 会显示类型提示，并且 Black 会在保存时格式化
4. 如果你有远程机器，设置 Remote SSH，并打开远程机器上的一个文件夹

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|----------------|----------------------|
| LSP | “自动补全引擎” | Language Server Protocol：一种标准，让编辑器从特定语言的 server 获取类型信息、补全和诊断 |
| Pylance | “Python 插件” | Microsoft 的 Python 语言服务器，使用 Pyright 做类型检查和 IntelliSense |
| Remote SSH | “在服务器上工作” | VS Code 扩展，在远程机器上运行一个轻量 server，并把 UI 流式传到本地编辑器 |
| Format on save | “自动 prettier” | 每次保存时编辑器运行格式化器（Black、Ruff），让代码风格始终一致 |
