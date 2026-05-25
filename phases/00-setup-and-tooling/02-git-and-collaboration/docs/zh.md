# Git 与协作

> 版本控制不是可选项。你在这里构建的每个实验、每个模型、每节课都要被追踪。

**类型：** 学习
**语言：** --
**前置要求：** 阶段 0，第 01 课
**时间：** ~30 分钟

## 学习目标

- 配置 git 身份，并使用 add、commit 和 push 的日常工作流
- 创建并合并分支，用于隔离实验，同时不破坏 main
- 编写 `.gitignore`，排除模型检查点和大型二进制文件
- 使用 `git log` 浏览提交历史，理解项目演进

## 问题

你即将在 20 个阶段中编写数百个代码文件。没有版本控制，你会丢失工作成果、弄坏无法撤销的内容，也无法与他人协作。

Git 是工具。GitHub 是代码存放的位置。本课只覆盖你在这门课程中需要的内容，不多讲其他东西。

## 概念

```mermaid
sequenceDiagram
    participant WD as Working Directory
    participant SA as Staging Area
    participant LR as Local Repo
    participant R as Remote (GitHub)
    WD->>SA: git add
    SA->>LR: git commit
    LR->>R: git push
    R->>LR: git fetch
    LR->>WD: git pull
```

记住三件事：
1. 经常保存（`git commit`）
2. 推送到远端（`git push`）
3. 为实验创建分支（`git checkout -b experiment`）

## 构建它

### 第 1 步：配置 git

```bash
git config --global user.name "Your Name"
git config --global user.email "you@example.com"
```

### 第 2 步：日常工作流

```bash
git status
git add file.py
git commit -m "Add perceptron implementation"
git push origin main
```

### 第 3 步：为实验创建分支

```bash
git checkout -b experiment/new-optimizer

# ... make changes, commit ...

git checkout main
git merge experiment/new-optimizer
```

### 第 4 步：使用本课程仓库

```bash
git clone https://github.com/rohitg00/ai-engineering-from-scratch.git
cd ai-engineering-from-scratch

git checkout -b my-progress
# work through lessons, commit your code
git push origin my-progress
```

## 使用它

对于本课程，你只需要这些命令：

| 命令 | 使用时机 |
|---------|------|
| `git clone` | 获取课程仓库 |
| `git add` + `git commit` | 保存你的工作 |
| `git push` | 备份到 GitHub |
| `git checkout -b` | 在不破坏 main 的情况下尝试新东西 |
| `git log --oneline` | 查看你已经做了什么 |

就这些。你在本课程中不需要 rebase、cherry-pick 或 submodules。

## 练习

1. 克隆此仓库，创建一个名为 `my-progress` 的分支，创建一个文件，提交它并推送它
2. 创建一个 `.gitignore`，排除模型检查点文件（`.pt`、`.pth`、`.safetensors`）
3. 使用 `git log --oneline` 查看此仓库的提交历史，并阅读课程是如何被添加进来的

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|----------------|----------------------|
| Commit | “保存” | 你的整个项目在某个时间点的快照 |
| Branch | “一个副本” | 指向某个提交的指针，会随着你的工作向前移动 |
| Merge | “合并代码” | 从一个分支取出改动，并应用到另一个分支 |
| Remote | “云端” | 托管在其他地方的仓库副本（GitHub、GitLab） |
