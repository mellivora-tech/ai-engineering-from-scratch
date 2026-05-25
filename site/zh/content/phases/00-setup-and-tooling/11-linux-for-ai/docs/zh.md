# 面向 AI 的 Linux

> 大多数 AI 都跑在 Linux 上。你需要知道足够多，别卡住。

**类型：** 学习
**语言：** --
**前置要求：** 阶段 0，第 01 课
**时间：** ~30 分钟

## 学习目标

- 浏览 Linux 文件系统，并从命令行执行基本文件操作
- 使用 `chmod` 和 `chown` 管理文件权限，解决 "Permission denied" 错误
- 使用 `apt` 安装系统包，并为 AI 工作设置一台新的 GPU 机器
- 识别从 macOS 切到 Linux 时常让开发者踩坑的差异

## 问题

你在 macOS 或 Windows 上开发。但一旦你 SSH 到云 GPU 机器、租 Lambda 实例，或启动一台 EC2 机器，你就会落进 Ubuntu。终端是你唯一的界面。没有 Finder，没有 Explorer，没有 GUI。如果你不能从命令行浏览文件系统、安装包、管理进程，你就会一边为闲置 GPU 付费，一边搜索“Linux 怎么解压文件”。

这是一份生存指南。它只覆盖你在远程 Linux 机器上做 AI 工作所需的内容。不多讲。

## 文件系统布局

Linux 把所有东西都组织在单一根目录 `/` 下。没有 `C:\`，也没有 `/Volumes`。你实际会碰到的目录：

```mermaid
graph TD
    root["/"] --> home["home/your-username/<br/>Your files — clone repos, run training"]
    root --> tmp["tmp/<br/>Temporary files, cleared on reboot"]
    root --> usr["usr/<br/>System programs and libraries"]
    root --> etc["etc/<br/>Config files"]
    root --> varlog["var/log/<br/>Logs — check when something breaks"]
    root --> mnt["mnt/ or /media/<br/>External drives and volumes"]
    root --> proc["proc/ and /sys/<br/>Virtual files — kernel and hardware info"]
```

你的 home 目录是 `~` 或 `/home/your-username`。你做的大多数事情都发生在这里。

## 必备命令

下面 15 个命令覆盖了你在远程 GPU 机器上 95% 的操作。

### 移动位置

```bash
pwd                         # Where am I?
ls                          # What's here?
ls -la                      # What's here, including hidden files with details?
cd /path/to/dir             # Go there
cd ~                        # Go home
cd ..                       # Go up one level
```

### 文件和目录

```bash
mkdir my-project            # Create a directory
mkdir -p a/b/c              # Create nested directories in one shot

cp file.txt backup.txt      # Copy a file
cp -r src/ src-backup/      # Copy a directory (recursive)

mv old.txt new.txt          # Rename a file
mv file.txt /tmp/           # Move a file

rm file.txt                 # Delete a file (no trash, it's gone)
rm -rf my-dir/              # Delete a directory and everything inside
```

`rm -rf` 是永久的。没有撤销。按回车前要反复确认路径。

### 读取文件

```bash
cat file.txt                # Print entire file
head -20 file.txt           # First 20 lines
tail -20 file.txt           # Last 20 lines
tail -f log.txt             # Follow a log file in real time (Ctrl+C to stop)
less file.txt               # Scroll through a file (q to quit)
```

### 搜索

```bash
grep "error" training.log           # Find lines containing "error"
grep -r "learning_rate" .           # Search all files in current directory
grep -i "cuda" config.yaml          # Case-insensitive search

find . -name "*.py"                 # Find all Python files under current dir
find . -name "*.ckpt" -size +1G     # Find checkpoint files larger than 1GB
```

## 权限

Linux 中每个文件都有所有者和权限位。当脚本不能执行，或你不能写入某个目录时，你会遇到它。

```bash
ls -l train.py
# -rwxr-xr-- 1 user group 2048 Mar 19 10:00 train.py
#  ^^^             owner permissions: read, write, execute
#     ^^^          group permissions: read, execute
#        ^^        everyone else: read only
```

常见修复：

```bash
chmod +x train.sh           # Make a script executable
chmod 755 deploy.sh         # Owner: full, others: read+execute
chmod 644 config.yaml       # Owner: read+write, others: read only

chown user:group file.txt   # Change who owns a file (needs sudo)
```

当某个东西提示 "Permission denied"，几乎总是权限问题。`chmod +x` 或 `sudo` 可以修好大多数情况。

## 包管理（apt）

Ubuntu 使用 `apt`。这是你安装系统级软件的方式。

```bash
sudo apt update             # Refresh the package list (always do this first)
sudo apt install -y htop    # Install a package (-y skips confirmation)
sudo apt install -y build-essential  # C compiler, make, etc. Needed by many Python packages
sudo apt install -y tmux    # Terminal multiplexer (keep sessions alive after disconnect)

apt list --installed        # What's installed?
sudo apt remove htop        # Uninstall
```

新 GPU 机器上常装的包：

```bash
sudo apt update && sudo apt install -y \
    build-essential \
    git \
    curl \
    wget \
    tmux \
    htop \
    unzip \
    python3-venv
```

## 用户和 sudo

你通常会以普通用户身份登录。有些操作需要 root（管理员）权限。

```bash
whoami                      # What user am I?
sudo command                # Run a single command as root
sudo su                     # Become root (exit to go back, use sparingly)
```

在云 GPU 实例上，你通常是唯一用户，并且已经有 sudo 权限。不要用 root 运行所有东西。只在需要时使用 sudo。

## 进程和 systemd

当训练卡住，或你需要检查正在运行什么时：

```bash
htop                        # Interactive process viewer (q to quit)
ps aux | grep python        # Find running Python processes
kill 12345                  # Gracefully stop process with PID 12345
kill -9 12345               # Force kill (use when graceful doesn't work)
nvidia-smi                  # GPU processes and memory usage
```

systemd 管理服务（后台 daemon）。当你运行推理服务器时会用到它：

```bash
sudo systemctl start nginx          # Start a service
sudo systemctl stop nginx           # Stop it
sudo systemctl restart nginx        # Restart it
sudo systemctl status nginx         # Check if it's running
sudo systemctl enable nginx         # Start automatically on boot
```

## 磁盘空间

GPU 机器的磁盘空间常常有限。模型和数据集会很快填满它。

```bash
df -h                       # Disk usage for all mounted drives
df -h /home                 # Disk usage for /home specifically

du -sh *                    # Size of each item in current directory
du -sh ~/.cache             # Size of your cache (pip, huggingface models land here)
du -sh /data/checkpoints/   # Check how big your checkpoints are

# Find the biggest space hogs
du -h --max-depth=1 / 2>/dev/null | sort -hr | head -20
```

常见省空间方法：

```bash
# Clear pip cache
pip cache purge

# Clear apt cache
sudo apt clean

# Remove old checkpoints you don't need
rm -rf checkpoints/epoch_01/ checkpoints/epoch_02/
```

## 网络

你会从命令行下载模型、传输文件、访问 API。

```bash
# Download files
wget https://example.com/model.bin                   # Download a file
curl -O https://example.com/data.tar.gz              # Same thing with curl
curl -s https://api.example.com/health | python3 -m json.tool  # Hit an API, pretty-print JSON

# Transfer files between machines
scp model.bin user@remote:/data/                     # Copy file to remote machine
scp user@remote:/data/results.csv .                  # Copy file from remote to local
scp -r user@remote:/data/checkpoints/ ./local-dir/   # Copy directory

# Sync directories (faster than scp for large transfers, resumes on failure)
rsync -avz --progress ./data/ user@remote:/data/
rsync -avz --progress user@remote:/results/ ./results/
```

任何大文件都优先用 `rsync`，不要用 `scp`。它只传输变化的字节，并且能处理连接中断。

## tmux：让会话保持存活

当你 SSH 到远程机器时，合上笔记本会杀掉训练运行。tmux 可以防止这件事。

```bash
tmux new -s train           # Start a new session named "train"
# ... start your training, then:
# Ctrl+B, then D            # Detach (training keeps running)

tmux ls                     # List sessions
tmux attach -t train        # Reattach to session

# Inside tmux:
# Ctrl+B, then %            # Split pane vertically
# Ctrl+B, then "            # Split pane horizontally
# Ctrl+B, then arrow keys   # Switch between panes
```

长时间训练作业一定放在 tmux 里。永远如此。

## Windows 用户的 WSL2

如果你在 Windows 上，WSL2 可以给你一个真正的 Linux 环境，不需要双系统。

```bash
# In PowerShell (admin)
wsl --install -d Ubuntu-24.04

# After restart, open Ubuntu from Start menu
sudo apt update && sudo apt upgrade -y
```

WSL2 运行真正的 Linux 内核。本课所有内容都能在里面工作。在 WSL 内部，你的 Windows 文件位于 `/mnt/c/Users/YourName/`。

GPU 透传需要在 Windows 侧安装 NVIDIA 驱动。安装 Windows NVIDIA 驱动（不是 Linux 驱动），CUDA 就能在 WSL2 里使用。

## 坑点：从 macOS 到 Linux

如果你来自 macOS，这些事情会绊倒你：

| macOS | Linux | 说明 |
|-------|-------|-------|
| `brew install` | `sudo apt install` | 包名有时不同。`brew install htop` 和 `sudo apt install htop` 类似，但 `brew install readline` 对应的可能是 `sudo apt install libreadline-dev`。 |
| `open file.txt` | `xdg-open file.txt` | 但远程机器上不会有 GUI。用 `cat` 或 `less`。 |
| `pbcopy` / `pbpaste` | 不可用 | 通过 SSH 没有剪贴板 pipe。 |
| `~/.zshrc` | `~/.bashrc` | macOS 默认 zsh。多数 Linux 服务器使用 bash。 |
| `/opt/homebrew/` | `/usr/bin/`, `/usr/local/bin/` | 二进制文件位于不同位置。 |
| `sed -i '' 's/a/b/' file` | `sed -i 's/a/b/' file` | macOS sed 在 `-i` 后需要空字符串。Linux 不需要。 |
| 大小写不敏感文件系统 | 大小写敏感文件系统 | `Model.py` 和 `model.py` 在 Linux 上是两个不同文件。 |
| 行尾 `\n` | 行尾 `\n` | 相同。但 Windows 使用 `\r\n`，这会破坏 bash 脚本。运行 `dos2unix` 修复。 |

## 快速参考卡

```
Navigation:     pwd, ls, cd, find
Files:          cp, mv, rm, mkdir, cat, head, tail, less
Search:         grep, find
Permissions:    chmod, chown, sudo
Packages:       apt update, apt install
Processes:      htop, ps, kill, nvidia-smi
Services:       systemctl start/stop/restart/status
Disk:           df -h, du -sh
Network:        curl, wget, scp, rsync
Sessions:       tmux new/attach/detach
```

## 练习

1. SSH 到任意 Linux 机器（或打开 WSL2），导航到你的 home 目录。创建一个项目文件夹，在里面用 `touch` 创建三个空文件，然后用 `ls -la` 列出它们。
2. 用 apt 安装 `htop`，运行它，并找出哪个进程使用最多内存。
3. 启动一个 tmux 会话，在里面运行 `sleep 300`，detach，列出会话，再 reattach。
4. 用 `df -h` 检查可用磁盘空间，然后用 `du -sh ~/.cache/*` 找出缓存中占空间的内容。
5. 使用 `scp` 从本地机器传一个文件到远程机器，然后用 `rsync` 做同样的传输，比较体验。
