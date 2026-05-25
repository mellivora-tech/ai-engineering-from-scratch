# ASCII Art 与 Visual Jailbreaks

> Jiang、Xu、Niu、Xiang、Ramasubramanian、Li、Poovendran，“ArtPrompt: ASCII Art-based Jailbreak Attacks against Aligned LLMs”（ACL 2024，arXiv:2402.11753）。把 harmful request 中 safety-relevant token 遮住，用同一字母的 ASCII-art render 替换，然后发送 cloaked prompt。GPT-3.5、GPT-4、Gemini、Claude、Llama-2 都无法 robustly 识别 ASCII-art token。该 attack 绕过 PPL（perplexity filter）、Paraphrase defenses 和 Retokenization。相关：ViTC benchmark 测量非语义 visual prompt 的识别；StructuralSleight 泛化到 Uncommon Text-Encoded Structures（tree、graph、nested JSON），把它们视作一类 encoding attack。

**类型：** 构建
**语言：** Python（stdlib，ArtPrompt token-masking harness）
**前置要求：** 阶段 18 · 12（PAIR），阶段 18 · 13（MSJ）
**时间：** ~60 分钟

## 学习目标

- 描述 ArtPrompt attack：word-identification step、ASCII-art substitution、final cloaked prompt。
- 解释为什么标准 defense（PPL、Paraphrase、Retokenization）会在 ArtPrompt 上失败。
- 定义 ViTC，并描述它测量什么。
- 描述 StructuralSleight 如何泛化到任意 Uncommon Text-Encoded Structures。

## 问题

通过 paraphrase 和 roleplay 的 attack（第 12 课）以及通过长 context 的 attack（第 13 课）作用在 text-level pattern 上。ArtPrompt 作用在 recognition level：模型没有解析 forbidden token。它解析的是由字符渲染出的图像。Safety filter 看到的是 harmless punctuation。模型看到的是一个 word。

## 概念

### ArtPrompt，两步

第 1 步。Word Identification。给定一个 harmful request，attacker 使用 LLM 识别 safety-relevant word（例如 “how to make a bomb” 中的 “bomb”）。

第 2 步。Cloaked Prompt Generation。把每个被识别的 word 替换为它的 ASCII-art rendering（由字符组成的 7x5 或 7x7 block，形成字母形状）。模型收到的是一组标点和空格构成的 grid；足够 capable 的模型能把它识别为 word，而 safety filter 只看到 grid。

结果：GPT-4、Gemini、Claude、Llama-2、GPT-3.5 都失败。在其 benchmark subset 上 attack success rate 超过 75%。

### 为什么标准 defense 失败

- **PPL（perplexity filter）。** ASCII art 有高 perplexity，但所有新颖输入也都有。能 block ArtPrompt 的 threshold 也会 block 合法 structured input。
- **Paraphrase。** Paraphrase prompt 会破坏 ASCII art。实践中，paraphrase LLM 经常保留或重构 art。
- **Retokenization。** 用不同方式切分 token，并不会改变模型的视觉识别正在识别字母形状这一事实。

底层问题是 safety filter 位于 token 或 semantic level；ArtPrompt 作用在 visual recognition level。

### ViTC benchmark

对非语义 visual prompt 的识别。它测量模型读取 ASCII-art、wingdings 和其他非 text-semantic visual content 的能力。ArtPrompt 的有效性与 ViTC accuracy 相关：模型越擅长读取 visual text，ArtPrompt 对它越有效。这是一个 capability-safety tradeoff。

### StructuralSleight

泛化 ArtPrompt：Uncommon Text-Encoded Structures（UTES）。Tree、graph、nested JSON、CSV-in-JSON、diff-style code block。如果某种结构在 training safety data 中罕见，但模型可以解析，它就能隐藏 harmful content。

Defense 含义：safety 必须泛化到模型能解析的 structured representation。这个集合很大，而且还在增长。

### Image-modality analog

Visual LLM（GPT-5.2、Gemini 3 Pro、Claude Opus 4.5、Grok 4.1）扩展了 attack surface。带实际图片的 ArtPrompt-style attack 比 ASCII-art analog 更强，因为 image encoder 产生更丰富的 signal。

### 它在阶段 18 中的位置

第 12-14 课描述三种正交 attack vector：iterative refinement（PAIR）、context length（MSJ）和 encoding（ArtPrompt/StructuralSleight）。第 15 课从 model-centric attack 转向 system-boundary attack（indirect prompt injection）。第 16 课描述防御工具回应。

## 使用它

`code/main.py` 构建一个玩具 ArtPrompt。你可以用 ASCII-art glyph cloak harmful query 中的特定 word，验证 cloaked string 能通过 keyword filter，并且（可选）用简单 recognizer 把 cloaked string decode 回来。

## 交付它

本课会生成 `outputs/skill-encoding-audit.md`。给定一份 jailbreak-defense report，它会枚举覆盖的 encoding attack family（ASCII art、base64、leet-speak、UTF-8 homoglyph、UTES）以及捕获每一种的 defense layer。

## 练习

1. 运行 `code/main.py`。验证 cloaked string 能通过简单 keyword filter。报告所需 character-level change。

2. 实现第二种 encoding：对同一个 target word 做 base64。比较它与 ArtPrompt 的 filter-bypass rate 和 recovery difficulty。

3. 阅读 Jiang et al. 2024 第 4.3 节（五模型结果）。提出一个原因，解释为什么 Claude 在同一 benchmark 上的 ArtPrompt-resistance 高于 Gemini。

4. 设计一个 pre-generation defense，用来检测 prompt 中 ASCII-art-shaped region。测量它在合法 code、table 和 mathematical notation 上的 false-positive rate。

5. StructuralSleight 列出了 10 种 encoding structure。勾勒一个能处理全部 10 种的 generalized defense，并估计每个 defended prompt 的 compute cost。

## 关键词汇

| 术语 | 人们常说 | 实际含义 |
|------|-----------------|------------------------|
| ArtPrompt | “ASCII-art attack” | 用 ASCII-art rendering 遮蔽 safety word 的两步 jailbreak |
| Cloaking | “hide the word” | 把 forbidden token 替换成模型能读、filter 读不到的 visual representation |
| UTES | “uncommon structure” | Uncommon Text-Encoded Structure，例如 tree、graph、nested JSON 等，用来 smuggle content |
| ViTC | “visual-text capability” | 测量模型读取非语义 visual encoding 能力的 benchmark |
| Perplexity filter | “PPL defense” | 拒绝高 perplexity prompt；失败原因是合法 structured input 也会高分 |
| Retokenization | “tokenizer shift defense” | 用不同 tokenizer 预处理 prompt；失败原因是 recognition 是 visual 的 |
| Homoglyph | “lookalike characters” | 看起来与拉丁字母相同的 Unicode 字符；绕过 substring check |

## 延伸阅读

- [Jiang et al. — ArtPrompt (ACL 2024, arXiv:2402.11753)](https://arxiv.org/abs/2402.11753) — ASCII-art jailbreak 论文
- [Li et al. — StructuralSleight (arXiv:2406.08754)](https://arxiv.org/abs/2406.08754) — UTES generalization
- [Chao et al. — PAIR (Lesson 12, arXiv:2310.08419)](https://arxiv.org/abs/2310.08419) — 互补 iterative attack
- [Anil et al. — Many-shot Jailbreaking (Lesson 13)](https://www.anthropic.com/research/many-shot-jailbreaking) — 互补 length attack
