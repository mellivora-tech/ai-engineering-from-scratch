# 从零构建 Tokenizer

> 第 01 课给了你一个玩具。这一课给你一件武器。

**类型：** 构建
**语言：** Python
**前置要求：** 阶段 10，第 01 课（Tokenizers：BPE、WordPiece、SentencePiece）
**时间：** ~90 分钟

## 学习目标

- 构建一个生产级 BPE tokenizer，能处理 Unicode、空白规范化和 special tokens
- 实现 byte-level fallback，使 tokenizer 可以编码任何输入（包括 emoji、CJK 和代码），且不会产生未知 token
- 添加 pre-tokenization 正则模式，在应用 BPE 合并前按词边界切分文本
- 在语料上训练自定义 tokenizer，并用多语言文本把它的 compression ratio 与 tiktoken 对比

## 问题

你在第 01 课实现的 BPE tokenizer 能处理英文文本。现在把日语丢给它。或者 emoji。或者混合了 tab 和空格的 Python 代码。

它会坏掉。

不是因为 BPE 错了，而是因为实现不完整。生产 tokenizer 要能处理任意编码的原始字节，在切分前规范化 Unicode，管理永远不会被合并的 special tokens，把 pre-tokenization 和 subword splitting 串起来，并且速度要快到不会拖慢一个处理 15 万亿 token 的训练 pipeline。

GPT-2 的 tokenizer 有 50,257 个 token。Llama 3 有 128,256 个。GPT-4 大约有 100,000 个。这些不是玩具数字。这些词表背后的 merge table 是在数百 GB 文本上训练出来的；而外围机制，normalization、pre-tokenization、special token injection、chat template formatting，才是区分“只能处理 hello world”的 tokenizer 和“能处理整个互联网”的 tokenizer 的关键。

你要构建的就是这套机制。

## 概念

### 完整 Pipeline

生产 tokenizer 不是一个算法。它是一个由五个阶段组成的 pipeline，每个阶段解决不同问题。

```mermaid
graph LR
    A[Raw Text] --> B[Normalize]
    B --> C[Pre-Tokenize]
    C --> D[BPE Merge]
    D --> E[Special Tokens]
    E --> F[Token IDs]

    style A fill:#1a1a2e,stroke:#e94560,color:#fff
    style B fill:#1a1a2e,stroke:#e94560,color:#fff
    style C fill:#1a1a2e,stroke:#e94560,color:#fff
    style D fill:#1a1a2e,stroke:#e94560,color:#fff
    style E fill:#1a1a2e,stroke:#e94560,color:#fff
    style F fill:#1a1a2e,stroke:#e94560,color:#fff
```

每个阶段都有具体职责：

| Stage | What It Does | Why It Matters |
|-------|-------------|----------------|
| Normalize | NFKC Unicode，可选 lowercase，可选 strip accents | `"fi"` 连字（U+FB01）变成 `"fi"`（两个字符）。没有这一步，同一个词会得到不同 token。 |
| Pre-Tokenize | 在 BPE 前把文本切成 chunk | 防止 BPE 跨词边界合并。`"the cat"` 不应该产生 token `"e c"`。 |
| BPE Merge | 对字节序列应用学到的 merge rules | 核心压缩步骤。把原始字节变成 subword token。 |
| Special Tokens | 注入 [BOS]、[EOS]、[PAD]、chat template markers | 这些 token 有固定 ID。它们永远不参与 BPE 合并。模型需要它们来表示结构。 |
| ID Mapping | 把 token 字符串转换成整数 ID | 模型看到的是整数，不是字符串。 |

### Byte-Level BPE

第 01 课的 tokenizer 作用在 UTF-8 字节上。这是正确选择。但我们跳过了一个重要问题：如果这些字节不是合法 UTF-8 会怎样？

Byte-level BPE 通过把每一种可能的字节值（0-255）都视为有效 token 来解决这个问题。基础词表正好有 256 个条目。任何文件，无论是文本、二进制还是损坏文件，都可以被 tokenized，且不会产生未知 token。

GPT-2 加了一个技巧：把每个字节映射到可打印 Unicode 字符，让词表保持可读。字节 `0x20`（空格）会映射成它们映射表中的某个字符。这纯粹是显示层面的。算法并不关心。

真正的威力在于：byte-level BPE 能处理地球上的每一种语言。中文字符每个是 3 个 UTF-8 字节。日文可能是 3-4 个字节。阿拉伯文、天城文、emoji，全都只是字节序列。BPE 算法在这些字节序列中寻找模式，方式与在英文 ASCII 字节中寻找模式完全一样。

### Pre-Tokenization

在 BPE 触碰文本前，你需要先把文本切成 chunk。这可以防止 merge 算法创造跨越词边界的 token。

GPT-2 使用一个正则模式切分文本：

```
'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
```

这个模式会切分缩写（`"don't"` 变成 `"don"` + `"'t"`）、带可选前导空格的单词、数字、标点和空白。前导空格会保留在单词上，所以 `"the cat"` 会变成 `[" the", " cat"]`，而不是 `["the", " ", "cat"]`。

Llama 使用 SentencePiece，完全跳过正则。它把原始字节流视作一个长序列，让 BPE 算法自己找边界。这更简单，但给 BPE 更多自由去创造跨词 token。

这个选择很重要。GPT-2 的正则会防止 tokenizer 学到一个词末尾的 `"the"` 和下一个词开头的 `"the"` 应该合并。SentencePiece 允许这种情况，有时压缩更高效，但 token 的可解释性更差。

### Special Tokens

每个生产 tokenizer 都会为结构标记保留 token ID：

| Token | Purpose | Used By |
|-------|---------|---------|
| `[BOS]` / `<s>` | 序列开始 | Llama 3, GPT |
| `[EOS]` / `</s>` | 序列结束 | 所有模型 |
| `[PAD]` | batch 对齐的 padding | BERT, T5 |
| `[UNK]` | 未知 token（byte-level BPE 会消除它） | BERT, WordPiece |
| `<\|im_start\|>` | Chat 消息边界开始 | ChatGPT, Qwen |
| `<\|im_end\|>` | Chat 消息边界结束 | ChatGPT, Qwen |
| `<\|user\|>` | 用户轮次标记 | Llama 3 |
| `<\|assistant\|>` | assistant 轮次标记 | Llama 3 |

Special tokens 永远不会被 BPE 拆分。它们会在 merge 算法运行前被精确匹配，替换成固定 ID；周围文本则正常 tokenized。

### Chat Templates

这是大多数人困惑、也是大多数实现出错的地方。

当你向 chat model 发送消息时，API 接收的是消息列表：

```
[
  {"role": "system", "content": "You are helpful."},
  {"role": "user", "content": "Hello"},
  {"role": "assistant", "content": "Hi there!"}
]
```

模型看不到 JSON。它看到的是一个扁平 token 序列。chat template 使用 special tokens 把消息转换成这个扁平序列。每个模型的格式都不同：

```
Llama 3:
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are helpful.<|eot_id|><|start_header_id|>user<|end_header_id|>

Hello<|eot_id|><|start_header_id|>assistant<|end_header_id|>

Hi there!<|eot_id|>

ChatGPT:
<|im_start|>system
You are helpful.<|im_end|>
<|im_start|>user
Hello<|im_end|>
<|im_start|>assistant
Hi there!<|im_end|>
```

模板写错，模型就会输出垃圾。它训练时见到的是一种精确格式。任何偏差，一个缺失的换行、一个调换的 token、一个多余空格，都会把输入推到训练分布之外。

### 速度

Python 对生产 tokenization 来说太慢。

tiktoken（OpenAI）用 Rust 编写，并提供 Python binding。HuggingFace tokenizers 也是 Rust。SentencePiece 是 C++。这些实现比纯 Python 快 10-100 倍。

做个量级感知：如果要为 Llama 3 pre-training tokenize 15 万亿 token，用每秒 100 万 token 的速度（很快的 Python）需要 174 天。用每秒 1 亿 token 的速度（Rust）需要 1.7 天。

你用 Python 构建是为了理解算法。在生产中，你会使用编译实现，只接触 Python wrapper。

## 构建它

### 第 1 步：Byte-Level Encoding

基础。把任意字符串转换成字节序列，把每个字节映射成可显示的字符，并实现反向转换。

```python
def bytes_to_tokens(text):
    return list(text.encode("utf-8"))

def tokens_to_text(token_bytes):
    return bytes(token_bytes).decode("utf-8", errors="replace")
```

在多语言文本上测试字节数：

```python
texts = [
    ("English", "hello"),
    ("Chinese", "你好"),
    ("Emoji", "🔥"),
    ("Mixed", "hello你好🔥"),
]

for label, text in texts:
    b = bytes_to_tokens(text)
    print(f"{label}: {len(text)} chars -> {len(b)} bytes -> {b}")
```

`"hello"` 是 5 个字节。`"你好"` 是 6 个字节（每个字符 3 个）。火焰 emoji 是 4 个字节。byte-level tokenizer 不关心它是哪种语言。字节就是字节。

### 第 2 步：使用 Regex 的 Pre-Tokenizer

用 GPT-2 正则模式把文本切成 chunk。每个 chunk 都由 BPE 独立 tokenize。

```python
import re

try:
    import regex
    GPT2_PATTERN = regex.compile(
        r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    )
except ImportError:
    GPT2_PATTERN = re.compile(
        r"""'(?:[sdmt]|ll|ve|re)| ?[a-zA-Z]+| ?[0-9]+| ?[^\s\w]+|\s+(?!\S)|\s+"""
    )

def pre_tokenize(text):
    return [match.group() for match in GPT2_PATTERN.finditer(text)]
```

`regex` 模块支持 Unicode property escapes（`\p{L}` 表示字母，`\p{N}` 表示数字）。标准库 `re` 模块不支持，所以我们 fallback 到 ASCII 字符类。生产多语言 tokenizer 应安装 `regex`。

试一下：

```python
print(pre_tokenize("Hello, world! Don't stop."))
# [' Hello', ',', ' world', '!', " Don", "'t", ' stop', '.']
```

前导空格保留在单词上。缩写在 apostrophe 处切分。标点成为自己的 chunk。BPE 永远不会跨这些边界合并 token。

### 第 3 步：在字节序列上执行 BPE

第 01 课的核心算法，但现在要在 pre-tokenized chunk 上独立运行。

```python
from collections import Counter

def get_byte_pairs(chunks):
    pairs = Counter()
    for chunk in chunks:
        byte_seq = list(chunk.encode("utf-8"))
        for i in range(len(byte_seq) - 1):
            pairs[(byte_seq[i], byte_seq[i + 1])] += 1
    return pairs

def apply_merge(byte_seq, pair, new_id):
    merged = []
    i = 0
    while i < len(byte_seq):
        if i < len(byte_seq) - 1 and byte_seq[i] == pair[0] and byte_seq[i + 1] == pair[1]:
            merged.append(new_id)
            i += 2
        else:
            merged.append(byte_seq[i])
            i += 1
    return merged
```

### 第 4 步：Special Token 处理

Special tokens 需要精确匹配和固定 ID。它们完全绕过 BPE。

```python
class SpecialTokenHandler:
    def __init__(self):
        self.special_tokens = {}
        self.pattern = None

    def add_token(self, token_str, token_id):
        self.special_tokens[token_str] = token_id
        escaped = [re.escape(t) for t in sorted(self.special_tokens.keys(), key=len, reverse=True)]
        self.pattern = re.compile("|".join(escaped))

    def split_with_specials(self, text):
        if not self.pattern:
            return [(text, False)]
        parts = []
        last_end = 0
        for match in self.pattern.finditer(text):
            if match.start() > last_end:
                parts.append((text[last_end:match.start()], False))
            parts.append((match.group(), True))
            last_end = match.end()
        if last_end < len(text):
            parts.append((text[last_end:], False))
        return parts
```

### 第 5 步：完整 Tokenizer 类

把所有阶段串起来：normalize、按 special token 切分、pre-tokenize、BPE merge、映射成 ID。

```python
import unicodedata

class ProductionTokenizer:
    def __init__(self):
        self.merges = {}
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.special_handler = SpecialTokenHandler()
        self.next_id = 256

    def normalize(self, text):
        return unicodedata.normalize("NFKC", text)

    def train(self, text, num_merges):
        text = self.normalize(text)
        chunks = pre_tokenize(text)
        chunk_bytes = [list(chunk.encode("utf-8")) for chunk in chunks]

        for i in range(num_merges):
            pairs = Counter()
            for seq in chunk_bytes:
                for j in range(len(seq) - 1):
                    pairs[(seq[j], seq[j + 1])] += 1
            if not pairs:
                break
            best = max(pairs, key=pairs.get)
            new_id = self.next_id
            self.next_id += 1
            self.merges[best] = new_id
            self.vocab[new_id] = self.vocab[best[0]] + self.vocab[best[1]]
            chunk_bytes = [apply_merge(seq, best, new_id) for seq in chunk_bytes]

    def add_special_token(self, token_str):
        token_id = self.next_id
        self.next_id += 1
        self.special_handler.add_token(token_str, token_id)
        self.vocab[token_id] = token_str.encode("utf-8")
        return token_id

    def encode(self, text):
        text = self.normalize(text)
        parts = self.special_handler.split_with_specials(text)
        all_ids = []
        for part_text, is_special in parts:
            if is_special:
                all_ids.append(self.special_handler.special_tokens[part_text])
            else:
                for chunk in pre_tokenize(part_text):
                    byte_seq = list(chunk.encode("utf-8"))
                    for pair, new_id in self.merges.items():
                        byte_seq = apply_merge(byte_seq, pair, new_id)
                    all_ids.extend(byte_seq)
        return all_ids

    def decode(self, ids):
        byte_parts = []
        for token_id in ids:
            if token_id in self.vocab:
                byte_parts.append(self.vocab[token_id])
        return b"".join(byte_parts).decode("utf-8", errors="replace")

    def vocab_size(self):
        return len(self.vocab)
```

### 第 6 步：多语言测试

真正的测试。把英文、中文、emoji 和代码都扔进去。

```python
corpus = (
    "The quick brown fox jumps over the lazy dog. "
    "The quick brown fox runs through the forest. "
    "Machine learning models process natural language. "
    "Deep learning transforms how we build software. "
    "def train(model, data): return model.fit(data) "
    "def predict(model, x): return model(x) "
)

tok = ProductionTokenizer()
tok.train(corpus, num_merges=50)

bos = tok.add_special_token("<|begin|>")
eos = tok.add_special_token("<|end|>")

test_texts = [
    "The quick brown fox.",
    "你好世界",
    "Hello 🌍 World",
    "def foo(x): return x + 1",
    f"<|begin|>Hello<|end|>",
]

for text in test_texts:
    ids = tok.encode(text)
    decoded = tok.decode(ids)
    print(f"Input:   {text}")
    print(f"Tokens:  {len(ids)} ids")
    print(f"Decoded: {decoded}")
    print()
```

中文字符每个产生 3 个字节。emoji 产生 4 个字节。它们都不会让 tokenizer 崩溃。也不会产生未知 token。这就是 byte-level BPE 的力量。

## 使用它

### 比较真实 Tokenizers

加载 Llama 3、GPT-4 和 Mistral 的实际 tokenizer。观察它们如何处理同一个多语言段落。

```python
import tiktoken

gpt4_enc = tiktoken.get_encoding("cl100k_base")

test_paragraph = "Machine learning is powerful. 机器学习很强大。 L'apprentissage automatique est puissant. 🤖💪"

tokens = gpt4_enc.encode(test_paragraph)
pieces = [gpt4_enc.decode([t]) for t in tokens]
print(f"GPT-4 ({len(tokens)} tokens): {pieces}")
```

```python
from transformers import AutoTokenizer

llama_tok = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
mistral_tok = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

for name, tok in [("Llama 3", llama_tok), ("Mistral", mistral_tok)]:
    tokens = tok.encode(test_paragraph)
    pieces = tok.convert_ids_to_tokens(tokens)
    print(f"{name} ({len(tokens)} tokens): {pieces[:20]}...")
```

你会看到同一段文本得到不同 token 数。128K 词表的 Llama 3 更积极地合并常见模式。100K 的 GPT-4 位于中间。32K 的 Mistral 产生更多 token，但 embedding 层更小。

权衡总是一样：更大词表意味着更短序列，但也意味着更多参数。

## 交付它

本课会产出一个用于构建和调试生产 tokenizer 的 prompt。见 `outputs/prompt-tokenizer-builder.md`。

## 练习

1. **简单：** 添加 `get_token_bytes(id)` 方法，显示任意 token ID 的原始字节。用它检查最常见的 merged tokens 实际代表什么。
2. **中等：** 实现 Llama 风格的 pre-tokenizer：按空白和数字切分，但保留前导空格。在同一语料上把它的词表与 GPT-2 正则方案比较。
3. **困难：** 添加 chat template 方法，接收 `{"role": ..., "content": ...}` 消息列表，并为 Llama 3 chat 格式生成正确 token 序列。用 HuggingFace 实现做对照测试。

## 关键词

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| Byte-level BPE | “作用在字节上的 tokenizer” | 基础词表包含 256 个字节值的 BPE；可以处理任何输入且没有未知 token |
| Pre-tokenization | “BPE 前的切分” | 正则或规则切分，用来防止 BPE 跨词边界合并 |
| NFKC normalization | “Unicode 清理” | 先做 canonical decomposition，再做 compatibility composition；`"fi"` 连字变成 `"fi"`，全角 `"A"` 变成 `"A"` |
| Chat template | “消息如何变成 token” | 把 role/content 消息列表转换成扁平 token 序列的精确格式；模型特定，必须匹配训练格式 |
| Special tokens | “控制 token” | 绕过 BPE 的保留 token ID，如 [BOS]、[EOS]、[PAD]、chat markers，在 merge 前精确匹配 |
| Fertility | “每个词多少 token” | 输出 token 与输入词的比例；GPT-4 英文约 1.3，韩语 2-3，更高意味着浪费 context |
| tiktoken | “OpenAI tokenizer” | 带 Python binding 的 Rust BPE 实现，比纯 Python 快 10-100 倍 |
| Merge table | “词表” | 训练中学到的 byte-pair merges 有序列表；这就是 tokenizer 学到的知识 |

## 延伸阅读

- [OpenAI tiktoken source](https://github.com/openai/tiktoken) -- GPT-3.5/4 使用的 Rust BPE 实现
- [HuggingFace tokenizers](https://github.com/huggingface/tokenizers) -- 支持 BPE、WordPiece、Unigram 的 Rust tokenizer 库
- [Llama 3 paper (Meta, 2024)](https://arxiv.org/abs/2407.21783) -- 128K 词表和 tokenizer 训练细节
- [SentencePiece (Kudo & Richardson, 2018)](https://arxiv.org/abs/1808.06226) -- 语言无关 tokenization
- [GPT-2 tokenizer source](https://github.com/openai/gpt-2/blob/master/src/encoder.py) -- 最初的 byte-to-Unicode 映射
