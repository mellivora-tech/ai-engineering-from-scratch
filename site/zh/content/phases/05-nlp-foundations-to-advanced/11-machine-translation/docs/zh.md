# 机器翻译

> 翻译是为 NLP 研究买单了三十年的任务，而且现在仍然在买单。

**类型：** 构建
**语言：** Python
**前置要求：** 阶段 5 第 10 课（Attention Mechanism）、阶段 5 第 04 课（GloVe, FastText, Subword）
**时间：** ~75 分钟

## 问题

模型读取一种语言的句子，并产出另一种语言的句子。长度会变。词序会变。有些源词会映射到多个目标词，反过来也一样。习语拒绝一对一映射。"I miss you" 在法语里是 "tu me manques"，字面意思是“你对我而言缺失”。没有任何词级 alignment 能经受住这个例子。

机器翻译迫使 NLP 发明了 encoder-decoders、attention、transformers，并最终走向整个 LLM 范式。每一步前进都来自一个事实：翻译质量可以衡量，而人类与机器之间的差距又非常顽固。

本课跳过历史课，直接教授 2026 年可工作的 pipeline：预训练多语言 encoder-decoder（NLLB-200 或 mBART）、subword tokenization、beam search、BLEU 和 chrF 评估，以及那些仍然会悄悄进入生产的失败模式。

## 概念

![MT pipeline: tokenize → encode → decode with attention → detokenize](../assets/mt-pipeline.svg)

现代 MT 是在平行文本上训练的 transformer encoder-decoder。Encoder 用源语言自己的 tokenization 读取源文本。Decoder 通过 cross-attention（第 10 课）使用 encoder 输出，一次生成一个 subword。Decoding 使用 beam search 来避开 greedy-decoding 陷阱。输出会被 detokenize、detruecase，并和 reference 打分。

三个操作选择会驱动真实世界的 MT 质量。

- **Tokenizer。** 在混合语言语料上训练的 SentencePiece BPE。跨语言共享词表是 NLLB 中 zero-shot 语言对能工作的原因。
- **Model size。** NLLB-200 distilled 600M 能装进笔记本。NLLB-200 3.3B 是论文中的生产默认。54.5B 是研究天花板。
- **Decoding。** 通用内容用 beam width 4-5。用 length penalty 避免输出太短。需要术语一致性时，用 constrained decoding。

## 构建它

### 第 1 步：一次预训练 MT 调用

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_id = "facebook/nllb-200-distilled-600M"
tok = AutoTokenizer.from_pretrained(model_id, src_lang="eng_Latn")
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

src = "The cats are running."
inputs = tok(src, return_tensors="pt")

out = model.generate(
    **inputs,
    forced_bos_token_id=tok.convert_tokens_to_ids("fra_Latn"),
    num_beams=5,
    length_penalty=1.0,
    max_new_tokens=64,
)
print(tok.batch_decode(out, skip_special_tokens=True)[0])
```

```text
Les chats courent.
```

这里有三件事很重要。`src_lang` 告诉 tokenizer 使用哪种 script 和 segmentation。`forced_bos_token_id` 告诉 decoder 生成哪种语言。二者都是 NLLB 专用技巧；mBART 和 M2M-100 使用各自约定，不能互换。

### 第 2 步：BLEU 与 chrF

BLEU 衡量输出和 reference 的 n-gram 重叠。四个 reference n-gram 大小（1-4）、precisions 的几何平均、对过短输出加 brevity penalty。分数位于 [0, 100]。常用，但解释起来很烦：30 BLEU 是“可用”，40 是“好”，50 是“非常好”；低于 1 BLEU 的差异基本是噪声。

chrF 衡量字符级 F-score。对于形态丰富、BLEU 会低估匹配的语言更敏感。通常和 BLEU 一起报告。

```python
import sacrebleu

hypotheses = ["Les chats courent."]
references = [["Les chats courent."]]

bleu = sacrebleu.corpus_bleu(hypotheses, references)
chrf = sacrebleu.corpus_chrf(hypotheses, references)
print(f"BLEU: {bleu.score:.1f}  chrF: {chrf.score:.1f}")
```

永远使用 `sacrebleu`。它会规范化 tokenization，使不同论文之间分数可比。自己手写 BLEU 计算，就是误导性 benchmark 的来源。

### 三层评估层级（2026）

现代 MT 评估使用三组互补指标。上线时至少带两组。

- **Heuristic**（BLEU、chrF）。快、reference-based、可解释、对 paraphrase 不敏感。用于 legacy comparison 和 regression detection。
- **Learned**（COMET、BLEURT、BERTScore）。在人类判断上训练的 neural models；比较翻译和 source/reference 的语义相似度。自 2023 年以来，COMET 与 MT 研究的相关性最高，是 2026 年重视质量场景的生产默认。
- **LLM-as-judge**（reference-free）。Prompt 大模型，从 fluency、adequacy、tone、cultural appropriateness 上给翻译打分。当 rubric 设计良好时，GPT-4-as-judge 与人类一致约 80%。用于没有 reference 的开放内容。

务实的 2026 栈：用 `sacrebleu` 计算 BLEU 和 chrF，用 `unbabel-comet` 计算 COMET，并用 prompted LLM 做最终面向人的信号。在信任任何指标进入生产数据前，先用 50-100 个，人类标注样例校准。

Reference-free metrics（COMET-QE、BLEURT-QE、LLM-as-judge）让你在没有 reference 的情况下评估翻译，这对没有参考译文的长尾语言对很重要。

### 第 3 步：生产中会坏掉什么

上面的工作 pipeline 80% 时间会给出流畅翻译，剩下 20% 会静默失败。具名失败模式：

- **Hallucination。** 模型发明源文本中不存在的内容。常见于不熟悉的领域词汇。症状：输出很流畅，但声称了源文本没说的事实。缓解：对领域术语做 constrained decoding，监管内容做人审，监控比输入长很多的输出。
- **Off-target generation。** 模型翻译成了错误语言。NLLB 在稀有语言对上意外容易这样。缓解：验证 `forced_bos_token_id`，并且永远用 language-ID model 检查输出。
- **Terminology drift。** "Sign up" 在文档 1 里变成 "s'inscrire"，在文档 2 里变成 "créer un compte"。对于 UI 文本和面向用户字符串，一致性比原始质量更重要。缓解：glossary-constrained decoding 或 post-edit dictionary。
- **Formality mismatch。** 法语 "tu" vs "vous"，日语敬语级别。模型会选择训练中更常见的形式。对于面向客户内容，这通常是错的。缓解：如果模型支持，使用带 formality token 的 prompt prefix，或在 formal-only 语料上 fine-tune 小模型。
- **短输入上的长度爆炸。** 很短的输入句子经常生成过长翻译，因为 source tokens 少于约 5 个时 length penalty 会失效。缓解：按源长度设置硬 max-length cap。

### 第 4 步：针对领域 fine-tuning

预训练模型是通才。法律、医疗或游戏对话翻译会明显受益于领域平行数据上的 fine-tuning。配方并不神秘：

```python
from transformers import Trainer, TrainingArguments
from datasets import Dataset

pairs = [
    {"src": "The defendant pleaded guilty.", "tgt": "L'accusé a plaidé coupable."},
]

ds = Dataset.from_list(pairs)


def preprocess(ex):
    return tok(
        ex["src"],
        text_target=ex["tgt"],
        truncation=True,
        max_length=128,
        padding="max_length",
    )


ds = ds.map(preprocess, remove_columns=["src", "tgt"])

args = TrainingArguments(output_dir="out", per_device_train_batch_size=4, num_train_epochs=3, learning_rate=3e-5)
Trainer(model=model, args=args, train_dataset=ds).train()
```

几千条高质量平行样例，胜过几十万条噪声网页抓取样例。训练数据质量是生产中最大的杠杆。

## 使用它

2026 年的 MT 生产栈：

| 使用场景 | 推荐起点 |
|---------|---------------------------|
| 任意到任意，200 种语言 | `facebook/nllb-200-distilled-600M`（笔记本）或 `nllb-200-3.3B`（生产） |
| 以英语为中心，高质量，50 种语言 | `facebook/mbart-large-50-many-to-many-mmt` |
| 短运行、低成本推理、英法/德/西 | Helsinki-NLP / Marian models |
| 延迟关键的浏览器端 | ONNX-quantized Marian（~50 MB） |
| 最高质量，愿意付费 | GPT-4 / Claude / Gemini with translation prompts |

截至 2026 年，LLMs 在若干语言对上已经超过专用 MT 模型，尤其是习语内容和长上下文。取舍是 per-token 成本和延迟。当上下文长度、风格一致性或通过 prompting 做领域适配比吞吐更重要时，选择 LLM。

## 交付它

保存为 `outputs/skill-mt-evaluator.md`：

```markdown
---
name: mt-evaluator
description: 评估机器翻译输出是否可以发布。
version: 1.0.0
phase: 5
lesson: 11
tags: [nlp, translation, evaluation]
---

给定 source text 和 candidate translation，输出：

1. Automatic score estimate。你预期的 BLEU 和 chrF 范围。说明是否有 reference。
2. 五点人类可验证 checklist：(a) content preservation（无 hallucinations），(b) correct language，(c) register / formality match，(d) 如提供 glossary，则术语一致，(e) 无 truncation 或 length explosion。
3. 一个领域特定问题要探测。例如法律：named entities 和 statute citations。医疗：drug names 和 dosages。UI：placeholder variables `{name}`。
4. Confidence flag。"Ship" / "Ship with review" / "Do not ship"。和第 2 步发现的问题严重程度绑定。

拒绝在没有 output language-ID check 的情况下发布翻译。除非用户明确选择 reference-free scoring（COMET-QE、BLEURT-QE），否则拒绝在没有 reference 的情况下评估。把超过 1000 tokens 的内容标记为可能需要 chunked translation。
```

## 练习

1. **简单。** 使用 `nllb-200-distilled-600M` 把 5 句英文段落翻译成法语，再翻回英语。衡量 round-trip 和原文有多接近。你应该会看到语义保留，但词语选择有漂移。
2. **中等。** 使用 `fasttext lid.176` 或 `langdetect` 在翻译输出上实现 language-ID check。集成到 MT 调用中，让 off-target generations 在返回前被捕获。
3. **困难。** 在你选择的 5,000 对领域语料上 fine-tune `nllb-200-distilled-600M`。在 held-out set 上衡量 fine-tuning 前后的 BLEU。报告哪些句子类型改善了，哪些退化了。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|-----------------|-----------------------|
| BLEU | 翻译分数 | 带 brevity penalty 的 n-gram precision。[0, 100]。 |
| chrF | Character F-score | 字符级 F-score。对形态丰富语言更敏感。 |
| NMT | Neural MT | 在平行文本上训练的 transformer encoder-decoder。2017+ 默认选择。 |
| NLLB | No Language Left Behind | Meta 的 200 语言 MT 模型家族。 |
| Constrained decoding | 受控输出 | 强制特定 tokens 或 n-grams 在输出中出现 / 不出现。 |
| Hallucination | 编造内容 | 模型输出不受源文本支持。 |

## 延伸阅读

- [Costa-jussà et al. (2022). No Language Left Behind: Scaling Human-Centered Machine Translation](https://arxiv.org/abs/2207.04672) — NLLB 论文。
- [Post (2018). A Call for Clarity in Reporting BLEU Scores](https://aclanthology.org/W18-6319/) — 为什么 `sacrebleu` 是报告 BLEU 的唯一正确方式。
- [Popović (2015). chrF: character n-gram F-score for automatic MT evaluation](https://aclanthology.org/W15-3049/) — chrF 论文。
- [Hugging Face MT guide](https://huggingface.co/docs/transformers/tasks/translation) — 实用 fine-tuning walkthrough。
