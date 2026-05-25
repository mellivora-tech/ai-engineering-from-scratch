# Long-Context Evaluation：NIAH、RULER、LongBench、MRCR

> Gemini 3 Pro 宣称 10M tokens context。到 1M tokens 时，8-needle MRCR 降到 26.3%。Advertised 不等于 usable。Long-context evaluation 告诉你正在交付的模型真实容量。

**类型：** 学习
**语言：** Python
**前置要求：** 阶段 5 · 13（问答），阶段 5 · 23（Chunking Strategies）
**时间：** ~60 分钟

## 问题

你有一份 200 页合同。模型声称有 1M-token context。你把合同贴进去问：“What is the termination clause?” 模型回答了，但它从封面页回答，因为 termination clause 位于 120k tokens 深处，超过了模型真正会 attend 的位置。

这就是 2026 年 context-capacity gap。规格表写 1M 或 10M。现实说可用的通常只有 60-70%，而且“可用”取决于任务。

- **Retrieval（single needle in haystack）：** frontier models 在 advertised max 之前几乎完美。
- **Multi-hop / aggregation：** 大多数模型过了 ~128k 后急剧退化。
- **Reasoning over dispersed facts：** 最先失败的任务。

Long-context evaluation 测量这些轴。本课命名 benchmarks、每个实际测什么，以及如何为你的领域构建 custom needle test。

## 概念

![NIAH baseline, RULER multi-task, LongBench holistic](../assets/long-context-eval.svg)

**Needle-in-a-Haystack（NIAH, 2023）。** 在长 context 的受控深度放置一个事实（“the magic word is pineapple”）。问模型取回它。扫 depth × length。最早的 long-context benchmark。Frontier models 现在能 saturate 它；它是必要但不充分的 baseline。

**RULER（Nvidia, 2024）。** 13 种 task types，跨 4 类：retrieval（single / multi-key / multi-value）、multi-hop tracing（variable tracking）、aggregation（common word frequency）、QA。可配置 context length（4k 到 128k+）。能揭示 saturate NIAH 但 multi-hop 失败的模型。2024 release 中，17 个声称 32k+ context 的模型，只有一半在 32k 保持质量。

**LongBench v2（2024）。** 503 个 multiple-choice questions，8k-2M word contexts，六类任务：single-doc QA、multi-doc QA、long in-context learning、long dialogue、code repo、long structured data。生产中评估真实 long-context 行为的 benchmark。

**MRCR（Multi-Round Coreference Resolution）。** 大规模 multi-turn coreference。8-needle、24-needle、100-needle variants。暴露模型在 attention 退化前能同时处理多少事实。

**NoLiMa。** “Non-lexical needle”。Needle 和 query 没有字面重叠；retrieval 需要一步 semantic reasoning。比 NIAH 更难。

**HELMET。** 拼接很多 documents，从任意一篇中提问。测试 selective attention。

**BABILong。** 把 bAbI reasoning chains 嵌进无关 haystacks。测试 reasoning-in-a-haystack，而不只是 retrieval。

### 实际应该报告什么

- **Advertised context window。** 规格表数字。
- **Effective retrieval length。** 在某阈值（例如 90%）下 NIAH 通过长度。
- **Effective reasoning length。** 在同阈值下 multi-hop 或 aggregation 通过长度。
- **Degradation curve。** Accuracy vs context length，按 task type 分开画。

给你的 spec sheet 两个数字：retrieval-effective 和 reasoning-effective。通常 reasoning-effective 只有 advertised window 的 25-50%。

## 构建它

### 第 1 步：为你的领域做 custom NIAH

见 `code/main.py`。骨架：

```python
def build_haystack(filler_text, needle, depth_ratio, total_tokens):
    if not (0.0 <= depth_ratio <= 1.0):
        raise ValueError(f"depth_ratio must be in [0, 1], got {depth_ratio}")
    if total_tokens <= 0:
        raise ValueError(f"total_tokens must be positive, got {total_tokens}")

    filler_tokens = tokenize(filler_text)
    needle_tokens = tokenize(needle)
    if not filler_tokens:
        raise ValueError("filler_text produced no tokens")

    # Repeat filler until long enough to fill the haystack body.
    body_len = max(total_tokens - len(needle_tokens), 0)
    while len(filler_tokens) < body_len:
        filler_tokens = filler_tokens + filler_tokens
    filler_tokens = filler_tokens[:body_len]

    insert_at = min(int(body_len * depth_ratio), body_len)
    haystack = filler_tokens[:insert_at] + needle_tokens + filler_tokens[insert_at:]
    return " ".join(haystack)


def score_niah(model, haystack, question, expected):
    answer = model.complete(f"Context: {haystack}\nQ: {question}\nA:", max_tokens=50)
    return 1 if expected.lower() in answer.lower() else 0
```

扫 `depth_ratio` ∈ {0, 0.25, 0.5, 0.75, 1.0} × `total_tokens` ∈ {1k, 4k, 16k, 64k}。画 heatmap。这就是目标模型的 NIAH card。

### 第 2 步：multi-needle variant

```python
def build_multi_needle(filler, needles, total_tokens):
    depths = [0.1, 0.4, 0.7]
    chunks = [filler[:int(total_tokens * 0.1)]]
    for depth, needle in zip(depths, needles):
        chunks.append(needle)
        next_chunk = filler[int(total_tokens * depth): int(total_tokens * (depth + 0.3))]
        chunks.append(next_chunk)
    return " ".join(chunks)
```

像 “What are the three magic words?” 这样的问题需要取回三个事实。Single-needle success 不能预测 multi-needle success。

### 第 3 步：multi-hop variable tracing（RULER-style）

```python
haystack = """X1 = 42. ... (filler) ... X2 = X1 + 10. ... (filler) ... X3 = X2 * 2."""
question = "What is X3?"
```

答案需要链接三次 assignment。Frontier models 在 128k 时常掉到 50-70% accuracy。

### 第 4 步：在你的 stack 上跑 LongBench v2

```python
from datasets import load_dataset
longbench = load_dataset("THUDM/LongBench-v2")

def eval_model_on_longbench(model, subset="single-doc-qa"):
    tasks = [x for x in longbench["test"] if x["task"] == subset]
    correct = 0
    for x in tasks:
        answer = model.complete(x["context"] + "\n\nQ: " + x["question"], max_tokens=20)
        if normalize(answer) == normalize(x["answer"]):
            correct += 1
    return correct / len(tasks)
```

按 category 报告 accuracy。Aggregate scores 会隐藏 task-level 差异。

## 坑

- **NIAH-only evaluation。** 1M tokens 通过 NIAH 不代表 multi-hop 能行。始终跑 RULER 或 custom multi-hop test。
- **Uniform depth sampling。** 很多实现只测 depth=0.5。测试 depth=0、0.25、0.5、0.75、1.0，“lost in the middle” 效应是真实的。
- **Lexical overlap with filler。** 如果 needle 和 filler 共享 keywords，retrieval 变得太简单。使用 NoLiMa 风格 non-overlapping needles。
- **忽略 latency。** 1M-token prompts 需要 30-120 秒 prefill。Accuracy 之外还要测 time-to-first-token。
- **Vendor-self-reported numbers。** OpenAI、Google、Anthropic 都发布自家 scores。始终在你的 use case 上独立重跑。

## 使用它

2026 年技术栈：

| 场景 | Benchmark |
|-----------|-----------|
| 快速 sanity check | Custom NIAH at 3 depths × 3 lengths |
| 生产模型选择 | RULER（13 tasks）at your target length |
| Real-world QA quality | LongBench v2 single-doc-QA subset |
| Multi-hop reasoning | BABILong 或 custom variable-tracing |
| Conversational / dialogue | MRCR 8-needle at your target length |
| Model upgrade regression | Fixed in-house NIAH + RULER harness，每次新模型都跑 |

生产经验法则：在目标长度上没有 NIAH + 1 个 reasoning task 前，不要相信 context window。

## 交付它

保存为 `outputs/skill-long-context-eval.md`：

```markdown
---
name: long-context-eval
description: 为给定 model 和 use case 设计 long-context evaluation battery。
version: 1.0.0
phase: 5
lesson: 28
tags: [nlp, long-context, evaluation]
---

给定 target model、target context length 和 use case，输出：

1. Tests。NIAH depth × length grid；RULER multi-hop；custom domain task。
2. Sampling。每个 length 上 depths 0、0.25、0.5、0.75、1.0。
3. Metrics。Retrieval pass rate；reasoning pass rate；time-to-first-token；cost-per-query。
4. Cutoff。Effective retrieval length（90% pass）和 effective reasoning length（70% pass）。两者都报告。
5. Regression。Fixed harness，每次 model upgrade 重跑，surface deltas。

拒绝只相信 model card 的 context window。拒绝任何 multi-hop workload 的 NIAH-only evaluation。拒绝把 vendor self-reported long-context scores 当作 independent evidence。
```

## 练习

1. **简单。** 构建 3 depths（0.25、0.5、0.75）× 3 lengths（1k、4k、16k）的 NIAH。在任意模型上运行。把 pass rate 画成 3×3 heatmap。
2. **中等。** 增加 3-needle variant。测量每个 length 下是否能取回全部 3 个。和同 length 的 single-needle pass rate 比较。
3. **困难。** 构造一个 variable-tracing task（X1 → X2 → X3，3 hops），嵌入 64k filler。比较 3 个 frontier models 的 accuracy。报告每个模型的 effective reasoning length。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|-----------------|-----------------------|
| NIAH | Needle in haystack | 在 filler 中埋一个事实，问模型取回。 |
| RULER | NIAH on steroids | 13 种任务，覆盖 retrieval / multi-hop / aggregation / QA。 |
| Effective context | 真实容量 | Accuracy 仍高于阈值的长度。 |
| Lost in the middle | Depth bias | 模型对长输入中间内容注意不足。 |
| Multi-needle | 同时很多事实 | 多个埋点；测试 attention juggling，不只是 retrieval。 |
| MRCR | Multi-round coref | 8、24 或 100-needle coreference；暴露 attention saturation。 |
| NoLiMa | Non-lexical needle | Needle 和 query 没有字面 tokens 重叠；需要 reasoning。 |

## 延伸阅读

- [Kamradt (2023). Needle in a Haystack analysis](https://github.com/gkamradt/LLMTest_NeedleInAHaystack) — 原始 NIAH repo。
- [Hsieh et al. (2024). RULER: What's the Real Context Size of Your Long-Context LMs?](https://arxiv.org/abs/2404.06654) — multi-task benchmark。
- [Bai et al. (2024). LongBench v2](https://arxiv.org/abs/2412.15204) — real-world long-context eval。
- [Modarressi et al. (2024). NoLiMa: Non-lexical needles](https://arxiv.org/abs/2404.06666) — 更难的 needles。
- [Kuratov et al. (2024). BABILong](https://arxiv.org/abs/2406.10149) — reasoning-in-haystack。
- [Liu et al. (2024). Lost in the Middle: How Language Models Use Long Contexts](https://arxiv.org/abs/2307.03172) — depth-bias 论文。
