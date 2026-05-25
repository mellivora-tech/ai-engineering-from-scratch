# LLM Evaluation：RAGAS、DeepEval、G-Eval

> Exact-match 和 F1 会错过语义等价。人工 review 无法扩展。LLM-as-judge 是生产答案，但要有足够 calibration 才能信这个数字。

**类型：** 构建
**语言：** Python
**前置要求：** 阶段 5 · 13（问答），阶段 5 · 14（信息检索）
**时间：** ~75 分钟

## 问题

你的 RAG 系统回答：“June 29th, 2007.”
Gold reference 是：“June 29, 2007.”
Exact Match 得 0。F1 约 75%。人类会给 100%。

现在乘以 10,000 个 test cases。再乘以 retriever、chunking、prompt 或 model 的每一次改动。你需要一个 evaluator：理解意义、能低成本大规模运行、不对 regressions 撒谎，并能暴露正确的 failure modes。

2026 年有三个 framework 主导这个问题。

- **RAGAS。** Retrieval-Augmented Generation ASsessment。四个 RAG metrics（faithfulness、answer-relevance、context-precision、context-recall），后端用 NLI + LLM-judge。有研究背书，轻量。
- **DeepEval。** LLM 的 pytest。G-Eval、task-completion、hallucination、bias metrics。CI/CD-native。
- **G-Eval。** 一种方法（也是 DeepEval metric）：带 chain-of-thought、自定义 criteria、0-1 score 的 LLM-as-judge。

三者都依赖 LLM-as-judge。本课建立方法直觉，以及围绕它的信任层。

## 概念

![Four evaluation dimensions, LLM-as-judge architecture](../assets/llm-evaluation.svg)

**LLM-as-judge。** 用 LLM 根据 rubric 给 outputs 打分，替代静态 metric。给定 `(query, context, answer)`，prompt 一个 judge LLM：“Score 0-1 on faithfulness.” 返回 score。

为什么有效：LLMs 能以极低成本近似人类判断。GPT-4o-mini 每个 scored case 约 $0.003，让 1000-sample regression eval 低于 $5。

为什么会静默失败：

1. **Judge bias。** Judges 偏好更长答案、来自自己模型家族的答案、匹配 prompt 风格的答案。
2. **JSON parsing failures。** Bad JSON → NaN score → 静默从 aggregate 排除。RAGAS 用户熟悉这个痛。用 try/except + explicit failure mode 做 gate。
3. **Model versions drift。** 升级 judge 会改变每个 metric。Freeze judge model + version。

**RAG 四件套。**

| Metric | 问题 | Backend |
|--------|----------|---------|
| Faithfulness | 答案中的每条 claim 是否来自 retrieved context？ | NLI-based entailment |
| Answer relevance | 答案是否回应问题？ | 从 answer 生成 hypothetical questions；和真实 question 比较 |
| Context precision | Retrieved chunks 中有多少比例相关？ | LLM-judge |
| Context recall | Retrieval 是否返回了所有需要的信息？ | LLM-judge against gold answer |

**G-Eval。** 定义一个 custom criterion：“Did the answer cite the correct source?” Framework 自动扩展为 chain-of-thought evaluation steps，再给 0-1 分。适合 RAGAS 没覆盖的 domain-specific quality dimensions。

**Calibration。** 在没有和 human labels 的 correlation 前，绝不要相信 raw judge score。运行 100 个手工标注样本。画 judge vs human。计算 Spearman rho。如果 rho < 0.7，你的 judge rubric 需要改。

## 构建它

### 第 1 步：用 NLI 做 faithfulness（RAGAS-style）

```python
from typing import Callable
from transformers import pipeline

nli = pipeline("text-classification",
               model="MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
               top_k=None)

# `llm` is any callable: prompt str -> generated str.
# Example: llm = lambda p: client.messages.create(model="claude-haiku-4-5", ...).content[0].text
LLM = Callable[[str], str]


def atomic_claims(answer: str, llm: LLM) -> list[str]:
    prompt = f"""Break this answer into simple factual claims (one per line):
{answer}
"""
    return llm(prompt).splitlines()


def faithfulness(answer: str, context: str, llm: LLM) -> float:
    claims = atomic_claims(answer, llm)
    if not claims:
        return 0.0
    supported = 0
    for claim in claims:
        result = nli({"text": context, "text_pair": claim})[0]
        entail = next((s for s in result if s["label"] == "entailment"), None)
        if entail and entail["score"] > 0.5:
            supported += 1
    return supported / len(claims)
```

把 answer 拆成 atomic claims。用 NLI 逐条检查 claim 是否被 retrieved context 支持。Faithfulness = 被支持的比例。

### 第 2 步：answer relevance

```python
import numpy as np
from sentence_transformers import SentenceTransformer

# encoder: any model implementing .encode(texts, normalize_embeddings=True) -> ndarray
# e.g., encoder = SentenceTransformer("BAAI/bge-small-en-v1.5")

def answer_relevance(question: str, answer: str, encoder, llm: LLM, n: int = 3) -> float:
    prompt = f"Write {n} questions this answer could be the answer to:\n{answer}"
    generated = [line for line in llm(prompt).splitlines() if line.strip()][:n]
    if not generated:
        return 0.0
    q_emb = np.asarray(encoder.encode([question], normalize_embeddings=True)[0])
    g_embs = np.asarray(encoder.encode(generated, normalize_embeddings=True))
    sims = [float(q_emb @ g_emb) for g_emb in g_embs]
    return sum(sims) / len(sims)
```

如果答案暗示的是和原问题不同的问题，relevance 会下降。

### 第 3 步：G-Eval custom metric

```python
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams, LLMTestCase

metric = GEval(
    name="Correctness",
    criteria="The answer should be factually accurate and match the expected output.",
    evaluation_steps=[
        "Read the expected output.",
        "Read the actual output.",
        "List factual claims in the actual output.",
        "For each claim, mark supported or unsupported by the expected output.",
        "Return score = fraction supported.",
    ],
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
)

test = LLMTestCase(input="When was the first iPhone released?",
                   actual_output="June 29th, 2007.",
                   expected_output="June 29, 2007.")
metric.measure(test)
print(metric.score, metric.reason)
```

Evaluation steps 就是 rubric。显式步骤比隐式 “score 0-1” prompts 更稳定。

### 第 4 步：CI gate

```python
import deepeval
from deepeval.metrics import FaithfulnessMetric, ContextualRelevancyMetric


def test_rag_system():
    cases = load_regression_cases()
    faith = FaithfulnessMetric(threshold=0.85)
    rel = ContextualRelevancyMetric(threshold=0.7)
    for case in cases:
        faith.measure(case)
        assert faith.score >= 0.85, f"faithfulness regression on {case.id}"
        rel.measure(case)
        assert rel.score >= 0.7, f"relevancy regression on {case.id}"
```

作为 pytest 文件交付。每个 PR 运行。Regression 阻塞 merge。

### 第 5 步：从零做 toy eval

见 `code/main.py`。Faithfulness（answer claims 和 context 的 overlap）与 relevance（answer tokens 和 question tokens 的 overlap）的 stdlib-only 近似。不是生产级，但展示了形状。

## 坑

- **No calibration。** 与 human labels 相关性 0.3 的 judge 只是噪声。交付前要求 calibration run。
- **Self-evaluation。** 用同一个 LLM 生成和评分会把分数抬高 10-20%。Judge 使用不同 model family。
- **Pairwise judging 的 positional bias。** Judges 偏好第一个选项。始终随机顺序并双向运行。
- **Raw aggregate hides failures。** Mean score 0.85 往往隐藏 5% catastrophic failures。始终检查 bottom quantile。
- **Golden dataset rot。** 未 version 的 eval sets 随时间漂移会破坏纵向比较。每次变更都给 dataset 打 tag。
- **LLM cost。** 大规模时 judge calls 主导成本。使用满足 calibration threshold 的最便宜模型：GPT-4o-mini、Claude Haiku、Mistral-small。

## 使用它

2026 年技术栈：

| 用例 | Framework |
|---------|-----------|
| RAG quality monitoring | RAGAS（4 metrics） |
| CI/CD regression gates | DeepEval + pytest |
| Custom domain criteria | DeepEval 内的 G-Eval |
| Online live-traffic monitoring | RAGAS reference-free mode |
| Human-in-the-loop spot checks | LangSmith 或 Phoenix + annotation UI |
| Red-teaming / safety eval | Promptfoo + DeepEval |

典型 stack：RAGAS 做 monitoring，DeepEval 做 CI，G-Eval 做新维度。三者都运行；它们的分歧很有用。

## 交付它

保存为 `outputs/skill-eval-architect.md`：

```markdown
---
name: eval-architect
description: 设计带 calibrated judge 和 CI gates 的 LLM evaluation plan。
version: 1.0.0
phase: 5
lesson: 27
tags: [nlp, evaluation, rag]
---

给定 use case（RAG / agent / generative task），输出：

1. Metrics。Faithfulness / relevance / context-precision / context-recall + 自定义 G-Eval metrics 与 criteria。
2. Judge model。命名 model + version，说明 cost vs accuracy 的理由。
3. Calibration。Hand-labeled set size，目标 Spearman rho vs human > 0.7。
4. Dataset versioning。Tag strategy、change log、stratification。
5. CI gate。每个 metric 的 thresholds、regression-window logic、bottom-quantile alert。

拒绝依赖没有在 ≥50 个人工标注样本上测试过的 judge。拒绝 self-evaluation（同一模型生成 + 评分）。拒绝只报告 aggregate 且不 surfaced bottom-10%。标记 judge upgrade 没有 parallel baseline eval 的 pipeline。
```

## 练习

1. **简单。** 在 10 个带已知 hallucinations 的 RAG examples 上使用 RAGAS。验证 faithfulness metric 能抓到每一个。
2. **中等。** 手工给 50 个 QA answers 按 correctness 标 0-1。用 G-Eval 打分。测量 judge 与 human 的 Spearman rho。
3. **困难。** 用 DeepEval 构建 pytest CI gate。故意让 retriever regression。验证 gate 失败。通过最低 10% threshold check 增加 bottom-quantile alerting。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|-----------------|-----------------------|
| LLM-as-judge | 用 LLM 评分 | Prompt judge model 根据 rubric 给 outputs 打 0-1 分。 |
| RAGAS | RAG metric library | 开源 eval framework，提供 4 个 reference-free RAG metrics。 |
| Faithfulness | 答案是否 grounded？ | 答案 claims 中被 retrieved context 蕴含的比例。 |
| Context precision | Retrieved chunks 相关吗？ | Top-K chunks 中真正有用的比例。 |
| Context recall | Retrieval 找全了吗？ | Gold-answer claims 中被 retrieved chunks 支持的比例。 |
| G-Eval | Custom LLM judge | Rubric + chain-of-thought eval steps + 0-1 score。 |
| Calibration | 信任但验证 | Judge score 与 human score 的 Spearman correlation。 |

## 延伸阅读

- [Es et al. (2023). RAGAS: Automated Evaluation of Retrieval Augmented Generation](https://arxiv.org/abs/2309.15217) — RAGAS 论文。
- [Liu et al. (2023). G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment](https://arxiv.org/abs/2303.16634) — G-Eval 论文。
- [DeepEval docs](https://deepeval.com/docs/metrics-introduction) — open production stack。
- [Zheng et al. (2023). Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena](https://arxiv.org/abs/2306.05685) — biases、calibration、limits。
- [MLflow GenAI Scorer](https://mlflow.org/blog/third-party-scorers) — 集成 RAGAS、DeepEval、Phoenix 的统一 framework。
