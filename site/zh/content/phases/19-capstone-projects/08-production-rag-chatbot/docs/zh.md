# Capstone 08 — 面向受监管垂直领域的 Production RAG Chatbot

> Harvey、Glean、Mendable 和 LlamaCloud 在 2026 年都运行着同一种生产形态。用 docling 或 Unstructured 摄入文档，用 ColPali 处理视觉内容。Hybrid search。用 bge-reranker-v2-gemma re-rank。用 Claude Sonnet 4.7 和 60-80% hit rate 的 prompt caching synthesize。用 Llama Guard 4 和 NeMo Guardrails 防护。用 Langfuse 和 Phoenix 观察。用 200-question golden set 上的 RAGAS 打分。在受监管 domain（legal、clinical、insurance）构建一个这样的系统，这个 capstone 的目标就是通过 golden set、red team 和 drift dashboard。

**类型：** Capstone
**语言：** Python（pipeline + API）、TypeScript（chat UI）
**前置要求：** 阶段 5（NLP）、阶段 7（transformers）、阶段 11（LLM engineering）、阶段 12（multimodal）、阶段 17（infrastructure）、阶段 18（safety）
**覆盖阶段：** P5 · P7 · P11 · P12 · P17 · P18
**时间：** 30 小时

## 问题

受监管领域 RAG（legal contracts、clinical trial protocols、insurance policies）是 2026 年出货最多的生产形态，因为 ROI 明确，风险也具体。Harvey（Allen & Overy）为 legal 构建了它。Mendable 交付 developer-docs 版本。Glean 覆盖 enterprise search。模式是：high-fidelity ingestion、hybrid retrieval with rerank、带 citation enforcement 和 prompt caching 的 synthesis、多层 safety guard，以及持续 drift monitoring。

难点不在模型，而在 jurisdiction-aware compliance（HIPAA、GDPR、SOC2）、citation-level auditability、cost control（prompt caching 在 hit rate 高时能带来 60-90% 折扣）、通过 RAGAS faithfulness 做 hallucination detection，以及当 source documents 更新但 index 没跟上时的 drift detection。这个 capstone 要求你在 200-question golden set 和 red-team suite 上一起通过。

## 概念

pipeline 有两侧。**Ingestion**：docling 或 Unstructured 解析结构化文档；ColPali 处理视觉丰富的文档；chunks 会获得 summaries、tags 和 role-based access labels。vectors 进入 pgvector + pgvectorscale（低于 5000 万 vectors）或 Qdrant Cloud；sparse BM25 并行运行。**Conversation**：LangGraph 处理 memory 和 multi-turn；每个 query 运行 hybrid retrieval，用 bge-reranker-v2-gemma-2b rerank，用 Claude Sonnet 4.7（prompt-cached）synthesize，再通过 Llama Guard 4 和 NeMo Guardrails，最终发出 citation-anchored response。

eval stack 有四层。**Golden set**（200 个带 citations 的标注 Q/A）用于 correctness。**Red team**（jailbreaks、PII extraction attempts、off-domain questions）用于 safety。**RAGAS** 自动按 turn 衡量 faithfulness / answer relevance / context precision。**Drift dashboard**（Arize Phoenix）每周观察 retrieval quality 和 hallucination score。

Prompt caching 是成本杠杆。Claude 4.5+ 和 GPT-5+ 支持缓存 system prompts + retrieved context。当 hit rate 达到 60-80% 时，每次 query 成本会下降 3-5x。pipeline 必须为 stable prefixes（system prompt + reranked context first）而设计，才能获得高 cache hit rate。

## 架构

```
documents (contracts, protocols, policies)
      |
      v
docling / Unstructured parse + ColPali for visuals
      |
      v
chunks + summaries + role-labels + jurisdiction tags
      |
      v
pgvector + pgvectorscale  +  BM25 (Tantivy)
      |
query + role + jurisdiction
      |
      v
LangGraph conversational agent
   +--- retrieve (hybrid)
   +--- filter by role + jurisdiction
   +--- rerank (bge-reranker-v2-gemma-2b or Voyage rerank-2)
   +--- synthesize (Claude Sonnet 4.7, prompt cached)
   +--- guard (Llama Guard 4 + NeMo Guardrails + Presidio output PII scrub)
   +--- cite + return
      |
      v
eval:
  RAGAS faithfulness / answer_relevance / context_precision (online)
  Langfuse annotation queue (sampled)
  Arize Phoenix drift (weekly)
  red team suite (pre-release)
```

## 技术栈

- Ingestion：Unstructured.io 或 docling 处理结构化文档；ColPali 处理 visually-rich PDFs
- Vector DB：5000 万 vectors 以下使用 pgvector + pgvectorscale；否则使用 Qdrant Cloud
- Sparse：带 field weights 的 Tantivy BM25
- Orchestration：LlamaIndex Workflows（ingestion）+ LangGraph（conversation）
- Re-ranker：自托管 bge-reranker-v2-gemma-2b 或 hosted Voyage rerank-2
- LLM：带 prompt caching 的 Claude Sonnet 4.7；fallback 为自托管 Llama 3.3 70B
- Eval：RAGAS 0.2 online，DeepEval 用于 hallucination 和 jailbreak suites
- Observability：自托管 Langfuse，带 annotation queue；Arize Phoenix 用于 drift
- Guardrails：Llama Guard 4 input/output classifier、NeMo Guardrails v0.12 policy、Presidio PII scrub
- Compliance：chunks 上的 role-based access labels；GDPR/HIPAA 的 jurisdiction tags

## 构建它

1. **Ingestion。** 用 Unstructured 或 docling 解析你的 corpus（严肃构建应有 1000-10000 documents）。对于 scanned / visual-heavy pages，route 到 ColPali。生成带 summaries、role-labels、jurisdiction tags 的 chunks。

2. **Index。** Dense embeddings（Voyage-3 或 Nomic-embed-v2）写入 pgvector + pgvectorscale。通过 Tantivy 建 BM25 side-index。role 和 jurisdiction filters 作为 payload。

3. **Hybrid retrieve。** 先按 role+jurisdiction filter；再并行 dense + BM25；用 reciprocal rank fusion 合并；top-20 进入 reranker；top-5 进入 synth。

4. **Synthesize with prompt caching。** System prompt + static policies 放在 cache header；reranked context 作为 cache extension；user question 作为 uncached suffix。稳态目标 60-80% cache hit rate。

5. **Guardrails。** input 先过 Llama Guard 4；NeMo Guardrails rails 阻止 off-domain questions 或 policy-forbidden topics；Presidio 清理输出中意外出现的 PII；citation enforcement post-filter。

6. **Golden set。** 由 domain expert 标注 200 个 Q/A pairs，包含 (answer, citations)。按 exact-citation match、answer correctness、faithfulness（RAGAS）给 agent 打分。

7. **Red team。** 50 个 adversarial prompts：jailbreaks（PAIR、TAP）、PII exfiltration attempts、off-domain、cross-jurisdiction leaks。按 pass/fail 和 severity 打分。

8. **Drift dashboard。** Arize Phoenix 每周跟踪 retrieval quality（nDCG、citation faithfulness）。下降 5% 时告警。

9. **Cost report。** Langfuse：prompt-caching hit rate、tokens per query、每个 stage 的 $/query breakdown。

## 使用它

```
$ chat --role=analyst --jurisdiction=GDPR
> what is the data-retention obligation for EU user profiles under our contract?
[retrieve]  hybrid top-20 filtered to GDPR + analyst-role
[rerank]    top-5 kept
[synth]     claude-sonnet-4.7, cache hit 74%, 0.8s
answer:
  The contract (Section 12.4, Master Services Agreement dated 2024-03-11)
  obligates EU user profile deletion within 30 days of termination per GDPR
  Article 17. The DPA amendment (DPA-v2.1, Section 5) extends this to 14 days
  for "restricted" category data.
  citations: [MSA-2024-03-11 s12.4, DPA-v2.1 s5]
```

## 交付它

`outputs/skill-production-rag.md` 描述交付物：一个部署好的 regulated-domain chatbot，带 compliance labels，通过 rubric，并由 live drift monitoring 观察。

| 权重 | 标准 | 衡量方式 |
|:-:|---|---|
| 25 | RAGAS faithfulness + answer relevance | golden set（200 Q/A）上的 online scores |
| 20 | Citation correctness | 带可验证 source anchors 的答案比例 |
| 20 | Guardrail coverage | Llama Guard 4 pass rate + jailbreak suite results |
| 20 | Cost / latency engineering | Prompt-cache hit rate、p95 latency、$/query |
| 15 | Drift monitoring dashboard | 带 weekly retrieval-quality trend 的 Phoenix live dashboard |
| **100** | | |

## 练习

1. 在不同 jurisdiction 下构建第二个 corpus slice（例如 HIPAA alongside GDPR）。用 20-question cross-jurisdiction probe 演示 role+jurisdiction filtering 阻止 cross-leak。

2. 衡量一周 production traffic 的 prompt-cache hit rate。识别哪些 query 破坏 cache prefix，并重构。

3. 添加带 10k-token summary buffer 的 multi-turn memory。衡量 conversation 变长时 faithfulness 是否下降。

4. 把 Claude Sonnet 4.7 换成自托管 Llama 3.3 70B。衡量 $/query 和 faithfulness delta。

5. 添加 “unsure” mode：如果 top reranked scores 低于阈值，agent 说 “I do not have confident citations”，而不是回答。衡量 false-confidence reduction。

## 关键词汇

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|------------------------|
| Prompt caching | “Cached system + context” | Claude/OpenAI 功能：命中时 cached prefix tokens 折扣 60-90% |
| RAGAS | “RAG evaluator” | 自动评分 faithfulness、answer relevance、context precision |
| Golden set | “Labeled eval” | 200+ 个专家标注且带 citations 的 Q/A；ground truth |
| Jurisdiction tag | “Compliance label” | 绑定到 chunks 的 GDPR/HIPAA/SOC2 scope；由 retrieval filter 强制执行 |
| Citation faithfulness | “Grounded answer rate” | 由可检索 source spans 支撑的 claims 比例 |
| Drift | “Retrieval quality decay” | nDCG 或 citation score 的每周变化；alert threshold 5% |
| Red team | “Adversarial eval” | 发布前的 jailbreak、PII extraction、off-domain probes |

## 延伸阅读

- [Harvey AI](https://www.harvey.ai) — reference legal production stack
- [Glean enterprise search](https://www.glean.com) — enterprise scale RAG 参考
- [Mendable documentation](https://mendable.ai) — developer-docs RAG reference
- [LlamaCloud Parse + Index](https://docs.llamaindex.ai/en/stable/examples/llama_cloud/llama_parse/) — managed ingestion
- [Anthropic prompt caching](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching) — cost-lever reference
- [RAGAS 0.2 documentation](https://docs.ragas.io/) — canonical RAG eval framework
- [Arize Phoenix](https://github.com/Arize-ai/phoenix) — reference drift observability
- [Llama Guard 4](https://ai.meta.com/research/publications/llama-guard-4/) — 2026 safety classifier
- [NeMo Guardrails v0.12](https://docs.nvidia.com/nemo-guardrails/) — policy rail framework
