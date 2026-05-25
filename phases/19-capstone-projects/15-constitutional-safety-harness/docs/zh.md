# Capstone 15 — Constitutional Safety Harness + Red-Team Range

> Anthropic 的 Constitutional Classifiers、Meta 的 Llama Guard 4、Google 的 ShieldGemma-2、NVIDIA 的 Nemotron 3 Content Safety，以及支持 multilingual coverage 的 X-Guard，定义了 2026 年的 safety-classifier stack。garak、PyRIT、NVIDIA Aegis 和 promptfoo 成为标准 adversarial evaluation tools。NeMo Guardrails v0.12 把它们连进 production pipeline。这个 capstone 会把所有东西串起来：围绕 target app 的 layered safety harness、运行 6+ attack families 的 autonomous red-team agent，以及一次 constitutional self-critique run，产出可衡量的 harmlessness delta。

**类型：** Capstone
**语言：** Python（safety pipeline、red team）、YAML（policy configs）
**前置要求：** 阶段 10（LLMs from scratch）、阶段 11（LLM engineering）、阶段 13（tools）、阶段 14（agents）、阶段 18（ethics, safety, alignment）
**覆盖阶段：** P10 · P11 · P13 · P14 · P18
**时间：** 25 小时

## 问题

2026 年 LLM safety 的前沿不再是 classifiers 是否有效（大体有效），而是如何围绕 production app 正确组合它们，既不过度拒绝，也不留下明显漏洞。Llama Guard 4 处理 English policy violations。X-Guard（132 种语言）处理 multilingual jailbreak。ShieldGemma-2 捕获 image-based prompt injection。NVIDIA Nemotron 3 Content Safety 覆盖 enterprise categories。Anthropic 的 Constitutional Classifiers 是另一种思路，用在 training 而不是 serving 阶段。

攻击演化也很重要。PAIR 和 TAP 会自动发现 jailbreak。GCG 运行 gradient-based suffix attacks。Multi-turn 和 code-switch attacks 利用 agent memory。任何已部署 LLM 都需要一个 red-team range：garak 和 PyRIT 是 canonical drivers；还要有记录好的 mitigations 和 CVSS-scored findings。

你将加固一个 target application（一个 8B instruction-tuned model，或其他 capstone 中的一个 RAG chatbot），对它运行 6+ attack families，并产出 before/after harmlessness measurement。

## 概念

safety pipeline 有五层。**Input sanitize**：移除 zero-width chars，解码 base64/rot13，规范化 Unicode。**Policy layer**：NeMo Guardrails v0.12 rails（off-domain、toxicity、PII extraction）。**Classifier gate**：input 上的 Llama Guard 4、非英语上的 X-Guard、image inputs 上的 ShieldGemma-2。**Model**：target LLM。**Output filter**：output 上的 Llama Guard 4、Presidio PII scrub，以及适用时的 citation enforcement。**HITL tier**：high-risk outputs 进入 Slack queue。

red-team range 在 scheduler 上运行。PAIR 和 TAP 自动发现 jailbreaks。GCG 运行 gradient-based suffix attacks。ASCII / base64 / rot13 encoding attacks。Multi-turn attacks（persona adoption、memory exploitation）。Code-switch attacks（混合英语与斯瓦希里语或泰语）。每次 run 都产出 structured findings file，包含 CVSS scoring 和 disclosure timeline。

constitutional-self-critique run 是 training-time intervention。取 1k harmful-attempt prompts，让模型草拟 response，再按照书面 constitution（do-not-harm rules）批评它，并基于 critique loop 重新训练。用 held-out eval 衡量 before/after harmlessness delta。

## 架构

```
request (text / image / multilingual)
      |
      v
input sanitize (strip zero-width, decode, normalize)
      |
      v
NeMo Guardrails v0.12 rails (off-domain, policy)
      |
      v
classifier gate:
  Llama Guard 4 (English)
  X-Guard (multilingual, 132 langs)
  ShieldGemma-2 (image prompts)
  Nemotron 3 Content Safety (enterprise)
      |
      v (allowed)
target LLM
      |
      v
output filter: Llama Guard 4 + Presidio PII + citation check
      |
      v
HITL tier for flagged outputs

parallel:
  red-team scheduler
    -> garak (classic attacks)
    -> PyRIT (orchestrated red team)
    -> autonomous jailbreak agent (PAIR + TAP)
    -> GCG suffix attacks
    -> multilingual / code-switch
    -> multi-turn persona adoption

output: CVSS-scored findings + disclosure timeline + before/after harmlessness delta
```

## 技术栈

- Safety classifiers：Llama Guard 4、ShieldGemma-2、NVIDIA Nemotron 3 Content Safety、X-Guard
- Guardrail framework：NeMo Guardrails v0.12 + OPA
- Red-team drivers：garak（NVIDIA）、PyRIT（Microsoft Azure）、NVIDIA Aegis、promptfoo
- Jailbreak agents：PAIR（Chao et al., 2023）、Tree-of-Attacks（TAP）、GCG suffix
- Constitutional training：Anthropic-style self-critique loop + 基于 critiques 的 SFT
- PII scrub：Presidio
- Target：一个 8B instruction-tuned model，或其他 capstones 的 RAG chatbots

## 构建它

1. **Target setup。** 在 vLLM 上启动一个 8B instruction-tuned model（或复用另一个 capstone 的 RAG chatbot）。这是待测 app。

2. **Safety pipeline wrap。** 围绕 target 接入五层 pipeline。验证每一层都可独立观察（Langfuse 中每层一个 span）。

3. **Classifier coverage。** 加载 Llama Guard 4、X-Guard（multilingual）、ShieldGemma-2（image）。在小型 labeled set 上运行每个 classifier，建立 baseline。

4. **Red-team scheduler。** 调度 garak、PyRIT、PAIR agent、TAP agent、GCG runner、multi-turn attacker 和 code-switch attacker。每个都在独立 queue 上运行。

5. **Attack suite。** 六类攻击：(1) PAIR automated jailbreak，(2) TAP tree-of-attacks，(3) GCG gradient suffix，(4) ASCII / base64 / rot13 encoding，(5) multi-turn persona，(6) multilingual code-switch。报告每个 family 的 success rate。

6. **Constitutional self-critique。** 整理 1k harmful-attempt prompts。对每个 prompt，target 草拟 response。critic LLM 按书面 constitution（“do no harm”、“cite evidence”、“refuse illegal requests”）评分。critic 反对的 prompts 会被重写；target 在 critique-improved pairs 上 fine-tune。用 held-out eval 衡量 before/after harmlessness。

7. **Over-refusal measurement。** 在 benign prompt suite（例如 XSTest）上跟踪 false-positive rate。target 必须在 benign questions 上保持 helpful。

8. **CVSS scoring。** 对每个成功 jailbreak，按 CVSS 4.0（attack vector、complexity、impact）打分。产出 disclosure timeline 和 mitigation plan。

9. **Range automation。** 以上全部跑在 cron 上；findings 写入 queue；over-refusal regression alerts 发到 Slack。

## 使用它

```
$ safety probe --model=target --family=PAIR --budget=50
[attacker]   PAIR agent running on target
[attack]     attempt 1/50: disguise query as academic research ... blocked
[attack]     attempt 2/50: appeal to roleplay ... blocked
[attack]     attempt 3/50: chain-of-thought coax ... SUCCEEDED
[finding]    CVSS 4.8 medium: roleplay bypass on target
[range]      7 successes out of 50 (14% success rate)
```

## 交付它

`outputs/skill-safety-harness.md` 是交付物。一个 production-grade layered safety pipeline，加一个可复现的 red-team range，并带 before/after harmlessness deltas。

| 权重 | 标准 | 衡量方式 |
|:-:|---|---|
| 25 | Attack-surface coverage | 演练 6+ attack families、2+ languages |
| 20 | True-positive / false-positive trade-off | Attack block rate vs XSTest benign pass rate |
| 20 | Self-critique delta | held-out eval 上的 before/after harmlessness |
| 20 | Documentation and disclosure | 带 timeline 的 CVSS-scored findings |
| 15 | Automation and repeatability | 所有内容跑在 cron 上并带 alerts |
| **100** | | |

## 练习

1. 在 RAG chatbot 上运行 garak 的 prompt-injection plugin，比较有无 output-filter layer 时的 attack success rate。

2. 添加第七个 attack family：通过 retrieved documents 进行 indirect prompt injection。衡量需要额外防御的部分。

3. 实现 “refuse-with-help” mode：guardrail 阻止时，target 给出更安全的相关回答，而不是 flat refusal。衡量 XSTest delta。

4. Multilingual coverage gap：找一种 X-Guard 表现不足的语言。提出一个针对它的 fine-tune dataset。

5. 在 30B model 上运行 constitutional self-critique，并衡量 delta 是否随规模增长。

## 关键词汇

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|------------------------|
| Layered safety | “Defense in depth” | input、gate、output、HITL 上的多重 guardrails |
| Llama Guard 4 | “Meta's safety classifier” | 2026 年 reference input/output content classifier |
| PAIR | “Jailbreak agent” | Chao 等人的论文，关于 LLM-driven jailbreak discovery |
| TAP | “Tree-of-Attacks” | PAIR 的 tree-search 变体 |
| GCG | “Greedy coordinate gradient” | Gradient-based adversarial suffix attack |
| Constitutional self-critique | “Anthropic-style training” | target drafts -> critic scores -> rewrite -> retrain |
| XSTest | “Benign probe set” | over-refusal regression benchmark |
| CVSS 4.0 | “Severity score” | safety findings 的标准 vulnerability scoring |

## 延伸阅读

- [Anthropic Constitutional Classifiers](https://www.anthropic.com/research/constitutional-classifiers) — training-time reference
- [Meta Llama Guard 4](https://ai.meta.com/research/publications/llama-guard-4/) — 2026 input/output classifier
- [Google ShieldGemma-2](https://huggingface.co/google/shieldgemma-2b) — image + multimodal safety
- [NVIDIA Nemotron 3 Content Safety](https://developer.nvidia.com/blog/building-nvidia-nemotron-3-agents-for-reasoning-multimodal-rag-voice-and-safety/) — enterprise reference
- [X-Guard (arXiv:2504.08848)](https://arxiv.org/abs/2504.08848) — 132-language multilingual safety
- [garak](https://github.com/NVIDIA/garak) — NVIDIA red-team toolkit
- [PyRIT](https://github.com/Azure/PyRIT) — Microsoft red-team framework
- [NeMo Guardrails v0.12](https://docs.nvidia.com/nemo-guardrails/) — rail framework
- [PAIR (arXiv:2310.08419)](https://arxiv.org/abs/2310.08419) — jailbreak agent paper
