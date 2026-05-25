# Capstone 07 — End-to-End Fine-Tuning Pipeline（Data 到 SFT 到 DPO 到 Serve）

> 一个用你自己的数据训练的 8B 模型，用你自己的偏好做 DPO alignment，经过 quantization、speculative decoding，并以可衡量的 $/1M tokens 服务。2026 年的 open stack 是 Axolotl v0.8、TRL 0.15、用于快速迭代的 Unsloth、用于 quantization 的 GPTQ/AWQ/GGUF、以及带 EAGLE-3 serving 的 vLLM 0.7。这个 capstone 要求你可复现地跑完整条 pipeline：YAML 输入，served endpoint 输出，并按照 2026 Model Openness Framework 发布 model card。

**类型：** Capstone
**语言：** Python（pipeline）、YAML（configs）、Bash（scripts）
**前置要求：** 阶段 2（ML）、阶段 3（DL）、阶段 7（transformers）、阶段 10（LLMs from scratch）、阶段 11（LLM engineering）、阶段 17（infrastructure）、阶段 18（safety）
**覆盖阶段：** P2 · P3 · P7 · P10 · P11 · P17 · P18
**时间：** 35 小时

## 问题

到 2026 年，每个严肃 AI 团队手边都会保留一条 fine-tuning pipeline。不是因为他们会发布 frontier base model，而是因为 downstream adaptation 里才有可衡量收益：domain SFT、针对 labeled preferences 的 DPO、用于 speculative decoding 的 distilled drafts、通过 EAGLE-3 serving。Axolotl v0.8 处理 multi-GPU SFT configs。TRL 0.15 处理 DPO 和 GRPO。Unsloth 支持快速 single-GPU iteration。带 EAGLE-3 的 vLLM 0.7 在不损失质量的情况下把 decode throughput 推高 2-3x。工具已经可用；手艺在 YAML、data hygiene 和 eval discipline 中。

你将把一个 8B base（Llama 3.3、Qwen3 或 Gemma 3）先用 task-specific data 做 SFT，再做 DPO，量化以便 serving，并用 lm-evaluation-harness、RewardBench-2、MT-Bench-v2 和 MMLU-Pro 衡量增益。你还将按照 2026 Model Openness Framework 产出 model card。重点是 reproducibility：一条命令 end-to-end 重新运行整条 pipeline。

## 概念

pipeline 有五个阶段。**Data**：dedup（MinHash / Datatrove）、quality filter（Nemotron-CC 风格 classifier）、PII scrub、对 public benchmark contamination 的 split-hygiene check。**SFT**：Axolotl YAML、8xH100 上的 ZeRO-3、cosine schedule、packed sequences、2-3 epochs。**DPO or GRPO**：TRL config、1 epoch、人工标注或模型评审得到的 preference pairs、beta tuning。**Quantize**：GPTQ + AWQ + GGUF，方便不同 deployment。**Serve**：带 EAGLE-3 speculative heads 的 vLLM 0.7（或带 SpecForge 的 SGLang）、K8s deployment、按 queue-wait 做 HPA。

Ablations 是交付物：在三个 task-specific benchmarks 上比较 SFT-only vs SFT+DPO vs SFT+GRPO。Serving metrics：batch 1 / 8 / 32 下的 tokens/s、EAGLE-3 acceptance rate、$/1M tokens。Safety eval：Llama Guard 4 pass rate。Model card：bias evaluations、reproducibility seeds、data licensing。

## 架构

```
raw data (HF datasets + internal)
    |
    v
Datatrove dedup + Nemotron-CC quality filter + PII scrub
    |
    v
split hygiene (MMLU-Pro contamination check)
    |
    v
Axolotl SFT config (YAML)  ---> 8xH100, ZeRO-3
    |
    v
TRL DPO / GRPO config       ---> 4xH100, 1 epoch
    |
    v
GPTQ + AWQ + GGUF quantize
    |
    v
vLLM 0.7 + EAGLE-3 speculative decoding
    |
    v
K8s deployment, HPA on queue-wait
    |
    v
lm-eval-harness + RewardBench-2 + MT-Bench-v2 + MMLU-Pro
    |
    v
model card (2026 MOF) + safety eval (Llama Guard 4)
```

## 技术栈

- Data：Datatrove 用于 dedup，Nemotron-CC classifier 用于 quality，Presidio 用于 PII
- Base：Llama 3.3 8B、Qwen3 14B 或 Gemma 3 12B
- SFT：Axolotl v0.8，带 ZeRO-3、Flash Attention 3、packed sequences
- Preference tuning：TRL 0.15 用于 DPO 或 GRPO；Unsloth 用于 single-GPU iteration
- Quantization：GPTQ（Marlin）、AWQ、通过 llama.cpp 生成 GGUF
- Serving：带 EAGLE-3 speculative decoding 的 vLLM 0.7（或 SGLang 0.4 + SpecForge）
- Eval：lm-evaluation-harness、RewardBench-2、MT-Bench-v2、MMLU-Pro
- Safety eval：Llama Guard 4、ShieldGemma-2
- Infrastructure：Kubernetes + NVIDIA device plugin，基于 queue-wait metric 的 HPA
- Observability：W&B 用于 training，Langfuse 用于 inference

## 构建它

1. **Data pipeline。** 在 raw corpus 上运行 Datatrove dedup。应用 Nemotron-CC-style quality classifier。Presidio 清理 PII。用明确 seed 写入 train/val splits。

2. **Contamination check。** 对每个 validation split，针对 MMLU-Pro、MT-Bench-v2、RewardBench-2 test sets 计算 MinHash。拒绝任何 overlap。

3. **Axolotl SFT。** YAML 配置 ZeRO-3、FA3、sequence packing。在 8xH100 上训练 2-3 epochs。记录到 W&B。

4. **TRL DPO / GRPO。** 取 SFT checkpoint，在 preference pairs 上运行一轮 DPO（或在 math/code 上用 verifiable reward 运行 GRPO）。扫 beta。

5. **Quantize。** 产出三种 quant：GPTQ-INT4-Marlin、AWQ-INT4、给 llama.cpp 的 GGUF-Q4_K_M。记录 size 和 nominal throughput。

6. **Serve with speculative decoding。** vLLM 0.7 config，使用通过 Red Hat Speculators 训练的 EAGLE-3 draft heads。衡量 batch 1 / 8 / 32 下的 acceptance rate 和 tail latency。报告同一 eval 上相对 Anthropic / OpenAI 的 $/1M tokens。

7. **Eval matrix。** 在 base、SFT-only、SFT+DPO、SFT+GRPO 上运行 lm-eval-harness、RewardBench-2、MT-Bench-v2、MMLU-Pro。产出表格。

8. **Safety eval。** 在 dev set 上衡量 Llama Guard 4 pass rate。使用 ShieldGemma-2 output filter。

9. **Model card。** MOF 2026 template：data、training、eval、safety、license，以及带 YAML 和 commit SHAs 的 reproducibility section。

## 使用它

```
$ ./pipeline.sh config/llama3.3-8b-domainX.yaml
[data]    300k deduped, 12k filtered, 280k accepted (seed=7)
[SFT]     3 epochs, 8xH100, 6h12m, val loss 1.42 -> 1.03
[DPO]     1 epoch, beta=0.08, 4xH100, 1h40m
[quant]   GPTQ-INT4 4.6 GB, AWQ-INT4 4.8 GB, GGUF-Q4_K_M 5.1 GB
[serve]   vLLM 0.7, EAGLE-3 acceptance 0.74, p99 126ms @ bs=8
[eval]    MMLU-Pro +3.2, MT-Bench-v2 +0.41, RewardBench-2 +0.08
[card]    model-card.md generated under 2026 MOF
```

## 交付它

`outputs/skill-finetuning-pipeline.md` 描述交付物。一条命令把 data 依次跑过 SFT、DPO、quant、serve 和 eval，并产出 model card + served endpoint。

| 权重 | 标准 | 衡量方式 |
|:-:|---|---|
| 25 | Eval delta vs base | target tasks（MMLU-Pro、MT-Bench-v2、task-specific）上的测量增益 |
| 20 | Pipeline reproducibility | 一条命令用相同 seeds end-to-end rerun |
| 20 | Data hygiene | Dedup rate、PII scrub coverage、contamination check green |
| 20 | Serving efficiency | bs=1/8/32 下的 tokens/s、EAGLE-3 acceptance rate、$/1M tokens |
| 15 | Model card + safety eval | 2026 MOF completeness + Llama Guard 4 pass rate |
| **100** | | |

## 练习

1. 在同一个 task-specific benchmark 上运行 SFT-only vs SFT+DPO vs SFT+GRPO。报告哪种 preference method 胜出，以及幅度。

2. 把 Llama 3.3 8B 换成 Qwen3 14B。在 matched quality 下衡量 $/1M tokens。

3. 衡量 domain data 与 generic ShareGPT 上的 EAGLE-3 acceptance rate。报告 delta 以及它对 latency budgets 的含义。

4. 注入 1% contamination（把 MMLU-Pro answers 泄漏进 training data）并重新 eval。观察 MMLU-Pro accuracy 非真实地跳升。构建一个能抓住它的 contamination-check CI gate。

5. 添加 LoRA SFT 作为 full fine-tune 的替代。以 10x 更低 memory 衡量 quality gap。

## 关键词汇

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|------------------------|
| Axolotl | “SFT trainer” | 由 YAML 驱动的统一 trainer，用于 SFT、DPO 和 distillation |
| TRL | “Preference tuner” | Hugging Face 的 LLM DPO、GRPO、PPO 库 |
| GRPO | “Group-relative policy optimization” | DeepSeek R1 使用 verifiable rewards 的 RL recipe |
| EAGLE-3 | “Speculative decoding draft” | 能预测前方 N 个 tokens 的 draft heads；vLLM 用 target model 验证 |
| MOF | “Model Openness Framework” | 2026 年按 data、code、license 给 model releases 分级的标准 |
| Contamination check | “Split hygiene” | 基于 MinHash 检测 test-set leakage 是否进入 training |
| Acceptance rate | “EAGLE / MTP metric” | drafted tokens 中被 target model 接受的比例 |

## 延伸阅读

- [Axolotl documentation](https://axolotl-ai-cloud.github.io/axolotl/) — reference SFT / DPO trainer
- [TRL documentation](https://huggingface.co/docs/trl) — DPO 和 GRPO reference implementations
- [Unsloth](https://github.com/unslothai/unsloth) — single-GPU iteration reference
- [DeepSeek R1 paper (arXiv:2501.12948)](https://arxiv.org/abs/2501.12948) — GRPO methodology
- [vLLM + EAGLE-3 documentation](https://docs.vllm.ai) — reference serving stack
- [SGLang SpecForge](https://github.com/sgl-project/SpecForge) — alternate speculative-decoding trainer
- [Model Openness Framework 2026](https://isocpp.org/) — open-release grading standard
- [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) — canonical eval runner
