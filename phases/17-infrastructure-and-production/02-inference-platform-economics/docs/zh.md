# Inference 平台经济学 — Fireworks、Together、Baseten、Modal、Replicate、Anyscale

> 2026 年的 inference 市场不再只是租 GPU 时间。它分成三类：custom silicon（Groq、Cerebras、SambaNova）、GPU platforms（Baseten、Together、Fireworks、Modal）和 API-first marketplaces（Replicate、DeepInfra）。Fireworks 在 2026 年 5 月 1 日把每 GPU 小时价格上调 $1，$4B valuation 和每天 10T+ tokens 说明 volume-driven 模型有效。Baseten 在 2026 年 1 月完成 $300M Series E，估值 $5B。竞争定位规则很简单：Fireworks 优化 latency，Together 优化目录广度，Baseten 优化企业级 polish，Modal 优化 Python-native DX，Replicate 优化多模态触达，Anyscale 优化 distributed Python。本课给你一张可以交给 founder 的矩阵。

**类型：** 学习
**语言：** Python（stdlib，玩具版 per-call economics 比较器）
**前置要求：** 阶段 17 · 01（Managed LLM Platforms），阶段 17 · 04（vLLM Serving Internals）
**时间：** ~60 分钟

## 学习目标

- 说出三个市场 segment（custom silicon、GPU platforms、API-first），并把每个 vendor 映射到 segment。
- 解释为什么“per-token”API pricing model 会向 serving engine 的成本曲线压缩，而不是向硬件成本压缩。
- 计算至少三个 vendor 的 effective cost per request，并解释什么时候 per-minute（Baseten、Modal）胜过 per-token。
- 判断给定 workload 的正确默认平台（serverless bursty、steady high-throughput、fine-tuned variants、多模态）。

## 问题

你已经评估过托管 hyperscaler 平台。你决定需要一个更窄、更快的 provider：Fireworks 用于 latency，Together 用于广度，Baseten 用于 fine-tuned custom model。现在你有六个真实选择，而 pricing pages 的单位都对不上。Fireworks 显示 $/M tokens；Baseten 显示 $/minute；Modal 显示 $/second；Replicate 显示 $/prediction。不对 workload 建模，就无法正面对比。

更糟的是，每张 pricing page 背后的商业模型不同。Fireworks 在共享 GPU 上运行自己的 custom engine（FireAttention）；per-token rate 反映它们的 utilization curve。Baseten 给你 Truss + dedicated GPUs；per-minute 反映 exclusivity。Modal 是真正的 Python serverless：per-second billing，sub-second cold starts。同样的输出（一条 LLM response），三种不同的成本函数。

本课会建模这六个平台，并告诉你它们分别什么时候胜出。

## 概念

### 三个 segment

**Custom silicon** — Groq（LPU）、Cerebras（WSE）、SambaNova（RDU）。在同一模型上，decode 通常比 GPU-based cluster 快 5-10x。per-token 价格更高（Groq 在 2025 年末 Llama-70B 约 ~$0.99/M），但对于 latency-sensitive use case 无可匹敌。Groq 是 voice agents 和 real-time translation 的 production 选择。

**GPU platforms** — Baseten、Together、Fireworks、Modal、Anyscale。运行在 NVIDIA（2026 年 H100、H200、B200）上，有时也运行在 AMD 上。它们是“raw GPU rental”（RunPod、Lambda）和“hyperscaler managed service”（Bedrock）之间的经济层。

**API-first marketplaces** — Replicate、DeepInfra、OpenRouter、Fal。目录广泛，pay-per-prediction 或 pay-per-second，强调 time-to-first-call。

### Fireworks — latency-optimized GPU platform

- FireAttention engine（custom）；营销口径是在等价配置上 latency 比 vLLM 低 4x。
- 非交互 workload 的 batch tier 约为 serverless rate 的 50%。
- Fine-tuned model 按 base model 同价服务，这是相对那些对你的 LoRA 加价的 provider 的真实差异点。
- 2026 年中：on-demand GPU rental 从 2026 年 5 月 1 日起上调 $1/hour。大规模 volume pricing 可谈判。
- 财务信号：$4B valuation，每天处理 10T+ tokens。

### Together — breadth-optimized

- 200+ models，包括 open-source releases 在 upstream 发布后几天内上线。
- 在等价 LLM models 上比 Replicate 便宜 50-70%：“AI Native Cloud”定位靠 volume 和 catalog。
- Inference + fine-tuning + training 都在同一个 API 中。

### Baseten — enterprise-polish-optimized

- Truss framework：用一个 manifest 打包 model dependencies、secrets 和 serving config。
- GPU 范围从 T4 到 B200。per-minute billing，并有合理的 cold-start mitigation。
- SOC 2 Type II，HIPAA-ready。常见 fintech 和 healthcare 选择。
- $5B valuation，2026 年 1 月 Series E（CapitalG、IVP、NVIDIA 投入 $300M）。

### Modal — Python-native-optimized

- 纯 Python infrastructure-as-code。给函数加上 `@modal.function(gpu="A100")`，一条命令部署。
- Per-second billing。预热后 cold starts 2-4s；小模型 <1s。
- 2025 年 $87M Series B，估值 $1.1B。在独立调查中 developer experience 得分最强。

### Replicate — multimodal breadth

- Pay-per-prediction。图像、视频和音频模型的默认平台。
- 集成生态（Zapier、Vercel、CMS plugins）。
- 在 LLM per-token rates 上竞争力较弱，但赢在多模态 variety。

### Anyscale — Ray-native

- 基于 Ray；RayTurbo 是 Anyscale 的 proprietary inference engine（与 vLLM 竞争）。
- 最适合 distributed Python workload，其中 inference step 是更大 graph 中的一个节点。
- Managed Ray clusters；与 Ray AIR 和 Ray Serve 紧密集成。

### Per-token vs per-minute — 什么时候各自胜出

Per-token 适合 latency-insensitive 且 bursty 的 workload：只为实际使用付费。Per-minute 适合利用率高且可预测的 workload：一旦 GPU 被打满，就会胜过 per-token。

粗略规则：当 workload 高于 dedicated GPU 约 30% sustained utilization，per-minute（Baseten、Modal）开始胜过 per-token（Fireworks、Together）。低于这个值，per-token 胜出，因为你避免为空闲付费。

### Custom engine 才是真正的 moat

vLLM 和 SGLang 之上的每个平台都声称有 custom engine。FireAttention、RayTurbo、Baseten 的 inference stack。custom-engine claim 带有营销色彩，更诚实的说法是：vLLM + SGLang 占据 production open-source inference 的约 80%，而平台层的差异点是 DX、attribution 和 SLA。

### 你应该记住的数字

- Fireworks GPU rental：2026 年 5 月 1 日起涨价 $1/hr。
- Fireworks claim：等价配置上 latency 比 vLLM 低 4x。
- Together：在 LLM 上比 Replicate 便宜 50-70%。
- Baseten valuation：$5B（Series E，2026 年 1 月，$300M round）。
- Modal valuation：$1.1B（Series B，2025）。
- Per-minute 在高于约 30% sustained utilization 时胜过 per-token。

## 使用它

`code/main.py` 在合成 workload 上跨 pricing models 比较六个 vendor。报告 $/day 和 effective $/M tokens。运行它，找出 per-token 和 per-minute 的 break-even。

## 交付它

本课会产出 `outputs/skill-inference-platform-picker.md`。给定 workload profile、SLA 和 budget，它会选择 primary inference platform 并指出 runner-up。

## 练习

1. 运行 `code/main.py`。对于一台 H100 上的 70B model，Baseten（per-minute）在什么 sustained utilization 下胜过 Fireworks（per-token）？自己推导 crossover，并与经验法则比较。
2. 你的产品提供图像生成、聊天和 speech-to-text。为每种 modality 选择平台，并说出统一它们的 gateway pattern。
3. Fireworks 对你的 primary model 提价 $1/hr。如果 40% 流量迁移到 batch tier（50% off），建模 blended cost impact。
4. 一个受监管客户要求 SOC 2 Type II + HIPAA + dedicated GPUs。哪三个平台可行，哪个在 FinOps 上胜出？
5. 比较 Llama 3.1 70B 在 Fireworks serverless、Together on-demand、Baseten dedicated 和 Replicate API 上每 1,000 predictions 的成本。10 predictions/day 时哪个最便宜？10,000 时呢？

## 关键词汇

| 术语 | 大家常说 | 实际含义 |
|------|----------|----------|
| Custom silicon | “non-GPU chips” | Groq LPU、Cerebras WSE、SambaNova RDU：为 decode 优化 |
| FireAttention | “Fireworks engine” | Custom attention kernel；营销口径是 latency 比 vLLM 低 4x |
| Truss | “Baseten's format” | Model packaging manifest；dependencies + secrets + serving config |
| Per-token | “API pricing” | 按消耗 tokens 收费；不为空闲付费 |
| Per-minute | “dedicated pricing” | 按 GPU wall-clock time 收费；高利用率时胜出 |
| Per-prediction | “Replicate pricing” | 按 model invocation 收费；常见于 image/video |
| RayTurbo | “Anyscale engine” | Ray 上的 proprietary inference；在 Ray clusters 上与 vLLM 竞争 |
| Batch tier | “50% off” | 降价的非交互队列；Fireworks、OpenAI 常见 |
| Fine-tuned at base rate | “Fireworks LoRA” | LoRA-served requests 按 base model rate 收费（差异点） |

## 延伸阅读

- [Fireworks Pricing](https://fireworks.ai/pricing) — per-token rates、batch tier、GPU rental。
- [Baseten Pricing](https://www.baseten.co/pricing/) — per-minute rates、committed capacity、enterprise tiers。
- [Modal Pricing](https://modal.com/pricing) — per-second GPU rates 和 free tier。
- [Together AI Pricing](https://www.together.ai/pricing) — model catalog 和 per-token rates。
- [Anyscale Pricing](https://www.anyscale.com/pricing) — RayTurbo 和 managed Ray pricing。
- [Northflank — Fireworks AI Alternatives](https://northflank.com/blog/7-best-fireworks-ai-alternatives-for-inference) — comparative assessment。
- [Infrabase — AI Inference API Providers 2026](https://infrabase.ai/blog/ai-inference-api-providers-compared) — vendor landscape。
