# 托管 LLM 平台 — Bedrock、Vertex AI、Azure OpenAI

> 三家 hyperscaler，三种截然不同的策略。AWS Bedrock 是模型市场：Claude、Llama、Titan、Stability、Cohere 都在同一个 API 后面。Azure OpenAI 是独家 OpenAI 合作关系，加上 Provisioned Throughput Units（PTUs）提供专用容量。Vertex AI 以 Gemini 为先，长上下文和多模态叙事最强。2026 年 Artificial Analysis 在 Llama 3.1 405B 等价部署上测得 Azure OpenAI 中位数约 50 ms，Bedrock 约 75 ms；PTU 解释了这个差距，因为专用容量胜过共享 on-demand。决策规则不是“哪个最快”，而是“哪个模型目录和 FinOps 表面最匹配我的产品”。本课教你把取舍写清楚，而不是凭感觉选择。

**类型：** 学习
**语言：** Python（stdlib，玩具版成本与 latency 比较器）
**前置要求：** 阶段 11（LLM Engineering），阶段 13（Tools & Protocols）
**时间：** ~60 分钟

## 学习目标

- 说出三种平台策略（marketplace vs exclusive vs Gemini-first），并把每种策略匹配到产品用例。
- 解释 Azure OpenAI 的 Provisioned Throughput Units（PTUs）买到什么，以及为什么 on-demand Bedrock 在 405B 规模通常慢约 25 ms。
- 画出每个平台的 FinOps 归因表面（Bedrock Application Inference Profiles、Vertex project-per-team、Azure scopes + PTU reservations）。
- 写下一条“至少两个 provider”的策略，并解释为什么单一供应商 lock-in 是 2026 年昂贵的错误。

## 问题

你为产品选择了 Claude 3.7 Sonnet。现在需要把它服务化。你可以直接调用 Anthropic API，也可以通过 AWS Bedrock 调用，或者走一个 gateway。直连 API 最简单；Bedrock 增加了 BAA、VPC endpoints、IAM 和 CloudWatch 归因。gateway 增加了 failover、统一计费，以及跨 provider 的 rate limit。

更深的问题是目录。如果你的同一个产品里需要 Claude、Llama 和 Gemini，除非同时使用 Bedrock、Vertex 和 Azure OpenAI，否则无法从同一个地方买到全部模型。hyperscaler 之间不可互换，因为它们各自对“谁拥有模型层”下注不同。

本课会梳理这三种下注、latency 差距、FinOps 差距和 lock-in 风险。

## 概念

### 三种策略

**AWS Bedrock** — marketplace。Claude（Anthropic）、Llama（Meta）、Titan（AWS first-party）、Stability（图像）、Cohere（embeddings）、Mistral，以及图像和 embedding 子目录。一个 API，一个 IAM 表面，一个 CloudWatch export。Bedrock 的押注是：客户想要可选性，胜过想要单一模型。

**Azure OpenAI** — 独家合作。你可以在 Azure 数据中心使用 GPT-4 / 4o / 5 / o-series、DALL·E、Whisper，以及 OpenAI 模型 fine-tuning。“Azure OpenAI Service”目录里没有非 OpenAI 模型，这些会进入 Azure AI Foundry（另一个产品）。Azure 的押注是：OpenAI 仍然处在 frontier，客户想要围绕这段特定关系的企业控制能力。

**Vertex AI** — Gemini first，其它其次。Gemini 1.5 / 2.0 / 2.5 Flash 和 Pro，加上 Model Garden（第三方）。Vertex 的押注是多模态长上下文，1M-token Gemini context 是差异点。

### 规模下的 latency 差距

Artificial Analysis 持续运行 benchmark。在等价的 Llama 3.1 405B 部署（共享 on-demand）上，Azure OpenAI 的中位 first-token latency 约 50 ms；Bedrock 约 75 ms。这个差距不是 AWS 失败，而是容量模型不同。Azure 销售 PTU（Provisioned Throughput Units），为你的 tenant 预留 GPU 容量。Bedrock 也有等价能力（Provisioned Throughput），但每 unit 起价约 $21/hour，大多数客户仍留在共享 on-demand。

On-demand 共享容量会和所有其他客户流量竞争。专用容量不会。如果你的产品 SLA 是 TTFT < 100 ms at P99，你要么购买 Azure PTU，要么购买 Bedrock Provisioned Throughput，要么接受默认 variance。

### Provisioned Throughput 经济学

Azure PTU：一块预留的 inference compute。对可预测 workload，相比 on-demand 最高可省约 70%。成本按小时固定，与流量无关，即使 idle 也要为 reservation 付费。break-even 通常在约 40-60% sustained utilization。

Bedrock Provisioned Throughput：每小时 $21-$50，取决于模型和 region。数学类似，break-even 大约是峰值利用率的一半。需要月度承诺。

Vertex provisioned capacity 按 Gemini SKU 销售；价格随模型和 region 变化，公开信息较少。

### FinOps 表面 — 真正的差异点

**Bedrock Application Inference Profiles** 是 marketplace 中最干净的归因方式。用 `team`、`product`、`feature` 给 profile 打 tag；所有模型调用都通过它路由；CloudWatch 可以按 profile 拆分成本，不需要 post-processing。该能力 2025 年加入，仍是 hyperscaler 原生里最细粒度的方案。

**Vertex** 的归因方式是 project-per-team 加上 everywhere labels。你把每个团队建模为一个 GCP project，在每个资源上打 label，然后用 BigQuery Billing Export + DataStudio 做汇总。工作更多，但 BigQuery 让你可以对成本数据写任意 SQL。

**Azure** 依赖 subscription/resource-group scope 加 tags，PTU reservation 是一级成本对象。Tags 从 resource group 继承，而不是从 request 继承，所以 per-request attribution 需要 Application Insights custom metrics，或者一个会打 header 的 gateway。

模式是：Bedrock 原生最清晰，Vertex 通过 BigQuery 最灵活，Azure 如果不 instrument 就最不透明。

### Lock-in 是 2026 年的风险

当一个模型占主导时，押注单一 hyperscaler 没问题。2026 年 frontier 按月移动：一个季度是 Claude 3.7，下一个季度是 Gemini 2.5，再下一个季度是 GPT-5。锁定一个平台，就把自己挡在三分之二的 frontier 之外。

有效团队采用的模式：任何产品关键 LLM 调用至少两个 provider。Bedrock 加 Azure OpenAI 是常见组合，一个提供 Claude，一个提供 GPT，在同一个 gateway 后面互相 failover。成本上浮可以忽略，因为 gateway 会路由到最优路径；在 outage 期间（比如 Azure OpenAI 2025 年 1 月事件、AWS us-east-1 outage），可用性收益是决定性的。

### Data residency、BAA 和受监管行业

Bedrock：大多数 region 支持 BAA；VPC endpoints；guardrails。常见 fintech 默认选择。
Azure OpenAI：HIPAA、SOC 2、ISO 27001；EU data residency；企业受监管场景默认选择。
Vertex：HIPAA、GDPR、按 region 的 data residency；Google Cloud 的合规栈。

三者都能满足基础 checkbox。差异在 data retention policy、日志处理方式，以及 abuse-monitoring 是否会读取你的流量（多数默认 opt-in；企业可 opt-out）。

### 你应该记住的数字

- Azure OpenAI 在 Llama 3.1 405B 等价部署上的中位 TTFT：~50 ms（使用 PTU）。
- Bedrock on-demand 中位 TTFT：~75 ms。
- Bedrock Provisioned Throughput：每 unit $21-$50/hr。
- Azure PTU break-even：~40-60% sustained utilization。
- 高利用率下 PTU 相比 on-demand 的节省：最高 70%。

## 使用它

`code/main.py` 在合成 workload 上比较三个平台：它建模 on-demand vs PTU economics、TTFT variance 和 cost attribution fidelity。运行它，看看 PTU 在哪里回本，以及 marketplace 的模型广度在哪里压过 TTFT 差距。

## 交付它

本课会产出 `outputs/skill-managed-platform-picker.md`。给定 workload profile（所需模型、TTFT SLA、每日 volume、compliance requirements），它会推荐 primary platform、fallback，以及 FinOps instrumentation plan。

## 练习

1. 运行 `code/main.py`。对于 70B class model，Azure PTU 在什么 sustained utilization 下胜过 on-demand？计算 break-even，并与宣传的 40-60% 区间比较。
2. 你的产品需要 Claude 3.7 Sonnet 和 GPT-4o。设计一个 two-provider deployment：哪个模型走哪个 hyperscaler，前面放什么 gateway，failover policy 是什么？
3. 一个受监管的医疗客户要求 BAA、US-East data residency，以及 sub-100ms P99 TTFT。选择一个平台，并用三个具体功能证明。
4. 你发现本月 Bedrock 账单在流量不变的情况下涨了 4x。没有 Application Inference Profiles 时，你会如何找出 culprit？有 profiles 时，需要多久？
5. 阅读 Azure OpenAI 和 Bedrock pricing pages。对于 100M-token/month 的 Claude workload，哪个更便宜：direct Anthropic API、Bedrock on-demand，还是 Bedrock Provisioned Throughput？

## 关键词汇

| 术语 | 大家常说 | 实际含义 |
|------|----------|----------|
| Bedrock | “AWS LLM service” | 跨 Claude、Llama、Titan、Mistral、Cohere 的模型 marketplace |
| Azure OpenAI | “Azure 的 ChatGPT” | Azure 数据中心里的独家 OpenAI 模型，带企业控制 |
| Vertex AI | “Google 的 LLM” | Gemini-first 平台，用 Model Garden 承载第三方模型 |
| PTU | “dedicated capacity” | Provisioned Throughput Unit：预留 inference GPUs，按小时计价 |
| Application Inference Profile | “Bedrock tagging” | 带 tags 的 per-product cost/usage profile，CloudWatch-native |
| Model Garden | “Vertex catalog” | Vertex AI 的第三方模型区，独立于 Gemini |
| Two-provider minimum | “LLM redundancy” | 每条关键 LLM 路径跨 ≥2 个 hyperscaler 运行的策略 |
| BAA | “HIPAA paperwork” | Business Associate Agreement；PHI 必需；三家都提供 |
| Abuse monitoring | “the log watcher” | Provider 侧对 prompts/outputs 的安全扫描；企业可 opt-out |

## 延伸阅读

- [AWS Bedrock Pricing](https://aws.amazon.com/bedrock/pricing/) — 权威 rate card 和 Provisioned Throughput pricing。
- [Azure OpenAI Service Pricing](https://azure.microsoft.com/en-us/pricing/details/cognitive-services/openai-service/) — PTU economics 和 rate cards。
- [Vertex AI Generative AI Pricing](https://cloud.google.com/vertex-ai/generative-ai/pricing) — Gemini tiers 和 Model Garden surcharges。
- [Artificial Analysis LLM Leaderboard](https://artificialanalysis.ai/) — 跨 provider 的持续 latency 和 throughput benchmarks。
- [The AI Journal — AWS Bedrock vs Azure OpenAI CTO Guide 2026](https://theaijournal.co/2026/03/aws-bedrock-vs-azure-openai/) — 企业决策框架。
- [Finout — Bedrock vs Vertex vs Azure FinOps](https://www.finout.io/blog/bedrock-vs.-vertex-vs.-azure-cognitive-a-finops-comparison-for-ai-spend) — attribution mechanics 横向对比。
