# Multi-Region LLM Serving 与 KV Cache Locality

> 对 cached LLM inference 来说，round-robin load balancing 会主动伤害性能。一个没有落到持有其 prefix 的 node 上的请求要支付完整 prefill cost：长 prompt 下 P50 约 800 ms，而 cache hit 约 80 ms。2026 年 production pattern 是 cache-aware router（Rust 写的 vLLM Router、llm-d router），它消费 KV-cache events，并按 prefix-hash match 路由。近期研究（GORGO）把 cross-region network latency 显式加入 routing objective。商业 “cross-region inference” 产品（Bedrock cross-region inference、GKE multi-cluster gateways）把 inference 当作 opaque，只处理 availability，不处理 TTFT。JPMorgan 和 Mayo Clinic 在 2024 年 11 月演练 us-east-1 failover，约 22 分钟恢复。DR 现实是：32% 的 LLM DR failures 是因为团队备份了 weights，却忘记 tokenizer files 或 quantization configs。

**类型：** 学习
**语言：** Python（stdlib，玩具版 prefix-cache-aware router simulator）
**前置要求：** 阶段 17 · 04（vLLM Serving），阶段 17 · 06（SGLang RadixAttention）
**时间：** ~60 分钟

## 学习目标

- 解释为什么 round-robin load balancing 会破坏 cached inference，并量化 TTFT penalty。
- 画出 cache-aware router：inputs（KV-cache events）、algorithm（prefix-hash match）、tie-breaker（GPU utilization）。
- 说出 LLM 的 32% DR failure driver（missing tokenizer files / quantization configs），并给出三文件 DR checklist。
- 区分商业 cross-region offerings（Bedrock CRI、GKE Multi-Cluster Gateway）和 KV-aware routing。

## 问题

你的服务运行在 us-east-1、us-west-2 和 eu-west-1。你在前面放了一个 ALB，使用 round-robin。Production 的 prefix cache hit rate 掉到 8%。TTFT P50 翻三倍。vLLM 日志显示每个请求都在支付完整 prefill cost。

Round-robin 对 stateless services 是最优的。LLM inference 天生是 stateful：KV cache 编码了模型看到的一切。盲目路由就是把请求路由进错误 cache。

另一个问题：你的团队有 DR plan。你把 model weights cross-region 备份到 S3。区域 outage 发生；你尝试 failover；replica 拒绝启动。你忘了 tokenizer.json、quantization config 和 RoPE scaling config 在另一个没有同步的 bucket 里。

Multi-region LLM serving 是 cache problem、routing problem 和 DR-hygiene problem，而不是 load-balancer problem。

## 概念

### Cache-aware routing

请求带着 prompt 到达。Router hash prefix（比如前 512 tokens）；它询问每个 replica：“你有这个 prefix cached 吗？”Replicas 在分配和 evict blocks 时，通过 pub/sub channel 发布 KV-cache events。Router 选择匹配的 replica；如果没有匹配，就 fallback 到基于 GPU-util 的 tie-breaker。

**vLLM Router**（Rust，2026 production-stack）：订阅 `kv.cache.block_added` events，维护 prefix-hash → replica index，用 O(1) lookup 路由。没有匹配时 fallback 到 least-queue-depth。

**llm-d router**：相同 pattern，Kubernetes-native。通过 ControlPlane API 发布 events。

**SGLang RadixAttention**（阶段 17 · 06）是 intra-replica 等价物。Cross-replica routing 严格在 upstream。

### 数字

2K-token prompt、Llama 3.3 70B FP8、H100 上的 TTFT P50：
- Cache hit（同 replica，prefix resident）：~80 ms。
- Cache miss（cold prefill）：~800 ms。

10x 差距。如果 router 在 replicas 之间达到 60-80% prefix cache 命中，你会在 N-replica capacity 下接近 single-replica performance。如果只命中 10%，你接近 naive scaling。

### Cross-region 有新的约束：network latency

Inter-region RTT：
- us-east-1 ↔ us-west-2：~65 ms。
- us-east-1 ↔ eu-west-1：~75 ms。
- us-east-1 ↔ ap-southeast-1：~220 ms。

如果 routing 把一个来自 us-east-1 的请求发到 ap-southeast-1 的 hot prefix，节省的 prefill（800 → 80 ms）会被 440 ms round-trip 吞掉。GORGO（2026 research）把这一点显式化：联合最小化 `prefill_time + network_latency`，而不是只最小化 prefill。很多时候答案是保持 regional routing，除非 prefix 是 massive multi-MB，prefill 成为绝对主导。

### 商业 “cross-region inference” 在这里帮不上忙

AWS Bedrock cross-region inference 会在 capacity pressure 时自动把请求路由到其他 regions。它优化 availability，不优化 TTFT，并把 inference 当作 opaque。GKE Multi-Cluster Gateway 也一样：service-level failover，没有 KV cache awareness。

即使使用这些产品，你仍然需要 app-layer cache-aware router。它们处理 “us-east-1 着火了” 的情况。Cache-aware routing 处理 TTFT 情况。

### DR hygiene — 32% missing-files 问题

广泛引用的 2026 统计：32% 的 LLM DR failures 发生是因为团队备份了 weights，却忘记：

- `tokenizer.json` 或 `tokenizer.model`
- Quantization configs（`quantize_config.json`、AWQ scales、GPTQ zero-points）
- Model-specific configs（RoPE scaling、attention masks、chat templates）
- Engine config（`vllm_config.yaml`、sampling defaults、LoRA adapter manifests）

修复是三文件 minimum DR manifest：

1. HF model repo 下的所有文件（weights + configs + tokenizer）。
2. Engine-specific serving config。
3. Deployment manifest（K8s YAML、Dockerfile、dependency lock）。

另外：每季度跑一次 DR drill。JPMorgan 在 2024 年 11 月 us-east-1 演练中达到 22 分钟 recovery，正是因为 playbook 被演练过。

### Data residency 是正交问题

EU customer PHI 不能离开 EU。如果你的 cache-aware router 为了 prefix match 把 Paris-originated request 发到 us-east-1，你就违反了 GDPR，不管 TTFT 收益多大。在优化 cache 之前，先按 residency boundary 分区 routers。

### 你应该记住的数字

- Cache hit vs miss TTFT gap：~10x（2K prompt 上 80 ms vs 800 ms）。
- Inter-region RTT US-EU：~75 ms。
- DR failure：32% 缺 tokenizer/quant configs。
- JPMorgan us-east-1 failover 2024 年 11 月：22 分钟（30-min SLA）。

## 使用它

`code/main.py` 在 multi-region workload 上模拟三种 routing strategies（round-robin、cache-aware regional、cache-aware global）。报告 cache hit rate、TTFT P50/P99 和 cross-region bill。

## 交付它

本课会产出 `outputs/skill-multi-region-router.md`。给定 regions、residency constraints 和 SLA，它会设计 routing plan。

## 练习

1. 运行 `code/main.py`。给定 75 ms RTT，prompt length 到多少时 cross-region routing 胜过 local-only routing？
2. 你的 cache hit rate 从 70% 掉到 12%。诊断三个可能原因，以及能确认每个原因的 observables。
3. 为一个在 vLLM 中服务、带 5 个 LoRA adapters 的 70B AWQ-quantized model 设计 DR manifest。列出每个 file 和 config。
4. 论证 Bedrock cross-region inference 对一个有严格 TTFT SLO 的 fintech 是否“足够”。引用具体 behaviors。
5. 一个 Paris-origin request 在 us-east-1 中匹配到 prefix。你会路由过去吗？写出 policy。

## 关键词汇

| 术语 | 大家常说 | 实际含义 |
|------|----------|----------|
| Cache-aware routing | “smart LB” | 按 prefix-hash match 路由到持有 KV-cache 的 replica |
| KV-cache events | “cache pub-sub” | Replicas 发布 block add/evict；router 建索引 |
| Prefix hash | “cache key” | 前 N tokens 的 hash，用作 router lookup |
| GORGO | “cross-region routing research” | arXiv 2602.11688；network latency 作为显式项 |
| Cross-region inference | “Bedrock CRI” | AWS 产品；availability failover，而不是 TTFT awareness |
| DR manifest | “the backup list” | 恢复所需的每个文件，不只是 weights |
| Data residency | “GDPR boundary” | 关于哪个 region 可以看到 user data 的法律约束 |
| RTT | “round-trip time” | Network latency；US-EU 75 ms，US-APAC 220 ms |
| LLM-aware LB | “cache-hit LB” | Cache-aware router 作为一个产品类别 |

## 延伸阅读

- [BentoML — Multi-cloud and cross-region inference](https://bentoml.com/llm/infrastructure-and-operations/multi-cloud-and-cross-region-inference)
- [arXiv — GORGO (2602.11688)](https://arxiv.org/html/2602.11688v1) — 带 network latency term 的 cross-region KV-cache reuse。
- [TianPan — Multi-Region LLM Serving Cache Locality](https://tianpan.co/blog/2026-04-17-multi-region-llm-serving-data-residency-routing)
- [AWS Bedrock Cross-Region Inference](https://docs.aws.amazon.com/bedrock/latest/userguide/cross-region-inference.html) — availability failover documentation。
- [vLLM Production Stack Router](https://github.com/vllm-project/production-stack) — cache-aware router source。
