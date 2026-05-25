# Serverless LLM 的 Cold Start Mitigation

> 一个 20 GB model image 从冷启动到 serving 需要 5-10 分钟（7B）到 20+ 分钟（70B）。在真正 serverless 的世界里，这不是 warm-up，而是 outage。缓解手段分布在五层：pre-seeded node images（AWS Bottlerocket，dual-volume architecture）、model streaming（NVIDIA Run:ai Model Streamer，vLLM 原生支持）、GPU memory snapshots（Modal checkpoints，restart 最高快 10x）、warm pools（`min_workers=1`）、tiered loading（ServerlessLLM 的 NVMe→DRAM→HBM pipeline，latency reduction 10-200x），以及移动 input tokens（KB）而不是 KV cache（GB）的 live migration。Modal 发布的 2-4s cold starts 是底线；Baseten 默认 5-10s，预热后 sub-second。本课教你测量、预算并叠加这五层。

**类型：** 学习
**语言：** Python（stdlib，玩具版 cold-start path simulator）
**前置要求：** 阶段 17 · 02（Inference Platform Economics），阶段 17 · 03（GPU Autoscaling）
**时间：** ~60 分钟

## 学习目标

- 枚举 cold-start mitigation 的五层，并为每层说出一个工具或 pattern。
- 对 70B model，把 total cold-start time 计算为（node provision）+（weights download）+（weights load into HBM）+（engine init）。
- 解释为什么 live migration 传输 input tokens（KB）而不是 KV cache（GB），以及代价是什么（recomputation）。
- 说出 warm-pool 取舍（为空闲 GPU 付费或接受 cold-start tail），以及 `min_workers > 0` 在什么 SLA threshold 下变成必需。

## 问题

你的 serverless LLM endpoint 夜间 scale to zero。早上 8 点流量 spike。第一个请求等待：

1. Karpenter provision 一个 GPU node：45-60s。
2. Container 拉取带 weights 的 30 GB image：120-300s。
3. Engine 把 weights 加载到 HBM：45-120s，取决于 model size 和 storage speed。
4. vLLM 或 TRT-LLM 初始化 CUDA graphs、KV cache pool、tokenizer：10-30s。

Total：220-510s（大约 3-8 分钟）之后才返回一个 token。你的 SLA 是 2s。你交付一个 warm-pool（`min_workers=1`），问题似乎消失了，但现在你要为一个 idle GPU 24x7 付费。如果服务有 5 个 products，每个都一个 warm replica，那就是 5 × 24 × 30 = 3,600 GPU-hours/month，不管有没有用户调用。

Cold-start mitigation 是如何保留 serverless economics，同时接近 always-on latency。

## 概念

### 第 1 层 — pre-seeded node images（Bottlerocket）

在 AWS 上，Bottlerocket 的 dual-volume architecture 把 OS 和 data 分开。把已经 pre-pulled 你的 container image 的 data volume 做 snapshot；在 `EC2NodeClass` 中引用 snapshot ID。新 nodes 启动时 weights 已经在 local NVMe 上，步骤 2 和部分步骤 3 消失。与 Karpenter 原生配合。典型节省：大模型每次 cold start 节省 2-4 分钟。

GCP 等价方案：带 pre-baked container layers 的 custom VM images。Azure：同样模式的 managed disk snapshots。

### 第 2 层 — model streaming（Run:ai Model Streamer）

不是先加载完整文件再回答第一个请求，而是逐层把 weights stream 进 GPU memory，并在第一个 transformer block 常驻后立即开始处理。NVIDIA Run:ai Model Streamer 在 vLLM 2026 中原生提供。支持 S3、GCS 和 local NVMe。通过把 I/O 与 compute setup 重叠，大模型 weight-load time 大约减半。

### 第 3 层 — GPU memory snapshots（Modal）

Modal 在首次 load 后对 GPU state（weights、CUDA graphs、KV cache region）做 checkpoint。后续 restarts 直接 deserialize 到 HBM，比重新初始化快 10x。这最接近“2 秒启动 warm GPU”。取舍：snapshots 绑定 GPU topology，所以如果 Karpenter 把你迁移到不同 SKU，就需要重新 checkpoint。

### 第 4 层 — warm pools（min_workers=1）

最简单的缓解：永远保留一个 replica ready。成本是一张 GPU 24x7 的 hourly rate。小模型上算术很残酷（你付 $0.85-$1.50/hr 来避免 30s cold start），大模型上较友好（付 $4/hr 来避免 5 分钟 cold start）。warm pools 变成 mandatory 的 SLA threshold：通常是 70B+ model 上 TTFT P99 < 60s。

### 第 5 层 — tiered loading（ServerlessLLM）

ServerlessLLM 把 storage 视为层级：NVMe（快但大）、DRAM（中等但分层）、HBM（小但即时）。Weights 预加载到 DRAM，并按需 load into HBM。论文报告相较 naive disk-to-HBM，cold loads latency reduction 为 10-200x。Production adoption 还早，但已有 vLLM integration。

### 第 6 层 — live migration（bonus pattern）

当 node 不可用（spot eviction、node drain）时，传统模式是 cold-start 另一个 replica，并 drain request queue。Live migration 会把 input tokens（kilobytes）移动到已加载 model 的 destination，并在 destination 上 recompute KV cache。重算比通过网络传输 GB 级 KV cache 更便宜。适用于 disaggregated deployments。

### Warm-pool 数学

对于 P99 TTFT SLA 为 2s 的服务，问题不是“要不要 warm pool”，而是“要几个 warm replicas，哪些 paths 需要它们”。

- 高价值交互路径（live chat、voice agent）：`min_workers=1-2`。
- 后台 batch 路径（nightly classification）：接受 scale-to-zero，容忍 5-10 分钟 cold start。
- Premium tier：每 tenant 的 `min_workers` 和 dedicated capacity。

### 优化前先测量

新 node 上 70B model 的 cold-start anatomy（示例）：

| Phase | Time | Mitigation |
|-------|------|-----------|
| Node provision | 50s | Bottlerocket + pre-seeded image, warm pool |
| Image pull | 180s | Pre-seeded data volume (eliminate) |
| Weights to HBM | 75s | Model streamer (halve); GPU snapshot (eliminate) |
| Engine init | 20s | Persistent CUDA graph cache |
| First forward | 3s | Min inherent latency |
| **Total cold** | **328s** | |
| **Total with mitigations** | **~15s** | 22x reduction |

### 你应该记住的数字

- Modal cold start：2-4s（with GPU snapshots）。
- Baseten default cold start：5-10s；pre-warming 后 sub-second。
- Raw 70B cold start：3-8 分钟。
- Run:ai Model Streamer：~2x weight-load speedup。
- ServerlessLLM tiered loading：10-200x latency reduction（论文数字）。

## 使用它

`code/main.py` 对有无各项 mitigation 的 cold-start path 建模。报告 total cold-start time、warm-pool cost，以及 warm pool 回本所需的 break-even request rate。

## 交付它

本课会产出 `outputs/skill-cold-start-planner.md`。给定 SLA、model size 和 traffic shape，它会选择要叠加的 mitigations。

## 练习

1. 运行 `code/main.py`。计算 warm replica 比通过 SLO 下 extra request drops 支付 cold-start tax 更便宜的 break-even request rate。
2. 你部署一个 P99 TTFT SLA 为 3s 的 13B model。选择能达到它的 minimum mitigation stack（层数最少）。
3. Bottlerocket pre-seeding 消除了 image pull，但 weights 仍要从 snapshot 加载到 HBM。如果 snapshot-backed NVMe 读取速度为 7 GB/s，计算 70B model 的 wall-clock。
4. 你的 serverless provider 提供 GPU snapshots（Modal），团队拒绝，理由是“snapshots leak PII”。为双方论证：现实风险是什么，mitigation 是什么（ephemeral snapshots、encryption、namespace isolation）？
5. 设计 tiered warm-pool policy：paid users、trial users 和 batch workloads 分别需要多少 warm replicas？展示数学。

## 关键词汇

| 术语 | 大家常说 | 实际含义 |
|------|----------|----------|
| Cold start | “the big pause” | Fresh replica 上从 request 到 first token 的时间 |
| Warm pool | “always-on minimum” | `min_workers >= 1`，至少保持一个 replica ready |
| Pre-seeded image | “baked AMI” | Node image 中已有 container weights |
| Bottlerocket | “AWS node OS” | 支持 dual-volume snapshot 的 AWS container-optimized OS |
| Model streamer | “streaming load” | 将 weights I/O 与 compute setup 重叠 |
| GPU snapshot | “checkpoint to HBM” | 序列化 post-load GPU state；restart 时 deserialize |
| Tiered loading | “NVMe + DRAM + HBM” | storage tiers 层级；按需加载 |
| Live migration | “move tokens” | 传输 input（KB），在 destination 重算 KV |
| `min_workers` | “warm replicas” | Serverless minimum keep-alive count |
| Scale-to-zero | “full serverless” | idle 时零成本；接受完整 cold-start tax |

## 延伸阅读

- [Modal — Cold start performance](https://modal.com/docs/guide/cold-start) — Modal 发布的 benchmarks 和 checkpoint architecture。
- [AWS Bottlerocket](https://github.com/bottlerocket-os/bottlerocket) — pre-seeded data volume snapshot pattern。
- [NVIDIA Run:ai Model Streamer](https://github.com/run-ai/runai-model-streamer) — overlap weights load with compute setup。
- [Baseten — Cold-start mitigation](https://www.baseten.co/blog/cold-start-mitigation/) — pre-warming playbook。
- [ServerlessLLM paper (USENIX OSDI'24)](https://www.usenix.org/conference/osdi24/presentation/fu) — tiered loading design。
- [NVIDIA — Disaggregated LLM Inference on Kubernetes](https://developer.nvidia.com/blog/deploying-disaggregated-llm-inference-workloads-on-kubernetes/) — disaggregated deployments 的 live migration。
