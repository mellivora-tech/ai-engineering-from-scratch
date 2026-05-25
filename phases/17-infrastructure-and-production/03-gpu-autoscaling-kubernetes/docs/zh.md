# Kubernetes 上的 GPU Autoscaling — Karpenter、KAI Scheduler、Gang Scheduling

> 是三层，不是一层。Karpenter 动态 provision nodes（不到一分钟，比 Cluster Autoscaler 快 40%）。KAI Scheduler 处理 gang scheduling、topology awareness 和 hierarchical queues：它能避免 8 个节点只分到 7 个、全都等待并烧钱的 partial allocation trap。Application-level autoscalers（NVIDIA Dynamo Planner、llm-d Workload Variant Autoscaler）根据 inference-specific signals 扩缩容：queue depth、KV cache utilization，而不是 CPU/DCGM duty cycle。经典 HPA 陷阱在于 `DCGM_FI_DEV_GPU_UTIL` 是 duty-cycle measurement：100% 可能是 10 个请求，也可能是 100 个。vLLM 会预分配 KV cache memory，所以 memory 永远不会触发 scale-down。本课教你组合三层，并避开默认 Karpenter `WhenEmptyOrUnderutilized` policy，它会在 inference 过程中终止正在运行的 GPU jobs。

**类型：** 学习
**语言：** Python（stdlib，玩具版 queue-depth autoscaler simulator）
**前置要求：** 阶段 17 · 02（Inference Platform Economics），阶段 17 · 04（vLLM Serving Internals）
**时间：** ~75 分钟

## 学习目标

- 画出三层 autoscaling（node provisioning、gang scheduling、application-level），并说出每层使用的工具。
- 解释为什么 `DCGM_FI_DEV_GPU_UTIL` 对 vLLM 来说是错误的 HPA signal，并说出两个替代指标（queue depth、KV cache utilization）。
- 描述 gang scheduling，以及 KAI Scheduler 防止的 partial-allocation failure mode（8 个 GPU 中 7 个 idle）。
- 说出会终止正在运行 GPU jobs 的 Karpenter consolidation policy（`WhenEmptyOrUnderutilized`），并说明 2026 年的安全替代方案。

## 问题

你的团队在 Kubernetes 上交付了一个 LLM-serving service。你把 `DCGM_FI_DEV_GPU_UTIL` 设为 HPA signal。业务时段服务固定在 100% utilization。HPA 从不 scale up，因为它已经认为你满了。你手动加一个 replica，TTFT 降了。HPA 仍然不扩容。这个 signal 在骗你。

另外，你用 Cluster Autoscaler 管理 nodes。凌晨 2 点来了一个 1M-token prompt；cluster 花 3 分钟 provision 一个 node，请求超时。

再另外，你部署一个需要跨 2 个 nodes 使用 8 个 GPU 的 70B model。cluster 有 7 个空闲 GPU，另有 1 个分散在 3 个 nodes 上。Cluster Autoscaler 为缺失的 1 个 GPU provision 一个 node。七个 nodes 等待 4 分钟，一边烧钱一边等 Kubernetes 把最后的 GPU 拉起来。

三层，三种不同 failure mode。2026 年的 GPU-aware autoscaling 不是“打开 HPA”。它是组合 node provisioning、gang scheduling 和 application-signal autoscaling。

## 概念

### 第 1 层 — node provisioning（Karpenter）

Karpenter 观察 pending pods，并在约 45-60 秒内 provision nodes（Cluster Autoscaler 对 GPU nodes 通常需要 90-120 秒）。它会根据 `NodePool` constraint 动态选择 instance types；如果你的 pod 需要 8 个 H100，而 cluster 没有匹配 node，Karpenter 会直接 provision 一个，而不是扩展现有 group。

**Consolidation 陷阱**：Karpenter 默认的 `consolidationPolicy: WhenEmptyOrUnderutilized` 对 GPU pools 很危险。它会终止一个正在运行的 GPU node，把 pods 迁移到更便宜、尺寸更合适的实例。对于 inference workload，这意味着驱逐正在运行的请求，并在新 node 上重新加载 70B model。损失是几分钟容量加上请求失败。

GPU pools 的安全设置：

```yaml
disruption:
  consolidationPolicy: WhenEmpty
  consolidateAfter: 1h
```

这允许 Karpenter 在一小时后 consolidate 真正空闲的 nodes，但绝不会 evict 正在运行的 job。

### 第 2 层 — gang scheduling（KAI Scheduler）

KAI Scheduler（项目原名 “Karp”，后改名）处理默认 kube-scheduler 不处理的事情：

**Gang scheduling** — all-or-nothing 调度。一个需要 8 个 GPU 的 distributed inference pod，要么 8 个全部一起启动，要么一个都不启动。没有它，就会出现 partial-allocation trap：8 个 pods 中 7 个启动、无限等待、持续烧钱。

**Topology awareness** — 知道哪些 GPU 共享 NVLink，哪些在同一 rack，哪些之间有 InfiniBand。相应地放置 pods。DeepSeek-V3 67B tensor-parallel workload 必须留在一个 NVLink domain 内；KAI Scheduler 会尊重这一点。

**Hierarchical queues** — 多个团队用 priority 和 quota 竞争同一个 GPU pool。只有 priority 规则允许时，Team A 的 production 紧急需求才会被 Team B 的 training job 抢占。

KAI 作为 secondary scheduler 与 kube-scheduler 一起部署；你通过注解让 workload 使用它。Ray 和 vLLM production-stack 都有集成。

### 第 3 层 — application-level signals

**HPA 陷阱**：`DCGM_FI_DEV_GPU_UTIL` 是 duty-cycle metric，它测量 GPU 在每个采样间隔是否在工作。100% utilization 可能意味着 10 个 concurrent requests，也可能意味着 100 个；无论如何 GPU 都很忙。按 duty cycle 扩缩容就是盲目扩缩容。

更糟的是，vLLM 和类似引擎会预分配 KV cache memory（最高到 `--gpu-memory-utilization`）。即使只有一个请求，memory usage 也会接近 90%。基于 memory 的 HPA 永远不会 scale down。

**2026 替代 signals**：

- Queue depth（等待 prefill 的请求数）。
- KV cache utilization（active sequences 已分配的 blocks 比例）。
- Per-replica P99 TTFT（你的 SLA signal）。
- Goodput（每秒满足所有 SLO 的请求数）。

NVIDIA Dynamo Planner 和 llm-d Workload Variant Autoscaler 会消费这些 signals 并扩缩 replicas。对于 LLM serving，它们直接替代 HPA。

### 什么时候用什么

| Scale decision | Tool |
|----------------|------|
| Add/remove nodes | Karpenter |
| Schedule multi-GPU jobs | KAI Scheduler |
| Add/remove replicas | Dynamo Planner / llm-d WVA（或基于 queue depth 的自定义 HPA） |
| Choose GPU type | Karpenter NodePool |
| Preempt low-priority | KAI Scheduler queues |

### Disaggregated prefill/decode 让一切更复杂

如果你运行 disaggregated prefill/decode（阶段 17 · 17），你有两类 pod，它们的 scaling triggers 不同：prefill pods 按 queue depth 扩缩，decode pods 按 KV cache pressure 扩缩。llm-d 会把它们暴露为独立 `Services`，并为每个 role 配置 HPA。不要试图在二者前面放一个单一 HPA。

### Cold start 在这里也重要

Cold-start mitigation（阶段 17 · 10）是 node provisioning time 变成用户可见 latency 的地方。Karpenter 的 45-60 秒 warm-up，加上 20GB model load，再加 engine init，意味着 from-zero request 需要 2-5 分钟。对 SLO-critical paths 保持 warm pool（`min_workers=1`），或者在 application layer 使用 Modal-style checkpointing。

### 你应该记住的数字

- Karpenter node provisioning：~45-60s；Cluster Autoscaler ~90-120s（GPU nodes）。
- KAI Scheduler 防止 partial-allocation waste：7-of-8 trap。
- `DCGM_FI_DEV_GPU_UTIL` 作为 HPA signal：坏的；使用 queue depth 或 KV utilization。
- Karpenter `WhenEmptyOrUnderutilized`：会终止正在运行的 GPU jobs。Inference 使用 `WhenEmpty + consolidateAfter: 1h`。

## 使用它

`code/main.py` 在 bursty GPU workload 上模拟三层 autoscaler。比较 naive HPA（duty cycle）、queue-depth HPA 和 KAI-gang-scheduled scaling。报告 unmet requests、idle-GPU minutes 和 composite score。

## 交付它

本课会产出 `outputs/skill-gpu-autoscaler-plan.md`。给定 cluster topology、workload shape 和 SLO，它会设计三层 autoscaling plan。

## 练习

1. 运行 `code/main.py`。在 bursty workload 下，naive duty-cycle HPA 会丢掉多少 queue-depth HPA 能接住的请求？差异来自哪里？
2. 为一个服务 Llama 3.3 70B FP8 on H100 SXM5 的 cluster 设计 Karpenter NodePool。指定 `capacity-type`、`disruption.consolidationPolicy`、`consolidateAfter`，以及一个让非 GPU workload 远离这些 nodes 的 taint。
3. 你的团队报告 deployments 卡在 Pending，因为“有 GPU 可用但 pod 不调度”。诊断：这是 Karpenter、kube-scheduler 还是 KAI Scheduler？哪些 metrics 能确认？
4. 为 disaggregated prefill pods 选择一个 autoscale signal，并为 decode pods 选择另一个。说明理由。
5. 计算 `WhenEmptyOrUnderutilized` consolidation trap 对一个 24x7 production service 的成本：该服务平均每天发生 60 次 request-dropping events，P99 TTFT > 10s。

## 关键词汇

| 术语 | 大家常说 | 实际含义 |
|------|----------|----------|
| Karpenter | “the node provisioner” | Kubernetes node autoscaler；sub-minute provisioning |
| Cluster Autoscaler | “the old scaler” | Kubernetes node autoscaler 前身；更慢，基于 group |
| KAI Scheduler | “the GPU scheduler” | 用于 gang + topology + queues 的 secondary scheduler |
| Gang scheduling | “all or nothing” | 原子地调度 N 个 pods，或者全部延后 |
| Topology awareness | “rack-aware” | 基于 NVLink/IB/rack placement 放置 pods |
| `DCGM_FI_DEV_GPU_UTIL` | “GPU utilization” | Duty-cycle metric；不是 LLM 的 scaling signal |
| Queue depth | “waiting requests” | Prefill-bound scaling 的正确 HPA signal |
| KV cache utilization | “memory pressure” | Decode-bound scaling 的正确 HPA signal |
| Consolidation | “Karpenter consolidation” | 终止 node 并迁移到更便宜实例类型 |
| `WhenEmpty + 1h` | “safe consolidation” | 不 evict 正在运行 GPU jobs 的 policy |

## 延伸阅读

- [KAI Scheduler GitHub](https://github.com/kai-scheduler/KAI-Scheduler) — design docs 和 configuration examples。
- [Karpenter Disruption Controls](https://karpenter.sh/docs/concepts/disruption/) — consolidation policy semantics 和 GPU-safe defaults。
- [NVIDIA — Disaggregated LLM Inference on Kubernetes](https://developer.nvidia.com/blog/deploying-disaggregated-llm-inference-workloads-on-kubernetes/) — Dynamo Planner scaling signals。
- [Ray docs — KAI Scheduler for RayClusters](https://docs.ray.io/en/latest/cluster/kubernetes/k8s-ecosystem/kai-scheduler.html) — Ray integration pattern。
- [AWS EKS Compute and Autoscaling Best Practices](https://docs.aws.amazon.com/eks/latest/best-practices/aiml-compute.html) — managed-Kubernetes-specific guidance。
- [llm-d GitHub](https://github.com/llm-d/llm-d) — Workload Variant Autoscaler design。
