# Disaggregated Prefill/Decode — NVIDIA Dynamo 和 llm-d

> Prefill 是 compute-bound；decode 是 memory-bound。把两者放在同一 GPU 上会浪费一种资源。Disaggregation 把它们拆到不同 pools，并通过 NIXL（RDMA/InfiniBand 或 TCP fallback）在二者之间传输 KV cache。NVIDIA Dynamo（GTC 2025 announce，1.0 GA）位于 vLLM/SGLang/TRT-LLM 之上：它的 Planner Profiler + SLA Planner 会自动按 SLO 匹配 prefill:decode ratios。NVIDIA 发布的 throughput gains 大致在这个区间：developer.nvidia.com（2025-06）显示 GB200 NVL72 + Dynamo 上 DeepSeek-R1 MoE 在 medium-latency regime 提升约 6x；Dynamo product page（developer.nvidia.com，未标日期）宣称 GB300 NVL72 + Dynamo 相比 Hopper 的 MoE throughput 最高 50x。“30x”数字是 full-stack Blackwell + Dynamo + DeepSeek-R1 报告的 community aggregate；我们没有找到单一 primary source 精确说明 30x，所以把它当作 directional claim。llm-d（Red Hat + AWS）是 Kubernetes-native：prefill / decode / router 作为独立 Services，并有 per-role HPA。llm-d 0.5 增加 hierarchical KV offloading、cache-aware LoRA routing、UCCL networking、scale-to-zero。Economics：多个客户披露的内部 rollup 表明，从 colocated serving 切换到带 Dynamo 的 disaggregated serving，在 SLA 不变时可在 $2M-class inference spend 上节省 30–40%（即 $600-800K/year）；具体 $2M→$600-800K 是 internal composite，不是单一公开 case study，请把它当数量级 anchor，而不是引用来源。短 prompts（<512 tokens、short output）不值得支付 transfer cost。

**类型：** 学习
**语言：** Python（stdlib，玩具版 disaggregated-vs-colocated simulator）
**前置要求：** 阶段 17 · 04（vLLM Serving Internals），阶段 17 · 08（Inference Metrics）
**时间：** ~75 分钟

## 学习目标

- 解释为什么 prefill 和 decode 有不同的 optimal GPU allocations，并量化 colocation 下的浪费。
- 画出 disaggregated architecture：prefill pool、decode pool、通过 NIXL 的 KV transfer、router。
- 说出 disaggregation 不划算的条件（short prompts、short outputs）。
- 区分 NVIDIA Dynamo（stack-above）和 llm-d（Kubernetes-native），并把每个匹配到 operational context。

## 问题

你在 8 张 H100 上运行 Llama 3.3 70B。在 mixed workload（long prompts + short outputs）下，GPU 在 decode 期间 idle，因为大部分 compute 花在 prefill 上。换一种 workload（short prompts + long outputs）时，情况相反。Colocated prefill + decode 意味着你同时 over-provision 两者。

预算影响：20-40% GPU time 浪费在错误资源上。你买 H100 compute 来跑 memory-bound decode，或者买 H100 HBM bandwidth 来跑 compute-bound prefill。两者都是昂贵浪费。

Disaggregation 把 prefill 和 decode 分到不同 pools，按各自 bottleneck 调整大小。KV cache 通过 high-bandwidth interconnect 从 prefill pool 转到 decode pool。

## 概念

### 为什么 bottlenecks 不同

**Prefill** — 在完整 input prompt 上跑一次 transformer forward。Matrix multiplications 主导；compute-bound。H100 FP8 给出约 2000 TFLOPS useful throughput。Batch efficiency 很好：一次 forward 处理许多 tokens。

**Decode** — 一次生成一个 token，每次 iteration 都读取完整 weights。Memory-bandwidth-bound。HBM3 给出约 3 TB/s。只有在高并发下 batch efficiency 才好，因为 weights read 会在 batch 中摊销。

Colocating 它们：你买的是同时优化两者的 GPUs。H100 两者都擅长，但无论如何成本相同。规模化时，你希望 prefill pool 用 H100 / compute-heavy，decode pool 用 H200 / memory-heavy，或者使用 aggressive quantization。

### Architecture

```
            ┌──────────────┐
  Request → │    Router    │ ───────────────────────┐
            └──────┬───────┘                        │
                   │                                │
                   ▼ (prompt only)                  │
            ┌──────────────┐    KV cache    ┌───────▼──────┐
            │ Prefill pool │ ─── NIXL ────► │ Decode pool  │
            │  (compute)   │                │  (memory)    │
            └──────────────┘                └──────┬───────┘
                                                   │ tokens
                                                   ▼
                                                 Client
```

NIXL 是 NVIDIA 的 inter-node transport。有 RDMA/InfiniBand 时使用它，否则 TCP fallback。Transfer latency 是真实成本：70B FP8 上 4K-token prompt 的 KV cache 通常需要 20-80 ms。这就是为什么 short prompts 不值得 disaggregation：transfer tax 超过节省。

### Dynamo vs llm-d

**NVIDIA Dynamo**（GTC 2025 announce，1.0 GA）：
- 作为 orchestrator 位于 vLLM、SGLang、TRT-LLM 之上。
- Planner Profiler 测量 workload，SLA Planner 自动配置 prefill:decode ratios。
- Rust core，Python extensibility。
- Throughput gains：NVIDIA 报告 GB200 NVL72 + Dynamo 上 DeepSeek-R1 MoE 在 medium-latency regime 提升 6x（developer.nvidia.com，2025-06）；full Blackwell + Dynamo + DeepSeek-R1 stacks 上 “up to 30x” 的 community reports 缺少单一 primary source，应视为 directional。
- GB300 NVL72 + Dynamo：根据 Dynamo product page（developer.nvidia.com，未标日期），相对 Hopper 的 MoE throughput 最高 50x。

**llm-d**（Red Hat + AWS，Kubernetes-native）：
- Prefill / decode / router 作为独立 Kubernetes Services。
- Per-role HPA，使用 queue depth（prefill）/ KV utilization（decode）signals。
- `topologyConstraint packDomain: rack` 把 prefill+decode cliques 打包到同一 rack，获得 high-bandwidth KV transfer。
- llm-d 0.5（2026）：hierarchical KV offloading、cache-aware LoRA routing、UCCL networking、scale-to-zero。

如果你想要 managed stack-above orchestrator，用 Dynamo。如果你想要 Kubernetes-native primitives，并承诺 CNCF ecosystem，用 llm-d。

### Economics

Internal composite（不是单一 published case study，只是数量级 anchor）：

- Colocated serving 每年 $2M inference spend。
- 切换到 Dynamo disaggregated。
- 同样 request volume、同样 P99 latency SLA。
- 报告节省：$600K–$800K/year（30–40% reduction）。
- 无新硬件。

我们从多个客户披露中综合出这个数字，而不是来自单一可引用 case study；最接近的公开数据点是 Baseten 的 Dynamo KV routing 带来 2x faster TTFT / 61% higher throughput（baseten.co，2025-10），以及 VAST + CoreWeave 对 40–60% KV hit rate 下 tokens/$ 增加 60–130% 的预测（vastdata.com，2025-12）。节省来自 right-sizing each pool；prefill-heavy workloads（RAG with 8K+ prefixes）比 balanced ones 受益更多。

### 什么时候不要 disaggregate

- Prompts < 512 tokens 且 outputs < 200 tokens：transfer tax 主导收益。
- Small cluster（< 4 GPUs）：没有足够 pool diversity。
- 团队无法运营带 per-role scaling 的两个 GPU pools：Dynamo 有帮助，但不是零成本。
- 没有 RDMA fabric：TCP transfer tax 更重。

### Router 与阶段 17 · 11 集成

Disaggregated routers 是 KV-cache-aware（阶段 17 · 11）。请求落到持有其 prefix 的 decode pool 上；如果没有 match，就走 prefill → decode。Hit rate 与 disaggregation 会叠加：cache-aware router 决定是否需要新的 prefill。

### MoE on Blackwell 才是真正数字所在

GB300 NVL72 + Dynamo 显示相对 Hopper baselines 的 50x MoE throughput。MoE expert routing 在 prefill 上 compute-heavy，在 decode 上 memory-heavy（expert caches），所以 disaggregation 是双重收益。2026 年 frontier model serving 以 MoE 为主导（DeepSeek-V3、未来 GPT-5 variants）。

### 你应该记住的数字

Benchmark numbers 会 drift：NVIDIA 和 inference stack 每季度发布更新结果。引用前重新检查。

- GB200 NVL72 + Dynamo 上的 DeepSeek-R1：medium-latency regime 中相对 baseline ~6x throughput（developer.nvidia.com，2025-06）；full Blackwell + Dynamo stacks 的 community “up to 30x” claims 是缺少单一 primary source 的 directional aggregates。
- GB300 NVL72 + Dynamo：相对 Hopper 最高 50x MoE throughput（developer.nvidia.com，未标日期）。
- Savings anchor（internal composite，不是单一 case study）：$2M annual spend、SLA 不变下每年省 $600-800K。
- Disaggregation threshold：prompts >512 tokens + outputs >200 tokens。
- NIXL KV transfer：70B FP8 上 4K-prompt KV 为 20-80 ms。

## 使用它

`code/main.py` 模拟 colocated vs disaggregated serving。报告 throughput、cost per request 和 prompt-length crossover。

## 交付它

本课会产出 `outputs/skill-disaggregation-decider.md`。给定 workload 和 cluster，它会判断是否 disaggregate。

## 练习

1. 运行 `code/main.py`。prompt length 到多少时 disaggregation 胜过 colocation？
2. 为一个 P99 prefix length 8K、output 300 的 RAG service 设计 prefill pool 和 decode pool。
3. Dynamo vs llm-d：为一个 pure-Kubernetes shop（无 Python runtime preference）选择一个。
4. 计算 KV transfer cost：70B FP8 上 4K prefill = ~500 MB KV。RDMA 100 GB/s 下 transfer = 5 ms。TCP 10 GB/s = 50 ms。哪个影响你的 SLA？
5. MoE expert routing 改变 KV access patterns。对于每个 token 激活不同 experts 的 MoE，disaggregation 表现如何？

## 关键词汇

| 术语 | 大家常说 | 实际含义 |
|------|----------|----------|
| Disaggregated serving | “split prefill/decode” | 每个 phase 使用独立 GPU pools |
| NIXL | “NVIDIA transport” | Dynamo 的 inter-node KV transfer（RDMA/TCP） |
| NVIDIA Dynamo | “the orchestrator” | vLLM/SGLang/TRT-LLM 的 stack-above coordinator |
| llm-d | “Kubernetes native” | Red Hat + AWS K8s disaggregated stack |
| Planner Profiler | “Dynamo auto-config” | 测量 workload，配置 pool ratios |
| SLA Planner | “Dynamo policy” | 自动匹配 prefill:decode rate 以满足 SLO |
| `packDomain: rack` | “llm-d topology” | 把 prefill+decode 打包到同 rack，加快 KV |
| UCCL | “unified collective” | llm-d 0.5 的 scale-to-zero networking layer |
| MoE expert routing | “expert per token” | DeepSeek-V3 pattern；disaggregation 有帮助 |

## 延伸阅读

- [NVIDIA — Introducing Dynamo](https://developer.nvidia.com/blog/introducing-nvidia-dynamo-a-low-latency-distributed-inference-framework-for-scaling-reasoning-ai-models/)
- [NVIDIA — Disaggregated LLM Inference on Kubernetes](https://developer.nvidia.com/blog/deploying-disaggregated-llm-inference-workloads-on-kubernetes/)
- [TensorRT-LLM Disaggregated Serving blog](https://nvidia.github.io/TensorRT-LLM/blogs/tech_blog/blog5_Disaggregated_Serving_in_TensorRT-LLM.html)
- [llm-d GitHub](https://github.com/llm-d/llm-d)
- [llm-d 0.5 release notes](https://github.com/llm-d/llm-d/releases)
