# Blackwell 上使用 FP8 和 NVFP4 的 TensorRT-LLM

> TensorRT-LLM 是 NVIDIA-only，但它在 Blackwell 上获胜。SemiAnalysis InferenceX 在 2026 年 Q1-Q2 测得：GB200 NVL72 + Dynamo orchestration 上，120B model 成本为每百万 tokens $0.012；H100 + vLLM 为 $0.09/M，经济差距 7x。这个 stack 由三种 floating-point regimes 叠加而成：FP8 对 KV cache 和 attention kernels 仍然关键，因为它们需要动态范围；NVFP4（4-bit microscaling）处理 weights 和 activations；multi-token prediction（MTP）和 disaggregated prefill/decode 再叠加 2-3x。Day-0 model support 可以直接加载 FP4 weights，无需 post-training conversion。2026 年工程团队的代价是：TRT-LLM 是封闭 NVIDIA stack，所以采用它是在用 portability 换 throughput。承诺前先对你的模型和硬件组合算账。

**类型：** 学习
**语言：** Python（stdlib，玩具版 FP8/NVFP4 memory and cost calculator）
**前置要求：** 阶段 17 · 04（vLLM Serving Internals），阶段 10 · 13（Quantization）
**时间：** ~75 分钟

## 学习目标

- 解释为什么即使 weights 使用 NVFP4，FP8 对 KV cache 和 attention 仍然关键。
- 计算 frontier model 在 BF16、FP8 和 NVFP4 下的 HBM footprint，并推理节省来自哪里。
- 说出 TRT-LLM 利用的 Blackwell-specific features（day-0 FP4、MTP、disaggregated serving、all-to-all primitives）。
- 判断 TRT-LLM 的 NVIDIA-lock 什么时候值得为它相对 Hopper 上 vLLM 的 7x cost gap 付出。

## 问题

2026 年 inference economics 的 frontier 是“每美元多少 tokens”。答案取决于四个叠加选择：hardware generation（Hopper H100/H200 vs Blackwell B200/GB200）、precision（BF16 → FP8 → NVFP4）、serving engine（vLLM vs SGLang vs TRT-LLM）和 orchestration（plain vs disaggregated vs Dynamo）。

在 Hopper + vLLM 上，一个 120B MoE 每百万 tokens 成本约 $0.09。在 Blackwell + TRT-LLM + Dynamo 上，同一模型约 $0.012，便宜 7x。其中一部分差距来自硬件（Blackwell 的 per-GPU LLM throughput 比 Hopper 高 11-15x）。另一部分来自 stack：FP4 weights、MTP draft、disaggregated prefill/decode，以及用于 MoE expert communication 的 NVLink 5 all-to-all。

你无法在 NVIDIA stack 之外复刻它。这就是取舍：portability 换 economics。理解哪些 stack choices 贡献了哪一部分差距，是本课重点。

## 概念

### 为什么 FP8 仍是 KV cache 的底线

2026 年常见错误：以为 NVFP4 可以应用在所有地方。它不行。KV cache 需要 FP8（8-bit floating point），因为它存储 attention keys and values，动态范围很宽。把 KV quantize 到 FP4 会导致灾难性 accuracy loss：distribution tail 掉落，attention scores 崩溃。FP8 的 exponent bits 给 KV cache 所需的范围。

NVFP4（2025-2026）适用于 weights 和 activations。Microscaling：每个 weights block 有自己的 scale factor，因此小 blocks 可以覆盖不同 dynamic ranges，而不会遭受 per-tensor scale loss。对于 activations，FP4 能撑住，因为 layer 内 activations 范围较小。

典型 Blackwell config：

- Weights：NVFP4（4-bit microscaling）。
- Activations：NVFP4。
- KV cache：FP8。
- Attention accumulator：FP32（softmax stability）。

### TRT-LLM 使用的 Blackwell-specific primitives

- **Day-0 FP4 weights**：model providers 直接发布 FP4 weights；TRT-LLM 加载时无需 post-training conversion。不需要 AWQ / GPTQ step 来得到 FP4。
- **Multi-token prediction（MTP）**：与 EAGLE（阶段 17 · 05）同一思路，但集成进 TRT-LLM build。
- **Disaggregated serving**：prefill 和 decode 分别在不同 GPU pools 上运行，KV cache 通过 NVLink 或 InfiniBand 传输。与 Dynamo（阶段 17 · 20）同一思路。
- **All-to-all communication primitives**：NVLink 5 将 MoE expert communication latency 相比 Hopper 降低 3x。TRT-LLM 的 MoE kernels 为此调优。
- **NVFP4 + MXFP8 microscaling**：Blackwell Tensor Cores 上硬件加速的 scale-factor handling。

### 你应该记住的数字

- HGX B200 通过 TRT-LLM 在 GPT-OSS-120B 上达到 $0.02/M tokens。
- GB200 NVL72 通过 Dynamo（编排 TRT-LLM）达到 $0.012/M tokens。
- H100 + vLLM 在可比 workload 上约 $0.09/M tokens。
- 2026 年三个月 TRT-LLM 更新带来 2.8x throughput gain。
- Blackwell vs Hopper：per-GPU LLM throughput 高 11-15x。
- MLPerf Inference v6.0（2026 年 4 月）：Blackwell 主导每个 submitted task。

### FP4 实际带来的质量成本

NVFP4 很激进。在 reasoning-heavy workloads（chain-of-thought、math、long context code-gen）上，FP4 weights 会出现可见退化。Per-block calibration 会缓解，但不能消除。交付 reasoning models 的团队常用 FP8 weights + FP4 activations 折中，或者在 H200 上全程 FP8。

规则：在承诺使用 NVFP4 weights 前，一定先在你的 eval set 上验证 task quality。

### 为什么这是 NVIDIA-lock 决策

TRT-LLM 是 C++ + CUDA + closed-source kernels。Models 需要为特定 GPU SKU 编译。没有 AMD、没有 Intel、没有 ARM。如果你的 infra strategy 是 multi-vendor，TRT-LLM 对被 TRT-LLM serving 的 tier 就不适用；你仍然可以在 mixed hardware 上用 vLLM 服务。如果你是 NVIDIA-only，7x 差距会为 lock 付费。

### 2026 实用 recipe

对于年 inference bill 超过 $100M 的场景，继续使用 Hopper + vLLM 等于把 7-10x 放在桌上。把 cost-dominant workloads 迁移到 Blackwell + TRT-LLM + Dynamo。把 experimentation tier 留在 H100 + vLLM 上，以便模型迭代更快。每个 NVFP4-converted model 进入 production 前都要验证质量。

### Disaggregation bonus

TRT-LLM 的 disaggregated serving（分离 prefill 和 decode pools）会在阶段 17 · 20 深入讲解。在 Blackwell 上，multiplier 会叠加：FP4 weights × MTP speedup × disaggregated placement × cache-aware routing。7x 数字假设使用完整 stack。

## 使用它

`code/main.py` 跨三个 stack 计算模型的 HBM footprint、decode throughput（memory-bound regime）和 $/M-tokens：H100 + BF16 + vLLM、H100 + FP8 + vLLM、B200 + NVFP4/FP8 + TRT-LLM。运行它，观察 compounding effect，以及每个变化贡献了多少差距。

## 交付它

本课会产出 `outputs/skill-trtllm-blackwell-advisor.md`。给定 workload、model size 和 annual token volume，它会判断 Blackwell + TRT-LLM stack 是否值得 NVIDIA-lock。

## 练习

1. 运行 `code/main.py`。对一个 active parameters 为 30% 的 120B MoE，计算 H100 BF16、H100 FP8 和 B200 NVFP4/FP8 上 memory-bandwidth-limited decode throughput。最大跃升来自哪里？
2. 一个客户每年在 H100 + vLLM 上花 $2M。给定 7x economic gap，他们需要购买多少 Blackwell GPUs，才能在 12 个月内摊销迁移到 TRT-LLM 的成本？
3. NVFP4 weight conversion 后，你在 MATH 上看到 accuracy 下降 3 points。说出两个恢复路径：一个 quality-first（保留 FP8 weights），一个 cost-first（用 in-domain data calibrate）。
4. 阅读 MLPerf v6.0 inference results。哪个 task 的 Blackwell-over-Hopper gap 最小，为什么？
5. 计算 405B model 在 NVFP4 weights + FP8 KV cache、128k context 下所需 HBM。它能放进单个 GB200 NVL72 node 吗？

## 关键词汇

| 术语 | 大家常说 | 实际含义 |
|------|----------|----------|
| FP8 | “eight-bit float” | 8-bit floating point；因 dynamic range 用于 KV cache 和 attention |
| NVFP4 | “four-bit micro” | NVIDIA 的 4-bit microscaling FP format；Blackwell 上用于 weights 和 activations |
| MXFP8 | “MX eight” | Microscaling FP8 variant；Blackwell Tensor Cores 硬件加速 |
| Day-0 FP4 | “ship FP4 weights” | Model providers 直接发布 FP4 weights；无 post-train conversion step |
| MTP | “multi-token prediction” | TRT-LLM 集成的 speculative-decoding draft（阶段 17 · 05） |
| Disaggregated serving | “split prefill/decode” | Prefill 和 decode 在不同 GPU pools；KV 通过 NVLink/IB 传输 |
| All-to-all | “MoE expert comm” | 将 tokens 路由到 expert GPUs 的通信模式；NVLink 5 降低 3x |
| InferenceX | “SemiAnalysis inference bench” | 2026 年行业接受的 cost-per-token benchmark |

## 延伸阅读

- [NVIDIA — Blackwell Ultra MLPerf Inference v6.0](https://developer.nvidia.com/blog/nvidia-blackwell-ultra-sets-new-inference-records-in-mlperf-debut/) — 2026 年 4 月 MLPerf results。
- [NVIDIA — MoE Inference on Blackwell](https://developer.nvidia.com/blog/delivering-massive-performance-leaps-for-mixture-of-experts-inference-on-nvidia-blackwell/) — NVLink 5 all-to-all 和 MoE kernels。
- [TensorRT-LLM Overview](https://nvidia.github.io/TensorRT-LLM/overview.html) — 官方 engine documentation。
- [NVIDIA — Introducing Dynamo](https://developer.nvidia.com/blog/introducing-nvidia-dynamo-a-low-latency-distributed-inference-framework-for-scaling-reasoning-ai-models/) — TRT-LLM 之上的 disaggregated orchestration。
- [MLPerf Inference](https://mlcommons.org/benchmarks/inference-datacenter/) — 发布 Blackwell 数字的 benchmark suite。
