# Production Quantization — AWQ、GPTQ、GGUF K-quants、FP8、MXFP4/NVFP4

> Quantization format 不是通用选择，而是 hardware、serving engine 和 workload 的函数。GGUF Q4_K_M 或 Q5_K_M 主导 CPU 和 edge，通过 llama.cpp 和 Ollama 交付。GPTQ 在 vLLM 内需要同一 base 上 multi-LoRA 时胜出。AWQ 搭配 Marlin-AWQ kernels，在 7B class model 上达到约 741 tok/s，并在 INT4 中有最佳 Pass@1，是 2026 年 datacenter production 默认选择。FP8 在 Hopper、Ada 和 Blackwell 上保持中间地带：近乎无损、广泛支持。NVFP4 和 MXFP4（Blackwell microscaling）很激进，需要 per-block validation。两个陷阱常咬到团队：calibration dataset 必须匹配 deployment domain；KV cache 与 weight quantization 是分离的，AWQ 课上的“我的模型现在 4 GB”忘了 production batch sizes 下 10-30 GB 的 KV cache。

**类型：** 学习
**语言：** Python（stdlib，玩具版跨格式 memory 和 throughput 比较）
**前置要求：** 阶段 10 · 13（Quantization foundations），阶段 17 · 04（vLLM Serving Internals）
**时间：** ~75 分钟

## 学习目标

- 说出六种 production quantization formats 以及它们在 2026 年的 sweet spots。
- 给定 hardware（CPU vs GPU，Hopper vs Blackwell）、engine（vLLM、TRT-LLM、llama.cpp）和 workload（routine chat、reasoning、multi-LoRA）选择格式。
- 计算所选格式节省的 weight memory，以及不受影响的 KV cache。
- 说出会让 quantized models 在 domain traffic 上退化的 calibration-dataset pitfall。

## 问题

Quantization 降低 memory 和 HBM bandwidth，而这正是 decode 需要的。一个 FP16 70B model 是 140 GB weights。把 weights quantize 到 INT4（AWQ 或 GPTQ）后，模型是 35 GB，可以放进一张 H100，还给 KV cache 留出空间；这很重要，因为 128 concurrent sequences、2k context 时，KV cache 单独就是 20-30 GB。

但 quantization 不是免费的。激进 quantization 会降低质量，尤其是 reasoning-heavy tasks。不同格式适配不同 engines。不同硬件原生支持不同 precision。2026 年的 format zoo 是真实存在的，不能复制别人的选择，必须基于你的 stack 选择。

## 概念

### 六种格式

| Format | Bits | Sweet spot | Engines |
|--------|------|-----------|---------|
| GGUF Q4_K_M / Q5_K_M | 4-5 | CPU, edge, laptops | llama.cpp, Ollama |
| GPTQ | 4-8 | Multi-LoRA on vLLM | vLLM, TGI |
| AWQ | 4 | Datacenter GPU production | vLLM (Marlin-AWQ), TGI |
| FP8 | 8 | Hopper/Ada/Blackwell datacenter | vLLM, TRT-LLM, SGLang |
| MXFP4 | 4 | Blackwell multi-user | TRT-LLM |
| NVFP4 | 4 | Blackwell multi-user | TRT-LLM |

### GGUF — CPU/edge 默认选择

GGUF 是 file format，不完全是 quantization scheme 本身。它把 K-quant variants（Q2_K、Q3_K_M、Q4_K_M、Q5_K_M、Q6_K、Q8_0）打包到一个 container 中。Q4_K_M 和 Q5_K_M 是 production defaults：4-5 bits 下接近 BF16 quality。它是 CPU 或 edge serving 的最佳选择，因为 llama.cpp 到目前为止是最快的 CPU inference engine。

在 vLLM 中的 throughput penalty：7B 上约 93 tok/s。这个格式没有为 GPU kernels 优化。deployment target 是 CPU/edge 时用 GGUF。否则不要用。

### GPTQ — vLLM 中的 multi-LoRA

GPTQ 是一种 post-training quantization algorithm，带 calibration pass。Marlin kernels 让它在 GPU 上很快（相比 non-Marlin GPTQ 2.6x speedup）。7B 上约 712 tok/s。

独特优势：GPTQ-Int4 在 vLLM 中支持 LoRA adapters。如果你要服务一个 base model 加 10-50 个 fine-tuned variants（每个都是一个 LoRA），GPTQ 是你的路径。截至 2026 年初，NVFP4 还不支持 LoRA。

### AWQ — datacenter GPU 默认选择

Activation-aware Weight Quantization。它在 quantization 期间保护约 1% 最显著的 weights。Marlin-AWQ kernels：相比 naive 提升 10.9x。7B 上约 741 tok/s，是 INT4 格式中 Pass@1 最好的。

新的 GPU serving 默认选择 AWQ，除非你需要 multi-LoRA（GPTQ）或激进 Blackwell FP4（NVFP4）。

### FP8 — 可靠中间地带

8-bit floating point。近乎无损。广泛支持。Hopper Tensor Cores 原生加速 FP8。Blackwell 继承。质量不可妥协时（reasoning、medical、code-gen），FP8 是 2026 年安全默认选择。Memory savings 是 INT4 的一半，但质量风险低得多。

### MXFP4 / NVFP4 — Blackwell aggressive

Microscaling FP4。每个 weights block 有自己的 scale factor。很激进，但在 Blackwell Tensor Cores 上硬件加速。相比 FP8，每 token bytes 减半，这是阶段 17 · 07 中的经济收益。

Caveats：
- 尚不支持 LoRA（2026 年初）。
- Reasoning-heavy workloads 上质量下降可见。
- 每个模型都要在自己的 eval set 上验证。

### Calibration 陷阱

AWQ 和 GPTQ 需要 calibration dataset，通常是 C4 或 WikiText。对于 domain models（code、medical、legal），用通用 web text 做 calibration 会让算法错误判断哪些 weights 该被保护。HumanEval 上的 Pass@1 可能掉好几个点。

修复：用 in-domain data 做 calibration。通常几百个 domain samples 就够。ship 前在 eval set 上测试。

### KV cache 陷阱

AWQ 把 weights 缩到 4 bits。KV cache 是分开的，并保持 FP16/FP8。对于 70B model + AWQ：

- Weights：~35 GB（从 140 GB INT4）。
- 128 concurrent × 2k context 的 KV cache：~20 GB。
- Activations：~5 GB。
- Total：~60 GB，可以放进 H100 80GB。

天真地说“我把模型 quantize 到 4 GB 了”会忘记另外 30-50 GB。要整体预算 HBM。

另外，KV cache quantization（FP8 KV 或 INT8 KV）是另一项选择，有自己的取舍：它直接影响 attention accuracy，并不是免费收益。

### AWQ INT4 对 reasoning 有风险

Chain-of-thought、math、long context code-gen 这些任务会明显受 aggressive quantization 影响。AWQ INT4 在 MATH 上会损失约 3-5 points。对于 reasoning-heavy workloads，交付 FP8 或 BF16，接受 memory cost。

### 2026 选择指南

- CPU/edge serve：GGUF Q4_K_M。完事。
- GPU serve、routine chat、无 LoRA：AWQ。
- GPU serve、multi-LoRA：GPTQ with Marlin。
- Reasoning workload：FP8。
- Blackwell datacenter、质量已验证：NVFP4 + FP8 KV。
- 模糊不清：对每个 candidate format 跑 1,000-sample eval。

## 使用它

`code/main.py` 对一系列 model sizes 和六种格式计算 memory footprint（weights + KV + activations）与 relative throughput。它会展示 KV cache 何时主导、weight compression 何时划算、FP8 何时是安全选择。

## 交付它

本课会产出 `outputs/skill-quantization-picker.md`。给定 hardware、model size、workload type 和 quality tolerance，它会选择格式并生成 calibration/validation plan。

## 练习

1. 运行 `code/main.py`。对于 70B model、128 concurrent、2k context，计算每种格式的 total HBM。哪种格式能让你放进一张 H100 80GB？
2. 你有一个 7B coding model。选择一个格式并说明理由。如果你对 quality tolerance 判断错了，恢复路径是什么？
3. 计算给 medical domain model 校准 AWQ 所需的 calibration-dataset size。为什么数据越多不一定越好？
4. 阅读 Marlin-AWQ kernel paper 或 release notes。用三句话解释为什么 AWQ 在 7B 上达到 741 tok/s，而 raw GPTQ 约 712。
5. 什么时候组合 AWQ weights + FP8 KV cache 有意义，什么时候应该让 KV 保持 BF16？

## 关键词汇

| 术语 | 大家常说 | 实际含义 |
|------|----------|----------|
| GGUF | “llama.cpp format” | 打包 K-quant variants 的 file format；CPU/edge 默认 |
| Q4_K_M | “Q4 K M” | 4-bit K-quant medium；production GGUF 默认 |
| GPTQ | “gee pee tee q” | 带 calibration 的 post-train INT4；在 vLLM 中支持 LoRA |
| AWQ | “a w q” | Activation-aware INT4；Marlin kernels；INT4 下最佳 Pass@1 |
| Marlin kernels | “fast INT4 kernels” | Hopper 上用于 INT4 的 custom CUDA kernels；10x speedup |
| FP8 | “eight-bit float” | Hopper/Ada/Blackwell 上的安全 precision 默认 |
| MXFP4 / NVFP4 | “microscaling four” | Blackwell 4-bit FP，带 per-block scale factors |
| Calibration dataset | “cal data” | 用于选择 quantization parameters 的输入文本；必须匹配 domain |
| KV cache quantization | “KV INT8” | 与 weights 分开的选择；影响 attention accuracy |

## 延伸阅读

- [VRLA Tech — LLM Quantization 2026](https://vrlatech.com/llm-quantization-explained-int4-int8-fp8-awq-and-gptq-in-2026/) — comparative benchmarks。
- [Jarvis Labs — vLLM Quantization Complete Guide](https://jarvislabs.ai/blog/vllm-quantization-complete-guide-benchmarks) — 各格式 throughput numbers。
- [PremAI — GGUF vs AWQ vs GPTQ vs bitsandbytes 2026](https://blog.premai.io/llm-quantization-guide-gguf-vs-awq-vs-gptq-vs-bitsandbytes-compared-2026/) — format-by-format picking。
- [vLLM docs — Quantization](https://docs.vllm.ai/en/latest/features/quantization/index.html) — supported formats 和 flags。
- [AWQ paper (arXiv:2306.00978)](https://arxiv.org/abs/2306.00978) — original AWQ formulation。
- [GPTQ paper (arXiv:2210.17323)](https://arxiv.org/abs/2210.17323) — original GPTQ formulation。
