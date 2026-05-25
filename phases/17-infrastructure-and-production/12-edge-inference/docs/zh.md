# Edge Inference — Apple Neural Engine、Qualcomm Hexagon、WebGPU/WebLLM、Jetson

> Edge 的核心约束是 memory bandwidth，不是 compute。Mobile DRAM 位于 50-90 GB/s；datacenter HBM3 超过 2-3 TB/s，是 30-50x 差距。Decode 是 memory-bound，所以差距是决定性的。2026 年格局分四路。Apple M4/A18 Neural Engine 峰值 38 TOPS，使用 unified memory（没有 CPU↔NPU copy）。Qualcomm Snapdragon X Elite / 8 Gen 4 Hexagon 达到 45 TOPS。WebGPU + WebLLM 在 M3 Max 上以约 41 tok/s 运行 Llama 3.1 8B（Q4），约为 native 的 70-80%；17.6k GitHub stars，OpenAI-compatible API，约 70-75% mobile coverage。NVIDIA Jetson Orin Nano Super（8GB）可放 Llama 3.2 3B / Phi-3；AGX Orin 通过 vLLM 以约 40 tok/s 运行 gpt-oss-20b；Jetson T4000（JetPack 7.1）性能是 AGX Orin 的 2x。TensorRT Edge-LLM 支持 EAGLE-3、NVFP4、chunked prefill，已在 CES 2026 由 Bosch、ThunderSoft、MediaTek 展示。

**类型：** 学习
**语言：** Python（stdlib，玩具版 bandwidth-bound decode simulator）
**前置要求：** 阶段 17 · 04（vLLM Serving Internals），阶段 17 · 09（Production Quantization）
**时间：** ~60 分钟

## 学习目标

- 解释为什么 mobile LLM inference 是 memory-bandwidth-bound，compute 是次要因素。
- 枚举四种 edge targets（Apple ANE、Qualcomm Hexagon、WebGPU/WebLLM、NVIDIA Jetson），并把每种匹配到 use case。
- 说出 2026 年 WebGPU coverage gap（Firefox Android 追赶中）以及 Safari iOS 26 landing。
- 为每个 target 选择 quantization format（ANE 用 Core ML INT4 + FP16，Hexagon 用 QNN INT8/INT4，browser 用 WebGPU Q4，Jetson Thor 用 NVFP4）。

## 问题

一个客户想要 on-device chatbot：voice-first、private-by-default、离线可用。在 MacBook Pro M3 Max 上，Llama 3.1 8B Q4 跑约 55 tok/s，没问题。在 iPhone 16 Pro 上，同一模型跑 3 tok/s，不行。在中端 Android + Snapdragon 8 Gen 3 上是 7 tok/s。通过 Chrome Android v121+ 上的 WebGPU 在浏览器中跑，根据设备为 4-8 tok/s。

Throughput variance 不是 porting issue。它是 bandwidth gap 乘以 quantization format，再乘以 NPU 是否能从 user-space 访问。2026 年 edge inference 是四个不同问题，对应四种不同解法。

## 概念

### Bandwidth 才是真正天花板

Decode 为每个 token 读取完整 weights。一个 Q4 的 7B model 是 3.5 GB。以 50 GB/s 读取 3.5 GB 需要 70 ms，理论上限约 14 tok/s。90 GB/s（高端 mobile DRAM）时天花板移动到约 25 tok/s。在这个数值以下，再多 compute 也帮不上忙。

Datacenter HBM3 以 3 TB/s 读取同样 3.5 GB 只需 1.2 ms，天花板是 830 tok/s。同一模型，同一 weights。不同 memory subsystem。

### Apple Neural Engine（M4 / A18）

- 最高 38 TOPS。Unified memory（CPU 和 ANE 共享同一 pool），无 copy overhead。
- 通过 Core ML + 编译后的 `.mlmodel` models 访问，或通过 PyTorch 的 Metal Performance Shaders（MPS）访问。
- Llama.cpp Metal backend 使用 MPS，不直接使用 ANE；原生 ANE 需要 Core ML conversion。
- 2026 年 iOS apps 最实用路径：Core ML + INT4 weights + FP16 activations。

### Qualcomm Hexagon（Snapdragon X Elite / 8 Gen 4）

- 最高 45 TOPS。与 SoC 中的 CPU 和 GPU 集成，但拥有独立 memory domain。
- QNN（Qualcomm Neural Network）SDK 和 AI Hub 提供从 PyTorch/ONNX 的 conversion。
- Chat templates、Llama 3.2、Phi-3 都作为 AI Hub first-class artifacts 发布。

### Intel / AMD NPUs（Lunar Lake、Ryzen AI 300）

- 40-50 TOPS。Software 落后于 Apple/Qualcomm；OpenVINO 在进步，但仍偏 niche。
- 最适合 Windows ARM copilot apps；也适合 AMD/Intel desktops 上的 local-first native。

### WebGPU + WebLLM

- 通过 WebGPU compute shaders 在浏览器中运行模型；无需安装。
- M3 Max 上 Llama 3.1 8B Q4 约 41 tok/s，使用同一 backend 时约为 native 的 70-80%。
- WebLLM 有 17.6k GitHub stars；OpenAI-compatible JS API；Apache 2.0。
- 2026 coverage：Chrome Android v121+、Safari iOS 26 GA，Firefox Android 仍在追赶。整体约 70-75% mobile coverage。

### NVIDIA Jetson family

- Orin Nano Super（8GB）：可放 Llama 3.2 3B、Phi-3，并有不错 tok/s。
- AGX Orin：通过 vLLM 以约 40 tok/s 运行 gpt-oss-20b。
- Thor / T4000（JetPack 7.1）：性能是 AGX Orin 的 2x，支持 EAGLE-3 和 NVFP4。
- TensorRT Edge-LLM（2026）支持 EAGLE-3 speculative decoding、NVFP4 weights、chunked prefill：datacenter optimizations 被移植到 edge。

### 每个 target 的 quantization 选择

| Target | Format | Notes |
|--------|--------|-------|
| Apple ANE | INT4 weights + FP16 activations | Core ML conversion path |
| Qualcomm Hexagon | QNN INT8 / INT4 | AI Hub converters |
| WebGPU / WebLLM | Q4 MLC (q4f16_1) | Use `mlc_llm convert_weight` + compiled `.wasm`; GGUF is not supported |
| Jetson Orin Nano | Q4 GGUF or TRT-LLM INT4 | Memory-bound |
| Jetson AGX / Thor | NVFP4 + FP8 KV | Edge-LLM path |

### Edge 上的 long-context 陷阱

Llama 3.1 的 128K context 是 datacenter feature。在 8 GB RAM 的手机上，4 GB model + 32K tokens 的 2 GB KV cache + OS overhead = OOM。Edge deployments 会把 context 保持在 4K-8K，除非接受激进 KV quantization（Q4 KV）。

### Voice 是 killer app

Voice agents 对 latency 敏感（first token < 500 ms）。Local inference 完全消除 network latency。与 speech-to-text（Whisper Turbo variants 可在 edge 运行）结合后，edge inference 成为 production-quality voice loop。

### 你应该记住的数字

- Apple M4 / A18 ANE：38 TOPS。
- Qualcomm Hexagon SD X Elite：45 TOPS。
- WebLLM M3 Max：Llama 3.1 8B Q4 上 ~41 tok/s。
- AGX Orin：通过 vLLM 在 gpt-oss-20b 上 ~40 tok/s。
- Datacenter-edge bandwidth gap：30-50x。
- WebGPU mobile coverage：~70-75%（Firefox Android 落后）。

## 使用它

`code/main.py` 从 bandwidth-bound 数学计算各 edge targets 的理论 decode throughput ceilings。它会与 observed benchmarks 比较，并突出 bottleneck 是 bandwidth，而不是 compute。

## 交付它

本课会产出 `outputs/skill-edge-target-picker.md`。给定 platform（iOS/Android/browser/Jetson）、model 和 latency/memory budget，它会选择 quantization format 和 conversion pipeline。

## 练习

1. 运行 `code/main.py`。对于 Snapdragon 8 Gen 3（约 77 GB/s bandwidth）上的 Q4 7B model，计算 decode ceiling。与 observed 6-8 tok/s 比较：runtime 高效吗？
2. Android 上 WebGPU 需要 Chrome v121+。为更旧浏览器设计 fallback：server-side via the same OpenAI-compatible API。
3. 你的 iOS app 需要 4K-context streaming。哪种 model/format combination 能让 iPhone 16 上 active memory 保持在 4 GB 以下？
4. Jetson AGX Orin 以 40 tok/s 运行 gpt-oss-20b。Jetson Nano 只能放 3B。如果产品同时面向二者，如何统一 inference stack？
5. 论证 “WebLLM is production-ready in 2026”。引用 coverage、performance 和 Firefox Android gap。

## 关键词汇

| 术语 | 大家常说 | 实际含义 |
|------|----------|----------|
| ANE | “Apple neural engine” | M-series 和 A-series 中的 on-device NPU；unified memory |
| Hexagon | “Qualcomm NPU” | Snapdragon NPU；通过 QNN SDK 访问 |
| WebGPU | “browser GPU” | W3C-standardized browser GPU API；Chrome/Safari 2026 |
| WebLLM | “browser LLM runtime” | MLC-LLM project；Apache 2.0；OpenAI-compatible JS |
| Jetson | “NVIDIA edge” | Orin Nano / AGX / Thor / T4000 family |
| TRT Edge-LLM | “edge TensorRT” | TensorRT-LLM 的 2026 edge port；EAGLE-3 + NVFP4 |
| Unified memory | “shared pool” | CPU 和 NPU 看到同一 RAM；无 copy overhead |
| Bandwidth-bound | “memory limited” | Decode 受读取 weights 的 bytes/sec 限制 |
| Core ML | “Apple conversion” | Apple 的 ANE-native models framework |
| QNN | “Qualcomm stack” | Qualcomm Neural Network SDK |

## 延伸阅读

- [On-Device LLMs State of the Union 2026](https://v-chandra.github.io/on-device-llms/) — landscape 和 benchmarks。
- [NVIDIA Jetson Edge AI](https://developer.nvidia.com/blog/getting-started-with-edge-ai-on-nvidia-jetson-llms-vlms-and-foundation-models-for-robotics/) — Orin / AGX / Thor。
- [NVIDIA TensorRT Edge-LLM](https://developer.nvidia.com/blog/accelerating-llm-and-vlm-inference-for-automotive-and-robotics-with-nvidia-tensorrt-edge-llm/) — 2026 edge port announcement。
- [WebLLM (arXiv:2412.15803)](https://arxiv.org/html/2412.15803v2) — design 和 benchmarks。
- [Apple Core ML](https://developer.apple.com/documentation/coreml) — ANE-native conversion。
- [Qualcomm AI Hub](https://aihub.qualcomm.com/) — 面向 Hexagon 的 pre-converted models。
