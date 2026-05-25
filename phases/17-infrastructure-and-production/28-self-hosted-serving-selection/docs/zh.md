# Self-Hosted Serving Selection — llama.cpp、Ollama、TGI、vLLM、SGLang

> 2026 年 self-hosted inference 由四个 engines 主导。根据 hardware、scale 和 ecosystem 选择。**llama.cpp** 在 CPU 上最快：model support 最广，对 quantization 和 threading 有完全控制。**Ollama** 是 dev-laptop 的一条命令安装，比 llama.cpp 慢约 15-30%（Go + CGo + HTTP serialization），prod-like load 下 throughput gap 为 3x。**TGI 在 2025 年 12 月 11 日进入 maintenance mode**：只修 bug，raw throughput 比 vLLM 慢约 10%，但历史上拥有顶级 observability 和 HF-ecosystem integration。这个 maintenance status 让它成为有风险的长期赌注：新项目默认选 SGLang 或 vLLM 更安全。**vLLM** 是 general-purpose production default：v0.15.1（2026 年 2 月）增加 PyTorch 2.10、RTX Blackwell SM120、H200 optimization。**SGLang** 是 agentic multi-turn / prefix-heavy specialist：production 中 400,000+ GPUs（xAI、LinkedIn、Cursor、Oracle、GCP、Azure、AWS）。Hardware constraints：CPU-only → 只能 llama.cpp。AMD / non-NVIDIA → vLLM only（TRT-LLM 是 NVIDIA-locked）。2026 pipeline pattern：dev = Ollama，staging = llama.cpp，prod = vLLM 或 SGLang。全程使用同一 GGUF/HF weights。

**类型：** 学习
**语言：** Python（stdlib，engine-decision tree walker）
**前置要求：** 所有覆盖 engines 的阶段 17 课程（04、06、07、09、18）
**时间：** ~45 分钟

## 学习目标

- 给定 hardware（CPU / AMD / NVIDIA Hopper / Blackwell）、scale（1 user / 100 / 10,000）和 workload（general chat / agent / long-context）选择 engine。
- 说出 2026 年 TGI maintenance-mode status（2025 年 12 月 11 日），以及它为什么让新项目偏向 vLLM 或 SGLang。
- 描述 dev/staging/prod pipeline：全程使用同一 GGUF 或 HF weights。
- 解释为什么 “CPU only” 强制使用 llama.cpp，而 “AMD” 排除 TRT-LLM。

## 问题

你的团队开始一个新的 self-hosted LLM project。一个工程师说 Ollama，另一个说 vLLM，第三个说“不是 TGI out of the box 就能工作吗？”三者在不同 context 下都对。但没有一个对所有场景都对。

2026 年 choice tree 很重要：hardware first，scale second，workload third。并且一个具体的 2025 事件：TGI 在 12 月 11 日进入 maintenance mode，会改变新项目的默认选择。

## 概念

### 五个 engines

| Engine | Best for | Notes |
|--------|----------|-------|
| **llama.cpp** | CPU / edge / minimal deps / widest model support | Fastest on CPU, full control |
| **Ollama** | Dev laptops, single user, one-command install | 15-30% slower than llama.cpp; 3x prod throughput gap |
| **TGI** | HF ecosystem, regulated industries | **Maintenance mode Dec 11, 2025** |
| **vLLM** | General-purpose production, 100+ users | Broad production default; v0.15.1 Feb 2026 |
| **SGLang** | Agentic multi-turn, prefix-heavy workloads | 400,000+ GPUs in production |

### Hardware-first decision

**CPU only** → llama.cpp。Ollama 也能工作，但更慢。没有其他 engine 在 CPU 上有竞争力。

**AMD GPU** → vLLM（AMD ROCm support）。SGLang 也能工作。TRT-LLM 是 NVIDIA-locked，所以排除。

**NVIDIA Hopper（H100 / H200）** → vLLM、SGLang 或 TRT-LLM。三者都是 top-tier。

**NVIDIA Blackwell（B200 / GB200）** → TRT-LLM 是 throughput leader（阶段 17 · 07）。vLLM 和 SGLang 紧随其后。

**Apple Silicon（M-series）** → llama.cpp（Metal）。Ollama 包了一层。

### Scale-second decision

**1 user / local dev** → Ollama。一条命令，first-token in seconds。

**10-100 users / small team** → vLLM single-GPU。

**100-10k users / production** → vLLM production-stack（阶段 17 · 18）或 SGLang。

**10k+ users / enterprise** → vLLM production-stack + disaggregated（阶段 17 · 17）+ LMCache（阶段 17 · 18）。

### Workload-third decision

**General chat / Q&A** → vLLM 在 broad default 上胜出。

**Agentic multi-turn（tools、planning、memory）** → SGLang 的 RadixAttention（阶段 17 · 06）占优。

**RAG with heavy prefix reuse** → SGLang。

**Code generation** → vLLM 可以；SGLang 在 cache 上略好。

**Long context（128K+）** → vLLM + chunked prefill；SGLang + tiered KV。

### TGI maintenance trap

Hugging Face TGI 于 2025 年 12 月 11 日进入 maintenance mode：之后只修 bug。历史上：顶级 observability、best-in-class HF-ecosystem integration（model cards、safety tools），raw throughput 稍落后 vLLM。

对 2026 年新项目：默认避开 TGI。现有 TGI deployments 可以继续，但最终应该迁移。SGLang 和 vLLM 是更安全默认值。

### Pipeline pattern

Dev（Ollama）→ staging（llama.cpp）→ prod（vLLM）。全程使用同一 GGUF 或 HF weights。工程师在 laptops 上快速迭代；staging 镜像 production quantization；prod 是 serving target。

### Ollama caveat

Ollama 非常适合 dev。不太适合 shared production：Go HTTP serialization 增加 overhead，concurrency management 比 vLLM 简单，OpenTelemetry support 滞后。让 Ollama 用在它擅长的地方：one user、one command；shared 场景切换到 vLLM。

### Self-hosted vs managed 是另一个决策

阶段 17 · 01（managed hyperscalers）和 · 02（inference platforms）覆盖 managed。本课假设你已经决定 self-host。选择 self-host 的原因：data residency、custom fine-tune、scale 下 total cost ownership、hosted 上没有的 domain model。

### 你应该记住的数字

- TGI maintenance mode：2025 年 12 月 11 日。
- vLLM v0.15.1：2026 年 2 月；PyTorch 2.10；Blackwell SM120 support。
- SGLang production footprint：400,000+ GPUs。
- Ollama throughput gap vs llama.cpp：慢 15-30%；prod load 下 3x。

## 使用它

`code/main.py` 是一个 decision-tree walker：给定 hardware + scale + workload，它会选择 engine 并解释原因。

## 交付它

本课会产出 `outputs/skill-engine-picker.md`。给定 constraints，它会选择 engine 并写出 migration plan。

## 练习

1. 用你的 hardware / scale / workload 运行 `code/main.py`。输出符合直觉吗？
2. 你的 infra 是 12 张 H100 和 8 张 MI300X AMD。选什么 engine？为什么 TRT-LLM 被排除？
3. 一个团队想在 2026 年使用 TGI，因为“这是我们熟悉的”。论证 migration case。
4. Ollama dev 到 vLLM prod：quantization、configuration 和 observability 有什么变化？
5. 一个 RAG product，P99 prefix length 8K，tenants 间 high reuse。选择 engine，并叠加阶段 17 · 11 + 18。

## 关键词汇

| 术语 | 大家常说 | 实际含义 |
|------|----------|----------|
| llama.cpp | “the CPU one” | model support 最广，CPU 上最快 |
| Ollama | “the laptop one” | One-command install，dev-grade throughput |
| TGI | “HF's serving” | 自 2025 年 12 月起 maintenance mode |
| vLLM | “the default” | 2026 broad production baseline |
| SGLang | “the agentic one” | Prefix-heavy，RadixAttention |
| TRT-LLM | “NVIDIA-locked” | Blackwell throughput leader，仅 NVIDIA |
| GGUF | “llama.cpp format” | 打包 K-quant variants |
| Production-stack | “vLLM K8s” | 阶段 17 · 18 reference deployment |
| Pipeline pattern | “dev→stage→prod” | 同一 weights 上 Ollama → llama.cpp → vLLM |

## 延伸阅读

- [AI Made Tools — vLLM vs Ollama vs llama.cpp vs TGI 2026](https://www.aimadetools.com/blog/vllm-vs-ollama-vs-llamacpp-vs-tgi/)
- [Morph — llama.cpp vs Ollama 2026](https://www.morphllm.com/comparisons/llama-cpp-vs-ollama)
- [n1n.ai — Comprehensive LLM Inference Engine Comparison](https://explore.n1n.ai/blog/llm-inference-engine-comparison-vllm-tgi-tensorrt-sglang-2026-03-13)
- [PremAI — 10 Best vLLM Alternatives 2026](https://blog.premai.io/10-best-vllm-alternatives-for-llm-inference-in-production-2026/)
- [TGI maintenance announcement](https://github.com/huggingface/text-generation-inference) — release notes。
- [vLLM v0.15.1 release notes](https://github.com/vllm-project/vllm/releases)
