# vLLM Serving Internals：PagedAttention、Continuous Batching、Chunked Prefill

> vLLM 在 2026 年的主导地位来自三个叠加的默认能力，而不是单一技巧。PagedAttention 总是开启。Continuous batching 会在 decode iterations 之间把新请求注入 active batch。Chunked prefill 会切分长 prompt，让 decode tokens 永远不被饿死。三者都打开后，单张 H100 SXM5 上的 Llama 3.3 70B FP8 在 128 concurrent 下可达到 2,200-2,400 tok/s，比 vLLM 自身默认值高约 25%，是 naive PyTorch loop 的 3-4x。本课会读 scheduler 和 attention kernel，读到你能画图解释的程度，并以 `code/main.py` 中的玩具 continuous batcher 结束，它像 vLLM 一样调度 prefill 和 decode。

**类型：** 学习
**语言：** Python（stdlib，玩具版 continuous batching scheduler）
**前置要求：** 阶段 17 · 01（Model Serving），阶段 11（LLM Engineering）
**时间：** ~75 分钟

## 学习目标

- 把 PagedAttention 解释为 KV cache allocator：blocks、block tables，以及为什么 production load 下 fragmentation 保持在 4% 以下。
- 在 iteration 级别画出 continuous batching：finished sequences 如何离开 batch，新 sequences 如何不等 drain 就加入。
- 用一句话描述 chunked prefill，并说出它保护哪个 latency metric（提示：是 TTFT tail，不是 mean throughput）。
- 说出 2026 年 vLLM v0.18.0 中会咬到那些一次性打开所有优化的团队的 gotcha。

## 问题

一个 naive PyTorch serve loop 一次跑一个请求：tokenize、prefill、decode 直到 EOS、return。一个用户时可行。一百个用户时，就是一队耐心等待的人。显而易见的修复是 static batching，但它会把每个请求 pad 到窗口里最长的 prompt，把每次 decode pad 到预期最长输出，并让整个 batch 卡在最慢 sequence 上。你为从未使用的 padding 付费，快请求也要等慢请求。

vLLM 同时解决三个问题。PagedAttention 阻止 KV cache fragmentation 像经典 contiguous allocation 那样吃掉 60-80% GPU memory。Continuous batching 允许请求在每次 decode iteration 之间加入和离开 batch，因此 batch 总是装满真实工作。Chunked prefill 把 32k-token prompt 切成约 512-token slices，与 decode 交错执行，所以一个长 prompt 不会冻结 GPU 上所有 decode token。

2026 年的 production default 是三者全开。你需要理解每个能力做什么，因为 failure modes 都在 scheduler 上，而不是 model 上。

## 概念

### PagedAttention 作为 virtual memory system

KV cache 对每个 sequence 的大小是 `num_layers × 2 × num_heads × head_dim × seq_len × bytes_per_element`。对于 Llama 3.3 70B、8192 tokens，BF16 下每个 sequence 大约 1.25 GB。如果你为每个请求预留 8192 slots，但平均请求只用 1500 tokens，你浪费了大约 82% 预留的 HBM。经典 batching 会支付这笔浪费。

PagedAttention 借鉴了 OS virtual memory 的想法。KV cache 对每个 sequence 不再连续。它被分配为 fixed-size blocks（默认 16 tokens）。每个 sequence 有一张 block table，把逻辑 token positions 映射到物理 block IDs。当 sequence 增长超过已分配 blocks 时，再添加一个 block。当它结束时，blocks 归还给 pool。

Fragmentation 从 60-80%（classic）降到 4% 以下（PagedAttention）。你不需要用 flag 开启 PagedAttention，它是 vLLM 提供的唯一 allocator。旋钮是 `--gpu-memory-utilization`（默认 0.9），告诉 vLLM 在加载 weights 和 activations 后，将多少 HBM 预留给 KV blocks。

### Iteration 级别的 continuous batching

旧的 “dynamic batching” 会等待一个窗口（比如 10 ms）填满 batch，然后运行 prefill + decode + decode + decode，直到每个 sequence 完成。快 sequences 早早离开并闲置，GPU 还在处理慢的。

Continuous batching 在每个 decode step 之间运行。把正在运行的 sequences 集合称为 `RUNNING` list。每次 iteration：

1. `RUNNING` 中刚刚 hit EOS 或 max_tokens 的 sequence 被移除。
2. Scheduler 查看 waiting queue。如果有空闲 KV blocks，它会接纳新的 sequences（prefill 或 resumed）。
3. Forward pass 在现在的 `RUNNING` 上运行，每个 sequence 发出一个新 token。

Batch size 永远不会 pad 到固定数字。处于不同 output 位置的 sequences 共享一次 fused forward。2026 年 vLLM 中这叫 `V1 scheduler`。关键不变量：scheduler 每次 decode iteration 运行一次，而不是每个 request 运行一次。

### Chunked prefill 保护 TTFT tail

Prefill 是 compute-bound。单张 H100 上，Llama 3.3 70B 的一个 32k-token prompt 需要约 800 ms 纯 prefill。prefill 运行期间，batch 中所有其他 sequence 的 decode tokens 都在等待。在 serving loop 中，一个长 prompt 的 first-token latency（TTFT）会变成几十个其他用户的 inter-token latency（ITL）尖刺。

Chunked prefill 把 prefill 切成 fixed-size chunks（默认 512 tokens），每个 chunk 作为一个单元调度。chunks 之间，scheduler 可以让 decode sequences 前进一个 token。你用一点绝对 prefill latency 损失（每 chunk 几 ms）换来低得多的 decode-time jitter。公开 benchmark 中，混合负载下 P99 ITL 从约 50 ms 降到约 15 ms。

### 三个默认能力相互作用

三个功能都假设彼此存在。PagedAttention 给 scheduler 一个细粒度 KV resource，可以用来取舍。Continuous batching 需要这种细粒度 resource，这样接纳新 sequence 才不会强制全局 reshuffle。Chunked prefill 是 scheduler 在同一张 `RUNNING` list 上做出的决定，它只是另一个 scheduler policy，而不是独立系统。

你不需要知道每个 flag。你需要知道 scheduler 优化什么：在 KV-block budget 下优化 goodput，并受 chunked prefill slicing 约束。

### 2026 年 v0.18.0 gotcha

在 vLLM v0.18.0 中，不能把 `--enable-chunked-prefill` 与 draft-model speculative decoding（`--speculative-model`）组合使用。文档中的例外是 V1 scheduler 中的 N-gram GPU speculative decoding。那些不读 release notes 就打开所有 flag 的团队，会在启动时遇到 run-time error，而不是 soft regression。如果 speculative gain 值得你启用 chunked prefill，就重新审视这个选择：2026 年正确答案通常是不用 chunked prefill 的 EAGLE-3，而不是无法编译的 draft model + chunked prefill。

### 你应该记住的数字

- Llama 3.3 70B FP8，H100 SXM5，128 concurrent，三者全开：2,200-2,400 tok/s。
- 同一模型，默认 vLLM（无 chunked prefill）：~1,800 tok/s。
- 同一模型，naive PyTorch forward loop：~600 tok/s。
- PagedAttention 下 production load 的 KV fragmentation waste：<4%。
- 混合负载下 P99 ITL：有 chunked prefill 约 15 ms，无则约 50 ms。

### Scheduler 看起来是什么样

```
while True:
    finished = [s for s in RUNNING if s.is_done()]
    for s in finished: release_blocks(s); RUNNING.remove(s)

    while WAITING and have_free_blocks_for(WAITING[0]):
        s = WAITING.pop(0)
        allocate_initial_blocks(s)
        RUNNING.append(s)

    # schedule prefill chunks + decode in one batch
    batch = []
    for s in RUNNING:
        if s.in_prefill:
            batch.append(next_prefill_chunk(s))   # e.g. 512 tokens
        else:
            batch.append(decode_one_token(s))     # 1 token

    run_forward(batch)                            # one fused GPU call
```

`code/main.py` 正是这个 loop 的 stdlib Python 版本，使用假的 token counts 和假的 forward latency。运行它会展示 chunked prefill 如何在长 prefill 期间让 decode sequences 保持活跃。

## 使用它

`code/main.py` 模拟一个 vLLM-style scheduler，功能可开关。运行它观察：

- `NAIVE` mode：一次一个请求，没有 batching。
- `STATIC` mode：pad 并等待，经典 batching。
- `CONTINUOUS` mode：iteration-level admission 和 release。
- `CONTINUOUS + CHUNKED` mode：prefill slices 与 decode 交错。

输出会展示 total throughput（tokens per virtual second）、TTFT mean 和 P99 ITL。`CONTINUOUS + CHUNKED` 这一行应该在混合流量上占优。

## 交付它

本课会产出 `outputs/skill-vllm-scheduler-reader.md`。给定 serving config（batch size、KV memory utilization、chunked prefill size、speculative config），它会生成 scheduler diagnosis，指出三个默认能力中哪个成为 bottleneck，以及该调什么。

## 练习

1. 运行 `code/main.py`。在混合短请求和长请求的 workload 上比较 `STATIC` 与 `CONTINUOUS`。throughput gap 来自哪里：prefill efficiency、decode efficiency，还是 tail latency？
2. 修改玩具 scheduler，添加 `--max-num-batched-tokens`。在 H100 上运行 Llama 3.3 70B FP8 时正确值是多少？（提示：它是 KV block size 和 free blocks 数量的函数，不是 raw HBM 的函数。）
3. 重新阅读 vLLM v0.18.0 release notes。哪些 flags 组合互斥？列出来。
4. 对 1,000 个请求的 trace 计算 KV cache fragmentation waste：mean 1,500 output tokens，std 600 tokens，分别在（a）按 8192 max 的 contiguous per-request allocation，（b）16-token blocks 的 PagedAttention 下。
5. 用一段话解释为什么 chunked prefill 帮助 P99 ITL，但单独看不提高 throughput。实践中的 throughput win 来自哪里？

## 关键词汇

| 术语 | 大家常说 | 实际含义 |
|------|----------|----------|
| PagedAttention | “the KV trick” | KV cache 的 fixed-size block allocator；fragmentation <4% |
| Block table | “the page table” | 每个 sequence 从逻辑 token position 到物理 KV block 的映射 |
| Continuous batching | “dynamic batching, but right” | 每个 decode iteration 做 admit/release 决策 |
| Chunked prefill | “prefill splitting” | 把长 prefill 切成 512-token slices，与 decode 交错 |
| TTFT | “first token time” | Prefill + queue + network；长 prompt 下由 prefill 主导 |
| ITL | “inter-token latency” | 连续 decode tokens 之间的时间；由 batch size 主导 |
| Goodput | “throughput that meets SLO” | 每个 request 仍满足 TTFT 和 ITL 目标的 tokens/sec |
| V1 scheduler | “the new scheduler” | vLLM 2026 scheduler；N-gram spec decode 是与 chunked prefill 兼容的路径 |
| `--gpu-memory-utilization` | “the memory knob” | weights 和 activations 后为 KV blocks 预留的 HBM 比例 |

## 延伸阅读

- [vLLM documentation — Speculative Decoding](https://docs.vllm.ai/en/latest/features/spec_decode/) — chunked-prefill 和 speculative-decoding compatibility 的官方来源。
- [vLLM Release Notes (NVIDIA)](https://docs.nvidia.com/deeplearning/frameworks/vllm-release-notes/index.html) — 2026 release cadence 和 version-specific behavior。
- [vLLM Blog — PagedAttention](https://blog.vllm.ai/2023/06/20/vllm.html) — 至今仍定义 allocator 思考方式的原始文章。
- [PagedAttention paper (arXiv:2309.06180)](https://arxiv.org/abs/2309.06180) — fragmentation analysis 和 scheduler design。
- [Aleksa Gordic — Inside vLLM](https://www.aleksagordic.com/blog/vllm) — 带 flame graphs 的 V1 scheduler 详细 walkthrough。
