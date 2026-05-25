# 面向 Prefix-Heavy Workloads 的 SGLang 和 RadixAttention

> SGLang 把 KV cache 当作一等、可复用资源，存储在 radix tree 中。vLLM 按 FCFS（first-come, first-served）调度请求，而 SGLang 的 cache-aware scheduler 会优先处理共享 prefix 更长的请求，本质上是 depth-first radix traversal，让 hot branches 留在 HBM 中。在 ShareGPT-like 1K prompts 的 Llama 3.1 8B 上，SGLang 达到约 16,200 tok/s，而 vLLM 约 12,500，领先约 29%。在 prefix-heavy RAG workloads 上，优势可达 6.4x。在 voice-cloning-shaped workloads 上，cache hit rate 超过 86%。2026 年已经在 xAI、LinkedIn、Cursor、Oracle、GCP、Azure、AWS 的 400,000+ GPUs 上部署。gotcha 是：如果 prefix ordering 不一致，6.4x 数字会消失；ordering 是工程师手里的杠杆。

**类型：** 学习
**语言：** Python（stdlib，玩具版 radix-tree cache + cache-aware scheduler）
**前置要求：** 阶段 17 · 04（vLLM Serving Internals），阶段 14（Agentic RAG）
**时间：** ~75 分钟

## 学习目标

- 画出 RadixAttention：prefixes 如何存储在 radix tree 中，以及 KV blocks 如何在同一 branch 下的 sequences 之间共享。
- 解释 cache-aware scheduling，以及为什么 FCFS 不适合 prefix-heavy traffic。
- 给定 prefix-cache hit rate 和 prompt length distribution，计算 workload 的 expected speedup。
- 说出让 6.4x 数字成立而不是丢失收益的 prompt-ordering discipline。

## 问题

经典 serving 把每个 request 的 prompt 当成 opaque。即使 5,000 个 RAG requests 都以相同的 2,000-token system prompt 加同样的 retrieval preamble 开头，vLLM 也会把这个 2,000-token prefix prefill 5,000 次。GPU 一遍又一遍做同样的工作。

观察是：agentic 和 RAG workloads 的 prompts 几乎总是共享长 prefix。System prompt、tool schemas、few-shot examples、retrieval headers、conversation history 都会在请求之间重复。如果你只存一次这个 prefix 的 KV cache 并复用，就不必再次 prefill。

RadixAttention 正是这样做。Tokens 被索引进 radix tree；每个 node 拥有从 root 到该 node 路径上的 token sequence 的 KV blocks。新请求会遍历这棵树：任何 token 匹配的 node 都会复用该 node 的 KV blocks。Prefill cost 变成与“新”suffix 成正比，而不是与完整 prompt 成正比。

挑战是调度。如果两个请求共享 2,000-token prefix，第三个只共享同一 prefix 的 200 tokens，你希望一起服务那两个长共享请求，让长 prefix 保持在 HBM 中。FCFS 反着来：谁先到服务谁，可能在下一个 long-prefix request 命中前就把 hot branch 驱逐掉。

## 概念

### Radix tree 作为 KV index

Radix tree（compact trie）存储 token sequences。每个 node 拥有一个 token range 以及为该 range 计算出的 KV blocks。Children 会把 sequence 延长一个或多个 tokens。

```
root
 |- "You are a helpful assistant..."  (2,000 tokens, 124 KV blocks)
      |- "Context: <doc A>..."        (500 tokens, 31 blocks)
           |- "Question: Alice..."    (80 tokens, 5 blocks)
           |- "Question: Bob..."      (95 tokens, 6 blocks)
      |- "Context: <doc B>..."        (520 tokens, 33 blocks)
```

一个新请求带着 system prompt + “Context: <doc A>” + “Question: Carol” 进入。Scheduler 遍历：system prefix 匹配（复用 124 blocks），doc-A branch 匹配（复用 31 blocks），然后只为 “Question: Carol” 分配新 blocks（4 blocks）。Prefill cost：4 blocks 的新 tokens。没有 tree：160 blocks。prefill 上约 40x 节省。

### Cache-aware scheduling

如果 cache 不停 churn，基于 radix tree 的复用没有意义。两个关键策略：

1. **Depth-first dispatch**。从 queue 中选择下一个请求时，优先选择和当前 running set 位于同一 branch 的请求。这会让 hot branch 被 pinned。
2. **Branch-level LRU，而不是 block-level LRU**。驱逐整条 branches（从 shortest-used leaves 开始），而不是单个 blocks，这样 cache shape 才匹配 radix shape。

FCFS 违反二者。一个共享 2,000 tokens 的请求排在一个只共享 50 tokens 的请求后面，然后 2,000-token branch 被驱逐，只为接纳那个 50-token 请求。

### 你应该记住的 benchmark 数字

- Llama 3.1 8B、H100、ShareGPT 1K prompts：SGLang ~16,200 tok/s vs vLLM ~12,500（~29% edge）。
- Prefix-heavy RAG（相同 system + 相同 doc，不同 question）：SGLang 最高 6.4x。
- Voice cloning workloads：86.4% prefix-cache hit rate。
- SGLang 客户的 production hit rates：50-99%，取决于 prompt discipline。
- 2026 年部署在 400,000+ GPUs 上。

### Ordering gotcha

6.4x 数字依赖 consistent prompt-template ordering。如果你的 client 在一些请求中构造 prompts 为 `[system, tools, context, history, question]`，在另一些请求中构造为 `[system, context, tools, history, question]`，tree 就无法找到 shared prefix。对人来说看起来共享的 prefix，对 radix tree 来说是两个不同 sequence。

工程师的杠杆：prompt template 就是 cache key。固定顺序。把所有 immutable 内容（system、tools、schemas）放在前面。retrieval context 放在后面。user question 放最后。不要把 dynamic content interleave 到 prefix 中。

研究中的真实案例：把 dynamic content 移出可缓存 prefix 后，一个 deployment 的 cache hit rate 一次变更从 7% 提升到 74%。

### RadixAttention 赢在哪里、输在哪里

赢：
- RAG（相同 retrieval preamble，不同 question）。
- Agents（相同 tool schemas，不同 query）。
- 带长 system prompt 的 chat。
- 带重复 preambles 的 voice / vision workloads。

输（退回 vLLM-level throughput）：
- Unique prompts 的 single-shot generation（code completion、没有 system prompt 的 open-ended chat）。
- 每个请求都把 unique content interleave 到 prefix 中的 dynamic prompts。

### 为什么这是 scheduler 问题，不只是 kernel 问题

你可以把 KV reuse 实现成 kernel trick。SGLang 的洞察是：只有 scheduler 让 hot branch 常驻时，reuse 才划算。一个 naive 的“可用就复用”policy 会在 mixed load 下让 cache churn。radix-tree-indexed scheduler 才是把 kernel trick 变成 29% production edge 的原因。

### 与 vLLM 的关系

两套系统不是严格竞争者。2026 年 vLLM 加入了 prefix caching（`--enable-prefix-caching`）和 cache-aware router（Rust 写的 vLLM Router）。差距缩小，但没有完全消失：SGLang 整个 stack 都是 radix-first；vLLM 是 grafted on。对 prefix reuse 占主导的 workload，SGLang 仍是默认选择。对没有强 prefix pattern 的通用 serving，vLLM 仍然相当或更好。

## 使用它

`code/main.py` 实现了一个玩具 radix-tree KV cache，以及带 FCFS 和 cache-aware 两种策略的 scheduler。它让同一个 workload 通过两者，报告 prefix-cache hit rate 和 throughput delta。然后运行一个 “scrambled ordering” workload，展示 6.4x 如何崩掉。

## 交付它

本课会产出 `outputs/skill-radix-scheduler-advisor.md`。给定 workload description（prompt-template shape、retrieval pattern、concurrent tenants 数量），它会生成 prompt-ordering prescription 和是否采用 SGLang 的 go/no-go。

## 练习

1. 运行 `code/main.py`。在同一 workload 上比较 FCFS 和 cache-aware。delta 来自哪里：prefill savings、decode savings，还是 queue delay？
2. 修改 workload，让 prompts 随机排列 `[system, tools, context]`。重新运行。hit rate 发生什么？为什么？
3. 计算在 Llama 3.1 8B 上，把一个 2,000-token system prompt 作为一个 radix branch 常驻所需的 HBM cost。与一个无 prefix reuse 的 16-sequence batch 成本比较。
4. 阅读 SGLang RadixAttention paper。用三句话解释为什么 tree-shaped LRU eviction 在 prefix-heavy load 下胜过 block-shaped LRU。
5. 一个客户报告 cache hit rate 只有 8%。说出三个可能原因，以及你会为每个原因运行的 diagnostic。

## 关键词汇

| 术语 | 大家常说 | 实际含义 |
|------|----------|----------|
| RadixAttention | “the SGLang thing” | 把 KV cache 索引为 radix tree，让 shared prefixes 复用 blocks |
| Radix tree | “compact trie” | 每个 node 拥有一个 token range 及其 KV blocks 的树 |
| Cache-aware scheduler | “hot-branch-first” | 优先处理共享 resident branch 请求的 scheduler |
| Prefix-cache hit rate | “how much of your prompt was free” | prompt tokens 中由复用 KV blocks 服务的比例 |
| FCFS | “first-come first-served” | 破坏 prefix locality 的默认调度 |
| Branch-level LRU | “evict the leaf” | 与 radix shape 匹配的 eviction policy |
| Prompt template ordering | “the cache key” | prompt 组件顺序决定 tree 能共享什么 |
| System prompt pinning | “resident prefix” | 保持 immutable system 部分 pinned，避免 eviction thrash |

## 延伸阅读

- [SGLang GitHub](https://github.com/sgl-project/sglang) — source 和 docs。
- [SGLang documentation](https://sgl-project.github.io/) — RadixAttention 和 scheduling details。
- [SGLang paper — Efficiently Programming Large Language Models (arXiv:2312.07104)](https://arxiv.org/abs/2312.07104) — design reference。
- [LMSYS blog — SGLang with RadixAttention](https://www.lmsys.org/blog/2024-01-17-sglang/) — benchmark numbers 和 scheduler rationale。
- [vLLM — Prefix Caching](https://docs.vllm.ai/en/latest/features/prefix_caching.html) — vLLM 自己的 radix-like implementation，用于对比。
