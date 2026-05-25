# Inference Optimization

> LLM inference 由两个阶段定义。Prefill 并行处理 prompt，是 compute-bound。Decode 一次生成一个 token，是 memory-bound。每个优化都针对其中一个或两个阶段。

**类型：** 构建
**语言：** Python
**前置要求：** 阶段 10，第 01-08 课（Transformer architecture, attention）
**时间：** ~120 分钟

## 学习目标

- 实现 KV-cache，消除 autoregressive token generation 中的重复计算
- 解释 LLM inference 的 prefill vs decode 阶段，以及为什么二者瓶颈不同（compute-bound vs memory-bound）
- 实现 continuous batching 和 PagedAttention 概念，在并发请求下最大化 GPU utilization
- 比较 inference optimization techniques（KV-cache、speculative decoding、flash attention）及其 throughput/latency tradeoffs

## 问题

你把 Llama 3 70B 部署到 4xA100 GPU 上。单个用户大约能得到 50 tokens per second。感觉很快。然后 100 个用户同时打到 endpoint。throughput 掉到每个用户 3 tokens/second。你每月 25,000 美元的 GPU 账单正在以比人打字还慢的速度服务 responses。

从 1 个用户到 100 个用户，模型本身没有改变。相同 weights、相同 architecture、相同数学。改变的是你如何调度工作。naive inference 会浪费 90%+ 可用 GPU compute。一个等第 47 个 token 的用户占着整个 batch slot，而 GPU memory bus 在 matmuls 之间空等。同时，一个新用户的 2,000-token prompt 本可以用这些死时间做有用 compute。

这不是 scaling problem。这是 scheduling problem。本课中的技术，KV caching、continuous batching、PagedAttention、speculative decoding、prefix caching，就是同样 traffic 下每月 25k 美元 inference bill 和 5k 美元 inference bill 的区别。

vLLM 在 4xA100-80GB 上 serving Llama 3 70B，低并发时约 50 tokens/second/user，并且通过 continuous batching 和 PagedAttention 在 100 concurrent requests 下保持 15-25 TPS/user。没有这些优化，同一硬件在该并发下只能服务 5 TPS/user。同样 GPU，同样模型，throughput 4 倍。

## 概念

### Prefill vs Decode

每个 LLM inference request 都有两个不同阶段。

**Prefill** 处理整个 input prompt。所有 tokens 已知，因此 attention 可以在完整序列上并行计算。这是大型 matrix multiplication，GPU cores 会保持忙碌。瓶颈是 compute：硬件每秒能提供多少 FLOPS。A100 有 312 TFLOPS（BF16）。70B 模型在单张 A100 上 prefill 一个 4,096-token prompt 大约需要 400ms。

**Decode** 一次生成一个 output token。每个新 token attend 到所有 previous tokens，但每个 forward pass 只生成一个 token。weight matrices 与 prefill 阶段一样大，但你用一个 vector 乘它们，而不是 matrix。GPU cores 微秒级完成计算，然后等待下一批 weights 从 memory 到达。瓶颈是 memory bandwidth：你能多快把 model weights 从 HBM stream 到 compute units。A100 有 2 TB/s bandwidth。FP16 的 70B 模型是 140 GB。读完整模型一次需要 70ms，这就是单个 decode step 的下限。

```mermaid
graph LR
    subgraph "Prefill (compute-bound)"
        P1["All prompt tokens"] --> P2["Parallel attention"]
        P2 --> P3["Full matmul utilization"]
    end

    subgraph "Decode (memory-bound)"
        D1["One token at a time"] --> D2["Sequential generation"]
        D2 --> D3["Waiting on memory reads"]
    end

    P3 --> D1
```

**ops:byte ratio**（也叫 arithmetic intensity）捕捉这个 tradeoff。它衡量每读取一个 byte memory 要执行多少 operations。

```
ops:byte ratio = FLOPs per token / bytes read from memory
```

prefill 中 batch 有 4,096 tokens 时，每加载一个 weight，你执行约 4,096 次 multiply-accumulate operations。ratio 高，因此 compute-bound。decode 中 batch size 为 1 时，每加载一个 weight 只执行约 1 次 operation。ratio 低，因此 memory-bound。

根本洞见是：*decode 之所以 memory-bound，是因为你读取整个模型只为了生成一个 token*。下面每个优化要么减少读取内容，要么增加每次读取处理的 token batch，要么完全避免读取。

### KV Cache

attention 中，每个 token 的 query attend 到每个 previous token 的 key 和 value vectors。没有 cache 时，生成第 N 个 token 需要重新计算前 N-1 个 tokens 的 key 和 value projections。token 1 在生成 token 2 时被 project 一次，生成 token 3 时又一次，生成 token 4 时又一次。到 token 1,000 时，token 1 已经被 project 999 次。

KV cache 存储所有 previous tokens 的 key 和 value projections。生成 token N 时，你只计算 token N 的 key 和 value，然后把它们与 tokens 1 到 N-1 的 cached K/V 拼接。

```mermaid
graph TD
    subgraph "Without KV Cache"
        A1["Token 5: recompute K,V for tokens 1-4"]
        A2["Token 6: recompute K,V for tokens 1-5"]
        A3["Token 7: recompute K,V for tokens 1-6"]
    end

    subgraph "With KV Cache"
        B1["Token 5: compute K5,V5, read K1-4,V1-4 from cache"]
        B2["Token 6: compute K6,V6, read K1-5,V1-5 from cache"]
        B3["Token 7: compute K7,V7, read K1-6,V1-6 from cache"]
    end
```

**KV cache 的内存公式：**

```
KV cache size = 2 * num_layers * num_kv_heads * head_dim * seq_len * bytes_per_param
```

对于 Llama 3 70B（80 layers、带 GQA 的 8 KV heads、head_dim=128、BF16）：

```
per token: 2 * 80 * 8 * 128 * 2 bytes = 327,680 bytes = 320 KB
at 4,096 tokens: 320 KB * 4,096 = 1.28 GB
at 128K tokens: 320 KB * 131,072 = 40 GB
```

Llama 3 70B 的单个 128K-context conversation 会消耗 40GB KV cache，也就是半张 A100 显存。100 个 4K tokens concurrent users，仅 KV cache 就需要 128GB。这就是为什么 KV cache management 是 inference optimization 的核心挑战。

### Continuous Batching

Static batching 会等到 N 个 requests 组成 batch，一起处理，并等到 *所有* requests 完成后才接收新请求。如果一个 request 需要 500 tokens，另一个需要 10，短 request 完成后还会在 490 个 decode steps 中空占 slot。

Continuous batching（也叫 iteration-level batching）会在任意 request 完成后立刻把新 request 插入 batch。batch 在每个 decode step 都会重新评估。一个 10 tokens 后完成的 request 会马上被等待队列中的 request 替换。

```mermaid
sequenceDiagram
    participant GPU
    participant R1 as Request 1 (50 tokens)
    participant R2 as Request 2 (10 tokens)
    participant R3 as Request 3 (30 tokens)
    participant R4 as Request 4 (waiting)

    Note over GPU: Static batching
    GPU->>R1: Process batch [R1, R2, R3]
    Note over R2: R2 done at step 10
    Note over R2: Wasting 40 steps...
    Note over R3: R3 done at step 30
    Note over R3: Wasting 20 steps...
    GPU->>R4: Finally start R4 at step 50

    Note over GPU: Continuous batching
    GPU->>R1: Process batch [R1, R2, R3]
    Note over R2: R2 done at step 10
    GPU->>R4: Insert R4 at step 11
    Note over R3: R3 done at step 30
```

throughput improvement 取决于 output lengths 的差异。长度均匀时，continuous batching 与 static batching 相同。长度变化大时（常见情况），continuous batching 可以提供 2-5 倍 higher throughput，因为 GPU slots 永远不空。

### PagedAttention

每个 request 的 KV cache 是一段 contiguous memory block。随着 requests 到达和离开，memory 会 fragment，就像 operating systems 中的 RAM fragmentation。一个 4K-token request 需要 1.28GB 连续空间。即使你总共有 2GB free，也可能没有 1.28GB *contiguous*。你要么浪费 memory，要么拒绝 request。

PagedAttention（来自 vLLM）把 OS-style virtual memory 应用于 KV cache。它不是为每个 request 分配一个 contiguous block，而是分配固定大小的 “pages”（通常每页 16 tokens）。pages 可以位于 physical GPU memory 的任何位置。page table 把每个 request 的 logical sequence positions 映射到 physical page locations。

```mermaid
graph TD
    subgraph "Contiguous allocation"
        C1["Request A: 2GB block"]
        C2["[free: 0.5GB]"]
        C3["Request B: 1GB block"]
        C4["[free: 1.5GB -- but fragmented]"]
    end

    subgraph "PagedAttention"
        P1["Page pool: 256 pages of 16 tokens each"]
        P2["Request A: pages 3,7,12,45,88..."]
        P3["Request B: pages 1,4,9,22,67..."]
        P4["No fragmentation, no waste"]
    end
```

PagedAttention 还为 shared prefixes 启用 **copy-on-write**。如果 50 个 requests 共享同一个 system prompt，该 system prompt 的 KV cache pages 只存一次，并被 50 个 requests 引用。只有当 request diverge（不同 user messages）时，它才获得自己的 pages。这会大幅削减具有共享 system prompts 的应用的 memory usage。

vLLM 报告通过 PagedAttention 实现近零 memory waste（约 4%，而 naive allocation 是 60-80%）。

### Speculative Decoding

Decode 慢是因为它 sequential：生成一个 token，喂回去，再生成下一个。但如果你能廉价猜出接下来的 5 个 tokens，然后一次性验证它们呢？

Speculative decoding 使用一个小而快的 **draft model** 生成 K 个 candidate tokens。大 **target model** 再用一次 forward pass 处理全部 K 个 candidates（看起来像 prefill：parallel、compute-bound、高效）。如果 target model 同意 draft model 的预测，你就在一次 target forward pass 的时间内接受全部 K 个 tokens。如果它在位置 j 不同意，你接受 tokens 1 到 j-1，丢弃其余。

```mermaid
graph LR
    D["Draft model (1B)"] -->|"Generate 5 tokens<br/>~5ms"| C["Candidates: the cat sat on the"]
    C --> T["Target model (70B)"]
    T -->|"Verify all 5 in one pass<br/>~70ms"| V{"Match?"}
    V -->|"4 of 5 match"| A["Accept 4 tokens in 75ms<br/>vs 280ms sequential"]
    V -->|"Mismatch at pos 5"| R["Reject token 5<br/>Resample from target"]
```

speedup 取决于 **acceptance rate**：draft model 的预测有多常匹配 target。Llama 3 8B 给 Llama 3 70B drafting 时，自然语言上的 acceptance rates 通常是 70-85%。这对应 2-3 倍 decode speedup。

三种 speculative decoding 方法：

| Method | Draft source | Acceptance rate | Overhead |
|--------|-------------|-----------------|----------|
| Draft-target (Leviathan et al.) | Separate small model | 70-85% | Draft model memory |
| EAGLE (Li et al.) | Lightweight head on target | 75-90% | ~1% extra parameters |
| N-gram lookup | Token n-gram table | 40-60% | Negligible |

**EAGLE** 在 target model hidden states 顶部训练一个小 autoregressive head。它用 target model 倒数第二层 features 预测下一个 token 的 embedding。因为它使用 target model 自己的 representations（而不是单独模型的），所以用极少额外 memory 达到更高 acceptance rates。EAGLE-2 添加 dynamic draft tree，根据 context 调整 candidate count。

**N-gram speculative decoding** 维护一个来自当前 context 或预建 corpus 的 n-gram continuations 表。如果 draft 匹配同一 conversation 中曾出现过的内容（重复模式、代码、结构化输出），它可以用零 neural network overhead 触发。平均 acceptance rates 更低，但每次 speculation 的成本几乎为零。

Speculative decoding 是 *mathematically exact*：output distribution 与 target model distribution 完全相同。它不是近似。verification step 保证每个 accepted token 的概率与 target model 本来会分配的一模一样。

### Prefix Caching

许多 requests 共享相同 prefix。chatbot system prompt。RAG context block。few-shot example set。没有 prefix caching 时，每个 request 都会从头 recompute 这些 shared tokens 的 KV cache。

Prefix caching 存储常见 prefixes 的 KV cache，并跨 requests 复用。当新 request 带有已知 prefix 到来时，系统复制（或引用）cached KV entries，只计算 unique suffix 的 KV。

一个 2,000-token system prompt 被所有 requests 共享时，prefix caching 可为每个 request 消除约 400ms prefill。100 requests/second 时，这每秒节省 40 秒 GPU compute，相当于超过一张 GPU 的工作量。

SGLang 的 RadixAttention 用 radix tree（trie）实现 prefix caching，根据 token content 索引 prefixes。任何匹配已存 prefix 的 request 都可以免费获得它的 KV cache。tree 支持 partial prefix matches：如果你和某 cached entry 共享 2,000 prefix tokens 中的 1,500，就复用这 1,500，只 recompute 500。

### Inference Engines

三个 engines 主导生产 LLM serving：

| Engine | Key innovation | Best for |
|--------|---------------|----------|
| vLLM | PagedAttention, continuous batching | General-purpose serving, highest compatibility |
| SGLang | RadixAttention (prefix caching), structured generation | Multi-turn chatbots, constrained decoding |
| TensorRT-LLM | NVIDIA kernel fusion, FP8 quantization | Maximum single-GPU throughput on NVIDIA hardware |

**vLLM** 是默认起点。它支持最广的模型范围，可在任意 GPU vendor 上运行（NVIDIA、AMD、Intel），并通过 PagedAttention + continuous batching 获得强 throughput。OpenAI-compatible API 意味着你可以把它直接作为 OpenAI API 调用的替代。

**SGLang** 建立在与 vLLM 类似的基础之上，但添加 RadixAttention 做 prefix caching，并提供用于 structured LLM programs 的 domain-specific language。如果 workload 包含 multi-turn conversations、tool use 或 constrained decoding（JSON output、regex-guided generation），SGLang 常通过 prefix reuse 比 vLLM 快 2-5 倍。

**TensorRT-LLM** 把模型编译成 optimized NVIDIA GPU kernels。它融合 operations（attention + linear + activation 在一个 kernel 中），在 H100 GPU 上使用 FP8，并集成 NVIDIA Triton Inference Server 做生产部署。它在 NVIDIA hardware 上实现最高 single-GPU throughput，但设置更复杂，且只适用于 NVIDIA GPU。

Llama 3 70B 的真实数字（4xA100-80GB，BF16）：

| Metric | vLLM | SGLang | TensorRT-LLM |
|--------|------|--------|---------------|
| Throughput (1 user) | ~50 TPS | ~55 TPS | ~65 TPS |
| Throughput (100 users) | ~2,500 total TPS | ~3,200 total TPS | ~3,000 total TPS |
| Time to first token | ~400ms | ~300ms (prefix hit) | ~350ms |
| Max context | 128K | 128K | 128K |

### Ops:Byte Framework

你无法优化自己没有测量的东西。ops:byte ratio 告诉你是 compute-bound 还是 memory-bound，从而决定哪些优化重要。

```
Compute roof: peak FLOPS of the GPU
Memory roof:  peak bandwidth * ops:byte ratio
```

ops:byte 低时（decode、小 batches），你撞到 memory bandwidth roof。增加更多 compute（更高 clock、更多 cores）没有帮助。你需要减少 memory reads（quantization、KV cache compression），或增加 batch size，把 reads 平摊到更多有用工作上。

ops:byte 高时（prefill、大 batches），你撞到 compute roof。memory bandwidth 优化没有帮助。你需要更快 GPU、kernel fusion，或 reduced precision 来榨出更多 FLOPS。

| Scenario | ops:byte | Bound | Optimize with |
|----------|----------|-------|---------------|
| Prefill, batch=1 | ~4,096 | Compute | Kernel fusion, FP8 |
| Decode, batch=1 | ~1 | Memory | Quantization, KV compression |
| Decode, batch=32 | ~32 | Memory | Larger batch, continuous batching |
| Decode, batch=256 | ~256 | Transitioning | Both matter |
| Decode, batch=1024 | ~1,024 | Compute | Kernel fusion, tensor parallelism |

A100 上 crossover point 约为 ops:byte = 156（312 TFLOPS / 2 TB/s）。低于 156 是 memory-bound。高于 156 是 compute-bound。continuous batching 通过每轮打包更多 tokens，把 decode 推向这个 crossover。

## 构建它

### 第 1 步：从零实现 KV Cache

我们构建一个 multi-head KV cache，按 layer、head 存储 key 和 value projections，并展示 memory growth pattern。

```python
import numpy as np

class KVCache:
    def __init__(self, num_layers, num_heads, head_dim, max_seq_len, dtype=np.float16):
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.dtype = dtype

        self.k_cache = np.zeros(
            (num_layers, num_heads, max_seq_len, head_dim), dtype=dtype
        )
        self.v_cache = np.zeros(
            (num_layers, num_heads, max_seq_len, head_dim), dtype=dtype
        )
        self.seq_len = 0

    def update(self, layer_idx, new_keys, new_values):
        num_new = new_keys.shape[1]
        end = self.seq_len + num_new
        self.k_cache[layer_idx, :, self.seq_len:end, :] = new_keys
        self.v_cache[layer_idx, :, self.seq_len:end, :] = new_values
        return (
            self.k_cache[layer_idx, :, :end, :],
            self.v_cache[layer_idx, :, :end, :]
        )

    def advance(self, num_tokens):
        self.seq_len += num_tokens

    def memory_bytes(self):
        return self.k_cache.nbytes + self.v_cache.nbytes

    def used_bytes(self):
        per_token = 2 * self.num_layers * self.num_heads * self.head_dim * np.dtype(self.dtype).itemsize
        return per_token * self.seq_len
```

### 第 2 步：带 KV Cache 的 Attention

一个简化 multi-head attention，在 decode steps 中使用 KV cache。

```python
def scaled_dot_product_attention(query, keys, values):
    head_dim = query.shape[-1]
    scores = np.matmul(query, keys.transpose(0, 1, 3, 2)) / np.sqrt(head_dim)
    seq_len_q = scores.shape[-2]
    seq_len_k = scores.shape[-1]
    if seq_len_q > 1:
        mask = np.triu(np.ones((seq_len_q, seq_len_k), dtype=np.float32), k=seq_len_k - seq_len_q + 1)
        scores = scores + mask * (-1e9)
    max_scores = np.max(scores, axis=-1, keepdims=True)
    exp_scores = np.exp(scores - max_scores)
    attn_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
    return np.matmul(attn_weights, values)


class MultiHeadAttention:
    def __init__(self, d_model, num_heads):
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        scale = np.sqrt(2.0 / d_model)
        self.W_q = np.random.randn(d_model, d_model).astype(np.float32) * scale
        self.W_k = np.random.randn(d_model, d_model).astype(np.float32) * scale
        self.W_v = np.random.randn(d_model, d_model).astype(np.float32) * scale
        self.W_o = np.random.randn(d_model, d_model).astype(np.float32) * scale

    def forward(self, x, kv_cache=None, layer_idx=0):
        batch, seq_len, d_model = x.shape
        Q = np.matmul(x, self.W_q).reshape(batch, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        K = np.matmul(x, self.W_k).reshape(batch, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        V = np.matmul(x, self.W_v).reshape(batch, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        if kv_cache is not None:
            K_full, V_full = kv_cache.update(layer_idx, K[0], V[0])
            K = K_full[np.newaxis, :, :, :]
            V = V_full[np.newaxis, :, :, :]
            if seq_len == 1:
                kv_cache.advance(1)

        attn_out = scaled_dot_product_attention(Q, K, V)
        attn_out = attn_out.transpose(0, 2, 1, 3).reshape(batch, -1, d_model)
        return np.matmul(attn_out, self.W_o)
```

### 第 3 步：Continuous Batching Simulator

模拟 static 与 continuous batching 之间的 scheduling 差异。

```python
import heapq

class Request:
    def __init__(self, request_id, prompt_tokens, output_tokens, arrival_step):
        self.request_id = request_id
        self.prompt_tokens = prompt_tokens
        self.output_tokens = output_tokens
        self.arrival_step = arrival_step
        self.tokens_generated = 0
        self.start_step = None
        self.end_step = None

    def is_done(self):
        return self.tokens_generated >= self.output_tokens


def simulate_static_batching(requests, batch_size):
    step = 0
    completed = []
    queue = list(requests)
    queue.sort(key=lambda r: r.arrival_step)

    while queue:
        batch = []
        while queue and len(batch) < batch_size:
            r = queue.pop(0)
            r.start_step = max(step, r.arrival_step)
            batch.append(r)

        if batch:
            step = max(step, max(r.start_step for r in batch))
            max_output = max(r.output_tokens for r in batch)
            for r in batch:
                r.tokens_generated = r.output_tokens
                r.end_step = step + max_output
            step += max_output
            completed.extend(batch)

    return completed


def simulate_continuous_batching(requests, batch_size):
    step = 0
    completed = []
    queue = sorted(requests, key=lambda r: r.arrival_step)
    queue_idx = 0
    active = []
    waiting = []

    while queue_idx < len(queue) or active or waiting:
        while queue_idx < len(queue) and queue[queue_idx].arrival_step <= step:
            waiting.append(queue[queue_idx])
            queue_idx += 1

        while waiting and len(active) < batch_size:
            r = waiting.pop(0)
            r.start_step = step
            active.append(r)

        if not active:
            if waiting:
                step += 1
                continue
            elif queue_idx < len(queue):
                step = queue[queue_idx].arrival_step
                continue
            else:
                break

        for r in active:
            r.tokens_generated += 1

        done = [r for r in active if r.is_done()]
        for r in done:
            r.end_step = step + 1
            completed.append(r)
        active = [r for r in active if not r.is_done()]

        step += 1

    return completed


def batching_stats(completed):
    latencies = [r.end_step - r.arrival_step for r in completed]
    total_time = max(r.end_step for r in completed) - min(r.arrival_step for r in completed)
    total_tokens = sum(r.output_tokens for r in completed)
    return {
        "avg_latency": np.mean(latencies),
        "p50_latency": np.median(latencies),
        "p99_latency": np.percentile(latencies, 99),
        "total_time": total_time,
        "throughput": total_tokens / total_time if total_time > 0 else 0,
    }
```

### 第 4 步：Prefix Cache

一个基于 trie 的 prefix cache，用来为 shared prefixes 存储 KV entries。

```python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.kv_data = None
        self.hit_count = 0


class PrefixCache:
    def __init__(self, max_entries=1000):
        self.root = TrieNode()
        self.max_entries = max_entries
        self.total_entries = 0
        self.hits = 0
        self.misses = 0

    def _walk(self, token_ids):
        node = self.root
        depth = 0
        for tid in token_ids:
            if tid not in node.children:
                break
            node = node.children[tid]
            depth += 1
        return node, depth

    def lookup(self, token_ids):
        node, depth = self._walk(token_ids)
        if depth > 0:
            self.hits += 1
            current = self.root
            for tid in token_ids[:depth]:
                current = current.children[tid]
                current.hit_count += 1
            kv_entries = []
            current = self.root
            for tid in token_ids[:depth]:
                current = current.children[tid]
                if current.kv_data is not None:
                    kv_entries.append(current.kv_data)
            return depth, kv_entries
        self.misses += 1
        return 0, []

    def insert(self, token_ids, kv_per_token):
        node = self.root
        for i, tid in enumerate(token_ids):
            if tid not in node.children:
                if self.total_entries >= self.max_entries:
                    return i
                node.children[tid] = TrieNode()
                self.total_entries += 1
            node = node.children[tid]
            if i < len(kv_per_token):
                node.kv_data = kv_per_token[i]
        return len(token_ids)

    def hit_rate(self):
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0
```

### 第 5 步：Speculative Decoding Simulator

模拟 draft-target speculative decoding，可配置 acceptance rates。

```python
class DraftModel:
    def __init__(self, vocab_size, acceptance_rate=0.8):
        self.vocab_size = vocab_size
        self.acceptance_rate = acceptance_rate

    def generate(self, context, num_tokens):
        tokens = np.random.randint(0, self.vocab_size, size=num_tokens)
        return tokens

    def get_probs(self, context, token):
        probs = np.random.dirichlet(np.ones(self.vocab_size))
        return probs


class TargetModel:
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size

    def get_probs(self, context, tokens=None):
        if tokens is not None:
            return [np.random.dirichlet(np.ones(self.vocab_size)) for _ in tokens]
        return np.random.dirichlet(np.ones(self.vocab_size))


def speculative_decode(draft_model, target_model, context, num_speculative=5,
                       draft_cost=1.0, target_cost=10.0, verify_cost=12.0):
    total_tokens = 0
    total_cost = 0.0
    accepted_counts = []
    context = list(context)

    max_tokens = 100

    while total_tokens < max_tokens:
        draft_tokens = draft_model.generate(context, num_speculative)
        total_cost += draft_cost * num_speculative

        target_probs = target_model.get_probs(context, draft_tokens)
        total_cost += verify_cost

        accepted = 0
        for i, token in enumerate(draft_tokens):
            draft_p = draft_model.get_probs(context + list(draft_tokens[:i]), token)
            target_p = target_probs[i]

            r = np.random.random()
            acceptance_prob = min(1.0, target_p[token] / (draft_p[token] + 1e-10))

            if r < draft_model.acceptance_rate:
                accepted += 1
                context.append(token)
                total_tokens += 1
            else:
                new_token = np.random.choice(draft_model.vocab_size, p=target_p)
                context.append(new_token)
                total_tokens += 1
                break

        accepted_counts.append(accepted)

        if accepted == num_speculative:
            bonus_probs = target_model.get_probs(context)
            bonus_token = np.random.choice(draft_model.vocab_size, p=bonus_probs)
            context.append(bonus_token)
            total_tokens += 1

    sequential_cost = total_tokens * target_cost
    return {
        "total_tokens": total_tokens,
        "speculative_cost": total_cost,
        "sequential_cost": sequential_cost,
        "speedup": sequential_cost / total_cost if total_cost > 0 else 1.0,
        "avg_accepted": np.mean(accepted_counts),
        "acceptance_rate": np.mean(accepted_counts) / num_speculative,
    }


def compare_speculation_strategies(vocab_size=1000, num_trials=20):
    results = {}

    for name, acceptance_rate, spec_tokens in [
        ("Draft-target (8B->70B)", 0.78, 5),
        ("EAGLE", 0.85, 6),
        ("N-gram", 0.50, 4),
        ("No speculation", 0.0, 0),
    ]:
        if spec_tokens == 0:
            results[name] = {
                "speedup": 1.0,
                "acceptance_rate": 0.0,
                "avg_accepted": 0.0,
            }
            continue

        trial_results = []
        for _ in range(num_trials):
            draft = DraftModel(vocab_size, acceptance_rate=acceptance_rate)
            target = TargetModel(vocab_size)
            context = list(np.random.randint(0, vocab_size, size=10))
            result = speculative_decode(draft, target, context, num_speculative=spec_tokens)
            trial_results.append(result)

        results[name] = {
            "speedup": np.mean([r["speedup"] for r in trial_results]),
            "acceptance_rate": np.mean([r["acceptance_rate"] for r in trial_results]),
            "avg_accepted": np.mean([r["avg_accepted"] for r in trial_results]),
        }

    return results
```

### 第 6 步：KV Cache Memory Profiler

计算真实 model configurations 的 KV cache memory requirements。

```python
MODEL_CONFIGS = {
    "Llama-3-8B": {
        "num_layers": 32, "num_kv_heads": 8, "head_dim": 128,
        "model_params_b": 8, "gqa": True,
    },
    "Llama-3-70B": {
        "num_layers": 80, "num_kv_heads": 8, "head_dim": 128,
        "model_params_b": 70, "gqa": True,
    },
    "Llama-3-405B": {
        "num_layers": 126, "num_kv_heads": 8, "head_dim": 128,
        "model_params_b": 405, "gqa": True,
    },
    "Mistral-7B": {
        "num_layers": 32, "num_kv_heads": 8, "head_dim": 128,
        "model_params_b": 7, "gqa": True,
    },
    "GPT-4-est": {
        "num_layers": 120, "num_kv_heads": 96, "head_dim": 128,
        "model_params_b": 1800, "gqa": False,
    },
}


def kv_cache_memory(config, seq_len, dtype_bytes=2):
    per_token = 2 * config["num_layers"] * config["num_kv_heads"] * config["head_dim"] * dtype_bytes
    total = per_token * seq_len
    return {
        "per_token_bytes": per_token,
        "per_token_kb": per_token / 1024,
        "total_bytes": total,
        "total_mb": total / (1024 ** 2),
        "total_gb": total / (1024 ** 3),
    }


def memory_budget(config, gpu_memory_gb, model_dtype_bytes=2, kv_dtype_bytes=2):
    model_memory_gb = config["model_params_b"] * 1e9 * model_dtype_bytes / (1024 ** 3)
    overhead_gb = gpu_memory_gb * 0.1
    available_for_kv = gpu_memory_gb - model_memory_gb - overhead_gb

    if available_for_kv <= 0:
        return {"error": "Model does not fit in GPU memory", "model_memory_gb": model_memory_gb}

    per_token = 2 * config["num_layers"] * config["num_kv_heads"] * config["head_dim"] * kv_dtype_bytes
    max_tokens = int(available_for_kv * (1024 ** 3) / per_token)

    return {
        "gpu_memory_gb": gpu_memory_gb,
        "model_memory_gb": round(model_memory_gb, 1),
        "overhead_gb": round(overhead_gb, 1),
        "available_for_kv_gb": round(available_for_kv, 1),
        "max_total_tokens": max_tokens,
        "max_users_at_2k": max_tokens // 2048,
        "max_users_at_4k": max_tokens // 4096,
        "max_users_at_32k": max_tokens // 32768,
    }
```

## 使用它

使用 vLLM：

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="meta-llama/Llama-3-70B-Instruct",
    tensor_parallel_size=4,
    enable_prefix_caching=True,
    max_model_len=8192,
    gpu_memory_utilization=0.9,
)

params = SamplingParams(temperature=0.7, max_tokens=256)
outputs = llm.generate(["Explain inference optimization in one paragraph."], params)
```

使用 SGLang 做 prefix caching + structured output：

```python
import sglang as sgl

@sgl.function
def classify(s, text):
    s += sgl.system("You are a classifier. Output JSON only.")
    s += sgl.user(f"Classify this text: {text}")
    s += sgl.assistant(sgl.gen("result", regex=r'\{"label": "(positive|negative|neutral)"\}'))

runtime = sgl.Runtime(model_path="meta-llama/Llama-3-70B-Instruct", tp_size=4)
sgl.set_default_backend(runtime)

results = classify.run_batch([
    {"text": "This product is amazing!"},
    {"text": "Terrible experience."},
    {"text": "It was okay I guess."},
])
```

使用 TensorRT-LLM：

```python
import tensorrt_llm
from tensorrt_llm.runtime import ModelRunner

runner = ModelRunner.from_dir("./llama-70b-trt-engine/", rank=0)

outputs = runner.generate(
    batch_input_ids=[tokenizer.encode("Explain KV caching.")],
    max_new_tokens=256,
    temperature=0.7,
)
```

## 交付它

本课会产出：

- `outputs/skill-inference-optimization.md`，用于诊断和优化 LLM inference serving 的 skill

## 练习

1. 修改 KV cache profiler，比较 FP16 vs FP8 vs INT4 KV cache quantization。对 Llama 3 70B 在 4K context 下，计算 4xA100-80GB 上每种精度的最大 concurrent users。KV quantization 到 INT4 应大约把 user capacity 提高 4 倍。

2. 扩展 continuous batching simulator，跟踪 GPU utilization（每 step 中被填充的 batch slots 比例）。对 50 个 output lengths 服从 Pareto distribution（shape=1.5, scale=20）的 requests，绘制 static 和 continuous batching 的 utilization over time。continuous batching 应保持 >80% utilization。

3. 实现 grouped-query attention（GQA）版本的 KV cache，其中 `num_kv_heads < num_query_heads`。Llama 3 70B 使用 64 query heads，但只有 8 KV heads。计算相对 full multi-head attention 的 memory savings（KV cache size 减少 8 倍）。

4. 构建使用 LRU eviction 的 prefix cache。设置 max_entries 为 500，生成 1,000 个 requests，其中 60% 共享 5 个 common prefixes 之一。测量 hit rate，并与 unlimited cache 比较。好的 eviction 下 hit rate 应保持高于 55%。

5. 扩展 speculative decoding simulator，实现 tree-based speculation（EAGLE-2 风格）。不要生成 K 个 draft tokens 的单链，而是生成候选树（例如 3 层、每层 2 个分支 = 8 个 leaf candidates）。比较每轮 verification 接受的 total tokens 与 linear speculation。

## 关键词

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| Prefill | “处理 prompt” | 并行计算所有 input tokens 上的 attention；compute-bound，因为完整 matrix multiplication 让 GPU cores 忙碌 |
| Decode | “生成 tokens” | 每次 forward pass 产生一个 token，每次读取完整 model weights；memory-bound，因为 compute 在下一批 weights 到达前就完成了 |
| KV cache | “缓存 attention states” | 存储所有 previous tokens 的 key 和 value projections，使每个 decode step 不用重复计算；用内存换 compute |
| Continuous batching | “Dynamic batching” | 任意 request 完成后立刻把新 requests 插入运行中的 batch，每次 decode iteration 都重新评估，而不是等待整批结束 |
| PagedAttention | “KV cache 的 virtual memory” | 用固定大小 pages 而不是 contiguous blocks 分配 KV cache，消除 memory fragmentation，并为 shared prefixes 启用 copy-on-write |
| Speculative decoding | “Draft and verify” | 用快速 draft model 提议多个 tokens，再用一次 target model forward pass 验证它们；数学上精确，2-3 倍 speedup |
| EAGLE | “Self-speculative decoding” | speculative decoding 变体，在 target model 自己的 hidden states 上训练 lightweight head，比单独 draft model 有更高 acceptance rates |
| Prefix caching | “复用 system prompt KV” | 存储 common prefixes（system prompts、few-shot examples）的 KV cache entries，并跨 requests 复用，跳过重复 prefill |
| Ops:byte ratio | “Arithmetic intensity” | compute operations 与读取 memory bytes 的比例，决定 workload 是 compute-bound（高 ratio）还是 memory-bound（低 ratio） |
| Time to first token | “TTFT” | 从接收 request 到产生第一个 output token 的 latency；long prompts 下主要由 prefill time 决定 |

## 延伸阅读

- Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention" (2023) -- 引入 paged KV cache management 的 vLLM 论文，现已成为 inference serving 行业标准
- Leviathan et al., "Fast Inference from Transformers via Speculative Decoding" (2023) -- 证明 draft-verify speculation 产生精确 target model distributions 并实现 2-3 倍 speedup 的基础论文
- Li et al., "EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty" (2024) -- 通过在 target model 自身 features 上训练 head，而不是使用单独 draft model，获得更高 acceptance rates
- Zheng et al., "SGLang: Efficient Execution of Structured Language Model Programs" (2024) -- 引入 RadixAttention 做 prefix caching，并提供 multi-call LLM programs 的 programming model
- Williams et al., "Roofline: An Insightful Visual Performance Model for Multicore Architectures" (2009) -- 原始 roofline paper，形式化 ops:byte framework，用于推理 compute vs memory bottlenecks
