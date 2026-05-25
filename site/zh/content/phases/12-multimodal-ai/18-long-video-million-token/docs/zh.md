# Million-Token Context 下的 Long-Video Understanding

> 一个 1 小时 4K、24 FPS 的视频，patch 并 embedding 后会产生约 6000 万 tokens。一个 2 小时 podcast episode 的 transcript 是 30,000 tokens。一部长篇 Blu-ray 电影，即使 aggressive pooling，也有数十万 tokens。Google 的 Gemini 1.5（2024 年 3 月）以 10-million-token context 开启了这个时代，能在小时级视频上可靠做 needle-in-a-haystack recall。LWM（Liu 等人，2024 年 2 月）展示了 ring attention 的 scaling path。LongVILA 和 Video-XL 进一步扩展 ingestion。VideoAgent 用 agentic retrieval 替换 raw context。每种路线都在 compute、recall 和 engineering complexity 上做不同取舍。本课把它们并排阅读。

**类型：** 构建
**语言：** Python（stdlib，needle-in-haystack simulator + agentic-retrieval router）
**前置要求：** 阶段 12 · 17（video temporal tokens）
**时间：** ~180 分钟

## 学习目标

- 计算不同 FPS 和 pooling 下 long-form video 的 total visual-token counts。
- 解释三条 scaling paths：brute context（Gemini 1.5）、ring attention（LWM）、token compression（LongVILA / Video-XL）。
- 在 accuracy 和 latency 上比较 raw-context video VLMs 与 agentic-retrieval video VLMs（VideoAgent）。
- 为 30 分钟视频设计 needle-in-a-haystack test，并测量特定分钟的 recall。

## 问题

Qwen2.5-VL 规模下 384 native resolution 的单帧 patches 约为 729 tokens。3x3 pooling 后是每帧 81 tokens。30 分钟 clip at 1 FPS = 1800 frames = 145,800 tokens。2025 年 open VLMs 可处理，但紧张。2 FPS 时是 291,600 tokens，只有最大 context 装得下。

2 小时电影 at 1 FPS 是 583k tokens。超过大多数 2026 open models；需要 Gemini 2.5 Pro，或更激进 pooling。

出现了三条 scaling paths。

## 概念

### Path 1：Brute context（Gemini 1.5、Claude Opus）

用硬件解决问题。把 context 扩展到数百万 tokens，在一次 forward pass 中处理全部内容。

Gemini 1.5 Pro 发布时有 1M tokens；Gemini 1.5 Ultra 到 10M；2026 年 Gemini 2.5 Pro 能可靠处理数小时视频。论文（arXiv:2403.05530）记录了直到约 9.5M tokens 时 needle-in-a-haystack recall 仍为 99.7%。

工程：自定义 attention implementation，带 memory hierarchy（local + global + sparse），再加 MoE expert routing 做 long-context efficiency。未完整公开。不开源。

### Path 2：Ring attention（LWM、LongVILA）

Ring attention 把长序列分布到多个设备，每个设备持有一个 chunk。Full-sequence attention 通过 ring pattern 实现：每个设备把自己的 chunk 发送给下一个设备，计算 partial attention，然后聚合。

LWM（Liu 等人，2024）用这种方式训练了 1M-token context model。Training compute 随 context 线性扩展，而不是二次扩展，因为 attention 的二次开销被 ring 中的设备摊销。

LongVILA（arXiv:2408.10188）把这个模式适配到 VLM。1400-frame videos、每帧 192 tokens = 268k context，使用 8-way parallelism 的 ring attention 训练。

### Path 3：Token compression（Video-XL、LongVA）

比 brute context 更便宜：在 LLM 看到序列前先 aggressive compression。

Video-XL（arXiv:2409.14485）使用 visual summary token：每个 N 帧 clip 产生一个 “summary” token，attend 到这 N 帧。Inference 时 LLM 看到每个 clip 一个 summary token，大幅缩小 context。

LongVA 用 “long context transfer” 技术把 LLM context 从 200k 扩展到 2M。在 long-context text 上训练，再通过 shared representation 转移到 long-context video。

Token compression 用特定 timestamp 的 recall 换可扩展性。模型通常知道发生了什么，但有时会错过确切帧。

### Path 4：Agentic retrieval（VideoAgent）

不要把完整视频喂给 LLM。把视频当成数据库，让 LLM 查询它。

VideoAgent（arXiv:2403.10517）：

1. LLM 阅读问题。
2. LLM 向 retrieval tool 请求相关 clips（“show me segments with a cat”）。
3. Tool 返回匹配 clip timestamps。
4. LLM 通过 VLM 阅读这些 clips。
5. LLM 组合答案，或提出 follow-up queries。

这是把 LLM-as-agent pattern 应用到 long video。Inference 更便宜（只编码相关 clips），工程更难（retrieval quality 成为 bottleneck）。

### Needle-in-a-haystack benchmarks

标准 long-context test：把一个独特 visual 或 textual marker 插入视频随机点，然后询问一个需要回忆它的问题。

Metric：跨 video length 与 marker position 的 Recall@k。

Gemini 2.5 Pro 在最高 90 分钟视频上 recall >99%。Open 72B models（Qwen2.5-VL-72B、InternVL3-78B）在 30 分钟约 85-90%，超过 60 分钟后下降。

如果 retrieval tool 好，VideoAgent 在 2+ 小时内容上能匹配或超过 raw-context models，因为 retrieval 会命中 needle。

### 如何选择路线

15 分钟 clip 且需要 frontier accuracy：open 72B + native context 通常可以。选 Qwen2.5-VL-72B。

30 分钟到 1 小时内容：open 用 LongVILA 或 Video-XL；closed 用 Gemini 2.5 Pro。质量门槛重要时 frontier 走 closed。

2+ 小时内容：VideoAgent 或类似 retrieval patterns。或者 summarize 成更小 chunks，再喂 hierarchical summaries。

### 2026 production pattern

实践中，production long-video pipelines 通常是 hybrid：

1. 对完整视频运行 dynamic-FPS sampling + aggressive pooling（得到 100k-token global representation）。
2. 传给 72B VLM 生成 global summary。
3. 如果用户问细节问题，用 summary 作为 index 运行 agentic retrieval。

这结合了 brute-context 的 global understanding 与 retrieval 的 local detail。

## 使用它

`code/main.py`：

- 计算 1 分钟到 3 小时视频在不同 FPS + pooling 下的 token budgets。
- 模拟 needle-in-a-haystack run：在随机 timestamp 注入 marker，提出问题，计算 recall。
- 包含 agentic-retrieval router simulator，选择要送给下游 VLM 的 specific clips。

运行 budget table，感受 scale gap。

## 交付它

本课产出 `outputs/skill-long-video-strategy-planner.md`。给定 video duration 和 query complexity，它会在 brute-context、compression 和 agentic retrieval 之间选择，并计算 latency + quality expectations。

## 练习

1. 一个 45 分钟 lecture，1 FPS，每帧 81 tokens。总 tokens？能放进哪些模型 context？

2. 设计 needle-in-a-haystack test：你在第几分钟注入 marker，query format 的确切形式是什么？

3. 对 1 小时视频比较 brute-context Qwen2.5-VL-72B（80k context）与 VideoAgent（Claude 3.5 + retrieval）。哪个 recall 胜出？哪个 latency 胜出？

4. Ring attention 的 memory cost 随 sequence length 和 device count 线性扩展。解释原因，以及如果去掉 ring-rotation phase 会坏在哪里。

5. 阅读 Gemini 1.5 第 5 节关于 needle-in-a-haystack 的内容。论文发现 1M 与 10M token boundary 上 recall 有什么变化？

## 关键术语

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| Brute context | “Just more tokens” | 把 LLM context 扩到数百万 tokens；一次 pass 处理全部内容 |
| Ring attention | “LWM-style parallel” | 分布式 attention pattern，每个设备持有一个 chunk 并轮转 |
| Token compression | “Summary tokens” | 在 LLM 前通过 learned compressor 减少每个 clip 的 tokens |
| Needle-in-haystack | “NIH test” | 在随机点插入 unique marker，测试时要求模型回忆 |
| Agentic retrieval | “LLM as query planner” | LLM 向 retrieval tool 请求相关 clips，经 VLM 阅读后组合答案 |
| VideoAgent | “Retrieval pattern for video” | 标准 agentic-retrieval design：question -> tool -> clip -> answer |

## 延伸阅读

- [Gemini Team — Gemini 1.5 (arXiv:2403.05530)](https://arxiv.org/abs/2403.05530)
- [Liu et al. — LWM / RingAttention (arXiv:2402.08268)](https://arxiv.org/abs/2402.08268)
- [Xue et al. — LongVILA (arXiv:2408.10188)](https://arxiv.org/abs/2408.10188)
- [Shu et al. — Video-XL (arXiv:2409.14485)](https://arxiv.org/abs/2409.14485)
- [Wang et al. — VideoAgent (arXiv:2403.10517)](https://arxiv.org/abs/2403.10517)
