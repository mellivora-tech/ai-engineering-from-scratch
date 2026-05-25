# InternVL3：Native Multimodal Pretraining

> InternVL3 之前的每个 open VLM 都遵循同一个三步 recipe：拿一个在 trillions of text tokens 上训练好的 text LLM，接上 vision encoder，再 fine-tune 接缝。这能工作，但有 alignment debt：text LLM 把完整 pretraining budget 都花在纯文本上，并不原生理解 visual tokens。当你 post-hoc 加入视觉时，LLM 必须重新学习如何把 visual input 关联到 text reasoning，同时还不能忘掉文本。InternVL3（Zhu 等人，2025 年 4 月）拒绝 post-hoc approach：一次 pretraining run，从第一步开始交错 text 和 multimodal。结果是在 78B open params 下匹配 Gemini 2.5 Pro 的 MMMU-Pro。本课阅读 native pretraining 的理由，以及选择它后会改变什么。

**类型：** 学习
**语言：** Python（stdlib，training-corpus mixer）
**前置要求：** 阶段 12 · 05，阶段 12 · 07（recipes）
**时间：** ~120 分钟

## 学习目标

- 解释为什么 post-hoc VLM training 会积累 alignment debt，并引用三个可测症状（catastrophic forgetting、answer drift、visual-text inconsistency）。
- 描述 InternVL3 的 native pretraining corpus mix，以及为什么 text : interleaved : caption 的比例重要。
- 比较 V2PE（variable visual position encoding）与 Qwen2-VL 的 M-RoPE。
- 说出 Visual Resolution Router（ViR）和 Decoupled Vision-Language（DvD）deployment optimizations。

## 问题

Post-hoc VLM training 是默认做法。LLaVA、BLIP-2、Qwen-VL、Idefics 都拿一个已经 pretrained LLM（Llama、Vicuna、Qwen、Mistral）并加入视觉。训练阶段通常是：

1. Frozen LLM + frozen vision encoder + trainable projector，在 caption pairs 上训练以对齐 embeddings。
2. Unfreeze LLM，在 instruction data（LLaVA-Instruct、ShareGPT4V）上训练。
3. 可选 task-specific fine-tune。

Alignment debt 会出现三个症状：

- Catastrophic forgetting。Post-hoc VLM 忘记 text-only skills。GSM8K 分数下降 5-10 分。Hellaswag 下降。纯文本 agents 回退。
- Answer drift。同一个视觉问题的小幅措辞变化会得到不同答案。Vision encoder 与 LLM 的连接比 LLM 自身 token 的绑定更弱。
- Visual-text inconsistency。VLM 可以正确描述图像，然后又回答出与自己描述矛盾的问题。Visual tokens 没有像 text tokens 一样参与 LLM 内部 consistency checks。

这些症状都有充分记录。MM1.5 第 4 节量化了它们。LLaVA-OneVision 的 ablations 也暗示了它们。Native pretraining 是答案。

## 概念

### Native multimodal pretraining

InternVL3 从零开始在一个原生多模态 corpus 上训练。Mix 是：

- 40% text-only data（FineWeb、Proof-Pile-2 等）
- 35% interleaved image-text data（OBELICS、MMC4-style）
- 20% paired image-caption data
- 5% video-text data

Vision tokens、text tokens 和 cross-modal interactions 从第一个梯度步骤开始就参与同一个 loss。没有 alignment pretraining，没有 projector freezing stage，也没有需要恢复的 catastrophic forgetting。

Base model 只需一个阶段训练。Instruction tuning 随后进行，但 base model 已经把 visual tokens 当作 first-class citizens 理解。

### V2PE（variable visual position encoding）

Qwen2-VL 使用固定 axis allocation 的 M-RoPE。InternVL3 引入 V2PE：position encoding 按 modality type（text、image、video）变化，并带 learnable scaling。实践中：

- Text tokens 获得 1D position（text index）。
- Image patches 获得 2D position（row, col）。
- Video frames 获得 3D position（time, row, col）。

三者共享同一个 RoPE frequency base，但每个 band 的 hidden-dim allocation 是 learned parameter，而不是固定拆分。预训练期间，模型可以自由权衡 temporal vs spatial frequency resolution。

V2PE 的 ablation claim：在相同 compute 下，相比 M-RoPE，video benchmarks 提升 1-2 分。不是革命，但更干净。

### Visual Resolution Router（ViR）

Deployment optimization。不是所有图像都需要 full-resolution encoding。一张只有一个低细节物体的照片，如果按 1280px native 编码就是浪费 token。ViR 是一个小 classifier，会在编码前预测回答问题所需的最低分辨率。

Routing 有三档：low-res（256 tokens）、medium（576）、high（2048+）。生产流量中 60% queries 用 low 或 medium 就够。净效果：相同质量下吞吐提升 2-3 倍。

### Decoupled Vision-Language deployment（DvD）

服务大型 VLM 时，vision encoder 每张图运行一次，而 LLM 对每个输出 token autoregressively 运行。两个组件的瓶颈不同（vision = GPU memory bandwidth for conv + attention；LLM = KV cache）。DvD 把它们拆到不同 GPU 上，并在两者之间 streaming。

对于 8B + 400M encoder 模型，DvD 相比 co-located 每节点吞吐大约翻倍。

### Single-stage vs multi-stage quality

InternVL3 的主要 benchmark claim：78B params 时匹配 Gemini 2.5 Pro 的 MMMU-Pro。38B 时匹配 GPT-4o。8B 时领先 open-8B leaderboard。全部基于 single-stage pretrain + instruction-tune recipe。

Alignment-debt hypothesis 是可测的：相对于 Qwen2.5-VL-7B，InternVL3-8B 每获得一单位 vision-benchmark gain，损失的 text-benchmark points（MMLU、GSM8K）更少。模型更像 generalist，因为训练是一整块，而不是两块拼接。

### InternVL3.5 与 InternVL-U

InternVL3.5（2025 年 8 月）扩展了 recipe。同样 native-pretrain approach，更多数据、更多参数。MMMU improvements 是增量的。

InternVL-U（2026）加入 unified generation，即在同一 backbone 上通过 MMDiT heads 输出图像。“U” 代表 “Understanding + generation”，追逐 Transfusion-style unified models（第 12.13 课）。同一个 native-pretrain backbone 同时支持 understanding 和 generation heads。

### Native pretraining 的 trade-offs

Native pretraining 不是免费的：

- Compute。从零训练一个新 VLM 的成本和训练 text LLM 相同，即数百万 GPU-hours。Post-hoc adaptation 复用已有 LLM 权重，节省大部分成本。
- Data。大规模 interleaved image-text corpora 很稀缺。OBELICS 有 141M documents；MMC4 有 571M。纯文本有 15T tokens。Multimodal pretraining data scarcity 是硬约束。
- Base-LLM reuse。Native pretraining 放弃了日后 drop in 新 LLM 的选项。Post-hoc 允许你只 retrain adapter，就把 Llama-3.1 换成 Llama-4。

InternVL3 的赌注是：alignment debt 比 reuse loss 更糟。Benchmarks 支持这个 claim。生产成本的门槛会阻止未来 lab 低成本复制。Post-hoc VLM 仍会继续存在，因为它对大多数项目更便宜。

## 使用它

`code/main.py` 是 training-corpus mixer 和 ViR router simulator。它会：

- 接收目标 corpus mix（%text、%interleaved、%caption、%video），并计算每种 modality 的 expected steps。
- 在一批 queries 上模拟 ViR routing（分布：50% low-detail、30% medium、20% high-detail），并报告 average token count。
- 给定 encoder vs LLM FLOPs，报告 DvD throughput estimates。
- 打印 post-hoc vs native pretraining 的 side-by-side，对比 params、compute、data 和 expected alignment-debt symptoms。

## 交付它

本课产出 `outputs/skill-native-vs-posthoc-auditor.md`。给定一个拟议 VLM training plan，它会审计应该走 native 还是 post-hoc，标记 alignment-debt risk，并推荐 corpus mix。当你为一个新的 open-VLM project 做 sizing 并选择训练策略时使用它。

## 练习

1. 估算 InternVL3-8B（native pretrain）与 LLaVA-OneVision-7B（post-hoc）之间的 compute delta。GPU-hours 比例大约是多少？差距来自哪里？

2. InternVL3 报告 40% text / 35% interleaved / 20% caption / 5% video。如果目标任务 video-heavy，提出新的比例，并说明为什么 base model 仍需要大量 text 和 caption data。

3. 阅读 MM1.5 第 4 节关于 forgetting 的内容。说出 post-hoc training 中回退最大的具体 benchmark。回退了多少？

4. ViR 把 60% 流量路由到低分辨率编码。它会误路由哪些 queries（需要 high-res 却送去 low-res）？提出三个 router-failure modes。

5. DvD 把 vision 和 LLM 拆到不同 GPU。什么 traffic pattern 下 DvD 会伤害吞吐，而不是帮助？

## 关键术语

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| Native multimodal pretraining | “From scratch together” | Text + image + video tokens 从第 1 步起参与 loss，而不是之后接上 |
| Alignment debt | “Post-hoc penalty” | 把视觉接到 frozen LLM 后产生的 text skills 与 answer consistency 可测回退 |
| V2PE | “Variable visual pos encoding” | Per-modality learnable position encoding allocation；InternVL3 的 M-RoPE 后继 |
| ViR | “Resolution router” | 在编码前按 query 选择所需最低 resolution 的小 classifier，用于节省 inference tokens |
| DvD | “Decoupled deployment” | Vision encoder 在一张 GPU，LLM 在另一张 GPU，中间 stream handoff；大 VLM 吞吐翻倍 |
| InternVL-U | “Unified understanding + generation” | 2026 follow-up，在 native-pretrain backbone 上加入 image-generation heads |
| Interleaved corpus | “OBELICS / MMC4” | 以自然 reading order 包含文本和图像的 documents；native pretraining 的原料 |

## 延伸阅读

- [Chen et al. — InternVL 1 (arXiv:2312.14238)](https://arxiv.org/abs/2312.14238)
- [Zhu et al. — InternVL3 (arXiv:2504.10479)](https://arxiv.org/abs/2504.10479)
- [InternVL3.5 (arXiv:2508.18265)](https://arxiv.org/abs/2508.18265)
- [InternVL-U (arXiv:2603.09877)](https://arxiv.org/abs/2603.09877)
- [Zhang et al. — MM1.5 (arXiv:2409.20566)](https://arxiv.org/abs/2409.20566)
