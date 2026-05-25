# Open-Weight VLM Recipes：真正重要的是什么

> 2024-2026 年的 open-weight VLM 文献是一片 ablation table 森林。Apple 的 MM1 测试了 image encoder、connector 和 data mix 的 13 种组合。Allen AI 的 Molmo 证明详细的人类 caption 优于 GPT-4V distillation。Cambrian-1 做了 20+ encoder 对比。Idefics2 形式化了五轴设计空间。Prismatic VLMs 在受控 benchmark 上比较了 27 种训练 recipe。在这些噪声里，有一小组结论横跨多篇论文都成立：image encoder 比 connector architecture 更重要，data mixture 比两者都重要，详细人类 caption 优于 distilled synthetic data。本课替你读这些表。

**类型：** 学习 + 实验
**语言：** Python（stdlib，ablation table parser + recipe picker）
**前置要求：** 阶段 12 · 05（LLaVA baseline）
**时间：** ~180 分钟

## 学习目标

- 说出 VLM 设计空间的五个轴：image encoder、connector、LLM、data mix、resolution schedule。
- 阅读 MM1 / Idefics2 / Cambrian-1 ablation table，并预测哪个旋钮会移动某个 benchmark。
- 给定 compute budget 和 task mix，为新 VLM 选择 recipe（encoder、connector、data、resolution）。
- 解释为什么在相同 token count 下，详细人类 captions 优于 GPT-4V distillation。

## 问题

Open-weight VLM 有几百个。大多数“还不错”与“state-of-the-art”的差距不在架构，而在数据、resolution schedule 和 encoder 选择。当模型表现不佳时，知道先转哪个旋钮，可以帮你省掉一次 5-million-GPU-hour 的错误。

2023 年浪潮（LLaVA-1.5、InstructBLIP、MiniGPT-4）运行在 caption-pair pretraining + LLaVA-Instruct-150k 上。是不错 baseline。MMMU 大约封顶在 35%。

2024 年浪潮（MM1、Idefics2、Molmo、Cambrian-1、Prismatic VLMs）做了详尽 ablations。结果既意外又实用。

## 概念

### 五轴设计空间

Idefics2（Laurençon 等人，2024）命名了这些轴：

1. Image encoder。CLIP ViT-L/14、SigLIP SO400m/14、DINOv2 ViT-g/14、InternViT-6B。Encoder 在 patch size、resolution 和 pretraining objective 上不同。
2. Connector。MLP（2-4 层）、Q-Former（32 queries + cross-attn）、Perceiver Resampler（64 queries）、C-Abstractor（convolutional + bilinear pooling）。
3. Language model。Llama-3 8B / 70B、Mistral 7B、Phi-3、Gemma-2、Qwen2.5。LLM size 是主导 parameter cost。
4. Training data。Caption pairs（CC3M、LAION）、interleaved（OBELICS、MMC4）、instruction（LLaVA-Instruct、ShareGPT4V、PixMo、Cauldron）。
5. Resolution schedule。Fixed 224/336/448、AnyRes、native dynamic。训练中逐步升高或保持常量。

每个生产 VLM 都在每个轴上做了选择。MMMU 分数的大部分 variance 由轴 1、4、5 解释，而不是你选了哪个 connector。

### Axis 1：encoder > connector

MM1 第 3.2 节显示：从 CLIP ViT-L/14 换成 SigLIP SO400m/14，MMMU 增加 3+ 分。把 connector 从 MLP 换成 Perceiver Resampler，增加不到 1 分。Idefics2 复现了同样结论：SigLIP > CLIP，在相同 token count 下 Q-Former ≈ MLP ≈ Perceiver。

Cambrian-1 的 “Cambrian Vision Encoders Match-Up”（Tong 等人，2024）在 vision-centric benchmark（CV-Bench）上跑了 20+ encoders。Leaderboard 顶部是 DINOv2 与 SigLIP 的混合；CLIP 在中游；ImageBind 和 ViT-MAE 更低。从 CLIP ViT-L 到 DINOv2 ViT-g/14，在 CV-Bench 上差距约 5-7 分。

2026 年 open VLM 的默认 encoder 是 SigLIP 2 SO400m/14，用于 semantic + dense features；如果需要 segmentation/grounding，有时会拼接 DINOv2 ViT-g/14 features（Cambrian 的 “Spatial Vision Aggregator” 就这样做）。

### Axis 2：connector design 基本打平

MM1、Idefics2、Prismatic 和 MM-Interleaved 都得出同样结论：在固定 visual-token count 下，connector architecture 几乎不重要。相同 token budget 下，对 mean-pooled patches 使用 2-layer MLP 的表现，与 32-query Q-Former 相差不到 1 分。

真正重要的是 token count。更多 visual tokens = 更多 LLM compute = 在某个点前更好，然后收益递减。每张图 64 tokens 对 OCR 太少。576-1024 tokens 是多数 open VLM 的甜点区。2048+ 只对 documents 和 charts 有帮助。

Q-Former vs MLP 是成本问题，不是质量问题：Q-Former 不管图像分辨率如何都把 token 限制在 32-64；MLP 输出全部 patch tokens。高分辨率输入时 Q-Former 节省 LLM context；低分辨率时差别只是噪声。

### Axis 3：LLM size 决定上限

把 LLM 从 7B 翻到 13B，几乎每篇 VLM 论文都稳定给 MMMU 增加 2-4 分。到 70B 时，大多数 benchmark 开始饱和。VLM 的 multimodal reasoning ceiling 就是 LLM 的 text reasoning ceiling；visual encoder 只能喂信息给它，不能替它推理。

这就是为什么 Qwen2.5-VL-72B 和 Claude Opus 4.7 能碾压 MMMU-Pro 与 ScreenSpot-Pro：语言大脑很大。7B VLM 不能靠聪明 connector design 取代 70B VLM。

### Axis 4：data — 详细人类 captions 胜过 distillation

Molmo + PixMo（Deitke 等人，2024）是每个人都该读的 2024 结果。Allen AI 让人类标注者用 1-3 分钟 dense speech-to-text 方式描述图像，得到 712K densely-captioned images。训练数据中完全没有 GPT-4V distillation。

Molmo-72B 在 11/11 个 benchmarks 上超过 Llama-3.2-90B-Vision。差距不在架构，而在 caption quality。详细人类 captions 每张图包含的信息量比短 web captions 多 5-10 倍，而且在 GPT-4V distillation 会 hallucinate 的地方保持事实 grounded。

ShareGPT4V（Chen 等人，2023）和 Cauldron（Idefics2）沿着同样路线，混合 human + GPT-4V captions。趋势很清楚：对于 2026 年 frontier，caption density > caption quantity > distillation convenience。

### Axis 5：resolution 与 schedule

Idefics2 ablations：384 -> 448 增加 1-2 分。448 -> 980 且使用 image splitting（AnyRes）在 OCR benchmarks 上再增加 3-5 分。Flat resolution training 在中等 accuracy 处平台化；resolution ramping（从 224 开始，最后到 448 或 native）训练更快，最终更高。

Cambrian-1 跑了 resolution vs tokens trade-off：在固定 compute 下，你可以选择更多低分辨率 token，或更少高分辨率 token。OCR 中高分辨率胜出；general scene understanding 中低分辨率更多 token 胜出。

2026 年生产 recipe：Stage 1 用 384 fixed 训练；Stage 2 对 OCR-heavy tasks 使用最高 1280 的 dynamic resolution。

### Prismatic 的受控对比

Prismatic VLMs（Karamcheti 等人，2024）是控制所有轴的论文。同一个 13B LLM、同一份 instruction data、同一套 evaluation，每次只变一个轴。结果：

- Per-image visual-token count 解释约 60% variance。
- Encoder choice 解释约 20%。
- Connector architecture 解释约 5%。
- 其他（data mix、scheduler、LR）解释剩下约 15%。

这是一个粗略分解，但它是文献中对“我应该先 ablate 什么”的最干净回答。

### 2026 年 picker

基于证据，2026 年新项目的默认 open-VLM recipe：

- Encoder：native resolution + NaFlex 下的 SigLIP 2 SO400m/14；如果需要 dense features，拼接 DINOv2 ViT-g/14。
- Connector：对 patch tokens 使用 2-layer MLP。除非 token-constrained，否则跳过 Q-Former。
- LLM：Qwen2.5 / Llama-3.1 / Gemma 2；成本选 7B，质量选 70B，由目标 latency 决定。
- Data：PixMo + ShareGPT4V + Cauldron，再补充 task-specific instruction data。
- Resolution：dynamic（min 256，max 1280 pixels per long side）。
- Schedule：Stage 1 alignment（projector-only）、Stage 2 full fine-tune、Stage 3 task-specific fine-tune。

这些默认项中的每一个，都可以追溯到本课末尾引用论文中的 measured ablation。

## 使用它

`code/main.py` 是 ablation table parser 和 recipe picker。它编码了 MM1 和 Idefics2 的 ablation tables（压缩版），并允许你查询：

- “给定 budget X 和 task Y，哪个 recipe 赢？”
- “如果我在 7B Llama 上把 SigLIP 换成 CLIP，预期 MMMU delta 是多少？”
- “为了得到 80% confidence answer，我应该先 ablate 哪个轴？”

输出是 ranked recipe list，带 expected benchmark deltas 和 “ablate first” 推荐。

## 交付它

本课产出 `outputs/skill-vlm-recipe-picker.md`。给定 target task mix、compute budget 和 latency target，它会输出完整 recipe（encoder、connector、LLM、data mix、resolution schedule），并为每个选择引用支撑它的 ablation。它能阻止工程师每启动一个新 VLM 项目都重新发明 Idefics2 ablation table。

## 练习

1. 阅读 MM1 第 3.2 节。在 fixed 2B LLM、budget 50M images 下，哪个 encoder 胜出？如果换成 13B LLM，答案会反转吗？为什么？

2. Cambrian-1 发现 DINOv2 + SigLIP 拼接在 vision-centric benchmarks 上优于任一单独 encoder，但在 MMMU 上没有额外信号。预测哪些 benchmarks 会提升，哪些保持不变。

3. 你的目标是一个运行在 2B LLM 上的 mobile UI agent。选择 encoder、connector、resolution 和 data mix。用具体 ablation table 为每个选择辩护。

4. Molmo 发布 4B 和 72B 模型。4B 能与 closed 7B VLMs 竞争；72B 在 11/11 个 benchmarks 上超过 Llama-3.2-90B-Vision。这告诉你什么关于 LLM-size plateau hypothesis？

5. 设计一个 ablation table，在 7B VLM 上隔离 data-mix quality 与 encoder quality。最少需要多少 training runs？提出四个 axis settings。

## 关键术语

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| Ablation | “Turning one knob” | 训练多次运行，每次只改变一个 design-space axis，其余全部保持不变 |
| Connector | “Bridge” / “projector” | 把 vision encoder output 映射到 LLM token space 的可训练模块（MLP、Q-Former、Perceiver） |
| Detailed human caption | “Dense caption” | 人类写的多句描述（通常 80-300 tokens），比 web alt text 更丰富 |
| Distillation | “GPT-4V captions” | 由更强 proprietary VLM 生成的训练数据；方便但容易继承 hallucination |
| AnyRes / dynamic res | “High-res path” | 通过 tiling 或 M-RoPE 输入大于 encoder native resolution 的图像 |
| Resolution ramp | “Curriculum” | 从低分辨率开始并逐步增加的训练 schedule，加速 alignment learning |
| Vision-centric bench | “CV-Bench / BLINK” | 强调细粒度视觉感知，而不是 language-heavy reasoning 的 evaluation |
| PixMo | “Molmo's data” | Allen AI 的 712K densely-captioned image dataset；人类语音转写为 dense captions |

## 延伸阅读

- [McKinzie et al. — MM1 (arXiv:2403.09611)](https://arxiv.org/abs/2403.09611)
- [Laurençon et al. — Idefics2 / What matters building VLMs (arXiv:2405.02246)](https://arxiv.org/abs/2405.02246)
- [Deitke et al. — Molmo and PixMo (arXiv:2409.17146)](https://arxiv.org/abs/2409.17146)
- [Tong et al. — Cambrian-1 (arXiv:2406.16860)](https://arxiv.org/abs/2406.16860)
- [Karamcheti et al. — Prismatic VLMs (arXiv:2402.07865)](https://arxiv.org/abs/2402.07865)
