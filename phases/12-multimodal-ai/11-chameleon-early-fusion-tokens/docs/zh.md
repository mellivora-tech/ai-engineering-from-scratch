# Chameleon 与 Early-Fusion Token-Only Multimodal Models

> 到目前为止我们看到的每个 VLM 都把图像和文本分开。Visual tokens 来自 vision encoder，流入 projector，然后在 LLM 内部遇到文本。Vision 和 text vocabulary 从不重叠。Chameleon（Meta，2024 年 5 月）问：如果它们重叠呢？训练一个 VQ-VAE，把图像变成共享 vocabulary 中的离散 token 序列。每个 multimodal document 现在都是一个序列：text tokens 与 image tokens 交错，一个 autoregressive loss。副作用是：模型可以生成 mixed-modality outputs，在一次 inference call 中交替输出 text 和 image tokens。本课阅读 early-fusion thesis，并端到端构建 toy 版本。

**类型：** 构建
**语言：** Python（stdlib，VQ-VAE tokenizer + interleaved decoder）
**前置要求：** 阶段 12 · 05，阶段 8（Generative AI）
**时间：** ~180 分钟

## 学习目标

- 解释 shared vocabulary + single loss 如何改变模型能力。
- 描述 VQ-VAE 如何把图像 tokenize 成与 transformer next-token objective 兼容的离散序列。
- 说出 Chameleon 的 training-stability tricks：QK-Norm、dropout placement、LayerNorm ordering。
- 比较 Chameleon 与 BLIP-2 的 Q-Former 方法，并说明什么时候各自是正确选择。

## 问题

Adapter-based VLM（LLaVA、BLIP-2、Qwen-VL）把文本和图像当成两种不同东西。Text token 经过 `embed(text_token)`；图像经过 `visual_encoder(image) → projector → ... pseudo_tokens`。模型有两条输入路径，中途合并。

三个后果：

1. LLM 只能消费图像，不能发出图像。输出只能是文本。
2. Mixed-modality documents（像文章一样交替出现段落和图片）很别扭，你要么在模型外解析 multimodal input，要么串联多次生成。
3. Distributional mismatch。Visual tokens 和 text tokens 位于 hidden space 的不同区域，造成微妙 alignment 问题。

Chameleon 拒绝这个前提：图像只是共享 vocabulary 中离散 token 的序列。用 interleaved documents 训练模型，一个 loss，一个 autoregressive decoder，于是 mixed-modality generation 自然解锁。

## 概念

### VQ-VAE 作为 image tokenizer

Tokenizer 是 vector-quantized variational autoencoder。架构：

- Encoder：CNN + ViT，把图像映射到 spatial feature map，比如 32x32 个 dim 256 的 features。
- Codebook：K 个向量组成的可学习 vocabulary（Chameleon 使用 8192），同样 dim 256。
- Quantization：对每个 spatial feature，按 L2 distance 查找最近的 codebook entry。用整数 index 替换 continuous feature。
- Decoder：CNN，把 quantized features 还原成 pixels。

训练：VAE reconstruction loss + commitment loss + codebook loss。Codebook indices 构成图像的离散 alphabet。

对于 Chameleon：一张图变成 32*32 = 1024 个 token，来自一个 8192 大小的 vocabulary。再与 text tokens（来自 LLM 的 BPE vocabulary，比如 32000）拼接。最终 vocabulary：40192。Transformer 看到一个序列，一个 loss。

### Shared vocabulary

Chameleon 的 vocabulary 结合 text tokens、image tokens 和 modality separators。每个 token 都有一个单独 ID。Input embedding layer 把每个 ID 映射到 D 维 hidden vector。Output projection 把 hidden 映射回 vocab logits。Softmax 选择下一个 token，无论 modality。

Separators 很重要：`<image>` 和 `</image>` tags 包住 image-token sequence。生成时，如果模型发出 `<image>`，下游软件就知道接下来的 1024 个 token 是要送给 decoder 渲染 pixels 的 VQ indices。

### Mixed-modality generation

Inference 是共享 vocabulary 上的 next-token prediction。示例 prompt：“Draw a cat and describe it.” Chameleon 输出：

```
<image> 4821 1029 2891 ... (1024 image tokens) </image>
The cat is orange, sitting on a windowsill...
```

模型自主选择顺序：它可能先图后文、先文后图，或交错输出。Same decoder，same loss。

对比 adapter VLM，它们 generation 只能是 text-only。Chameleon 重新打开了模型输出 modality 的问题。

### Training stability：QK-Norm、dropout、LayerNorm ordering

Early-fusion training 在大规模下不稳定。Chameleon 论文记录了三个技巧：

- QK-Norm。在 attention 内部、dot product 前，对 query 和 key projections 应用 LayerNorm。防止深层 logit magnitude 爆炸。多个 2024 后大型模型使用它。
- Dropout placement。每次 residual-add 后都加 dropout，而不仅仅是在 attention 和 MLP 后。当 image tokens 的梯度可能占主导时，需要更多 regularization。
- LayerNorm ordering。在 residual branch 上用 Pre-LN（标准），并在最后一个 block 的 skip connection 上加额外 LN。稳定 final-layer gradient flow。

没有这些技巧，34B-param Chameleon 在多个 checkpoint 发散。加上它们后才收敛。训练 recipe 与架构本身同样是贡献。

### Tokenizer 的 reconstruction ceiling

VQ-VAE 是有损的。在 8192 codebook entries、每张 512x512 图 1024 tokens 下，reconstruction PSNR 上限大约 26-28 dB。这足以生成可识别图像，但明显差于 continuous-space diffusion（Stable Diffusion 3 达到 32+ dB）。

Tokenizer 是 bottleneck。更好的 tokenizers（MAGVIT-v2、IBQ、SBER-MoVQGAN）会提高 ceiling。Emu3（第 12.12 课）仅靠更好的 tokenizer 就达到了 SDXL-quality generation。

### Chameleon vs BLIP-2 / LLaVA

Chameleon（early fusion，shared vocab）：
- 一个 loss，一个 decoder。
- 生成 mixed-modality output。
- Tokenizer 是质量上限。
- 昂贵：inference path 上每次生成图像都要运行 VQ-VAE decoder。

BLIP-2 / LLaVA（late fusion，separate towers）：
- Vision in，text out only。
- 复用 pretrained LLM。
- Understanding 没有 tokenizer bottleneck。
- 便宜：single forward pass。

按任务选择。如果需要 image generation，选择 Chameleon family。如果只需要 understanding，adapter-VLM 更简单，也复用更多 pretrained compute。

### Fuyu 与 AnyGPT

Fuyu（Adept，2023）是相关路线：完全跳过单独 vision encoder，把 raw image patches 像 token 一样通过 LLM input projection 输入，没有 tokenizer。比 Chameleon 更简单，但失去了 shared-vocab output generation。

AnyGPT（Zhan 等人，2024）把 Chameleon 扩展到四种 modality：text、image、speech、music。每种 modality 都使用同样 VQ-VAE 技巧，共享 transformer。Any-to-any generation。第 12.16 课会讲更多。

## 使用它

`code/main.py` 构建一个 toy end-to-end early-fusion model：

- 一个小型 VQ-VAE-style quantizer，把 8x8 patches 映射到 codebook indices（K=16）。
- 共享 vocabulary：（text ids 0..31）+（image ids 32..47）+（separators 48, 49）。
- 一个 toy autoregressive decoder（bigram table），在 synthetic captions + image-token sequences 上训练。
- Sampling loop：给定 prompt 后输出交替的 text + image tokens。

代码故意把 transformer 保持得很小（bigrams），方便你端到端追踪 signal flow。

## 交付它

本课产出 `outputs/skill-tokenizer-vs-adapter-picker.md`。给定 product spec（仅理解 vs 理解 + 生成、required image quality、cost budget），它会在 Chameleon-family（early fusion）和 LLaVA-family（late fusion）之间选择，并用定量经验法则解释。

## 练习

1. Chameleon 使用 K=8192 codebook entries，每张 512x512 图 1024 tokens。估算相对 24-bit RGB 图像的 compression ratio。它是 lossy 吗？有多 lossy？

2. 一张 4K 图像（3840x2160）在相同 VQ-VAE 密度下会产生多少 image tokens？Chameleon-style model 能在一次 inference call 中生成 4K 图像吗？最先坏掉的是 context、tokenizer quality 还是 KV cache？

3. 用纯 Python 实现 QK-Norm。给定 64-dim query 和 key，展示 LayerNorm 前后的 dot product。为什么深层中 magnitude control 很重要？

4. 阅读 Chameleon 第 2.3 节关于 training stability 的内容。描述没有 QK-Norm 时，论文在 34B 上观察到的确切 failure mode。什么是 “norm explosion” signature？

5. 扩展 toy decoder，让它根据 text-only prompt 输出 mixed-modality response。在 training-data distribution 为 60% text-first / 40% image-first 时，测量模型选择 image-first vs text-first 的频率。

## 关键术语

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| Early fusion | “Unified tokens” | 图像从第一步起就被转换为共享 transformer vocabulary 中的离散 token |
| VQ-VAE | “Image tokenizer” | CNN + ViT + codebook，把图像映射为 transformer 可预测的整数 indices |
| Shared vocabulary | “One dictionary” | 单一 token ID space，覆盖 text + image + modality separators |
| QK-Norm | “Attention stabilizer” | Dot product 前对 query 和 key 应用 LayerNorm，防止 norm blowup |
| Mixed-modality generation | “Text + image output” | 一次 inference 中自主生成交错 text 和 image tokens |
| Codebook size | “K entries” | VQ-VAE 可量化到的离散向量数量；在 compression 与 fidelity 之间取舍 |
| Tokenizer ceiling | “Reconstruction limit” | 解码 VQ tokens 可达到的最佳 PSNR；限制模型 image quality |

## 延伸阅读

- [Chameleon Team — Chameleon: Mixed-Modal Early-Fusion Foundation Models (arXiv:2405.09818)](https://arxiv.org/abs/2405.09818)
- [Aghajanyan et al. — CM3 (arXiv:2201.07520)](https://arxiv.org/abs/2201.07520)
- [Yu et al. — CM3Leon (arXiv:2309.02591)](https://arxiv.org/abs/2309.02591)
- [Zhan et al. — AnyGPT (arXiv:2402.12226)](https://arxiv.org/abs/2402.12226)
- [Adept — Fuyu-8B blog (adept.ai)](https://www.adept.ai/blog/fuyu-8b)
