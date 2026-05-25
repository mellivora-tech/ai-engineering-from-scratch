# Model、System 与 Dataset Cards

> 三种 documentation format 构成 AI transparency。Model Cards（Mitchell et al. 2019）：模型的 nutrition labels，包括 training data、quantitative disaggregated analyses、ethical considerations、caveats；只有 0.3% 的 Hugging Face model card 记录 ethical considerations（Oreamuno et al. 2023）。Datasheets for Datasets（Gebru et al. 2018，CACM）：motivation、composition、collection process、labeling、distribution、maintenance；类比 electronics datasheet。Data Cards（Pushkarna et al., Google 2022）：模块化分层细节（telescopic、periscopic、microscopic），作为面向多样读者的 boundary object。2024-2025 进展：通过 LLM 自动生成（CardGen, Liu et al. 2024）；model-card detail 与 HF 上最高 29% 的 download increase 相关（Liang et al. 2024）；verifiable attestations（Laminator, Duddu et al. 2024）；面向 carbon/water 的 sustainability reporting additions（Jouneaux et al. 2025 年 7 月）；EU/ISO regulatory cards 正在出现。System Cards（Sidhpurwala 2024；Meta system-level transparency；“Blueprints of Trust” arXiv:2509.20394）：覆盖 security capabilities、prompt-injection protection、data-exfiltration detection、alignment with human values 的 end-to-end AI system documentation。

**类型：** 构建
**语言：** Python（stdlib，model-card + datasheet + system-card generator）
**前置要求：** 阶段 18 · 18（safety frameworks），阶段 18 · 24（regulatory）
**时间：** ~60 分钟

## 学习目标

- 描述原始 Mitchell et al. 2019 model card 和 Gebru et al. 2018 datasheet。
- 描述 Data Cards 的 telescopic/periscopic/microscopic layering。
- 描述 System Cards 及其 end-to-end coverage。
- 说出三项 2024-2025 进展（automated generation、verifiable attestations、sustainability reporting）。

## 问题

Regulatory framework（第 24 课）和 lab safety policy（第 18 课）都要求 documentation。Documentation format 从 model-specific（model card）演化到 dataset-specific（datasheet），再到 system-specific（system card）。每种格式处理不同 transparency scope。2024-2025 的 automation 与 verifiable-attestation 工作，处理了长期存在的 adoption problem。

## 概念

### Model Cards（Mitchell et al. 2019）

章节：
- Model details。
- Intended use。
- Factors（与 evaluation 相关的人口统计或环境因素）。
- Metrics。
- Evaluation data。
- Training data。
- Quantitative analyses（按 factor disaggregated）。
- Ethical considerations。
- Caveats and recommendations。

Adoption problem：Oreamuno et al. 2023 对 Hugging Face model card 的 audit 发现，只有 0.3% 记录 ethical considerations。

### Datasheets for Datasets（Gebru et al. 2018）

Electronics-datasheet 类比。章节：
- Motivation（为什么创建这个 dataset）。
- Composition（里面有什么）。
- Collection process（如何组装）。
- Labeling（如适用）。
- Uses（intended、prohibited、risks）。
- Distribution。
- Maintenance。

2021 年发表于 CACM。Datasheet 是 upstream documentation；model card 依赖 datasheet 准确。

### Data Cards（Pushkarna et al., Google 2022）

模块化分层细节。三个 zoom level：
- **Telescopic。** 面向非专家的 high-level summary。
- **Periscopic。** 面向 ML practitioner 的 middle-level overview。
- **Microscopic。** 面向 auditor 的详细 feature-level documentation。

Boundary-object framing：不同读者从同一文档提取不同信息。

### System Cards

Scope：包含 model + safety stack + deployment context 的 end-to-end AI system。典型章节包括：
- Security capabilities。
- Prompt-injection protection。
- Data-exfiltration detection。
- Alignment with stated human values。
- Incident response。

Sidhpurwala 2024 和 Meta system-level transparency work。“Blueprints of Trust”（arXiv:2509.20394）把 System Card 形式化为 Model Card 的 deployment-layer complement。

### 2024-2025 进展

- **CardGen（Liu et al. 2024）。** 通过 LLM 自动生成 model card；在标准化 Mitchell 2019 field 上，报告称比许多人写的 card 更客观。
- **Download correlation（Liang et al. 2024）。** 详细 model card 与 HF 上最高 29% 的更高 download rate 相关，adoption pressure 现在不仅来自 compliance，也来自市场。
- **Laminator（Duddu et al. 2024）。** 通过 hardware TEE / cryptographic signature 提供 verifiable attestations，让 model card 携带 claim 的 proof，而不仅是 claim。
- **Sustainability（Jouneaux et al. 2025 年 7 月）。** 增加 carbon、water 和 compute-energy footprint；ISO standard 正在出现。
- **Regulatory cards。** EU AI Act（第 24 课）GPAI Code of Practice Transparency chapter 要求 model card 作为 compliance artifact。

### 它在阶段 18 中的位置

第 24-25 课是 regulatory 与 CVE layer。第 26 课是 documentation layer。第 27 课是 training-data governance，也就是 datasheet 的 upstream。第 28 课是产生 card 中引用 evaluation 的 research ecosystem。

## 使用它

`code/main.py` 为玩具 deployment 生成最小 model card、datasheet 和 system card。每个都遵循 canonical section structure。你可以检查格式，并比较三者的 scope。

## 交付它

本课会生成 `outputs/skill-card-audit.md`。给定 model card、datasheet 或 system card，它会 audit section coverage、numerical disaggregation，以及是否存在 verifiable attestations。

## 练习

1. 运行 `code/main.py`。检查生成的 cards。识别薄弱章节（只有 placeholder），并说明什么 evidence 会增强它们。

2. 在 model card 中扩展跨两个人口群体的 quantitative disaggregated analysis（第 20 课）。

3. 阅读 Oreamuno et al. 2023 关于 0.3% adoption rate 的研究。提出一个 model card specification 的结构性变化，用来提高 ethical-considerations adoption。

4. Laminator（Duddu et al. 2024）使用 TEE 做 verifiable attestations。设计一个 model-card field，携带 evaluation result 的 cryptographic attestation，并描述 verifier 的角色。

5. 为你过去的一个项目或一个假想 deployment 写一份 System Card（System Card，不是 Model Card）。识别对 third-party auditor 价值最高的章节。

## 关键词汇

| 术语 | 人们常说 | 实际含义 |
|------|-----------------|------------------------|
| Model Card | “Mitchell card” | Mitchell et al. 2019 针对 ML model 的标准 documentation |
| Datasheet | “Gebru datasheet” | Gebru et al. 2018 针对 dataset 的标准 documentation |
| Data Card | “Pushkarna card” | Google 2022 模块化分层 data documentation |
| System Card | “deployment card” | 包含 safety stack 的 end-to-end AI system documentation |
| Boundary object | “不同读者，一个文档” | Data Cards framing：同一文档服务多样受众 |
| Verifiable attestation | “Laminator attestation” | 附着在 documentation claim 上的 cryptographic 或 TEE proof |
| Sustainability field | “carbon / water footprint” | 2025 年 emerging addition，用于 environmental accounting |

## 延伸阅读

- [Mitchell et al. — Model Cards for Model Reporting (arXiv:1810.03993, FAT* 2019)](https://arxiv.org/abs/1810.03993) — canonical model card
- [Gebru et al. — Datasheets for Datasets (CACM 2021, arXiv:1803.09010)](https://arxiv.org/abs/1803.09010) — datasheet paper
- [Pushkarna et al. — Data Cards (Google 2022)](https://arxiv.org/abs/2204.01075) — layered data documentation
- [Sidhpurwala et al. — Blueprints of Trust (arXiv:2509.20394)](https://arxiv.org/abs/2509.20394) — System Card formalization
