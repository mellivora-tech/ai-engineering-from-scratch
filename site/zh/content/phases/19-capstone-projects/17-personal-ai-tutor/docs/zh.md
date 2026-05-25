# Capstone 17 — 个人 AI Tutor（自适应、多模态、带记忆）

> Khanmigo（Khan Academy）、Duolingo Max、Google LearnLM / Gemini for Education、Quizlet Q-Chat 和 Synthesis Tutor 都在 2026 年大规模交付了 adaptive multimodal tutoring。共同形态是 Socratic policy（永远不只是直接甩答案）、每次 interaction 后更新的 learner model（Bayesian knowledge tracing 风格）、voice + text + photo-math input、curriculum graph retrieval、spaced-repetition scheduling，以及面向 age-appropriate content 的硬 safety filters。这个 capstone 是交付一个 subject-specific tutor（K-12 algebra 或 intro Python），用 10 名学习者运行两周 efficacy study，并通过 content-safety audit。

**类型：** Capstone
**语言：** Python（backend、learner model）、TypeScript（web app）、SQL（通过 Postgres + Neo4j 管理 curriculum graph）
**前置要求：** 阶段 5（NLP）、阶段 6（speech）、阶段 11（LLM engineering）、阶段 12（multimodal）、阶段 14（agents）、阶段 17（infrastructure）、阶段 18（safety）
**覆盖阶段：** P5 · P6 · P11 · P12 · P14 · P17 · P18
**时间：** 30 小时

## 问题

Adaptive tutoring 曾经只是 ed-tech research niche。到 2026 年，它已经是消费级产品。Khanmigo 部署在美国大多数 school districts。Duolingo Max 达到数千万 MAUs。Google 的 LearnLM / Gemini for Education 支撑 Google Classroom 中的 tutoring。Quizlet Q-Chat 与 flashcards 并列。Synthesis Tutor 因 tutor-for-curious-kids 出圈。共同元素是：multimodal input（打字、说话、拍方程）、Socratic pedagogy（先提问，后解释）、每次 interaction 后更新的 learner model，以及严格 age-appropriate safety。

你将为一个特定 cohort 构建其中一个。衡量门槛是真实 efficacy study：10 名学习者，两周内进行 pre-test 和 post-test。voice loop 必须自然（capstone 03 sub-stack）。memory 必须尊重隐私。safety filter 必须通过面向 K-12 的 COPPA-aware red-team。

## 概念

四个组件。**Tutor policy** 是 Socratic loop：当学习者索要答案时，policy 会问一个 leading question；当他们答对时，它移动到下一个 concept；当他们卡住时，它给出 scaffolded hint。**Learner model** 是 Bayesian knowledge tracing（或一个简单变体），每次 interaction 后更新每个 curriculum node 的 mastery probability。**Curriculum graph** 是一个带 prerequisite edges 的 Neo4j concepts 图；policy 遍历 graph 选择下一个 concept。**Memory** 是一个 episodic + semantic store（agentmemory-style），保存 past interactions、mistakes 和 preferences。

UX 是 multimodal 的。Text input 用于 typed answers。Voice input 通过 LiveKit + Whisper（复用 capstone 03）。Photo input 通过 dots.ocr 或 PaliGemma 2 处理数学题。Voice output 使用 Cartesia Sonic-2。Safety 使用 Llama Guard 4 加 age-appropriate filter（阻止 adult content、violence、self-harm），并带 COPPA-aware memory retention policy。

efficacy study 是交付物。10 名学习者，pre-test 和 post-test，两周。报告 learning gain delta 和 confidence interval。与 non-adaptive baseline（相同内容线性呈现，不使用 tutor policy）对比。

## 架构

```
learner device
  |
  +-- text         -> web app
  +-- voice        -> LiveKit Agents (ASR + TTS)
  +-- photo math   -> dots.ocr / PaliGemma 2
       |
       v
  tutor policy (LangGraph)
       - Socratic decision head
       - next-concept chooser (curriculum graph walk)
       - hint scaffolder
       - mastery update
       |
       v
  learner model (BKT / item-response theory)
       - per-concept mastery probability
       - spaced-repetition scheduler (SM-2 or FSRS)
       |
       v
  memory (agentmemory-style)
       - episodic: every interaction
       - semantic: learned mistakes, preferences
       - retention policy: COPPA / GDPR aware
       |
       v
  curriculum graph (Neo4j)
       - prerequisite edges
       - OER content attached
       |
       v
  safety:
    Llama Guard 4 + age-appropriate filter
    memory access guarded by learner ID scope
```

## 技术栈

- Subject choice：K-12 algebra 或 intro Python（选择一个深入）
- Tutor policy：基于 Claude Sonnet 4.7（带 prompt caching）的 LangGraph
- Learner model：Bayesian knowledge tracing（经典）或用于 spacing 的 FSRS
- Curriculum graph：concepts + prerequisite edges + OER content 的 Neo4j
- Memory：agentmemory-style persistent vector + episodic + semantic store
- Voice：LiveKit Agents 1.0 + Cartesia Sonic-2（复用 capstone 03 sub-stack）
- Photo math：dots.ocr 或 PaliGemma 2 做 equation recognition
- Safety：Llama Guard 4 + custom age-appropriate filter
- Eval：Bloom-level question generation、pre/post test harness、efficacy study tooling

## 构建它

1. **Curriculum graph。** 构建一个包含 50-150 个 concept nodes 的 Neo4j（例如 K-12 algebra，从 “number line” 到 “quadratic formula”），带 prerequisite edges。每个 node 附 OER content（Open Textbook、OpenStax）。

2. **Learner model。** 用 priors 初始化 Bayesian knowledge tracing：guess、slip、learn-rate。每次 interaction 后按 concept 更新 mastery。按 learner 持久化。

3. **Tutor policy。** LangGraph nodes：`read_signal`（学习者回答是 correct / partial / stuck？）、`select_concept`（遍历 curriculum graph，选择 highest-priority concept）、`scaffold`（Socratic prompt）、`update_mastery`。

4. **Memory。** 每次 interaction 写入 episodic store。Mistakes 和 preferences 提升为 semantic memory。COPPA-aware retention policy：1 年后自动删除，parent-accessible。

5. **Voice path。** 把 LiveKit Agents worker 连接到 tutor policy。ASR 使用 Whisper-v3-turbo。TTS 使用 Cartesia Sonic-2。支持 barge-in（复用 capstone 03 mechanics）。

6. **Photo-math path。** 上传或拍摄图像；运行 dots.ocr 或 PaliGemma 2 识别方程；把它作为 structured input 传给 tutor。

7. **Safety。** 每个 model output 都通过 Llama Guard 4 + age-appropriate filter（阻止 self-harm、adult content、violence）。Memory access 按 learner ID scoped；提供 parental access surface 用于删除。

8. **Efficacy study。** 10 名学习者，pre-test（standardized 30-question baseline），两周 tutor interaction（每周 3 sessions），post-test。与 10 名学习者的 non-adaptive baseline cohort 对比，内容相同。

9. **Weekly progress reports。** 每个 learner 自动生成 PDF summary：探索过的 topics、mastery trajectories、recommended next steps。

## 使用它

```
learner: "I don't understand why 3x + 6 = 12 means x = 2"
[signal]   stuck
[concept]  'isolating variables' (prerequisite: addition-subtraction-equality)
[scaffold] "what number would you subtract from both sides to start?"
learner: "6"
[signal]   correct
[mastery]  addition-subtraction-equality: 0.62 -> 0.77
[concept]  continue 'isolating variables'
[scaffold] "great. now what is 3x / 3 equal to?"
```

## 交付它

`outputs/skill-ai-tutor.md` 是交付物。一个 subject-specific adaptive tutor，带 multimodal input、learner model、memory、safety 和 measured efficacy。

| 权重 | 标准 | 衡量方式 |
|:-:|---|---|
| 25 | Learning gain delta | 10-learner 两周研究中的 pre/post-test delta |
| 20 | Socratic fidelity | transcript samples 上的 rubric score |
| 20 | Multimodal UX | Voice + photo + text end-to-end coherence |
| 20 | Safety + privacy posture | Llama Guard 4 pass rate + COPPA-aware retention |
| 15 | Curriculum breadth and graph quality | Concept coverage + prerequisite graph consistency |
| **100** | | |

## 练习

1. 在有无 adaptive learner model（random concept order）两种情况下运行 efficacy study。报告 delta。预期 adaptive 会赢，但幅度才是有趣的数字。

2. 添加 multimodal probe：同一个 concept question 分别以 text、voice 和 photo 形式呈现。衡量学习者是否在自己偏好的 modality 下收敛更快。

3. 构建 parent dashboard：练习过的 topics、mastery trajectories、upcoming concepts、safety events（任何 guardrail hits）。保持 COPPA-aligned。

4. 添加 language-switch mode：tutor 接受 Spanish input，并用 Spanish 教学。衡量 X-Guard coverage。

5. 压测 memory privacy：验证 learner A 无法看到 learner B 的数据，即使通过 voice-clip re-ingest attack。记录尝试访问并告警。

## 关键词汇

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|------------------------|
| Socratic policy | “Ask, do not dump” | Tutor 提 leading question，而不是直接给答案 |
| Bayesian knowledge tracing | “BKT” | 用于每个 concept mastery probability 的经典 learner-model equations |
| FSRS | “Free Spaced Repetition Scheduler” | 2024 spaced-repetition scheduler，比 SM-2 更好 |
| Curriculum graph | “Concept DAG” | 带 prerequisite edges 的 Neo4j concepts 图 |
| Episodic memory | “Per-interaction log” | 每次 interaction 都存储，供后续 retrieval |
| Semantic memory | “Learned pattern store” | 从 episodic 提升出的压缩 mistakes 和 preferences |
| COPPA | “Kids privacy law” | 美国限制收集 13 岁以下儿童数据的法律 |

## 延伸阅读

- [Khanmigo (Khan Academy)](https://www.khanmigo.ai) — reference consumer K-12 tutor
- [Duolingo Max](https://blog.duolingo.com/duolingo-max/) — reference language-learning tutor
- [Google LearnLM / Gemini for Education](https://blog.google/technology/google-deepmind/learnlm) — hosted reference model
- [Quizlet Q-Chat](https://quizlet.com) — alternate reference
- [Synthesis Tutor](https://www.synthesis.com) — startup reference
- [FSRS algorithm](https://github.com/open-spaced-repetition/fsrs4anki) — spaced-repetition scheduler
- [Bayesian Knowledge Tracing](https://en.wikipedia.org/wiki/Bayesian_knowledge_tracing) — learner-model classic
- [LiveKit Agents](https://github.com/livekit/agents) — voice stack
