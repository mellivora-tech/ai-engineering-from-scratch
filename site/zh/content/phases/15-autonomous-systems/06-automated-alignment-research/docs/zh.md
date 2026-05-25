# Automated Alignment Research（Anthropic AAR）

> Anthropic 让多组 Claude Opus 4.6 Autonomous Alignment Researchers 在独立 sandboxes 中并行运行，并通过一个共享 forum 协作；该 forum 的日志存放在任何 sandbox 之外（因此 agents 不能删除自己的记录）。在 weak-to-strong training 问题上，AARs 超过了人类研究者。Anthropic 自己的总结指出，规定好的 workflows 往往会限制 AAR flexibility 并降低 performance。自动化 alignment research 是一个压缩步骤，它会把时间线压缩到 RSP 试图检测的那些 misalignment risks 上。

**类型：** 学习
**语言：** Python（stdlib，parallel-research-forum simulator）
**前置要求：** 阶段 15 · 05（AI Scientist v2），阶段 15 · 04（DGM）
**时间：** ~60 分钟

## 问题

Alignment research 需要大量人类研究者时间。像 scalable oversight、reward specification、weak-to-strong training 这样的问题，每次迭代实验都可能需要数周。随着 frontier capabilities 前进，alignment workload 的增长速度超过合格研究者供给。

Automated Alignment Research（AAR）问的是：这些能力正在超过 alignment 的前沿模型，是否也能帮助缩小这个差距？Anthropic 2026 年关于 AAR-run weak-to-strong-training study 的报告（alignment.anthropic.com/2026/automated-w2s-researcher/），是这类已部署系统的首批公开结果之一。

结果确实正面：AARs 比人类 baseline 更好地解决了一个研究问题。这个结果也把本阶段发展出的每一种担忧集中在一个系统里。如果 alignment research 可以被自动化，那么破坏 safeguards 的那些部分也可以被自动化。RSP 中针对 AI R&D capability 的阈值，正是围绕这个 loop 写的。

## 概念

### 架构（按公开描述）

- **Parallel agents。** 多个 Claude Opus 4.6 instances 作为 AARs 运行，每个都在独立 sandbox 中。Sandboxes 限制 filesystem、network 和 process surface。
- **Shared forum。** Findings 发布到一个共享 forum，其 storage backend 位于每个 sandbox 之外。Agents 可以读取 forum；agents 不能从自己的 sandbox 内删除或编辑 forum records。
- **Task allocation。** 人类设计的 task queue 将研究问题分配给 AARs。AARs 提出 decompositions，运行 experiments，并写出 findings。
- **Weak-to-strong training focus。** 已发表研究处理的是 weak-to-strong training——使用较弱 supervisor model 训练更强 student model。AARs 在这个任务上超过了人类研究者。

### 为什么 out-of-sandbox log 很重要

log-integrity 属性是一项受控实验与一个可能已被攻破的实验之间的区别。如果 AARs 可以编辑自己的 logs，它们就能掩盖失败或可疑实验。Anthropic 直接指出了这一点：这个属性是信任研究输出的必要条件，而且验证起来并不简单。

在基础设施术语中，对应的是 append-only logs，并 write-through 到 external storage（例如 S3 object-lock、immutable Datadog logs）。agent 没有凭据修改历史记录。这与 2026 年企业安全团队用于构建 ransomware-resistant audit trails 的模式相同。

### prescribed-workflow 的权衡

Anthropic 的报告指出，规定好的 workflows 往往会限制 AAR flexibility 并降低 performance。换句话说：如果人类设计者一步一步写好 research plan，AARs 的表现会差于只给 high-level goals 并允许自由分解。这与 agentic research 中的结果一致（LangGraph、Microsoft's Agent Framework）——过度规定的 scaffolds 会伤害有能力的 agents。

安全含义很微妙。提升 AAR performance 的 flexibility，也正是允许 unsanctioned decomposition、novel tool use 和 goal-drift 的 flexibility。这个权衡是 design-space decision：要给多少 flexibility，才能买到足够能力以超过人类，同时又不让 agent 有空间把 goals 泛化到 off-distribution？

### 压缩风险

RSP v3.0（第 19 课）引入了 AI R&D capability threshold：以相对 human + AI tools 有竞争力的成本完全自动化 AI R&D pipeline 的能力。DeepMind 的 FSF v3 包含类似的 ML R&D autonomy level。两个框架都把这个阈值视为触发更高 controls 的节点。

AAR 距离这个阈值还差一步：它自动化了 pipeline 的一部分（针对具体、良好 scoped tasks 的 alignment research），但还没有自动化 end-to-end capability development loop。时间线问题是这个差距会多快闭合。

Compressed timelines 是 compounding-failure 的担忧。如果 alignment research 与 capability research 以相近速度复合，那么 misalignment risk surface 至少会和 capability 一样快增长。如果 capability 复合更快（历史趋势），差距就会变宽。这也是为什么 AAR 是有条件的好事：每个额外 alignment result 都会缩小差距，当且仅当 research process 是可信的。

### AAR 不能取代什么

人类研究者设置 task queue、review results，并持有 constitutional authority。AARs 加速 pipeline 中间部分，而不是两端。Anthropic 公开输出同时包含 AAR contributions 和人类研究者对 what to publish、what to retract、what to refine 的判断。

这匹配第 15 课的 propose-then-commit 模式，只是应用到了 research 本身：AARs propose；humans commit。

## 使用它

`code/main.py` 会模拟一个 multi-AAR research forum。三个独立的“research agents”并行运行，将 findings 发布到一个 append-only forum，并带有可选 tamper attempts。

观察：

- out-of-sandbox logging 如何捕捉 log-tampering attempts。
- 带自由分解的 task allocation 如何在更难问题上超过 fixed workflows。
- 如果 log 不是 tamper-evident，一个被攻破的 AAR 如何污染 forum 的 signal。

## 交付它

`outputs/skill-aar-deployment-review.md` 是 automated-alignment-research pipeline 的 pre-deployment review。

## 练习

1. 运行 `code/main.py`。比较“fixed-workflow”和“free-decomposition”设置。free decomposition 总是赢吗，还是有些 problem classes 中 fixed workflow 也有竞争力？

2. 修改 simulator，让一个 agent 尝试 log tampering。确认 append-only log 能检测到它。写一段描述 log 中 detection signature 的具体样子。

3. 阅读 Anthropic 的 weak-to-strong AAR report。找出 AARs 超过人类研究者的具体 sub-task。是什么让它适合自动化？

4. 设计一个 task-queue allocation policy，平衡 AAR flexibility（更好结果）与 prescribed-workflow constraints（更易审计）。描述你会如何 A/B test 两者。

5. 阅读 RSP v3.0 的 AI R&D-4 threshold。用一段话描述你认为哪种能力会跨过它，而 AAR 当前尚未做到。

## 关键术语

| Term | What people say | What it actually means |
|---|---|---|
| AAR | “Automated Alignment Researcher” | Claude Opus 4.6 instance，在 alignment problems 上 autonomous operation |
| Weak-to-strong training | “用较弱 supervisor 训练更强模型” | AARs 超过人类的经典 scalable-oversight benchmark |
| Shared forum | “agents 发布 findings 的地方” | Append-only、out-of-sandbox storage |
| Out-of-sandbox log | “agent 不能编辑自己的记录” | Tamper-evident write-through 到 external storage |
| Prescribed workflow | “人类设计者给出的 step-by-step plan” | 约束 AAR；相对 free decomposition 往往降低 performance |
| Free decomposition | “agent 决定如何拆解任务” | 更有能力，更难审计 |
| AI R&D threshold | “RSP/FSF capability level” | 以有竞争力成本完全自动化 R&D pipeline |
| Compressed timeline | “Alignment vs capability race” | 如果 capability 复合快于 alignment，misalignment risk 会增长 |

## 延伸阅读

- [Anthropic — Automated Weak-to-Strong Researcher](https://alignment.anthropic.com/2026/automated-w2s-researcher/) — primary source。
- [Anthropic Responsible Scaling Policy v3.0](https://anthropic.com/responsible-scaling-policy/rsp-v3-0) — AI R&D threshold framing。
- [Anthropic — Measuring AI agent autonomy](https://www.anthropic.com/research/measuring-agent-autonomy) — 更宽的 agent-autonomy framing。
- [DeepMind Frontier Safety Framework v3](https://deepmind.google/blog/strengthening-our-frontier-safety-framework/) — 与 RSP 平行的 ML R&D autonomy levels。
- [Burns et al. (2023). Weak-to-Strong Generalization (OpenAI)](https://openai.com/index/weak-to-strong-generalization/) — AARs 攻克的底层问题。
