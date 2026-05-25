# Failure Modes — MAST、Groupthink、Monoculture、Cascading Errors

> 2026 年的参考 taxonomy 是 **MAST**（Cemri et al., NeurIPS 2025, arXiv:2503.13657），来自 7 个 state-of-the-art open-source MAS 的 1642 条 execution traces，显示 **41–86.7% failure rate**。三类 root categories：**Specification Problems**（41.77%）— role ambiguity、unclear task definitions；**Coordination Failures**（36.94%）— communication breakdowns、state desync；**Verification Gaps**（21.30%）— missing validation、absent quality checks。**Groupthink** family（arXiv:2508.05687）补充：monoculture collapse（same base model → correlated failures）、conformity bias（agents 强化彼此 errors）、deficient theory of mind、mixed-motive dynamics、cascading reliability failures。Cascading example：retry storms，payment failure 触发 order retries，order retries 触发 inventory retries，并在数秒内把 inventory service 压到 10x load — 需要 circuit breakers。Memory poisoning：一个 agent 的 hallucination 进入 shared memory，下游 agents 当作 fact；accuracy 逐渐衰减，root-cause diagnosis 很痛苦。**STRATUS**（NeurIPS 2025）报告，通过 specialized detection / diagnosis / validation agents，mitigation-success 提高 1.5x。本课把 failure modes 当作 first-class engineering targets。

**类型：** 学习
**语言：** Python（stdlib）
**前置要求：** 阶段 16 · 13（Shared Memory），阶段 16 · 14（Consensus and BFT），阶段 16 · 15（Voting and Debate Topology）
**时间：** ~75 分钟

## 问题

Multi-agent systems 在真实 tasks 上失败率为 41-86.7%（Cemri et al. 2025 在 7 个 open-source MAS 上测得）。这不能靠“just add more agents”来 debug。失败有结构性原因。MAST taxonomy 给了 categories。本课把每个 category 映射到具体 detection、diagnosis 和 mitigation pattern，让这些数字不再显得任意。

2026 production practice 是把 failure modes 当作 design inputs。你的 architecture 只有在能指向每个 MAST category 并说出部署了什么 mitigation 时，才算 “good enough”。

## 概念

### MAST categories

**Specification Problems（41.77% failures）。** agent 的 task 没有定义足够紧。例子：

- Role ambiguity：两个 agents 都以为自己是 reviewer。
- Task underspecified：“summarize this”，但用户想要特定角度。
- Success criteria implicit：agent 无法判断自己是否成功。

Mitigations：
- 写 explicit role contracts。每个 agent prompt 说明它做什么 *以及不做什么*。
- 每个 task 有 acceptance tests。agent 开始前定义 “done looks like X”。
- Pre-flight spec check：dispatch 前由 separate agent review task definition。

**Coordination Failures（36.94%）。** communication 或 state breakdowns。

例子：
- 两个 agents 在无 synchronization 下更新 shared state。
- agents 之间 message lost（queue failure、timeout）。
- State drift：agent A 认为 task done；agent B 仍在执行。

Mitigations：
- 带 optimistic concurrency 的 versioned shared state。
- 关键 messages 显式 acknowledgment（retry until acked）。
- 定期 state-sync checkpoints；早期检测 drift。

**Verification Gaps（21.30%）。** outputs 没有 independent check。

例子：
- 一个 agent 声称 success；无人 verify。
- 一条 chain 中每个 agent 都信任前一个 output。
- emergent composed behavior 上缺少 test coverage。

Mitigations：
- independent verifier agent（第 13 课）。Read-only，有 independent source access。
- 显式 handoff contract：“A's output must pass checker C before B starts。”
- outcome logging for post-hoc analysis。

### Groupthink family（arXiv:2508.05687）

当 agents homogenize 或 mimic 时出现的五类相关 failures：

**Monoculture collapse。** same base model 或 training data → correlated errors。三个 agents 共享一个 LLM 时，也共享它的 hallucinations。

**Conformity bias。** agents 朝最大声或最自信的 peer 调整，即使 peer 是错的。

**Deficient ToM。** agents 无法建模彼此 beliefs；coordination 坍塌（第 18 课）。

**Mixed-motive dynamics。** incentives 部分对齐的 agents 漂向 compromise-middle，结果谁都不满意。

**Cascading reliability failures。** 一个 component 的 error pattern 触发 dependent components 的 error patterns。

### Cascading example — retry storm

2026 年经典 incident pattern：

```
payment service fails 10% of requests
   ↓
order agent retries payment (exponential backoff but naive)
   ↓
each retry is a new order-inventory check
   ↓
inventory service sees 2x normal load
   ↓
inventory service starts timing out
   ↓
every order retries inventory check
   ↓
inventory service sees 10x normal load
   ↓
cluster goes down
```

修复是 classical：**circuit breakers**。当 downstream error rate 超过 threshold，用 cached 或 default results short-circuit。再加上每个 request 的 capped retry budgets。

Circuit breakers 是少数可以从 distributed systems 直接借来、无需修改的 multi-agent failure mitigations。

### Memory poisoning（再次）

来自第 13 课：一个 agent 的 hallucination 成为 shared-memory fact；downstream agents 在 poisoned fact 上推理。用 MAST 术语，这是 shared-memory layer 的 verification gap。

症状是 gradual accuracy decay。你不会得到 crash；你会得到 slow drift，很难 root-cause。

Mitigation：append-only log、provenance、unwritable verifier。第 13 课已覆盖。

### STRATUS — failure detection 的 specialized agents

STRATUS（NeurIPS 2025）报告，当你部署以下 agents 时，mitigation-success 提高 1.5x：

- **Detection agent。** 监控 symptom patterns（high disagreement、retry spikes、accuracy drift）。
- **Diagnosis agent。** 给定 symptoms，从 MAST taxonomy 推断 likely root cause。
- **Validation agent。** mitigation 应用后，检查 symptoms 是否消失。

这是应用到 agent systems 的 SRE-style incident response。三种 roles 都可以是带 specialized prompts 的 LLM agents。

### Failure-mode audit

2026 best practice 是每年（或每次 major release）做 failure-mode audit：

1. **Trace sample。** 收集约 1000 条真实 execution traces。
2. **Categorize。** 对每条 trace 的 failures，映射到 MAST + Groupthink categories。
3. **Compute failure-by-category rate。** 哪些 categories 主导你的系统？
4. **Rank mitigations。** 哪个 fix 会消除最多 failures？
5. **Pick 2-3 mitigations。** 实现；下季度重新 audit。

纪律比具体选择更重要。没有 audits，failures 会混成噪音，永远无法被系统性处理。

### Silent failures

最危险的 failure category 是 silent correctness failure。loud failures（crash、exception、alert）可以被监控。产生 plausible-but-wrong outputs 的系统无法靠 exception logs 检测。这就是为什么 verification gaps 虽然只占 21.30% by count，但 per-failure 成本最高。

投入：
- Sample-based human review。
- Golden-dataset regression tests。
- 重要 outputs 上的 cross-agent cross-checking。

### Failure vs slow failure

有些 failures 立即发生；有些是 slow。Immediate failures（timeout、schema mismatch、auth error）容易检测。Slow failures（memory poisoning、monoculture drift、role ambiguity）检测和预防都更贵。

2026 engineering move：instrument slow-failure proxies，在 drift 变成 visible error 前捕捉它。agreement rate、retry rate、output-length distribution，以及 consecutive agent versions 之间的 edit-distance 都是有用 proxies。

## 构建它

`code/main.py` 实现：

- `FailureTaxonomy` — 把 simulated incidents 分类到 MAST + Groupthink categories。
- `CircuitBreaker` — classic pattern；error rate 超过 threshold 时 open。
- `RetryStormSimulator` — 展示 cascading failure；可开关 circuit breaker。
- `DetectionAgent` — scripted STRATUS-style symptom matcher。

运行：

```
python3 code/main.py
```

预期输出：
- 无 circuit breaker 的 retry storm：inventory errors 爆炸（simulated）。
- 有 circuit breaker：cap 在 threshold；提供 degraded-mode responses。
- detection agent 标记 pattern 并命名 MAST category。

## 使用它

`outputs/skill-mast-auditor.md` 对 multi-agent system 运行 MAST-style failure-mode audit。Traces → categorization → mitigation ranking。

## 发布它

Production 中的 failure-mode discipline：

- **MAST audit per quarter。** 不是 annually。categories 会随系统成长而变化。
- **Circuit breakers everywhere。** 每个 outbound call to dependent service。默认 open threshold 为 5-10% error rate。
- **Golden datasets。** 小而高质量，hand-audited。每周 regression-test。
- **STRATUS trio。** Detection + Diagnosis + Validation agents 监控 production。先从 detection agent 开始；symptoms 嘈杂后再加 diagnosis。
- **Failure budget。** 为 failure rate by category 设置 explicit SLO。超过 budget 触发 stop-shipping conversation。

## 练习

1. 运行 `code/main.py`。确认 circuit breaker cap retry storm。改变 failure threshold 并观察 tradeoff。
2. 实现 **slow-failure proxy**：3 个 parallel agents 的 agreement rate。当它急剧下降时触发 alert。通过逐步关联 agent outputs 模拟 monoculture drift。
3. 阅读 Cemri et al.（arXiv:2503.13657）。选他们的 7 个 MAS systems 之一，映射其 top 3 failure categories。这些与 MAST 预测相比如何？
4. 阅读 Groupthink paper（arXiv:2508.05687）。识别五个 patterns 中哪个最难在 production 检测。提出一个 proxy metric。
5. 为你熟悉的一个 multi-agent system 设计 STRATUS-style detection-diagnosis-validation trio。detection 观察哪些 symptoms？diagnosis 推荐哪些 mitigations？validation 如何确认它们有效？

## 关键词汇

| 术语 | 人们常说 | 实际含义 |
|------|----------------|------------------------|
| MAST | “2026 taxonomy” | Cemri 2025；failures 的 3 个 root categories + 14 sub-types。 |
| Specification Problem | “Role ambiguity” | task 或 role under-defined；agents 不知道要做什么。 |
| Coordination Failure | “State drift” | agents 之间 communication 或 sync breakdown。 |
| Verification Gap | “No one checked” | outputs 未经 independent validation 就被接受。 |
| Groupthink family | “Homogeneity failures” | monoculture、conformity、deficient ToM、mixed-motive、cascading。 |
| Monoculture collapse | “Same model, same hallucinations” | shared base model 或 training data 导致 correlated errors。 |
| Retry storm | “Cascading error amplification” | 一个 failure 触发 retries，retries 放大 downstream load。 |
| Circuit breaker | “Fail fast on error rate” | error rate 超 threshold 时 open；用 default short-circuit。 |
| STRATUS | “Incident response trio” | Detection + diagnosis + validation agents。mitigation success 1.5x。 |
| Memory poisoning | “Hallucinations propagate” | shared-memory fact 被污染；downstream agents 在 poison 上推理。 |

## 延伸阅读

- [Cemri et al. — Why Do Multi-Agent LLM Systems Fail?](https://arxiv.org/abs/2503.13657) — MAST taxonomy，NeurIPS 2025
- [Groupthink failures in multi-agent LLMs](https://arxiv.org/abs/2508.05687) — monoculture、conformity 和 five-family taxonomy
- [STRATUS — specialized agents for MAS incident response](https://neurips.cc/) — NeurIPS 2025 proceedings entry（detection + diagnosis + validation）
- [Release It! — stability patterns (Nygard)](https://pragprog.com/titles/mnee2/release-it-second-edition/) — canonical circuit-breaker reference
- [Anthropic — Multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system) — production failure-mode notes
