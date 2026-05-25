# LLM 的 Shadow Traffic、Canary Rollout 和 Progressive Deployment

> LLM rollouts 结合了软件部署中最难的部分：没有 unit tests、failure modes 分散、signals 延迟。顺序是：（1）shadow mode：把 prod requests 复制给 candidate model，记录并比较，对用户零影响；能捕获明显 distribution issues，但不是 quality guarantee；（2）canary rollout：progressive traffic shift 10% → 25% → 50% → 75% → 100%，每步设 gates；跟踪 latency percentiles、cost/request、error/refusal rate、output length distribution、user-feedback rate；（3）稳定确认后，对明显不同的 alternatives 做 A/B testing。Non-determinism 不可约化：由于 GPU FP non-associativity 加上 batch-size variance，相同 inputs 跨 runs 会有最高 15% accuracy variation。Cost 是变量，不是常量：一个好 20% 的模型可能每 call 贵 3x。Rollback speed 是决定性的：如果 rollback 需要 redeploy，你太慢了。Policy 放在 config/flags 中；model 放在 registry 中并 pinned digests；rollback = flip policy + revert threshold + pin old model，几秒内完成。

**类型：** 学习
**语言：** Python（stdlib，玩具版 canary-progression simulator）
**前置要求：** 阶段 17 · 13（Observability），阶段 17 · 21（A/B Testing）
**时间：** ~60 分钟

## 学习目标

- 区分 shadow mode（zero-impact compare）、canary（live traffic progressive）和 A/B（stability-confirmed comparison）。
- 枚举五个 LLM-specific canary metrics（latency、cost/request、error/refusal、output-length distribution、user feedback）。
- 解释为什么 LLM non-determinism（最高 15%）会改变 rollout 中 “stable” 的含义。
- 设计一个用 seconds（policy flip）而不是 hours（redeploy）完成的 rollback path。

## 问题

你交付了一个新模型。Offline evals 显示 accuracy gain 3%。你在 production 直接打开它。24 小时内，cost 上升 40%，user thumbs-down 上升 8%，三个 customer tickets 报告“weird answers”。你 rollback。Redeploy 花 3 小时。你的周末被毁了。

这一切都可以避免。Shadow mode 会在任何用户看到前捕获 40% cost spike。Canary 会在 thumbs-down 变动时停在 10%。Policy-flag rollback 只需 30 秒。Discipline 填补了“offline evals 看起来不错”和“真实用户满意”之间的鸿沟。

## 概念

### Shadow mode

Candidate 接收与 production 相同的 requests；outputs 被记录，不返回给用户。对用户零影响。记录：

- Output content（与 production diff）。
- Token counts（cost delta）。
- Latency。
- Refusal 和 error。

能捕获：cost blow-ups、length regressions、明显 refusal changes、hard errors。不能捕获：用户感知的 quality delta。Shadow 是 smoke test，不是 quality test。

### Canary rollout

带 gates 的 progressive traffic shift。典型 progression：1% → 10% → 25% → 50% → 75% → 100%。每步用 5 个 metrics gate：

1. **Latency percentiles** — P50、P95、P99。Breach：canary 的 P99 > 1.5x baseline。
2. **Cost per request** — blended $。Breach：>20% above baseline。
3. **Error / refusal rate** — 5xx 加 explicit refusals。Breach：2x baseline。
4. **Output length distribution** — mean + P99。Breach：distributional shift。
5. **User-feedback rate** — thumbs-down / ticket filings。Breach：1.5x baseline。

### Non-determinism 是新的 variance

相同 inputs 产生不相同 outputs。原因：

- GPU FP non-associativity（floating-point reduction order 随 batch 变化）。
- Batch-size variance（同一 prompt 在 batch of 128 vs batch of 16 中）。
- Sampling（temperature > 0）。

实测：相同 eval sets 跨 runs 最高 15% accuracy variation。Rollout 中的 “stable” 意味着 metrics 位于 expected variance 内，而不是与 baseline 完全相同。把 gates 设在 noise floor 之上。

### Cost 是变量

一个好 20% 的模型可能每 call 贵 3x。Cost/request 是五个 gates 之一。交付一个“更好”但破坏 unit economics 的模型，就是 rollback case。

### Rollback 是武器

- Policy flag（feature flag system）：在 config 中 flip percentage；几秒完成。
- Model pinning（registry digest）：pinned model 不会 auto-upgrade。
- Rollback = revert flag + set pinned digest to previous。几秒，不是几小时。

如果你的 stack 需要 redeploy 才能 rollback，在 rollout 前先修它。

### Tooling

**Argo Rollouts** / **Flagger** — Kubernetes progressive delivery controllers。与 Istio/Linkerd weighted routing 集成。

**Istio weighted routing** — service-mesh-level traffic split。

**KServe / Seldon Core** — 带 built-in canary 的 model serving。

**Feature flags** — LaunchDarkly、Flagsmith、Unleash。Policy-level flip，无需 redeploy。

### Metrics cadence

Canary gates 每 5-15 分钟检查一次，取决于 traffic volume。1% traffic、10 req/min 时，每个 window 有 50-150 data points，对 latency 足够，但对 user feedback noisy。10% 会带来约 10x 更多。每步 progression 应暂停足够长时间，以积累足够样本。

### A/B step 是可选的

如果新模型明显不同（behavior、cost curve、tone 不同），canary 通过后在 50% 做 A/B test。如果只是改进版，canary gates 通过后直接到 100%。

### 你应该记住的数字

- Canary progression：1% → 10% → 25% → 50% → 75% → 100%。
- Non-determinism ceiling：相同 inputs 上 run-to-run variance 最高 15%。
- 五个 canary metrics：latency、cost、error/refusal、output length、user feedback。
- Cost gate：>20% above baseline 是 breach。
- Rollback：几秒，不是几小时。

## 使用它

`code/main.py` 模拟带 injected regressions 的 canary rollout。报告 rollout 停在哪个 stage，以及触发哪个 gate。

## 交付它

本课会产出 `outputs/skill-rollout-runbook.md`。给定 candidate model、baseline 和 risk tolerance，它会设计 shadow→canary→100% plan。

## 练习

1. 运行 `code/main.py`。注入 25% cost regression。Canary 会停在哪个 stage？
2. 你的新模型 offline 有 3% accuracy gain，但 cost/request 是 +18%。能 ship 吗？取决于 policy，写出两条路径。
3. 设计一个 60 秒以内 end-to-end 完成的 rollback。列出所需 infrastructure。
4. Non-determinism 在你的 eval 上显示 ±7%。设置 canary gates，避免 false alarm。你用什么 multipliers？
5. Shadow mode 在 canary 前捕获 40% cost spike。写出触发的 alert rule。

## 关键词汇

| 术语 | 大家常说 | 实际含义 |
|------|----------|----------|
| Shadow mode | “duplicate to new” | Zero-impact send-to-candidate for logging |
| Canary | “progressive traffic” | 带 gates 的逐步用户可见 rollout |
| Gates | “rollout checks” | 阻止 progression 的 metric thresholds |
| Non-determinism | “LLM variance” | 不可约化的 run-to-run differences |
| Policy flag | “flag flip rollback” | Config-level rollback，几秒而不是几小时 |
| Model pin | “registry digest” | 指向 model version 的 immutable reference |
| Argo Rollouts | “K8s progressive” | Kubernetes-native canary/rollback controller |
| KServe | “inference K8s” | 带 canary primitives 的 model serving |
| Istio weighted | “mesh split” | Service-mesh traffic splitter |

## 延伸阅读

- [TianPan — Releasing AI Features Without Breaking Production](https://tianpan.co/blog/2026-04-09-llm-gradual-rollout-shadow-canary-ab-testing)
- [MarkTechPost — Safely Deploying ML Models](https://www.marktechpost.com/2026/03/21/safely-deploying-ml-models-to-production-four-controlled-strategies-a-b-canary-interleaved-shadow-testing/)
- [APXML — Advanced LLM Deployment Patterns](https://apxml.com/courses/mlops-for-large-models-llmops/chapter-4-llm-deployment-serving-optimization/advanced-llm-deployment-patterns)
- [Argo Rollouts docs](https://argo-rollouts.readthedocs.io/)
- [Flagger docs](https://docs.flagger.app/)
