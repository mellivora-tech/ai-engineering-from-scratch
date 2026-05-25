# Kill Switches、Circuit Breakers 与 Canary Tokens

> kill switch 是一个位于 agent edit surface 之外的 boolean——Redis key、feature flag、signed config——用于完全禁用 agent。circuit breaker 更细粒度：它在特定 pattern（连续五次相同 tool calls）上 trip，暂停 offending path，并升级给人类。canary token 继承自经典 deception：一个 fake credential 或 honeypot record，agent 没有合法理由触碰它，一旦访问就触发 alert。基于 eBPF 的 datapaths（例如 Cilium）可以在 kernel layer 将 quarantined pod 的 egress 重写到 forensic honeypot；已发布的 Cilium benchmarks 报告在负载下有 sub-millisecond P99 datapath latency（你的 propagation budget 取决于 policy update 如何到达 node，而不是 datapath 本身）。会适应 moving baseline 的 statistical detectors（EWMA、CUSUM）会静默接受 drift——把它们与不会弯曲的 hard constitutional limits 叠加。

**类型：** 学习
**语言：** Python（stdlib，three-detector simulator: kill switch, circuit breaker, canary）
**前置要求：** 阶段 15 · 13（Cost governors），阶段 15 · 10（Permission modes）
**时间：** ~60 分钟

## 问题

Cost governors（第 13 课）约束 agent 能花多少钱。它们不约束 agent 在预算内能做什么。一个有 $50 velocity limit 的 agent 仍然可以 exfiltrate secret、发布错误帖子、删除资源——在 token 上，昂贵 action 往往很便宜。

本课覆盖紧挨 cost layer 的三个 detectors：

1. **Kill switch**：位于 agent reach 之外的 boolean off-button。
2. **Circuit breaker**：暂停特定 path 的 action-pattern detector。
3. **Canary token**：agent 没有合法理由触碰的 bait，一旦触碰就暴露自己。

三者都是 pre-LLM engineering。经典 deception、rate-limit breakers 和 feature-flag kills 早于 autonomous agents。新的地方是 attack surface：agents 读取不可信内容（第 11 课），编辑自己的 memory，并且可以把许多看似安全的 actions 组合成不安全行为。这里命名的 detectors 有效，是因为它们不信任 agent 的 self-report。

## 概念

### Kill switches

kill switch 是 agent 可以读取但不能写入的 boolean。实现选项：

- **Managed service 中的 feature flag。** LaunchDarkly、Statsig、Unleash。更新会在数秒内传播。
- **agent 轮询的 Redis key。** 简单；要求 agent process 每轮检查。
- **Object storage 中的 signed config。** agent 在 boot 时验证 signature；拒绝 unsigned states。
- **OS-level signal 或 container-lifecycle kill。** Docker `kill`、Kubernetes `kubectl delete pod`、systemd stop。

正确 kill switch 的属性：

- agent 不能把它设为 `off`。（它存在于 agent credentials 无法写入的系统中。）
- 它在每个 consequential action 上检查，而不只是 startup。
- 当它 off 时，agent 不做任何 externally observable 的事，包括 logging 到 agent 可达的系统。
- 重新启用它是明确的人类行动，不是自动 timeout。

### Circuit breakers

circuit breaker 暂停特定 pattern，而不是整个 agent。经典形态（来自 2007 年 Nygard 的书，仍然适用）：

- **Closed**：action 允许。
- **Open**：action 被阻止。
- **Half-open**：cool-down 后允许 1–3 次 probe attempts（默认 1）；成功则关闭 breaker，任何残余失败都会重新打开。

agent 相关 triggers：

- 连续五次相同 tool calls（repetitive loop）。
- 同一 tool 连续五次 failure，但 inputs 不同（systemic failure）。
- tool calls rate 高于阈值（第 13 课 velocity）。
- 在 out-of-trust content read 后调用特定 tool（例如 `file.delete`）（第 11 课）。

### Canary tokens

Canary tokens（也叫 honeytokens）是 agent 永远不应触碰的 inputs。访问就是 alarm。

agent 示例：

- environment 中的 fake `AWS_SECRET_ACCESS_KEY`，没有实际权限。exfiltration attempts 会立刻 attribution。
- 带 known sentinel value 的 fake database record。任何 read 或 update 都触发 alert。
- workspace 中的 fake file。读取或编辑该文件很可疑。
- session start 注入的 fake memory entry。如果它出现在后续 output 中，memory 已被 exfiltrated。

Canary design 特定于 workflow。agent 有合法理由触碰的 canary 是噪声。

### 为什么要叠加 statistical 和 hard limits

Statistical detectors（EWMA、CUSUM、z-score on tool-call rates）会适应 moving baseline。对于 legitimate drift，这是 feature；对于 adversarial drift，这是 bug。有耐心的攻击者可以把 baseline 缓慢移动到没有 alarm 触发。

Hard constitutional limits（第 17 课）不会弯曲。“Never email `secrets@`” 作为 constitutional rule 更简单、可审计，且不会被 slow drift game。layering pattern：statistical detectors 捕捉大多数噪声，hard limits 捕捉走过 detector 的攻击。

### 通过 eBPF datapath redirect 进行 quarantine

Cilium 的 eBPF-based network policy 可以在 kernel layer 将 quarantined pod 的 egress 重写到 forensic honeypot。Cilium 的公开 benchmarks 报告 TCP_RR 上的 sub-millisecond P99 datapath latency；在你的 cluster 中，*触发* quarantine 的 wall-clock budget 主要由 policy update 如何从 control plane 传播到 node 决定（在 Kubernetes 中预加载 operator 时，几百毫秒是现实的；tail latency 取决于 cluster size 和 reconcile cadence）。这给你两样东西：agent 不会 exfiltrate 任何真实东西，并且 agent 在 honeypot 中的行为被完整观察。同一种模式，不同 layer，就像 canary token：agent 通过被 contained 时试图做的事情暴露 intent。

### 没有 detector 能捕捉什么

- **由 allowed primitives 组合而成的 semantic attacks。** 每个单独 action 都通过每个 detector；组合才是问题。这是第 10 课的 Auto Mode composite failure。
- **把 detector 本身作为 payload 一部分的攻击。** 如果 agent 读取一个页面，上面写着“行动前先 disable canary”，且 agent 有能力这么做，detector 就被 compromise。Canary tokens 应位于 agent 无法修改的系统中。

## 使用它

`code/main.py` 会模拟一个短 agent trajectory 通过三个 detectors。一个位于 external dict 中的 kill switch；一个在连续五次相同 tool calls 上 trip 的 circuit breaker；一个读取时触发 alert 的 canary file。输入 synthetic trajectory：legitimate actions、repetitive loop、canary probe，以及一个 kill-switch-triggered scenario，其中 agent actions 被 halted。

## 交付它

`outputs/skill-tripwire-design.md` 会审查一个 agent deployment 的 detector stack，并标记 gaps（missing kill switch、missing canary、circuit breaker threshold too loose）。

## 练习

1. 运行 `code/main.py`。确认 circuit breaker 在 turn 5（第五次 identical call）触发，canary 在 turn 9（fake-key read）触发。

2. 添加一个 statistical detector：tool-call rate 上的 EWMA z-score。输入一个缓慢 drift 的 trajectory，并展示 detector 从不触发。然后添加 hard limit（10 分钟内不超过 50 个 tool calls），展示 hard limit 在同一 trajectory 上触发。

3. 为 browser agent（第 11 课）设计 canary token set。列出至少三个 canaries，以及每个检测什么。

4. 阅读 Cilium network-policy docs。具体描述一个 egress-redirect quarantine flow：哪个 policy selector、哪个 pod、哪个 egress rewrite、哪个 alert。从“决定 quarantine”到“第一个 redirected packet”的 wall-clock latency 由什么决定？

5. 为 kill-switched agent 定义 re-enable procedure。谁可以 re-enable？必须记录什么？在 re-enable 前 agent 必须改变什么？

## 关键术语

| Term | What people say | What it actually means |
|---|---|---|
| Kill switch | “Off button” | 位于 agent edit surface 之外的 boolean；每个 consequential action 都检查 |
| Circuit breaker | “Pattern pause” | 基于 repetition、failure rate 或 rate-limit 的 action-specific trip |
| Canary token | “Honeytoken” | agent 无合法理由触碰的 bait；访问触发 alert |
| Honeypot | “Forensic sandbox” | 用于观察 quarantined agent 的 redirected traffic / workspace |
| EWMA | “Moving average” | Exponentially weighted；适应 drift（feature + bug） |
| CUSUM | “Cumulative sum” | 检测相对 baseline 的持续 shift |
| Hard limit | “Constitutional rule” | 不适应；不随历史变化 |
| Constitutional limit | “Always-true rule” | 绑定第 17 课 constitution；不能被 agent 编辑 |

## 延伸阅读

- [Anthropic — Measuring agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) — autonomous agents 的 kill-switch 和 circuit-breaker framing。
- [Microsoft Agent Framework — HITL and oversight](https://learn.microsoft.com/en-us/agent-framework/workflows/human-in-the-loop) — production governance patterns。
- [OWASP LLM / Agentic Top 10](https://owasp.org/www-project-top-10-for-large-language-model-applications/) — detection-and-response requirements。
- [Cilium — Network policy and eBPF](https://docs.cilium.io/en/stable/security/network/) — pod-level egress redirect 和 forensic honeypot patterns。
- [Anthropic — Claude's Constitution (January 2026)](https://www.anthropic.com/news/claudes-constitution) — 作为“constitutional limits”的 hardcoded prohibitions。
