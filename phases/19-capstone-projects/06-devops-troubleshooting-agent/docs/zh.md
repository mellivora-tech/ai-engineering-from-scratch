# Capstone 06 — 面向 Kubernetes 的 DevOps Troubleshooting Agent

> AWS 的 DevOps Agent GA，Resolve AI 发布了 K8s playbooks，NeuBird 演示了 semantic monitoring，Metoro 把 AI SRE 绑定到 per-service SLO。生产形态已经确定：alert webhook 触发后，agent 读取 telemetry，遍历 K8s objects 的 graph，排序 root-cause hypotheses，并发布一条带 approval buttons 的 Slack brief。默认 read-only。每个 remediation 都由人类 gate。这个 capstone 就是这样的 agent，在 20 个 synthetic incidents 上评测，并与 AWS 的 Agent 在三个共享案例上比较。

**类型：** Capstone
**语言：** Python（agent）、TypeScript（Slack integration）
**前置要求：** 阶段 11（LLM engineering）、阶段 13（tools and MCP）、阶段 14（agents）、阶段 15（autonomous）、阶段 17（infrastructure）、阶段 18（safety）
**覆盖阶段：** P11 · P13 · P14 · P15 · P17 · P18
**时间：** 30 小时

## 问题

2025-2026 年的 SRE 叙事变成了：“AI agents triage incidents, humans approve remediations.” AWS DevOps Agent、Resolve AI、NeuBird、Metoro、PagerDuty AIOps 都在生产环境中交付了这种形态。agent 会读取 Prometheus metrics、Loki logs、Tempo traces、kube-state-metrics，以及 K8s objects 的 knowledge graph。它在五分钟内产出带 telemetry citations 的 ranked root-cause hypothesis。没有通过 Slack 明确获得人类批准前，它永远不会执行 destructive commands。

大部分难点在 scoping 和 safety，而不是 reasoning。agent 需要 read-only-by-default RBAC surface、hardened MCP tool server，以及每个 considered vs executed command 的 audit logs。它需要知道自己什么时候超出能力边界并升级处理。并且它的运行成本必须足够低，不能让 OOM-kill cascades 生成一张 $5k 的 agent 账单。

## 概念

agent 运行在 knowledge graph 上。节点包括 K8s objects（Pods、Deployments、Services、Nodes、HPAs、PVCs）和 telemetry sources（Prometheus series、Loki streams、Tempo traces）。边表示 ownership（Pod -> ReplicaSet -> Deployment）、scheduling（Pod -> Node）和 observation（Pod -> Prometheus series）。graph 由 kube-state-metrics sync 保持新鲜，并在每次 alert 时重新采样。

alert 触发时，agent 从 affected object 开始 root-cause。它遍历 edges，拉取相关 telemetry slices（最近 15 分钟），并起草 hypothesis。hypothesis 按 evidence 排序：支持它的 telemetry citations 有多少、是否足够新、是否足够具体。top-3 hypotheses 会发到 Slack，附带 graph-path visualizations 和 remediation actions 的 approval buttons。

Remediation 有 gate。默认允许的 actions 都是 read-only。destructive actions（scaling down、rolling back、deleting Pods）需要 Slack approval；ArgoCD rollback hooks 需要 agent 永远不持有的 auth token。audit log 会记录 agent *considered* 的每条 command，而不只是 executed 的 command，这样 review process 才能抓到 near-misses。

## 架构

```
PagerDuty / Alertmanager webhook
           |
           v
     FastAPI receiver
           |
           v
   LangGraph root-cause agent
           |
           +---- read-only MCP tools ----+
           |                             |
           v                             v
   K8s knowledge graph              telemetry slices
     (Neo4j / kuzu)              Prometheus, Loki, Tempo
   ownership + scheduling          last 15m, scoped
           |
           v
   hypothesis ranking (evidence weight)
           |
           v
   Slack brief + approval buttons
           |
           v (approved)
   ArgoCD rollback hook / PagerDuty escalate
           |
           v
   audit log: considered vs executed, every command
```

## 技术栈

- Observability sources：Prometheus、Loki、Tempo、kube-state-metrics
- Knowledge graph：K8s objects + telemetry edges 的 Neo4j（managed）或 kuzu（embedded）
- Agent：带 per-tool allow-list 的 LangGraph，默认 read-only
- Tool transport：基于 StreamableHTTP 的 FastMCP；destructive tools 放在 approval gate 后的独立 server
- Models：Claude Sonnet 4.7 用于 root-cause reasoning，Gemini 2.5 Flash 用于 log summarization
- Remediation：ArgoCD rollback webhook、PagerDuty escalate、Slack approval card
- Audit：append-only structured log（considered、executed、approved、outcome）
- Deployment：带独立 narrow RBAC role 的 K8s deployment；独立 namespace

## 构建它

1. **Graph ingestion。** 每 30 秒把 kube-state-metrics 同步到 Neo4j/kuzu。Nodes：Pod、Deployment、Node、Service、PVC、HPA。Edges：OWNED_BY、SCHEDULED_ON、EXPOSES、MOUNTS、SCALES。Telemetry overlay edges：OBSERVED_BY（一个 Pod 被某个 Prometheus series 观察）。

2. **Alert receiver。** FastAPI endpoint，接收 PagerDuty 或 Alertmanager webhooks。提取 affected object(s) 和 SLO breach。

3. **Read-only tool surface。** 通过 FastMCP 包装 kubectl、Prometheus query、Loki logql、Tempo traceql。每个工具只有很窄的 RBAC verb（“get”、“list”、“describe”）。默认 server 中没有 “delete”、“exec”、“scale”。

4. **Root-cause agent。** 三节点 LangGraph：`sample` 拉取最近 15 分钟 telemetry slice，`walk` 查询 graph 中的相邻 objects，`hypothesize` 起草带 telemetry citations 的 ranked root-cause candidates。

5. **Evidence scoring。** 每个 hypothesis 的 score = recency * specificity * graph-path length inverse * citation count。返回 top-3。

6. **Slack brief。** 发布一个 attachment，包含 hypothesis、graph-path visualization（server-side 渲染的 subgraph image），以及最多一个 remediation action 的 approval buttons。

7. **Remediation gate。** Destructive tools（scale down、roll back、delete）位于第二个 MCP server 上，并受 approval token 保护。agent 只有在人类批准 Slack card 后才能调用它们。

8. **Audit log。** Append-only JSONL：对每个 candidate command，记录它是否被 considered、是否被 executed、谁批准了它。每天发送到 S3。

9. **Synthetic incident suite。** 构建 20 个场景：OOMKill cascade、DNS flap、HPA thrash、PVC fill、noisy neighbor、faulty sidecar、bad ConfigMap rollout、certificate rotation、image-pull backoff 等。按 root-cause accuracy 和 time-to-hypothesis 给 agent 打分。

## 使用它

```
webhook: alert.pagerduty.com -> checkout-api SLO breach, error rate 14%
[graph]   affected: Deployment checkout-api (3 Pods, Node ip-10-2-3-4)
[walk]    neighbors: ReplicaSet checkout-api-abc, Service checkout-api,
           recent rollout 14m ago
[sample]  prometheus error_rate 14%, up-trend; loki 500s on /api/v2/pay
[hypo]    #1 bad rollout: latest image checkout-api:v2.41 fails /healthz
          citations: deploy.yaml (rev 42), prometheus errorRate, loki 500 stack
[slack]   [ROLL BACK to v2.40]  [ESCALATE]  [IGNORE]
          (approval required; agent does not roll back unilaterally)
```

## 交付它

`outputs/skill-devops-agent.md` 是交付物。给定一个 K8s cluster 和 alert source，agent 会产出 ranked root-cause hypotheses 和一个 Slack-gated remediation flow。

| 权重 | 标准 | 衡量方式 |
|:-:|---|---|
| 25 | RCA accuracy on scenario suite | 20 个 synthetic incidents 中 ≥80% root cause 正确 |
| 20 | Safety | audit log 中 destructive-action guard 从不在没有 Slack approval 时触发 |
| 20 | Time-to-hypothesis | 从 alert 到 Slack brief 的 p50 低于 5 分钟 |
| 20 | Explainability | 每个 hypothesis 都有 graph paths 和 telemetry citations |
| 15 | Integration completeness | PagerDuty、Slack、ArgoCD、Prometheus end-to-end 工作 |
| **100** | | |

## 练习

1. 在 AWS 的 DevOps Agent 演示过的同三个 incidents 上运行你的 agent。发布 side-by-side。报告 agent 在哪里出现分歧。

2. 添加一个 “near-miss” audit，标记 agent *considered* 过但如果没有 approval 就会是 destructive 的 command。衡量一周内的 near-miss rate。

3. 把 hypothesis model 从 Claude Sonnet 4.7 换成自托管 Llama 3.3 70B。衡量 RCA accuracy delta 和每次 incident 的 dollar cost。

4. 构建 causal filter：区分 correlated telemetry spikes 和真正 root cause。用 20-scenario labels 训练一个小 classifier。

5. 添加 rollback dry-run：针对有相同 manifest 的 staging cluster 执行 ArgoCD rollback。在 Slack approval button 出现前，在 live cluster 中验证 rollback plan。

## 关键词汇

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|------------------------|
| K8s knowledge graph | “Cluster graph” | Nodes = K8s objects + telemetry series；edges = ownership、scheduling、observation |
| Read-only-by-default | “Scoped RBAC” | agent 的 service account 只有 get/list/describe verbs；destructive verbs 位于 approval 后的独立 server |
| Audit log | “Considered vs executed” | 每个 candidate command 的 append-only 记录：是否运行、谁批准 |
| Hypothesis ranking | “Evidence score” | Recency × specificity × graph-path length inverse × citation count |
| Slack approval card | “HITL gate” | 带 remediation buttons 的 interactive Slack message；人类点击前 agent 不能继续 |
| Telemetry citation | “Evidence pointer” | 支持某个 claim 的 Prometheus query、Loki selector 或 Tempo trace URL |
| MTTR | “Time to resolution” | 从 alert fire 到 SLO recovery 的 wall-clock 时间 |

## 延伸阅读

- [AWS DevOps Agent GA](https://aws.amazon.com/blogs/aws/aws-devops-agent-helps-you-accelerate-incident-response-and-improve-system-reliability-preview/) — 2026 canonical reference
- [Resolve AI K8s troubleshooting](https://resolve.ai/blog/kubernetes-troubleshooting-in-resolve-ai) — competitor reference
- [NeuBird semantic monitoring](https://www.neubird.ai) — semantic-graph approach
- [Metoro AI SRE](https://metoro.io) — SLO-first production framing
- [kube-state-metrics](https://github.com/kubernetes/kube-state-metrics) — cluster-state source
- [LangGraph](https://langchain-ai.github.io/langgraph/) — reference agent orchestrator
- [FastMCP](https://github.com/jlowin/fastmcp) — Python MCP server framework
- [ArgoCD rollback](https://argo-cd.readthedocs.io/en/stable/user-guide/commands/argocd_app_rollback/) — gated remediation target
