# Security — Secrets、API Key Rotation、Audit Logs、Guardrails

> 通过 centralized vaults（HashiCorp Vault、AWS Secrets Manager、Azure Key Vault）消除 secret sprawl。永远不要把 credentials 存在 config files、VCS 中的 env files、spreadsheets。优先使用 IAM roles，而不是 static keys；CI/CD 使用 OIDC。AI-gateway pattern 是 2026 年解法：apps → gateway → model provider，gateway 在 runtime 从 vault 拉取 credentials。在 vault 中 rotate，所有 apps 几分钟内生效：无需 redeploy，无需 Slack 里问“谁有新 key”。Rotation policy ≤90 days；每次 commit 用 TruffleHog / GitGuardian / Gitleaks 扫描。Zero-trust：MFA、SSO、RBAC/ABAC、short-lived tokens、device posture。PII scrubbing 使用 entity recognition 在转发前 mask PHI/PII；consistent tokenization（Mesh approach）把 sensitive values 映射到 stable placeholders，让 LLM 保留 code/relationship semantics。Network egress：LLM services 在 dedicated VPC/VNet subnet 中，只 whitelist `api.openai.com`、`api.anthropic.com` 等；阻断所有其他 outbound。2026 incident driver：Vercel supply-chain attack 通过 compromised CI/CD credentials exfiltrated env vars，影响数千个 customer deployments。

**类型：** 学习
**语言：** Python（stdlib，玩具版 PII-scrubber + audit-log writer）
**前置要求：** 阶段 17 · 19（AI Gateways），阶段 17 · 13（Observability）
**时间：** ~60 分钟

## 学习目标

- 枚举四个 secret-management anti-patterns（config files in VCS、hardcoded env、spreadsheets、static keys），并说出替代方案。
- 解释 AI-gateway-pulls-from-vault pattern 作为 2026 production standard。
- 实现带 consistent tokenization 的 PII scrubber（同一 value → 同一 placeholder），让 semantics 保留下来。
- 说出 2026 Vercel supply-chain incident，以及它对 CI/CD credential hygiene 的教训。

## 问题

实习生提交了带 API keys 的 `.env`。他们很快删除了它。Keys 已经在 git history 中：GitGuardian scan 捕获到它，你的 rotation process 是“Slack 全团队、更新 40 个 config files、redeploy 所有 services”。8 小时后，一半 services 已 live，另一半还在等 deploy windows。

另外，user prompts 包含 “My SSN is 123-45-6789.” Prompt 发到了 OpenAI。你有 BAA，但内部 policy 要求转发前 mask PII。你没有做。

再另外，你的 EKS cluster 中 LLM pod 可以访问任意 internet host。有人通过 DNS lookup 向 attacker-controlled domain exfil data。没有任何东西阻止。

LLM services 的 security 必须处理这三个 vectors：vault-backed credentials、PII scrubbing、network egress filtering、audit logs。

## 概念

### Centralized vault + IAM-role pull

**Vault**：HashiCorp Vault、AWS Secrets Manager、Azure Key Vault、GCP Secret Manager。唯一 truth source。

**IAM role**：app/gateway 通过 IAM identity 认证，而不是 static key。Vault 在 token lifetime 内返回 secret。

**AI-gateway pattern**：gateway 在 request time 从 vault 拉取 `OPENAI_API_KEY`。在 vault 中 rotate；下一个 request 获取新 key。无需 redeploy。

### Rotation policy ≤ 90 days

适用于所有 API keys、vault root tokens、CI/CD credentials。能自动就自动。手动 rotation 要记录并跟踪。

### Secret scanning

- **TruffleHog** — 对 commits 做 regex + entropy。
- **GitGuardian** — commercial，高 accuracy。
- **Gitleaks** — OSS，在 CI 中运行。

每次 commit 都运行。检测到新 secret 时 block PR。

### Zero-trust posture

- 所有 accounts 必须 MFA。
- 通过 SAML/OIDC 做 SSO。
- 用 RBAC（role-based）或 ABAC（attribute-based）做 fine grained access。
- Short-lived tokens（小时，不是天）。
- Device posture：只允许带 disk encryption 的 corp devices。

### PII / PHI scrubbing

在 prompt 离开你的 infra 前：

1. Entity recognition（spaCy NER、Presidio、commercial）。
2. Mask matched entities：`"My SSN is 123-45-6789"` → `"My SSN is [SSN_TOKEN_A3F]"`。
3. Consistent tokenization（Mesh approach）：同一 value 映射到同一 placeholder，让 LLM 保留 relationships。
4. 可选：对 LLM response 做 reverse mapping。

Static regex filters 捕获基础 patterns；NER 捕获更多。两者都用。

### Input + output guardrails

Input：阻断 known jailbreaks、forbidden topics；按 user 做 rate-limit。

Output：对 leaked secrets 做 regex scrub（API key patterns、refusal contexts 中的 email patterns），用 classifier 检测 policy violations。

### Network egress whitelist

LLM services 放在 dedicated subnet：
- Whitelist：`api.openai.com`、`api.anthropic.com`、vector DB endpoints、vault endpoints。
- 其他全部 drop。
- DNS 通过 allowlist-only resolver（避免 DNS-tunneling exfil）。

### Audit log

每次 LLM call 的 immutable log 包括：
- Timestamp。
- User / tenant。
- Prompt hash（为隐私不存 raw prompt）。
- Model + version。
- Token counts。
- Cost。
- Response hash。
- Any guardrail trips。

按 regulatory requirement 保留（SOC 2 1 年，HIPAA 6 年）。

### 2026 Vercel incident

Supply-chain attack：compromised CI/CD credentials exfiltrated env vars，影响数千个 customer deployments。教训：CI/CD credentials 等同于 prod。存入 vault。Scope 要窄。积极 rotate。

### 你应该记住的数字

- Rotation policy：≤ 90 days。
- 每次 commit 扫描：TruffleHog / GitGuardian / Gitleaks。
- Vercel 2026：CI/CD creds compromised → thousands of customer env vars leaked。
- Audit log retention：SOC 2 = 1 year，HIPAA = 6 years。

## 使用它

`code/main.py` 实现一个带 consistent tokenization 的玩具 PII scrubber，以及 append-only audit log。

## 交付它

本课会产出 `outputs/skill-llm-security-plan.md`。给定 regulatory scope 和 current state，它会规划 vault migration、scrubber、egress、audit log。

## 练习

1. 运行 `code/main.py`。发送两个引用同一 SSN 的 prompts。确认两者得到同一 placeholder。
2. 为一个调用 OpenAI + Anthropic + Weaviate 的 vLLM-on-EKS deployment 设计 network egress policy。
3. 你在 git history 中发现一个 key（2 年前）。正确响应是什么：rotate key、scrub history，还是两者？说明理由。
4. 你的 audit log 每天增长 10 GB。设计 retention tiers（hot 30d、warm 12mo、cold 6yr）。
5. 论证 reverse-tokenization（把真实值替回 LLM response）相较于保持 placeholders visible 是否值得复杂性。

## 关键词汇

| 术语 | 大家常说 | 实际含义 |
|------|----------|----------|
| Vault | “secrets store” | Centralized credential management service |
| IAM role | “identity-based auth” | app assume 的 role；返回 short-lived creds |
| OIDC for CI/CD | “cloud-issued tokens” | CI 中无 static keys，通过 OIDC identity |
| TruffleHog / GitGuardian / Gitleaks | “secret scanners” | Commit-time secret detection |
| RBAC / ABAC | “access control” | Role-based vs attribute-based |
| PII scrubbing | “data masking” | 移除或 tokenize sensitive entities |
| Consistent tokenization | “stable placeholders” | 同一 value 每次得到同一 token |
| Mesh approach | “Mesh tokenization” | 语义保留的 tokenization pattern |
| Egress whitelist | “outbound allowlist” | 只有允许 domains 可达 |
| Audit log | “immutable history” | 用于 compliance 的 append-only record |

## 延伸阅读

- [Doppler — Advanced LLM Security](https://www.doppler.com/blog/advanced-llm-security)
- [Portkey — Manage LLM API keys with secret references](https://portkey.ai/blog/secret-references-ai-api-key-management/)
- [Datadog — LLM Guardrails Best Practices](https://www.datadoghq.com/blog/llm-guardrails-best-practices/)
- [JumpServer — Secrets Management Best Practices 2026](https://www.jumpserver.com/blog/secret-management-best-practices-2026)
- [Microsoft Presidio](https://github.com/microsoft/presidio) — PII detection 和 anonymization。
- [HashiCorp Vault docs](https://developer.hashicorp.com/vault/docs)
