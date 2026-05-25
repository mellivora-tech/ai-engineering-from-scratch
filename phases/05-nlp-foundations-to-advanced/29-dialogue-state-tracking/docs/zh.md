# Dialogue State Tracking

> “I want a cheap restaurant in the north... actually make it moderate... and add Italian.” 三轮对话，三次 state update。DST 让 slot-value dict 保持同步，这样 booking 才能工作。

**类型：** 构建
**语言：** Python
**前置要求：** 阶段 5 · 17（Chatbots），阶段 5 · 20（Structured Outputs）
**时间：** ~75 分钟

## 问题

在 task-oriented dialogue system 中，用户目标被编码为一组 slot-value pairs：`{cuisine: italian, area: north, price: moderate}`。每个 user turn 都可能新增、更改或删除一个 slot。系统必须读取整段 conversation，并正确输出当前 state。

只要一个 slot 错了，系统就会订错餐厅、安排错航班或扣错卡。DST 是用户说的话和后端执行动作之间的铰链。

为什么 2026 年有 LLMs 之后它仍然重要：

- 合规敏感领域（银行、医疗、航空订票）需要 deterministic slot values，而不是 free-form generation。
- Tool-use agents 在调用 APIs 前仍然需要 slot resolution。
- Multi-turn correction 比看起来更难：“actually no, make it Thursday.”

现代 pipeline：经典 DST 概念 + LLM extractors + structured-output guardrails。

## 概念

![DST: dialog history → slot-value state](../assets/dst.svg)

**任务结构。** Schema 定义 domains（restaurant、hotel、taxi）及其 slots（cuisine、area、price、people）。每个 slot 可以为空、填入 closed set 中的值（price: {cheap, moderate, expensive}），或 free-form value（name: “The Copper Kettle”）。

**两种 DST formulation。**

- **Classification。** 对每个（slot, candidate_value）pair 预测 yes/no。适用于 closed-vocab slots。2020 前标准。
- **Generation。** 给定 dialogue，生成 slot values 作为 free text。适用于 open-vocab slots。现代默认。

**Metric。** Joint Goal Accuracy（JGA）：每一轮中 *所有* slots 都正确的比例。All-or-nothing。MultiWOZ 2.4 leaderboard 在 2026 年约 83% 顶部。

**架构。**

1. **Rule-based（slot regex + keyword）。** 窄领域强 baseline。可调试。
2. **TripPy / BERT-DST。** 基于 copy 的 generation，用 BERT encoding。Pre-LLM 标准。
3. **LDST（LLaMA + LoRA）。** Instruction-tuned LLM + domain-slot prompting。在 MultiWOZ 2.4 上达到 ChatGPT-level 质量。
4. **Ontology-free（2024-26）。** 跳过 schema，直接生成 slot names 和 values。处理 open domains。
5. **Prompt + structured output（2024-26）。** LLM + Pydantic schema + constrained decoding。5 行代码，production-ready。

### 经典失败模式

- **Co-reference across turns。** “Let's stay with the first option.” 需要解析是哪一个 option。
- **Over-write vs append。** 用户说 “add Italian.” 你是替换 cuisine，还是追加？
- **Implicit confirmations。** “OK cool”——这是否接受了系统提供的 booking？
- **Correction。** “Actually make it 7 pm.” 必须更新时间，不清空其他 slots。
- **Coreference to previous system utterance。** “Yes, that one.” 是哪个 “that”？

## 构建它

### 第 1 步：rule-based slot extractor

见 `code/main.py`。Regex + synonym dictionaries 能覆盖窄领域 70% 的 canonical utterances：

```python
CUISINE_SYNONYMS = {
    "italian": ["italian", "pasta", "pizza", "italy"],
    "chinese": ["chinese", "chow mein", "noodles"],
}


def extract_cuisine(utterance):
    for canonical, synonyms in CUISINE_SYNONYMS.items():
        if any(syn in utterance.lower() for syn in synonyms):
            return canonical
    return None
```

在 canonical vocabulary 外很脆。对 deterministic slot confirmations 有用。

### 第 2 步：state update loop

```python
def update_state(state, utterance):
    new_state = dict(state)
    for slot, extractor in SLOT_EXTRACTORS.items():
        value = extractor(utterance)
        if value is not None:
            new_state[slot] = value
    for slot in NEGATION_CLEARS:
        if is_negated(utterance, slot):
            new_state[slot] = None
    return new_state
```

三个不变量：

- 永远不要重置用户没有触碰的 slot。
- 显式否定（“never mind the cuisine”）必须清空。
- 用户纠正（“actually...”）必须 overwrite，而不是 append。

### 第 3 步：LLM-driven DST with structured output

```python
from pydantic import BaseModel
from typing import Literal, Optional
import instructor

class RestaurantState(BaseModel):
    cuisine: Optional[Literal["italian", "chinese", "indian", "thai", "any"]] = None
    area: Optional[Literal["north", "south", "east", "west", "center"]] = None
    price: Optional[Literal["cheap", "moderate", "expensive"]] = None
    people: Optional[int] = None
    day: Optional[str] = None


def llm_dst(history, llm):
    prompt = f"""You track the slot values of a restaurant booking across turns.
Dialogue so far:
{render(history)}

Update the state based on the latest user turn. Output only the JSON state."""
    return llm(prompt, response_model=RestaurantState)
```

Instructor + Pydantic 保证 valid state object。没有 regex、没有 schema mismatches、没有 hallucinated slots。

### 第 4 步：JGA evaluation

```python
def joint_goal_accuracy(predicted_states, gold_states):
    correct = sum(1 for p, g in zip(predicted_states, gold_states) if p == g)
    return correct / len(predicted_states)
```

校准：系统有多少轮能把所有 slots 都弄对？MultiWOZ 2.4 的 2026 顶部系统：80-83%。你的 in-domain 系统在窄 vocabulary 上应该超过这个，否则 LLM baseline 会打败你。

### 第 5 步：handling correction

```python
CORRECTION_CUES = {"actually", "no wait", "on second thought", "change that to"}


def is_correction(utterance):
    return any(cue in utterance.lower() for cue in CORRECTION_CUES)
```

检测到 correction 时，overwrite 最近更新的 slot，而不是 append。没有 LLM 帮助很难做对。现代模式：始终让 LLM 从 history 重新生成整个 state，而不是 incremental update；这自然处理 corrections。

## 坑

- **Full-history regeneration cost。** 每轮让 LLM 重新生成 state，总 token 成本是 O(n²)。限制 history 或总结旧 turns。
- **Schema drift。** 事后添加新 slots 会打破旧训练数据。Version your schema。
- **Case sensitivity。** “Italian” vs “italian” vs “ITALIAN”——到处 normalize。
- **Implicit inheritance。** 如果用户之前说过 “for 4 people”，之后请求不同时间不应清空 people。始终传入 full history。
- **Free-form vs closed-set。** Names、times、addresses 需要 free-form slots；cuisines 和 areas 是 closed。Schema 中要混合使用。

## 使用它

2026 年技术栈：

| 场景 | Approach |
|-----------|----------|
| Narrow domain（一两个 intents） | Rule-based + regex |
| Broad domain，有 labeled data | LDST（LLaMA + LoRA on MultiWOZ-style data） |
| Broad domain，无 labels，prod-ready | LLM + Instructor + Pydantic schema |
| Spoken / voice | ASR + normalizer + LLM-DST |
| Multi-domain booking flow | Schema-guided LLM + per-domain Pydantic models |
| Compliance-sensitive | Rule-based primary，LLM fallback + confirmation flow |

## 交付它

保存为 `outputs/skill-dst-designer.md`：

```markdown
---
name: dst-designer
description: 设计 dialogue state tracker：schema、extractor、update policy、evaluation。
version: 1.0.0
phase: 5
lesson: 29
tags: [nlp, dialogue, task-oriented]
---

给定 use case（domain、languages、vocab openness、compliance needs），输出：

1. Schema。Domain list、每个 domain 的 slots、每个 slot 是 open vocabulary 还是 closed vocabulary。
2. Extractor。Rule-based / seq2seq / LLM-with-Pydantic。说明理由。
3. Update policy。Regenerate-whole-state / incremental；correction handling；negation handling。
4. Evaluation。Held-out dialogue set 上的 Joint Goal Accuracy、slot-level precision/recall、hardest slot 的 confusion。
5. Confirmation flow。什么时候明确要求用户确认（destructive actions、low-confidence extractions）。

拒绝对 compliance-sensitive slots 使用没有 rule-based secondary check 的 LLM-only DST。拒绝任何不能在 user correction 后 roll back slot 的 DST。标记没有 version tags 的 schemas。
```

## 练习

1. **简单。** 为 3 个 slots（cuisine、area、price）构建 `code/main.py` 中的 rule-based state tracker。在 10 个手写 dialogues 上测试。测量 JGA。
2. **中等。** 同一数据集使用 Instructor + Pydantic + 小型 LLM。比较 JGA。检查最难的 turns。
3. **困难。** 两者都实现并路由：rule-based primary；当 rule-based 发出 <2 个 slots 且 confidence 低时用 LLM fallback。测量 combined JGA 和每轮 inference cost。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|-----------------|-----------------------|
| DST | Dialogue state tracking | 在 dialogue turns 中维护 slot-value dict。 |
| Slot | 用户意图单位 | 后端需要的命名参数（cuisine、date）。 |
| Domain | 任务区域 | Restaurant、hotel、taxi，一组 slots。 |
| JGA | Joint Goal Accuracy | 每一轮所有 slots 都正确的比例。All-or-nothing。 |
| MultiWOZ | benchmark | Multi-domain WOZ dataset；标准 DST evaluation。 |
| Ontology-free DST | 没有 schema | 直接生成 slot names 和 values，不用固定列表。 |
| Correction | “Actually...” | 覆盖之前已填 slot 的 turn。 |

## 延伸阅读

- [Budzianowski et al. (2018). MultiWOZ — A Large-Scale Multi-Domain Wizard-of-Oz](https://arxiv.org/abs/1810.00278) — 经典 benchmark。
- [Feng et al. (2023). Towards LLM-driven Dialogue State Tracking (LDST)](https://arxiv.org/abs/2310.14970) — LLaMA + LoRA instruction tuning for DST。
- [Heck et al. (2020). TripPy — A Triple Copy Strategy for Value Independent Neural Dialog State Tracking](https://arxiv.org/abs/2005.02877) — copy-based DST workhorse。
- [King, Flanigan (2024). Unsupervised End-to-End Task-Oriented Dialogue with LLMs](https://arxiv.org/abs/2404.10753) — EM-based unsupervised TOD。
- [MultiWOZ leaderboard](https://github.com/budzianowski/multiwoz) — canonical DST results。
