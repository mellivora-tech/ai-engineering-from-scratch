# Capstone 12 — 视频理解 Pipeline（Scene、QA、Search）

> Twelve Labs 把 Marengo + Pegasus 产品化。VideoDB 发布了 CRUD-for-video API。AI2 的 Molmo 2 发布了 open VLM checkpoints。Gemini long-context 可以原生处理数小时视频。TimeLens-100K 定义了大规模 temporal grounding。2026 年的 pipeline 已经定型：scene segmentation、per-scene caption + embedding、transcript alignment、multi-vector index，以及一个能用 (start, end) timestamps 加 frame previews 回答的 query。这个 capstone 是摄入 100 小时视频，达到公开 benchmark，并衡量 counting 和 action questions 上的 hallucination。

**类型：** Capstone
**语言：** Python（pipeline）、TypeScript（UI）
**前置要求：** 阶段 4（CV）、阶段 6（speech）、阶段 7（transformers）、阶段 11（LLM engineering）、阶段 12（multimodal）、阶段 17（infrastructure）
**覆盖阶段：** P4 · P6 · P7 · P11 · P12 · P17
**时间：** 30 小时

## 问题

long-form video QA 是 2026 年规模下最吃 bandwidth 的 multimodal problem。Gemini 2.5 Pro 可以原生阅读 2 小时视频，但把 100 小时视频摄入成一个可查询 corpus，仍然需要 scene-level index。生产形态结合了 scene segmentation（TransNetV2 或 PySceneDetect）、用 VLM 做 per-scene captioning（Gemini 2.5、Qwen3-VL-Max 或 Molmo 2）、transcript alignment（带 word timestamps 的 Whisper-v3-turbo），以及一个并排存储 caption、frame embedding 和 transcript 的 multi-vector index。query pipeline 会返回带 (start, end) timestamps 和 frame previews 的答案。

benchmarks 是公开的（ActivityNet-QA、NeXT-GQA），再加上你自己的 100-query custom set。counting 和 action-type questions 上的 hallucination 是已知困难 failure class；这个 capstone 明确要求衡量它。

## 概念

摄入时并行运行三条 pipeline。**Scene segmentation** 把视频切成 scenes。**VLM captioning** 为每个 scene 生成 caption，并从 keyframe 生成 frame embedding。**ASR alignment** 产出 word-level timestamps。三条流通过 (scene_id, time range) join。每个 scene 在 multi-vector index（Qdrant）里有三种 vector type：caption embedding、keyframe embedding、transcript embedding。

query time 时，自然语言问题会同时打到三种 vectors；结果用 RRF merge；temporal-grounding adapter（TimeLens-style）会在 top scene 内细化 (start, end) window。VLM synthesizer（Gemini 2.5 Pro 或 Qwen3-VL-Max）接收 query + top scenes + cropped frames，并用 cited timestamps 和 frame preview 回答。

hallucination measurement 很重要。Counting（“how many people enter the room?”）和 action-type（“does the chef pour before stirring?”）问题出了名不可靠。请把它们的准确率与 descriptive questions 分开报告。

## 架构

```
video file / URL
      |
      v
PySceneDetect / TransNetV2  (scene segmentation)
      |
      +--- per-scene keyframe --- VLM caption + frame embedding
      |                            (Gemini 2.5 Pro / Qwen3-VL-Max / Molmo 2)
      |
      +--- audio channel --- Whisper-v3-turbo ASR + word timestamps
      |
      v
multi-vector Qdrant: {caption_emb, keyframe_emb, transcript_emb}
      |
query:
  dense queries against all three -> RRF merge -> top-k scenes
      |
      v
TimeLens / VideoITG temporal grounding (refine start/end within scene)
      |
      v
VLM synth: query + top scenes + frame previews
      |
      v
answer + (start, end) timestamps + frame thumbs + citations
```

## 技术栈

- Scene segmentation：TransNetV2（2024-26 state-of-the-art）或 PySceneDetect
- ASR：通过 faster-whisper 使用 Whisper-v3-turbo，带 word timestamps
- VLM captioner + answerer：Gemini 2.5 Pro、Qwen3-VL-Max 或 Molmo 2
- Temporal grounding：TimeLens-100K-trained adapter 或 VideoITG
- Index：带 multi-vector support 的 Qdrant（caption / frame / transcript）
- UI：Next.js 15，带 HTML5 video player 和 scene thumbnails
- Eval：ActivityNet-QA、NeXT-GQA、custom 100-question hand-labeled set
- Hallucination benchmark：带 hand labels 的 counting 和 action-type subsets

## 构建它

1. **Ingest walker。** 接受 YouTube URLs 或 local MP4s。必要时 downscale 到 720p。持久化 `{video_id, file_path}`。

2. **Scene segmentation。** 运行 TransNetV2 或 PySceneDetect，产出 `[{scene_id, start_ms, end_ms, keyframe_path}]`。目标 100 小时：约 6k-8k scenes。

3. **ASR pass。** 在 audio 上运行 Whisper-v3-turbo；导出 word-level timestamps；切成 per-scene transcript slices。

4. **VLM captioning。** 对每个 scene，用 keyframe 和 short caption template 调用 Gemini 2.5 Pro（或 Qwen3-VL-Max）。产出 caption + frame embedding。

5. **Multi-vector index。** Qdrant collection，包含三个 named vectors。Payload：`{video_id, scene_id, start_ms, end_ms, keyframe_url}`。

6. **Query。** 自然语言问题触发三次 dense queries；用 reciprocal rank fusion 合并；top-k=5 scenes。

7. **Temporal grounding。** 在 top scene 上运行 TimeLens-style adapter，以细化 scene 内的 (start, end) window。

8. **VLM synth。** 用 query + top-3 scene clips（作为 images 或 short clips）+ transcripts 调用 Gemini 2.5 Pro。要求 `(video_id, start_ms, end_ms)` citations。

9. **Eval。** 运行 ActivityNet-QA 和 NeXT-GQA。构建 100-query custom set。报告 overall accuracy + per-class breakdown（counting、action、descriptive）。

## 使用它

```
$ video-qa ask --url=https://youtube.com/watch?v=X "how many cars pass the intersection in the first minute?"
[scene]    23 scenes detected
[asr]      transcript complete, 4m12s
[index]    69 vectors written (23 scenes x 3)
[query]    top scene: scene 3 [01:32-01:54], confidence 0.84
[ground]   refined window: [00:12-00:58]
[synth]    gemini 2.5 pro, 1.4s
answer:    5 cars pass the intersection between 00:12 and 00:58.
citations: [scene 3: 00:12-00:58]
          [frame preview at 00:14, 00:27, 00:44, 00:51, 00:57]
```

## 交付它

`outputs/skill-video-qa.md` 是交付物。给定 YouTube URL 或上传视频，pipeline 会索引 scenes，并用 timestamped citations 回答问题。

| 权重 | 标准 | 衡量方式 |
|:-:|---|---|
| 25 | Temporal grounding IoU | held-out grounding set 上的 Intersection-over-union |
| 20 | QA accuracy | NeXT-GQA 和 custom 100-query |
| 20 | Ingest throughput | 每美元可处理的视频小时数 |
| 20 | UI and citation UX | Timestamp links、thumbnail strip、jump-to-frame |
| 15 | Hallucination rate | Counting 和 action-type accuracy 分开衡量 |
| **100** | | |

## 练习

1. 在 captioning pass 中把 Gemini 2.5 Pro 换成 Qwen3-VL-Max。用人工评分的 50-scene sample 报告 caption quality delta。

2. 把 per-scene frame embedding 降为一个 pooled vector，而不是 multi-vector。衡量 retrieval regression。

3. 构建 “counting strict” mode：synthesizer 为每个 counted instance 提取 timestamp，用户点击验证。衡量 user-verification 是否降低 hallucination。

4. 基准测试 ingest cost：比较三种 VLM 的 hours-of-video-per-dollar。选择 sweet spot。

5. 添加 speaker-diarized transcript：在 audio 上运行 pyannote speaker diarization，并嵌入 per-speaker transcripts。演示 “what did Alice say about X?” queries。

## 关键词汇

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|------------------------|
| Scene segmentation | “Shot detection” | 在 shot boundaries 处把 video 切成 scenes |
| Multi-vector index | “Caption + frame + transcript” | 每种 representation 都有 named vectors 的 Qdrant collection |
| Temporal grounding | “When exactly did it happen” | 为 query answer 细化 (start, end) window |
| Frame embedding | “Visual representation” | keyframe 的 vector embedding，用于 scene-visual similarity |
| RRF fusion | “Reciprocal rank fusion” | 多个 ranked lists 的合并策略；经典 hybrid-retrieval trick |
| Counting hallucination | “Miscount” | VLM 在 “how many X” 问题上的已知 failure mode |
| ActivityNet-QA | “Video-QA benchmark” | long-form video QA accuracy benchmark |

## 延伸阅读

- [AI2 Molmo 2](https://allenai.org/blog/molmo2) — open VLM checkpoints
- [TimeLens (CVPR 2026)](https://github.com/TencentARC/TimeLens) — temporal grounding at scale
- [Gemini Video long-context](https://deepmind.google/technologies/gemini) — hosted reference
- [VideoDB](https://videodb.io) — CRUD-for-video API reference
- [Twelve Labs Marengo + Pegasus](https://www.twelvelabs.io) — commercial reference
- [TransNetV2](https://github.com/soCzech/TransNetV2) — scene segmentation model
- [PySceneDetect](https://github.com/Breakthrough/PySceneDetect) — classic open alternative
- [ActivityNet-QA](https://arxiv.org/abs/1906.02467) — reference eval benchmark
