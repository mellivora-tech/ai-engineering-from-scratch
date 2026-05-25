# 汉化与双语静态站进度

最后更新：2026-05-25

## 目标

- 使用静态网站方案完成课程汉化。
- 中文站课程页同时显示中文译文和英文原文。
- 不使用外部翻译 API；课程正文翻译由 agent 分批完成。
- 删除原 `web/` 方向，保留 `site/zh/` 静态站作为汉化入口。
- 将进度持续记录在本文件中，便于后续按阶段推进。

## 当前完成情况

| 项目 | 状态 | 说明 |
| --- | --- | --- |
| 中文静态站目录 | 已完成 | 已创建 `site/zh/`，页面导航和主要 UI 已汉化。 |
| 双语课程页 | 已完成 | 有 `zh.md` 的课程同时显示中文和 English；无译文课程回退英文。 |
| 本地 Markdown 副本 | 已完成 | 中文构建会复制 `zh.md` 和对应 `en.md` 到 `site/zh/content/`。 |
| 原 `web/` 目录 | 已处理 | 已删除原 `web/.gitkeep`，原 Web 方向不再作为交付入口。 |
| 全部 435 课翻译 | 已完成 | 已由 agent teams 分阶段翻译，未使用外部翻译 API。 |
| GitHub 仓库 | 已完成 | 已推送到 `mellivora-tech/ai-engineering-from-scratch`。 |

## 翻译总进度

| 指标 | 数量 |
| --- | ---: |
| 英文课文总数 `docs/en.md` | 435 |
| 已有中文课文 `docs/zh.md` | 435 |
| 未翻译课文 | 0 |
| 完成比例 | 100% |
| 未翻译英文词数估算 | 0 |
| 未翻译英文行数估算 | 0 |

## 分阶段进度

| Phase | 已完成 / 总数 | 剩余 |
| --- | ---: | ---: |
| 00-setup-and-tooling | 12 / 12 | 0 |
| 01-math-foundations | 22 / 22 | 0 |
| 02-ml-fundamentals | 18 / 18 | 0 |
| 03-deep-learning-core | 13 / 13 | 0 |
| 04-computer-vision | 28 / 28 | 0 |
| 05-nlp-foundations-to-advanced | 29 / 29 | 0 |
| 06-speech-and-audio | 17 / 17 | 0 |
| 07-transformers-deep-dive | 16 / 16 | 0 |
| 08-generative-ai | 15 / 15 | 0 |
| 09-reinforcement-learning | 12 / 12 | 0 |
| 10-llms-from-scratch | 24 / 24 | 0 |
| 11-llm-engineering | 17 / 17 | 0 |
| 12-multimodal-ai | 25 / 25 | 0 |
| 13-tools-and-protocols | 23 / 23 | 0 |
| 14-agent-engineering | 42 / 42 | 0 |
| 15-autonomous-systems | 22 / 22 | 0 |
| 16-multi-agent-and-swarms | 25 / 25 | 0 |
| 17-infrastructure-and-production | 28 / 28 | 0 |
| 18-ethics-safety-alignment | 30 / 30 | 0 |
| 19-capstone-projects | 17 / 17 | 0 |

## 已翻译课程

- 全部 `phases/**/docs/en.md` 均已有对应 `docs/zh.md`，共 435 篇。

## 下一步待办

| 优先级 | 待办 | 验收标准 |
| --- | --- | --- |
| P0 | 抽样人工校对重点章节 | 每个大主题至少抽查 1-2 篇，确认术语、代码块、链接和中英双栏展示正常。 |
| P1 | 建立术语统一表 | 固定常见译法，如 tensor、gradient、backpropagation、agent、prompt、MCP。 |
| P2 | 页面 UI 深度汉化 | 课程标题、目录、侧栏中仍来自英文数据的内容逐步补中文字段。 |
| P2 | 发布站点 | 配置 GitHub Pages 或其他静态托管，确认 `/zh/` 可访问。 |

## 每批翻译流程

1. 选择一批互不重叠的 `phases/**/docs/en.md`。
2. 派发 agent 翻译为对应 `docs/zh.md`，不使用翻译 API。
3. 保留 Markdown 结构、代码块、命令、路径、变量名和链接。
4. 检查中英文代码围栏数量、一级标题数量和文件非空。
5. 运行 `node site/build.js --locale zh --out site/zh/data.js`。
6. 运行 `node scripts/test_static_zh_site.js`。
7. 浏览器打开对应 `/zh/lesson.html?path=...`，确认中文和 English 同时显示。
8. 提交并推送。

## 注意事项

- 不提交 `.agents/`、`.playwright-mcp/`、截图或本地临时文件。
- 不使用 `scripts/translate_lessons.py` 这类外部 API 翻译路径。
- 课程正文翻译优先保证准确、结构一致和可验证，再逐步统一风格。
- 当前中文站仍有部分课程标题、phase 名称来自英文数据源，属于后续 UI/数据层汉化范围。
