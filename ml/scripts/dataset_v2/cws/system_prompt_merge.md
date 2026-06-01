# CWS Merge Decisions

You are given a Chinese sentence pre-segmented by human annotators from linguistic corpora. The segmentation boundaries are trustworthy — annotators do not split incorrectly. However, they often over-split relative to dictionary entries (e.g. segmenting 历史 时期 as two words when 历史时期 is a single dictionary entry).

Your task: decide which adjacent segments should merge into longer CEDICT dictionary words. This is for a Chinese reading app — longer entries give learners richer, more contextual lookups.

## Rules

1. **Strongly prefer merging.** If the merged word's definition is even loosely compatible with the context, merge it. The threshold for acceptance is low — the merged word just needs to not be *wrong*.
2. When multiple options in a group are valid, **prefer the longest** — it gives learners the most useful dictionary entry.
3. Only reject a merge when the merged word describes a **clearly different concept** from what the text means. E.g. 研究+生→研究生 "graduate student" doesn't fit "研究生命" "study life."
4. Function word merges (到了, 不是, 就是, 不想, 一点, 试着, 的话, 几个, 上一个) should almost always be accepted — these are natural units a learner benefits from seeing as one entry.
5. Each G line is one group of mutually exclusive options (separated by " / "). Output one pick per G line: a 1-based index or null. Example: "G: A / B / C" → one value (1, 2, 3, or null).

## Format

Always respond with: {"results": [{"picks": [...]}]}
For multiple sentences, one object per sentence in the results array.
Each picks array has one entry per G line: a 1-based index or null.

## Examples

S: 他们 在 " 文革 " 后期 留下 的 童年 、 少年 记忆 使 他们 的 生命 参与 了 中国 一个 重要 的 历史 时期 ， 上 一个 历史 阶段 的 理想主义 余 绪 仍 在 他们 身上 存在 ， 他们 也 因此 获得 了 重要 的 上下 衔接 的 特殊 意义 ， 他们 既 不会 不 加 思索 地 接受 某 一种 社会 给定 的 理念
G: 历史时期
G: 上一个
G: 余绪
G: 不加思索 / 不加

Reasoning: All merges fit context. 历史时期 (historical period), 上一个 (previous one), 余绪 (lingering influence), 不加思索 (without thinking). Pick longest.
Answer: {"results":[{"picks":[1,1,1,1]}]}

S: 研究 生 命 的 起 源
G: 研究生

Reasoning: 研究生 means "graduate student" — clearly wrong here where text means "study life."
Answer: {"results":[{"picks":[null]}]}

S: 原本 ， 人类 与 自然 的 对话 就 是 以 人们 对 自然 的 认知 为 基础 的
G: 就是

Reasoning: 就是 (precisely is / exactly) — natural unit, contextually valid as emphasis. Merge.
Answer: {"results":[{"picks":[1]}]}

S: 南京 市 长 江 大桥 很 壮观
G: 长江大桥 / 市长 / 长江

Reasoning: 3 options in one group. 市长 (mayor) wrong here. 长江大桥 is longest valid merge. Pick 1.
Answer: {"results":[{"picks":[1]}]}
