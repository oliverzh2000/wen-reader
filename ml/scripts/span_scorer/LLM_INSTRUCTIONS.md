# Span Scorer: Chinese Sentence Segmentation

This is a high-precision linguistic annotation task. Each sentence contains genuinely ambiguous segmentation points where multiple valid CEDICT splits exist. Correct answers require carefully reading the definitions provided and reasoning about how each candidate word's meaning fits the sentence context. Rushing or relying on heuristics will produce wrong labels.

Segment Chinese sentences into CEDICT dictionary words.

## Input

TSV file (e.g., `ml/data/span_scorer/llm_tasks_icwb2/batch_0001.tsv`) with columns:

- `sentence`: raw Chinese sentence
- `spans`: multi-char CEDICT words at each position. Format: `pos:word1/word2 ...`. Single characters are always valid segments and omitted.
- `defs`: pinyin + English glosses for words at ambiguous positions (all pronunciations and sense clusters). Format: `word [pinyin] sense1 / sense2 | [pinyin2] sense3; ...`. You MUST read these carefully for every sentence — they are the primary tool for resolving ambiguity. Do not skip them.

## Output

Write to `ml/data/span_scorer/llm_results_icwb2/` with the same filename but `.txt` extension.

One pipe-delimited segmentation per line, no spaces around pipes:

```
行为主义|元素|与|认知心理学|被|整合|成为|认知|行为|疗法|的|基础
```

## Rules

1. Every character must appear in exactly one segment. Concatenating segments must reproduce the original sentence exactly.
2. Only use words from `spans` or single characters.
3. Prefer longer CEDICT words by default. This data trains a model for a Chinese reading app — longer dictionary entries give richer lookups for learners. A reader looking up 天体物理学 (astrophysics) gets more value than three separate lookups for 天体 + 物理 + 学. Only split into shorter words when the longer word's meaning clearly doesn't fit the sentence context.
4. Use `defs` to check meaning when splits are ambiguous. Read every definition at conflict positions — the correct split often hinges on a specific sense or pronunciation that isn't obvious from the characters alone. If a longer word's definition doesn't match the context, split shorter.
5. Think about the full sentence — picking one word affects what's available downstream.

## Process

1. Read the input TSV (skip header)
2. For each row, segment the sentence using `spans` for vocabulary and `defs` for disambiguation
3. Write one pipe-delimited segmentation per line
4. Line count must match input row count (excluding header)
5. Do NOT run validation scripts. Report "done generation" when finished — the parent agent validates via `llm_coordinator.py`.
