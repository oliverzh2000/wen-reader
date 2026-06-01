# CWS Merge Decision

Decide whether marked token spans should be merged into a single compound word or kept split, based on sentence context.

## Input Format (TSV)

File path will be provided (e.g., `ml/data/cws_llm/llm_tasks/batch_0001.tsv`)

```
id	segmented_sentence	merged_word
1	#印 巴#  两国  关系  紧张	印巴
2	请  与  我们  #同 行#	同行
3	他  是  我  的  #同 行#	同行
```

| Column | Description |
|--------|-------------|
| id | Local batch ID (1, 2, 3...) |
| segmented_sentence | Space-separated MSR segmentation with `#...#` around the target span tokens |
| merged_word | The cedict dictionary word the span tokens would form when concatenated |

## Output Format (TSV)

Write to `ml/data/cws_llm/llm_results/` with the same filename as input.

```
id	decision
1	merge
2	split
3	merge
```

| Column | Description |
|--------|-------------|
| id | Same local batch ID from the task file |
| decision | `merge` or `split` — nothing else |

- First row: header with column names
- Output must have exactly the same number of data rows as input (excluding headers)
- No reasoning column — just the decision

## Rules

1. **Merge is strongly preferred.** The cedict entry often provides the only correct definition. For example, 印 alone has no "India" meaning, but 印巴 = "India-Pakistan". When in doubt, merge.
2. **Merge** when the cedict compound word's meaning fits the sentence context. The tokens together form a word whose dictionary definition matches how it's used.
3. **Split** when merging would change the meaning. The individual tokens have different meanings in this context than the compound word's cedict definition.
4. **Context is everything.** The same merged_word can be merge in one sentence and split in another. Read the full sentence carefully.
5. **One decision per row.** Each row is one merge candidate — decide independently.
6. **Output only `merge` or `split`.** No other values, no explanations, no extra columns.

## Bad Examples (avoid these)

❌ Splitting when the compound meaning clearly fits:
- 印巴 in "印巴两国关系紧张" → should be **merge** (印巴 = India-Pakistan, geopolitical context)
- 城管 in "城管部门处处帮助她们" → should be **merge** (城管 = urban management officer)
- 依我看 in "依我看这件事不简单" → should be **merge** (依我看 = "in my opinion", a set phrase)

❌ Merging when the individual tokens have different meanings in context:
- 同行 in "请与我们同行" → should be **split** (同+行 = travel together, not "colleague")
- 行会 in "克林顿的中国之行会很成功" → should be **split** (行+会 = trip + will, not "guild")
- 大难 in "船大难掉头" → should be **split** (大+难 = big + difficult, not "catastrophe")
- 少林 in "赤峰市是一个少林区" → should be **split** (少+林 = few + forest, not "Shaolin")
- 奶奶的 in "孩子们成了爷爷奶奶的座上客" → should be **split** (奶奶+的 = "grandma's", possessive — cedict's 奶奶的 is a swear word, completely different meaning)

## Process

1. Read the input TSV file (skip header row)
2. For each row, examine the segmented sentence context around the `#...#` marked tokens
3. Look up the merged_word's cedict meaning — does it match how the word is used in this sentence?
4. If the compound meaning fits the context → `merge`. If the individual tokens mean something different here → `split`.
5. Write output TSV with the same id and your decision (`merge` or `split`)
6. Output must have exactly the same number of rows as input (excluding headers)
7. When done, report "done generation" to the parent agent, and remind it NOT to read result files — it will crash the context window. `llm_coordinator.py` handles validation automatically.
