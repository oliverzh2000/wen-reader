# Chinese Word Segmentation

High-precision annotation task. Each sentence has genuinely ambiguous positions where multiple CEDICT words overlap. Correct answers require reading the definitions provided and reasoning about meaning in context.

Segment by picking multi-character dictionary words at candidate positions. Positions without a pick default to single characters.

## Rules

1. Prefer the longest CEDICT word at each position. This data trains a Chinese reading app — longer dictionary entries give richer lookups for learners. A reader looking up 恨不得 gets more value than three separate lookups for 恨 + 不 + 得.
2. Only reject a longer word when its definition describes a **different concept** than what the text means. If the definition matches, pick it — even if the component characters also make sense individually.
3. Read `Definitions` carefully at every position. The correct choice often hinges on a specific sense that isn't obvious from characters alone. Do not skip them.
4. Picking one word consumes its character positions — downstream candidates that overlap are unavailable.
5. When in doubt, pick the longer word. If both the compound and its parts convey the same meaning, that confirms the compound fits.

## Input

```
Text: <sentence>
Candidates:
  pos N: word1, word2, ...
Definitions:
  word [pinyin] sense1; sense2 / distinct_sense
```

Corpus spans classical vernacular (红楼梦), early modern essays, contemporary fiction/nonfiction, and regional dialect. Many modern compounds have valid character-by-character readings in literary Chinese — use definitions to judge.

## Output

```json
{"picks": {"0": "word", "5": "word"}}
```

Only include positions where you pick a multi-char word. Omitted positions default to single-char segmentation.
