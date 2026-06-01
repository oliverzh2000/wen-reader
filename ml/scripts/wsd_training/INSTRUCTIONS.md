# WSD Example Generation

Generate natural Chinese sentences demonstrating specific word senses.

## Input Format (TSV)

File path will be provided (e.g., `ml/data/wsd_llm/llm_tasks/batch_0001.tsv` or `batch_0042_missing.tsv`)

```
word	pinyin	sense_en	sense_zh	num_sentences_needed
检举	jian3 ju3	to report (an offense to the authorities)	向当局举报违法或不当行为	3
检举	jian3 ju3	to inform against sb	告发或揭发他人的过失	3
用人	yong4 ren2	to employ sb for a job	雇用某人从事工作	4
用人	yong4 ren2	to manage people	管理和调配人员	4
```

**Important**: Use the `sense_zh` column (not `sense_en`) in your output. The `sense_en` column is for reference only.

## Output Format (TSV)

Write to `ml/data/wsd_llm/llm_results/` with the same filename as input.

Fixed columns: `word`, `pinyin`, `sense_zh`, `sentence1`, `sentence2`, ..., `sentence10`

- First row: header with column names
- Generate exactly `num_sentences_needed` sentences per row
- Leave remaining sentence columns empty
- **Use the `sense_zh` column from input, NOT the `sense_en` column**

```
word	pinyin	sense_zh	sentence1	sentence2	sentence3	sentence4	sentence5	sentence6	sentence7	sentence8	sentence9	sentence10
检举	jian3 ju3	告发或揭发他人的过失	他因为#检举#了同事的贪污行为而遭到排挤，但他并不后悔自己的选择。	在那个特殊年代，互相#检举#揭发成为一种普遍现象，很多人因此家破人亡。	她不愿意#检举#自己的朋友，即使知道对方确实做了错事。
用人	yong4 ren2	雇用某人从事工作	面试，又称口试，是#用人#单位和应聘者之间见面交流的一种就业测试，是用人单位选择员工最常用的方法。	该公司的#用人#标准非常严格，不仅要求专业技能过硬，还要有良好的团队协作能力。	政府出台了一系列优惠政策，鼓励企业#用人#时优先考虑应届毕业生和退役军人。	随着人工智能技术的发展，很多企业开始重新思考#用人#策略，更加注重员工的创新能力。
用人	yong4 ren2	管理和调配人员	作为部门经理，他深谙#用人#之道，总能把合适的人放在合适的岗位上。	古人云"用人不疑，疑人不用"，这是#用人#的基本原则。	她在#用人#方面很有一套，手下的员工都心服口服，工作效率极高。	领导者的#用人#艺术直接影响着整个团队的凝聚力和战斗力。
```

## Rules

1. **Sentence length**: 40-100 characters. Must have rich context that clearly disambiguates the sense.
2. **Complete sentences**: Full grammatical sentences with subject, predicate, and meaningful context. No fragments.
3. **Mark target word**: Use `#word#` delimiters around the word instance showing that sense.
4. **Disambiguation**: The sentence context must make it clear which sense is intended. Similar senses require more distinctive context.
5. **Variety**: Mix sentence structures, topics, and registers (formal/informal, news/literary/conversational).
6. **Natural**: Sentences should sound like real Chinese from news, books, or natural speech — not textbook examples.
7. **One sense per mark**: If word appears multiple times, only mark the instance matching the target sense.

## Bad Examples (avoid these)

❌ Too short (under 40 characters):
- `他走#了#。`
- `我#打#他。`

❌ Fragments (not complete sentences):
- `#立#老师。` (just a name)
- `关于#检举#的规定。` (noun phrase, not a sentence)

❌ Genuinely ambiguous (even a native speaker can't tell which sense) — MOST IMPORTANT TO AVOID:
- `他#走#了。` (could be "walk away", "leave", or "pass away" — no way to tell)
- `她#没#了。` (could be "not have" or "passed away")
- `他#打#人。` (could be "hit someone" or "call someone")
- `这个#行#。` (could be "okay/fine" or "this profession")

The sentence must make ONE specific sense undeniably correct. If a native speaker could reasonably interpret it as multiple senses, it's noise — reject it and write a more specific sentence.

## Process

1. Read the input TSV file (skip header row)
2. For each row, generate exactly `num_sentences_needed` sentences demonstrating that specific sense
3. Write output TSV with same word/pinyin/sense_zh, followed by the generated sentences
4. **Output must have exactly the same number of rows as input (excluding headers)**
5. When done task, report back to the parent agent that you are "done generation", and remind it to not check your work, because the llm_coordinator.py script will do all that automatically. Additionally, checking your work will crash context window!
