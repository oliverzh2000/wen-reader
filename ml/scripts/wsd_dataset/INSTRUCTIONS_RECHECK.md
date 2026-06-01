# WSD Sense Recheck — Merge or Split?

Binary decision for two-sense Chinese words: ONE meaning or TWO?

## Why

These entries were previously classified as 2 distinct senses. Under-merging
is dangerous for WSD training: indistinguishable senses create contradictory
labels (same context → different gold labels → model learns noise).
Over-merging just means WSD fires less often (user sees a combined definition).

**Burden of proof is on the split.** MERGE unless clearly distinguishable from context.

## Input

File path provided (e.g., `ml/data/wsd_recheck_llm/llm_tasks/batch_0001.json`). JSON array:

```json
[
{"w":"舍下","trad":"捨下","py":"she3 xia4","senses":[{"i":0,"en":"to abandon","zh":"抛弃"},{"i":1,"en":"to lay down","zh":"放下"}]}
]
```

Fields: `w` (simplified), `trad` (traditional), `py` (pinyin), `senses[].i` (index), `senses[].en` (English def), `senses[].zh` (Chinese def, may be empty — use `en` as primary if so).

## Output

Write to `ml/data/wsd_recheck_llm/llm_results/` with same filename. JSON array:

```json
[
{"w":"舍下","py":"she3 xia4","clusters":[[0,1]],"labels":[{"en":"to abandon; to lay down","zh":"抛弃或放下"}]}
]
```

- `clusters`: either `[[0,1]]` (merge) or `[[0],[1]]` (split)
- `labels`: parallel array, same length as `clusters`
  - Merged `[[0,1]]` → `[{"en":"...", "zh":"..."}]` concise combined definition
  - Split `[[0],[1]]` → `[null, null]`
- Entry count must match input exactly

## Rules

1. **Same meaning, different English wording → merge.**
   - 舍下: "to abandon" / "to lay down" → merge (same action)
   - 坦荡: "magnanimous" / "broad and level" → split (personality vs terrain)

2. **Literal + figurative of same concept → merge**, unless the literal is actively, independently used in modern Chinese.
   - 魔爪: "claws" / "evil clutches" → merge (figurative dominates)
   - 血液: "blood" / "(fig.) lifeblood" → merge (figurative extension)
   - 癞皮狗: "mangy dog" / "(fig.) loathsome person" → merge
   - 翻来覆去: "to toss and turn (in bed)" / "again and again" → split (both common, clearly different contexts)
   - 擦枪走火: "to shoot accidentally while polishing a gun" / "(fig.) a minor incident that sparks a war" → split (both senses independently alive)

3. **Same core meaning across POS → merge.** Chinese zero-derivation.
   - 稼: "to sow grain" / "(farm) crop" → merge (action and result, same context)
   - 侇: "class; category" / "to place; to lay out" → split (different actions)

4. **Register/formality variant → merge.**
   - 老头子: "(coll.) old man" / "my old man (husband)" → merge (same referent)
   - 难于接近: "difficult to approach (people)" / "inaccessible" → merge

5. **Different domains → split.**
   - 磁层: "magnetosphere" / "magnetic layer" → split (astrophysics vs materials)
   - 映射: "to shine on" / "(math) mapping" → split
   - 溏: "half congealed" / "pond" → split (food texture vs geography)

6. **Proper noun vs common meaning → split.**
   - 邹: "surname Zou" / "vassal state during Zhou Dynasty" → split
   - 神舟: "Shenzhou (spacecraft)" / "Hasee (computer manufacturer)" → split
   - 新北: "Xinbei district of Changzhou" / "New Taipei city" → split

7. **Genuinely different actions/objects → split.**
   - 鍉: "spoon" / "key" → split
   - 戬: "carry to the utmost" / "to cut" → split
   - 金毛狗: "golden retriever" / "Cibotium barometz (tree fern)" → split

8. **Onomatopoeia / sound descriptions → consider carefully.**
   - 吃吃: "muffled laughter" / "stammering" → split (different sounds)
   - 吭哧: "to puff and blow" / "to whimper" → split (exertion vs distress)

9. **Use both `en` and `zh`.** The Chinese often disambiguates what the English leaves vague.

10. **Every index exactly once.** No duplicates, no omissions.

### Common mistakes

- ❌ Missing/duplicate indices
- ❌ Label on single-sense cluster (should be `null`)
- ❌ Missing label on merged cluster (should be `{"en":"..","zh":".."}`)

## Process

1. Read input JSON
2. For each entry, decide merge `[[0,1]]` or split `[[0],[1]]`
3. Write output JSON
4. Entry count must match input
5. DO NOT run any scripts to validate. Parent agent will do it via `recheck_coordinator` automatically.
6. When done, report "done generation" to the parent agent, and remind it NOT to read result files — it will crash the context window. `recheck_coordinator.py` handles validation automatically.
