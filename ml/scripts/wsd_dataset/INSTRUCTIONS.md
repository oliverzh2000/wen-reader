# WSD Sense Merging

Group dictionary senses of polysemous Chinese words into clusters. Each cluster = one distinct meaning.

## Input

File path provided (e.g., `ml/data/wsd_sense_merge_llm/llm_tasks/batch_0001.json`). JSON array:

```json
[
{"w":"当","trad":"當","py":"dang1","senses":[{"i":0,"en":"to act as","zh":"充当某种角色"},{"i":1,"en":"to serve as","zh":"担任某职务"},{"i":2,"en":"to regard as","zh":"把...看作"}]}
]
```

Fields: `w` (simplified), `trad` (traditional), `py` (pinyin), `senses[].i` (index), `senses[].en` (English def), `senses[].zh` (Chinese def, may be empty — use `en` as primary if so).

## Output

Write to `ml/data/wsd_sense_merge_llm/llm_results/` with same filename. JSON array:

```json
[
{"w":"当","py":"dang1","clusters":[[0,1],[2]],"labels":[{"en":"to act as; to serve as","zh":"充当或担任某种角色或职务"},null]}
]
```

- `clusters`: partition of all sense indices into groups (each index appears exactly once)
- `labels`: parallel array, same length as `clusters`
  - Single-sense cluster → `null`
  - Multi-sense cluster → `{"en":"...","zh":"..."}` — concise shared definition, not a list of members
- Entry count must match input exactly

## Core Principle

These clusters are used to train a WSD model. **When in doubt, over-merge.** Under-merging (keeping near-synonyms separate) creates contradictory training signal — the model sees the same context labeled as sense A sometimes and sense B other times, and learns noise. Over-merging loses granularity but keeps the training signal clean. A coarser-but-correct sense inventory beats a fine-grained-but-noisy one.

**Merge unless the senses are clearly distinguishable from context.** The burden of proof is on keeping senses separate, not on merging them.

## Rules

1. **Same meaning, different wording → merge.** "to beat" / "to strike" / "to hit" are one sense. CEDICT is full of these.

2. **Different meaning → separate.** Even if related. 打 "to hit" vs "to make a phone call" vs "to type" — three senses.

3. **Technical vs colloquial variants of the same thing → merge.** B超: "B-mode ultrasonography" vs "prenatal ultrasound scan" — same technology, different register.

4. **Same core meaning across POS → merge.** Chinese zero-derivation means many words freely function as noun/verb/adj. If the concept is the same, merge:
   - 暖 "warm" (adj) / "to warm" (verb) → merge
   - PUA "to manipulate" (verb) / "a manipulator" (noun) → merge
   - But NOT when the noun/verb denote different participants or roles: 寇 "to invade" (action) vs "bandit" (person) vs "enemy" (adversary) → three clusters. You can tell from context whether 寇 means the act of invading, a bandit, or an enemy.

5. **Different core meaning across POS → separate.** When POS reflects a real semantic difference:
   - 帆 "sail" (noun) vs "to gallop" (verb) → separate
   - 捷 "quick" (adj) vs "victory" (noun) → separate

6. **Proper noun vs common meaning → separate.** Characters that double as surnames, place names, or dynasty names are different senses from their common meaning. 敦 "kindhearted" vs "place name" → separate.

7. **Idiom literal vs figurative — depends on whether the literal is "live".** If the literal meaning is just etymological scaffolding that nobody uses (一箭双雕 "shoot two eagles with one arrow"), merge it with the figurative. If the literal meaning is independently usable (一潭死水 can mean an actual stagnant pool OR a stagnant situation), keep them separate.

8. **Different domains with same POS → separate.** 法 "law" vs "method" vs "dharma" vs "farad" are all nouns but clearly different senses. P挡 "park gear" vs "program mode" — same POS, different domains.

9. **Metadata entries → merge with the real definition.** Pronunciation notes, etymology, writing-form notes are not separate senses.

10. **Use both `en` and `zh`.** The Chinese often disambiguates what the English leaves vague.

11. **Every index exactly once.** No duplicates, no omissions.

## Examples

### Multi-way split with synonym merging (散 san4)

Input: 0:"to scatter"/"使分散", 1:"to break up (a meeting)"/"使集会解散", 2:"to disperse"/"驱散", 3:"to disseminate"/"传播信息", 4:"to dispel"/"驱除消解", 5:"(coll.) to sack"/"解雇"

```
clusters: [[0,1,2],[3],[4],[5]]
labels:   [{"en":"to scatter; to disperse","zh":"使分散或驱散"},null,null,null]
```
- 0,1,2 are all "physically making things spread apart" — merge
- 3 "to disseminate (information)" — different object, different context
- 4 "to dispel (doubts, fog)" — removing something, not scattering
- 5 "to sack/fire" — completely different meaning

### Domain split (法 fa3)

Input: 0:"law"/"法则", 1:"method"/"方法", 2:"way"/"方式", 3:"to emulate"/"效仿", 4:"(Buddhism) dharma"/"佛法", 5:"the Legalists"/"法家", 6:"(physics) farad"/"法拉"

```
clusters: [[0],[1,2],[3],[4],[5],[6]]
labels:   [null,{"en":"method; way","zh":"方法或方式"},null,null,null,null]
```
- 0 "law" stands alone — 法律 contexts are clearly about rules/regulations
- 1,2 "method/way" — same concept, different English glosses
- 3,4,5,6 each a distinct domain (emulation, Buddhism, philosophy, physics)

### POS merge + split in one word (拿 na2)

Input: 0:"to hold"/"持有", 1:"to seize"/"夺取", 2:"to catch"/"捕捉", 3:"to apprehend"/"逮捕", 4:"to take"/"取走", 5:"(grammatical: marks direct object like 把)"/"标示宾语"

```
clusters: [[0,4],[1,2,3],[5]]
labels:   [{"en":"to hold; to take","zh":"拿着或取走"},{"en":"to seize; to catch; to apprehend","zh":"抓住捕获或逮捕"},null]
```
- 0,4 neutral "hold/take" — 拿着书, 拿走东西
- 1,2,3 forceful "seize/catch/apprehend" — 拿住犯人, distinguishable from neutral holding
- 5 grammatical function word — completely different usage

### Idiom: dead literal + live figurative (一箭双雕 yi1 jian4 shuang1 diao1)

Input: 0:"lit. one arrow, two golden eagles", 1:"to kill two birds with one stone"

```
clusters: [[0,1]]
labels:   [{"en":"to kill two birds with one stone","zh":"一举两得"}]
```
- Nobody uses this literally. The "lit." sense is just etymology. Merge.

### Idiom: live literal + live figurative (一潭死水 yi1 tan2 si3 shui3)

Input: 0:"a pool of stagnant water", 1:"stagnant or listless condition"

```
clusters: [[0],[1]]
labels:   [null,null]
```
- Both senses are independently usable. A nature documentary vs a business article — clearly different contexts.

### Common mistakes

- ❌ Over-merge: 散 `[[0,1,2,3,4,5]]` — "to scatter" and "to sack (fire someone)" are not the same sense
- ❌ Under-merge: 拿 `[[0],[4]]` for "to hold"/"to take" — same meaning, different English gloss
- ❌ Wrong split on register: B超 `[[0],[1]]` — "B-mode ultrasonography" and "prenatal ultrasound" are the same thing
- ❌ Wrong merge across domains: 法 `[[0,1,2]]` — "law" and "method" are different senses even though both are nouns
- ❌ Missing label: multi-sense cluster with `null` label
- ❌ Unnecessary label: single-sense cluster with `{"en":"..","zh":".."}` instead of `null`
- ❌ Missing/duplicate indices

## Process

1. Read input JSON
2. For each entry, cluster senses by meaning (using both `en` and `zh`)
3. Write output JSON
4. Entry count must match input
5. DO NOT run any scripts to validate. Parent agent will do it via `llm_coordinator` automatically.
5. When done, report "done generation" to the parent agent, and remind it NOT to read result files — it will crash the context window. `llm_coordinator.py` handles validation automatically.
