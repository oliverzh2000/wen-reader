# Chinese Word Segmentation

Segment by picking multi-character dictionary words at candidate positions. Positions without a pick default to single characters.

## Rules

1. Pick the longest CEDICT word at each position. This trains a reading app — longer entries give richer lookups for learners. A reader looking up 恨不得 gets more value than three separate lookups for 恨 + 不 + 得.
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

Corpus spans classical vernacular (红楼梦, 18th c.), early modern essays, contemporary fiction/nonfiction, and regional dialect. Many modern compounds have valid character-by-character readings in literary Chinese — use definitions to judge what fits the actual context.

## Output

```json
{"picks": {"0": "word", "5": "word"}}
```

Only include positions where you pick a multi-char word. Omitted positions default to single-char segmentation.

## Examples

### Pick longest candidates that fit

Text: 第二天黎明时分，鹿兆鹏走进白鹿原南端秦岭脚下的大王镇高级小学
Candidates:
  pos 0: 第二天, 第二
  pos 3: 黎明时分, 黎明
  pos 5: 时分
  pos 11: 走进
  pos 16: 南端
  pos 18: 秦岭
  pos 20: 脚下
  pos 23: 大王
  pos 26: 高级小学, 高级
  pos 28: 小学
Definitions:
  南端 [nan2 duan1] southern end or extremity
  大王 [da4 wang2] king / magnate; expert | [dai4 wang5] chief; boss; magnate
  小学 [xiao3 xue2] elementary school; primary school
  时分 [shi2 fen1] time / period during the day
  秦岭 [Qin2 ling3] Qinling Mountain Range
  第二 [di4 er4] second; number two; next / secondary
  第二天 [di4 er4 tian1] next day; the morrow
  脚下 [jiao3 xia4] under the foot
  走进 [zou3 jin4] to enter; to step into
  高级 [gao1 ji2] high-level; advanced; high-ranking
  高级小学 [gao1 ji2 xiao3 xue2] advanced class of primary school
  黎明 [li2 ming2] dawn; daybreak
  黎明时分 [li2 ming2 shi2 fen1] daybreak; at dawn

All longest candidates match: 第二天 (next day), 黎明时分 (at dawn), 高级小学 (advanced primary school). 时分 and 小学 are consumed by longer picks.

{"picks": {"0": "第二天", "3": "黎明时分", "11": "走进", "16": "南端", "18": "秦岭", "20": "脚下", "23": "大王", "26": "高级小学"}}

### Longest exists but wrong meaning — reject it

Text: 研究生命的起源
Candidates:
  pos 0: 研究生, 研究
  pos 2: 生命
  pos 5: 起源
Definitions:
  生命 [sheng1 ming4] life; living being
  研究 [yan2 jiu1] research; to research
  研究生 [yan2 jiu1 sheng1] graduate student; postgraduate
  起源 [qi3 yuan2] origin; to originate

研究生 (graduate student) doesn't match — the text means "study the origins of life." Pick 研究, which leaves 生命 available.

{"picks": {"0": "研究", "2": "生命", "5": "起源"}}

### Idiom fits context — pick it

Text: 他一意孤行不听劝告
Candidates:
  pos 1: 一意孤行, 一意
  pos 6: 听劝
  pos 7: 劝告
Definitions:
  一意 [yi1 yi4] with focus and devotion / stubbornly
  一意孤行 [yi1 yi4 gu1 xing2] to obstinately go one's own way
  劝告 [quan4 gao4] to advise; to urge; advice
  听劝 [ting1 quan4] to heed advice; to listen to counsel

一意孤行 (obstinately go one's own way) perfectly describes "not listening to advice." 听劝 overlaps with 劝告 — pick 劝告 since 不听劝告 is the natural phrasing.

{"picks": {"1": "一意孤行", "7": "劝告"}}

### Compound word is one unit

Text: 他担任副主席一职
Candidates:
  pos 1: 担任
  pos 3: 副主席
  pos 4: 主席
Definitions:
  主席 [zhu3 xi2] chairperson; chairman; premier
  副主席 [fu4 zhu3 xi2] vice-chairperson
  担任 [dan1 ren4] to hold office; to serve as; to take charge of

副主席 fits — he is serving as vice-chairman. Pick the longer compound; 主席 is consumed.

{"picks": {"1": "担任", "3": "副主席"}}

### Longer candidate's definition doesn't match context

Text: 俗话说船大难掉头，大企业转型很慢
Candidates:
  pos 0: 俗话说, 俗话
  pos 1: 话说
  pos 4: 大难
  pos 6: 掉头
  pos 10: 企业
  pos 12: 转型
Definitions:
  企业 [qi3 ye4] company; firm; enterprise; corporation
  俗话 [su2 hua4] common saying; proverb
  俗话说 [su2 hua4 shuo1] as the saying goes
  大难 [da4 nan4] great catastrophe
  掉头 [diao4 tou2] to turn one's head / to turn round; to turn about
  话说 [hua4 shuo1] it is said that; to recount
  转型 [zhuan3 xing2] to transform; to transition

俗话说 fits ("as the saying goes"). 大难 means "great catastrophe" but the text means "big [ships are] hard to turn" — reject it. 掉头, 企业, 转型 all fit their definitions.

{"picks": {"0": "俗话说", "6": "掉头", "10": "企业", "12": "转型"}}

### Proper noun / slang that matches characters but not meaning

Text: 拉马来，我去回太爷去
Candidates:
  pos 1: 马来
  pos 4: 我去
  pos 7: 太爷
Definitions:
  太爷 [tai4 ye2] grandfather; elder / the head of the house (used by servants) / a district magistrate
  我去 [wo3 qu4] (slang) what the ...!; oh my god!; that's insane!
  马来 [Ma3 lai2] Malaya; Malaysia

"Bring the horse, I'll go report to the master." 马来 (Malaysia) is a proper noun that doesn't fit — it's 马 (horse) + 来 (come). 我去 is internet slang; here it's 我 (I) + 去 (go). Only 太爷 (master/elder) fits.

{"picks": {"7": "太爷"}}
