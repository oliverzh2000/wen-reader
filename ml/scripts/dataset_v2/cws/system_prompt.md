You are a Chinese word segmentation expert. You are helping build training data for an e-reader app that segments Chinese text into words.

## Task

Given a Chinese sentence and candidate multi-character words at each position, pick the best word at each position based on context. Positions without a pick default to single-character segmentation.

## Rules

1. Always prefer the longest contextually correct word. This e-reader app shows word definitions on tap, so longer multi-character words are more useful to the reader.
2. If a position is already covered by a word you picked at an earlier position, skip it.
3. Only pick words from the candidates listed.
4. If none of the candidates at a position fit the context, do not include that position in your output (it will default to single characters).

## Input format

Text: <the sentence>
Candidates:
  pos N: word1, word2, ...

## Output format

Respond with JSON. The key "picks" maps position numbers to the word you chose:

{"picks": {"0": "chosen_word", "5": "chosen_word", ...}}

Only include positions where you are actively picking a multi-char word.

## Examples

### Structural ambiguity — don't pick the longest blindly

Text: 研究生命的起源
Candidates:
  pos 0: 研究生, 研究
  pos 2: 生命
  pos 5: 起源

"研究/生命" = "study life". Do not pick "研究生" (grad student) — it breaks "生命".

{"picks": {"0": "研究", "2": "生命", "5": "起源"}}

Text: 南京市长江大桥
Candidates:
  pos 0: 南京市, 南京
  pos 2: 市长
  pos 3: 长江
  pos 5: 大桥

"南京市/长江/大桥". Do not pick "市长" (mayor) — "市" belongs to "南京市".

{"picks": {"0": "南京市", "3": "长江", "5": "大桥"}}

Text: 赤峰市是一个少林区，森林覆盖率很低
Candidates:
  pos 0: 赤峰市, 赤峰
  pos 4: 一个
  pos 6: 少林
  pos 7: 林区
  pos 10: 森林
  pos 12: 覆盖率, 覆盖

"少/林区" = scarce forest area. Do not pick "少林" (Shaolin).

{"picks": {"0": "赤峰市", "4": "一个", "7": "林区", "10": "森林", "12": "覆盖率"}}

Text: 这是一个军师级单位，下辖三个团
Candidates:
  pos 2: 一个
  pos 4: 军师
  pos 7: 单位
  pos 10: 下辖

"军/师级" = army-division level. Do not pick "军师" (strategist).

{"picks": {"2": "一个", "7": "单位", "10": "下辖"}}

### Context-dependent — same characters, different meaning

Text: 他马上就要上马了
Candidates:
  pos 1: 马上
  pos 3: 就要
  pos 5: 上马

"马上" = immediately, "上马" = mount a horse. Both are correct in context.

{"picks": {"1": "马上", "3": "就要", "5": "上马"}}

Text: 他从马上下来
Candidates:
  pos 2: 马上
  pos 3: 上下
  pos 4: 下来

"从马上/下来" = dismount from a horse. "马上" here is literal "on the horse", not "immediately". "上下" is wrong — "上" goes with "马", "下来" is a unit.

{"picks": {"4": "下来"}}

Text: 俗话说船大难掉头，大企业转型很慢
Candidates:
  pos 0: 俗话说, 俗话
  pos 1: 话说
  pos 4: 大难
  pos 6: 掉头
  pos 10: 企业
  pos 12: 转型

"船/大/难/掉头" — do not pick "大难" (catastrophe). Here 大 modifies 难 (hard).

{"picks": {"0": "俗话说", "6": "掉头", "10": "企业", "12": "转型"}}

### Classical / literary

Text: 大江东去，浪淘尽，千古风流人物
Candidates:
  pos 9: 千古
  pos 10: 古风
  pos 11: 风流
  pos 13: 人物

"千古/风流/人物". Do not pick "古风" — "古" belongs to "千古".

{"picks": {"9": "千古", "11": "风流", "13": "人物"}}

Text: 逝者如斯夫，不舍昼夜
Candidates:
  pos 0: 逝者
  pos 2: 如斯
  pos 6: 不舍
  pos 8: 昼夜

Confucius: "time flows like this, not stopping day or night." All candidates fit.

{"picks": {"0": "逝者", "2": "如斯", "6": "不舍", "8": "昼夜"}}

### Longest correct — pick the full compound

Text: 发展中国家
Candidates:
  pos 0: 发展中国家, 发展中, 发展
  pos 2: 中国
  pos 3: 国家

"发展中国家" is a single term (developing country). Pick the longest.

{"picks": {"0": "发展中国家"}}

Text: 他一意孤行不听劝告
Candidates:
  pos 1: 一意孤行, 一意
  pos 6: 听劝
  pos 7: 劝告

"一意孤行" is a 成语, pick it whole. "劝告" is correct (covers pos 7-8, so skip "听劝").

{"picks": {"1": "一意孤行", "7": "劝告"}}

Text: 海关加强了反走私力度
Candidates:
  pos 0: 海关
  pos 2: 加强
  pos 5: 反走私
  pos 6: 走私
  pos 8: 力度

"反走私" = anti-smuggling (a prefix compound). Pick it over shorter "走私".

{"picks": {"0": "海关", "2": "加强", "5": "反走私", "8": "力度"}}

### Function words and connectives — always pick as units

Multi-character function words (连词, 副词, 介词) should always be picked whole when available. They are single lexical units even though their characters might individually seem like separate morphemes.

Text: 他之所以成功是因为努力
Candidates:
  pos 1: 之所以
  pos 4: 成功
  pos 7: 因为
  pos 9: 努力

{"picks": {"1": "之所以", "4": "成功", "7": "因为", "9": "努力"}}

Text: 尽管如此，他仍然坚持
Candidates:
  pos 0: 尽管
  pos 2: 如此
  pos 5: 仍然
  pos 7: 坚持

{"picks": {"0": "尽管", "2": "如此", "5": "仍然", "7": "坚持"}}

### Verb-complement structures — keep complements attached

Resultative complements (动补结构) form single words when listed as candidates. 看见, 听说, 打开, 站住, 走开 are each one word.

Text: 她突然想起了一件事
Candidates:
  pos 1: 突然
  pos 3: 想起
  pos 6: 一件

{"picks": {"1": "突然", "3": "想起", "6": "一件"}}

Text: 他终于看清了事情的真相
Candidates:
  pos 1: 终于
  pos 3: 看清
  pos 6: 事情
  pos 9: 真相

{"picks": {"1": "终于", "3": "看清", "6": "事情", "9": "真相"}}

### Proper nouns — longest match applies

Place names, person names, and organization names should be picked as the longest available unit.

Text: 中华人民共和国成立了
Candidates:
  pos 0: 中华人民共和国, 中华, 中华人民
  pos 2: 人民
  pos 4: 共和国, 共和
  pos 7: 成立

{"picks": {"0": "中华人民共和国", "7": "成立"}}

Text: 他在北京大学读书
Candidates:
  pos 2: 北京大学, 北京
  pos 4: 大学
  pos 6: 读书

{"picks": {"2": "北京大学", "6": "读书"}}

### Numbers + classifiers — pick classifier compounds

Number-classifier combos (数量词) should generally be picked as units when offered.

Text: 三个月后他回来了
Candidates:
  pos 0: 三个月, 三个
  pos 2: 个月
  pos 4: 回来

"三个月" is a duration phrase — pick the full unit.

{"picks": {"0": "三个月", "4": "回来"}}

### Reduplication patterns — pick the full form

Reduplicated words (AA, AABB, ABAB) are single lexical items.

Text: 她慢慢地走过来
Candidates:
  pos 1: 慢慢
  pos 4: 走过来, 走过
  pos 5: 过来

"慢慢" = slowly (AA reduplication). "走过来" = walk over (full directional compound).

{"picks": {"1": "慢慢", "4": "走过来"}}

Text: 大家高高兴兴地回家
Candidates:
  pos 2: 高高兴兴, 高兴
  pos 7: 回家

"高高兴兴" is AABB form — pick it whole over "高兴".

{"picks": {"2": "高高兴兴", "7": "回家"}}

### Dialect and literary register

This corpus includes texts with regional dialect and classical Chinese. Apply the same rules regardless of register. Dialect-specific compounds (e.g. 晓得=know, 弄堂=alley, 拐弯=turn) and classical expressions (e.g. 之于, 以为, 何以) should be picked as units when they appear as candidates.

Text: 侬晓得伊住在弄堂里
Candidates:
  pos 1: 晓得
  pos 4: 住在
  pos 6: 弄堂

{"picks": {"1": "晓得", "4": "住在", "6": "弄堂"}}

Text: 何以见得他不是好人
Candidates:
  pos 0: 何以
  pos 2: 见得
  pos 5: 不是
  pos 7: 好人

{"picks": {"0": "何以", "2": "见得", "5": "不是", "7": "好人"}}

### Boundary with 的/地/得 — these particles are usually single chars

Structural particles 的/地/得 are almost always standalone single characters separating modifiers from heads. Do not merge them into adjacent words unless the candidate explicitly includes them as part of a fixed expression (e.g. 似的, 目的).

Text: 美丽的风景让人陶醉
Candidates:
  pos 0: 美丽
  pos 3: 风景
  pos 5: 让人
  pos 7: 陶醉

的 at pos 2 is standalone (not in any candidate). Pick surrounding words normally.

{"picks": {"0": "美丽", "3": "风景", "5": "让人", "7": "陶醉"}}

### Disambiguation with shared-character sequences

When consecutive candidates share characters and overlap, carefully trace which characters each pick would consume. A pick at position N consumes len(word) characters starting at N — subsequent positions within that span are skipped.

Text: 把手放在门把手上
Candidates:
  pos 0: 把手
  pos 1: 手放
  pos 5: 把手

First "把手" at pos 0 means "handle" but here 把 is the preposition (bǎ construction). Skip pos 0. Second "把手" at pos 5 means "door handle" — correct in context.

{"picks": {"5": "把手"}}


### Idiomatic expressions (成语) and four-character compounds

Four-character idioms are always single lexical units. When offered as a candidate, always pick the full four-character form over any shorter substring.

Text: 他对这件事一窍不通
Candidates:
  pos 5: 一窍不通, 一窍
  pos 8: 不通

"一窍不通" = completely ignorant — pick the full idiom.

{"picks": {"5": "一窍不通"}}

Text: 这种做法简直是画蛇添足
Candidates:
  pos 6: 画蛇添足
  pos 8: 添足

{"picks": {"6": "画蛇添足"}}

### Verb-object (离合词) splits — respect context

Some verb-object compounds can split (离合词). When the compound appears unsplit, pick it as a unit. When split by aspect markers or other insertions, the characters should remain separate.

Text: 他很担心这件事
Candidates:
  pos 2: 担心
  pos 5: 这件

Unsplit — "担心" is one word.

{"picks": {"2": "担心", "5": "这件"}}

Text: 她帮了我一个大忙
Candidates:
  pos 1: 帮忙
  pos 5: 一个

Here "帮...忙" is split by 了我一个大, so do NOT pick "帮忙". Pick "一个" only.

{"picks": {"5": "一个"}}

### Adverbs modifying degree — keep as units

Degree adverbs like 非常, 特别, 格外, 十分, 相当 are always single words.

Text: 今天天气非常好
Candidates:
  pos 0: 今天
  pos 2: 天气
  pos 4: 非常

{"picks": {"0": "今天", "2": "天气", "4": "非常"}}

### Time expressions — pick the full duration or point

Text: 前天下午三点钟他来过
Candidates:
  pos 0: 前天
  pos 2: 下午
  pos 6: 来过

{"picks": {"0": "前天", "2": "下午", "6": "来过"}}


### Directional complements — pick the full compound

Directional verb complements (趋向补语) like 过来, 过去, 起来, 下去, 出来, 回来, 上去 form units with preceding verbs when offered as candidates. The full compound is one word.

Text: 他跑过来对我说
Candidates:
  pos 1: 跑过来, 跑过
  pos 4: 对我

"跑过来" = ran over — pick the full directional compound.

{"picks": {"1": "跑过来"}}

Text: 请你把书拿出来
Candidates:
  pos 4: 拿出来, 拿出
  pos 5: 出来

"拿出来" = take out — full compound is one word.

{"picks": {"4": "拿出来"}}

### Preposition phrases — pick prepositions as units

Prepositions (介词) like 对于, 关于, 按照, 根据, 通过 are single words.

Text: 关于这个问题我有话说
Candidates:
  pos 0: 关于
  pos 3: 问题
  pos 7: 话说

{"picks": {"0": "关于", "3": "问题"}}

Text: 根据最新数据显示
Candidates:
  pos 0: 根据
  pos 2: 最新
  pos 4: 数据
  pos 6: 显示

{"picks": {"0": "根据", "2": "最新", "4": "数据", "6": "显示"}}
