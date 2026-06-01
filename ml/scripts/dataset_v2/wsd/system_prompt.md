# Word Sense Disambiguation

You are given a Chinese sentence with one or more target words marked by ★...★. Each marked word has multiple sense clusters with pinyin and English definitions from CEDICT. For each marked word, pick which cluster best matches how the word is used in this specific context.

## Rules

1. Read the sentence context carefully. Pick the cluster whose definitions match the word's meaning in this sentence.
2. Pick exactly one cluster per word (1-based index).
3. For function words (的, 了, 在, 就, 把, etc.), focus on the grammatical role: is it a structural particle, a verbal complement, a conjunction, etc.?
4. If the word is part of a fixed expression or idiom, pick the cluster matching the expression's meaning.
5. When genuinely ambiguous, pick the most natural interpretation for the context.
6. Use the pinyin to help distinguish — different pronunciations often indicate fundamentally different words (e.g. 长 cháng "long" vs zhǎng "chief").

## Format

For a single word in the sentence:
```json
{"senses": [2]}
```

For multiple words in the sentence:
```json
{"senses": [2, 1, 4]}
```

The order of senses matches the order of words as they appear in the sentence (left to right).

For batched sentences:
```json
{"results": [{"senses": [2, 1]}, {"senses": [4]}, ...]}
```

## Examples

Sentence: 白巡★长★乘这个机会解释给钱先生听。
Words:
  [1] 长 (pos 2)
    1. (chang2) long
    2. (chang2) length; distance from one end to the other
    3. (chang2) forte; strong suit
    4. (zhang3) chief; head; elder
    5. (zhang3) older; eldest; senior
    6. (zhang3) to grow; to develop

白巡长 = a patrol chief. 长 here is zhǎng meaning "chief/head." Cluster 4.
{"senses": [4]}

Sentence: 他★开★了一张支票，支票★上★写着五千元。
Words:
  [1] 开 (pos 1)
    1. (kai1) to open (transitive or intransitive)
    2. (kai1) (of ships, vehicles, troops etc) to start; to turn on
    3. (kai1) to boil
    4. (kai1) to write out (a prescription, check, invoice etc)
    5. (kai1) (directional complement) away; off
    6. (kai1) carat (gold)
  [2] 上 (pos 9)
    1. (shang4) on top; upon; above
    2. (shang4) to go up; to attend; to board
    3. (shang4) previous; last; preceding
    4. (shang5) (aspect particle after verb indicating completion or result)

开 + 支票 = write out a check → cluster 4. 支票上 = on the check → cluster 1.
{"senses": [4, 1]}

Sentence: 他★鼓★着勇气走了进去。
Words:
  [1] 鼓 (pos 1)
    1. (gu3) drum; to beat a drum
    2. (gu3) to rouse; to agitate
    3. (gu3) to bulge; to swell

鼓着勇气 = muster/rouse courage. Cluster 2.
{"senses": [2]}

Sentence: 她对此事一★了★百了，★自然★也就放★下★了心来。
Words:
  [1] 了 (pos 6)
    1. (le5) (completed action marker)
    2. (le5) (modal particle indicating change of state)
    3. (liao3) to finish; to end
    4. (liao3) completely; entirely (used as a complement)
    5. (liao3) to understand; to know
    6. (liao4) unofficial variant of 瞭[liao4]
  [2] 自然 (pos 11)
    1. (zi4 ran2) nature; the natural world
    2. (zi4 ran2) natural; naturally
  [3] 下 (pos 16)
    1. (xia4) down; downwards; below; lower
    2. (xia4) to go down; to descend; to get off
    3. (xia4) next (week etc); second (of two parts)
    4. (xia4) (verbal complement: down, off, away)
    5. (xia4) to issue; to give (an order)

一了百了 idiom, 了 = liǎo "to finish" → 3. 自然 = naturally → 2. 放下 = put down (complement) → 4.
{"senses": [3, 2, 4]}
