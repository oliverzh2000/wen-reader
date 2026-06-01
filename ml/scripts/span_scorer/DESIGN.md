# Span Scorer: Lexicon-Constrained Chinese Word Segmentation

## Goal

Given a Chinese sentence, find the correct CEDICT word at each character
position. Unlike traditional CWS (per-character B/I tagging), we enumerate
all CEDICT spans and score them directly. The output is guaranteed to be
valid CEDICT entries by construction.

## Why Span Scoring Over CWS

Traditional CWS predicts B/I labels per character, then we post-process
(greedy decode → OOV sub-split) to get dictionary-compatible words. This
creates a mismatch: the model optimizes for character-level boundary
accuracy, but we actually want "which CEDICT entry is correct here?"

Span scoring is a more direct formulation:
- Enumerate candidate CEDICT spans at each position (trie lookup)
- Score each candidate in context
- Pick the best non-overlapping cover (DP, same as Viterbi)

Benefits:
- CEDICT-constrained by construction — no OOV, no sub-splitting
- Directly optimizes for the real task (dictionary lookup)
- Simpler on-device pipeline: one encoder pass + span scoring + DP

## Architecture

### Encoder

`ckiplab/bert-base-chinese-ws` — 12-layer, 768-hidden, ~102M params.

This is a `BertForTokenClassification` fine-tuned on CWS (B/I tagging).
We strip the classification head and use only the encoder (`model.bert`),
then attach our span scoring head on top.

Why ckiplab over vanilla MacBERT:
- Already fine-tuned for Chinese word segmentation — hidden representations
  encode word boundary knowledge out of the box.
- Same vocab (21128 tokens) and architecture as bert-base-chinese and
  MacBERT — drop-in replacement, no code changes needed.
- Character-level tokenizer: each CJK character = one token, giving a 1:1
  mapping from characters to hidden states. No subword alignment needed.
- Same proven CoreML export path as bert-base-chinese.
- Warmer start means faster convergence and potentially better results,
  since the encoder already "understands" word boundaries and just needs
  to adapt to the CEDICT-constrained span scoring formulation.

Previously used `hfl/chinese-macbert-base` (MacBERT). Switched to ckiplab
because starting from a CWS-tuned encoder is strictly better — MacBERT's
whole-word masking gives implicit boundary knowledge, but ckiplab has
explicit boundary knowledge from supervised CWS training.

### Span Scoring Head

For a sentence of length N, the encoder produces hidden states H of shape
(N, 768). For each candidate span (i, j), the span representation is:

```
span_repr = [H[i] ; H[j-1] ; width_embedding(j - i)]
```

Where:
- `H[i]` — hidden state at span start (768-dim)
- `H[j-1]` — hidden state at span end (768-dim)
- `width_embedding(j - i)` — learned embedding for span length (64-dim)

This concatenated vector (768 + 768 + 64 = 1600-dim) feeds into a small MLP:

```
Linear(1600, 256) → ReLU → Dropout(0.1) → Linear(256, 1) → scalar score
```

Why start + end is sufficient: after 12 transformer layers of self-attention,
H[i] already encodes information about all other positions in the sentence.
The start and end hidden states together carry rich information about the
span and its context. This is well-established in span-based NER literature
(SpanBERT, 2019).

### Width Embedding

A small learned lookup table encoding span length:

```python
width_embed = nn.Embedding(num_embeddings=max_word_len, embedding_dim=64)
```

CEDICT words are 1–6 characters, so this is 6 × 64 = 384 parameters.
Gives the MLP an explicit length signal so it can learn length preferences
without inferring them indirectly from hidden states.

### Candidate Enumeration

At inference, for each character position, candidates are all CEDICT entries
whose span covers that position. Enumerated via a trie built from CEDICT
simplified forms. Typically 3–8 candidates per position.

For full-sentence segmentation (highlighting all words in the reader view),
enumerate all CEDICT spans at every position and find the best non-overlapping
cover via dynamic programming:

```
score[0] = 0
for i in 0..N:
    for each span (i, j) in cedict_spans_starting_at(i):
        score[j] = max(score[j], score[i] + span_score(i, j))
backtrack to recover spans
```

This is the same DP as the existing CedictSegmentationService Viterbi,
but with learned scores instead of the hand-tuned length heuristic.

## Training Data Generation

### Why New Data

The old CWS dataset was a naive LLM-merge of the PKU corpus. The span
scorer needs data that's specifically CEDICT-aligned and dense with
ambiguous cases. We generate it from scratch using Wikipedia zh +
OpenSubtitles zh, targeting sentences with the most training signal.

### Key Insight: Target Ambiguous Sentences

Most text has many positions where multiple CEDICT spans overlap. But
not all overlaps are equally interesting. We want sentences dense with
genuinely hard segmentation decisions — where context determines the
correct split.

Important lesson learned: you CANNOT use rules like "longer word always
wins" to auto-label even seemingly trivial cases. Example: 奶奶的 is a
CEDICT entry (swear word), but 奶奶 + 的 (grandma + possessive) is the
correct split in most contexts. Every multi-char vs constituent-chars
conflict is potentially context-dependent. No shortcuts — LLM labels
for everything.

### Pipeline

#### Phase 1: Corpus Scan — Find Ambiguous Sentences

1. Build a CEDICT trie from simplified forms
2. Scan Wikipedia zh + OpenSubtitles zh using multiprocessing — each
   worker builds its own trie and OpenCC converter, processes batches
   of ~2000 sentences independently (embarrassingly parallel)
3. Extract sentences from Wikipedia articles using regex-based splitting
   (must start at document start or after 。！？, end with 。！？).
   Subtitles are already single sentences.
4. Filter: convert traditional → simplified, strip spaces, reject
   sentences with Latin letters, unbalanced quotes, ASCII quotes,
   empty parenthetical cruft, or outside the 15–40 character range
5. At each character position, enumerate all CEDICT spans starting there
6. Identify "conflict points": positions where 2+ multi-char CEDICT
   spans partially overlap (e.g., 还好 and 好赌 share character 好)
7. Score each sentence by ambiguity density:
   - Number of conflict points (weighted 0.6)
   - Diversity of conflict patterns (weighted 0.4)
   - Normalized by sentence length
8. Overcollect 5× per source (parallel), then deduplicate across workers

#### Phase 1b: Coverage-Aware Selection (single-threaded)

After overcollection, select the final N sentences per source using a
blended score that balances ambiguity density with CEDICT coverage:

```
final_score = ambiguity_score * (1 - coverage_weight)
            + coverage_bonus * coverage_weight
```

Where `coverage_bonus = uncovered_words_in_sentence / total_words_in_sentence`.
This naturally boosts sentences containing CEDICT words not yet seen in
any previously selected sentence.

Selection runs in multiple passes (default 5) — each pass re-scores
remaining candidates against the current coverage state, picks a chunk,
and updates coverage. This gives the coverage signal a chance to adapt
without full O(n²) recomputation. Pattern and zone dedup caps (max 5
per exact conflict pattern, max 10 per individual conflict zone) are
applied during selection to prevent frequency skew.

Why coverage matters: pure ambiguity ranking tends to over-represent
common conflict patterns (e.g., 中华/华人 appears in many sentences)
while rare CEDICT words that only appear in a few corpus sentences
never get selected. The coverage bonus ensures the model sees a broad
slice of the dictionary at training time.

Sweet spot: sentences 15–40 characters with 3–8 conflict points. Very
long sentences with many conflicts risk cascading LLM errors.

#### Phase 2: LLM Labeling

For each selected sentence, the LLM segments the full sentence using
a compact vocabulary format. Each batch is a TSV file with the sentence
and a `spans` column listing all multi-char CEDICT entries at each
position (single characters are always valid and omitted to save space).

Batch TSV format:
```
sentence	spans	defs
他还好赌博	1:还好 2:好赌 3:赌博	还好[hái hǎo]=not bad; 好赌[hào dǔ]=to be fond of gambling; 赌博[dǔ bó]=to gamble
```

The `spans` column uses the format `pos:word1/word2` listing all
multi-char CEDICT entries at each position. Single characters are always
valid and omitted to save space. The `defs` column provides pinyin and
English glosses for words at conflict positions only — this gives the
LLM the semantic signal needed to disambiguate overlapping candidates
(e.g., knowing 教徒 means "religious follower" makes 犹太|教徒 obvious
over 犹太教|徒). Definitions are omitted for non-conflict positions to
keep token costs down.

The LLM sees the full sentence (for context) and all CEDICT multi-char
options at each position (so it knows what splits are available). It
segments the entire sentence, not just the ambiguous parts — because
even "boring" positions can have context-dependent surprises.

LLM labeling is manual, guided by `LLM_INSTRUCTIONS.md`. Batch
progress is tracked by `llm_coordinator.py`, which checks line counts
in the results directory against the task files.

#### Phase 3: Conversion to Span Training Data

From each LLM-segmented sentence:
1. Reconstruct gold span positions from the segmentation
2. For each character position, enumerate all CEDICT spans covering it
3. Gold span = positive, other CEDICT spans = negatives
4. Every position with 2+ candidates becomes a training example

### Volume Estimates

- 60k LLM-labeled sentences (40k wiki + 20k subtitles) × ~3-5 conflict
  points each = 180-300k hard training examples
- Plus all the unambiguous positions in those sentences as easy examples
  (model needs some easy cases to stay calibrated)
- Total: likely 500k-1M training examples from 60k LLM calls

### Edge Cases

**Cascading ambiguity**: In "他还好赌博", picking 好赌 at position 2
affects whether 赌博 is available at position 3. The LLM must segment
the full sentence to handle these interactions — local per-conflict
labeling would miss cascades.

**Correct word not in CEDICT**: Sometimes the right segmentation
produces a word not in CEDICT (proper nouns, neologisms). Label the
best available CEDICT segmentation — this is what the model will face
at inference anyway.

**LLM quality for rare words**: The `defs` column provides pinyin and
glosses for all words at conflict positions, so the LLM can disambiguate
even obscure entries it might not otherwise know. Definitions are only
included at conflict zones to keep token costs manageable.

**Pattern deduplication**: Two-level caps to avoid frequency skew.
Per exact conflict pattern: max 5 sentences sharing the same full set
of overlapping word pairs. Per individual conflict zone: max 10
sentences featuring any single zone (e.g., the 还好/好赌 overlap).
The zone cap prevents a single common ambiguity from dominating across
many different pattern combinations.

### Future: ICWB2 Hard-Case Mining

Analysis of the ICWB2 human-annotated datasets (PKU + MSR, ~111k
sentences) against greedy-longest-CEDICT segmentation revealed a rich
source of additional training data. See `analyze_icwb2_mismatches.py`.

Results across PKU + MSR training and test sets:
- 13.5% exact match (greedy-longest = gold) — free training data
- 47.9% genuine mismatch — gold chose different boundaries than greedy
- 9.3% oversplit-only — gold splits finer than CEDICT longest match
- 29.3% undersplit-only — mostly numbers/proper nouns not in CEDICT

The genuine mismatches are exactly the "hard cases" the model needs:
sentences where the longest CEDICT match is wrong and context determines
the correct shorter split. Common patterns include:
- 中国人|民进|入 → 中国|人民|进入 (cascading ambiguity, 21x)
- 改革开放 → 改革|开放 (compound word oversplit, 22x)
- 就是 → 就|是, 才能 → 才|能 (function word ambiguity)
- 江泽民 → 江|泽民 (proper name boundary errors, 22x)

To use this data, both matching and mismatched sentences should be
included together — using only matches would bias toward "greedy is
always right." The mismatched sentences need LLM re-segmentation with
our "prefer longer CEDICT word" rule, since ICWB2's gold standard was
designed for NLP tasks (tends to oversplit compounds like 改革开放 that
our reader app would want as single dictionary lookups).

Not yet integrated — waiting to evaluate the ckiplab encoder switch
first before adding more data.

### Future: LLM Labeling Efficiency

Greedy-longest-CEDICT gets ~78% of sentences correct on our eval set.
This means most LLM-labeled training data from the corpora pipeline
confirms what greedy already gets right — expensive but low-value.

The high-value training signal comes exclusively from sentences where
greedy gets the boundaries wrong. The ICWB2 harvest already exploits
this: only "genuine mismatch" sentences (where human annotators chose
different boundaries than greedy) go to the LLM. The "exact match"
sentences are auto-labeled for free.

Future corpora harvesting should apply the same principle: run greedy
on each candidate sentence, and only send sentences to the LLM where
greedy's segmentation has at least one boundary that differs from what
a context-aware segmenter would choose. Sentences where greedy has no
ambiguous positions, or where all ambiguities resolve to the longest
match, can be auto-labeled without LLM cost. The model still needs
these easy examples for calibration, but they don't need human/LLM
judgment — greedy is correct by definition when there's no ambiguity.

## Training

### Data Format

Input: LLM-segmented sentences (from the pipeline above), stored as
pipe-delimited words in plain text files (one segmentation per line).

Conversion to span training data:
1. Reconstruct raw sentence and gold span positions from segments
2. For each character position, enumerate all CEDICT spans covering it
3. The gold span (from the segmentation) is the positive; other CEDICT
   spans at that position are negatives

### Loss Function

Cross-entropy over the candidate set at each position:

```
loss = -log(exp(score_positive) / sum(exp(score_candidate) for all candidates))
```

This is softmax cross-entropy where the "classes" are the candidate spans
at each position. The candidate set varies per position (unlike standard
classification with a fixed label set).

Why not a ranking loss? A margin-based or listwise ranking loss could
explicitly teach the model to rank candidates. But ranking losses need
graded labels (span A is better than B is better than C), and our CWS
data only gives binary signal (correct / not correct). Cross-entropy with
hard positives/negatives is the standard in span-based NER and works well
for small candidate sets (3–8 spans). If close-call errors become a
problem later, we can switch to margin loss to push scores further apart.

### Training Strategy

**Default: Full fine-tuning (encoder + span head).**
- Train the entire model end-to-end
- Discriminative learning rates: encoder 2e-5, span head 1e-3 (AdamW)
- Epochs: 3–5
- With ~50–80k sentences this is safe — catastrophic forgetting is not
  a real concern at this data scale
- Gives the best results because the encoder can adapt its representations
  to the span scoring task (especially for ambiguous cases like 还好赌)

**Quick validation option: Frozen encoder, train span head only.**
- Freeze all ckiplab encoder parameters, train only MLP + width embeddings
- Learning rate: 1e-3
- Epochs: 5–10, converges in minutes
- Useful for validating the pipeline before committing to a full run

### Handling Single-Character Fallback

Every character position must have at least one candidate. If no multi-char
CEDICT entry covers a position, the single character itself is the only
candidate (score doesn't matter, it wins by default). If the single character
IS in CEDICT, it's a valid entry. If not, it's an unknown character — the
span scorer passes it through as-is.

## On-Device (CoreML) Deployment

The model is one unit during training but split into two pieces for
inference efficiency.

### Why Split

The encoder is expensive (~12ms on iPad). The span scoring head is
trivially cheap (two matrix multiplies). By splitting them, we run the
encoder once per sentence and reuse the hidden states to score all
candidate spans. If the head were baked into the CoreML model, we'd
need to re-run the entire encoder for each candidate span.

### Encoder
Export the fine-tuned encoder to CoreML — same path as the existing CWS
model export. One forward pass per sentence → hidden states H.

### Span Scoring Head
Trivially small (~410K params) — runs as raw Swift math rather than a
separate CoreML model:
1. Extract H[i] and H[j-1] from encoder output
2. Look up width embedding (6 × 64 table hardcoded in Swift)
3. Concatenate → two matrix multiplies with ReLU → scalar score

~10 lines of Swift, no CoreML overhead.

### DP Decoding
Same Viterbi-style DP as existing CedictSegmentationService, replacing
the hand-tuned scoring function with the learned span scores.
