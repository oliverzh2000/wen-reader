# WSD Dataset Pipeline

Three-phase pipeline: sense merging → corpus scan → sentence ranking.

## Phase 1: Sense Merging → `entries_after_merging.json` ✅

LLM-based merging of CC-CEDICT senses. Each polysemous word (2+ non-trivial
senses) gets one LLM call that groups senses into clusters where each cluster
= one distinct meaning.

Scripts: `build_dataset.py` (gen-tasks / assemble), `llm_coordinator.py`,
`INSTRUCTIONS.md` (LLM prompt), `sense_merge_stats.py` (stats below).

### Phase 1 Stats

122,821 total CEDICT entries (word, pinyin pairs); 120,318 unique simplified words.

**Polysemous entry counts at each stage:**

| Stage | Polysemous (≥2) | % of total |
|---|---:|---:|
| Raw CEDICT (all senses) | 46,010 | 37.5% |
| After trivial sense removal | 43,544 | 35.5% |
| After trivial + LLM merge | 11,558 | 9.4% |

Trivial senses (variant-of, abbr-for, see-also, etc.): 11,662 / 200,072 total
senses removed (5.8%). 10,965 entries (8.9%) had at least one trivial sense.

**LLM merge outcomes** (43,544 entries with ≥2 non-trivial senses):

| Outcome | Entries | % |
|---|---:|---:|
| Collapsed to 1 cluster (was "polysemous" only due to re◊dundant glosses) | 31,986 | 73.5% |
| Reduced but still 2+ clusters | 5,868 | 13.5% |
| Unchanged (all senses genuinely distinct) | 5,690 | 13.1% |

Among the 11,558 still-polysemous entries: avg 3.28 senses → 2.29 clusters (30.1% reduction).

**Reduction by raw sense count:**

| Raw senses | Entries | Avg after trivial rm | Avg after LLM merge | Reduction (trivial→raw) | Reduction (merge→raw) |
|---:|---:|---:|---:|---:|---:|
| 2 | 28,722 | 1.91 | 1.16 | 4.5% | 42.1% |
| 3 | 10,313 | 2.88 | 1.33 | 3.9% | 55.7% |
| 4 | 3,828 | 3.83 | 1.61 | 4.2% | 59.9% |
| 5 | 1,553 | 4.74 | 1.89 | 5.2% | 62.2% |
| 6 | 713 | 5.65 | 2.24 | 5.8% | 62.7% |
| 7 | 366 | 6.62 | 2.70 | 5.4% | 61.4% |
| 8 | 204 | 7.40 | 3.29 | 7.5% | 58.8% |
| 9 | 125 | 8.34 | 3.98 | 7.3% | 55.8% |
| 10 | 58 | 9.26 | 4.02 | 7.4% | 59.8% |
| 11–15 | 117 | 11.56 | 5.27 | 5.1% | 57.8% |
| 16+ | 11 | 17.36 | 7.73 | 6.4% | 58.3% |

Key pattern: trivial removal is a small effect (~4–7%), while LLM merging
drives the real reduction (~42–63%), increasing with sense count. The more
senses CEDICT lists, the more redundancy the LLM finds.

**Cluster size distribution** (131,395 total clusters):

| Cluster size | Count | % |
|---:|---:|---:|
| 1 sense | 91,541 | 69.7% |
| 2 senses | 27,908 | 21.2% |
| 3 senses | 8,409 | 6.4% |
| 4 senses | 2,429 | 1.8% |
| 5+ senses | 1,108 | 0.8% |

30.3% of clusters are multi-sense (i.e., the LLM merged ≥2 CEDICT definitions
into a single meaning). Quality ~85–90% correct; errors lean toward mild
under-merging.

## Phase 2: Corpus Scan → `sentence_index.json`

1. Load CWS model
2. Scan Wikipedia zh + OpenSubtitles zh in one pass
3. For each sentence: quick string match → CWS verification → add to index
4. Deduplicate and cap at 500 sentences per word
5. Save sentence_index.json

Model-independent — run once, reuse for any ranking model.

Script: `build_sentence_index.py` (uses batched CWS inference).
Supports `--resume` for crash recovery.

## Phase 3: Ranking → `wsd_dataset.tsv`

Two-stage retrieval: bge-m3 bi-encoder → bge-reranker-v2-m3 cross-encoder.

1. Load bge-m3 + bge-reranker-v2-m3
2. Load sentence_index.json + entries_after_merging.json
3. For each polysemous word:
   - Encode candidate sentences with bge-m3 (no star markers — tested, no effect)
   - Encode sense definitions with retrieval prompt
   - Cosine similarity → top-20 per sense
   - Cross-encoder rescore top-20 → final top-K
   - Skip senses where best reranker score < threshold (no good examples)
4. Export dataset

Why two stages: bge-m3 is fast (~0.3s/batch) with good sense separation and
transliteration resistance. The reranker adds precision — widens the gap between
correct hits and noise (e.g. 拉/to pull vs transliteration 班杜拉), eliminates
code contamination (队/team vs EnQueue), and gives honest low scores for senses
with no good corpus match (easy to auto-filter).

Sentence embedding reuse: no markers means same sentence = same embedding
regardless of which word we're ranking for. Encode all unique sentences once.

Logic already exists in `wsd_dataset_common.py:rank_sentences()`.

## Open Questions

- Transliteration filtering for single-char words like 拉? CWS segments foreign
  names into single chars, so 拉 in 班杜拉 passes word boundary check.
  Possible fix: check if surrounding CWS segments form a transliteration pattern.
- Minimum reranker score threshold for auto-skipping bad senses?
- Function words (啊, 已, 再, 当) need dialogue corpora — Wikipedia is too formal.
  OpenSubtitles helps but may not be enough. Consider adding a novel/web fiction corpus.

## File Reference

- `build_dataset.py` — Phase 1: gen-tasks / assemble (LLM sense merging)
- `build_sentence_index.py` — Phase 2: corpus scan → sentence_index.json
- `llm_coordinator.py` — tracks LLM batch generation progress + validation
- `INSTRUCTIONS.md` — LLM prompt for sense merging
- `wsd_dataset_common.py` — shared logic (constants, CEDICT parsing, CWS,
  corpus scan, ranking, model loading)
- `lookup_cedict.py` — utility to look up words in CEDICT
- `sense_merge_stats.py` — Phase 1 stats (polysemy reduction, merge outcomes)
- `run_build_dataset.sh` — runner script (phases commented in/out as needed)
