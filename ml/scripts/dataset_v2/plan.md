# Dataset v2 — Plan

## Goal

Build training data for both CWS and WSD from ebook text. Better domain
match than wiki/subtitles → better models for the reading app.

## History

- **v0** (`wsd_training/`): Sentences fully LLM-generated. No real corpus.
- **v1** (`wsd_dataset/`): Sentences from wiki/subtitles corpus + embedding
  ranking + LLM labeling (incomplete). Wiki is encyclopedic, subtitles too
  colloquial/short.
- **v2** (`dataset_v2/`, this): Sentences from real ebooks. LLM does both
  segmentation (CWS data) and WSD labeling. Two datasets from one pipeline.

## Corpus Source

14 books — wide coverage from classical through contemporary, fiction and
nonfiction, multiple regional dialects:

**Classical (pre-1900):**
- **红楼梦** (tennessine/corpus, txt) — 18th century, classical vernacular

**Early modern / 五四 era (1920s-40s):**
- **周作人文集** (hankinghu, txt) — 1920s-30s essays
- **四世同堂** (hankinghu, txt) — Lao She, 1940s Beijing vernacular

**Mid-century (1950s-80s):**
- **冬天里的春天** (hankinghu, txt) — 1981, scar literature

**Contemporary literary (1990s-2010s):**
- **白鹿原** (hankinghu, txt) — 1993, rural realist, Shaanxi dialect
- **务虚笔记** (hankinghu, txt) — Shi Tiesheng, 1996, philosophical
- **繁花** (hankinghu, txt) — 2012, Shanghai/Wu dialect, dialogue-heavy
- **三体（全集）** (fancy88/ibook, epub) — Liu Cixin, 2006-2010, sci-fi

**Essay / 散文:**
- **俗世奇人** (hankinghu, txt) — Feng Jicai, Tianjin dialect vignettes
- **沉默的大多数** (fancy88/ibook, epub) — Wang Xiaobo, intellectual/irreverent
- **我从未如此眷恋人间** (local epub) — contemporary multi-author essays

**Nonfiction:**
- **万历十五年** (hankinghu, txt) — Huang Renyu, 1982, historical prose
- **《常识》梁文道** (hankinghu, txt) — modern social commentary
- **看见** (fancy88/ibook, epub) — Chai Jing, 2013, investigative journalism

Txt books downloaded by `download_books.py` (GitHub raw URLs).
Epubs downloaded from fancy88/ibook or from local_test_data.

Full corpus: ~51k paragraphs, ~189k sentences, ~6M chars.
After weighted sampling (TARGET_CHARS=3M): ~25k paragraphs, ~84k sentences,
~2.5M chars. Sampling uses per-book weights to control relative representation
(modern/nonfiction weighted 1.5×, older sources 1.0×).

## Pipeline

Entry point: `build_dataset.py`. Run steps individually or all at once.

### Usage

```bash
python build_dataset.py                              # all steps, default model
python build_dataset.py 3                            # just step 3 (CWS annotation)
python build_dataset.py 3 -m claude-opus-4.6 -v      # override model, verbose
python build_dataset.py 3 -m deepseek-v4-pro-thinking -c 4 -o results_test.jsonl
python build_dataset.py 5 -m claude-haiku-4.5 -c 16  # ICWB2 annotation
```

**Flags:**
- `-m / --model` — override model (see available models below)
- `-c / --concurrency` — parallel API calls per batch (default: 8)
- `-v / --verbose` — print per-task details
- `-o / --output` — override output filename (in `data/dataset_v2/`)

### Available models

| Short name | Provider | Pricing (in/out per 1M) | Notes |
|---|---|---|---|
| `deepseek-v4-flash` | DeepSeek | $0.14 / $0.28 | Cheapest, no thinking |
| `deepseek-v4-pro` | DeepSeek | $0.435 / $0.87 | No thinking |
| `deepseek-v4-pro-thinking` | DeepSeek | $0.435 / $0.87 | Thinking enabled |
| `claude-opus-4.6` | Bedrock | $15 / $75 | No thinking, structured JSON |
| `claude-opus-4.6-thinking` | Bedrock | $15 / $75 | Adaptive thinking, effort=high |
| `claude-opus-4.6-thinking-medium` | Bedrock | $15 / $75 | Adaptive thinking, effort=medium |
| `claude-opus-4.6-thinking-low` | Bedrock | $15 / $75 | Adaptive thinking, effort=low |
| `claude-sonnet-4.6` | Bedrock | $3 / $15 | No thinking, structured JSON |
| `claude-sonnet-4.6-thinking` | Bedrock | $3 / $15 | Adaptive thinking, effort=high |
| `claude-sonnet-4.6-thinking-medium` | Bedrock | $3 / $15 | Adaptive thinking, effort=medium |
| `claude-sonnet-4.6-thinking-low` | Bedrock | $3 / $15 | Adaptive thinking, effort=low |
| `claude-haiku-4.5` | Bedrock | $0.80 / $4 | Structured JSON, cheapest Claude |

### Steps

1. **Extract sentences** (`extract_sentences.py`)
   - Downloads books via `download_books.py`
   - Extracts .txt (utf-8/gb18030) and .epub (ebooklib + BeautifulSoup)
   - Splits on Chinese terminators (。！？…), filters non-CJK, deduplicates
   - Weighted sampling to TARGET_CHARS (3M)
   - Output: `sentences.json`

2. **Build CWS tasks** (`cws/build_tasks.py`)
   - Chunks sentences to ≤64 chars (merges within paragraphs, never across)
   - Finds CEDICT candidate words at each position
   - Output: `ebooks_cws_tasks.jsonl`

3. **LLM-annotate CWS** (`cws/annotate.py`)
   - Sends tasks to LLM: picks which multi-char word to use at each position
   - Fork-and-join concurrency with adaptive batch sizing
   - Resume support (skips completed IDs)
   - Prompt caching (DeepSeek auto, Bedrock via cache_control)
   - Output: `ebooks_cws_results.jsonl`

4. **Extract ICWB2** (`cws/icwb2.py`)
   - Loads gold-standard ICWB2 corpora (MSR, PKU, AS, CityU)
   - Sentences where greedy-longest-match == gold → "free" training data
   - Disagreements → LLM tasks (with confirmed positions pre-filled)
   - Proportional sampling to ICWB2_TARGET_CHARS (700k)
   - Output: `icwb2_free.jsonl` + `icwb2_cws_tasks.jsonl`

5. **LLM-annotate ICWB2** (`cws/annotate.py`)
   - Same annotator as step 3, on ICWB2 disagreement tasks
   - Output: `icwb2_cws_results.jsonl`

6. **Assemble CWS training data** (`cws/assemble.py`)
   - Merges ebook results + ICWB2 free + ICWB2 LLM results
   - Reports agreement stats vs greedy baseline
   - Output: `cws_dataset_v2.tsv` (~37k sentences)

7. **Merge old span-scorer data (Opus)**
   - Reads old span-scorer pipeline's batch pairs (wiki/subs + ICWB2)
   - Annotated by Opus thinking — higher quality labels
   - Filters out sentences with multi-char non-CEDICT segments
   - Deduplicates against v2 (v2 wins ties, quote-normalized matching)
   - Adds ~67k unique sentences from wiki/subs/ICWB2 domains
   - Output: `cws_dataset_v2.tsv` (~104k combined sentences)

8. **Build span-scorer dataset (JSONL)**
   - Converts TSV → span-scorer training format
   - For each CJK position: enumerates CEDICT candidates + single char
   - Emits one example per ambiguous position (≥2 candidates)
   - Quote normalization for round-trip validation
   - Output: `span_scorer_dataset_v2.jsonl` (~914k training examples)

9. **Build WSD tasks** — not yet implemented
10. **LLM-annotate WSD** — not yet implemented
11. **Assemble WSD** — not yet implemented

## Architecture: Index-Based CWS Annotation

Instead of asking the LLM to reproduce segmented text (which causes garbling),
we present ambiguous positions with candidate dictionary words and ask the
model to pick which words to use. Segmentation is reconstructed deterministically
from the picks — 100% structurally valid by construction.

This was validated in a test harness comparing 5 prompt variants × 3 models.
Index-based achieved 100% validation rate vs ~55% for text-reproduction.

## Data Sources Summary

The final `span_scorer_dataset_v2.jsonl` combines:

| Source | Sentences | Annotator | Domain |
|---|---|---|---|
| Ebook LLM-annotated | ~7k | Sonnet thinking | Literary/nonfiction |
| ICWB2 free (sampled) | ~16k | Auto (greedy=gold) | News/general |
| ICWB2 LLM-annotated | ~14k | Sonnet thinking | News/general |
| Old wiki/subs (corpora) | ~38k | Opus thinking | Encyclopedic/colloquial |
| Old ICWB2 auto-matched | ~12k | Auto (greedy=gold) | News (MSR only) |
| Old ICWB2 LLM-labeled | ~17k | Opus thinking | News (MSR only) |

After dedup and filtering: ~104k sentences → ~914k span-scorer examples.

## Progress

- [x] Step 1: extraction — 14 books, ~84k sentences, ~2.5M chars
- [x] Step 2: CWS task building — ~54k tasks
- [x] Step 3: CWS annotation — Sonnet thinking
- [x] Step 4: ICWB2 extraction — free + disagreement split (sampled 700k chars)
- [x] Step 5: ICWB2 LLM annotation — Sonnet thinking
- [x] Step 6: CWS assembly — 37k sentences
- [x] Step 7: Merge old Opus data — +67k sentences (104k total)
- [x] Step 8: Span-scorer JSONL — 914k training examples
- [ ] Train span scorer on v2 dataset
- [ ] Steps 9-11: WSD pipeline (not yet implemented)

## Span Scorer Training Results

Trained on `span_scorer_dataset_v2.jsonl` (914k examples, 104k sentences).
Eval split: 2% (2,068 sentences, ~18.4k ambiguous positions).

| Encoder | Params | Best Eval Loss | Eval Acc (plateau) | Notes |
|---|---|---|---|---|
| hfl/chinese-macbert-base | 102M | 0.0294 | ~99.0% | Best at epoch ~1.74 |
| hfl/chinese-electra-180g-base-discriminator | 102M | — | ~99.2% | Nearly identical to MacBERT |
| hfl/chinese-electra-180g-large-discriminator | 324M | — | ~99.2% | No improvement over base |
| hfl/chinese-electra-180g-small-discriminator | 12M | — | TBD | |

**Conclusion:** Model capacity is not the bottleneck — base and large converge
to the same accuracy. The remaining ~1% errors are due to training data issues
(missing words like 中华人民共和国, label noise, distributional gaps). Further
improvement requires better/more targeted training data, not bigger models.

## Hard-Case Eval Results (97 curated test cases)

DP decode uses raw scores (not log-prob) — consistently +3 to +7 across models.

| Model | Dataset | Epochs | Perfect | Oversplit | Passed | Wrong |
|---|---|---|---|---|---|---|
| GreedyLongest | — | — | 74 | 0 | 74/97 | 23 |
| ElectraSmall2 | v2 | 1 | 81 | 3 | 84/97 | 13 |
| ElectraSmall1 | v1 | 4 | 85 | 6 | 91/97 | 6 |
| **ElectraSmall1-int8** | v1 | 4 | 83 | 9 | **92/97** | 5 |
| ElectraBase2 | v2 | 1 | 83 | 5 | 88/97 | 9 |
| ElectraBase1 | v1 | 2 | 90 | 3 | 93/97 | 4 |
| **ElectraBase1-int8** | v1 | 2 | 88 | 5 | **93/97** | 4 |
| SpanMacBERT (all) | v1/v2 | — | 85-89 | 4-8 | 93/97 | 4 |
| CkipFinetuned | — | — | 81 | 13 | 94/97 | 3 |

  Greedy beat (greedy passed but model wrong):
    SpanMacBERT-v2: 0
    SpanMacBERT-new: 0
    SpanMacBERT-old: 0
    ElectraSmall2: 1
    ElectraSmall1: 1
    ElectraSmall1-int8: 1
    ElectraBase2: 2
    ElectraBase1: 0
    ElectraBase1-int8: 0
    CkipBaseline: 0
    CkipFinetuned: 0

**Key findings:**
- v1 dataset (Opus-annotated wiki/subs) >> v2 (ICWB2-heavy) for hard cases
- Int8 quantization: zero degradation for base, +1 for small (noise → oversplit)
- ElectraSmall1-int8 achieves 92/97 at ~12MB — 1 point behind base at ~100MB

**Deployment recommendation:** ElectraSmall1-int8. The 1-point gap vs base
appears only on curated hard cases; on typical ebook text the difference is
negligible. The 8× size reduction (12MB vs 100MB) and faster inference make
it the clear choice for iOS deployment. Distillation from base could close
the remaining gap if needed.


## WSD Pipeline (Steps 9-12)

### Steps 9-11: LLM WSD Annotation

9. **Segment ebooks with span scorer** — Run the trained span scorer on ebook
   sentences to produce word-level segmentation for WSD input.
   Output: `wsd_segments.jsonl`

10. **Build WSD tasks** (`wsd/build_tasks.py`) — Find all polysemous words
    (2+ clusters in `entries_after_merging.json`), cap at 100 occurrences per
    word, output one task per sentence with all its polysemous words.
    Output: `wsd_tasks.jsonl`

11. **LLM-annotate WSD** (`wsd/annotate.py`) — Batch API (Sonnet, 50% discount).
    Each task presents a sentence + all polysemous words + their cluster options.
    LLM picks the correct cluster for each word.
    Output: `wsd_results.jsonl` (23.8k sentences, 102k word disambiguations)

### Step 12: Assemble WSD Training Data

`assemble_wsd()` in `build_dataset.py` merges all WSD sources:

| Source | Examples | Description |
|---|---|---|
| Ebooks LLM-annotated | 102k | Real ebook text, LLM-labeled (Sonnet batch) |
| V1 LLM-generated | 81k | Formulaic sentences crafted per-sense (includes MICS) |

Total: **183k positive examples** → `wsd_dataset_v2.tsv`

sense_zh is built by concatenating Chinese translations from `translation_cache.json`
for each raw English sense in a cluster, joined with "；". This is the text the
biencoder embeds at both train and inference time.

V1 data is auto-converted from 0-based to 1-based cluster IDs via
`_convert_v1_if_needed()` (runs once, caches to `wsd_v1.jsonl`).

### WSD Eval Findings

**Key insight: v1 biencoder (79.1%) outperformed v2 biencoder (74.0%) on the
hand-crafted eval set despite v2 having 3.5× more training data.**

Root cause: distribution mismatch, not model quality.

- The eval set is 209/239 single-char classical/literary words.
- V1's LLM-generated data was deliberately balanced across senses (each sense
  gets N crafted examples). This produces strong signal for rare senses.
- V2's ebook data has natural Zipfian distribution: common senses dominate,
  rare senses barely appear. The eval specifically tests rare sense distinctions.
- V2 models achieve 94.8% on their own training distribution (sanity check),
  confirming no bug — it's purely a distribution gap.

**Solution:** Merge v1 + v2 data. V1 provides balanced rare-sense anchors,
v2 provides natural text distribution and domain matching. Combined dataset
gives both coverage and realism.

### WSD Training Configs

Base model: `thenlper/gte-base-zh` (or `gte-small-zh` for lightweight).
Training: grouped cross-entropy over cosine similarities. ★markers★ around
target word in context to help attention.

| Config | Encoder | Batch | LR | Epochs |
|---|---|---|---|---|
| small | gte-small-zh | 64 | 5e-5 | 5 |
| base | gte-base-zh | 32 | 2e-5 | 5 |
| small-h100 | gte-small-zh | 256 | 5e-5 | 5 |
| base-h100 | gte-base-zh | 128 | 2e-5 | 5 |

### WSD Biencoder Results (Hard Eval, 239 examples)

| Model | TOP-1 | TOP-3 | MRR | Notes |
|---|---|---|---|---|
| GTE-base v2 | 212/239 (88.7%) | 238/239 (99.6%) | 0.936 | Best. fp16 model + int8 emb = lossless |
| GTE-base int8 model | 175/239 (73.2%) | 224/239 (93.7%) | 0.839 | Naive quantize_dynamic wrecks it |
| Distilled small (base→small) | 193/239 (80.8%) | 235/239 (98.3%) | 0.889 | int8 model + int8 emb, ~20MB |
| GTE-small v2 (direct) | 155/239 (64.9%) | 217/239 (90.8%) | 0.785 | Distillation >> direct small |
| Biencoder v1 | 202/239 (84.5%) | 235/239 (98.3%) | 0.912 | Older model, fewer senses |
| Cross-encoder (zh) | 184/239 (77.0%) | 229/239 (95.8%) | 0.864 | |

### Dataset Distribution Analysis

Training data: 183k positive examples, 6,625 words, 13,376 sense clusters.
Full inventory: 10,092 polysemous words, 23,579 sense clusters.

**Coverage gaps:**
- 3,107 sense clusters (20% of those in training words) have 0 examples
- 3,828 words (38%) have zero training data — mostly obscure (3+ char, rare chars)
- Top 500 most-trained words: only 81/1,724 clusters have 0 examples (well-covered)
- Tail 3001+ words: 2,938/7,703 clusters have 0 examples (38%, but rare words)

**Imbalance:** Median Gini = 0.39. 658 words severely imbalanced (Gini > 0.5).

### Error Analysis (GTE-base, 27/239 wrong)

**Error breakdown:**
- 17/27: model picks the *more frequent* cluster over the correct rare one
- 13/27: correct cluster has 11+ training examples (not data-starved — genuinely hard)
- ~13/27: function words (的/了/都/从/再/给/把 etc.) — low user impact

**Sense description issues found:**
- 对 clusters 2 ("towards/regarding" = 朝向) and 4 ("to face" = 面向) too similar
- 开 clusters 1 ("open" = 打开或**开启**) and 2 ("start" = **开启**设备) share "开启"
- 将 cluster 5 (把-marker) vs 2 ("to use/take") — eval label arguably wrong

**Practical impact assessment:**
- Errors concentrate on function words (users rarely look up 的/了) and near-synonym
  clusters (predicted sense is semantically adjacent, not misleading)
- TOP-3 = 99.6% — user almost always sees correct sense within first 3
- High-value disambiguations (方便 convenient vs toilet, 结果 result vs kill,
  打发 dispatch vs pass time) work at ~100%
- **Conclusion: ready to ship.** Improvements are incremental, not blocking.

### Potential Improvements (ordered by expected ROI)

1. **Hard negative mining** — standard technique (FaceNet, DPR, ANCE). Mine
   examples where model margin is small, upsample or generate more for those
   sense pairs. Expected +2-3pp.
2. **Rewrite overlapping sense_zh** — fix 对/开/白 descriptions to be more
   contrastive. Zero-cost, could fix 2-3 specific errors.
3. **Augment tail senses on high-frequency words** — generate 5-10 examples
   for the ~150 zero-count clusters on top-500 words. Expected +1-2pp.
4. **Better distillation** — try alpha=0.3, distill_temp=4.0 for small model.
   Or freeze teacher sense embeddings, only train student context encoder.
5. **QAT for int8** — if need <110MB model. fp16 is lossless so low priority
   unless hard size constraint.


## iOS Deployment — Completed

### Problem (resolved)

CoreML with enumerated shapes used ~2.6GB for ELECTRA-base on iPhone 13 Pro.
Both CWS + WSD models couldn't coexist in memory.

### Solution deployed

Switched to small models with fixed shapes and int8 per-block quantization:

| Model | Encoder | Params | Seq length | Quantization | Bundle size |
|---|---|---|---|---|---|
| CWS | Electra-small-v1 | 12M | 32 (fixed) | int8 per-block (block_size=32) | ~6 MB |
| WSD | GTE-small distilled | ~30M | 16 (fixed) | int8 per-block (block_size=32) | ~8 MB |

**CoreML export config:**
- iOS 18 deployment target (enables per-block quantization)
- fp16 compute precision, int8 weight storage
- Fixed input shapes (no enumerated) — enables ANE, eliminates buffer pre-allocation
- Explicit position_ids passed to avoid Electra int-cast issue in coremltools

**Real CoreML eval results (eval_coreml.py):**
- CWS: 85 perfect + 6 oversplit = 91/97 passed (6 wrong) — matches unquantized PyTorch
- WSD: TOP-1=79.1%, TOP-3=97.9%, MRR=0.880

**Key decisions:**
- Raw score DP (not log-prob) — consistently better on hard cases
- WSD embeddings stored as int8 in SQLite (4B scale + 512B int8 = 516B/cluster, ~12MB total)
- Single-slot score cache covers entire sentence context (encoder runs once per sentence)
- Per-block quantization eliminates accuracy loss vs naive per-channel int8

### Completed items
- [x] Export with fixed shapes: CWS=32, WSD=16
- [x] Switch to Electra-small-v1-int8 for CWS
- [x] Switch to distilled GTE-small-int8 for WSD
- [x] Per-block quantization (block_size=32) for better int8 accuracy
- [x] eval_coreml.py script for end-to-end CoreML validation
- [x] Upload small + distilled models to HuggingFace
- [x] Remove simulator guard (mlprogram works on sim since Xcode 15)
- [x] Re-enable WSD in app (was disabled for memory debugging)
- [x] Fix Swift hiddenDim (768→256), embeddingDim (768→512)
- [x] Simplify BertCoreMLRunner (single sequenceLength, no enumerated array)
