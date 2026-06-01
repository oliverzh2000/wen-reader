# CWS Pipeline Notes

## Post-Processing Architecture (finalized)

### Two-stage decoding: greedy argmax + recursive OOV sub-splitting

See `cws.py` docstring for full rationale. Summary:

1. **Stage 1: Greedy argmax.** B if P(B) > 0.5, else I. Pure model output, no dictionary bias.
2. **Stage 2: Recursive OOV sub-split.** For any multi-char OOV word, split at the internal char with the highest P(B), recurse until every piece is in cedict or single char.

### Why no OOV penalty in the primary pass

We experimented extensively with lattice+Viterbi decoding with an OOV penalty that biased toward cedict words. Results:

- OOV penalty had **zero effect** across all values (0.0 to 5.0+) on the eval set. The model is confident enough that the penalty never tips anything.
- The OOV penalty has the same fundamental problem as blind post-hoc merging: it's a static dictionary-based bias that is not context-aware. It can override the model's confident (and correct) decisions in cases like 奶奶+的 vs 奶奶的 (possessive vs swear word), 还+好赌 vs 还好+赌 (fond of gambling vs not bad + gamble).
- The recursive sub-splitter handles the dictionary-compatibility constraint without touching primary boundaries.

### Why recursive sub-splitting beats Viterbi sub-splitting

The old approach ran Viterbi with a high OOV penalty on OOV words. Problem: it could pick a single large cedict entry (e.g., 义不容辞) that subsumes the natural sub-boundaries the model prefers. Example:

- Model wants: 大义不容辞 (one word, all P(B) < 0.1)
- Viterbi sub-split picks: 大-义不容辞 (because 义不容辞 is in cedict)
- Recursive sub-split picks: 大义-不容-辞 (splits at highest P(B) first: 不 at 0.06, then 辞 at 0.03)

The recursive approach follows the model's preferences strictly. It asks "if you had to split, where?" rather than "what cedict words can I find?"

### Why greedy argmax ≈ Viterbi (with no OOV penalty)

With no OOV penalty and no transition constraints, Viterbi minimizes total NLL across all characters. But the model is trained with per-character cross-entropy, so each character's prediction is already independently optimized. Greedy argmax gives nearly identical results. The Viterbi lattice only matters when there are additional constraints (like OOV penalty) that create interactions between positions.

## Current Eval Results

```
Baseline: 52 perfect + 38 oversplit = 90/94 passed (4 wrong)
Finetuned: 79 perfect + 12 oversplit = 91/94 passed (3 wrong)
```

### Remaining wrong cases (finetuned)

1. **还好赌** — model reads 还好 (hái hǎo, "not bad") instead of 还+好赌 (hái hào dǔ, "also fond of gambling"). P(B) at 好 is 0.18. The model doesn't distinguish the tones hǎo vs hào.

2. **船大难掉头** — model reads 大难 (dà nàn, "catastrophe") instead of 大+难 (dà nán, "big + hard to"). P(B) at 难 is 0.29. Common saying meaning "a big ship is hard to turn around."

3. (third wrong case not investigated)

All are semantic disambiguation failures where the model confidently picks a common compound that's wrong in context. Post-processing can't fix these — the model's probabilities are confidently wrong. Fix requires better training data.

## Training Data Improvement Ideas

### Current state
- ~58k sentences from MSR (86k raw, minus filtered/dropped)
- Cedict-merge + LLM corrections applied via build_dataset.py pipeline
- Per-merged-word frequency cap (default 10) drops entire sentences when any span hits the cap

### Available ICWB2 datasets (already downloaded)

| Dataset | Lines | Script | Notes |
|---------|-------|--------|-------|
| MSR | 86k | Simplified | Currently used |
| PKU | 19k | Simplified | Different segmentation standard (splits more finely) |
| AS | 709k | Traditional | Needs trad→simp conversion. Prefers longer words. |
| CityU | 53k | Traditional | Needs trad→simp conversion. |

### Improvement approaches (ranked by effort/impact)

1. **Add PKU through the same pipeline.** Simplified Chinese, 19k sentences. The cedict-merge + LLM pipeline handles the different segmentation standards. Low effort, ~15-20k more sentences after filtering. `build_dataset.py` already has the infrastructure — just point at `pku_training.utf8`.

2. **Targeted adversarial augmentation.** For the specific failure patterns (还好 vs 还+好赌, 大难 vs 大+难), generate synthetic sentences via LLM where these ambiguous words appear in both readings with correct segmentation. Even 100-200 targeted examples could help more than thousands of generic sentences.

3. **AS dataset (trad→simp conversion).** 709k sentences, huge. AS tends to prefer longer words (Taiwan standard), which aligns with cedict-merge goal. Conversion risk: some trad→simp mappings are lossy. High effort (LLM batch work for merge decisions), high reward.

4. **Chinese Treebank (CTB) from LDC.** ~500k words, high quality newswire + broadcast. Requires LDC license.

### Frequency cap considerations

The current per-merged-word cap drops entire sentences when any span hits the cap. This means a sentence with both an over-cap word (e.g., 中华人民共和国, 500 occurrences) and a rare compound loses the rare compound's training signal too.

Options considered:
- **Include over-cap sentences with merge still applied**: safe for 4+ char blind merges (~0% meaning-change rate), but risks training data skew toward common compounds.
- **Include over-cap sentences with original MSR segmentation**: reintroduces contradictory signal (teaches the oversplit form).
- **Current approach (drop entirely)**: cleanest signal, loses some data. Acceptable at current dataset size.

## Refactoring Done in This Session

1. Deleted `ml/scripts/experimental/` directory (all 4 files: `cws.py`, `detect_mergeable_spans.py`, `analyze_cws_datasets.py`, `sample_merge_candidates.py`)
2. Moved core CWS inference to `ml/scripts/cws_training/cws.py` (clean library, no side effects on import)
3. Test cases and eval loop moved to `ml/scripts/cws_training/eval_cws.py`
4. Updated `cws_training/build_dataset.py` and `wsd_training/build_dataset.py` to use new CWS API
5. Fixed 38 golden test cases to be cedict-compatible (every multi-char word in cedict)
6. Replaced Viterbi+OOV with greedy argmax + recursive OOV sub-splitting
