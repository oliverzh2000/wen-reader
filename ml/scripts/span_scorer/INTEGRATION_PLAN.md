# Span Scorer Integration Plan

Replace the CWS (B/I tagging) segmentation pipeline with the span scorer.
The span scorer is structurally constrained to only output CEDICT words,
eliminating the OOV sub-splitting stage entirely. It also enables a
"continue splitting" feature where the user can progressively break a
long word into shorter CEDICT constituents.

## Status: Implemented

All phases complete. The old CWS pipeline (`CoreMLSegmentationService`)
has been removed.

## Architecture

### On-Device Pipeline

1. **Encoder** (CoreML, ~75ms on simulator CPU): Full BERT encoder
   (`span_scorer_encoder.mlpackage`). Takes the full paragraph as context
   (truncated to 510 chars centered on the run). Outputs hidden states
   `[1, seq_len, 768]`.

2. **MLP Span Head** (pure Swift math, ~1.5ms): For each candidate CEDICT
   span at each position, scores it using `[H[start]; H[end]; width_emb(len)]`
   → Linear(1600,256) → ReLU → Linear(256,1) → scalar. Weights loaded from
   `span_head_weights.bin` (~1.6MB flat float32 binary).

3. **DP Decode** (pure Swift, <0.01ms): Viterbi-style DP over precomputed
   scores to find the best non-overlapping CEDICT cover.

### Caching

Single-slot cache: sentence text → all precomputed span scores (a dict of
position → [(word, score)]). ~50-100 floats per sentence. Encoder + MLP
run once per sentence. DP re-runs every time (it's free). Sub-splitting
reuses the same cached scores with a different mask.

### Continue Splitting (Sub-split)

When the user long-presses an already-highlighted multi-char word:
- Mask that word (and any longer words at the same position) to `-inf`
- Re-run DP on that word's character range → next-best CEDICT cover
- Sub-splits accumulate within a session (e.g. 4→2+2→1+1+2→1+1+1+1)
- Dismissing the dictionary popover resets to fresh segmentation

### Context Window

The encoder receives the full paragraph (`block` from JS) as context,
truncated to 510 chars centered on the run if needed. The run (contiguous
Chinese characters between punctuation) is the only part segmented by DP.
This gives the encoder maximum surrounding context for better hidden states.

## Files

### ML Pipeline (Python)

| File | Purpose |
|------|---------|
| `ml/scripts/export_coreml.py` | Export encoder to CoreML + head weights to binary. `python scripts/export_coreml.py span` |
| `ml/scripts/run_pipeline.sh` | Full pipeline: gen-tasks, assemble, train, export, bundle |
| `ml/scripts/eval_pipeline.py` | End-to-end eval: span scorer segmentation + WSD |
| `ml/scripts/span_scorer/debug_scores.py` | Dump per-span scores for cross-checking with Swift |

### iOS App (Swift)

| File | Purpose |
|------|---------|
| `WenReader/Services/SpanScorerSegmentationService.swift` | CoreML encoder + Swift MLP + DP + score cache + sub-splitting |
| `WenReader/Services/CedictTrie.swift` | In-memory prefix trie for candidate enumeration |
| `WenReader/Services/SegmentationService.swift` | Protocol + dictionary-based fallback |
| `WenReader/Services/SegmentationServiceFactory.swift` | Instantiation + fallback logic |
| `WenReader/Utilities/LongPressGestureHandler.swift` | Gesture handling + sub-split UX |
| `WenReader/Services/BertTokenizer.swift` | Fixed `buildVocab` to split on `\n` only (NEL bug) |

### App Bundle Resources

| File | Source |
|------|--------|
| `span_scorer_encoder.mlpackage` | CoreML encoder model |
| `span_head_weights.bin` | MLP + width embedding weights (flat float32) |
| `cws_vocab.txt` | BERT vocab (21128 tokens, from span scorer model) |
| `cedict.sqlite` | CEDICT dictionary (for trie + lookups) |

### Removed

| File | Reason |
|------|--------|
| `CoreMLSegmentationService.swift` | Replaced by SpanScorerSegmentationService |
| `cws_encoder.mlpackage` | Replaced by span_scorer_encoder.mlpackage |
