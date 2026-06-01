# ML Integration for Wen Reader

## Approach

### CWS (Chinese word segmentation)

**Best model found:** `ckiplab/bert-base-chinese-ws`
- Best publicly available transformer-based CWS model on HuggingFace
- Preserves word boundaries well, though tends to oversplit (e.g. splits 研究生命 → 研究-生-命 instead of 研究-生命)
- Oversplitting is acceptable for reader app use case — boundaries are still correct, just finer-grained
- Trained on Traditional Chinese but works well on Simplified
- Outperforms `AimanGh/bert-base-chinese-word-segmentation` on hard cases (76/78 vs 72/78 passed)

**Models tested (78 test cases):**

| Model | Perfect | Oversplit | Wrong | Notes |
|-------|---------|-----------|-------|-------|
| `ckiplab/bert-base-chinese-ws` | 29 | 47 | 2 | **Best choice** |
| `AimanGh/bert-base-chinese-word-segmentation` | 39 | 33 | 6 | More perfect matches but more errors |
| `ckiplab/bert-tiny-chinese-ws` | 4 | 50 | 24 | Too small, not usable |

Note: ALBERT/tiny variants sacrifice too much accuracy for size savings. Stick with bert-base.

**Post-processing with Viterbi:**
- Use lattice-based Viterbi decoding on top of model's P(B) probabilities
- **OOV penalty is the only static rule worth keeping** — penalize multi-char words not in cedict
- Do NOT add other static rules (e.g. "prefer longer words") — any rule that helps some cases will hurt others since it's not context-aware
- The model's probabilities already encode context; fighting them with heuristics is counterproductive

**What didn't work:**
- CKIP NER to augment segmentation — didn't improve results, not worth the complexity
- Static rules to bias toward longer/shorter words — helps and hurts equally, creates whack-a-mole

**Future work:**
- LLM-generated training data may be needed for optimal cedict-aligned segmentation
- Approach: mine corpus sentences, provide cedict word list for each, have LLM segment using only those words contextually
- Cost estimate: ~$5-15 for 100k labeled sentences using GPT-4o-mini
- Would allow fine-tuning `hfl/chinese-roberta-wwm-ext` or similar on cedict-style segmentation

**Test harness:** `ml/scripts/experimental/cws.py`
- 78 validated test cases covering classical poetry, ambiguous boundaries, idioms, named entities
- Scoring: perfect match, oversplit (acceptable), wrong (boundary error)

### WSD (Word sense disambiguation)