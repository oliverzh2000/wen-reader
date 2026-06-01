#!/usr/bin/env python3
"""Evaluate exported CoreML models against the same test cases used in eval_cws/eval_wsd.

Runs the actual .mlpackage files (not PyTorch simulations) to verify end-to-end
accuracy after quantization and export.

Usage:
    uv run python scripts/eval_coreml.py
    uv run python scripts/eval_coreml.py --span-only
    uv run python scripts/eval_coreml.py --wsd-only
"""

import sys
import struct
from pathlib import Path

import coremltools as ct
import numpy as np
import torch

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).parent
ML_DIR = SCRIPT_DIR.parent

sys.path.insert(0, str(SCRIPT_DIR / "cws_training"))
sys.path.insert(0, str(SCRIPT_DIR / "span_scorer"))

from cws import score_segmentation  # noqa: E402
from eval_cws import VALIDATED_TEST_CASES as CWS_TEST_CASES  # noqa: E402
from span_scorer import (  # noqa: E402
    SpanScoringHead,
    build_cedict_trie,
    dp_decode,
    MAX_WORD_LEN,
)

sys.path.insert(0, str(SCRIPT_DIR / "wsd_training"))
from eval_wsd import EXAMPLES as WSD_EXAMPLES, load_merged_entries, get_merged_senses, _matches  # noqa: E402

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

COREML_DIR = ML_DIR / "output" / "coreml"
SPAN_ENCODER_PATH = COREML_DIR / "span_scorer_encoder.mlpackage"
SPAN_HEAD_PATH = COREML_DIR / "span_head_weights.bin"
WSD_ENCODER_PATH = COREML_DIR / "wsd_encoder.mlpackage"

CEDICT_PATH = ML_DIR / "data" / "cedict_ts.u8"
ENTRIES_PATH = ML_DIR / "data" / "entries_after_merging.json"
TRANSLATION_CACHE_PATH = ML_DIR / "data" / "translation_cache.json"

# Must match export_coreml.py
SPAN_SEQ_LENGTH = 32
WSD_SEQ_LENGTH = 16

# Span scorer architecture (Electra-small)
SPAN_HIDDEN_DIM = 256
SPAN_WIDTH_EMBED_DIM = 64
SPAN_MLP_HIDDEN = 256
SPAN_MAX_WORD_LEN = 19

# WSD architecture (GTE-small distilled)
WSD_EMBEDDING_DIM = 512

# Model directories (for tokenizer/vocab only)
SPAN_MODEL_DIR = ML_DIR / "models" / "cws_span_scorer_electra_small" / "final"
WSD_MODEL_DIR = ML_DIR / "models" / "wsd_distilled_gte_v2_small" / "20260529_021557" / "final"


# ---------------------------------------------------------------------------
# CoreML Span Scorer
# ---------------------------------------------------------------------------

class CoreMLSpanScorer:
    """Span scorer using exported CoreML encoder + raw binary head weights."""

    def __init__(self):
        from transformers import BertTokenizerFast

        print("Loading CoreML span encoder...")
        self.model = ct.models.MLModel(
            str(SPAN_ENCODER_PATH), compute_units=ct.ComputeUnit.CPU_ONLY
        )
        self.tokenizer = BertTokenizerFast.from_pretrained(str(SPAN_MODEL_DIR))
        self.trie = build_cedict_trie(CEDICT_PATH)
        self._load_head_weights()

    def _load_head_weights(self):
        """Load span head weights from binary file."""
        data = SPAN_HEAD_PATH.read_bytes()
        floats = list(struct.unpack(f"<{len(data) // 4}f", data))

        offset = 0
        hDim = SPAN_HIDDEN_DIM
        wDim = SPAN_WIDTH_EMBED_DIM
        mlpH = SPAN_MLP_HIDDEN
        inputDim = hDim * 2 + wDim

        # width_embed: [19, 64]
        self.width_embed = np.array(
            floats[offset:offset + SPAN_MAX_WORD_LEN * wDim]
        ).reshape(SPAN_MAX_WORD_LEN, wDim)
        offset += SPAN_MAX_WORD_LEN * wDim

        # mlp.0.weight: [256, 576]
        self.w1 = np.array(floats[offset:offset + mlpH * inputDim]).reshape(mlpH, inputDim)
        offset += mlpH * inputDim

        # mlp.0.bias: [256]
        self.b1 = np.array(floats[offset:offset + mlpH])
        offset += mlpH

        # mlp.3.weight: [1, 256]
        self.w2 = np.array(floats[offset:offset + mlpH]).reshape(1, mlpH)
        offset += mlpH

        # mlp.3.bias: [1]
        self.b2 = floats[offset]
        offset += 1

        print(f"  Loaded head weights: {offset} floats ({len(data)} bytes)")

    def _encode(self, sentence: str) -> np.ndarray | None:
        """Run CoreML encoder, return hidden states (seq_len, hidden_dim)."""
        chars = list(sentence)
        encoding = self.tokenizer(
            chars,
            is_split_into_words=True,
            return_tensors="np",
            truncation=True,
            max_length=SPAN_SEQ_LENGTH,
        )
        input_ids = encoding["input_ids"].astype(np.int32)
        seq_len = input_ids.shape[1]

        if seq_len > SPAN_SEQ_LENGTH:
            return None

        # Pad to fixed length
        if seq_len < SPAN_SEQ_LENGTH:
            pad = np.zeros((1, SPAN_SEQ_LENGTH - seq_len), dtype=np.int32)
            input_ids = np.concatenate([input_ids, pad], axis=1)

        output = self.model.predict({"input_ids": input_ids})
        hidden = output["hidden_states"]  # (1, seq_len, hidden_dim)

        # Only take real tokens (not padding)
        return hidden[0, :seq_len, :]

    def _score_span(self, hidden: np.ndarray, start: int, end_inclusive: int, width: int) -> float | None:
        """Score a single span using the head weights."""
        start_tok = start + 1  # +1 for [CLS]
        end_tok = end_inclusive + 1
        seq_len = hidden.shape[0]

        if start_tok >= seq_len or end_tok >= seq_len:
            return None
        if width < 1 or width > SPAN_MAX_WORD_LEN:
            return None

        h_start = hidden[start_tok]
        h_end = hidden[end_tok]
        w_emb = self.width_embed[width - 1]

        inp = np.concatenate([h_start, h_end, w_emb])  # (576,)

        # Layer 1: ReLU(W1 @ inp + b1)
        h = self.w1 @ inp + self.b1
        h = np.maximum(h, 0)

        # Layer 2: W2 @ h + b2
        score = float((self.w2 @ h)[0] + self.b2)
        return score

    def segment(self, sentence: str) -> str:
        """Segment a sentence, return dash-separated result."""
        if not sentence:
            return ""

        # Truncate to max input chars
        max_chars = SPAN_SEQ_LENGTH - 2  # [CLS] + [SEP]
        if len(sentence) > max_chars:
            sentence = sentence[:max_chars]

        hidden = self._encode(sentence)
        if hidden is None:
            return "-".join(sentence)

        def score_fn(start: int, end_exclusive: int) -> float | None:
            width = end_exclusive - start
            return self._score_span(hidden, start, end_exclusive - 1, width)

        segments = dp_decode(sentence, self.trie, score_fn, use_log_prob=False)
        return "-".join(segments)

    def print_scores(self, sentence: str):
        """Print per-span scores for debugging."""
        max_chars = SPAN_SEQ_LENGTH - 2
        if len(sentence) > max_chars:
            sentence = sentence[:max_chars]

        hidden = self._encode(sentence)
        if hidden is None:
            print("       (encoder returned None)")
            return

        chars = list(sentence)
        for i in range(len(chars)):
            words = self.trie.get_words_at(sentence, i)
            single = chars[i]
            if not any(len(w) == 1 for w in words):
                words.append(single)

            for word in words:
                w_len = len(word)
                if w_len > SPAN_MAX_WORD_LEN or i + w_len > len(chars):
                    continue
                score = self._score_span(hidden, i, i + w_len - 1, w_len)
                if score is not None:
                    print(f"       pos {i}: '{word}' (len={w_len}) score={score:.4f}")


# ---------------------------------------------------------------------------
# CoreML WSD Scorer
# ---------------------------------------------------------------------------

class CoreMLWSDScorer:
    """WSD bi-encoder using exported CoreML model."""

    def __init__(self):
        from transformers import AutoTokenizer

        print("Loading CoreML WSD encoder...")
        self.model = ct.models.MLModel(
            str(WSD_ENCODER_PATH), compute_units=ct.ComputeUnit.CPU_ONLY
        )
        self.tokenizer = AutoTokenizer.from_pretrained(str(WSD_MODEL_DIR))

    def encode(self, text: str) -> np.ndarray | None:
        """Encode text, return L2-normalized embedding."""
        encoding = self.tokenizer(
            text, return_tensors="np", truncation=True, max_length=WSD_SEQ_LENGTH
        )
        input_ids = encoding["input_ids"].astype(np.int32)
        seq_len = input_ids.shape[1]

        if seq_len > WSD_SEQ_LENGTH:
            input_ids = input_ids[:, :WSD_SEQ_LENGTH]
            seq_len = WSD_SEQ_LENGTH

        # Pad to fixed length
        if seq_len < WSD_SEQ_LENGTH:
            pad = np.zeros((1, WSD_SEQ_LENGTH - seq_len), dtype=np.int32)
            input_ids = np.concatenate([input_ids, pad], axis=1)

        output = self.model.predict({"input_ids": input_ids})
        embedding = output["embedding"]  # (1, dim)
        return embedding[0]

    def encode_batch(self, texts: list[str]) -> np.ndarray:
        """Encode a batch of texts (one at a time through CoreML)."""
        embeddings = []
        for text in texts:
            emb = self.encode(text)
            if emb is not None:
                embeddings.append(emb)
            else:
                embeddings.append(np.zeros(WSD_EMBEDDING_DIM))
        return np.array(embeddings)


# ---------------------------------------------------------------------------
# CWS Evaluation
# ---------------------------------------------------------------------------

def eval_cws(scorer: CoreMLSpanScorer):
    """Run CWS evaluation using CoreML model."""
    print(f"\n{'=' * 70}")
    print("CWS EVALUATION (CoreML int8)")
    print(f"{'=' * 70}")

    perfect = 0
    oversplit = 0
    wrong = 0
    total = len(CWS_TEST_CASES)

    # Print scores for first test case (for comparison with on-device output)
    first_sentence = CWS_TEST_CASES[0][0]
    print(f"\n  Scores for '{first_sentence}':")
    scorer.print_scores(first_sentence)
    print()

    for i, (sentence, gold) in enumerate(CWS_TEST_CASES, 1):
        result = scorer.segment(sentence)
        score = score_segmentation(result, gold)

        if score == "perfect":
            perfect += 1
        elif score == "oversplit":
            oversplit += 1
        else:
            wrong += 1

        mark = {"perfect": "✓✓", "oversplit": "✓~", "wrong": "✗"}[score]
        if score == "wrong":
            print(f"  [{i:02d}] ✗ {sentence}")
            print(f"       Gold: {gold}")
            print(f"       Got:  {result}")


    passed = perfect + oversplit
    print(f"\n{'=' * 70}")
    print(f"CWS RESULTS (CoreML int8)")
    print(f"{'=' * 70}")
    print(f"  {perfect} perfect + {oversplit} oversplit = {passed}/{total} passed ({wrong} wrong)")
    print()


# ---------------------------------------------------------------------------
# WSD Evaluation
# ---------------------------------------------------------------------------

def eval_wsd(scorer: CoreMLWSDScorer):
    """Run WSD evaluation using CoreML model."""
    print(f"\n{'=' * 70}")
    print("WSD EVALUATION (CoreML int8)")
    print(f"{'=' * 70}")

    print("Loading merged entries...")
    merged = load_merged_entries(ENTRIES_PATH)
    print(f"  {len(merged)} polysemous words loaded")

    # Pre-encode all sense texts (cached per word)
    sense_emb_cache: dict[str, np.ndarray] = {}

    correct = 0
    top3_correct = 0
    mrr_sum = 0.0
    total = 0

    for word, context, expected in WSD_EXAMPLES:
        senses = get_merged_senses(word, merged)
        if not senses:
            continue

        # Encode context
        context_emb = scorer.encode(context)
        if context_emb is None:
            continue

        # Encode senses (cached)
        if word not in sense_emb_cache:
            chinese_senses = [s[1] for s in senses]
            sense_emb_cache[word] = scorer.encode_batch(chinese_senses)

        sense_embs = sense_emb_cache[word]

        # Dot product (embeddings are L2-normalized)
        scores = context_emb @ sense_embs.T
        ranked_indices = scores.argsort()[::-1]

        top1_english = senses[ranked_indices[0]][0]
        is_correct = _matches(expected, top1_english)

        top3_english = [senses[ranked_indices[j]][0] for j in range(min(3, len(ranked_indices)))]
        is_top3 = any(_matches(expected, eng) for eng in top3_english)

        if is_correct:
            correct += 1
        if is_top3:
            top3_correct += 1

        for rank, idx in enumerate(ranked_indices, 1):
            if _matches(expected, senses[idx][0]):
                mrr_sum += 1.0 / rank
                break

        total += 1

        if not is_correct:
            status = "◐" if is_top3 else "✗"
            print(f"  {status} [{word}] {context}")
            print(f"    Expected: {expected}")
            print(f"    Top-1: {top1_english}")

    print(f"\n{'=' * 70}")
    print(f"WSD RESULTS (CoreML int8)")
    print(f"{'=' * 70}")
    print(f"  TOP-1: {correct}/{total} ({100*correct/total:.1f}%)")
    print(f"  TOP-3: {top3_correct}/{total} ({100*top3_correct/total:.1f}%)")
    print(f"  MRR:   {mrr_sum/total:.3f}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse
    p = argparse.ArgumentParser(description="Evaluate exported CoreML models")
    p.add_argument("--span-only", action="store_true", help="Only run CWS eval")
    p.add_argument("--wsd-only", action="store_true", help="Only run WSD eval")
    args = p.parse_args()

    run_span = not args.wsd_only
    run_wsd = not args.span_only

    if run_span:
        if not SPAN_ENCODER_PATH.exists():
            print(f"ERROR: {SPAN_ENCODER_PATH} not found. Run export first.")
            sys.exit(1)
        scorer = CoreMLSpanScorer()
        eval_cws(scorer)

    if run_wsd:
        if not WSD_ENCODER_PATH.exists():
            print(f"ERROR: {WSD_ENCODER_PATH} not found. Run export first.")
            sys.exit(1)
        scorer = CoreMLWSDScorer()
        eval_wsd(scorer)


if __name__ == "__main__":
    main()
