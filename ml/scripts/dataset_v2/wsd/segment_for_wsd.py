#!/usr/bin/env python3
"""Segment full ebook corpus using the trained span scorer model for WSD task generation.

This replaces the LLM CWS results as input for WSD task building (step 9).
Using the span scorer ensures WSD training data matches what the model sees in production.

The script:
1. Loads the full sentences.json (all ebooks from step 1)
2. Segments each sentence with the ElectraSmall1 span scorer model (batched encoder)
3. Outputs JSONL in the same format as ebooks_cws_results.jsonl:
   {"text": str, "segments": [str, ...], "source": str, "id": str}

Usage:
    python segment_for_wsd.py                         # default (all sentences)
    python segment_for_wsd.py --max-sentences 1000    # limit for testing
    python segment_for_wsd.py --batch-size 64         # adjust batch size
"""
import json
import os
import sys
import time
from pathlib import Path

import torch

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).parent
DATASET_V2_DIR = SCRIPT_DIR.parent
ML_DIR = DATASET_V2_DIR.parent.parent
SPAN_SCORER_DIR = ML_DIR / "scripts" / "span_scorer"

sys.path.insert(0, str(SPAN_SCORER_DIR))
from span_scorer import (  # noqa: E402
    CedictTrie,
    build_cedict_trie,
    dp_decode,
    MAX_WORD_LEN,
)
from eval_span_scorer import load_span_scorer  # noqa: E402

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_DATA = ML_DIR / "data"
CEDICT_PATH = _DATA / "cedict_ts.u8"

# Model config: ElectraSmall1 (best speed/quality tradeoff for batch segmentation)
MODEL_PATH = ML_DIR / "models" / "cws_span_scorer_electra_small" / "final"
BASE_ENCODER = "hfl/chinese-electra-180g-small-discriminator"

# Default I/O paths
SENTENCES_PATH = _DATA / "dataset_v2" / "sentences.json"
OUTPUT_PATH = _DATA / "dataset_v2" / "wsd_segments.jsonl"

DEFAULT_BATCH_SIZE = 128

DEVICE = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)


# ---------------------------------------------------------------------------
# Batched segmentation
# ---------------------------------------------------------------------------

def _batch_segment(
    sentences: list[str],
    tokenizer,
    model,
    trie: CedictTrie,
    device: str,
    use_log_prob: bool = True,
) -> list[list[str]]:
    """Segment a batch of sentences using batched encoder + per-sentence DP decode.

    Args:
        sentences: list of raw Chinese text strings
        tokenizer: HF tokenizer
        model: SpanScorer (encoder + head)
        trie: CEDICT trie for DP decode
        device: torch device string
        use_log_prob: whether to use log-softmax normalization in DP

    Returns:
        list of segment lists (one per sentence)
    """
    if not sentences:
        return []

    # Tokenize all sentences (char-level, is_split_into_words)
    all_chars = [list(s) for s in sentences]
    encoding = tokenizer(
        all_chars,
        is_split_into_words=True,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True,
    )
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    # Batched encoder forward pass
    with torch.no_grad():
        hidden_states = model.encoder(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state  # (batch, seq_len, hidden_dim)

    # Get max width from head
    max_w = model.head.width_embed.num_embeddings

    # DP decode each sentence individually
    results = []
    for i, sentence in enumerate(sentences):
        h = hidden_states[i]  # (seq_len, hidden_dim)

        def make_score_fn(hidden):
            def score_fn(start: int, end_exclusive: int) -> float | None:
                width = end_exclusive - start
                if width < 1 or width > max_w:
                    return None
                s = torch.tensor([start], device=device)
                e = torch.tensor([end_exclusive - 1], device=device)
                w = torch.tensor([width], device=device)
                score = model.head(hidden, s + 1, e + 1, w)  # +1 for [CLS]
                return score.item()
            return score_fn

        segments = dp_decode(sentence, trie, make_score_fn(h), use_log_prob=use_log_prob)
        results.append(segments)

    return results


def segment_corpus(
    sentences_path: Path,
    output_path: Path,
    model_path: Path = MODEL_PATH,
    base_encoder: str = BASE_ENCODER,
    max_sentences: int | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> None:
    """Segment all sentences from the ebook corpus using the span scorer.

    Args:
        sentences_path: Path to sentences.json (step 1 output)
        output_path: Where to write segmented JSONL
        model_path: Path to span scorer checkpoint
        base_encoder: HF model name for the encoder architecture
        max_sentences: Optional limit for testing
        batch_size: Encoder batch size (default 128)
    """
    # Disable MPS high watermark for large batches
    if DEVICE == "mps":
        os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

    print(f"  Device: {DEVICE}")
    print(f"  Model: {model_path.name} ({base_encoder})")
    print(f"  Batch size: {batch_size}")

    # Load CEDICT trie
    print("  Building CEDICT trie...")
    trie = build_cedict_trie(CEDICT_PATH)
    print(f"  {len(trie.words):,} words in trie")

    # Load model
    print("  Loading span scorer model...")
    tokenizer, model = load_span_scorer(model_path, DEVICE, base_encoder=base_encoder)
    model.eval()
    print("  Model loaded.")

    # Load sentences
    print(f"  Loading sentences from {sentences_path.name}...")
    books = json.loads(sentences_path.read_text(encoding="utf-8"))

    # Flatten all sentences with source/id metadata
    all_items = []
    for book in books:
        source = book["source"]
        for pi, para in enumerate(book["paragraphs"]):
            for si, sent in enumerate(para):
                if sent.strip():
                    all_items.append({
                        "text": sent,
                        "source": source,
                        "id": f"{source}_{pi}_{si}",
                    })

    total = len(all_items)
    if max_sentences is not None:
        all_items = all_items[:max_sentences]
        print(f"  Limited to {max_sentences:,} / {total:,} sentences")
    else:
        print(f"  {total:,} sentences to segment")

    # Segment in batches
    output_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    count = 0
    chars_total = 0

    with open(output_path, "w", encoding="utf-8") as f:
        for batch_start in range(0, len(all_items), batch_size):
            batch_items = all_items[batch_start:batch_start + batch_size]
            batch_texts = [item["text"] for item in batch_items]

            segments_batch = _batch_segment(
                batch_texts, tokenizer, model, trie, device=DEVICE, use_log_prob=True
            )

            for item, segments in zip(batch_items, segments_batch):
                result = {
                    "text": item["text"],
                    "segments": segments,
                    "source": item["source"],
                    "id": item["id"],
                }
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
                count += 1
                chars_total += len(item["text"])

            # Progress
            done = batch_start + len(batch_items)
            if done % (batch_size * 10) == 0 or done == len(all_items):
                elapsed = time.time() - t0
                rate = done / elapsed
                print(f"  [{done:,}/{len(all_items):,}] {rate:.0f} sent/s, {chars_total/elapsed:.0f} char/s")

    elapsed = time.time() - t0
    print(f"  → {count:,} sentences segmented ({chars_total:,} chars) in {elapsed:.1f}s")
    print(f"  → {count / elapsed:.0f} sent/s average")
    print(f"  → Output: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Segment ebook corpus with span scorer for WSD task generation"
    )
    parser.add_argument("--input", type=Path, default=SENTENCES_PATH,
                        help=f"Input sentences JSON (default: {SENTENCES_PATH.name})")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH,
                        help=f"Output segments JSONL (default: {OUTPUT_PATH.name})")
    parser.add_argument("--model", type=Path, default=MODEL_PATH,
                        help=f"Span scorer model checkpoint (default: {MODEL_PATH.name})")
    parser.add_argument("--base-encoder", type=str, default=BASE_ENCODER,
                        help=f"HF base encoder name (default: {BASE_ENCODER})")
    parser.add_argument("--max-sentences", type=int, default=None,
                        help="Max sentences to process (for testing)")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                        help=f"Encoder batch size (default: {DEFAULT_BATCH_SIZE})")
    args = parser.parse_args()

    segment_corpus(
        sentences_path=args.input,
        output_path=args.output,
        model_path=args.model,
        base_encoder=args.base_encoder,
        max_sentences=args.max_sentences,
        batch_size=args.batch_size,
    )
