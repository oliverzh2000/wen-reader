#!/usr/bin/env python3
"""
Phase 2: Scan corpora and build CWS-verified sentence index for polysemous words.

Reads entries_after_merging.json (Phase 1 output), scans Wikipedia zh +
OpenSubtitles zh, and writes sentence_index.json.

Usage:
    python build_sentence_index.py
    python build_sentence_index.py --cws-batch-size 128
    python build_sentence_index.py --resume  # resume from checkpoint
"""
import argparse
import json
import time
from pathlib import Path

from wsd_dataset_common import (
    build_sentence_index,
    detect_device,
    load_cws_model,
)

_ROOT = Path(__file__).parent.parent.parent  # ml/
DEFAULT_ENTRIES = _ROOT / "data" / "entries_after_merging.json"
DEFAULT_OUTPUT = _ROOT / "data" / "sentence_index.json"


def load_polysemous_words(entries_path: Path) -> set[str]:
    """Extract unique words from polysemous entries (2+ clusters)."""
    with open(entries_path, encoding="utf-8") as f:
        entries = json.load(f)
    words = set()
    for e in entries:
        if len(e.get("clusters", [])) >= 2:
            words.add(e["word"])
    return words


def main() -> None:
    p = argparse.ArgumentParser(description="Phase 2: Build sentence index")
    p.add_argument("--entries", type=Path, default=DEFAULT_ENTRIES,
                   help="Path to entries_after_merging.json")
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--cws-batch-size", type=int, default=64)
    p.add_argument("--max-per-word", type=int, default=100,
                   help="Max sentences per word before saturation")
    p.add_argument("--resume", action="store_true",
                   help="Resume from existing output (skip words already indexed)")
    args = p.parse_args()

    # Load target words
    words = load_polysemous_words(args.entries)
    print(f"Polysemous words to index: {len(words)}")

    # Resume support: load existing index and skip words that already have sentences
    existing_index = {}
    if args.resume and args.output.exists():
        with open(args.output, encoding="utf-8") as f:
            existing_index = json.load(f)
        already_done = words & set(existing_index.keys())
        words -= already_done
        print(f"Resuming: {len(already_done)} words already indexed, "
              f"{len(words)} remaining")
        if not words:
            print("Nothing to do.")
            return

    # Load span scorer model
    device = detect_device()
    print(f"Device: {device}")
    cws_tok, cws_mod, cws_trie = load_cws_model(device)

    # Scan corpora
    t0 = time.time()
    new_index = build_sentence_index(
        words, cws_tok, cws_mod, cws_trie, device,
        cws_batch_size=args.cws_batch_size,
        max_per_word=args.max_per_word,
        checkpoint_path=args.output,
    )
    elapsed = time.time() - t0

    # Merge with existing if resuming
    if existing_index:
        existing_index.update(new_index)
        final_index = existing_index
    else:
        final_index = new_index

    # Write output (one line per word for readability)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write("{\n")
        items = list(final_index.items())
        for i, (word, sents) in enumerate(items):
            line = json.dumps(word, ensure_ascii=False) + ": " + json.dumps(sents, ensure_ascii=False)
            f.write(line)
            if i < len(items) - 1:
                f.write(",")
            f.write("\n")
        f.write("}\n")

    total_sents = sum(len(v) for v in final_index.values())
    print(f"\nDone in {elapsed:.0f}s")
    print(f"Words with sentences: {len(final_index)}")
    print(f"Total sentences: {total_sents}")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
