#!/usr/bin/env python3
"""Check what percentage of polysemous CEDICT words appear in the ebook corpus.

Uses simple substring matching (no CWS needed). Shows coverage at various
sample sizes to help decide how many sentences are needed for WSD annotation.

Usage:
    python -m wsd.coverage_check
    python -m wsd.coverage_check --samples 1000 2000 5000 10000 20000
"""
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

_ROOT = Path(__file__).parent.parent.parent.parent  # ml/
_DATA = _ROOT / "data"

SENTENCES_PATH = _DATA / "dataset_v2" / "sentences.json"
ENTRIES_PATH = _DATA / "entries_after_merging.json"

_CJK_RE = re.compile(r'[\u4e00-\u9fff]')


def load_polysemous_words() -> set[str]:
    """Load all polysemous words (2+ clusters across all pinyins)."""
    from collections import defaultdict
    entries = json.loads(ENTRIES_PATH.read_text(encoding="utf-8"))

    by_word = defaultdict(list)
    for e in entries:
        by_word[e["word"]].append(e)

    poly = set()
    for word, word_entries in by_word.items():
        total_clusters = sum(len(e["clusters"]) for e in word_entries)
        if total_clusters >= 2:
            poly.add(word)
    return poly


def load_all_sentences() -> list[str]:
    """Load all sentences from sentences.json, flattened."""
    books = json.loads(SENTENCES_PATH.read_text(encoding="utf-8"))
    sentences = []
    for book in books:
        for para in book["paragraphs"]:
            for sent in para:
                if sent.strip():
                    sentences.append(sent)
    return sentences


def check_coverage(sentences: list[str], poly_words: set[str]) -> dict:
    """Check which polysemous words appear as substrings in the sentences."""
    # Build corpus text for fast substring search
    corpus = "\n".join(sentences)

    found = set()
    for word in poly_words:
        if word in corpus:
            found.add(word)

    return {
        "total_poly": len(poly_words),
        "found": len(found),
        "pct": 100 * len(found) / len(poly_words) if poly_words else 0,
        "missing_sample": sorted(poly_words - found)[:20],
    }


def count_occurrences(sentences: list[str], poly_words: set[str]) -> dict[str, int]:
    """Count how many sentences each polysemous word appears in."""
    from collections import defaultdict
    counts = defaultdict(int)
    for sent in sentences:
        for word in poly_words:
            if word in sent:
                counts[word] += 1
    return counts


def check_depth_at_samples(
    sentences: list[str],
    poly_words: set[str],
    sample_sizes: list[int],
    thresholds: list[int] = [1, 3, 5, 10, 20],
    seed: int = 42,
) -> None:
    """Check how many words have N+ occurrences at various sample sizes."""
    rng = random.Random(seed)

    print(f"\n{'='*70}")
    print(f"DEPTH ANALYSIS: words with N+ sentence occurrences")
    print(f"{'='*70}")
    print(f"Total polysemous words: {len(poly_words):,}")

    header = f"{'Sentences':<12}" + "".join(f"{'≥'+str(t):<10}" for t in thresholds)
    print(f"\n{header}")
    print(f"{'─'*60}")

    # Full corpus
    counts = count_occurrences(sentences, poly_words)
    row = f"{'ALL':<12}"
    for t in thresholds:
        n = sum(1 for c in counts.values() if c >= t)
        row += f"{n:<10}"
    print(row)

    for n in sorted(sample_sizes):
        if n >= len(sentences):
            continue
        sample = rng.sample(sentences, n)
        counts = count_occurrences(sample, poly_words)
        row = f"{n:<12,}"
        for t in thresholds:
            n_words = sum(1 for c in counts.values() if c >= t)
            row += f"{n_words:<10}"
        print(row)

    # Distribution for full corpus
    print(f"\n{'─'*60}")
    print(f"Full corpus occurrence distribution:")
    full_counts = count_occurrences(sentences, poly_words)
    buckets = [(0, 0), (1, 1), (2, 2), (3, 5), (6, 10), (11, 20), (21, 50), (51, 100), (101, float('inf'))]
    for lo, hi in buckets:
        if hi == float('inf'):
            n = sum(1 for c in full_counts.values() if c >= lo)
            label = f"{lo}+"
        elif lo == hi:
            n = sum(1 for c in full_counts.values() if c == lo)
            label = str(lo)
        else:
            n = sum(1 for c in full_counts.values() if lo <= c <= hi)
            label = f"{lo}-{hi}"
        # Include zero-count words (not found at all)
        if lo == 0 and hi == 0:
            n = len(poly_words) - len(full_counts)
            label = "0"
        print(f"  {label:>6} occurrences: {n:>5} words ({100*n/len(poly_words):.1f}%)")


def check_coverage_at_samples(
    sentences: list[str],
    poly_words: set[str],
    sample_sizes: list[int],
    seed: int = 42,
    n_trials: int = 3,
) -> None:
    """Check coverage at various sample sizes."""
    rng = random.Random(seed)

    print(f"\nTotal sentences: {len(sentences):,}")
    print(f"Total polysemous words: {len(poly_words):,}")
    print(f"{'─'*60}")
    print(f"{'Sentences':<12} {'Coverage':<12} {'Words found':<15} {'Missing'}")
    print(f"{'─'*60}")

    # Full corpus first
    result = check_coverage(sentences, poly_words)
    print(f"{'ALL':<12} {result['pct']:>6.1f}%     {result['found']:>5}/{result['total_poly']:<5}   —")

    for n in sorted(sample_sizes):
        if n >= len(sentences):
            continue
        coverages = []
        for trial in range(n_trials):
            sample = rng.sample(sentences, n)
            r = check_coverage(sample, poly_words)
            coverages.append(r["found"])
        avg = sum(coverages) / len(coverages)
        pct = 100 * avg / len(poly_words)
        missing = len(poly_words) - int(avg)
        print(f"{n:<12,} {pct:>6.1f}%     {int(avg):>5}/{len(poly_words):<5}   {missing:,} missing")

    # Show some missing words from full corpus
    print(f"\n{'─'*60}")
    print(f"Words NOT found in full corpus (first 30):")
    result = check_coverage(sentences, poly_words)
    for w in result["missing_sample"][:30]:
        print(f"  {w}")
    if len(poly_words) - result["found"] > 30:
        print(f"  ... and {len(poly_words) - result['found'] - 30} more")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Check polysemous word coverage in corpus")
    parser.add_argument("--samples", type=int, nargs="+",
                        default=[500, 1000, 2000, 5000, 10000, 15000, 20000, 25000],
                        help="Sample sizes to check")
    args = parser.parse_args()

    print("Loading polysemous words...")
    poly_words = load_polysemous_words()
    print(f"  {len(poly_words):,} polysemous words")

    print("Loading sentences...")
    sentences = load_all_sentences()
    print(f"  {len(sentences):,} sentences")

    check_coverage_at_samples(sentences, poly_words, args.samples)
    check_depth_at_samples(sentences, poly_words, args.samples)


if __name__ == "__main__":
    main()
