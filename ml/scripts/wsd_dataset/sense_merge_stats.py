#!/usr/bin/env python3
"""
Compare original CEDICT senses with post-merge senses.

Reports three stages:
  1. Raw CEDICT (all senses)
  2. After trivial-sense removal (variant-of, abbr-for, see-also, etc.)
  3. After trivial removal + LLM merging

Usage:
    python scripts/wsd_dataset/sense_merge_stats.py
    python scripts/wsd_dataset/sense_merge_stats.py --entries path/to/entries.json
"""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent))
from wsd_dataset_common import is_trivial

_ROOT = Path(__file__).parent.parent.parent  # ml/
DEFAULT_ENTRIES = _ROOT / "data" / "entries_after_merging.json"


def load_entries(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def fmt_pct(n: int, total: int) -> str:
    return f"{n:,} ({n / total * 100:.1f}%)" if total else f"{n:,}"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--entries", type=Path, default=DEFAULT_ENTRIES)
    args = p.parse_args()

    entries = load_entries(args.entries)
    total = len(entries)

    # ── Per-word aggregation (across all pinyin readings) ──
    # Group entries by simplified word to get word-level stats
    by_word: dict[str, list[dict]] = defaultdict(list)
    for e in entries:
        by_word[e["word"]].append(e)

    # ── Stage counts per (word, pinyin) entry ──
    # raw_senses: all senses in original CEDICT for this entry
    # nontrivial_senses: raw minus trivial
    # merged_clusters: final cluster count after LLM

    raw_sense_counts = []          # per entry: total raw sense count
    nontrivial_sense_counts = []   # per entry: non-trivial sense count
    merged_cluster_counts = []     # per entry: cluster count after LLM

    for e in entries:
        trivial = e.get("trivial_senses", [])
        clusters = e.get("clusters", [])

        # Reconstruct raw sense count: non-trivial senses + trivial senses
        n_nontrivial = sum(len(c["senses"]) for c in clusters)
        n_trivial = len(trivial)
        n_raw = n_nontrivial + n_trivial

        raw_sense_counts.append(n_raw)
        nontrivial_sense_counts.append(n_nontrivial)
        merged_cluster_counts.append(len(clusters))

    # ── Polysemous counts at each stage ──
    raw_poly = sum(1 for n in raw_sense_counts if n >= 2)
    nontrivial_poly = sum(1 for n in nontrivial_sense_counts if n >= 2)
    merged_poly = sum(1 for n in merged_cluster_counts if n >= 2)

    print("=" * 70)
    print("WSD Sense Merge Statistics")
    print("=" * 70)

    print(f"\nTotal CEDICT entries (word, pinyin): {total:,}")
    print(f"Unique simplified words:            {len(by_word):,}")

    print(f"\n{'Stage':<40} {'Polysemous':>12} {'% of total':>12}")
    print("-" * 66)
    print(f"{'Raw CEDICT (all senses)':<40} {raw_poly:>12,} {raw_poly/total*100:>11.1f}%")
    print(f"{'After trivial removal':<40} {nontrivial_poly:>12,} {nontrivial_poly/total*100:>11.1f}%")
    print(f"{'After trivial + LLM merge':<40} {merged_poly:>12,} {merged_poly/total*100:>11.1f}%")

    # ── Trivial sense stats ──
    entries_with_trivial = sum(1 for e in entries if e.get("trivial_senses"))
    total_trivial = sum(len(e.get("trivial_senses", [])) for e in entries)
    total_raw = sum(raw_sense_counts)
    print(f"\nTrivial senses removed:  {total_trivial:,} / {total_raw:,} total senses "
          f"({total_trivial/total_raw*100:.1f}%)")
    print(f"Entries with ≥1 trivial: {entries_with_trivial:,} / {total:,} "
          f"({entries_with_trivial/total*100:.1f}%)")

    # ── Reduction by original polysemous count ──
    # For entries that were polysemous in raw CEDICT, show average reduction
    # at each stage, bucketed by raw sense count.
    print(f"\n{'─' * 70}")
    print("Reduction by raw sense count (entries with ≥2 raw senses)")
    print(f"{'─' * 70}")
    print(f"{'Raw senses':>10} {'Entries':>8} {'Avg after':>12} {'Avg after':>12} "
          f"{'Reduction':>12} {'Reduction':>12}")
    print(f"{'':>10} {'':>8} {'trivial rm':>12} {'LLM merge':>12} "
          f"{'trivial→raw':>12} {'merge→raw':>12}")
    print("-" * 70)

    # Bucket by raw sense count
    buckets: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
    for raw, nt, mc in zip(raw_sense_counts, nontrivial_sense_counts, merged_cluster_counts):
        if raw >= 2:
            buckets[raw].append((raw, nt, mc))

    max_display = 15
    for raw_n in sorted(buckets.keys()):
        if raw_n > max_display:
            break
        items = buckets[raw_n]
        count = len(items)
        avg_nt = sum(nt for _, nt, _ in items) / count
        avg_mc = sum(mc for _, _, mc in items) / count
        red_trivial = (1 - avg_nt / raw_n) * 100
        red_merge = (1 - avg_mc / raw_n) * 100
        print(f"{raw_n:>10} {count:>8,} {avg_nt:>12.2f} {avg_mc:>12.2f} "
              f"{red_trivial:>11.1f}% {red_merge:>11.1f}%")

    remaining = {k: v for k, v in buckets.items() if k > max_display}
    if remaining:
        all_items = [item for items in remaining.values() for item in items]
        r_count = len(all_items)
        r_avg_raw = sum(r for r, _, _ in all_items) / r_count
        r_avg_nt = sum(nt for _, nt, _ in all_items) / r_count
        r_avg_mc = sum(mc for _, _, mc in all_items) / r_count
        r_red_trivial = (1 - r_avg_nt / r_avg_raw) * 100
        r_red_merge = (1 - r_avg_mc / r_avg_raw) * 100
        print(f"{f'{max_display+1}+':>10} {r_count:>8,} {r_avg_nt:>12.2f} {r_avg_mc:>12.2f} "
              f"{r_red_trivial:>11.1f}% {r_red_merge:>11.1f}%")

    # ── Word-level polysemy (across all pinyin readings) ──
    print(f"\n{'─' * 70}")
    print("Word-level polysemy (aggregated across all pinyin readings)")
    print(f"{'─' * 70}")

    word_raw_poly = 0
    word_merged_poly = 0
    for word, word_entries in by_word.items():
        raw_total = sum(
            sum(len(c["senses"]) for c in e["clusters"]) + len(e.get("trivial_senses", []))
            for e in word_entries
        )
        merged_total = sum(len(e["clusters"]) for e in word_entries)
        if raw_total >= 2:
            word_raw_poly += 1
        if merged_total >= 2:
            word_merged_poly += 1

    print(f"Words with ≥2 raw senses (across readings):    {word_raw_poly:,}")
    print(f"Words with ≥2 merged clusters (across readings): {word_merged_poly:,}")

    # ── LLM merge outcome distribution ──
    # For entries that went through LLM (had ≥2 non-trivial senses),
    # what happened?
    print(f"\n{'─' * 70}")
    print("LLM merge outcomes (entries with ≥2 non-trivial senses)")
    print(f"{'─' * 70}")

    llm_candidates = [(nt, mc) for nt, mc in zip(nontrivial_sense_counts, merged_cluster_counts)
                       if nt >= 2]
    llm_total = len(llm_candidates)
    collapsed_to_1 = sum(1 for nt, mc in llm_candidates if mc == 1)
    reduced = sum(1 for nt, mc in llm_candidates if 1 < mc < nt)
    unchanged = sum(1 for nt, mc in llm_candidates if mc == nt)

    print(f"Total entries sent to LLM:  {llm_total:,}")
    print(f"  Collapsed to 1 cluster:   {fmt_pct(collapsed_to_1, llm_total)}")
    print(f"  Reduced (but still 2+):   {fmt_pct(reduced, llm_total)}")
    print(f"  Unchanged:                {fmt_pct(unchanged, llm_total)}")

    # Average cluster reduction for those that remained polysemous
    still_poly = [(nt, mc) for nt, mc in llm_candidates if mc >= 2]
    if still_poly:
        avg_before = sum(nt for nt, _ in still_poly) / len(still_poly)
        avg_after = sum(mc for _, mc in still_poly) / len(still_poly)
        print(f"\n  Among still-polysemous ({len(still_poly):,} entries):")
        print(f"    Avg senses before merge: {avg_before:.2f}")
        print(f"    Avg clusters after merge: {avg_after:.2f}")
        print(f"    Avg reduction:           {(1 - avg_after/avg_before)*100:.1f}%")

    # ── Cluster size distribution ──
    print(f"\n{'─' * 70}")
    print("Cluster size distribution (all entries)")
    print(f"{'─' * 70}")

    cluster_sizes = Counter()
    for e in entries:
        for c in e.get("clusters", []):
            cluster_sizes[len(c["senses"])] += 1

    total_clusters = sum(cluster_sizes.values())
    for size in sorted(cluster_sizes.keys()):
        count = cluster_sizes[size]
        print(f"  {size} sense(s): {count:>8,}  ({count/total_clusters*100:.1f}%)")

    print(f"\n  Total clusters: {total_clusters:,}")
    multi = sum(v for k, v in cluster_sizes.items() if k >= 2)
    print(f"  Multi-sense clusters (merged): {multi:,} ({multi/total_clusters*100:.1f}%)")


if __name__ == "__main__":
    main()
