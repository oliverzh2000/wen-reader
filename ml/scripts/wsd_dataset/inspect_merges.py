#!/usr/bin/env python3
"""
Sample entries where LLM kept senses separate (potential under-merging).

Writes readable output to ml/data/merge_inspection.txt for manual review.
"""
import json
import random
from pathlib import Path

_ROOT = Path(__file__).parent.parent.parent
ENTRIES_PATH = _ROOT / "data" / "entries_after_merging.json"
OUTPUT_PATH = _ROOT / "data" / "merge_inspection.txt"

random.seed(99)


def fmt_cluster(cluster: dict, idx: int) -> str:
    lines = []
    label_parts = []
    if "en" in cluster:
        label_parts.append(cluster["en"])
    if "zh" in cluster:
        label_parts.append(cluster["zh"])
    label = f" — {' / '.join(label_parts)}" if label_parts else ""
    lines.append(f"    Cluster {idx}{label}")
    for s in cluster["senses"]:
        lines.append(f"      • {s}")
    return "\n".join(lines)


def fmt_entry(e: dict) -> str:
    lines = []
    word = e["word"]
    pinyin = e["pinyin"]
    trad = e.get("traditional", "")
    n_senses = sum(len(c["senses"]) for c in e["clusters"])
    n_clusters = len(e["clusters"])
    trivial = e.get("trivial_senses", [])

    lines.append(f"  {word} ({trad}) [{pinyin}]  —  {n_senses} senses → {n_clusters} clusters")
    for i, c in enumerate(e["clusters"]):
        lines.append(fmt_cluster(c, i))
    if trivial:
        lines.append(f"    Trivial: {'; '.join(trivial)}")
    return "\n".join(lines)


def main():
    with open(ENTRIES_PATH, encoding="utf-8") as f:
        entries = json.load(f)

    # Category 1: 2 non-trivial senses → stayed 2 clusters
    two_stayed_two = [
        e for e in entries
        if len(e["clusters"]) == 2
        and sum(len(c["senses"]) for c in e["clusters"]) == 2
    ]

    # Category 2: 3 non-trivial senses → stayed 3 clusters (no merging at all)
    three_stayed_three = [
        e for e in entries
        if len(e["clusters"]) == 3
        and sum(len(c["senses"]) for c in e["clusters"]) == 3
    ]

    # Category 3: 3 non-trivial senses → 2 clusters (partial merge)
    three_to_two = [
        e for e in entries
        if len(e["clusters"]) == 2
        and sum(len(c["senses"]) for c in e["clusters"]) == 3
    ]

    # Category 4: 4 non-trivial senses → 3 or 4 clusters (minimal merging)
    four_to_three_or_four = [
        e for e in entries
        if sum(len(c["senses"]) for c in e["clusters"]) == 4
        and len(e["clusters"]) >= 3
    ]

    # Category 5: 5+ non-trivial senses → still 4+ clusters
    five_plus_high = [
        e for e in entries
        if sum(len(c["senses"]) for c in e["clusters"]) >= 5
        and len(e["clusters"]) >= 4
    ]

    N = 20  # samples per category

    out = []
    out.append("=" * 70)
    out.append("MERGE INSPECTION — Potential Under-Merging Cases")
    out.append("=" * 70)

    categories = [
        ("2 senses → 2 clusters (no merging)", two_stayed_two, 40),
        ("3 senses → 3 clusters (no merging)", three_stayed_three, 25),
        ("3 senses → 2 clusters (partial merge)", three_to_two, 25),
        ("4 senses → 3–4 clusters (minimal merging)", four_to_three_or_four, 20),
        ("5+ senses → 4+ clusters (high residual)", five_plus_high, 15),
    ]

    for title, pool, n in categories:
        out.append(f"\n{'─' * 70}")
        out.append(f"  {title}  ({len(pool)} total, showing {min(n, len(pool))})")
        out.append(f"{'─' * 70}")
        sample = random.sample(pool, min(n, len(pool)))
        for e in sample:
            out.append("")
            out.append(fmt_entry(e))

    text = "\n".join(out) + "\n"
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Wrote {OUTPUT_PATH} ({len(text)} chars)")
    for title, pool, n in categories:
        print(f"  {title}: {len(pool)} total")


if __name__ == "__main__":
    main()
