#!/usr/bin/env python3
"""
Apply recheck results to entries_after_merging.json.

Reads all result files from wsd_recheck_llm/llm_results/, finds entries
where the LLM decided to merge (clusters == [[0,1]]), and collapses
the corresponding 2-cluster entries in entries_after_merging.json.

Usage:
    python apply_recheck.py              # dry run (preview)
    python apply_recheck.py --apply      # write updated file
"""
import argparse
import json
from pathlib import Path

_ROOT = Path(__file__).parent.parent.parent
ENTRIES_PATH = _ROOT / "data" / "entries_after_merging.json"
RESULTS_DIR = _ROOT / "data" / "wsd_recheck_llm" / "llm_results"


def load_recheck_results() -> dict[tuple[str, str], dict]:
    """Load all recheck results keyed by (word, pinyin)."""
    results = {}
    for path in sorted(RESULTS_DIR.glob("batch_*.json")):
        with open(path, encoding="utf-8") as f:
            batch = json.load(f)
        for r in batch:
            results[(r["w"], r["py"])] = r
    return results


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--apply", action="store_true",
                   help="Write updated entries_after_merging.json")
    args = p.parse_args()

    recheck = load_recheck_results()
    print(f"Loaded {len(recheck)} recheck results")

    # Classify recheck decisions
    to_merge = {}
    to_keep = set()
    for key, r in recheck.items():
        clusters = r.get("clusters", [])
        if len(clusters) == 1 and sorted(clusters[0]) == [0, 1]:
            to_merge[key] = r
        else:
            to_keep.add(key)

    print(f"  Merge: {len(to_merge)}")
    print(f"  Keep:  {len(to_keep)}")

    # Load entries
    with open(ENTRIES_PATH, encoding="utf-8") as f:
        entries = json.load(f)

    # Apply merges
    merged_count = 0
    for e in entries:
        key = (e["word"], e["pinyin"])
        if key not in to_merge:
            continue
        clusters = e.get("clusters", [])
        if len(clusters) != 2 or sum(len(c["senses"]) for c in clusters) != 2:
            continue

        r = to_merge[key]
        labels = r.get("labels", [])
        label = labels[0] if labels else None

        # Merge: combine both senses into one cluster
        all_senses = []
        for c in clusters:
            all_senses.extend(c["senses"])

        new_cluster = {"senses": all_senses}
        if label and isinstance(label, dict):
            if "en" in label:
                new_cluster["en"] = label["en"]
            if "zh" in label:
                new_cluster["zh"] = label["zh"]

        e["clusters"] = [new_cluster]
        merged_count += 1

    poly_after = sum(1 for e in entries if len(e.get("clusters", [])) >= 2)

    print(f"\nApplied {merged_count} merges")
    print(f"Polysemous entries remaining: {poly_after}")

    if not args.apply:
        print("\nDry run. Examples of merged entries:")
        shown = 0
        for key, r in to_merge.items():
            if shown >= 10:
                break
            labels = r.get("labels", [{}])
            label = labels[0] if labels else {}
            en = label.get("en", "") if isinstance(label, dict) else ""
            print(f"  {r['w']} [{r['py']}] -> {en}")
            shown += 1
        print("\nUse --apply to write changes.")
    else:
        with open(ENTRIES_PATH, "w", encoding="utf-8") as f:
            json.dump(entries, f, ensure_ascii=False, indent=2)
        print(f"Wrote {ENTRIES_PATH}")


if __name__ == "__main__":
    main()
