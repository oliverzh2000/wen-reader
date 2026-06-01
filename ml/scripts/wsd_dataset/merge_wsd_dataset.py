#!/usr/bin/env python3
"""
Merge wsd_dataset.tsv using entries_after_merging.json sense clusters.

For each (word, context) group in the TSV:
1. Map each sense (by sense_en) to its merged cluster
2. If the positive sense and any negative sense(s) land in the SAME cluster,
   those negatives become positives (they were under-merged before)
3. Monosemous words and trivial senses are dropped

Output: wsd_dataset_merged.tsv at CLUSTER level — one row per (word, context, cluster_id).
  Columns: word, context, cluster_id, sense_zh, sense_en, label
  sense_zh = all Chinese translations in the cluster joined by ；
  sense_en = cluster_label_en if available, else all English senses joined by "; "

Usage:
    python merge_wsd_dataset.py
    python merge_wsd_dataset.py --dry-run   # stats only, no write
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

_ROOT = Path(__file__).parent.parent.parent
TSV_PATH = _ROOT / "data" / "wsd_dataset.tsv"
ENTRIES_PATH = _ROOT / "data" / "entries_after_merging.json"
OUTPUT_PATH = _ROOT / "data" / "wsd_dataset_merged.tsv"


def build_sense_to_cluster(entries: list[dict]) -> dict[tuple[str, str], dict]:
    """Build lookup: (word, sense_en) -> cluster info including pinyin."""
    lookup = {}
    for e in entries:
        word = e["word"]
        pinyin = e["pinyin"]
        clusters = e.get("clusters", [])
        trivial = set(e.get("trivial_senses", []))
        n_clusters = len(clusters)

        for ci, c in enumerate(clusters):
            for sense in c["senses"]:
                lookup[(word, sense)] = {
                    "cluster_id": ci,
                    "pinyin": pinyin,
                    "n_clusters": n_clusters,
                    "is_trivial": False,
                }

        for ts in trivial:
            lookup[(word, ts)] = {
                "cluster_id": -1,
                "pinyin": pinyin,
                "n_clusters": n_clusters,
                "is_trivial": True,
            }

    return lookup


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    with open(ENTRIES_PATH, encoding="utf-8") as f:
        entries = json.load(f)
    lookup = build_sense_to_cluster(entries)

    with open(TSV_PATH, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        rows = list(reader)

    # Group by (word, context)
    groups = defaultdict(list)
    for r in rows:
        groups[(r["word"], r["context"])].append(r)

    stats = {
        "total_groups": len(groups),
        "total_rows_in": len(rows),
        "groups_monosemous": 0,
        "groups_polysemous": 0,
        "groups_unmapped": 0,
        "labels_flipped_pos": 0,
        "trivial_rows": 0,
    }

    # Collect all sense_zh per (word, pinyin, cluster_id) across the entire dataset
    # so we get the full set of Chinese translations for each cluster.
    # Keyed on (word, pinyin, cluster_id) to avoid collisions between entries
    # with different pinyin (e.g. 落 la4 vs 落 luo4).
    global_cluster_senses_zh: dict[tuple[str, str, int], list[str]] = defaultdict(list)
    global_cluster_senses_en: dict[tuple[str, str, int], list[str]] = defaultdict(list)

    for r in rows:
        key = (r["word"], r["sense_en"])
        info = lookup.get(key)
        if info is None or info["is_trivial"]:
            continue
        ckey = (r["word"], info["pinyin"], info["cluster_id"])
        sz = r["sense_zh"]
        if sz and sz not in global_cluster_senses_zh[ckey]:
            global_cluster_senses_zh[ckey].append(sz)
        se = r["sense_en"]
        if se and se not in global_cluster_senses_en[ckey]:
            global_cluster_senses_en[ckey].append(se)

    # Now process each (word, context) group and output one row per cluster
    # Key: (word, pinyin, context, cluster_id) -> {label}
    out_clusters: dict[tuple, dict] = {}

    for (word, context), group in groups.items():
        mapped = []
        for r in group:
            info = lookup.get((r["word"], r["sense_en"]))
            mapped.append((r, info))

        mapped_infos = [i for _, i in mapped if i is not None]
        if not mapped_infos:
            stats["groups_unmapped"] += 1
            continue

        n_clusters = mapped_infos[0]["n_clusters"]

        # Find gold cluster
        gold_cluster = None
        gold_pinyin = None
        for r, info in mapped:
            if float(r["label"]) == 1.0 and info is not None and not info["is_trivial"]:
                gold_cluster = info["cluster_id"]
                gold_pinyin = info["pinyin"]
                break

        if gold_cluster is None:
            stats["groups_unmapped"] += 1
            continue

        if n_clusters < 2:
            stats["groups_monosemous"] += 1
            continue
        stats["groups_polysemous"] += 1

        for r, info in mapped:
            if info is None or info["is_trivial"]:
                if info and info["is_trivial"]:
                    stats["trivial_rows"] += 1
                continue

            cid = info["cluster_id"]
            pinyin = info["pinyin"]
            old_label = float(r["label"])
            new_label = 1.0 if cid == gold_cluster and pinyin == gold_pinyin else 0.0

            if old_label == 0.0 and new_label == 1.0:
                stats["labels_flipped_pos"] += 1

            okey = (word, pinyin, context, cid)
            if okey not in out_clusters:
                out_clusters[okey] = {"label": new_label}
            else:
                out_clusters[okey]["label"] = max(out_clusters[okey]["label"], new_label)

    # Build final output rows
    out_rows = []
    for (word, pinyin, context, cid), data in out_clusters.items():
        ckey = (word, pinyin, cid)
        senses_zh = global_cluster_senses_zh.get(ckey, [])
        sense_zh = "；".join(senses_zh) if senses_zh else ""
        senses_en = global_cluster_senses_en.get(ckey, [])
        sense_en = "; ".join(senses_en) if senses_en else ""

        out_rows.append({
            "word": word,
            "context": context,
            "cluster_id": cid,
            "sense_zh": sense_zh,
            "sense_en": sense_en,
            "label": data["label"],
        })

    # Print stats
    print(f"Input:  {stats['total_rows_in']:,} rows, {stats['total_groups']:,} groups")
    print(f"Output: {len(out_rows):,} rows (cluster level)")
    print(f"  Polysemous groups: {stats['groups_polysemous']:,}")
    print(f"  Monosemous (dropped): {stats['groups_monosemous']:,}")
    print(f"  Unmapped (dropped): {stats['groups_unmapped']:,}")
    print(f"  Labels flipped 0->1: {stats['labels_flipped_pos']:,}")
    print(f"  Trivial rows (dropped): {stats['trivial_rows']:,}")

    if not args.dry_run:
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = ["word", "context", "cluster_id", "sense_zh", "sense_en", "label"]
        with open(OUTPUT_PATH, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
            writer.writeheader()
            writer.writerows(out_rows)
        print(f"\nWrote {OUTPUT_PATH}")
    else:
        print("\nDry run — no write.")


if __name__ == "__main__":
    main()
