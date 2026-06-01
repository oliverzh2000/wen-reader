#!/usr/bin/env python3
"""
Build WSD sense-merged dictionary (Phase 1 of WSD dataset pipeline).

Usage:
    python build_dataset.py gen-tasks   # Generate LLM task files from CEDICT
    python build_dataset.py assemble    # Build entries_after_merging.json from LLM results
"""
import json
import re
import shutil
import argparse
from collections import defaultdict
from pathlib import Path

from wsd_dataset_common import (
    CEDICT_PATH,
    TRANSLATION_CACHE_PATH,
    is_trivial,
    load_translation_cache,
)

_ROOT = Path(__file__).parent.parent.parent          # ml/
DEFAULT_LLM_DIR = _ROOT / "data" / "wsd_sense_merge_llm"
DEFAULT_OUTPUT = _ROOT / "data" / "entries_after_merging.json"

# CEDICT line: 傳統 传统 [chuan2 tong3] /tradition/traditional/.../
CEDICT_RE = re.compile(r'^(\S+)\s+(\S+)\s+\[([^\]]+)\]\s+/(.+)/')


# ---------------------------------------------------------------------------
# CEDICT parsing (extended — captures traditional + simplified)
# ---------------------------------------------------------------------------

def parse_cedict_full(path: Path = CEDICT_PATH) -> list[dict]:
    """Parse CC-CEDICT into a list of entry dicts.

    Each entry: {traditional, simplified, pinyin, senses: [str, ...]}
    Multiple lines with the same (simplified, pinyin) are merged.
    """
    merged: dict[tuple[str, str], dict] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            m = CEDICT_RE.match(line.strip())
            if not m:
                continue
            trad, simp, pinyin = m.group(1), m.group(2), m.group(3).strip()
            senses = [s.strip() for s in m.group(4).split("/") if s.strip()]
            key = (simp, pinyin)
            if key not in merged:
                merged[key] = {
                    "traditional": trad,
                    "simplified": simp,
                    "pinyin": pinyin,
                    "senses": [],
                }
            merged[key]["senses"].extend(senses)
    return list(merged.values())


# ---------------------------------------------------------------------------
# Sense preprocessing
# ---------------------------------------------------------------------------

def clean_senses_for_merging(senses: list[str]) -> tuple[list[str], list[str]]:
    """Split senses into non-trivial (for LLM merging) and trivial (carried through).

    Non-trivial senses are also deduplicated (case-insensitive).

    Returns (non_trivial, trivial).
    """
    non_trivial: list[str] = []
    trivial: list[str] = []
    seen: set[str] = set()
    for s in senses:
        if is_trivial(s):
            trivial.append(s)
            continue
        key = s.lower().strip()
        if key not in seen:
            seen.add(key)
            non_trivial.append(s)
    return non_trivial, trivial


# ---------------------------------------------------------------------------
# gen-tasks
# ---------------------------------------------------------------------------

def gen_tasks(args: argparse.Namespace) -> None:
    """Generate LLM task batches from CEDICT."""
    entries = parse_cedict_full(args.cedict_path)
    trans_cache = load_translation_cache(args.translation_cache)

    # Preprocess: find candidates for LLM merging (2+ non-trivial senses)
    candidates: list[dict] = []
    stats = {"total": len(entries), "single_sense": 0, "all_trivial": 0}

    for entry in entries:
        non_trivial, trivial = clean_senses_for_merging(entry["senses"])
        if len(non_trivial) < 2:
            if len(non_trivial) == 0:
                stats["all_trivial"] += 1
            else:
                stats["single_sense"] += 1
            continue

        # Build task with sense indices and translations
        task_senses = []
        for i, sense in enumerate(non_trivial):
            cache_key = f"{entry['simplified']}|{entry['pinyin']}|{sense}"
            chinese = trans_cache.get(cache_key, "")
            task_senses.append({
                "i": i,
                "en": sense,
                "zh": chinese,
            })

        candidates.append({
            "w": entry["simplified"],
            "trad": entry["traditional"],
            "py": entry["pinyin"],
            "senses": task_senses,
        })

    # Prepare output directory
    tasks_dir = args.llm_dir / "llm_tasks"
    if tasks_dir.exists():
        shutil.rmtree(tasks_dir)
    tasks_dir.mkdir(parents=True)

    # Write batch JSON files (~batch_size tasks each)
    num_batches = 0
    for batch_start in range(0, len(candidates), args.batch_size):
        batch = candidates[batch_start : batch_start + args.batch_size]
        num_batches += 1
        path = tasks_dir / f"batch_{num_batches:04d}.json"
        with open(path, "w", encoding="utf-8") as f:
            # One entry per line: compact but still readable
            f.write("[\n")
            for i, entry in enumerate(batch):
                line = json.dumps(entry, ensure_ascii=False, separators=(",", ":"))
                f.write(line)
                if i < len(batch) - 1:
                    f.write(",")
                f.write("\n")
            f.write("]\n")

    print(f"Total CEDICT entries:      {stats['total']}")
    print(f"  All senses trivial:      {stats['all_trivial']}")
    print(f"  Single non-trivial sense:{stats['single_sense']}")
    print(f"  LLM candidates (2+):     {len(candidates)}")
    print(f"  Batch files written:     {num_batches}")
    print(f"  Output dir:              {tasks_dir}")


# ---------------------------------------------------------------------------
# assemble
# ---------------------------------------------------------------------------

def load_llm_results(results_dir: Path) -> dict[tuple[str, str], dict]:
    """Load LLM cluster assignments and labels from result batch files.

    Returns {(word, pinyin): {"clusters": [[idx, ...], ...], "labels": [label, ...]}}
    where label is None for single-sense clusters or {"en": ..., "zh": ...} for merged.
    """
    results: dict[tuple[str, str], dict] = {}
    if not results_dir.exists():
        return results

    for path in sorted(results_dir.glob("batch_*.json")):
        with open(path, encoding="utf-8") as f:
            batch = json.load(f)
        for item in batch:
            word = item["w"]
            pinyin = item["py"]
            clusters = item.get("clusters", [])
            labels = item.get("labels")
            if clusters:
                results[(word, pinyin)] = {
                    "clusters": clusters,
                    "labels": labels,
                }

    return results


def assemble(args: argparse.Namespace) -> None:
    """Assemble entries_after_merging.json from LLM results."""
    entries = parse_cedict_full(args.cedict_path)
    trans_cache = load_translation_cache(args.translation_cache)

    # Load LLM results
    results_dir = args.llm_dir / "llm_results"
    llm_clusters = load_llm_results(results_dir)

    output_entries: list[dict] = []
    stats = {
        "total": len(entries),
        "all_trivial": 0,
        "single_sense": 0,
        "llm_merged": 0,
        "llm_missing": 0,
        "single_cluster": 0,
    }

    for entry in entries:
        non_trivial, trivial = clean_senses_for_merging(entry["senses"])

        if len(non_trivial) < 2:
            if len(non_trivial) == 0:
                stats["all_trivial"] += 1
            else:
                stats["single_sense"] += 1
            # Still include in output — just no clusters to disambiguate
            out = {
                "word": entry["simplified"],
                "traditional": entry["traditional"],
                "pinyin": entry["pinyin"],
                "clusters": [],
                "trivial_senses": trivial,
            }
            if len(non_trivial) == 1:
                out["clusters"] = [{"senses": non_trivial}]
            output_entries.append(out)
            continue

        key = (entry["simplified"], entry["pinyin"])
        llm_result = llm_clusters.get(key)

        if llm_result is None:
            # LLM result missing — fall back to each sense as its own cluster
            stats["llm_missing"] += 1
            clusters = [{"senses": [s]} for s in non_trivial]
        else:
            # Apply LLM cluster assignments
            cluster_assignments = llm_result["clusters"]
            labels = llm_result.get("labels") or [None] * len(cluster_assignments)
            clusters = []
            for ci, idx_list in enumerate(cluster_assignments):
                cluster_senses = []
                for idx in idx_list:
                    if 0 <= idx < len(non_trivial):
                        cluster_senses.append(non_trivial[idx])
                if cluster_senses:
                    cluster_obj = {"senses": cluster_senses}
                    # Use LLM-generated label for multi-sense clusters
                    label = labels[ci] if ci < len(labels) else None
                    if label and isinstance(label, dict) and len(cluster_senses) >= 2:
                        if "en" in label:
                            cluster_obj["en"] = label["en"]
                        if "zh" in label:
                            cluster_obj["zh"] = label["zh"]
                    clusters.append(cluster_obj)
            stats["llm_merged"] += 1

        if len(clusters) < 2:
            stats["single_cluster"] += 1

        output_entries.append({
            "word": entry["simplified"],
            "traditional": entry["traditional"],
            "pinyin": entry["pinyin"],
            "clusters": clusters,
            "trivial_senses": trivial,
        })

    # Write output
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output_entries, f, ensure_ascii=False, indent=2)

    polysemous = sum(1 for e in output_entries if len(e["clusters"]) >= 2)

    print(f"\nAssembly complete:")
    print(f"Total CEDICT entries:      {stats['total']}")
    print(f"  All senses trivial:      {stats['all_trivial']}")
    print(f"  Single non-trivial sense:{stats['single_sense']}")
    print(f"  LLM merged:              {stats['llm_merged']}")
    print(f"  LLM missing (fallback):  {stats['llm_missing']}")
    print(f"  Single cluster after merge:{stats['single_cluster']}")
    print(f"  Polysemous (2+ clusters):{polysemous}")
    print(f"Output entries:            {len(output_entries)}")
    print(f"Wrote {args.output}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description="Build WSD sense-merged dictionary (Phase 1)"
    )
    p.add_argument("--cedict-path", type=Path, default=CEDICT_PATH)
    p.add_argument("--translation-cache", type=Path,
                   default=TRANSLATION_CACHE_PATH)
    p.add_argument("--llm-dir", type=Path, default=DEFAULT_LLM_DIR)
    p.add_argument("--batch-size", type=int, default=50,
                   help="Entries per LLM task batch file")

    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("gen-tasks", help="Generate LLM task files from CEDICT")

    asm = sub.add_parser("assemble",
                         help="Build entries_after_merging.json from LLM results")
    asm.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)

    args = p.parse_args()
    if args.cmd == "gen-tasks":
        gen_tasks(args)
    else:
        assemble(args)


if __name__ == "__main__":
    main()
