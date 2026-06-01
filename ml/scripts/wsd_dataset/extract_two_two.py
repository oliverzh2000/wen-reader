#!/usr/bin/env python3
"""
Extract 2-sense → 2-cluster entries from entries_after_merging.json and write
them as LLM task batches in the same format as build_dataset.py gen-tasks.

These are re-checked with a tighter binary prompt via the LLM coordinator
workflow (INSTRUCTIONS_RECHECK.md).

Output: ml/data/wsd_recheck_llm/llm_tasks/batch_NNNN.json
"""
import json
import shutil
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent))
from wsd_dataset_common import load_translation_cache

_ROOT = Path(__file__).parent.parent.parent  # ml/
ENTRIES_PATH = _ROOT / "data" / "entries_after_merging.json"
TRANSLATION_CACHE = _ROOT / "data" / "translation_cache.json"
OUTPUT_DIR = _ROOT / "data" / "wsd_recheck_llm"
TASKS_DIR = OUTPUT_DIR / "llm_tasks"

BATCH_SIZE = 50


def main() -> None:
    with open(ENTRIES_PATH, encoding="utf-8") as f:
        entries = json.load(f)

    trans_cache = load_translation_cache(TRANSLATION_CACHE)

    # Find 2-sense → 2-cluster entries (no merging happened)
    candidates = []
    for e in entries:
        clusters = e.get("clusters", [])
        if len(clusters) != 2:
            continue
        if sum(len(c["senses"]) for c in clusters) != 2:
            continue

        # Rebuild task in the same format as gen-tasks
        sense_a = clusters[0]["senses"][0]
        sense_b = clusters[1]["senses"][0]

        # Look up Chinese translations from cache
        key_a = f"{e['word']}|{e['pinyin']}|{sense_a}"
        key_b = f"{e['word']}|{e['pinyin']}|{sense_b}"
        zh_a = trans_cache.get(key_a, "")
        zh_b = trans_cache.get(key_b, "")

        candidates.append({
            "w": e["word"],
            "trad": e.get("traditional", ""),
            "py": e["pinyin"],
            "senses": [
                {"i": 0, "en": sense_a, "zh": zh_a},
                {"i": 1, "en": sense_b, "zh": zh_b},
            ],
        })

    # Write task batches
    if TASKS_DIR.exists():
        shutil.rmtree(TASKS_DIR)
    TASKS_DIR.mkdir(parents=True)

    num_batches = 0
    for batch_start in range(0, len(candidates), BATCH_SIZE):
        batch = candidates[batch_start : batch_start + BATCH_SIZE]
        num_batches += 1
        path = TASKS_DIR / f"batch_{num_batches:04d}.json"
        with open(path, "w", encoding="utf-8") as f:
            f.write("[\n")
            for i, entry in enumerate(batch):
                line = json.dumps(entry, ensure_ascii=False, separators=(",", ":"))
                f.write(line)
                if i < len(batch) - 1:
                    f.write(",")
                f.write("\n")
            f.write("]\n")

    # Also create empty results dir
    results_dir = OUTPUT_DIR / "llm_results"
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"Extracted {len(candidates)} two-sense → two-cluster entries")
    print(f"Batch files written: {num_batches} (batch size {BATCH_SIZE})")
    print(f"Output dir: {TASKS_DIR}")


if __name__ == "__main__":
    main()
