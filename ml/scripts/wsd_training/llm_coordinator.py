#!/usr/bin/env python3
"""
LLM Batch Coordinator - tracks WSD example generation progress.

Usage: python llm_coordinator.py [N]

Prints status, collects missing rows from partial batches, returns next N incomplete batches.
"""
import csv
import sys
from pathlib import Path

TASKS_DIR = Path(__file__).parent.parent.parent / "data" / "wsd_llm" / "llm_tasks"
RESULTS_DIR = Path(__file__).parent.parent.parent / "data" / "wsd_llm" / "llm_results"


def load_tasks(batch: str) -> list[tuple[str, ...]]:
    """Load task rows (word, pinyin, english, chinese, num_needed)."""
    rows = []
    with open(TASKS_DIR / f"{batch}.tsv", 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader, None)  # skip header
        for row in reader:
            if len(row) >= 5:
                rows.append(tuple(row[:5]))
    return rows


def load_result_keys(batch: str) -> set[tuple[str, str, str]]:
    """Load completed (word, pinyin, english) keys from results."""
    path = RESULTS_DIR / f"{batch}.tsv"
    if not path.exists():
        return set()
    keys = set()
    try:
        with open(path, 'r', encoding='utf-8') as f:
            for row in csv.reader(f, delimiter='\t'):
                if len(row) >= 4 and all('#' in s for s in row[3:] if s.strip()):
                    keys.add((row[0], row[1], row[2]))
    except Exception:
        pass
    return keys


def get_status(batch: str) -> tuple[str, int, int]:
    """Returns (status, done, total). Status: complete/partial/missing."""
    tasks = load_tasks(batch)
    results = load_result_keys(batch)
    done, total = len(results), len(tasks)
    if done >= total:
        return "complete", done, total
    return ("partial" if done > 0 else "missing"), done, total


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    
    batches = sorted(f.stem for f in TASKS_DIR.glob("batch_*.tsv")) if TASKS_DIR.exists() else []
    if not batches:
        print("No batches. Run build_dataset.py gen-tasks first.")
        return
    
    # Gather stats
    complete, partial, missing = 0, 0, 0
    done_rows, total_rows = 0, 0
    for b in batches:
        status, done, total = get_status(b)
        done_rows += done
        total_rows += total
        if status == "complete": complete += 1
        elif status == "partial": partial += 1
        else: missing += 1
    
    print(f"Batches: {complete}/{len(batches)} complete" + 
          (f", {partial} partial" if partial else "") +
          (f", {missing} missing" if missing else ""))
    print(f"Rows: {done_rows}/{total_rows} ({done_rows/total_rows*100:.1f}%)" if total_rows else "Rows: 0")
    
    # Return missing batches
    missing = [b for b in batches if get_status(b)[0] == "missing"]
    if not missing:
        print("DONE")
    else:
        print("---")
        for b in missing[:n]:
            print(f"ml/data/wsd_llm/llm_tasks/{b}.tsv")


if __name__ == "__main__":
    main()
