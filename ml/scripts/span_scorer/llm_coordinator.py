#!/usr/bin/env python3
"""
LLM Batch Coordinator — tracks span-scorer segmentation progress.

Usage: python llm_coordinator.py [N]

Prints status, returns next N incomplete batches.
"""
import csv
import sys
from pathlib import Path

TASKS_DIR = Path(__file__).parent.parent.parent / "data" / "span_scorer" / "llm_tasks_icwb2"
RESULTS_DIR = Path(__file__).parent.parent.parent / "data" / "span_scorer" / "llm_results_icwb2"


def count_tasks(batch: str) -> int:
    with open(TASKS_DIR / f"{batch}.tsv", encoding="utf-8") as f:
        return sum(1 for _ in f) - 1  # minus header


def get_status(batch: str) -> tuple[str, int, int]:
    total = count_tasks(batch)
    path = RESULTS_DIR / f"{batch}.txt"
    if not path.exists():
        return "missing", 0, total
    with open(path, encoding="utf-8") as f:
        lines = sum(1 for l in f if l.strip())
    if lines >= total:
        return "complete", lines, total
    return ("partial" if lines > 0 else "missing"), lines, total


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 5

    batches = sorted(
        f.stem for f in TASKS_DIR.glob("batch_*.tsv")
    ) if TASKS_DIR.exists() else []
    if not batches:
        print("No batches. Run build_dataset.py gen-tasks first.")
        return

    complete, partial, missing = 0, 0, 0
    done_lines, total_lines = 0, 0
    for b in batches:
        status, done, total = get_status(b)
        done_lines += done
        total_lines += total
        if status == "complete": complete += 1
        elif status == "partial": partial += 1
        else: missing += 1

    print(f"Batches: {complete}/{len(batches)} complete"
          + (f", {partial} partial" if partial else "")
          + (f", {missing} missing" if missing else ""))
    pct = done_lines / total_lines * 100 if total_lines else 0
    print(f"Lines: {done_lines}/{total_lines} ({pct:.1f}%)")

    incomplete = [b for b in batches if get_status(b)[0] != "complete"]
    if not incomplete:
        print("DONE")
    else:
        print("---")
        for b in incomplete[:n]:
            print(f"ml/data/span_scorer/llm_tasks_icwb2/{b}.tsv")


if __name__ == "__main__":
    main()
