#!/usr/bin/env python3
"""
LLM Batch Coordinator - tracks CWS merge-decision generation progress.

Usage: python llm_coordinator.py [N]

Prints status, returns next N incomplete batches (missing or partial).
"""
import csv
import sys
from pathlib import Path

TASKS_DIR = Path(__file__).parent.parent.parent / "data" / "cws_llm" / "llm_tasks"
RESULTS_DIR = Path(__file__).parent.parent.parent / "data" / "cws_llm" / "llm_results"


def count_task_rows(batch: str) -> int:
    """Count data rows in a task file (excluding header)."""
    count = 0
    with open(TASKS_DIR / f"{batch}.tsv", "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader, None)  # skip header
        for row in reader:
            if row:
                count += 1
    return count


def count_result_rows(batch: str) -> int:
    """Count valid result rows (at least 2 columns, decision is merge/split)."""
    path = RESULTS_DIR / f"{batch}.tsv"
    if not path.exists():
        return 0
    count = 0
    try:
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            next(reader, None)  # skip header
            for row in reader:
                if len(row) >= 2 and row[1].strip() in ("merge", "split"):
                    count += 1
    except Exception:
        pass
    return count


def get_status(batch: str) -> tuple[str, int, int]:
    """Returns (status, done, total). Status: complete/partial/missing."""
    total = count_task_rows(batch)
    done = count_result_rows(batch)
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
        if status == "complete":
            complete += 1
        elif status == "partial":
            partial += 1
        else:
            missing += 1

    print(
        f"Batches: {complete}/{len(batches)} complete"
        + (f", {partial} partial" if partial else "")
        + (f", {missing} missing" if missing else "")
    )
    print(
        f"Rows: {done_rows}/{total_rows} ({done_rows / total_rows * 100:.1f}%)"
        if total_rows
        else "Rows: 0"
    )

    # Return incomplete batches (both missing and partial)
    incomplete = [b for b in batches if get_status(b)[0] != "complete"]
    if not incomplete:
        print("DONE")
    else:
        print("---")
        for b in incomplete[:n]:
            print(f"ml/data/cws_llm/llm_tasks/{b}.tsv")


if __name__ == "__main__":
    main()
