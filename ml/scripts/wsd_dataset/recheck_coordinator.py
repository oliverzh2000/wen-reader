#!/usr/bin/env python3
"""
LLM Batch Coordinator for 2→2 recheck — tracks progress and validates results.

Same logic as llm_coordinator.py but pointed at wsd_recheck_llm directory.

Usage: python recheck_coordinator.py [N]
"""
import json
import sys
from pathlib import Path

TASKS_DIR = Path(__file__).parent.parent.parent / "data" / "wsd_recheck_llm" / "llm_tasks"
RESULTS_DIR = Path(__file__).parent.parent.parent / "data" / "wsd_recheck_llm" / "llm_results"


def count_task_entries(batch: str) -> int:
    with open(TASKS_DIR / f"{batch}.json", "r", encoding="utf-8") as f:
        return len(json.load(f))


def validate_result(batch: str) -> tuple[int, int, list[str]]:
    path = RESULTS_DIR / f"{batch}.json"
    if not path.exists():
        return 0, count_task_entries(batch), []

    with open(TASKS_DIR / f"{batch}.json", "r", encoding="utf-8") as f:
        tasks = json.load(f)
    try:
        with open(path, "r", encoding="utf-8") as f:
            results = json.load(f)
    except (json.JSONDecodeError, Exception) as e:
        return 0, len(tasks), [f"JSON parse error: {e}"]

    if not isinstance(results, list):
        return 0, len(tasks), ["Result is not a list"]

    errors = []
    valid = 0
    for i, task in enumerate(tasks):
        if i >= len(results):
            errors.append(f"Entry {i} ({task['w']}): missing from results")
            continue

        result = results[i]
        clusters = result.get("clusters", [])
        if not clusters:
            errors.append(f"Entry {i} ({task['w']}): no clusters")
            continue

        n_senses = len(task["senses"])
        expected = set(range(n_senses))
        found = set()
        dupes = []
        for cluster in clusters:
            for idx in cluster:
                if idx in found:
                    dupes.append(idx)
                found.add(idx)

        missing_idx = expected - found
        extra = found - expected

        if dupes:
            errors.append(f"Entry {i} ({task['w']}): duplicate indices {dupes}")
        elif missing_idx:
            errors.append(f"Entry {i} ({task['w']}): missing indices {sorted(missing_idx)}")
        elif extra:
            errors.append(f"Entry {i} ({task['w']}): extra indices {sorted(extra)}")
        else:
            labels = result.get("labels")
            if labels is None:
                errors.append(f"Entry {i} ({task['w']}): missing 'labels' field")
            elif len(labels) != len(clusters):
                errors.append(
                    f"Entry {i} ({task['w']}): labels length {len(labels)} "
                    f"!= clusters length {len(clusters)}"
                )
            else:
                label_ok = True
                for ci, (cluster, label) in enumerate(zip(clusters, labels)):
                    if len(cluster) >= 2:
                        if label is None or not isinstance(label, dict):
                            errors.append(
                                f"Entry {i} ({task['w']}): cluster {ci} has "
                                f"{len(cluster)} senses but label is {label!r} "
                                f"(expected {{\"en\":..., \"zh\":...}})"
                            )
                            label_ok = False
                        elif "en" not in label or "zh" not in label:
                            errors.append(
                                f"Entry {i} ({task['w']}): cluster {ci} label "
                                f"missing 'en' or 'zh' key"
                            )
                            label_ok = False
                    elif len(cluster) == 1 and label is not None:
                        errors.append(
                            f"Entry {i} ({task['w']}): cluster {ci} is single-sense "
                            f"but has a label (should be null)"
                        )
                        label_ok = False
                if label_ok:
                    valid += 1

    if len(results) > len(tasks):
        errors.append(f"Result has {len(results)} entries but task has {len(tasks)}")

    return valid, len(tasks), errors


def get_status(batch: str) -> tuple[str, int, int]:
    path = RESULTS_DIR / f"{batch}.json"
    if not path.exists():
        total = count_task_entries(batch)
        return "missing", 0, total
    valid, total, errors = validate_result(batch)
    if valid >= total:
        return "complete", valid, total
    return ("partial" if valid > 0 else "invalid"), valid, total


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    verbose = "--verbose" in sys.argv or "-v" in sys.argv

    batches = sorted(
        f.stem for f in TASKS_DIR.glob("batch_*.json")
    ) if TASKS_DIR.exists() else []
    if not batches:
        print("No batches. Run extract_two_two.py first.")
        return

    complete, partial, missing, invalid = 0, 0, 0, 0
    valid_entries, total_entries = 0, 0
    for b in batches:
        status, valid, total = get_status(b)
        valid_entries += valid
        total_entries += total
        if status == "complete":
            complete += 1
        elif status == "partial":
            partial += 1
        elif status == "invalid":
            invalid += 1
        else:
            missing += 1

    print(
        f"Batches: {complete}/{len(batches)} complete"
        + (f", {partial} partial" if partial else "")
        + (f", {invalid} invalid" if invalid else "")
        + (f", {missing} missing" if missing else "")
    )
    pct = valid_entries / total_entries * 100 if total_entries else 0
    print(f"Entries: {valid_entries}/{total_entries} ({pct:.1f}%)")

    if verbose:
        for b in batches:
            _, _, errors = validate_result(b)
            if errors:
                print(f"\n{b}:")
                for e in errors[:10]:
                    print(f"  {e}")
                if len(errors) > 10:
                    print(f"  ... and {len(errors) - 10} more")

    incomplete = [b for b in batches if get_status(b)[0] != "complete"]
    if not incomplete:
        print("DONE")
    else:
        print("---")
        for b in incomplete[:n]:
            print(f"ml/data/wsd_recheck_llm/llm_tasks/{b}.json")


if __name__ == "__main__":
    main()
