"""Compare eval result files and output only lines where models disagree.

Usage:
  python compare_eval.py                  # uses hardcoded FILES dict
  python compare_eval.py --dir path/to/eval/  # auto-discover eval_*.jsonl in dir
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "dataset_v2"

FILES = {
    "opus_think": DATA_DIR / "eval_opus_thinking.jsonl",
    "opus": DATA_DIR / "eval_opus.jsonl",
    "opus_med": DATA_DIR / "eval_opus_thinking_med.jsonl",
    "sonnet_think": DATA_DIR / "eval_sonnet_thinking.jsonl",
    "sonnet_med": DATA_DIR / "eval_sonnet_thinking_med.jsonl",
    "deepseek": DATA_DIR / "eval_deepseek_thinking.jsonl",
}


def load_results(path: Path) -> dict[str, list[str]]:
    """Load jsonl file, return {id: segments}."""
    results = {}
    for line in path.read_text().splitlines():
        if line.strip():
            d = json.loads(line)
            results[d["id"]] = d["segments"]
    return results


def discover_eval_files(directory: Path) -> dict[str, Path]:
    """Auto-discover eval_*.jsonl files in a directory."""
    files = {}
    for p in sorted(directory.glob("eval_*.jsonl")):
        # Strip "eval_" prefix and ".jsonl" suffix to get model name
        name = p.stem.removeprefix("eval_")
        files[name] = p
    return files


def main():
    parser = argparse.ArgumentParser(description="Compare eval results across models")
    parser.add_argument("--dir", type=Path, default=None,
                        help="Directory with eval_*.jsonl files (auto-discover mode)")
    parser.add_argument("--out", type=Path, default=None,
                        help="Write diff output to file instead of stdout")
    args = parser.parse_args()

    if args.dir:
        file_map = discover_eval_files(args.dir)
        if not file_map:
            print(f"No eval_*.jsonl files found in {args.dir}", file=sys.stderr)
            sys.exit(1)
        print(f"Found {len(file_map)} eval files in {args.dir}:", file=sys.stderr)
        for name in file_map:
            print(f"  {name}", file=sys.stderr)
    else:
        file_map = FILES

    # Load all
    data = {}
    for name, path in file_map.items():
        if not path.exists():
            print(f"WARNING: {path} not found, skipping", file=sys.stderr)
            continue
        data[name] = load_results(path)

    # Collect all IDs that appear in 2+ models
    id_counts = defaultdict(set)
    for name, results in data.items():
        for task_id in results:
            id_counts[task_id].add(name)

    # Sort IDs
    def sort_key(id_str):
        parts = id_str.split(":")
        try:
            return (int(parts[0]), int(parts[1]))
        except (ValueError, IndexError):
            return (0, 0)  # fallback for non-numeric IDs

    multi_ids = sorted(
        [tid for tid, models in id_counts.items() if len(models) >= 2],
        key=sort_key,
    )

    disagreements = 0
    output = sys.stdout if not args.out else open(args.out, "w", encoding="utf-8")

    # Track per-model agreement stats
    model_names = sorted(data.keys())
    # How often each model is in the majority group
    majority_count = {name: 0 for name in model_names}
    # How often each model is a lone dissenter
    lone_dissent_count = {name: 0 for name in model_names}
    # Pairwise agreement matrix
    pairwise_agree = {a: {b: 0 for b in model_names} for a in model_names}
    pairwise_total = {a: {b: 0 for b in model_names} for a in model_names}

    for task_id in multi_ids:
        # Gather segmentations from all models that have this ID
        segs_by_model = {}
        for name in data:
            if task_id in data[name]:
                segs_by_model[name] = data[name][task_id]

        # Group models by their segmentation (use tuple for hashing)
        groups = defaultdict(list)
        for name, segs in segs_by_model.items():
            groups[tuple(segs)].append(name)

        # Update pairwise stats
        present_models = sorted(segs_by_model.keys())
        for i, a in enumerate(present_models):
            for b in present_models[i+1:]:
                pairwise_total[a][b] += 1
                pairwise_total[b][a] += 1
                if segs_by_model[a] == segs_by_model[b]:
                    pairwise_agree[a][b] += 1
                    pairwise_agree[b][a] += 1

        # If only one group, all agree — skip
        if len(groups) == 1:
            for name in segs_by_model:
                majority_count[name] += 1
            continue

        disagreements += 1

        # Find majority group
        sorted_groups = sorted(groups.values(), key=len, reverse=True)
        majority_set = set(sorted_groups[0])
        for name in segs_by_model:
            if name in majority_set:
                majority_count[name] += 1

        # Track lone dissenters
        for models in groups.values():
            if len(models) == 1:
                lone_dissent_count[models[0]] += 1

        # Reconstruct the text
        text = "".join(list(segs_by_model.values())[0])
        print(f"{'='*70}", file=output)
        print(f"ID: {task_id}  Text: {text}", file=output)
        print(f"Models present: {', '.join(sorted(segs_by_model.keys()))}", file=output)
        print(file=output)
        for segs_tuple, models in sorted(groups.items(), key=lambda x: -len(x[1])):
            seg_str = " / ".join(segs_tuple)
            print(f"  [{', '.join(sorted(models))}]", file=output)
            print(f"    {seg_str}", file=output)
            print(file=output)

    # Summary statistics
    print(f"{'='*70}", file=output)
    if len(multi_ids) > 0:
        print(f"Total disagreements: {disagreements} / {len(multi_ids)} shared IDs "
              f"({disagreements/len(multi_ids)*100:.0f}%)", file=output)
    else:
        print(f"Total disagreements: 0 (no shared IDs across models — need ≥2 models to compare)", file=output)
    print(file=output)

    # Per-model stats
    print("Per-model stats:", file=output)
    print(f"  {'Model':<30} {'With majority':<16} {'Lone dissent':<14} {'Tasks'}", file=output)
    for name in model_names:
        total_tasks = sum(1 for tid in multi_ids if tid in data[name])
        maj_pct = majority_count[name] / total_tasks * 100 if total_tasks else 0
        lone = lone_dissent_count[name]
        print(f"  {name:<30} {majority_count[name]:>4}/{total_tasks} ({maj_pct:.0f}%)   "
              f"{lone:>3} lone", file=output)
    print(file=output)

    # Pairwise agreement
    print("Pairwise agreement:", file=output)
    header = f"  {'':20}" + "".join(f"{n[:12]:>13}" for n in model_names)
    print(header, file=output)
    for a in model_names:
        row = f"  {a:20}"
        for b in model_names:
            if a == b:
                row += f"{'—':>13}"
            elif pairwise_total[a][b] > 0:
                pct = pairwise_agree[a][b] / pairwise_total[a][b] * 100
                row += f"{pct:>12.0f}%"
            else:
                row += f"{'n/a':>13}"
        print(row, file=output)

    if args.out:
        output.close()
        print(f"Diff written to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
