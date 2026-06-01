"""Run ICWB2 merge tasks through multiple models for comparison + cost estimate.

Samples N tasks from the merge task file, runs each model, outputs results.
Also provides dry-run cost estimates for the full task set.

Usage:
  python eval_merge.py --n 50 --models gpt-4o-mini,gpt-4o,claude-sonnet-4.6
  python eval_merge.py --dry-run  # cost estimate only, no API calls
  python eval_merge.py --list-models
"""
import argparse
import asyncio
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from cws.annotate_merge import (
    _get_model_config, _run_async, _format_user_prompt, _MODELS, _dry_run,
    _SYSTEM_PROMPT,
)

_ROOT = Path(__file__).parent.parent.parent  # ml/
_DEFAULT_TASKS = _ROOT / "data" / "dataset_v2" / "icwb2_cws_tasks.jsonl"
_DEFAULT_OUT_DIR = _ROOT / "data" / "dataset_v2" / "eval_merge"

_DEFAULT_MODELS = [
    "gpt-4o-mini",
    "gpt-4o",
    "claude-sonnet-4.6",
    "claude-sonnet-4.6-thinking",
]


def sample_tasks(tasks_path: Path, n: int, seed: int) -> list[dict]:
    """Load all tasks and return a random sample of n (seeded)."""
    all_tasks = [
        json.loads(line)
        for line in tasks_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if n >= len(all_tasks):
        print(f"  Requested {n} but only {len(all_tasks)} tasks available — using all.")
        return all_tasks
    rng = random.Random(seed)
    sample = rng.sample(all_tasks, n)
    sample.sort(key=lambda t: t["id"])
    return sample


def compare_results(out_dir: Path, models: list[str], tasks: list[dict]) -> None:
    """Compare model outputs — agreement rates and disagreements."""
    results_by_model: dict[str, dict[str, list]] = {}

    for model in models:
        p = out_dir / f"eval_{model}.jsonl"
        if not p.exists():
            continue
        model_results = {}
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            model_results[r["id"]] = r["segments"]
        results_by_model[model] = model_results

    if len(results_by_model) < 2:
        return

    print(f"\n{'─'*60}")
    print("Agreement matrix (% identical segmentation):")
    model_names = list(results_by_model.keys())
    # Header
    print(f"  {'':30}", end="")
    for m in model_names:
        print(f"{m[:12]:>13}", end="")
    print()

    for m1 in model_names:
        print(f"  {m1[:30]:30}", end="")
        for m2 in model_names:
            common_ids = set(results_by_model[m1].keys()) & set(results_by_model[m2].keys())
            if not common_ids:
                print(f"{'—':>13}", end="")
                continue
            agree = sum(1 for id_ in common_ids
                        if results_by_model[m1][id_] == results_by_model[m2][id_])
            pct = agree * 100 // len(common_ids)
            print(f"{pct:>12}%", end="")
        print()

    # Show disagreements (first 10)
    print(f"\n  Sample disagreements (first 10):")
    shown = 0
    task_lookup = {t["id"]: t for t in tasks}
    for id_ in sorted(set.intersection(*[set(r.keys()) for r in results_by_model.values()])):
        segs = [results_by_model[m].get(id_) for m in model_names]
        if len(set(str(s) for s in segs)) > 1:
            task = task_lookup.get(id_, {})
            print(f"\n    {id_}: {task.get('segments', '')[:60]}...")
            for m, s in zip(model_names, segs):
                print(f"      {m[:20]:20} → {' '.join(s) if s else '(failed)'}")
            shown += 1
            if shown >= 10:
                break


def run_eval(
    tasks_path: Path,
    out_dir: Path,
    models: list[str],
    n: int,
    seed: int,
    concurrency: int,
    verbose: bool,
    tasks_per_call: int = 5,
) -> None:
    """Sample tasks, run each model, save results."""
    print(f"Sampling {n} tasks (seed={seed}) from {tasks_path.name}...")
    tasks = sample_tasks(tasks_path, n, seed)
    print(f"  {len(tasks)} tasks sampled")

    # Stats on merge complexity
    total_groups = sum(len(t["groups"]) for t in tasks)
    multi_option = sum(1 for t in tasks for g in t["groups"] if len(g) > 1)
    print(f"  Total merge groups: {total_groups} ({multi_option} with multiple options)")

    out_dir.mkdir(parents=True, exist_ok=True)

    # Validate models
    for m in models:
        _get_model_config(m)

    print(f"\nRunning {len(models)} models × {len(tasks)} tasks (concurrency={concurrency}):\n")
    total_start = time.time()

    cost_summary = {}
    for model in models:
        output_path = out_dir / f"eval_{model}.jsonl"
        if output_path.exists():
            output_path.unlink()
        print(f"{'─'*60}")
        print(f"  Model: {model}")
        result = asyncio.run(_run_async(tasks, output_path, model, concurrency,
                                        verbose=verbose, tasks_per_call=tasks_per_call))
        cost_summary[model] = result
        print()

    total_elapsed = time.time() - total_start

    # Results summary
    print(f"{'─'*60}")
    print(f"All done in {total_elapsed:.0f}s.\n")
    print(f"  {'Model':<28} {'Cost':>8} {'Input':>8} {'Cached':>8} {'Output':>8} {'ok/fail':>8} {'Time':>6}")
    total_cost = 0
    for model in models:
        r = cost_summary.get(model, {})
        if not r:
            continue
        total_cost += r.get("cost", 0)
        t = r.get("tokens", {})
        print(f"  {model:<28} ${r.get('cost', 0):>6.4f} "
              f"{t.get('input', 0):>8,} {t.get('input_cached', 0):>8,} {t.get('output', 0):>8,} "
              f"{r.get('ok', 0):>4}/{r.get('failed', 0):<3} "
              f"{r.get('elapsed', 0):>5.0f}s")
    print(f"  {'TOTAL':<28} ${total_cost:>6.4f}")

    # Extrapolate to full dataset using actual token ratios
    all_tasks_count = sum(1 for line in tasks_path.read_text().splitlines() if line.strip())
    scale = all_tasks_count / len(tasks)
    print(f"\n  Full dataset extrapolation ({all_tasks_count:,} tasks, {scale:.0f}x):")
    for model in models:
        r = cost_summary.get(model, {})
        est = r.get("cost", 0) * scale
        print(f"    {model:<28} ~${est:.2f}")

    # Compare
    compare_results(out_dir, models, tasks)

    # Save
    costs_path = out_dir / "costs.json"
    with open(costs_path, "w", encoding="utf-8") as f:
        json.dump(cost_summary, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Multi-model eval for ICWB2 merge tasks")
    parser.add_argument("--tasks", type=Path, default=_DEFAULT_TASKS,
                        help="Path to icwb2_cws_tasks.jsonl")
    parser.add_argument("--out-dir", type=Path, default=_DEFAULT_OUT_DIR,
                        help="Output directory")
    parser.add_argument("--models", type=str, default=None,
                        help="Comma-separated model names")
    parser.add_argument("-n", type=int, default=50,
                        help="Number of tasks to sample (default: 50)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--tasks-per-call", type=int, default=5)
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--dry-run", action="store_true",
                        help="Cost estimate only, no API calls")
    parser.add_argument("--list-models", action="store_true")

    args = parser.parse_args()

    if args.list_models:
        print("Available models:")
        for name, cfg in sorted(_MODELS.items()):
            p = cfg["pricing"]
            print(f"  {name:<25} {cfg['provider']:<10} "
                  f"in=${p['input']:.3f} out=${p['output']:.3f} cached=${p.get('cached_input', p['input']*0.5):.3f}")
        return

    if args.dry_run:
        tasks_path = args.tasks
        if not tasks_path.exists():
            sys.exit(f"Tasks file not found: {tasks_path}")
        tasks = [json.loads(l) for l in tasks_path.read_text().splitlines() if l.strip()]
        print(f"Dry-run cost estimate for {len(tasks):,} tasks:\n")
        models = args.models.split(",") if args.models else list(_MODELS.keys())
        for model in models:
            cfg = _get_model_config(model)
            pricing = cfg["pricing"]
            # Estimate
            system_tokens = len(_SYSTEM_PROMPT)
            total_input = sum(system_tokens + len(_format_user_prompt(t)) for t in tasks)
            cached = system_tokens * (len(tasks) - 1)
            uncached = total_input - cached
            cached_rate = pricing["cached_input"]
            input_cost = uncached / 1_000_000 * pricing["input"] + cached / 1_000_000 * cached_rate
            est_output = len(tasks) * 15  # ~15 tokens per response
            output_cost = est_output / 1_000_000 * pricing["output"]
            total = input_cost + output_cost
            print(f"  {model:<25} ${total:.3f} (in=${input_cost:.3f} out=${output_cost:.3f})")
        return

    models = args.models.split(",") if args.models else _DEFAULT_MODELS
    run_eval(args.tasks, args.out_dir, models, args.n, args.seed,
             args.concurrency, args.verbose, args.tasks_per_call)


if __name__ == "__main__":
    main()
