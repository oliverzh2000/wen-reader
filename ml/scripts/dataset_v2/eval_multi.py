"""Run the same sampled tasks through multiple models for comparison.

Randomly samples N tasks (seeded) from the full task file, then runs each
model on the exact same sample. Outputs go to separate files in an eval
directory. Models run sequentially (to avoid cross-provider rate-limit chaos),
but each model run uses full concurrency internally.

Usage:
  python eval_multi.py --tasks ../../data/dataset_v2/ebooks_cws_tasks.jsonl \
      --models deepseek-v4-pro,claude-sonnet-4.6,claude-haiku-4.5 \
      --n 100 --seed 42 --concurrency 8 --verbose

  # Short form with defaults (all non-thinking models, 100 tasks):
  python eval_multi.py
"""
import argparse
import asyncio
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from cws.annotate import _get_model_config, _run_async, run_batch_api, _MODELS

_ROOT = Path(__file__).parent.parent.parent  # ml/
_DEFAULT_TASKS = _ROOT / "data" / "dataset_v2" / "ebooks_cws_tasks.jsonl"
_DEFAULT_OUT_DIR = _ROOT / "data" / "dataset_v2" / "eval"

# Default model set for eval comparison
_DEFAULT_MODELS = [
    # "deepseek-v4-pro",
    # "deepseek-v4-pro-thinking",
    # "claude-haiku-4.5",
    "claude-sonnet-4.6",
    "claude-sonnet-4.6-thinking",
    "claude-opus-4.6",
    "claude-opus-4.6-thinking",
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
    # Sort by id for deterministic ordering in output
    sample.sort(key=lambda t: t["id"])
    return sample


def run_eval(
    tasks_path: Path,
    out_dir: Path,
    models: list[str],
    n: int,
    seed: int,
    concurrency: int,
    verbose: bool,
    tasks_per_call: int = 1,
    batch: bool = False,
) -> None:
    """Sample tasks, run each model, save results."""
    print(f"Sampling {n} tasks (seed={seed}) from {tasks_path.name}...")
    tasks = sample_tasks(tasks_path, n, seed)
    print(f"  {len(tasks)} tasks sampled from {len(set(t['source'] for t in tasks))} sources")

    # Show source distribution
    source_counts: dict[str, int] = {}
    for t in tasks:
        source_counts[t["source"]] = source_counts.get(t["source"], 0) + 1
    for src, count in sorted(source_counts.items(), key=lambda x: -x[1]):
        print(f"    {src}: {count}")

    # Save the sample manifest so we know exactly what was tested
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / f"sample_seed{seed}_n{len(tasks)}.jsonl"
    with open(manifest_path, "w", encoding="utf-8") as f:
        for t in tasks:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    print(f"  Sample manifest → {manifest_path.name}")

    # Validate models
    for m in models:
        _get_model_config(m)  # raises if invalid

    print(f"\nRunning {len(models)} models × {len(tasks)} tasks (concurrency={concurrency}):\n")
    total_start = time.time()

    cost_summary = {}
    for model in models:
        output_path = out_dir / f"eval_{model}.jsonl"
        # Clear previous results for this eval run
        if output_path.exists():
            output_path.unlink()
        print(f"{'─'*60}")
        print(f"  Model: {model}")
        print(f"  Output: {output_path.name}")
        if batch:
            result = run_batch_api(tasks, output_path, model,
                                   tasks_per_call=tasks_per_call, verbose=verbose)
        else:
            result = asyncio.run(_run_async(tasks, output_path, model, concurrency, verbose=verbose,
                                            tasks_per_call=tasks_per_call))
        cost_summary[model] = result
        print()

    total_elapsed = time.time() - total_start
    print(f"{'─'*60}")
    print(f"All done in {total_elapsed:.0f}s. Results in {out_dir}/")
    print(f"Files:")
    for model in models:
        p = out_dir / f"eval_{model}.jsonl"
        count = sum(1 for _ in open(p)) if p.exists() else 0
        print(f"  {p.name}: {count}/{len(tasks)} tasks")

    # Auto-run comparison
    print(f"\n{'─'*60}")
    print("Running comparison...")
    compare_path = out_dir / "compare.txt"

    import subprocess
    script = Path(__file__).parent / "compare_eval.py"
    result = subprocess.run(
        [sys.executable, str(script), "--dir", str(out_dir), "--out", str(compare_path)],
        capture_output=True, text=True,
    )
    if result.stderr:
        print(result.stderr.strip())

    # Print just the summary stats from compare.txt (everything after last "======")
    if compare_path.exists():
        lines = compare_path.read_text().splitlines()
        # Find last separator
        last_sep = 0
        for i, line in enumerate(lines):
            if line.startswith("=" * 60):
                last_sep = i
        print()
        for line in lines[last_sep:]:
            print(line)

    # Cost summary
    print(f"\n{'─'*60}")
    print("Cost summary:")
    print(f"  {'Model':<30} {'Cost':>8} {'Input':>10} {'Cached':>10} {'Output':>10} {'ok/fail':>10} {'Time':>6}")
    total_cost = 0
    for model in models:
        r = cost_summary.get(model)
        if not r:
            continue
        t = r["tokens"]
        total_cost += r["cost"]
        print(f"  {model:<30} ${r['cost']:>6.3f} {t['input']:>10,} {t['input_cached']:>10,} "
              f"{t['output']:>10,} {r['ok']:>4}/{r['failed']:<4} {r['elapsed']:>5.0f}s")
    print(f"  {'TOTAL':<30} ${total_cost:>6.3f}")

    # Save cost summary to file
    costs_path = out_dir / "costs.json"
    with open(costs_path, "w", encoding="utf-8") as f:
        json.dump(cost_summary, f, indent=2)
    print(f"\n  Costs saved → {costs_path.name}")


def main():
    parser = argparse.ArgumentParser(description="Multi-model eval comparison")
    parser.add_argument("--tasks", type=Path, default=_DEFAULT_TASKS,
                        help="Path to cws_tasks.jsonl")
    parser.add_argument("--out-dir", type=Path, default=_DEFAULT_OUT_DIR,
                        help="Output directory for eval results")
    parser.add_argument("--models", type=str, default=None,
                        help="Comma-separated model names (default: all non-thinking)")
    parser.add_argument("-n", type=int, default=100,
                        help="Number of tasks to sample")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for sampling")
    parser.add_argument("--concurrency", type=int, default=8,
                        help="Concurrent API calls per model")
    parser.add_argument("--tasks-per-call", type=int, default=5,
                        help="Number of sentences batched per API call (default: 5)")
    parser.add_argument("--batch", action="store_true",
                        help="Use Anthropic Batch API (50%% discount, requires 'anthropic' provider models)")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--list-models", action="store_true",
                        help="List available models and exit")

    args = parser.parse_args()

    if args.list_models:
        print("Available models:")
        for name, cfg in sorted(_MODELS.items()):
            thinking = cfg.get("thinking", False)
            tag = " (thinking)" if thinking else ""
            print(f"  {name}{tag} — {cfg['provider']} ${cfg['pricing']['input']}/{cfg['pricing']['output']} per 1M tok")
        return

    models = args.models.split(",") if args.models else _DEFAULT_MODELS
    run_eval(args.tasks, args.out_dir, models, args.n, args.seed, args.concurrency,
             args.verbose, tasks_per_call=args.tasks_per_call, batch=args.batch)


if __name__ == "__main__":
    main()
