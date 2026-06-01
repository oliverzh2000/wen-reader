"""Run the same WSD tasks through multiple models for comparison.

Samples N tasks from the task file, runs each model on the same sample,
compares inter-model agreement. Models run sequentially, each with full
concurrency internally. Tasks are batched per API call for prompt caching.

Usage:
  python wsd/eval_multi.py
  python wsd/eval_multi.py --models claude-sonnet-4.6,deepseek-v4-flash -n 100
  python wsd/eval_multi.py --tasks ../../data/dataset_v2/wsd_tasks.jsonl -n 200 -c 20
"""
import argparse
import asyncio
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from wsd.annotate import _get_model_config, _run_async, _MODELS, _format_user_prompt

_ROOT = Path(__file__).parent.parent.parent.parent  # ml/
_DEFAULT_TASKS = _ROOT / "data" / "dataset_v2" / "wsd_tasks.jsonl"
_DEFAULT_OUT_DIR = _ROOT / "data" / "dataset_v2" / "eval_wsd"

# Default model set
_DEFAULT_MODELS = [
    "claude-sonnet-4.6",
    "claude-sonnet-4.6-thinking",
    "gpt-5-mini",
    "gpt-4o",
    "gpt-4o-mini",
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


def compare_results(out_dir: Path, model_names: list[str], tasks: list[dict]) -> None:
    """Compare model outputs and print disagreement analysis."""
    # Load results: model -> {task_id: [sense_per_word]}
    data: dict[str, dict[str, list[int]]] = {}
    for model in model_names:
        path = out_dir / f"eval_{model}.jsonl"
        if not path.exists():
            continue
        results = {}
        for line in path.read_text().splitlines():
            if line.strip():
                d = json.loads(line)
                # Extract senses list from sentence-level result
                senses = [w["sense"] for w in d["words"]]
                results[d["id"]] = senses
        data[model] = results

    if len(data) < 2:
        print("  Need at least 2 models with results to compare")
        return

    # Build task lookup
    task_lookup = {t["id"]: t for t in tasks}

    # Find all IDs present in 2+ models
    id_counts = defaultdict(set)
    for model, results in data.items():
        for task_id in results:
            id_counts[task_id].add(model)

    shared_ids = sorted([tid for tid, models in id_counts.items() if len(models) >= 2])

    # Track stats per word (not per sentence)
    model_list = sorted(data.keys())
    word_agreements = 0
    word_disagreements = 0
    pairwise_agree = {a: {b: 0 for b in model_list} for a in model_list}
    pairwise_total = {a: {b: 0 for b in model_list} for a in model_list}
    majority_count = {m: 0 for m in model_list}
    total_words_per_model = {m: 0 for m in model_list}

    disagreement_lines = []

    for task_id in shared_ids:
        task = task_lookup.get(task_id)
        if not task:
            continue

        n_words = len(task["words"])
        # For each word position in the sentence
        for wi in range(n_words):
            senses_by_model = {}
            for model in data:
                if task_id in data[model] and wi < len(data[model][task_id]):
                    senses_by_model[model] = data[model][task_id][wi]
                    total_words_per_model[model] = total_words_per_model.get(model, 0) + 1

            if len(senses_by_model) < 2:
                continue

            # Pairwise
            present = sorted(senses_by_model.keys())
            for i, a in enumerate(present):
                for b in present[i+1:]:
                    pairwise_total[a][b] += 1
                    pairwise_total[b][a] += 1
                    if senses_by_model[a] == senses_by_model[b]:
                        pairwise_agree[a][b] += 1
                        pairwise_agree[b][a] += 1

            unique_senses = set(senses_by_model.values())
            if len(unique_senses) == 1:
                word_agreements += 1
                for m in senses_by_model:
                    majority_count[m] += 1
            else:
                word_disagreements += 1
                # Find majority
                sense_counts = defaultdict(list)
                for model, sense in senses_by_model.items():
                    sense_counts[sense].append(model)
                majority_sense = max(sense_counts.keys(), key=lambda s: len(sense_counts[s]))
                for model, sense in senses_by_model.items():
                    if sense == majority_sense:
                        majority_count[model] += 1

                # Record disagreement
                if task:
                    w_info = task["words"][wi]
                    word = w_info["word"]
                    pos = w_info["pos"]
                    sent = task["sentence"]
                    marked = sent[:pos] + f"★{word}★" + sent[pos+len(word):]
                    clusters = w_info["clusters"]

                    lines = [f"{'─'*70}"]
                    lines.append(f"ID: {task_id}  Word[{wi+1}]: {word}")
                    lines.append(f"  {marked[:100]}")
                    for model, sense in sorted(senses_by_model.items()):
                        cluster_info = clusters[sense-1] if sense <= len(clusters) else {}
                        label = "; ".join(cluster_info.get("senses_en", [])[:2]) if cluster_info else "???"
                        pinyin = cluster_info.get("pinyin", "")
                        lines.append(f"    {model:<30} → {sense} ({pinyin}: {label[:40]})")
                    disagreement_lines.append("\n".join(lines))

    # Print summary
    total_words = word_agreements + word_disagreements
    print(f"\n{'='*70}")
    print(f"WSD EVAL COMPARISON (sentence-level)")
    print(f"{'='*70}")
    print(f"\nShared sentences: {len(shared_ids)}")
    print(f"Total word disambiguations: {total_words}")
    print(f"  Agreements:    {word_agreements} ({100*word_agreements/total_words:.1f}%)" if total_words else "")
    print(f"  Disagreements: {word_disagreements} ({100*word_disagreements/total_words:.1f}%)" if total_words else "")

    # Per-model stats
    print(f"\nPer-model majority alignment:")
    print(f"  {'Model':<30} {'With majority':>15} {'Words':>8}")
    for model in model_list:
        total_w = total_words_per_model.get(model, 0)
        pct = 100 * majority_count[model] / total_w if total_w else 0
        print(f"  {model:<30} {majority_count[model]:>4}/{total_w} ({pct:.0f}%)   ")

    # Pairwise agreement
    print(f"\nPairwise agreement:")
    header = f"  {'':25}" + "".join(f"{n[:12]:>13}" for n in model_list)
    print(header)
    for a in model_list:
        row = f"  {a:25}"
        for b in model_list:
            if a == b:
                row += f"{'—':>13}"
            elif pairwise_total[a][b] > 0:
                pct = pairwise_agree[a][b] / pairwise_total[a][b] * 100
                row += f"{pct:>12.0f}%"
            else:
                row += f"{'n/a':>13}"
        print(row)

    # Show disagreements (first 20)
    if disagreement_lines:
        print(f"\n{'='*70}")
        print(f"DISAGREEMENTS (showing first 20 of {len(disagreement_lines)}):")
        for line in disagreement_lines[:20]:
            print(line)

    # Save full comparison
    compare_path = out_dir / "compare_wsd.txt"
    with open(compare_path, "w", encoding="utf-8") as f:
        f.write(f"WSD Eval: {len(shared_ids)} sentences, {total_words} words, "
                f"{word_agreements} agree, {word_disagreements} disagree\n\n")
        for line in disagreement_lines:
            f.write(line + "\n\n")
    print(f"\n  Full comparison → {compare_path.name}")


def run_eval(
    tasks_path: Path,
    out_dir: Path,
    models: list[str],
    n: int,
    seed: int,
    concurrency: int,
    verbose: bool,
    tasks_per_call: int = 10,
) -> None:
    """Sample tasks, run each model, compare results."""
    print(f"Sampling {n} tasks (seed={seed}) from {tasks_path.name}...")
    tasks = sample_tasks(tasks_path, n, seed)
    print(f"  {len(tasks)} sentence tasks sampled")

    # Show word distribution in sample
    from collections import Counter
    word_dist = Counter(w["word"] for t in tasks for w in t["words"])
    total_words = sum(len(t["words"]) for t in tasks)
    print(f"  {total_words} total word disambiguations across {len(tasks)} sentences")
    print(f"  {len(word_dist)} unique words in sample")
    print(f"  Top words: {', '.join(f'{w}({c})' for w, c in word_dist.most_common(10))}")

    # Save manifest
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / f"sample_seed{seed}_n{len(tasks)}.jsonl"
    with open(manifest_path, "w", encoding="utf-8") as f:
        for t in tasks:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")

    # Validate models
    for m in models:
        _get_model_config(m)

    print(f"\nRunning {len(models)} models × {len(tasks)} tasks "
          f"(concurrency={concurrency}, batch={tasks_per_call}/call):\n")
    total_start = time.time()

    cost_summary = {}
    for model in models:
        output_path = out_dir / f"eval_{model}.jsonl"
        if output_path.exists():
            n_existing = sum(1 for l in output_path.read_text().splitlines() if l.strip())
            print(f"{'─'*60}")
            print(f"  Model: {model} — SKIP (already have {n_existing} results in {output_path.name})")
            cost_summary[model] = {"ok": n_existing, "failed": 0, "cost": 0, "elapsed": 0}
            continue
        print(f"{'─'*60}")
        print(f"  Model: {model}")
        result = asyncio.run(_run_async(tasks, output_path, model, concurrency,
                                         verbose=verbose, tasks_per_call=tasks_per_call))
        cost_summary[model] = result
        print()

    total_elapsed = time.time() - total_start
    print(f"{'─'*60}")
    print(f"All done in {total_elapsed:.0f}s.")

    # Compare
    compare_results(out_dir, models, tasks)

    # Cost summary
    print(f"\n{'─'*60}")
    print("Cost summary:")
    print(f"  {'Model':<30} {'Cost':>8} {'ok/fail':>10} {'Time':>6}")
    total_cost = 0
    for model in models:
        r = cost_summary.get(model)
        if not r:
            continue
        total_cost += r["cost"]
        print(f"  {model:<30} ${r['cost']:>6.3f} {r['ok']:>4}/{r['failed']:<4} {r['elapsed']:>5.0f}s")
    print(f"  {'TOTAL':<30} ${total_cost:>6.3f}")


def main():
    parser = argparse.ArgumentParser(description="Multi-model WSD eval comparison")
    parser.add_argument("--tasks", type=Path, default=_DEFAULT_TASKS,
                        help="Path to wsd_tasks.jsonl")
    parser.add_argument("--out-dir", type=Path, default=_DEFAULT_OUT_DIR,
                        help="Output directory for eval results")
    parser.add_argument("--models", type=str, default=None,
                        help="Comma-separated model names")
    parser.add_argument("-n", type=int, default=100,
                        help="Number of tasks to sample")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--concurrency", "-c", type=int, default=20)
    parser.add_argument("--tasks-per-call", type=int, default=10,
                        help="Tasks batched per API call (default: 10)")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--list-models", action="store_true")

    args = parser.parse_args()

    if args.list_models:
        print("Available models:")
        for name, cfg in sorted(_MODELS.items()):
            thinking = cfg.get("thinking", False)
            tag = " (thinking)" if thinking else ""
            print(f"  {name}{tag} — {cfg['provider']} ${cfg['pricing']['input']}/{cfg['pricing']['output']} per 1M tok")
        return

    models = args.models.split(",") if args.models else _DEFAULT_MODELS
    run_eval(args.tasks, args.out_dir, models, args.n, args.seed,
             args.concurrency, args.verbose, tasks_per_call=args.tasks_per_call)


if __name__ == "__main__":
    main()
