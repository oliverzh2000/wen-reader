"""Extract ICWB2 data for the CWS pipeline (merge-only approach).

Key insight: ICWB2 human annotations have trustworthy *boundaries*. They may
over-split relative to our "prefer longest CEDICT word" standard, but they
never split incorrectly. So the only decision is whether to *merge* adjacent
gold segments into a longer CEDICT word.

Two outputs:
  1. "Free" sentences — no valid merge candidates exist (gold is already
     maximally merged per CEDICT), goes directly to training.
  2. Merge-decision tasks — positions where 2+ consecutive gold segments
     concatenate to a CEDICT word. LLM decides: merge or keep split.

Task format is compatible with the existing annotate.py pipeline:
  - candidates[pos] = list of merge options (longer CEDICT words)
  - prefilled[pos] = pre-resolved positions (greedy-confirmed words)
  - The "keep split" option is implicit (LLM omits the position from picks)
"""
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from opencc import OpenCC

from cws.cedict_lookup import load_cedict_words, greedy_segment
from filters import passes_filter

_ROOT = Path(__file__).parent.parent.parent.parent  # ml/
_ICWB2 = _ROOT / "data" / "icwb2-data" / "training"

_T2S = OpenCC("t2s")

CORPORA = [
    ("MSR", _ICWB2 / "msr_training.utf8", False),
    ("PKU", _ICWB2 / "pku_training.utf8", False),
    ("AS", _ICWB2 / "as_training.utf8", True),
    ("CityU", _ICWB2 / "cityu_training.utf8", True),
]


def _load_gold_sentences(path: Path, to_simplified: bool) -> list[list[str]]:
    """Load gold-segmented sentences from ICWB2 training file."""
    sentences = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            words = line.split()
            if not words:
                continue
            if to_simplified:
                words = [_T2S.convert(w) for w in words]
            sentences.append(words)
    return sentences


def _find_merge_candidates(
    gold_segments: list[str], word_set: set[str], max_len: int
) -> list[tuple[int, int, str]]:
    """Find all valid merges: consecutive gold segments that form a CEDICT word.

    Returns list of (start_seg_index, num_segments_consumed, merged_word).
    Only merges of 2+ segments that produce a multi-char CEDICT entry.
    """
    merges = []
    n = len(gold_segments)

    for i in range(n):
        # Try concatenating segments[i:j] for j = i+2, i+3, ...
        concat = gold_segments[i]
        for j in range(i + 1, n):
            concat += gold_segments[j]
            if len(concat) > max_len:
                break
            if len(concat) >= 2 and concat in word_set:
                merges.append((i, j - i + 1, concat))

    return merges


def _group_overlapping_merges(
    merges: list[tuple[int, int, str]],
) -> list[list[str]]:
    """Group merges that share segment indices into mutually exclusive groups.

    Each group is a list of merged words (longest first) that conflict with
    each other. The LLM picks at most one from each group.
    """
    if not merges:
        return []

    # Sort by start index, then by length descending
    merges_sorted = sorted(merges, key=lambda m: (m[0], -m[1]))

    groups: list[list[tuple[int, int, str]]] = []
    for merge in merges_sorted:
        start, span, word = merge
        end = start + span - 1  # inclusive end segment index

        # Find if this merge overlaps with any existing group
        placed = False
        for group in groups:
            for g_start, g_span, _ in group:
                g_end = g_start + g_span - 1
                if start <= g_end and end >= g_start:
                    # Overlaps — add to this group
                    group.append(merge)
                    placed = True
                    break
            if placed:
                break

        if not placed:
            groups.append([merge])

    # Convert to just the merged words, longest first within each group
    return [
        [word for _, _, word in sorted(group, key=lambda m: -m[1])]
        for group in groups
    ]


def _resolve_sentence(
    gold_segments: list[str],
    word_set: set[str],
    max_len: int,
) -> tuple[list[str] | None, list[list[str]]]:
    """Determine if a sentence needs LLM help or can be auto-resolved.

    Logic: if any 2+ consecutive gold segments concatenate to a CEDICT word,
    that's a merge candidate requiring LLM judgment. Otherwise the sentence
    is "free" — already maximally merged.

    Returns:
      (segments, groups)
      - If segments is not None: sentence is "free" (no LLM needed)
      - If segments is None: groups contains mutually-exclusive merge options
    """
    text = "".join(gold_segments)
    merges = _find_merge_candidates(gold_segments, word_set, max_len)

    if not merges:
        # No consecutive gold segments form a CEDICT word → already maximally
        # merged. Use greedy for canonical format (single chars for non-CEDICT).
        greedy = greedy_segment(text, word_set, max_len)
        return greedy, []

    # Has merge candidates → group overlapping ones and send to LLM.
    groups = _group_overlapping_merges(merges)
    return None, groups


def extract_icwb2(
    free_path: Path,
    tasks_path: Path,
    max_chars: int = 64,
    target_chars: int | None = None,
    seed: int = 42,
) -> None:
    """Extract ICWB2 into free training data + merge-decision LLM tasks.

    Args:
        free_path: Output path for free sentences (segmented.jsonl format)
        tasks_path: Output path for LLM tasks (cws_tasks.jsonl format)
        max_chars: Max chunk size (unused in merge-only, kept for API compat)
        target_chars: If set, randomly sample sentences to stay within this
            character budget (distributed proportionally across corpora).
        seed: Random seed for sampling.
    """
    print("  Loading CEDICT words...")
    word_set, max_len = load_cedict_words()
    print(f"  {len(word_set):,} words, max length {max_len}")

    free_path.parent.mkdir(parents=True, exist_ok=True)
    tasks_path.parent.mkdir(parents=True, exist_ok=True)

    # Pre-compute total chars across all corpora for proportional sampling
    _total_all_chars = 0
    if target_chars is not None:
        for _, path, is_traditional in CORPORA:
            if not path.exists():
                continue
            sents = _load_gold_sentences(path, to_simplified=is_traditional)
            _total_all_chars += sum(len("".join(s)) for s in sents)
        print(f"  Total ICWB2 chars: {_total_all_chars:,} (target: {target_chars:,})")

    total_free = 0
    total_tasks = 0
    task_id = 0

    f_free = open(free_path, "w", encoding="utf-8")
    f_tasks = open(tasks_path, "w", encoding="utf-8")

    for corpus_name, path, is_traditional in CORPORA:
        if not path.exists():
            print(f"  Skipping {corpus_name}: {path} not found")
            continue

        print(f"  Processing {corpus_name}...")
        gold_sentences = _load_gold_sentences(path, to_simplified=is_traditional)

        # Sample if target_chars is set
        if target_chars is not None:
            total_corpus_chars = sum(len("".join(s)) for s in gold_sentences)
            corpus_budget = target_chars * total_corpus_chars // _total_all_chars
            if total_corpus_chars > corpus_budget:
                rng = random.Random(seed)
                rng.shuffle(gold_sentences)
                sampled = []
                chars_so_far = 0
                for s in gold_sentences:
                    chars_so_far += len("".join(s))
                    sampled.append(s)
                    if chars_so_far >= corpus_budget:
                        break
                gold_sentences = sampled
                print(f"    Sampled {len(gold_sentences):,} sentences ({chars_so_far:,} chars, budget {corpus_budget:,})")

        corpus_free = 0
        corpus_tasks = 0

        for gold_segments in gold_sentences:
            text = "".join(gold_segments)
            if not passes_filter(text):
                continue

            segments, groups = _resolve_sentence(
                gold_segments, word_set, max_len
            )

            if segments is not None:
                # Free — no LLM needed
                result = {
                    "id": f"icwb2_{corpus_name}:{task_id}",
                    "source": f"icwb2-{corpus_name.lower()}",
                    "text": text,
                    "segments": segments,
                }
                f_free.write(json.dumps(result, ensure_ascii=False) + "\n")
                corpus_free += 1
            else:
                # Needs LLM merge decision
                task = {
                    "id": f"icwb2_{corpus_name}:{task_id}",
                    "source": f"icwb2-{corpus_name.lower()}",
                    "segments": " ".join(gold_segments),
                    "groups": groups,
                }
                f_tasks.write(json.dumps(task, ensure_ascii=False) + "\n")
                corpus_tasks += 1

            task_id += 1

        print(f"    {corpus_name}: {corpus_free:,} free, {corpus_tasks:,} tasks")
        total_free += corpus_free
        total_tasks += corpus_tasks

    f_free.close()
    f_tasks.close()

    print(f"  → {total_free:,} free sentences → {free_path.name}")
    print(f"  → {total_tasks:,} LLM tasks → {tasks_path.name}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract ICWB2 data for CWS pipeline (merge-only)")
    parser.add_argument("--target-chars", type=int, default=None,
                        help="Target total characters (samples proportionally if set)")
    args = parser.parse_args()

    data = _ROOT / "data" / "dataset_v2"
    extract_icwb2(
        free_path=data / "icwb2_free.jsonl",
        tasks_path=data / "icwb2_cws_tasks.jsonl",
        target_chars=args.target_chars,
    )
