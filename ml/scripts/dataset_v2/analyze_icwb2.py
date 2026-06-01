#!/usr/bin/env python3
"""Analyze ICWB2 datasets: how much agrees with greedy longest-match cedict?

Reports:
  - Sentence-level: % of sentences where greedy == gold (full match)
  - Boundary-level: % of B/I positions that agree between greedy and gold
  - Per-corpus breakdown (MSR, PKU, AS, CityU)

Usage:
    python -m dataset_v2.analyze_icwb2
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from opencc import OpenCC

from cws.cedict_lookup import load_cedict_words, words_at, greedy_segment
from filters import passes_filter

_ROOT = Path(__file__).parent.parent.parent  # ml/
_ICWB2 = _ROOT / "data" / "icwb2-data" / "training"

# Traditional → Simplified converter
_T2S = OpenCC("t2s")

CORPORA = [
    ("MSR", _ICWB2 / "msr_training.utf8", False),
    ("PKU", _ICWB2 / "pku_training.utf8", False),
    ("AS", _ICWB2 / "as_training.utf8", True),       # traditional
    ("CityU", _ICWB2 / "cityu_training.utf8", True),  # traditional
]


def segments_to_bi(segments: list[str]) -> list[int]:
    """Convert segment list to B/I labels (0=B, 1=I)."""
    labels = []
    for word in segments:
        for i in range(len(word)):
            labels.append(0 if i == 0 else 1)
    return labels


def load_gold_sentences(path: Path, to_simplified: bool) -> list[list[str]]:
    """Load space-separated gold segmentation, return list of segment lists."""
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


def _cedict_spans(segments: list[str], word_set: set[str]) -> set[tuple[int, str]]:
    """Extract (position, word) for every multi-char cedict word in a segmentation."""
    spans = set()
    pos = 0
    for seg in segments:
        if len(seg) > 1 and seg in word_set:
            spans.add((pos, seg))
        pos += len(seg)
    return spans


def _cedict_positions(text: str, word_set: set[str], max_len: int) -> dict[int, list[str]]:
    """Find all positions in text with at least one cedict word."""
    positions = {}
    for pos in range(len(text)):
        matches = words_at(text, pos, word_set, max_len)
        if matches:
            positions[pos] = matches
    return positions


def analyze_corpus(
    name: str, gold_sentences: list[list[str]], word_set: set[str], max_len: int
):
    """Compare greedy vs gold: which cedict words does each pick?"""
    total_sents = 0
    usable_sents = 0
    total_greedy_spans = 0
    total_gold_spans = 0
    disagree_spans = 0  # spans in one but not the other
    total_disagree_positions = 0  # unique positions needing LLM

    # Per-position stats
    total_cedict_positions = 0  # positions with any cedict word
    positions_greedy_confirmed = 0  # gold uses same cedict word as greedy
    positions_greedy_rejected = 0  # gold doesn't use greedy's cedict word (LLM decision pt)
    # Breakdown of rejections:
    reject_shorter_cedict = 0  # gold picks a shorter cedict word at same pos
    reject_no_cedict = 0  # gold doesn't use any cedict word at that pos (single-char split)

    disagreement_examples = []

    for gold_segments in gold_sentences:
        text = "".join(gold_segments)
        if not passes_filter(text):
            continue

        total_sents += 1
        greedy_segments = greedy_segment(text, word_set, max_len)

        gold_spans = _cedict_spans(gold_segments, word_set)
        greedy_spans = _cedict_spans(greedy_segments, word_set)

        total_gold_spans += len(gold_spans)
        total_greedy_spans += len(greedy_spans)

        # Per-position analysis: at every position where greedy picked a word,
        # did gold agree?
        greedy_span_map = {pos: word for pos, word in greedy_spans}
        cedict_positions = _cedict_positions(text, word_set, max_len)
        total_cedict_positions += len(cedict_positions)

        for pos, candidates in cedict_positions.items():
            greedy_word = greedy_span_map.get(pos)
            if greedy_word is None:
                # Greedy didn't start a word here (this pos is inside a longer
                # greedy word starting earlier) — not a decision point for greedy
                continue

            # Did gold also pick this word?
            if (pos, greedy_word) in gold_spans:
                positions_greedy_confirmed += 1
            else:
                positions_greedy_rejected += 1
                # Why did gold reject? Did it pick a shorter cedict word or no cedict word?
                gold_at_pos = [(p, w) for p, w in gold_spans if p == pos]
                if gold_at_pos:
                    reject_shorter_cedict += 1
                else:
                    reject_no_cedict += 1

        if gold_spans == greedy_spans:
            usable_sents += 1
        else:
            # Symmetric difference = spans in one but not the other
            diff = gold_spans.symmetric_difference(greedy_spans)
            disagree_spans += len(diff)
            disagree_pos_in_sent = len({pos for pos, _ in diff})
            total_disagree_positions += disagree_pos_in_sent
            if len(disagreement_examples) < 10:
                disagreement_examples.append((text, gold_segments, greedy_segments, diff))

    usable_pct = usable_sents / total_sents * 100 if total_sents else 0
    need_llm = total_sents - usable_sents
    avg_disagree = total_disagree_positions / need_llm if need_llm else 0

    total_greedy_decisions = positions_greedy_confirmed + positions_greedy_rejected
    confirmed_pct = positions_greedy_confirmed / total_greedy_decisions * 100 if total_greedy_decisions else 0

    print(f"\n{'='*60}")
    print(f"  {name}: {total_sents:,} sentences")
    print(f"  Greedy cedict spans: {total_greedy_spans:,}")
    print(f"  Gold cedict spans: {total_gold_spans:,}")
    print(f"  ---")
    print(f"  Usable without LLM: {usable_sents:,}/{total_sents:,} = {usable_pct:.1f}%")
    print(f"  Need LLM: {need_llm:,} sentences")
    print(f"    Disagreement positions: {total_disagree_positions:,} (avg {avg_disagree:.1f} per sentence)")
    print(f"  ---")
    print(f"  Greedy decision points (positions where greedy picks a word): {total_greedy_decisions:,}")
    print(f"    Gold confirms greedy: {positions_greedy_confirmed:,} ({confirmed_pct:.1f}%)")
    print(f"    Gold rejects greedy: {positions_greedy_rejected:,} ({100-confirmed_pct:.1f}%) ← LLM decision pts")
    print(f"      → gold picks shorter cedict: {reject_shorter_cedict:,}")
    print(f"      → gold uses no cedict word:  {reject_no_cedict:,}")
    print(f"{'='*60}")

    if disagreement_examples:
        print(f"\n  Sample disagreements ({name}):")
        for text, gold, greedy, diff in disagreement_examples[:5]:
            print(f"    Text:   {text[:60]}")
            print(f"    Gold:   {'/'.join(gold)[:80]}")
            print(f"    Greedy: {'/'.join(greedy)[:80]}")
            print(f"    Diff:   {sorted(diff)[:6]}")
            print()

    return {
        "name": name,
        "total_sents": total_sents,
        "usable_sents": usable_sents,
        "usable_pct": usable_pct,
        "total_greedy_spans": total_greedy_spans,
        "total_gold_spans": total_gold_spans,
        "disagree_spans": disagree_spans,
        "total_disagree_positions": total_disagree_positions,
        "total_greedy_decisions": total_greedy_decisions,
        "positions_greedy_confirmed": positions_greedy_confirmed,
        "positions_greedy_rejected": positions_greedy_rejected,
        "reject_shorter_cedict": reject_shorter_cedict,
        "reject_no_cedict": reject_no_cedict,
    }

    if disagreement_examples:
        print(f"\n  Sample disagreements ({name}):")
        for text, gold, greedy, diff in disagreement_examples[:5]:
            print(f"    Text:   {text[:60]}")
            print(f"    Gold:   {'/'.join(gold)[:80]}")
            print(f"    Greedy: {'/'.join(greedy)[:80]}")
            print(f"    Diff:   {sorted(diff)[:6]}")
            print()

    return {
        "name": name,
        "total_sents": total_sents,
        "usable_sents": usable_sents,
        "usable_pct": usable_pct,
        "total_greedy_spans": total_greedy_spans,
        "total_gold_spans": total_gold_spans,
        "disagree_spans": disagree_spans,
        "total_disagree_positions": total_disagree_positions,
        "total_ambiguous_positions": total_ambiguous_positions,
        "ambig_agrees_greedy": ambig_agrees_greedy,
        "ambig_disagrees_greedy": ambig_disagrees_greedy,
    }


def main():
    print("Loading cedict words...")
    word_set, max_len = load_cedict_words()
    print(f"  {len(word_set):,} words, max length {max_len}")

    results = []
    for name, path, is_traditional in CORPORA:
        if not path.exists():
            print(f"  Skipping {name}: {path} not found")
            continue
        print(f"\nLoading {name}...")
        gold = load_gold_sentences(path, to_simplified=is_traditional)
        print(f"  {len(gold):,} sentences loaded")
        r = analyze_corpus(name, gold, word_set, max_len)
        results.append(r)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    total_s = sum(r["total_sents"] for r in results)
    total_u = sum(r["usable_sents"] for r in results)
    total_gs = sum(r["total_greedy_spans"] for r in results)
    total_golds = sum(r["total_gold_spans"] for r in results)
    total_d = sum(r["disagree_spans"] for r in results)
    total_dp = sum(r["total_disagree_positions"] for r in results)
    total_decisions = sum(r["total_greedy_decisions"] for r in results)
    total_confirmed = sum(r["positions_greedy_confirmed"] for r in results)
    total_rejected = sum(r["positions_greedy_rejected"] for r in results)
    total_reject_shorter = sum(r["reject_shorter_cedict"] for r in results)
    total_reject_none = sum(r["reject_no_cedict"] for r in results)
    need_llm = total_s - total_u
    confirmed_pct = total_confirmed / total_decisions * 100 if total_decisions else 0

    print(f"  Overall: {total_s:,} sentences")
    print(f"  Usable sentences (no LLM): {total_u:,}/{total_s:,} = {total_u/total_s*100:.1f}%")
    print(f"  Sentences needing LLM: {need_llm:,}")
    print(f"    Disagreement positions: {total_dp:,} (avg {total_dp/need_llm:.1f} per sentence)")
    print(f"  ---")
    print(f"  Greedy decision points: {total_decisions:,}")
    print(f"    Gold confirms greedy: {total_confirmed:,} ({confirmed_pct:.1f}%)")
    print(f"    Gold rejects greedy: {total_rejected:,} ({100-confirmed_pct:.1f}%) ← LLM decision pts")
    print(f"      → gold picks shorter cedict: {total_reject_shorter:,}")
    print(f"      → gold uses no cedict word:  {total_reject_none:,}")
    print(f"  ---")
    print(f"  Cost estimation:")
    print(f"    {total_rejected:,} positions need LLM adjudication across {need_llm:,} sentences")
    print(f"    Avg {total_rejected/need_llm:.1f} LLM decision pts per sentence" if need_llm else "")


if __name__ == "__main__":
    main()
