"""Generate formatted prompt examples from selected sentences.

Uses the same cedict_lookup logic as build_tasks.py to find candidates
and definitions, then formats them as system prompt examples.

Run from ml/scripts/dataset_v2/:
  python -m cws.gen_prompt_examples
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from cws.cedict_lookup import load_cedict_words, words_at, load_merged_senses, lookup_defs_for_candidates

# Selected sentences with known gold segmentations (from eval_cws.py)
# Chosen to demonstrate diverse patterns and provide enough tokens for caching.
EXAMPLES = [
    # Typical longest-match (from v2, keep for continuity)
    {
        "text": "第二天黎明时分，鹿兆鹏走进白鹿原南端秦岭脚下的大王镇高级小学……",
        "gold": "第二天-黎明时分-，-鹿兆鹏-走进-白鹿原-南端-秦岭-脚下-的-大王-镇-高级小学-……",
        "note": "Typical case — pick longest candidates that fit",
    },
    # Edge case: longest exists but wrong meaning (from v2)
    {
        "text": "研究生命的起源",
        "gold": "研究-生命-的-起源",
        "note": "Edge case — longest exists but wrong meaning",
    },
    # Idiom in context
    {
        "text": "他一意孤行不听劝告",
        "gold": "他-一意孤行-不-听-劝告",
        "note": "Idiom — pick the 4-char idiom when its meaning fits",
    },
    # Ambiguous boundary: 副主席 vs 副 + 主席
    {
        "text": "他担任副主席一职",
        "gold": "他-担任-副主席-一-职",
        "note": "Compound word — 副主席 (vice-chairman) is one word here",
    },
    # Numbers: always pick
    {
        "text": "俗话说船大难掉头，大企业转型很慢",
        "gold": "俗话说-船-大-难-掉头-，-大-企业-转型-很-慢",
        "note": "Reject longer candidate when it doesn't match — 大难 (catastrophe) vs 大 + 难 (big + hard)",
    },
    # Classical / literary
    {
        "text": "拉马来，我去回太爷去",
        "gold": "拉-马-来-，-我-去-回-太爷-去",
        "note": "Classical vernacular — 马来 (Malaysia) doesn't fit; it's 马 (horse) + 来 (come)",
    },
]


def build_task(text: str, word_set: set[str], max_len: int, merged: dict) -> dict:
    """Build a task dict for a sentence (same logic as build_tasks.py)."""
    candidates = {}
    for pos in range(len(text)):
        words = words_at(text, pos, word_set, max_len)
        if words:
            candidates[str(pos)] = words

    defs = lookup_defs_for_candidates(candidates, merged)
    return {"text": text, "candidates": candidates, "defs": defs}


def format_example(task: dict, gold: str, note: str) -> str:
    """Format a single example in the system prompt style."""
    lines = [f"### {note}", ""]
    lines.append(f"Text: {task['text']}")

    candidates = task["candidates"]
    if candidates:
        lines.append("Candidates:")
        for pos in sorted(candidates.keys(), key=int):
            words = candidates[pos]
            lines.append(f"  pos {pos}: {', '.join(words)}")

    defs = task.get("defs", {})
    if defs:
        lines.append("Definitions:")
        for word in sorted(defs.keys()):
            lines.append(f"  {word} {defs[word]}")

    # Build picks from gold
    gold_segments = gold.split("-")
    picks = {}
    pos = 0
    for seg in gold_segments:
        if len(seg) > 1 and str(pos) in candidates:
            # Only include if it's actually a candidate
            candidate_words = candidates[str(pos)]
            if seg in candidate_words:
                picks[str(pos)] = seg
        pos += len(seg)

    lines.append("")
    picks_json = json.dumps({"picks": picks}, ensure_ascii=False)
    lines.append(picks_json)
    return "\n".join(lines)


def main():
    print("Loading CEDICT words...", file=sys.stderr)
    word_set, max_len = load_cedict_words()
    print(f"  {len(word_set)} words (max len {max_len})", file=sys.stderr)

    print("Loading merged senses...", file=sys.stderr)
    merged = load_merged_senses()
    print(f"  {len(merged)} words", file=sys.stderr)

    print("\n--- FORMATTED EXAMPLES ---\n")
    for ex in EXAMPLES:
        task = build_task(ex["text"], word_set, max_len, merged)
        formatted = format_example(task, ex["gold"], ex["note"])
        print(formatted)
        print()


if __name__ == "__main__":
    main()
