"""Build CWS task file from sentences.json → cws_tasks.jsonl.

Each task contains a text chunk plus positions where multi-char CEDICT
words start. The LLM picks which word to use at each position (or
implicitly falls back to single-char segmentation by not picking).

Output schema (one JSON object per line):
    {
        "id": "book_idx:chunk_idx",
        "source": str,
        "text": str,
        "candidates": {"pos": ["word1", "word2", ...], ...}
    }

"candidates" maps character positions (as strings) to lists of multi-char
words starting there (longest first). Positions with no multi-char matches
are omitted (single-char is always the implicit fallback).
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from chunking import chunk_paragraphs
from cws.cedict_lookup import load_cedict_words, words_at, load_merged_senses, lookup_defs_for_candidates


def build_cws_tasks(
    sentences_path: Path, output_path: Path, max_chars: int = 64,
    target_chars: int | None = None, seed: int = 42,
) -> None:
    """Read sentences.json, chunk, find candidate words, write tasks.

    Args:
        target_chars: If set, randomly sample tasks until reaching this
            character budget. Useful to reserve corpus for WSD.
    """
    books = json.loads(sentences_path.read_text(encoding="utf-8"))

    print("  Loading CEDICT words...", end=" ")
    word_set, max_len = load_cedict_words()
    print(f"{len(word_set)} words (max len {max_len})")

    print("  Loading merged senses...", end=" ")
    merged = load_merged_senses()
    print(f"{len(merged)} words")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Build all tasks in memory first (for optional sampling)
    all_tasks = []
    for book_idx, book in enumerate(books):
        source = book["source"]
        chunks = chunk_paragraphs(book["paragraphs"], max_chars=max_chars)
        for chunk_idx, text in enumerate(chunks):
            candidates = {}
            for pos in range(len(text)):
                words = words_at(text, pos, word_set, max_len)
                if words:
                    candidates[str(pos)] = words

            if not candidates:
                continue

            defs = lookup_defs_for_candidates(candidates, merged)

            task = {
                "id": f"{book_idx}:{chunk_idx}",
                "source": source,
                "text": text,
                "candidates": candidates,
                "defs": defs,
            }
            all_tasks.append(task)

    total_chars = sum(len(t["text"]) for t in all_tasks)

    # Sample by character budget if target_chars is set
    if target_chars is not None and total_chars > target_chars:
        import random
        rng = random.Random(seed)
        rng.shuffle(all_tasks)
        sampled = []
        chars_so_far = 0
        for task in all_tasks:
            chars_so_far += len(task["text"])
            sampled.append(task)
            if chars_so_far >= target_chars:
                break
        sampled.sort(key=lambda t: t["id"])
        print(f"  Sampled {len(sampled)} tasks ({chars_so_far:,} chars) from {len(all_tasks)} ({total_chars:,} chars)")
        all_tasks = sampled

    with open(output_path, "w", encoding="utf-8") as f:
        for task in all_tasks:
            f.write(json.dumps(task, ensure_ascii=False) + "\n")

    print(f"  {len(all_tasks)} tasks → {output_path.name}")


if __name__ == "__main__":
    root = Path(__file__).parent.parent.parent.parent  # ml/
    data = root / "data" / "dataset_v2"
    build_cws_tasks(data / "sentences.json", data / "ebooks_cws_tasks.jsonl")
