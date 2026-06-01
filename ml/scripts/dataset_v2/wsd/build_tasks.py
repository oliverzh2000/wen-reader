"""Build WSD tasks from segmented ebook sentences (sentence-level).

Each task = one sentence with ALL its polysemous words. This avoids
repeating the same sentence across multiple API calls and gives the LLM
full context to disambiguate all words at once.

Flow:
1. Load span-scorer-segmented sentences
2. For each sentence, find all polysemous word occurrences
3. Cap at MAX_PER_WORD occurrences per word across the corpus (randomly sampled)
4. Output one task per sentence (only sentences that survived capping)

Output schema (one JSON object per line):
    {
        "id": "wsd_<source>_<pi>_<si>",
        "source": str,
        "sentence": str,
        "words": [
            {
                "word": str,
                "pos": int,
                "clusters": [
                    {"idx": 1, "pinyin": "...", "senses_en": ["...", ...]},
                    ...
                ]
            },
            ...
        ]
    }
"""
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

_ROOT = Path(__file__).parent.parent.parent.parent  # ml/
_DATA = _ROOT / "data"

ENTRIES_PATH = _DATA / "entries_after_merging.json"

_CJK_RE = re.compile(r'[\u4e00-\u9fff]')

MAX_PER_WORD = 100  # cap per polysemous word


def _load_polysemous_entries() -> dict:
    """Load entries with combined clusters across all pinyins, indexed by word.

    A word is "polysemous" if it has 2+ clusters total across all its entries
    (pronunciation variants). Each cluster carries its own pinyin.
    """
    entries = json.loads(ENTRIES_PATH.read_text(encoding="utf-8"))

    # Group entries by word (same word can have multiple pinyins)
    from collections import defaultdict
    by_word = defaultdict(list)
    for e in entries:
        by_word[e["word"]].append(e)

    poly = {}
    for word, word_entries in by_word.items():
        # Combine all clusters from all entries, tagging each with its pinyin
        all_clusters = []
        for entry in word_entries:
            pinyin = entry["pinyin"]
            for cluster in entry["clusters"]:
                all_clusters.append({
                    "pinyin": pinyin,
                    "zh": cluster.get("zh", ""),
                    "en": cluster.get("en", ""),
                    "senses": cluster["senses"],
                })

        # Only include if 2+ clusters total (needs disambiguation)
        if len(all_clusters) >= 2:
            poly[word] = {
                "word": word,
                "clusters": all_clusters,
            }

    return poly


def _find_sentence_occurrences(segments_source: Path, poly_words: dict) -> list[dict]:
    """Find all polysemous word occurrences grouped by sentence.

    Args:
        segments_source: JSONL file with {"text": ..., "segments": [...], "source": ..., "id": ...}
        poly_words: dict of word -> entry for polysemous words

    Returns list of sentence dicts: {text, source, id, words: [{word, pos}, ...]}
    """
    sentences = []

    with open(segments_source, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            text = r["text"]
            segments = r["segments"]
            source = r.get("source", "ebook")
            sent_id = r.get("id", "")

            # Find all polysemous words in this sentence
            words_in_sent = []
            pos = 0
            for seg in segments:
                if seg in poly_words and len(seg) >= 1 and _CJK_RE.search(seg):
                    words_in_sent.append({"word": seg, "pos": pos})
                pos += len(seg)

            if words_in_sent:
                sentences.append({
                    "text": text,
                    "source": source,
                    "id": sent_id,
                    "words": words_in_sent,
                })

    return sentences


def build_wsd_tasks(
    segments_source: Path,
    output_path: Path,
    max_per_word: int = MAX_PER_WORD,
    seed: int = 42,
) -> None:
    """Build WSD task file from segmented sentences (sentence-level).

    Args:
        segments_source: JSONL with segmented sentences
        output_path: Where to write WSD tasks
        max_per_word: Maximum occurrences per word to include
        seed: Random seed for sampling
    """
    print("  Loading polysemous entries...")
    poly_words = _load_polysemous_entries()
    print(f"  {len(poly_words):,} polysemous words")

    print("  Finding occurrences in segmented text...")
    sentences = _find_sentence_occurrences(segments_source, poly_words)
    total_occ = sum(len(s["words"]) for s in sentences)
    print(f"  {len(sentences):,} sentences with polysemous words ({total_occ:,} occurrences)")

    # Build per-word occurrence index for capping
    # Each entry: (sentence_idx, word_idx_within_sentence)
    by_word: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for si, sent in enumerate(sentences):
        for wi, w in enumerate(sent["words"]):
            by_word[w["word"]].append((si, wi))

    # Cap: for words exceeding max_per_word, randomly select which occurrences to keep
    rng = random.Random(seed)
    keep_set: set[tuple[int, int]] = set()  # (sentence_idx, word_idx) pairs to keep
    capped_count = 0

    for word, occurrences in by_word.items():
        if len(occurrences) > max_per_word:
            capped_count += 1
            selected = rng.sample(occurrences, max_per_word)
        else:
            selected = occurrences
        for pair in selected:
            keep_set.add(pair)

    print(f"  {len(by_word):,} unique polysemous words found in corpus")
    print(f"  {capped_count:,} words capped at {max_per_word}")

    # Build final tasks: filter each sentence's words to only kept occurrences
    output_path.parent.mkdir(parents=True, exist_ok=True)
    task_count = 0
    total_words = 0

    with open(output_path, "w", encoding="utf-8") as f:
        for si, sent in enumerate(sentences):
            # Filter to kept words only
            kept_words = []
            for wi, w in enumerate(sent["words"]):
                if (si, wi) in keep_set:
                    entry = poly_words[w["word"]]
                    clusters = []
                    for ci, cluster in enumerate(entry["clusters"]):
                        clusters.append({
                            "idx": ci + 1,  # 1-based for LLM
                            "pinyin": cluster["pinyin"],
                            "senses_en": cluster["senses"],
                        })
                    kept_words.append({
                        "word": w["word"],
                        "pos": w["pos"],
                        "clusters": clusters,
                    })

            if not kept_words:
                continue

            task = {
                "id": f"wsd_{sent['id']}",
                "source": sent["source"],
                "sentence": sent["text"],
                "words": kept_words,
            }
            f.write(json.dumps(task, ensure_ascii=False) + "\n")
            task_count += 1
            total_words += len(kept_words)

    print(f"  → {task_count:,} sentence tasks ({total_words:,} word disambiguations) → {output_path.name}")

    # Stats
    words_per_sent = total_words / task_count if task_count else 0
    cluster_counts = [len(poly_words[w]["clusters"]) for w in by_word]
    avg_clusters = sum(cluster_counts) / len(cluster_counts) if cluster_counts else 0
    print(f"  Average words per sentence: {words_per_sent:.1f}")
    print(f"  Average clusters per word: {avg_clusters:.1f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build WSD tasks from segmented ebook sentences")
    parser.add_argument("--input", type=Path,
                        default=_DATA / "dataset_v2" / "wsd_segments.jsonl",
                        help="Segmented sentences JSONL (from span scorer)")
    parser.add_argument("--output", type=Path,
                        default=_DATA / "dataset_v2" / "wsd_tasks.jsonl",
                        help="Output tasks JSONL")
    parser.add_argument("--max-per-word", type=int, default=MAX_PER_WORD,
                        help=f"Max occurrences per word (default: {MAX_PER_WORD})")
    args = parser.parse_args()

    build_wsd_tasks(args.input, args.output, max_per_word=args.max_per_word)
