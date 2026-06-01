"""Convert MiCLS mappings to WSD results format (merged sense clusters).

Maps each row in mappings.tsv to the corresponding cluster index in
entries_after_merging.json. Outputs rows compatible with wsd_results.jsonl
for merging with LLM-labeled data downstream.

Usage:
    python -m ml.scripts.dataset_v2.wsd.convert_micls [--threshold 0.85]
"""
import csv
import json
import re
import argparse
from collections import defaultdict
from pathlib import Path

_ROOT = Path(__file__).parent.parent.parent.parent  # ml/
_DATA = _ROOT / "data"

ENTRIES_PATH = _DATA / "entries_after_merging.json"
MAPPINGS_PATH = _DATA / "mappings.tsv"
OUTPUT_PATH = _DATA / "dataset_v2" / "wsd_micls.jsonl"

DEFAULT_THRESHOLD = 0.85


def _build_sense_to_cluster_index(entries: list[dict]) -> dict[str, dict[str, int]]:
    """Build lookup: (word, pinyin) -> {english_sense: cluster_idx (1-based)}.

    A word+pinyin pair maps each raw English sense string to its parent cluster index.
    """
    # Group entries by word (multiple pinyins possible)
    by_word_pinyin: dict[tuple[str, str], list[dict]] = {}
    for e in entries:
        by_word_pinyin[(e["word"], e["pinyin"])] = e["clusters"]

    # Now build: for each (word, pinyin), combine all clusters across all pinyins
    # to match the combined-cluster logic in build_tasks.py
    by_word: dict[str, list[tuple[str, list[str]]]] = defaultdict(list)
    for e in entries:
        by_word[e["word"]].append((e["pinyin"], e["clusters"]))

    # Final index: word -> {sense_en: (cluster_idx_1based, pinyin)}
    lookup: dict[str, dict[str, tuple[int, str]]] = {}
    for word, entries_list in by_word.items():
        sense_map: dict[str, tuple[int, str]] = {}
        cluster_idx = 0
        for pinyin, clusters in entries_list:
            for cluster in clusters:
                cluster_idx += 1
                for sense in cluster["senses"]:
                    sense_lower = sense.lower().strip()
                    sense_map[sense_lower] = (cluster_idx, pinyin)
        lookup[word] = sense_map

    return lookup


def _count_clusters_for_word(entries: list[dict], word: str) -> int:
    """Count total clusters across all pinyins for a word."""
    count = 0
    for e in entries:
        if e["word"] == word:
            count += len(e["clusters"])
    return count


def convert_micls(
    mappings_path: Path = MAPPINGS_PATH,
    entries_path: Path = ENTRIES_PATH,
    output_path: Path = OUTPUT_PATH,
    threshold: float = DEFAULT_THRESHOLD,
) -> None:
    """Convert MiCLS mappings to wsd_results.jsonl format."""
    print(f"Loading entries from {entries_path.name}...")
    entries = json.loads(entries_path.read_text(encoding="utf-8"))

    print("Building sense -> cluster index...")
    lookup = _build_sense_to_cluster_index(entries)

    # Pre-compute cluster counts per word
    cluster_counts: dict[str, int] = defaultdict(int)
    for e in entries:
        cluster_counts[e["word"]] += len(e["clusters"])

    print(f"Loading mappings from {mappings_path.name}...")
    with open(mappings_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    print(f"  {len(rows)} total rows")

    # Convert
    converted = 0
    skipped_threshold = 0
    skipped_not_found = 0
    skipped_monosemous = 0

    results = []
    for i, row in enumerate(rows):
        word = row["word"]
        context = row["context"]
        cedict_key = row["cedict_key"]
        score = float(row["similarity_score"])

        # Filter by threshold
        if score < threshold:
            skipped_threshold += 1
            continue

        # Parse cedict_key: word|pinyin|english_definition
        parts = cedict_key.split("|", 2)
        if len(parts) != 3:
            skipped_not_found += 1
            continue
        key_word, key_pinyin, key_en = parts

        # Skip if word is monosemous in merged entries
        n_clusters = cluster_counts.get(word, 0)
        if n_clusters < 2:
            skipped_monosemous += 1
            continue

        # Look up cluster index
        word_senses = lookup.get(word, {})
        key_en_lower = key_en.lower().strip()
        match = word_senses.get(key_en_lower)

        if match is None:
            skipped_not_found += 1
            continue

        cluster_idx, pinyin = match

        # Find word position in context (MiCLS uses #word# markers)
        marker_match = re.search(r"#" + re.escape(word) + r"#", context)
        if marker_match:
            # Position is where the word starts (excluding the leading #)
            # But in clean text, we need to compute position without markers
            clean_before = context[:marker_match.start()].replace("#", "")
            pos = len(clean_before)
            clean_sentence = context.replace("#", "")
        else:
            skipped_not_found += 1
            continue

        results.append({
            "id": f"wsd_micls_{i}",
            "source": "micls",
            "sentence": clean_sentence,
            "words": [{
                "word": word,
                "pos": pos,
                "sense": cluster_idx,
                "n_clusters": n_clusters,
            }],
        })
        converted += 1

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nResults:")
    print(f"  Converted: {converted}")
    print(f"  Skipped (below threshold {threshold}): {skipped_threshold}")
    print(f"  Skipped (sense not found in merged entries): {skipped_not_found}")
    print(f"  Skipped (monosemous after merging): {skipped_monosemous}")
    print(f"  → {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert MiCLS mappings to WSD results format")
    parser.add_argument("--mappings", type=Path, default=MAPPINGS_PATH)
    parser.add_argument("--entries", type=Path, default=ENTRIES_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                        help=f"Minimum similarity score (default: {DEFAULT_THRESHOLD})")
    args = parser.parse_args()

    convert_micls(args.mappings, args.entries, args.output, args.threshold)
