"""Assemble CWS training data from segmented.jsonl.

Compares LLM segmentation against greedy longest-match baseline and
reports stats. Outputs final training file.

Output: cws_training.tsv (text \t segmentation)
"""
import json
from pathlib import Path

from cws.cedict_lookup import load_cedict_words, greedy_segment


def assemble_cws(segmented_path: Path, output_path: Path) -> None:
    """Assemble training TSV and print comparison stats."""
    word_set, max_len = load_cedict_words()

    results = [
        json.loads(line)
        for line in segmented_path.read_text().splitlines()
        if line.strip()
    ]

    total = len(results)
    agrees = 0
    disagrees_shorter = 0  # LLM chose shorter word than greedy
    disagrees_different = 0  # LLM chose different word

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for r in results:
            text = r["text"]
            llm_segs = r["segments"]
            greedy_segs = greedy_segment(text, word_set, max_len)

            if llm_segs == greedy_segs:
                agrees += 1
            else:
                # Classify disagreement
                llm_set = set()
                pos = 0
                for seg in llm_segs:
                    if len(seg) > 1:
                        llm_set.add((pos, seg))
                    pos += len(seg)

                greedy_set = set()
                pos = 0
                for seg in greedy_segs:
                    if len(seg) > 1:
                        greedy_set.add((pos, seg))
                    pos += len(seg)

                # Words in greedy but not in LLM (LLM chose shorter)
                only_greedy = greedy_set - llm_set
                only_llm = llm_set - greedy_set

                if only_greedy and not only_llm:
                    disagrees_shorter += 1
                else:
                    disagrees_different += 1

            # Write training line: text <tab> space-separated segments
            f.write(f"{text}\t{' '.join(llm_segs)}\n")

    disagrees_total = total - agrees
    print(f"  Total: {total}")
    print(f"  Agrees with greedy: {agrees} ({agrees*100//total}%)")
    print(f"  Disagrees: {disagrees_total} ({disagrees_total*100//total}%)")
    print(f"    LLM chose shorter: {disagrees_shorter}")
    print(f"    LLM chose different: {disagrees_different}")
    print(f"  → {output_path.name}")
