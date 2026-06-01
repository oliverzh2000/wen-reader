"""Check how many CC-CEDICT simplified words appear in the extracted corpus."""
import json
import re
from collections import Counter
from pathlib import Path

_ROOT = Path(__file__).parent.parent.parent


def load_cedict_words() -> set[str]:
    """Load simplified Chinese words from cedict_ts.u8."""
    words = set()
    cedict_path = _ROOT.parent / "cedict" / "cedict_ts.u8"
    with open(cedict_path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.strip().split(" ", 2)
            if len(parts) >= 2:
                simplified = parts[1]
                if re.fullmatch(r'[\u4e00-\u9fff]+', simplified):
                    words.add(simplified)
    return words


def count_words_in_text(text: str, vocab: set[str]) -> Counter:
    """Scan text once, check all n-grams against vocab set."""
    max_len = max(len(w) for w in vocab)
    counts: Counter = Counter()
    n = len(text)
    for i in range(n):
        for length in range(1, min(max_len, n - i) + 1):
            ngram = text[i:i + length]
            if ngram in vocab:
                counts[ngram] += 1
    return counts


def main():
    cedict_words = load_cedict_words()
    print(f"CEDICT: {len(cedict_words)} pure-CJK simplified words\n")

    sentences_path = _ROOT / "data" / "dataset_v2" / "sentences.json"
    with open(sentences_path, encoding="utf-8") as f:
        books = json.load(f)

    # Concatenate all text
    all_text = "".join(s for b in books for p in b["paragraphs"] for s in p)
    print(f"Corpus: {len(all_text):,} chars\n")

    print("Scanning corpus (one pass)...")
    found = count_words_in_text(all_text, cedict_words)
    not_found = cedict_words - set(found)

    print(f"Covered: {len(found):,} / {len(cedict_words):,} "
          f"({100 * len(found) / len(cedict_words):.1f}%)")
    print(f"Not found: {len(not_found):,}\n")

    print("=== Top 50 most frequent words ===")
    for word, count in found.most_common(50):
        print(f"  {word}: {count:,}")

    print(f"\n=== Frequency distribution ===")
    brackets = [(1, 1), (2, 5), (6, 10), (11, 50), (51, 100),
                (101, 500), (501, 1000), (1001, None)]
    for lo, hi in brackets:
        if hi is None:
            n = sum(1 for c in found.values() if c >= lo)
            label = f"{lo}+"
        else:
            n = sum(1 for c in found.values() if lo <= c <= hi)
            label = f"{lo}-{hi}" if lo != hi else str(lo)
        print(f"  {label} occurrences: {n:,} words")

    # Per-book coverage
    print(f"\n=== Per-book coverage ===")
    for book in books:
        book_text = "".join(s for p in book["paragraphs"] for s in p)
        book_words = set()
        max_len = max(len(w) for w in cedict_words)
        n = len(book_text)
        for i in range(n):
            for length in range(1, min(max_len, n - i) + 1):
                ngram = book_text[i:i + length]
                if ngram in cedict_words:
                    book_words.add(ngram)
        print(f"  {book['source']}: {len(book_words):,} / {len(cedict_words):,} "
              f"({100 * len(book_words) / len(cedict_words):.1f}%)")

    # Sample not-found words (shortest first)
    not_found_sorted = sorted(not_found, key=len)
    print(f"\n=== Sample not-found words (shortest first) ===")
    for w in not_found_sorted[:50]:
        print(f"  {w}")


if __name__ == "__main__":
    main()
