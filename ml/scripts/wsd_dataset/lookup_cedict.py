#!/usr/bin/env python3
"""Look up all senses for given words in CEDICT.

Usage:
    python scripts/wsd_training/lookup_cedict.py 干 意 深 负 绝
    python scripts/wsd_training/lookup_cedict.py --file words.txt
"""

import re
import sys
from pathlib import Path

CEDICT_PATH = Path(__file__).parent.parent.parent.parent / "cedict" / "cedict_ts.u8"


def load_cedict(path: Path) -> dict[str, list[tuple[str, str, str]]]:
    """Load CEDICT indexed by simplified. Returns {word: [(pinyin, english, trad), ...]}"""
    entries: dict[str, list[tuple[str, str, str]]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("#") or line.startswith("%"):
                continue
            m = re.match(r"(\S+)\s+(\S+)\s+\[([^\]]+)\]\s+/(.+)/", line.strip())
            if not m:
                continue
            trad, simp, pinyin, defs = m.groups()
            for eng in defs.split("/"):
                eng = eng.strip()
                if eng:
                    entries.setdefault(simp, []).append((pinyin, eng, trad))
    return entries


def print_senses(word: str, cedict: dict):
    senses = cedict.get(word, [])
    if not senses:
        print(f"\n❌ '{word}' — not found in CEDICT")
        return
    # Group by pinyin
    by_pinyin: dict[str, list[str]] = {}
    for pinyin, eng, trad in senses:
        by_pinyin.setdefault(pinyin, []).append(eng)
    print(f"\n{'='*60}")
    print(f"  {word}  ({len(senses)} sense entries, {len(by_pinyin)} pinyin readings)")
    print(f"{'='*60}")
    for i, (pinyin, engs) in enumerate(by_pinyin.items(), 1):
        print(f"  [{pinyin}]")
        for eng in engs:
            print(f"    • {eng}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python lookup_cedict.py WORD1 WORD2 ...")
        print("       python lookup_cedict.py --file words.txt")
        sys.exit(1)

    print(f"Loading CEDICT from {CEDICT_PATH}...")
    cedict = load_cedict(CEDICT_PATH)
    print(f"Loaded {len(cedict)} unique simplified entries.")

    if sys.argv[1] == "--file":
        with open(sys.argv[2]) as f:
            words = [w.strip() for w in f if w.strip()]
    else:
        words = sys.argv[1:]

    for word in words:
        print_senses(word, cedict)


if __name__ == "__main__":
    main()
