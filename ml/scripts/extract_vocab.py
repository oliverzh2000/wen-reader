#!/usr/bin/env python3
"""Extract vocab.txt from a HuggingFace tokenizer.json file.

Usage:
    python scripts/extract_vocab.py <tokenizer_dir> <output_path>

Reads tokenizer.json (or falls back to vocab.txt if it already exists)
from <tokenizer_dir> and writes a standard vocab.txt to <output_path>.
"""

import json
import sys
from pathlib import Path


def extract_vocab(tokenizer_dir: Path, output_path: Path):
    # If vocab.txt already exists, just copy it
    existing = tokenizer_dir / "vocab.txt"
    if existing.exists():
        output_path.write_text(existing.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"Copied existing vocab.txt → {output_path}")
        return

    # Extract from tokenizer.json
    tokenizer_json = tokenizer_dir / "tokenizer.json"
    if not tokenizer_json.exists():
        print(f"Error: neither vocab.txt nor tokenizer.json found in {tokenizer_dir}", file=sys.stderr)
        sys.exit(1)

    with open(tokenizer_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    vocab = data.get("model", {}).get("vocab", {})
    if not vocab:
        print(f"Error: no vocab found in {tokenizer_json}", file=sys.stderr)
        sys.exit(1)

    # Sort by token ID and write one token per line
    sorted_tokens = sorted(vocab.items(), key=lambda x: x[1])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for token, _ in sorted_tokens:
            f.write(token + "\n")

    print(f"Extracted {len(sorted_tokens)} tokens → {output_path}")


def main():
    if len(sys.argv) != 3:
        print("Usage: python scripts/extract_vocab.py <tokenizer_dir> <output_path>", file=sys.stderr)
        sys.exit(1)

    extract_vocab(Path(sys.argv[1]), Path(sys.argv[2]))


if __name__ == "__main__":
    main()
