#!/usr/bin/env python3
"""End-to-end test: CWS segmentation + WSD sense ranking on passages.

Usage:
    python scripts/test_pipeline.py
    python scripts/test_pipeline.py --debug
    python scripts/test_pipeline.py --file data/passages.txt
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import BertTokenizerFast, BertForTokenClassification

sys.path.insert(0, str(Path(__file__).parent / "cws_training"))
from cws import load_cedict_vocab, segment_sentence

ML_DIR = Path(__file__).parent.parent
CWS_MODEL = ML_DIR / "models" / "cws_finetuned" / "final"
WSD_MODEL = ML_DIR / "models" / "wsd_finetuned" / "final"
CEDICT_PATH = ML_DIR / "data" / "cedict_ts.u8"
ENTRIES_PATH = ML_DIR / "data" / "entries_after_merging.json"
CACHE_PATH = ML_DIR / "data" / "translation_cache.json"
PASSAGES_PATH = ML_DIR / "data" / "passages.txt"


def load_wsd_senses(entries_path, cache_path):
    """word → [(zh_text, en_text, senses_list)]"""
    with open(entries_path) as f:
        entries = json.load(f)
    with open(cache_path) as f:
        cache = json.load(f)

    by_word = {}
    for e in entries:
        w, p = e["word"], e["pinyin"]
        clusters = e.get("clusters", [])
        if len(clusters) < 2:
            continue
        for c in clusters:
            zh_parts = [cache.get(f"{w}|{p}|{s}", s) for s in c["senses"]]
            zh = "；".join(zh_parts)
            en = "; ".join(c["senses"])
            by_word.setdefault(w, []).append((zh, en))
    return by_word


def run_wsd(word, context, wsd_model, by_word):
    """Returns ranked list of (score, en_label) for the word in context."""
    if word not in by_word:
        return None
    clusters = by_word[word]
    zh_texts = [c[0] for c in clusters]
    marked = context.replace(word, f"★{word}★", 1)

    ctx_emb = wsd_model.encode(marked, normalize_embeddings=True)
    sense_embs = wsd_model.encode(zh_texts, normalize_embeddings=True)
    scores = ctx_emb @ sense_embs.T

    ranked = sorted(zip(scores, clusters), key=lambda x: -x[0])
    return ranked


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--file", default=str(PASSAGES_PATH))
    args = parser.parse_args()

    # Load models
    print("Loading CWS model...")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    cws_tok = BertTokenizerFast.from_pretrained(str(CWS_MODEL))
    cws_model = BertForTokenClassification.from_pretrained(str(CWS_MODEL)).to(device)
    cws_model.eval()
    cedict_vocab = load_cedict_vocab(str(CEDICT_PATH))

    print("Loading WSD model...")
    wsd_model = SentenceTransformer(str(WSD_MODEL), device="cpu")

    print("Loading sense inventory...")
    by_word = load_wsd_senses(ENTRIES_PATH, CACHE_PATH)
    print(f"  {len(by_word)} polysemous words\n")

    # Load passages
    with open(args.file, encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]

    wsd_total = 0
    wsd_confident = 0  # top score > second by margin

    for line in lines:
        # Segment
        seg_result = segment_sentence(
            line, cws_tok, cws_model, device=device, cedict_vocab=cedict_vocab
        )
        words = seg_result.split("-")

        print(f"{'─'*60}")
        print(f"原文: {line}")
        print(f"分词: {' / '.join(words)}")

        # WSD for polysemous words
        wsd_results = []
        for word in words:
            ranked = run_wsd(word, line, wsd_model, by_word)
            if ranked and len(ranked) >= 2:
                wsd_total += 1
                top_score, (top_zh, top_en) = ranked[0]
                second_score = ranked[1][0]
                margin = top_score - second_score
                if margin > 0.05:
                    wsd_confident += 1
                wsd_results.append((word, ranked, margin))

        if wsd_results:
            print(f"  WSD:")
            for word, ranked, margin in wsd_results:
                top_score, (_, top_en) = ranked[0]
                conf = "✓" if margin > 0.05 else "~"
                print(f"    {conf} {word} → {top_en}  (Δ={margin:.3f})")
                if args.debug:
                    for score, (zh, en) in ranked:
                        print(f"        {score:.4f}  {en}")
                        print(f"                {zh}")
        print()

    print(f"{'═'*60}")
    print(f"WSD: {wsd_total} polysemous words encountered")
    if wsd_total:
        print(f"  Confident (Δ>0.05): {wsd_confident}/{wsd_total} ({100*wsd_confident/wsd_total:.0f}%)")


if __name__ == "__main__":
    main()
