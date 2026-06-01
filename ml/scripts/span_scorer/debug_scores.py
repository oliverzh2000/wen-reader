#!/usr/bin/env python3
"""Dump per-span scores and hidden state previews for cross-checking with Swift.

Usage (from ml/):
    python scripts/span_scorer/debug_scores.py "倭寇侵华日"
"""

import sys
from pathlib import Path

import torch
from transformers import BertModel, BertTokenizerFast

SCRIPT_DIR = Path(__file__).parent
ML_DIR = SCRIPT_DIR.parent.parent

sys.path.insert(0, str(SCRIPT_DIR))
from span_scorer import (
    SpanScoringHead,
    SpanScorer,
    build_cedict_trie,
    MAX_WORD_LEN,
)

CEDICT_PATH = ML_DIR / "data" / "cedict_ts.u8"
MODEL_DIR = ML_DIR / "models" / "span_scorer_macbert_new" / "final"


def main():
    sentence = sys.argv[1] if len(sys.argv) > 1 else "倭寇侵华日"

    print(f"Sentence: {sentence}")
    print(f"Model: {MODEL_DIR}")

    trie = build_cedict_trie(CEDICT_PATH)
    tokenizer = BertTokenizerFast.from_pretrained(str(MODEL_DIR))
    encoder = BertModel.from_pretrained(str(MODEL_DIR))
    head = SpanScoringHead()
    head.load_state_dict(torch.load(MODEL_DIR / "span_head.pt", map_location="cpu", weights_only=True))
    model = SpanScorer(encoder, head)
    model.eval()

    chars = list(sentence)
    encoding = tokenizer(chars, is_split_into_words=True, return_tensors="pt", truncation=True, max_length=512)
    input_ids = encoding["input_ids"]
    attention_mask = encoding["attention_mask"]

    print(f"input_ids: {input_ids.tolist()[0]}")
    print(f"seq_len: {input_ids.shape[1]}")

    with torch.no_grad():
        hidden = encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state.squeeze(0)

    print(f"hidden shape: {hidden.shape}")
    print(f"H[0][0:5] = {hidden[0, :5].tolist()}")
    print(f"H[1][0:5] = {hidden[1, :5].tolist()}")
    print()

    # Score every candidate span, same as DP would
    for i in range(len(chars)):
        words = trie.get_words_at(sentence, i)
        char = chars[i]
        if char not in [w for w in words if len(w) == 1]:
            words.append(char)

        for w in words:
            width = len(w)
            if width > MAX_WORD_LEN:
                continue
            start = i
            end_inclusive = i + width - 1

            s = torch.tensor([start + 1])   # +1 for [CLS]
            e = torch.tensor([end_inclusive + 1])
            wt = torch.tensor([width])

            with torch.no_grad():
                score = head(hidden, s, e, wt).item()

            print(f"  pos {i}: '{w}' (len={width}) score={score:.4f}")

    print()


if __name__ == "__main__":
    main()
