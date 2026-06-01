#!/usr/bin/env python3
"""
Evaluate span scorer vs CWS models on shared test cases.

Loads the span scorer (encoder + head + DP decode), the ckiplab CWS
baseline, and the finetuned CWS model, runs all three on the same
validated test cases from cws_training/eval_cws.py.

Usage:
    python eval_span_scorer.py
"""

import sys
from pathlib import Path

import torch
from transformers import BertModel, BertTokenizerFast, BertForTokenClassification

# ---------------------------------------------------------------------------
# Path setup for cross-directory imports
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).parent
ML_DIR = SCRIPT_DIR.parent.parent
CWS_DIR = SCRIPT_DIR.parent / "cws_training"

sys.path.insert(0, str(CWS_DIR))
from cws import (  # noqa: E402
    load_cedict_vocab,
    score_segmentation,
    segment_sentence as cws_segment_sentence,
)
from eval_cws import VALIDATED_TEST_CASES  # noqa: E402

sys.path.insert(0, str(SCRIPT_DIR))
from span_scorer import (  # noqa: E402
    SpanScoringHead,
    SpanScorer,
    MAX_WORD_LEN,
    build_cedict_trie,
    segment_sentence as span_segment_sentence,
)
from build_dataset import _greedy_longest  # noqa: E402

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CEDICT_PATH = ML_DIR / "data" / "cedict_ts.u8"
SPAN_MACBERT_V2 = ML_DIR / "models" / "span_scorer_v2" / "final"
SPAN_MACBERT_NEW = ML_DIR / "models" / "span_scorer_macbert_new" / "final"
SPAN_MACBERT_OLD = ML_DIR / "models" / "span_scorer_macbert_old" / "final"
SPAN_ELECTRA_SMALL2 = ML_DIR / "models" / "span_scorer_electra_small2" / "final"
SPAN_ELECTRA_SMALL1 = ML_DIR / "models" / "cws_span_scorer_electra_small" / "final"
SPAN_ELECTRA_BASE2 = ML_DIR / "models" / "span_scorer_electra_base2" / "final"
SPAN_ELECTRA_BASE1 = ML_DIR / "models" / "cws_span_scorer_electra_base" / "final"
CWS_BASELINE = "ckiplab/bert-base-chinese-ws"
CWS_FINETUNED = ML_DIR / "models" / "cws_finetuned" / "final"

DEVICE = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

# (display_name, kind, path_or_name, base_encoder, quantize)
MODELS = [
    ("GreedyLongest",    "greedy", None,              None, False),
    ("SpanMacBERT-v2",  "span",   SPAN_MACBERT_V2,  "hfl/chinese-macbert-base", False),
    ("SpanMacBERT-new",  "span",   SPAN_MACBERT_NEW,  "hfl/chinese-macbert-base", False),
    ("SpanMacBERT-old",  "span",   SPAN_MACBERT_OLD,  "hfl/chinese-macbert-base", False),
    ("ElectraSmall2",    "span",   SPAN_ELECTRA_SMALL2, "hfl/chinese-electra-180g-small-discriminator", False),
    ("ElectraSmall1",    "span",   SPAN_ELECTRA_SMALL1, "hfl/chinese-electra-180g-small-discriminator", False),
    ("ElectraSmall1-int8", "span", SPAN_ELECTRA_SMALL1, "hfl/chinese-electra-180g-small-discriminator", "int8"),
    ("ElectraBase2",     "span",   SPAN_ELECTRA_BASE2,  "hfl/chinese-electra-180g-base-discriminator", False),
    ("ElectraBase1",     "span",   SPAN_ELECTRA_BASE1,  "hfl/chinese-electra-180g-base-discriminator", False),
    ("ElectraBase1-fp16", "span",  SPAN_ELECTRA_BASE1,  "hfl/chinese-electra-180g-base-discriminator", "fp16"),
    ("ElectraBase1-int8", "span",  SPAN_ELECTRA_BASE1,  "hfl/chinese-electra-180g-base-discriminator", "int8"),
    ("CkipBaseline",     "cws",    CWS_BASELINE,      None, False),
    ("CkipFinetuned",    "cws",    CWS_FINETUNED,     None, False),
]


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def quantize_model_int8(model: SpanScorer) -> SpanScorer:
    """Apply dynamic int8 quantization to the encoder and head (CPU only)."""
    # qnnpack backend works on ARM/macOS; fbgemm on x86
    torch.backends.quantized.engine = "qnnpack"
    model = model.to("cpu")
    model.encoder = torch.quantization.quantize_dynamic(
        model.encoder, {torch.nn.Linear}, dtype=torch.qint8
    )
    model.head = torch.quantization.quantize_dynamic(
        model.head, {torch.nn.Linear}, dtype=torch.qint8
    )
    model.eval()
    return model


def quantize_model_fp16(model: SpanScorer) -> SpanScorer:
    """Cast model to fp16 (simulates CoreML fp16 inference)."""
    model = model.half()
    model.eval()
    return model


def load_span_scorer(checkpoint_dir: Path, device: str,
                     base_encoder: str = "ckiplab/bert-base-chinese-ws",
                     quantize: bool = False) -> tuple:
    """Load tokenizer + SpanScorer from checkpoint.

    Handles two formats:
    - Final checkpoint: encoder saved via save_pretrained + separate span_head.pt
    - Trainer checkpoint: single model.safetensors with full SpanScorerForTraining state

    base_encoder: HF model name used as fallback for tokenizer and for
    loading encoder architecture when using trainer checkpoint format.
    """
    from transformers import ElectraModel

    # Tokenizer: fall back to base encoder if not saved in checkpoint
    tok_path = (
        str(checkpoint_dir)
        if (checkpoint_dir / "tokenizer_config.json").exists()
        else base_encoder
    )
    tokenizer = BertTokenizerFast.from_pretrained(tok_path)

    head = SpanScoringHead()

    def _load_encoder(path_or_name):
        """Load encoder, trying ElectraModel first then BertModel."""
        try:
            return ElectraModel.from_pretrained(path_or_name)
        except Exception:
            return BertModel.from_pretrained(path_or_name)

    if (checkpoint_dir / "span_head.pt").exists():
        # Final checkpoint format: encoder via save_pretrained + separate head
        encoder = _load_encoder(str(checkpoint_dir))
        state = torch.load(
            checkpoint_dir / "span_head.pt", map_location=device, weights_only=True,
        )
        # Backwards compatibility: old checkpoints may have smaller width_embed
        ckpt_max_word_len = state["width_embed.weight"].shape[0]
        hidden_dim = state["mlp.0.weight"].shape[1] - MAX_WORD_LEN * 0  # infer from encoder
        hidden_dim = encoder.config.hidden_size
        if ckpt_max_word_len != MAX_WORD_LEN:
            head = SpanScoringHead(hidden_dim=hidden_dim, max_word_len=ckpt_max_word_len)
        else:
            head = SpanScoringHead(hidden_dim=hidden_dim)
        head.load_state_dict(state)
    else:
        # Trainer checkpoint format: full model state dict
        from safetensors.torch import load_file
        state = load_file(str(checkpoint_dir / "model.safetensors"), device=device)
        encoder_state = {k.removeprefix("encoder."): v for k, v in state.items() if k.startswith("encoder.")}
        head_state = {k.removeprefix("head."): v for k, v in state.items() if k.startswith("head.")}
        # Backwards compatibility: old checkpoints may have smaller width_embed
        ckpt_max_word_len = head_state["width_embed.weight"].shape[0]
        encoder = _load_encoder(base_encoder)
        hidden_dim = encoder.config.hidden_size
        if ckpt_max_word_len != MAX_WORD_LEN:
            head = SpanScoringHead(hidden_dim=hidden_dim, max_word_len=ckpt_max_word_len)
        else:
            head = SpanScoringHead(hidden_dim=hidden_dim)
        encoder.load_state_dict(encoder_state)
        head.load_state_dict(head_state)

    model = SpanScorer(encoder, head).to(device)
    model.eval()
    if quantize:
        model = quantize_model_int8(model)
    return tokenizer, model


def load_cws_model(model_path: str, device: str) -> tuple:
    """Load a BertForTokenClassification model + tokenizer."""
    tokenizer = BertTokenizerFast.from_pretrained(model_path)
    model = BertForTokenClassification.from_pretrained(model_path).to(device)
    model.eval()
    return tokenizer, model


# ---------------------------------------------------------------------------
# Eval
# ---------------------------------------------------------------------------

def run_eval(test_cases, models, trie, cedict_vocab, device, use_log_prob=True):
    """Run all models on test cases and print side-by-side results.

    models: list of (name, "span"|"cws", tokenizer, model)
    """
    names = [name for name, *_ in models]
    tallies = {name: {"perfect": 0, "oversplit": 0, "wrong": 0} for name in names}

    total = len(test_cases)
    for i, (sentence, gold) in enumerate(test_cases, 1):
        print(f"\n[{i:02d}] Input: {sentence}")
        print(f"     Gold:  {gold}")

        for name, kind, tok, mod in models:
            if kind == "span":
                # Detect model device (quantized models are on CPU)
                mod_device = next(mod.parameters()).device if mod else device
                result = span_segment_sentence(
                    sentence, tok, mod, trie, device=str(mod_device),
                    use_log_prob=use_log_prob,
                )
            elif kind == "greedy":
                result = "-".join(_greedy_longest(sentence, trie))
            else:
                result = cws_segment_sentence(
                    sentence, tok, mod, device=device, cedict_vocab=cedict_vocab,
                )
            score = score_segmentation(result, gold)
            tallies[name][score] += 1
            mark = {"perfect": "✓✓", "oversplit": "✓~", "wrong": "✗"}[score]
            print(f"     {name}: {result} {mark}")

    # Summary
    print(f"\n{'=' * 70}")
    print(f"RESULTS (log_prob={use_log_prob})")
    print(f"{'=' * 70}")
    for name in names:
        t = tallies[name]
        passed = t["perfect"] + t["oversplit"]
        print(
            f"  {name}: {t['perfect']} perfect + {t['oversplit']} oversplit "
            f"= {passed}/{total} passed ({t['wrong']} wrong)"
        )
    print()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate span scorer models")
    parser.add_argument("--raw-scores", action="store_true",
                        help="Use raw score addition instead of log-prob in DP")
    args = parser.parse_args()

    use_log_prob = not args.raw_scores

    print(f"DP mode: {'log-prob' if use_log_prob else 'raw scores'}")
    print("Building CEDICT trie + vocab...")
    trie = build_cedict_trie(CEDICT_PATH)
    cedict_vocab = load_cedict_vocab(str(CEDICT_PATH))

    loaded = []
    for name, kind, path, base_encoder, quantize in MODELS:
        if kind == "greedy":
            loaded.append((name, kind, None, None))
            continue
        print(f"Loading {name}...")
        if kind == "span":
            # Quantized models must run on CPU; fp16 stays on device
            dev = "cpu" if quantize == "int8" else DEVICE
            tok, mod = load_span_scorer(path, dev, base_encoder=base_encoder,
                                        quantize=False)
            if quantize == "int8":
                mod = quantize_model_int8(mod)
            elif quantize == "fp16":
                mod = quantize_model_fp16(mod)
        else:
            tok, mod = load_cws_model(str(path), DEVICE)
        loaded.append((name, kind, tok, mod))

    print(f"Device: {DEVICE}")
    print(f"Test cases: {len(VALIDATED_TEST_CASES)}")

    run_eval(VALIDATED_TEST_CASES, loaded, trie, cedict_vocab, DEVICE,
             use_log_prob=use_log_prob)


if __name__ == "__main__":
    main()
