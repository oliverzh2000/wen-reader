"""
Core CWS (Chinese Word Segmentation) inference library.

## Decoding Strategy

We use a two-stage approach that cleanly separates the model's
context-aware segmentation from the dictionary-compatibility constraint:

### Stage 1: Greedy argmax (primary segmentation)

For each character, if P(B) > 0.5 it's a word boundary, else it
continues the current word. This is pure model output with zero
dictionary bias.

We considered lattice+Viterbi decoding (which finds the segmentation
that minimizes total negative log-likelihood across all characters),
but with no OOV penalty the difference from greedy argmax is negligible.
The model is trained with per-character cross-entropy, so each
character's prediction is already independently optimized. Viterbi
only helps when there are additional constraints (like an OOV penalty)
that create interactions between positions.

We explicitly chose NOT to use an OOV penalty in the primary pass.
While it can tip ambiguous cases toward cedict words, it also fights
the model when it's confident about a segmentation that produces OOV
words. This is the same problem as blind post-hoc merging rules: any
static dictionary-based bias is not context-aware and can always find
counterexamples (e.g., 奶奶+的 = "grandma's" vs 奶奶的 = swear word).
We learned this the hard way through extensive experimentation.

### Stage 2: Recursive OOV sub-splitting

After the primary segmentation, any multi-char word not in cedict gets
recursively split. The algorithm:

1. Find the internal character with the highest P(B) — this is where
   the model thinks a boundary is most likely, if forced to place one.
2. Split there into two pieces.
3. For each piece: if it's in cedict or a single char, stop. Otherwise
   recurse.

This is still model-driven: we're asking "if you had to split this
word, where would you?" The model makes the decision, we just force
it to make more decisions than it wanted to. This is a legitimate
constraint (the user needs lookupable words), not a static rule.

This approach correctly handles cases like 大义不容辞 where the model
wants one big word but the sub-splitter follows P(B) to find
大义-不容-辞 (splitting at the highest P(B) positions first), rather
than the Viterbi sub-splitter which would pick 大-义不容辞 because
义不容辞 is a single cedict entry.

No side effects on import — caller is responsible for loading
models and cedict vocab.
"""

import re
import numpy as np
import torch

LINE_RE = re.compile(r"^\S+\s+(\S+)\s+\[.+?]\s+/.*/", re.VERBOSE)


def load_cedict_vocab(cedict_path: str) -> set[str]:
    """Load simplified Chinese vocab from cedict file."""
    vocab = set()
    with open(cedict_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            m = LINE_RE.match(line)
            if m:
                vocab.add(m.group(1))
    return vocab


def get_begin_probs(
    sentences: str | list[str],
    tokenizer,
    model,
    device: str = "cpu",
) -> tuple[list[str], np.ndarray] | list[tuple[list[str], np.ndarray]]:
    """
    Run inference and return per-char B-label probability.

    Accepts a single sentence (str) or a batch (list[str]).

    Single: returns (chars, b_probs) where b_probs[i] = P(B) at char i.
    Batch:  returns list of (chars, b_probs) tuples.
    """
    single = isinstance(sentences, str)
    if single:
        sentences = [sentences]

    inputs = tokenizer(
        sentences, return_tensors="pt", return_offsets_mapping=True,
        padding=True, truncation=True,
    )
    all_offsets = inputs.pop("offset_mapping")  # (batch, seq_len, 2)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits
    # (batch, seq_len, num_labels) → P(B) is label 0
    all_probs = torch.softmax(logits, dim=-1)[:, :, 0].cpu().numpy()

    results = []
    for i, sent in enumerate(sentences):
        offsets = all_offsets[i, 1:-1].tolist()  # strip [CLS] and [SEP]
        probs = all_probs[i, 1:-1]  # strip [CLS] and [SEP]

        # Skip padding tokens (offset == [0, 0] after the real tokens)
        real_len = sum(1 for o in offsets if o != [0, 0])
        offsets = offsets[:real_len]
        probs = probs[:real_len]

        chars = list(sent)
        char_probs = []
        token_idx = 0
        for char_idx in range(len(chars)):
            while token_idx < len(offsets) - 1 and offsets[token_idx][1] <= char_idx:
                token_idx += 1
            char_probs.append(probs[token_idx])

        results.append((chars, np.array(char_probs)))

    return results[0] if single else results


def greedy_segment(chars: list[str], b_probs: np.ndarray) -> list[str]:
    """
    Greedy argmax segmentation: B if P(B) > 0.5, else I.

    Pure model output, no dictionary bias. First char is always B.
    """
    if not chars:
        return []

    segments = []
    current_word = chars[0]

    for i in range(1, len(chars)):
        if b_probs[i] > 0.5:
            segments.append(current_word)
            current_word = chars[i]
        else:
            current_word += chars[i]

    segments.append(current_word)
    return segments


def _recursive_subsplit(
    word: str,
    word_probs: np.ndarray,
    cedict_vocab: set[str],
) -> list[str]:
    """
    Recursively split an OOV word at the internal char with the highest
    P(B), then recurse on each piece until everything is in cedict or
    is a single char.

    This is purely model-driven: we're asking the model "if you had to
    put a boundary somewhere inside this word, where would it go?" and
    splitting at its best guess.
    """
    # Base cases: single char or in cedict — done
    if len(word) <= 1 or word in cedict_vocab:
        return [word]

    # Find the internal char (index 1..n-1) with the highest P(B)
    internal_probs = word_probs[1:]  # skip position 0 (always B)
    best_internal = int(np.argmax(internal_probs)) + 1  # +1 to get back to word index

    left_word = word[:best_internal]
    left_probs = word_probs[:best_internal]
    right_word = word[best_internal:]
    right_probs = word_probs[best_internal:]

    left_result = _recursive_subsplit(left_word, left_probs, cedict_vocab)
    right_result = _recursive_subsplit(right_word, right_probs, cedict_vocab)

    return left_result + right_result


def subsplit_oov_words(
    segments: list[str],
    b_probs: np.ndarray,
    cedict_vocab: set[str],
) -> list[str]:
    """
    Post-process segmentation to split OOV multi-char words.

    For each word that is >1 char and not in cedict_vocab, recursively
    splits at the internal position with the highest P(B) until every
    piece is in cedict or a single char.

    Primary word boundaries are never changed — only OOV words get
    internally re-segmented.
    """
    result = []
    char_offset = 0

    for word in segments:
        word_len = len(word)

        if word_len > 1 and word not in cedict_vocab:
            word_probs = b_probs[char_offset : char_offset + word_len]
            sub_segments = _recursive_subsplit(word, word_probs, cedict_vocab)
            result.extend(sub_segments)
        else:
            result.append(word)

        char_offset += word_len

    return result


def segment_sentence(
    sentence: str | list[str],
    tokenizer,
    model,
    device: str = "cpu",
    cedict_vocab: set[str] | None = None,
    subsplit_oov: bool = True,
    batch_size: int = 64,
) -> str | list[str]:
    """
    Full segmentation pipeline:
    1. Model inference → per-char P(B)
    2. Greedy argmax → primary segmentation (pure model, no bias)
    3. Recursive OOV sub-splitting → every word is in cedict or single char

    Accepts a single sentence (str) or a batch (list[str]).
    Returns dash-separated segmentation string(s).

    When given a batch, inference is batched in chunks of *batch_size*
    for GPU efficiency. Decoding is still per-sentence (CPU-bound, fast).
    """
    single = isinstance(sentence, str)
    sentences = [sentence] if single else sentence

    # Batch inference in chunks
    all_results: list[tuple[list[str], np.ndarray]] = []
    for start in range(0, len(sentences), batch_size):
        chunk = sentences[start : start + batch_size]
        all_results.extend(get_begin_probs(chunk, tokenizer, model, device=device))

    # Decode each sentence
    outputs = []
    for chars, b_probs in all_results:
        segments = greedy_segment(chars, b_probs)
        if subsplit_oov and cedict_vocab is not None:
            segments = subsplit_oov_words(segments, b_probs, cedict_vocab)
        outputs.append("-".join(segments))

    return outputs[0] if single else outputs


def score_segmentation(result: str, golden: str) -> str:
    """
    Score a segmentation result against golden reference.

    Returns: "perfect", "oversplit", or "wrong".

    Logic: if we can merge consecutive segments in result to reconstruct
    golden, it's just over-splitting (acceptable). Otherwise wrong.
    """
    if result == golden:
        return "perfect"

    result_segs = result.split("-")
    golden_segs = golden.split("-")

    i = 0
    for golden_seg in golden_segs:
        merged = ""
        while i < len(result_segs) and len(merged) < len(golden_seg):
            merged += result_segs[i]
            i += 1
        if merged != golden_seg:
            return "wrong"

    return "oversplit" if i == len(result_segs) else "wrong"
