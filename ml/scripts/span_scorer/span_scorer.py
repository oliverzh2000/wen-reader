"""
Core span scorer model and inference library.

Architecture (from DESIGN.md):
- Encoder: ckiplab/bert-base-chinese-ws (char-level tokenizer, 1:1 char→token)
- Span scoring head: [H[start] ; H[end] ; width_emb(len)] → MLP → scalar
- Decoding: DP (Viterbi-style) over all CEDICT spans to find best cover

Shared utilities: CedictTrie, CEDICT_RE, build_cedict_trie,
enumerate_cedict_spans, parse_segmentation — used by build_dataset.py,
train_span_scorer.py, and eval_span_scorer.py.

No side effects on import — caller loads models and CEDICT trie.
"""

import math
import re
from pathlib import Path

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# CEDICT regex + trie (shared across all span_scorer scripts)
# ---------------------------------------------------------------------------

CEDICT_RE = re.compile(r"^(\S+)\s+(\S+)\s+\[([^\]]+)\]\s+/(.+)/")


class CedictTrie:
    """Prefix trie of CEDICT simplified forms.

    Stores the set of all inserted words in `self.words` for quick
    membership checks and iteration.
    """

    def __init__(self):
        self.root: dict = {}
        self.words: set[str] = set()

    def insert(self, word: str):
        node = self.root
        for ch in word:
            node = node.setdefault(ch, {})
        node["$"] = True
        self.words.add(word)

    def get_words_at(self, text: str, start: int) -> list[str]:
        """Return all CEDICT words starting at position `start`."""
        words = []
        node = self.root
        for i in range(start, len(text)):
            ch = text[i]
            if ch not in node:
                break
            node = node[ch]
            if not isinstance(node, dict):
                break
            if "$" in node:
                words.append(text[start : i + 1])
        return words


def build_cedict_trie(cedict_path: str | Path) -> CedictTrie:
    """Parse CEDICT and build a trie of simplified forms."""
    trie = CedictTrie()
    with open(cedict_path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            m = CEDICT_RE.match(line.strip())
            if m:
                trie.insert(m.group(2))  # simplified form
    return trie


def enumerate_cedict_spans(sentence: str, trie: CedictTrie) -> dict[int, list[str]]:
    """For each character position, list all CEDICT words starting there."""
    spans_at = {}
    for i in range(len(sentence)):
        words = trie.get_words_at(sentence, i)
        if words:
            spans_at[i] = words
    return spans_at


def parse_segmentation(segmented: str) -> list[str]:
    """Parse a pipe-delimited (or space-delimited) segmentation into words."""
    if "|" in segmented:
        return [w.strip() for w in segmented.split("|") if w.strip()]
    return [w.strip() for w in segmented.split() if w.strip()]


# ---------------------------------------------------------------------------
# Span Scoring Head
# ---------------------------------------------------------------------------

MAX_WORD_LEN = 19
WIDTH_EMBED_DIM = 64
HIDDEN_DIM = 768  # MacBERT hidden size
MLP_HIDDEN = 256


class SpanScoringHead(nn.Module):
    """
    Scores candidate spans from encoder hidden states.

    Input per span: [H[start] ; H[end] ; width_emb(length)]
    Output: scalar score.
    """

    def __init__(
        self,
        hidden_dim: int = HIDDEN_DIM,
        width_embed_dim: int = WIDTH_EMBED_DIM,
        mlp_hidden: int = MLP_HIDDEN,
        max_word_len: int = MAX_WORD_LEN,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.width_embed = nn.Embedding(max_word_len, width_embed_dim)
        input_dim = hidden_dim * 2 + width_embed_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, 1),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        span_starts: torch.Tensor,
        span_ends: torch.Tensor,
        span_widths: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: (seq_len, hidden_dim) — encoder output (single sentence)
            span_starts:   (total_spans,) — start indices (into seq dim)
            span_ends:     (total_spans,) — end indices (inclusive)
            span_widths:   (total_spans,) — span lengths (1-indexed)
        Returns:
            scores: (total_spans,) — scalar score per span
        """
        w_emb = self.width_embed(span_widths - 1)
        start_hidden = hidden_states[span_starts]
        end_hidden = hidden_states[span_ends]
        span_repr = torch.cat([start_hidden, end_hidden, w_emb], dim=-1)
        return self.mlp(span_repr).squeeze(-1)


# ---------------------------------------------------------------------------
# Full model wrapper (encoder + head) — used for training
# ---------------------------------------------------------------------------

class SpanScorer(nn.Module):
    """MacBERT encoder + span scoring head."""

    def __init__(self, encoder, head: SpanScoringHead):
        super().__init__()
        self.encoder = encoder
        self.head = head


# ---------------------------------------------------------------------------
# DP decoding
# ---------------------------------------------------------------------------

def dp_decode(
    sentence: str,
    trie: CedictTrie,
    score_fn,
    use_log_prob: bool = True,
) -> list[str]:
    """
    Find the highest-scoring non-overlapping cover of the sentence
    using CEDICT spans + single-char fallback.

    Args:
        sentence: raw Chinese text
        trie: CEDICT trie
        score_fn: callable(start, end_exclusive) → float score for span
                  sentence[start:end_exclusive], or None if the span
                  cannot be scored (e.g. exceeds max width)
        use_log_prob: if True, apply log-softmax per position before summing
                      (maximizes joint probability). If False, sum raw scores.

    Returns:
        list of segmented words
    """
    n = len(sentence)
    best_score = [-float("inf")] * (n + 1)
    best_score[0] = 0.0
    backtrack = [0] * (n + 1)

    for i in range(n):
        if best_score[i] == -float("inf"):
            continue

        # Gather all candidate spans at position i
        words = trie.get_words_at(sentence, i)
        char = sentence[i]
        if char not in [w for w in words if len(w) == 1]:
            words.append(char)

        # Score all candidates, filtering out un-scorable spans
        scored_words = []
        raw_scores = []
        for w in words:
            s = score_fn(i, i + len(w))
            if s is not None:
                scored_words.append(w)
                raw_scores.append(s)

        if not scored_words:
            # Fallback: single char with score 0
            j = i + 1
            total = best_score[i] + 0.0
            if total > best_score[j]:
                best_score[j] = total
                backtrack[j] = i
            continue

        # Normalize scores per position if using log-prob mode
        if use_log_prob:
            scores_t = torch.tensor(raw_scores)
            dp_scores = torch.log_softmax(scores_t, dim=0).tolist()
        else:
            dp_scores = raw_scores

        # Additive DP: try ALL candidates at this position.
        for w, s in zip(scored_words, dp_scores):
            j = i + len(w)
            total = best_score[i] + s
            if total > best_score[j]:
                best_score[j] = total
                backtrack[j] = i

    segments = []
    pos = n
    while pos > 0:
        start = backtrack[pos]
        segments.append(sentence[start:pos])
        pos = start
    segments.reverse()
    return segments


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def segment_sentence(
    sentence: str,
    tokenizer,
    model: SpanScorer,
    trie: CedictTrie,
    device: str = "cpu",
    use_log_prob: bool = True,
) -> str:
    """
    Full span-scorer segmentation pipeline.
    Returns dash-separated segmentation string.
    """
    if not sentence:
        return ""

    model.eval()
    chars = list(sentence)

    encoding = tokenizer(
        chars,
        is_split_into_words=True,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        hidden = model.encoder(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state.squeeze(0)  # (seq_len, hidden_dim)

    def score_fn(start: int, end_exclusive: int) -> float | None:
        width = end_exclusive - start
        # Respect the model's actual width embedding size
        max_w = model.head.width_embed.num_embeddings
        if width < 1 or width > max_w:
            return None
        s = torch.tensor([start], device=device)
        e = torch.tensor([end_exclusive - 1], device=device)
        w = torch.tensor([width], device=device)
        with torch.no_grad():
            score = model.head(hidden, s + 1, e + 1, w)  # +1 for [CLS]
        return score.item()

    segments = dp_decode(sentence, trie, score_fn, use_log_prob=use_log_prob)
    return "-".join(segments)
