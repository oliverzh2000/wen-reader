#!/usr/bin/env python3
"""Shared logic and constants for the WSD dataset pipeline.

Provides:
- CEDICT parsing and sense cleaning
- CWS-based corpus scanning
- Sentence ranking (bi-encoder + cross-encoder reranker)
- Model loading helpers
- Display utilities
"""
import json
import random
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import torch

# ---------------------------------------------------------------------------
# Span scorer import (needs path hack — keep it here so other scripts stay clean)
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent.parent / "cws_training"))
from cws import load_cedict_vocab, score_segmentation  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent.parent / "span_scorer"))
from span_scorer import (  # noqa: E402
    SpanScorer,
    SpanScoringHead,
    build_cedict_trie,
    segment_sentence as span_segment_sentence,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).parent.parent.parent          # ml/
CEDICT_PATH = _ROOT / "data" / "cedict_ts.u8"
TRANSLATION_CACHE_PATH = _ROOT / "data" / "translation_cache.json"
SPAN_MODEL_PATH = _ROOT / "models" / "span_scorer_macbert_new" / "final"
SPAN_BASE_ENCODER = "hfl/chinese-macbert-base"

RETRIEVAL_MODEL_NAME = "BAAI/bge-m3"
RETRIEVAL_PROMPT = "给定一个中文词的释义，找出使用该词义的例句\nQuery: "
EMBED_BATCH_SIZE = 256
TRIVIAL_PATTERNS = [
    # --- Variant / cross-reference patterns ---
    r"^variant of ",
    r"^old variant of ",
    r"^archaic variant of ",
    r"^ancient variant of ",
    r"^Japanese variant of ",
    r"^Taiwan variant of ",
    r"^erhua variant of ",
    r"^erhua form of ",
    r"^euphemistic variant of ",
    r"^literary variant of ",
    r"^dialectal variant of ",
    r"^nonstandard simplified variant of ",
    r"^classical variant of ",
    r"^erroneous variant of ",
    r"^erroneous written form of ",
    r"^obscure variant of ",
    r"^popular variant of ",
    # Parenthesized variant forms
    r"^\(variant of ",
    r"^\(old\) variant of ",
    r"^\(literary\) variant of ",
    r"^\(dialect\) contracted form of ",
    r"^\(Tw\) \(coll\.\) variant of ",
    r"^\(Cantonese\) variant of ",
    # --- See / see also ---
    r"^see [^\s]",
    r"^see also[: ]",
    r"^\(idiom\) see ",
    r"^\(Tw\) see ",
    r"^\(Taiwan\) see ",
    r"^\(Beijing dialect\) see ",
    r"^\(dialect\) see ",
    r"^\(old\) see ",
    r"^\(archaic\) see ",
    r"^\(classical\) see ",
    # --- Equivalence / aliasing ---
    r"^also written ",
    r"^same as ",
    r"^single-character equivalent of ",
    r"^dialectal equivalent of ",
    r"^Mandarin equivalent:",
    r"^equivalent to ",
    r"^also called ",
    r"^also known as ",
    r"^also termed ",
    r"^another name for ",
    r"^alternative name for ",
    r"^aka ",
    r"^a\.k\.a\. ",
    r"^formerly called ",
    r"^formerly known as ",
    r"^formerly written ",
    r"^Internet slang for ",
    # --- Abbreviation / short form ---
    r"^abbr\. for ",
    r"^abbr\. to ",
    r"^abbr\. of ",
    r"^short for ",
    # --- Pronunciation notes ---
    r"^Taiwan pr\. ",
    r"^also pr\. ",
    r"^pr\. ",
    r"^pronounced ",
    # --- Classifier notes (not a sense) ---
    r"^CL:",
    r"^classifier for ",
    # --- Spelling / form notes ---
    r"^old form of ",
    r"^old spelling of ",
    r"^old for ",
    r"^old name for ",
    r"^old term for ",
    r"^old word for ",
    r"^archaic form of ",
    r"^now written ",
    r"^now mostly replaced by ",
    r"^wrongly used for ",
    r"^misspelling of ",
    r"^misprint of ",
    r"^contracted form of ",
    r"^often written ",
    r"^sometimes written ",
    # --- Bound character notes (only used in compounds) ---
    r"^used in [^\s]+\[",
    # --- Etymology-only notes (no actual definition) ---
    r"^\(loanword from [^)]+\)$",
]

# Corpus / ranking settings
MIN_SENT_LEN = 10
MAX_SENT_LEN = 200
MAX_PER_DOC = 3
TOP_K = 5
RERANKER_MODEL_NAME = "BAAI/bge-reranker-v2-m3"
RERANK_TOP_N = 20  # retrieve this many with bi-encoder, then rerank to TOP_K


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class MergedSense:
    """A cluster of near-duplicate sense definitions."""
    label: str                  # representative label (first sense in cluster)
    members: list[str]          # all original sense strings in this cluster


@dataclass
class WordEntry:
    """A CC-CEDICT word with its pronunciation and merged senses."""
    word: str
    pinyin: str
    raw_senses: list[str]
    merged_senses: list[MergedSense] = field(default_factory=list)


# ---------------------------------------------------------------------------
# CEDICT parsing
# ---------------------------------------------------------------------------

def is_trivial(sense: str) -> bool:
    """True if *sense* is a cross-reference / variant note, not a real def."""
    return any(re.match(p, sense, re.IGNORECASE) for p in TRIVIAL_PATTERNS)


def parse_cedict(path: Path = CEDICT_PATH) -> dict[tuple[str, str], list[str]]:
    """Parse CC-CEDICT → {(simplified, pinyin): [sense1, …]}."""
    entries: dict[tuple[str, str], list[str]] = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            m = re.match(r'^(\S+)\s+(\S+)\s+\[([^\]]+)\]\s+/(.+)/', line.strip())
            if not m:
                continue
            simplified, pinyin = m.group(2), m.group(3).strip()
            senses = [s.strip() for s in m.group(4).split("/") if s.strip()]
            entries[(simplified, pinyin)].extend(senses)
    return dict(entries)


def clean_senses(senses: list[str]) -> list[str]:
    """Remove trivial cross-references and exact duplicates."""
    seen: set[str] = set()
    result: list[str] = []
    for s in senses:
        if is_trivial(s):
            continue
        key = s.lower().strip()
        if key not in seen:
            seen.add(key)
            result.append(s)
    return result


# ---------------------------------------------------------------------------
# Translation cache (loaded once)
# ---------------------------------------------------------------------------

def load_translation_cache(path: Path = TRANSLATION_CACHE_PATH) -> dict[str, str]:
    """Load word|pinyin|sense → Chinese translation mapping."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# CWS loading
# ---------------------------------------------------------------------------

def load_cws_model(device: str):
    """Load the span scorer model, tokenizer, and cedict trie."""
    from transformers import BertModel, BertTokenizerFast

    print(f"Loading span scorer from {SPAN_MODEL_PATH}…")

    # Tokenizer: fall back to base encoder if not saved in checkpoint
    tok_path = (
        str(SPAN_MODEL_PATH)
        if (SPAN_MODEL_PATH / "tokenizer_config.json").exists()
        else SPAN_BASE_ENCODER
    )
    tokenizer = BertTokenizerFast.from_pretrained(tok_path)

    encoder = BertModel.from_pretrained(str(SPAN_MODEL_PATH))
    head = SpanScoringHead()
    head.load_state_dict(torch.load(
        SPAN_MODEL_PATH / "span_head.pt", map_location=device, weights_only=True,
    ))
    model = SpanScorer(encoder, head).to(device)
    model.eval()

    trie = build_cedict_trie(CEDICT_PATH)
    return tokenizer, model, trie


# ---------------------------------------------------------------------------
# Corpus scanning
# ---------------------------------------------------------------------------

def split_sentences(text: str) -> list[str]:
    """Split Chinese text into sentences on common terminators."""
    for sep in ["。", "！", "？", "；", "\n"]:
        text = text.replace(sep, "。")
    return [s.strip() for s in text.split("。") if s.strip()]


def build_sentence_index(
    words: set[str],
    cws_tok, cws_model, cws_trie,
    device: str,
    cws_batch_size: int = 256,
    max_per_word: int = 100,
    checkpoint_path: Path | None = None,
    checkpoint_interval: int = 500,
) -> dict[str, list[str]]:
    """Scan Wikipedia + OpenSubtitles, collect CWS-verified sentences.

    Uses Aho-Corasick for fast multi-pattern pre-filtering, then the
    span scorer to verify word boundaries. The Aho-Corasick automaton
    is rebuilt periodically as words saturate, so the pre-filter gets
    cheaper over time.

    Saves a checkpoint every *checkpoint_interval* segmented batches
    if *checkpoint_path* is provided.

    Returns ``{word: [sentence, …]}``.
    """
    import ahocorasick
    from datasets import load_dataset
    from tqdm import tqdm

    try:
        import opencc
        t2s = opencc.OpenCC("t2s")
    except ImportError:
        print("Warning: opencc not installed, skipping traditional→simplified conversion")
        t2s = None

    index: dict[str, list[str]] = defaultdict(list)
    total_hits = 0
    total_segmented = 0
    batches_since_checkpoint = 0
    unsaturated = set(words)

    def build_automaton(target_words: set[str]):
        a = ahocorasick.Automaton()
        for w in target_words:
            a.add_word(w, w)
        a.make_automaton()
        return a

    automaton = build_automaton(unsaturated)
    rebuild_countdown = 500  # rebuild automaton every N saturations

    pending_sents: list[str] = []
    pending_matches: list[list[str]] = []

    def flush_batch(pbar):
        nonlocal total_hits, total_segmented, automaton, rebuild_countdown
        nonlocal batches_since_checkpoint
        if not pending_sents:
            return

        # Batch the encoder forward pass, then loop DP decode per sentence
        chars_batch = [list(s) for s in pending_sents]
        encoding = cws_tok(
            chars_batch,
            is_split_into_words=True,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)

        with torch.no_grad():
            hidden_batch = cws_model.encoder(
                input_ids=input_ids, attention_mask=attention_mask,
            ).last_hidden_state  # (batch, seq_len, hidden_dim)

        from span_scorer import dp_decode, MAX_WORD_LEN, CedictTrie

        seg_results = []
        for i, sent in enumerate(pending_sents):
            seq_len = int(attention_mask[i].sum().item())
            h = hidden_batch[i, :seq_len]  # (seq_len, hidden_dim)
            n = len(sent)

            # Pre-compute all candidate spans and score them in one batch
            all_starts = []
            all_ends = []
            all_widths = []
            span_keys = []  # (start, end_exclusive) for lookup

            for pos in range(n):
                cedict_words = cws_trie.get_words_at(sent, pos)
                # Single-char fallback
                candidates = {len(w) for w in cedict_words}
                candidates.add(1)
                for width in candidates:
                    end_exc = pos + width
                    if end_exc > n or width > MAX_WORD_LEN:
                        continue
                    si = pos + 1       # +1 for [CLS]
                    ei = end_exc - 1 + 1  # end inclusive + [CLS]
                    if ei >= h.shape[0]:
                        continue
                    all_starts.append(si)
                    all_ends.append(ei)
                    all_widths.append(width)
                    span_keys.append((pos, end_exc))

            # One batched head call for all spans
            if all_starts:
                s_t = torch.tensor(all_starts, device=device)
                e_t = torch.tensor(all_ends, device=device)
                w_t = torch.tensor(all_widths, device=device)
                with torch.no_grad():
                    scores = cws_model.head(h, s_t, e_t, w_t)
                score_map = {k: scores[j].item() for j, k in enumerate(span_keys)}
            else:
                score_map = {}

            def make_score_fn(smap):
                def score_fn(start, end_exclusive):
                    return smap.get((start, end_exclusive), -100.0)
                return score_fn

            segments = dp_decode(sent, cws_trie, make_score_fn(score_map))
            seg_results.append("-".join(segments))
        total_segmented += len(pending_sents)
        newly_saturated = []
        for sent, seg, matching in zip(pending_sents, seg_results, pending_matches):
            seg_words = set(seg.split("-"))
            for w in matching:
                if w in seg_words:
                    index[w].append(sent)
                    total_hits += 1
                    if len(index[w]) >= max_per_word and w in unsaturated:
                        unsaturated.discard(w)
                        newly_saturated.append(w)
        pending_sents.clear()
        pending_matches.clear()

        # Rebuild automaton when enough words saturate (cheaper pre-filter)
        if newly_saturated:
            rebuild_countdown -= len(newly_saturated)
            if rebuild_countdown <= 0 and unsaturated:
                automaton = build_automaton(unsaturated)
                rebuild_countdown = 500

        pbar.set_postfix(
            hits=total_hits, covered=len(index),
            saturated=len(words) - len(unsaturated),
            remaining=len(unsaturated), seg=total_segmented,
            refresh=False,
        )

        # Periodic checkpoint
        if checkpoint_path:
            batches_since_checkpoint += 1
            if batches_since_checkpoint >= checkpoint_interval:
                batches_since_checkpoint = 0
                _save_checkpoint(index, checkpoint_path)


    def _save_checkpoint(idx, path):
        tmp = Path(str(path) + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            f.write("{\n")
            items = list(idx.items())
            for i, (word, sents) in enumerate(items):
                line = json.dumps(word, ensure_ascii=False) + ": " + json.dumps(sents, ensure_ascii=False)
                f.write(line)
                if i < len(items) - 1:
                    f.write(",")
                f.write("\n")
            f.write("}\n")
        tmp.rename(path)
        print(f"\n  Checkpoint: {len(idx)} words, "
              f"{sum(len(v) for v in idx.values())} sentences → {path}")

    def process_sentence(sent: str, pbar):
        # Traditional → simplified
        if t2s is not None:
            sent = t2s.convert(sent)
        # Strip spaces, invisible unicode
        sent = sent.replace(" ", "").replace("\u3000", "")
        sent = re.sub(r'[\u200b-\u200f\u202a-\u202e\ufeff\u00ad\u2028\u2029]', '', sent)
        # Remove short parenthetical content (often English glosses)
        sent = re.sub(r'[(\uff08][^)\uff09]{0,20}[)\uff09]', '', sent)
        # Reject sentences containing ASCII letters (English fragments,
        # Latin taxonomy names, etc.)
        if re.search(r'[a-zA-Z]', sent):
            return
        sent = sent.strip()
        if len(sent) < MIN_SENT_LEN or len(sent) > MAX_SENT_LEN:
            return
        matching = list({w for _, w in automaton.iter(sent) if w in unsaturated})
        if not matching:
            return
        pending_sents.append(sent)
        pending_matches.append(matching)
        if len(pending_sents) >= cws_batch_size:
            flush_batch(pbar)

    def scan_corpus(name, dataset_iter, extract_fn):
        print(f"Scanning {name}…")
        t0 = time.time()
        with tqdm(dataset_iter, desc=name, unit="item") as pbar:
            for item in pbar:
                for sent in extract_fn(item):
                    process_sentence(sent, pbar)
                if not unsaturated:
                    print(f"  All words saturated — stopping early")
                    break
        flush_batch(tqdm(total=0, disable=True))
        print(f"  {name} done in {time.time() - t0:.0f}s")

    # --- Interleaved scan: alternate between corpora for balanced coverage ---
    wiki = load_dataset("wikimedia/wikipedia", "20231101.zh", split="train")
    subs = load_dataset("FradSer/OpenSubtitles-en-zh-cn-20m", split="train")

    wiki_iter = iter(wiki)
    subs_iter = iter(subs)
    wiki_exhausted = False
    subs_exhausted = False
    CHUNK = 500  # sentences per corpus before switching

    print("Scanning (interleaved Wikipedia + OpenSubtitles)…")
    t0 = time.time()
    with tqdm(desc="Interleaved", unit="sent") as pbar:
        while unsaturated and not (wiki_exhausted and subs_exhausted):
            # Wikipedia chunk
            if not wiki_exhausted:
                sent_count = 0
                while sent_count < CHUNK:
                    try:
                        doc = next(wiki_iter)
                    except StopIteration:
                        wiki_exhausted = True
                        break
                    for sent in split_sentences(doc["text"]):
                        process_sentence(sent, pbar)
                        sent_count += 1
                        pbar.update(1)
                        if sent_count >= CHUNK:
                            break
                    if not unsaturated:
                        break

            # OpenSubtitles chunk
            if not subs_exhausted and unsaturated:
                sent_count = 0
                while sent_count < CHUNK:
                    try:
                        row = next(subs_iter)
                    except StopIteration:
                        subs_exhausted = True
                        break
                    process_sentence(row["target"].strip(), pbar)
                    sent_count += 1
                    pbar.update(1)
                    if not unsaturated:
                        break

        flush_batch(pbar)

    if not unsaturated:
        print("  All words saturated — stopped early")
    print(f"  Done in {time.time() - t0:.0f}s")

    print(f"\n  Total segmented: {total_segmented}")
    print(f"  Total hits: {total_hits}")
    print(f"  Words with sentences: {len(index)}/{len(words)}")
    print(f"  Words saturated ({max_per_word}): {len(words) - len(unsaturated)}")
    print(f"  Words with 0 sentences: {len(words) - len(index)}")

    # Deduplicate and cap
    for w in list(index):
        unique = list(dict.fromkeys(index[w]))
        if len(unique) > max_per_word:
            unique = random.sample(unique, max_per_word)
        index[w] = unique

    return dict(index)


# ---------------------------------------------------------------------------
# Sentence ranking
# ---------------------------------------------------------------------------

def mark_word(sentence: str, word: str) -> str:
    """Wrap *word* in ★markers★ for attention focusing.
    Disabled — markers don't meaningfully affect bge-m3 ranking."""
    return sentence


def rank_sentences(
    entry: WordEntry,
    sentences: list[str],
    model,
    trans_cache: dict[str, str],
    top_k: int = TOP_K,
    batch_size: int = EMBED_BATCH_SIZE,
    reranker=None,
    rerank_top_n: int = RERANK_TOP_N,
) -> dict[str, list[dict]]:
    """Rank *sentences* for each merged sense of *entry*.

    Sense queries use ``RETRIEVAL_PROMPT``; sentences are encoded as-is
    (asymmetric retrieval).

    If *reranker* is provided, retrieves *rerank_top_n* candidates with the
    bi-encoder, then rescores them with the cross-encoder to pick the final *top_k*.

    Returns ``{sense_label: [{sentence, marked, score, rank, sense_members}, …]}``.
    """
    if not sentences or not entry.merged_senses:
        return {}

    marked = [mark_word(s, entry.word) for s in sentences]
    sent_embs = model.encode(
        marked, normalize_embeddings=True,
        show_progress_bar=False, batch_size=batch_size,
    )
    sent_t = torch.tensor(sent_embs)

    # Build sense queries with retrieval prompt
    sense_texts: list[str] = []
    for ms in entry.merged_senses:
        zh_members = []
        for member in ms.members:
            cache_key = f"{entry.word}|{entry.pinyin}|{member}"
            zh_members.append(trans_cache.get(cache_key, member))
        sense_texts.append(RETRIEVAL_PROMPT + "；".join(zh_members))

    sense_embs = model.encode(
        sense_texts, normalize_embeddings=True,
        show_progress_bar=False, batch_size=batch_size,
    )

    retrieve_k = rerank_top_n if reranker else top_k

    results: dict[str, list[dict]] = {}
    for i, ms in enumerate(entry.merged_senses):
        sense_t = torch.tensor(sense_embs[i]).unsqueeze(0)
        scores = torch.mm(sense_t, sent_t.T).squeeze(0)
        k = min(retrieve_k, len(sentences))
        top_scores, top_idx = scores.topk(k)

        candidates = [
            {"sentence": sentences[idx], "marked": marked[idx],
             "bi_score": float(s), "idx": int(idx)}
            for s, idx in zip(top_scores, top_idx)
        ]

        if reranker and candidates:
            # Cross-encoder rerank: score (sense_text, sentence) pairs
            sense_query = sense_texts[i]
            pairs = [[sense_query, c["sentence"]] for c in candidates]
            ce_scores = reranker.predict(pairs)
            for c, ce_s in zip(candidates, ce_scores):
                c["ce_score"] = float(ce_s)
            candidates.sort(key=lambda c: c["ce_score"], reverse=True)
            candidates = candidates[:top_k]
            score_key = "ce_score"
        else:
            candidates = candidates[:top_k]
            score_key = "bi_score"

        results[ms.label] = [
            {
                "sentence": c["sentence"],
                "marked": c["marked"],
                "score": c[score_key],
                "rank": r + 1,
                "sense_members": ms.members,
            }
            for r, c in enumerate(candidates)
        ]
    return results


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_embedding_model(model_name: str = RETRIEVAL_MODEL_NAME):
    """Load a SentenceTransformer (device auto-detected)."""
    from sentence_transformers import SentenceTransformer
    print(f"Loading model: {model_name}")
    return SentenceTransformer(model_name)


def load_reranker(model_name: str = RERANKER_MODEL_NAME):
    """Load a cross-encoder reranker model."""
    from sentence_transformers import CrossEncoder
    print(f"Loading reranker: {model_name}")
    return CrossEncoder(model_name)


def detect_device() -> str:
    """Return the best available torch device string."""
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


# ---------------------------------------------------------------------------
# Display helper
# ---------------------------------------------------------------------------

def print_results(
    entry: WordEntry,
    results: dict[str, list[dict]],
    trans_cache: dict[str, str] | None = None,
) -> None:
    """Pretty-print ranked sentences for one word entry."""
    print(f"\n{'─' * 80}")
    print(f"  {entry.word} [{entry.pinyin}]")
    print(f"  {len(entry.raw_senses)} raw → {len(entry.merged_senses)} merged senses")
    print(f"{'─' * 80}")

    for ms in entry.merged_senses:
        hits = results.get(ms.label, [])
        print(f"\n  ◆ {ms.label}")
        if trans_cache:
            for member in ms.members:
                cache_key = f"{entry.word}|{entry.pinyin}|{member}"
                zh = trans_cache.get(cache_key)
                if zh and zh != member:
                    print(f"    {member}  →  {zh}")
                else:
                    print(f"    {member}")
        elif len(ms.members) > 1:
            print(f"    (merged: {' / '.join(ms.members)})")
        for h in hits:
            print(f"    [{h['rank']}] (sim={h['score']:.3f}) {h['marked']}")
        if not hits and results:
            print("    (no sentences found)")
