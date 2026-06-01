#!/usr/bin/env python3
"""
Build span-scorer training dataset.

Subcommands:
    gen-tasks corpora  — Scan wiki/subs for ambiguous sentences, write LLM batches
    gen-tasks icwb2    — Mine hard cases from ICWB2 MSR dataset
    assemble           — Build final dataset from all sources

Data layout (ml/data/span_scorer/):
    auto_matched/          — ICWB2 exact matches (greedy = gold, auto-labeled)
    llm_tasks_corpora/     — wiki/subs LLM task batches
    llm_results_corpora/   — wiki/subs LLM results
    llm_tasks_icwb2/       — ICWB2 mismatch LLM task batches
    llm_results_icwb2/     — ICWB2 mismatch LLM results

Usage:
    python build_dataset.py gen-tasks corpora      # harvest wiki/subs
    python build_dataset.py gen-tasks icwb2        # harvest ICWB2 MSR
    python build_dataset.py assemble               # combine all into JSONL
"""
import argparse
import csv
import json
import multiprocessing as mp
import os
import random
import re
import shutil
import time
from collections import defaultdict
from pathlib import Path

from span_scorer import (
    CedictTrie,
    CEDICT_RE,
    enumerate_cedict_spans,
    parse_segmentation,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).parent.parent.parent  # ml/
CEDICT_PATH = _ROOT / "data" / "cedict_ts.u8"
DATA_DIR = _ROOT / "data" / "span_scorer"
TASKS_DIR_CORPORA = DATA_DIR / "llm_tasks_corpora"
RESULTS_DIR_CORPORA = DATA_DIR / "llm_results_corpora"
TASKS_DIR_ICWB2 = DATA_DIR / "llm_tasks_icwb2"
RESULTS_DIR_ICWB2 = DATA_DIR / "llm_results_icwb2"
AUTO_MATCHED_DIR = DATA_DIR / "auto_matched"
FINAL_OUTPUT = _ROOT / "data" / "span_scorer_dataset.jsonl"
MERGED_SENSES_PATH = _ROOT / "data" / "entries_after_merging.json"
ICWB2_DATA_DIR = _ROOT / "data" / "icwb2-data"

# Sentence length bounds (design doc sweet spot: 15-40 chars) = 15
MAX_SENT_LEN = 40
# Target conflict points per sentence
MIN_CONFLICTS = 3
MAX_CONFLICTS = 8
# Max sentences per conflict pattern to avoid skew
MAX_PER_PATTERN = 5
# Max sentences per individual conflict zone (word set at one position).
# Relaxed from 10 — coverage-aware selection handles diversity, this is
# just a safety valve against extreme zone repetition.
MAX_PER_ZONE = 50
# Per-source harvested sentences targets
WIKI_TARGET = 40000
SUBS_TARGET = 20000


def load_merged_senses(path: Path = MERGED_SENSES_PATH) -> dict[str, list[dict]]:
    """Load WSD sense-merged entries, keyed by simplified word.

    Returns {word: [entry, ...]} where each entry has pinyin, clusters,
    and trivial_senses. Multiple entries per word = different pronunciations.
    """
    lookup: dict[str, list[dict]] = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        entries = json.load(f)
    for e in entries:
        lookup[e["word"]].append(e)
    return lookup


def format_merged_defs(word: str, merged: dict[str, list[dict]]) -> str:
    """Format merged sense clusters for a word into a compact string.

    Shows all pronunciations and all sense clusters, using LLM-generated
    cluster labels when available, falling back to raw senses.
    """
    entries = merged.get(word, [])
    if not entries:
        return ""
    parts = []
    for e in entries:
        pinyin = e["pinyin"]
        cluster_strs = []
        for c in e.get("clusters", []):
            # Prefer LLM-generated summary label if available
            if "en" in c:
                cluster_strs.append(c["en"])
            else:
                # Join raw senses, truncate if too long
                joined = "; ".join(c["senses"])
                if len(joined) > 60:
                    joined = joined[:57] + "..."
                cluster_strs.append(joined)
        for s in e.get("trivial_senses", []):
            cluster_strs.append(s)
        if cluster_strs:
            senses_str = " / ".join(cluster_strs)
            parts.append(f"[{pinyin}] {senses_str}")
    return f"{word} " + " | ".join(parts) if parts else ""


def build_cedict_trie(cedict_path: Path = CEDICT_PATH) -> CedictTrie:
    """Parse CEDICT and build a trie of simplified forms."""
    trie = CedictTrie()
    with open(cedict_path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            m = CEDICT_RE.match(line.strip())
            if not m:
                continue
            trie.insert(m.group(2))  # simplified form
    return trie


# ---------------------------------------------------------------------------
# Conflict analysis
# ---------------------------------------------------------------------------

def find_conflicts(text: str, trie: CedictTrie) -> list[dict]:
    """Find positions where 2+ multi-char CEDICT spans truly overlap.

    Two multi-char spans conflict when they share at least one character
    position.  We group all transitively-overlapping spans into a single
    conflict zone so cascading ambiguities (e.g. 还好/好赌/赌博) are
    captured together.

    No frequency filtering — every CEDICT entry is a legitimate candidate
    and only the LLM can decide what's correct in context.
    """
    # Collect all multi-char spans: (start, end_exclusive, word)
    all_spans: list[tuple[int, int, str]] = []
    for i in range(len(text)):
        for w in trie.get_words_at(text, i):
            if len(w) >= 2:
                all_spans.append((i, i + len(w), w))

    if not all_spans:
        return []

    # Sort by start then by end so we can sweep left-to-right
    all_spans.sort()

    # Group into conflict zones: spans that transitively overlap
    conflicts = []
    zone_start, zone_end = all_spans[0][0], all_spans[0][1]
    zone_words = [all_spans[0][2]]

    for s, e, w in all_spans[1:]:
        if s < zone_end:
            # Overlaps current zone — extend it
            zone_end = max(zone_end, e)
            zone_words.append(w)
        else:
            # No overlap — flush previous zone
            if len(zone_words) >= 2:
                conflicts.append({"pos": zone_start, "words": zone_words})
            zone_start, zone_end = s, e
            zone_words = [w]

    # Flush last zone
    if len(zone_words) >= 2:
        conflicts.append({"pos": zone_start, "words": zone_words})

    return conflicts


def conflict_pattern_key(conflicts: list[dict]) -> str:
    """Create a dedup key from the set of overlapping word pairs."""
    pairs = set()
    for c in conflicts:
        words = sorted(c["words"])
        for i in range(len(words)):
            for j in range(i + 1, len(words)):
                pairs.add((words[i], words[j]))
    return "|".join(f"{a}/{b}" for a, b in sorted(pairs))


def conflict_zone_keys(conflicts: list[dict]) -> list[str]:
    """Create a dedup key per conflict zone (frozenset of words)."""
    return [",".join(sorted(c["words"])) for c in conflicts]


def score_sentence(conflicts: list[dict], sent_len: int) -> float:
    """Score a sentence by ambiguity density (per character) and diversity."""
    if not conflicts or sent_len == 0:
        return 0.0
    n_conflicts = len(conflicts)
    pairs = set()
    for c in conflicts:
        words = sorted(c["words"])
        for i in range(len(words)):
            for j in range(i + 1, len(words)):
                pairs.add((words[i], words[j]))
    raw = n_conflicts * 0.6 + len(pairs) * 0.4
    return raw / sent_len  # normalize by length so short dense sentences compete fairly


# Regex: match sentences that start at document start OR right after
# sentence-ending punctuation (。！？), and end with sentence-ending punct.
# This ensures we only get proper sentences, not mid-sentence fragments.
_SENT_RE = re.compile(
    r'(?:^|(?<=[。！？\n]))'   # must be at start of text or after a terminator
    r'\s*'                      # optional whitespace
    r'([\u4e00-\u9fff]'         # first char must be CJK
    r'[^\n。！？]*'             # anything except terminators
    r'[。！？])'                # ends with sentence-ending punct
)


def split_sentences(text: str) -> list[str]:
    """Extract proper Chinese sentences terminated by 。！？.

    Only extracts sentences that begin at document start or after a
    sentence terminator — never mid-sentence fragments.
    """
    results = []
    for m in _SENT_RE.finditer(text):
        sent = m.group(1).strip()
        if sent and '\u4e00' <= sent[0] <= '\u9fff':
            results.append(sent)
    return results


# ---------------------------------------------------------------------------
# Parallel sentence processing
# ---------------------------------------------------------------------------

# Per-worker global — initialized once per process via _init_worker
_worker_trie: CedictTrie | None = None
_worker_t2s = None

_QUOTE_PAIRS = [
    ("\u201c", "\u201d"),
    ("\u2018", "\u2019"),
    ("\u300c", "\u300d"),
    ("\u300e", "\u300f"),
    ("\uff08", "\uff09"),
    ("(", ")"),
]


def _init_worker(cedict_path: str) -> None:
    """Initialize trie and OpenCC converter in each worker process."""
    global _worker_trie, _worker_t2s
    from opencc import OpenCC
    _worker_trie = build_cedict_trie(Path(cedict_path))
    _worker_t2s = OpenCC("t2s")


def _filter_and_score(sent: str) -> dict | None:
    """Filter and score a single sentence. Pure function (no shared state).

    Returns a candidate dict or None if the sentence is rejected.
    """
    trie = _worker_trie
    t2s = _worker_t2s
    sent = t2s.convert(sent)
    sent = sent.replace(" ", "").replace("\u3000", "")
    # Strip invisible Unicode control characters (LRM, RLM, ZWS, ZWNJ, ZWJ,
    # BOM/ZWNBSP, soft hyphen, bidi marks) — these break 1:1 char↔token mapping
    sent = re.sub(r'[\u200b-\u200f\u202a-\u202e\ufeff\u00ad\u2028\u2029]', '', sent)
    if re.search(r'[a-zA-Z]', sent):
        return None
    sent = re.sub(r'[(\uff08][^)\uff09]{0,2}[)\uff09]', '', sent)
    if '"' in sent or "'" in sent:
        return None
    sent = sent.strip()
    if len(sent) < MIN_SENT_LEN or len(sent) > MAX_SENT_LEN:
        return None
    for open_q, close_q in _QUOTE_PAIRS:
        if sent.count(open_q) != sent.count(close_q):
            return None
        if open_q + close_q in sent:
            return None
    conflicts = find_conflicts(sent, trie)
    n_conflicts = len(conflicts)
    if n_conflicts < MIN_CONFLICTS or n_conflicts > MAX_CONFLICTS:
        return None
    # Collect all multi-char CEDICT words present in this sentence
    cedict_words: set[str] = set()
    for i in range(len(sent)):
        for w in trie.get_words_at(sent, i):
            if len(w) >= 2:
                cedict_words.add(w)
    return {
        "sentence": sent,
        "conflicts": conflicts,
        "n_conflicts": n_conflicts,
        "ambiguity_score": score_sentence(conflicts, len(sent)),
        "cedict_words": cedict_words,
    }


def _process_batch(batch: list[tuple[str, str]]) -> list[tuple[str, dict]]:
    """Process a batch of (source, raw_sentence) pairs in a worker.

    Returns list of (source, candidate_dict) for sentences that pass filters.
    """
    results = []
    for source, raw_sent in batch:
        item = _filter_and_score(raw_sent)
        if item is not None:
            item["source"] = source
            results.append((source, item))
    return results


def _select_with_coverage(
    source_pools: list[tuple[str, list[dict], int]],
    all_cedict_words: set[str],
    coverage_weight: float = 0.35,
    passes: int = 5,
) -> dict[str, list[dict]]:
    """Select sentences from multiple source pools with shared coverage tracking.

    Interleaves chunk-based selection across sources so coverage gained by
    one source benefits the next. Each pass picks a chunk from each source
    in round-robin order, re-scoring against the shared coverage state.

    Args:
        source_pools: list of (name, candidates, target) per source
        all_cedict_words: full set of multi-char CEDICT words (for stats)
        coverage_weight: blend weight for coverage vs ambiguity (0-1)
        passes: number of selection passes (more = better coverage adaptation)

    Returns:
        dict mapping source name to selected items
    """
    # Shared state across all sources
    covered_words: set[str] = set()
    seen_sentences: set[str] = set()

    # Per-source state
    sources: dict[str, dict] = {}
    for name, candidates, target in source_pools:
        candidates.sort(key=lambda x: x["ambiguity_score"], reverse=True)
        sources[name] = {
            "remaining": list(candidates),
            "selected": [],
            "target": target,
            "chunk_size": max(target // passes, 1),
            "pattern_counts": defaultdict(int),
            "zone_counts": defaultdict(int),
        }

    for pass_idx in range(passes):
        for name, candidates, target in source_pools:
            s = sources[name]
            if len(s["selected"]) >= s["target"]:
                continue
            this_chunk = min(s["chunk_size"], s["target"] - len(s["selected"]))

            # Score remaining against shared coverage
            scored = []
            for item in s["remaining"]:
                words = item["cedict_words"]
                n_words = len(words)
                if n_words > 0:
                    uncovered = len(words - covered_words)
                    coverage_bonus = uncovered / n_words
                else:
                    coverage_bonus = 0.0
                blended = (
                    item["ambiguity_score"] * (1 - coverage_weight)
                    + coverage_bonus * coverage_weight
                )
                scored.append((blended, item))

            scored.sort(key=lambda x: x[0], reverse=True)

            next_remaining = []
            picked = 0
            for _, item in scored:
                if picked >= this_chunk:
                    next_remaining.append(item)
                    continue
                sent = item["sentence"]
                if sent in seen_sentences:
                    continue
                pattern = conflict_pattern_key(item["conflicts"])
                if s["pattern_counts"][pattern] >= MAX_PER_PATTERN:
                    next_remaining.append(item)
                    continue
                zone_keys = conflict_zone_keys(item["conflicts"])
                if any(s["zone_counts"][zk] >= MAX_PER_ZONE for zk in zone_keys):
                    next_remaining.append(item)
                    continue
                # Accept
                seen_sentences.add(sent)
                s["pattern_counts"][pattern] += 1
                for zk in zone_keys:
                    s["zone_counts"][zk] += 1
                covered_words.update(item["cedict_words"])
                s["selected"].append(item)
                picked += 1

            s["remaining"] = next_remaining

    # Print per-source stats
    results = {}
    for name, _, _ in source_pools:
        s = sources[name]
        results[name] = s["selected"]
        # Per-source coverage (just this source's contribution)
        src_covered = set()
        for item in s["selected"]:
            src_covered.update(item["cedict_words"])
        src_pct = len(src_covered) / len(all_cedict_words) * 100 if all_cedict_words else 0
        print(f"    {name}: {len(s['selected'])} selected, "
              f"{len(src_covered)} unique words ({src_pct:.1f}% of CEDICT)")

    coverage_pct = len(covered_words) / len(all_cedict_words) * 100 if all_cedict_words else 0
    print(f"    Combined CEDICT coverage: {len(covered_words)}/{len(all_cedict_words)} "
          f"multi-char words ({coverage_pct:.1f}%)")
    return results


# ---------------------------------------------------------------------------
# gen-tasks: harvest corpora + write LLM batches in one step
# ---------------------------------------------------------------------------

def gen_tasks_corpora(args: argparse.Namespace) -> None:
    """Scan wiki/subs corpora for ambiguous sentences and write LLM task batches."""
    from datasets import load_dataset
    from tqdm import tqdm

    print("Building CEDICT trie...")
    trie = build_cedict_trie(args.cedict_path)
    print(f"  {len(trie.words)} unique words in trie")

    # All multi-char CEDICT words — the universe we want to cover
    all_cedict_multichar = {w for w in trie.words if len(w) >= 2}
    print(f"  {len(all_cedict_multichar)} multi-char words to cover")

    print("Loading merged senses...")
    merged_senses = load_merged_senses(args.merged_senses_path)
    print(f"  {len(merged_senses)} words with merged senses")

    # --- Harvest phase: scan full corpora, parallel filter + score ---
    print("\nLoading corpora...")
    wiki = load_dataset("wikimedia/wikipedia", "20231101.zh", split="train")
    subs = load_dataset("FradSer/OpenSubtitles-en-zh-cn-20m", split="train")

    print("\nExtracting raw sentences...")
    raw_wiki: list[str] = []
    raw_subs: list[str] = []
    wiki_buf: list[str] = []

    for row in tqdm(wiki, desc="Wiki articles", unit="doc"):
        wiki_buf.extend(split_sentences(row["text"]))
        while len(wiki_buf) > 1000:
            raw_wiki.extend(wiki_buf[:1000])
            wiki_buf = wiki_buf[1000:]
    raw_wiki.extend(wiki_buf)

    for row in tqdm(subs, desc="Subtitles", unit="line"):
        raw_subs.append(row["target"].strip())

    print(f"  Raw sentences: wiki={len(raw_wiki)}, subs={len(raw_subs)}")

    # Build work batches: list of (source, sentence) tuples, chunked
    CHUNK_SIZE = 2000
    work_batches: list[list[tuple[str, str]]] = []
    chunk: list[tuple[str, str]] = []
    wi, si = 0, 0
    while wi < len(raw_wiki) or si < len(raw_subs):
        if wi < len(raw_wiki):
            chunk.append(("wiki", raw_wiki[wi]))
            wi += 1
        if si < len(raw_subs):
            chunk.append(("subs", raw_subs[si]))
            si += 1
        if len(chunk) >= CHUNK_SIZE:
            work_batches.append(chunk)
            chunk = []
    if chunk:
        work_batches.append(chunk)

    del raw_wiki, raw_subs

    # Parallel filter + score
    n_workers = min(args.workers or os.cpu_count() or 1, len(work_batches))
    print(f"\nParallel scan: {len(work_batches)} chunks across {n_workers} workers...")
    t0 = time.time()

    all_candidates_wiki: list[dict] = []
    all_candidates_subs: list[dict] = []

    with mp.Pool(
        processes=n_workers,
        initializer=_init_worker,
        initargs=(str(args.cedict_path),),
    ) as pool:
        for batch_results in tqdm(
            pool.imap_unordered(_process_batch, work_batches),
            total=len(work_batches),
            desc="Scanning",
            unit="chunk",
        ):
            for source, item in batch_results:
                if source == "wiki":
                    all_candidates_wiki.append(item)
                else:
                    all_candidates_subs.append(item)

    elapsed = time.time() - t0
    print(f"  Scan done in {elapsed:.0f}s")
    print(f"  Candidates: wiki={len(all_candidates_wiki)}, subs={len(all_candidates_subs)}")

    # Exact dedup across sources (parallel workers may find the same sentence)
    seen: set[str] = set()
    def dedup(candidates: list[dict]) -> list[dict]:
        deduped = []
        for item in candidates:
            if item["sentence"] not in seen:
                seen.add(item["sentence"])
                deduped.append(item)
        return deduped

    all_candidates_wiki = dedup(all_candidates_wiki)
    all_candidates_subs = dedup(all_candidates_subs)
    print(f"  After dedup: wiki={len(all_candidates_wiki)}, subs={len(all_candidates_subs)}")

    # --- Selection phase: coverage-aware, interleaved across sources ---
    print(f"\nSelecting with coverage-aware scoring (interleaved)...")
    print(f"  wiki: {len(all_candidates_wiki)} candidates → {args.wiki_target} target")
    print(f"  subs: {len(all_candidates_subs)} candidates → {args.subs_target} target")
    results = _select_with_coverage(
        source_pools=[
            ("wiki", all_candidates_wiki, args.wiki_target),
            ("subs", all_candidates_subs, args.subs_target),
        ],
        all_cedict_words=all_cedict_multichar,
        coverage_weight=args.coverage_weight,
    )
    selected_wiki = results["wiki"]
    selected_subs = results["subs"]
    print(f"  Selected: wiki={len(selected_wiki)}, subs={len(selected_subs)}")

    # Merge and shuffle so batches have a mix of both sources
    harvested = selected_wiki + selected_subs
    random.seed(42)
    random.shuffle(harvested)
    print(f"    total: {len(harvested)} sentences")

    # --- Write LLM task batches ---
    if args.tasks_dir.exists():
        # Check if there are existing LLM results that reference these tasks
        results_dir = RESULTS_DIR_CORPORA
        if results_dir.exists() and any(results_dir.glob("batch_*.txt")):
            print(f"\n  WARNING: {results_dir} contains existing LLM results.")
            print(f"  Re-generating tasks will invalidate them (different sentences).")
            resp = input("  Delete existing tasks and results? [y/N] ").strip().lower()
            if resp != "y":
                print("  Aborted.")
                return
            shutil.rmtree(results_dir)
            print(f"  Deleted {results_dir}")
        shutil.rmtree(args.tasks_dir)
    args.tasks_dir.mkdir(parents=True)

    num_batches = 0
    for batch_start in range(0, len(harvested), args.batch_size):
        batch = harvested[batch_start : batch_start + args.batch_size]
        num_batches += 1
        tasks = []
        for item in batch:
            sent = item["sentence"]
            # Compact span format: only positions with multi-char words
            # Format: "pos:word1/word2 pos:word1/word2/word3 ..."
            # Single chars are always valid and omitted to save tokens.
            span_parts = []
            for i in range(len(sent)):
                words = trie.get_words_at(sent, i)
                multi = [w for w in words if len(w) >= 2]
                if multi:
                    span_parts.append(f"{i}:{'/'.join(multi)}")

            # Definitions at conflict zones only — helps the LLM
            # disambiguate overlapping spans by showing meaning.
            # Uses WSD sense-merged entries for richer, cleaner glosses.
            conflicts = item["conflicts"]
            conflict_words: set[str] = set()
            for c in conflicts:
                conflict_words.update(c["words"])
            def_parts = []
            for w in sorted(conflict_words):
                formatted = format_merged_defs(w, merged_senses)
                if formatted:
                    def_parts.append(formatted)

            tasks.append((sent, " ".join(span_parts), "; ".join(def_parts)))

        path = args.tasks_dir / f"batch_{num_batches:04d}.tsv"
        with open(path, "w", encoding="utf-8") as f:
            f.write("sentence\tspans\tdefs\n")
            for sent, spans, defs in tasks:
                f.write(f"{sent}\t{spans}\t{defs}\n")

    print(f"  Batches: {num_batches} files ({args.batch_size} per batch)")
    print(f"  Output: {args.tasks_dir}/")


# ---------------------------------------------------------------------------
# harvest-icwb2: mine hard cases from ICWB2 human-annotated data
# ---------------------------------------------------------------------------

def _greedy_longest(text: str, trie: CedictTrie) -> list[str]:
    """Greedy longest-match CEDICT segmentation."""
    segments = []
    i = 0
    while i < len(text):
        words = trie.get_words_at(text, i)
        if words:
            longest = max(words, key=len)
            segments.append(longest)
            i += len(longest)
        else:
            segments.append(text[i])
            i += 1
    return segments


def _classify_mismatch(gold_words: list[str], greedy_words: list[str]) -> str:
    """Classify mismatch type using boundary sets."""
    if gold_words == greedy_words:
        return "match"
    gold_bounds = set()
    pos = 0
    for w in gold_words:
        pos += len(w)
        gold_bounds.add(pos)
    greedy_bounds = set()
    pos = 0
    for w in greedy_words:
        pos += len(w)
        greedy_bounds.add(pos)
    only_gold = gold_bounds - greedy_bounds
    only_greedy = greedy_bounds - gold_bounds
    if only_gold and not only_greedy:
        return "oversplit_only"
    if only_greedy and not only_gold:
        return "undersplit_only"
    return "genuine_mismatch"


def _write_task_batches(
    sentences: list[str],
    trie: CedictTrie,
    merged_senses: dict,
    out_dir: Path,
    batch_size: int,
) -> int:
    """Write sentences as LLM task TSV batches. Returns number of batches."""
    out_dir.mkdir(parents=True, exist_ok=True)
    num_batches = 0
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i : i + batch_size]
        num_batches += 1
        tasks = []
        for sent in batch:
            span_parts = []
            for j in range(len(sent)):
                words = trie.get_words_at(sent, j)
                multi = [w for w in words if len(w) >= 2]
                if multi:
                    span_parts.append(f"{j}:{'/'.join(multi)}")
            conflicts = find_conflicts(sent, trie)
            conflict_words: set[str] = set()
            for c in conflicts:
                conflict_words.update(c["words"])
            def_parts = []
            for w in sorted(conflict_words):
                formatted = format_merged_defs(w, merged_senses)
                if formatted:
                    def_parts.append(formatted)
            tasks.append((sent, " ".join(span_parts), "; ".join(def_parts)))
        path = out_dir / f"batch_{num_batches:04d}.tsv"
        with open(path, "w", encoding="utf-8") as f:
            f.write("sentence\tspans\tdefs\n")
            for sent, spans, defs in tasks:
                f.write(f"{sent}\t{spans}\t{defs}\n")
    return num_batches


def harvest_icwb2(args: argparse.Namespace) -> None:
    """Mine hard cases from ICWB2 human-annotated CWS datasets.

    Compares each sentence to greedy-longest-CEDICT and splits into:
    - Exact matches: written as auto-labeled task/result pairs
    - Genuine mismatches: scored by ambiguity density + coverage, top N
      selected and written as LLM task batches for re-segmentation
    """
    print("Building CEDICT trie...")
    trie = build_cedict_trie(args.cedict_path)
    print(f"  {len(trie.words)} words in trie")

    all_cedict_multichar = {w for w in trie.words if len(w) >= 2}

    print("Loading merged senses...")
    merged_senses = load_merged_senses(args.merged_senses_path)
    print(f"  {len(merged_senses)} words with merged senses")

    # MSR only — PKU has annotation inconsistencies
    datasets = [
        ("msr_train", args.icwb2_data_dir / "training" / "msr_training.utf8"),
        ("msr_test", args.icwb2_data_dir / "gold" / "msr_test_gold.utf8"),
    ]

    matched_segs: list[tuple[str, str]] = []  # (sentence, pipe-delimited seg)
    mismatch_candidates: list[dict] = []
    seen: set[str] = set()
    counts: dict[str, int] = defaultdict(int)

    for name, path in datasets:
        if not path.exists():
            print(f"  Skipping {name}: not found")
            continue
        print(f"\nProcessing {name}...")
        ds_counts: dict[str, int] = defaultdict(int)
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                gold_words = [w for w in re.split(r'\s+', line) if w]
                if not gold_words:
                    continue
                sentence = "".join(gold_words)
                if len(sentence) < 5 or sentence in seen:
                    continue
                seen.add(sentence)
                greedy_words = _greedy_longest(sentence, trie)
                cat = _classify_mismatch(gold_words, greedy_words)
                ds_counts[cat] += 1
                counts[cat] += 1
                if cat == "match":
                    matched_segs.append((sentence, "|".join(greedy_words)))
                elif cat == "genuine_mismatch":
                    # Score by ambiguity density (same formula as corpora)
                    conflicts = find_conflicts(sentence, trie)
                    cedict_words: set[str] = set()
                    for i in range(len(sentence)):
                        for w in trie.get_words_at(sentence, i):
                            if len(w) >= 2:
                                cedict_words.add(w)
                    mismatch_candidates.append({
                        "sentence": sentence,
                        "conflicts": conflicts,
                        "ambiguity_score": score_sentence(conflicts, len(sentence)),
                        "cedict_words": cedict_words,
                    })
        for cat in ["match", "genuine_mismatch", "oversplit_only", "undersplit_only"]:
            print(f"  {cat}: {ds_counts[cat]}")

    print(f"\nTotals:")
    print(f"  Exact matches (auto-labeled): {len(matched_segs)}")
    print(f"  Genuine mismatches (candidates): {len(mismatch_candidates)}")
    for cat in ["oversplit_only", "undersplit_only"]:
        print(f"  {cat} (skipped): {counts[cat]}")

    # Select top N mismatches by ambiguity density + coverage
    target = args.icwb2_target
    print(f"\nSelecting top {target} mismatches (coverage-aware)...")
    selected = _select_with_coverage(
        source_pools=[("icwb2", mismatch_candidates, target)],
        all_cedict_words=all_cedict_multichar,
        coverage_weight=args.coverage_weight,
    )
    mismatch_sents = [item["sentence"] for item in selected["icwb2"]]
    print(f"  Selected {len(mismatch_sents)} mismatch sentences")

    # --- Write output ---
    auto_dir = AUTO_MATCHED_DIR
    tasks_dir = TASKS_DIR_ICWB2
    results_dir = RESULTS_DIR_ICWB2

    # Clean existing output dirs if they exist
    for d in [auto_dir, tasks_dir, results_dir]:
        if d.exists():
            if not hasattr(args, '_confirmed'):
                resp = input(f"\nICWB2 output dirs exist. Delete and regenerate? [y/N] ").strip().lower()
                if resp != "y":
                    print("Aborted.")
                    return
                args._confirmed = True
            shutil.rmtree(d)

    results_dir.mkdir(parents=True)

    # Write exact matches as paired task/result files
    print(f"\nWriting {len(matched_segs)} auto-matched sentences...")
    auto_dir.mkdir(parents=True)
    num = 0
    for i in range(0, len(matched_segs), args.batch_size):
        batch = matched_segs[i : i + args.batch_size]
        num += 1
        fname = f"batch_{num:04d}"
        with open(auto_dir / f"{fname}.tsv", "w", encoding="utf-8") as f:
            f.write("sentence\tspans\tdefs\n")
            for sent, _ in batch:
                f.write(f"{sent}\t\t\n")
        with open(auto_dir / f"{fname}.txt", "w", encoding="utf-8") as f:
            for _, seg in batch:
                f.write(seg + "\n")
    print(f"  Wrote {num} batch files to {auto_dir}/")

    # Write mismatches as LLM task batches
    print(f"\nWriting {len(mismatch_sents)} mismatch sentences as LLM tasks...")
    num = _write_task_batches(
        mismatch_sents, trie, merged_senses, tasks_dir, args.batch_size,
    )
    print(f"  Wrote {num} batch files to {tasks_dir}/")
    print(f"  LLM results go in {results_dir}/")
    print(f"\nDone. Run LLM labeling on the tasks, then assemble.")


# ---------------------------------------------------------------------------
# assemble: build dataset from all sources
# ---------------------------------------------------------------------------


def _load_batch_pairs(tasks_dir: Path, results_dir: Path) -> list[dict]:
    """Load sentence/segmentation pairs from matched task/result batch files.

    Reads batch_*.txt from results_dir, pairs each line with the
    corresponding sentence from the matching batch_*.tsv in tasks_dir.
    """
    if not results_dir.exists():
        return []
    pairs = []
    for result_path in sorted(results_dir.glob("batch_*.txt")):
        task_path = tasks_dir / result_path.with_suffix(".tsv").name
        if not task_path.exists():
            print(f"  Warning: no task file for {result_path.name}, skipping")
            continue
        sentences = []
        with open(task_path, encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            next(reader, None)
            for row in reader:
                if row:
                    sentences.append(row[0])
        with open(result_path, encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]
        if len(lines) < len(sentences):
            print(f"  Warning: {result_path.name} has {len(lines)}/{len(sentences)} lines (incomplete batch)")
        for i, sent in enumerate(sentences):
            if i >= len(lines):
                break
            pairs.append({"sentence": sent, "segmentation": lines[i]})
    return pairs


def assemble(args: argparse.Namespace) -> None:
    """Build final span-scorer dataset from all segmentation sources."""
    print("Building CEDICT trie...")
    trie = build_cedict_trie(args.cedict_path)

    all_sentences = []

    # Source 1: Corpora LLM-labeled data (wiki/subs)
    print(f"\nLoading corpora LLM data...")
    corpora_pairs = _load_batch_pairs(TASKS_DIR_CORPORA, RESULTS_DIR_CORPORA)
    print(f"  {len(corpora_pairs)} sentences from corpora")
    all_sentences.extend(corpora_pairs)

    # Source 2: ICWB2 auto-matched (greedy = gold)
    if AUTO_MATCHED_DIR.exists():
        print(f"\nLoading ICWB2 auto-matched...")
        auto_pairs = _load_batch_pairs(AUTO_MATCHED_DIR, AUTO_MATCHED_DIR)
        print(f"  {len(auto_pairs)} sentences from auto-matched")
        all_sentences.extend(auto_pairs)

    # Source 3: ICWB2 LLM-labeled mismatches
    if RESULTS_DIR_ICWB2.exists():
        print(f"\nLoading ICWB2 LLM-labeled mismatches...")
        icwb2_pairs = _load_batch_pairs(TASKS_DIR_ICWB2, RESULTS_DIR_ICWB2)
        print(f"  {len(icwb2_pairs)} sentences from ICWB2 mismatches")
        all_sentences.extend(icwb2_pairs)

    print(f"\nTotal: {len(all_sentences)} segmented sentences")

    # Dedup by sentence text
    seen = set()
    deduped = []
    for item in all_sentences:
        if item["sentence"] not in seen:
            seen.add(item["sentence"])
            deduped.append(item)
    if len(deduped) < len(all_sentences):
        print(f"  Deduped: {len(all_sentences)} -> {len(deduped)}")
    all_sentences = deduped

    total_examples = 0
    total_positions = 0
    total_ambiguous = 0
    skipped_mismatch = 0
    gold_not_in_cedict = 0

    # Regex to strip invisible Unicode control characters
    _INVIS_RE = re.compile(r'[\u200b-\u200f\u202a-\u202e\ufeff\u00ad\u2028\u2029]')

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as out_f:
        for item in all_sentences:
            sentence = _INVIS_RE.sub('', item["sentence"])
            segmentation = _INVIS_RE.sub('', item["segmentation"])
            seg_words = parse_segmentation(segmentation)
            if "".join(seg_words) != sentence:
                skipped_mismatch += 1
                continue

            pos = 0
            cedict_spans_at = enumerate_cedict_spans(sentence, trie)
            for w in seg_words:
                start = pos
                pos += len(w)
                total_positions += 1
                candidates = cedict_spans_at.get(start, [])
                if not candidates:
                    continue
                # Ensure gold word is in the candidate list — it might
                # be a single char not returned by the trie, or a valid
                # CEDICT entry starting at a different position.
                if w not in candidates:
                    gold_not_in_cedict += 1
                    candidates = candidates + [w]
                if len(candidates) < 2:
                    continue
                total_ambiguous += 1
                example = {
                    "sentence": sentence,
                    "start": start,
                    "gold_word": w,
                    "candidates": candidates,
                }
                out_f.write(json.dumps(example, ensure_ascii=False) + "\n")
                total_examples += 1

    print(f"\nAssembly complete:")
    print(f"  Sentences: {len(all_sentences)}, skipped {skipped_mismatch} (mismatch)")
    print(f"  Positions: {total_positions}, ambiguous: {total_ambiguous}")
    if gold_not_in_cedict:
        print(f"  Gold word not in CEDICT candidates: {gold_not_in_cedict} (added as fallback)")
    print(f"  Training examples (ambiguous only): {total_examples}")
    print(f"  Wrote {args.output}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Build span-scorer training dataset")
    p.add_argument("--cedict-path", type=Path, default=CEDICT_PATH)
    sub = p.add_subparsers(dest="cmd", required=True)

    # gen-tasks <source>
    g = sub.add_parser("gen-tasks", help="Harvest sentences + create LLM batches")
    g.add_argument("source", choices=["corpora", "icwb2"],
                   help="'corpora' = wiki/subs, 'icwb2' = ICWB2 MSR dataset")
    g.add_argument("--batch-size", type=int, default=10,
                   help="Sentences per LLM batch file")
    g.add_argument("--merged-senses-path", type=Path, default=MERGED_SENSES_PATH)
    # corpora-specific
    g.add_argument("--wiki-target", type=int, default=WIKI_TARGET)
    g.add_argument("--subs-target", type=int, default=SUBS_TARGET)
    g.add_argument("--workers", type=int, default=None)
    g.add_argument("--coverage-weight", type=float, default=0.35)
    # icwb2-specific
    g.add_argument("--icwb2-data-dir", type=Path, default=ICWB2_DATA_DIR)
    g.add_argument("--icwb2-target", type=int, default=20000,
                   help="Max mismatch sentences to select for LLM labeling")

    # assemble
    a = sub.add_parser("assemble", help="Build dataset from all sources")
    a.add_argument("--output", type=Path, default=FINAL_OUTPUT)

    args = p.parse_args()
    if args.cmd == "gen-tasks":
        if args.source == "corpora":
            args.tasks_dir = TASKS_DIR_CORPORA
            gen_tasks_corpora(args)
        else:
            harvest_icwb2(args)
    else:
        assemble(args)


if __name__ == "__main__":
    main()
