#!/usr/bin/env python3
"""
Build CWS + WSD training data from ebook corpus via LLM annotation.

    python build_dataset.py                          # all steps, default model
    python build_dataset.py 3                        # just step 3
    python build_dataset.py 3 -m claude-opus-4.6 -v  # step 3, opus, verbose
    python build_dataset.py 3 -m deepseek-v4-pro-thinking -c 4 -o results_ds.jsonl
"""
import json
import random
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_ROOT = Path(__file__).parent.parent.parent  # ml/
_DATA = _ROOT / "data" / "dataset_v2"
_TXT = _ROOT / "data" / "txt-books"
_EPUB = _ROOT / "data" / "epub-books"

SENTENCES_PATH = _DATA / "sentences.json"
CWS_TRAINING_PATH = _DATA / "cws_dataset_v2.tsv"
WSD_TSV_PATH = _DATA / "wsd_dataset_v2.tsv"
ENTRIES_PATH = _ROOT / "data" / "entries_after_merging.json"

# Ebooks CWS paths
EBOOKS_CWS_TASKS_PATH = _DATA / "ebooks_cws_tasks.jsonl"
EBOOKS_CWS_RESULTS_PATH = _DATA / "ebooks_cws_results.jsonl"

# ICWB2 CWS paths
ICWB2_FREE_PATH = _DATA / "icwb2_free.jsonl"
ICWB2_CWS_TASKS_PATH = _DATA / "icwb2_cws_tasks.jsonl"
ICWB2_CWS_RESULTS_PATH = _DATA / "icwb2_cws_results.jsonl"

# ICWB2 sampling - None to skip sampling
ICWB2_TARGET_CHARS = None

# Upsampling multipliers (repeat rows to rebalance vs large ICWB2 corpus)
EBOOKS_UPSAMPLE = 5       # ebooks are target domain, heavily underrepresented
WIKI_SUBS_UPSAMPLE = 2    # different domain but Opus-quality labels

# WSD paths
WSD_SEGMENTS_PATH = _DATA / "wsd_segments.jsonl"  # span-scorer segmented sentences for WSD
WSD_TASKS_PATH = _DATA / "wsd_tasks.jsonl"
WSD_RESULTS_PATH = _DATA / "wsd_results.jsonl"

# ---------------------------------------------------------------------------
# Corpus — (path, sampling weight)
# ---------------------------------------------------------------------------

BOOKS: list[tuple[Path, float]] = [
    # Classical — reduced weight, learners may eventually read these
    (_TXT / "红楼梦.txt", 0.2),
    # Early modern
    (_TXT / "周作人文集.txt", 0.4),
    (_TXT / "四世同堂.txt", 1.0),
    # Mid-century
    (_TXT / "冬天里的春天.txt", 1.0),
    # Contemporary literary
    (_TXT / "《白鹿原》全集.txt", 1.0),
    (_TXT / "务虚笔记.txt", 1.0),
    (_TXT / "繁花.txt", 0.3),  # dialectal, reduced weight
    # Essay / dialect
    (_TXT / "俗世奇人.txt", 0.3),  # Tianjin dialect, slightly reduced
    # Nonfiction — main target for learners, high weight
    (_TXT / "万历十五年.txt", 1.5),
    (_TXT / "《常识》梁文道.txt", 1.5),
    # Epub — modern, accessible, high weight
    (_ROOT.parent / "local_test_data" / "我从未如此眷恋人间.epub", 1.5),
    (_EPUB / "看见.epub", 1.5),
    (_EPUB / "沉默的大多数(精排版).epub", 1.5),
    (_EPUB / "【精】三体（全集）.epub", 1.5),
]

TARGET_CHARS = 1_000_000

# ---------------------------------------------------------------------------
# LLM config
# ---------------------------------------------------------------------------

DEEPSEEK_API_KEY_FILE = Path(__file__).parent / "DEEPSEEK_API_DO_NOT_COMMIT.txt"

CWS_MODEL = "claude-opus-4.6"
ICWB2_MODEL = "gpt-4o-mini"
WSD_MODEL = "deepseek-v4-flash"
MAX_CHUNK_CHARS = 64
CWS_TARGET_CHARS = 300000

# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------


def extract():
    """Extract + sample sentences from ebook corpus → sentences.json"""
    from download_books import download_books
    from extract_sentences import extract_book

    download_books()

    raw: list[tuple[dict, float]] = []
    for path, weight in BOOKS:
        if not path.exists():
            print(f"  skip: {path.name} (not found)")
            continue
        raw.append((extract_book(path), weight))

    if not raw:
        sys.exit("No books found. Run download_books.py first.")

    # Weighted sampling
    total_weight = sum(w for _, w in raw)
    char_per_unit = TARGET_CHARS / total_weight
    rng = random.Random(42)
    books = []

    for book, weight in raw:
        target = weight * char_per_unit
        paras = book["paragraphs"]
        total_chars = sum(len(s) for p in paras for s in p)

        if total_chars > target:
            k = max(1, int(len(paras) * target / total_chars))
            idxs = sorted(rng.sample(range(len(paras)), k))
            book["paragraphs"] = [paras[i] for i in idxs]

        n_s = sum(len(p) for p in book["paragraphs"])
        n_c = sum(len(s) for p in book["paragraphs"] for s in p)
        print(f"  {book['source']}: {n_s} sent, {n_c:,} chars")
        books.append(book)

    _DATA.mkdir(parents=True, exist_ok=True)
    SENTENCES_PATH.write_text(json.dumps(books, ensure_ascii=False, indent=2))

    total_s = sum(len(p) for b in books for p in b["paragraphs"])
    total_c = sum(len(s) for b in books for p in b["paragraphs"] for s in p)
    print(f"  → {total_s} sentences, {total_c:,} chars → {SENTENCES_PATH.name}")


def cws_tasks():
    """Build CWS task file from sentences → ebooks_cws_tasks.jsonl"""
    from cws.build_tasks import build_cws_tasks
    build_cws_tasks(SENTENCES_PATH, EBOOKS_CWS_TASKS_PATH, max_chars=MAX_CHUNK_CHARS,
                    target_chars=CWS_TARGET_CHARS)


def cws_llm():
    """LLM-annotate CWS segmentation → ebooks_cws_results.jsonl"""
    from cws.annotate import annotate_cws
    annotate_cws(EBOOKS_CWS_TASKS_PATH, EBOOKS_CWS_RESULTS_PATH, model=CWS_MODEL)


def cws_llm_dry_run():
    """Dry-run: dump prompts + estimate cost (no API calls)"""
    from cws.annotate import annotate_cws
    annotate_cws(EBOOKS_CWS_TASKS_PATH, EBOOKS_CWS_RESULTS_PATH, model=CWS_MODEL, dry_run_mode=True)


def wsd_segment():
    """Segment full ebook corpus with span scorer for WSD → wsd_segments.jsonl"""
    from wsd.segment_for_wsd import segment_corpus
    segment_corpus(SENTENCES_PATH, WSD_SEGMENTS_PATH, max_sentences=wsd_max_sentences)


def wsd_tasks():
    """Build WSD task file from span-scorer segments → wsd_tasks.jsonl"""
    from wsd.build_tasks import build_wsd_tasks
    build_wsd_tasks(WSD_SEGMENTS_PATH, WSD_TASKS_PATH, max_per_word=100)


def wsd_llm():
    """LLM-annotate WSD labels → wsd_results.jsonl"""
    from wsd.annotate import annotate_wsd
    annotate_wsd(WSD_TASKS_PATH, WSD_RESULTS_PATH, model=WSD_MODEL)


def icwb2_extract():
    """Extract ICWB2 → free sentences + LLM tasks"""
    from cws.icwb2 import extract_icwb2
    extract_icwb2(ICWB2_FREE_PATH, ICWB2_CWS_TASKS_PATH, max_chars=MAX_CHUNK_CHARS,
                  target_chars=ICWB2_TARGET_CHARS)


def icwb2_llm():
    """LLM-annotate ICWB2 merge decisions → icwb2_cws_results.jsonl"""
    from cws.annotate_merge import annotate_merges
    annotate_merges(ICWB2_CWS_TASKS_PATH, ICWB2_CWS_RESULTS_PATH, model=ICWB2_MODEL)


def icwb2_llm_dry_run():
    """Dry-run: dump prompts + estimate cost for ICWB2 (no API calls)"""
    from cws.annotate_merge import annotate_merges
    annotate_merges(ICWB2_CWS_TASKS_PATH, ICWB2_CWS_RESULTS_PATH, model=ICWB2_MODEL, dry_run_mode=True)


def assemble_all():
    """Assemble CWS training data from all sources (ebook + ICWB2)"""
    from cws.assemble import assemble_cws
    import json

    # 1. Assemble ebook LLM-annotated data (upsampled)
    print(f"  --- Ebook (LLM-annotated, {EBOOKS_UPSAMPLE}x upsample) ---")
    assemble_cws(EBOOKS_CWS_RESULTS_PATH, CWS_TRAINING_PATH)
    # Upsample by repeating ebook rows
    if EBOOKS_UPSAMPLE > 1:
        with open(CWS_TRAINING_PATH, encoding="utf-8") as f:
            ebook_lines = f.readlines()
        with open(CWS_TRAINING_PATH, "w", encoding="utf-8") as f:
            for line in ebook_lines:
                f.write(line * EBOOKS_UPSAMPLE)
        print(f"  Upsampled {len(ebook_lines):,} ebook rows → {len(ebook_lines) * EBOOKS_UPSAMPLE:,}")

    # 2. Append ICWB2 free sentences (greedy == gold, no LLM needed)
    free_count = 0
    if ICWB2_FREE_PATH.exists():
        print("  --- ICWB2 free sentences ---")
        with open(CWS_TRAINING_PATH, "a", encoding="utf-8") as out:
            for line in ICWB2_FREE_PATH.read_text().splitlines():
                if not line.strip():
                    continue
                r = json.loads(line)
                out.write(f"{r['text']}\t{' '.join(r['segments'])}\n")
                free_count += 1
        print(f"  Appended {free_count:,} ICWB2 free sentences")

    # 3. Append ICWB2 LLM-annotated data
    llm_count = 0
    if ICWB2_CWS_RESULTS_PATH.exists():
        print("  --- ICWB2 LLM-annotated ---")
        with open(CWS_TRAINING_PATH, "a", encoding="utf-8") as out:
            for line in ICWB2_CWS_RESULTS_PATH.read_text().splitlines():
                if not line.strip():
                    continue
                r = json.loads(line)
                out.write(f"{r['text']}\t{' '.join(r['segments'])}\n")
                llm_count += 1
        print(f"  Appended {llm_count:,} ICWB2 LLM-annotated sentences")

    with open(CWS_TRAINING_PATH, encoding="utf-8") as f:
        total = sum(1 for _ in f)
    print(f"  → Total CWS training rows: {total:,} → {CWS_TRAINING_PATH.name}")


def merge_old_span_scorer():
    """Merge wiki/subs data (Opus-annotated) into CWS training data.

    Reads batch pairs from the old span_scorer pipeline (wiki/subs source),
    converts pipe-delimited segmentation to space-delimited, and deduplicates
    against existing cws_dataset_v2.tsv.

    Filters out sentences where any multi-char segment is not a valid CEDICT
    word (e.g. foreign names, numbers, invisible unicode — the old pipeline
    didn't enforce CEDICT-only as strictly).

    On ties (same sentence text), the old Opus-annotated segmentation wins.
    """
    import csv
    import re
    from cws.cedict_lookup import load_cedict_words

    _CJK_RE = re.compile(r'^[\u4e00-\u9fff]+$')

    print("  Loading CEDICT words for validation...")
    word_set, _ = load_cedict_words()

    SPAN_SCORER_DATA = _ROOT / "data" / "span_scorer"
    SOURCES = [
        ("corpora (wiki/subs)", SPAN_SCORER_DATA / "llm_tasks_corpora", SPAN_SCORER_DATA / "llm_results_corpora"),
        # Old ICWB2 sources removed — superseded by new full-corpora merge pipeline (steps 4-6)
    ]

    def load_batch_pairs(tasks_dir: Path, results_dir: Path) -> list[tuple[str, str]]:
        """Load (sentence, segmentation) pairs from batch files."""
        if not results_dir.exists():
            return []
        pairs = []
        for result_path in sorted(results_dir.glob("batch_*.txt")):
            task_path = tasks_dir / result_path.with_suffix(".tsv").name
            if not task_path.exists():
                continue
            sentences = []
            with open(task_path, encoding="utf-8") as f:
                reader = csv.reader(f, delimiter="\t")
                next(reader, None)  # skip header
                for row in reader:
                    if row:
                        sentences.append(row[0])
            with open(result_path, encoding="utf-8") as f:
                lines = [l.strip() for l in f if l.strip()]
            for i, sent in enumerate(sentences):
                if i >= len(lines):
                    break
                pairs.append((sent, lines[i]))
        return pairs

    def has_invalid_segments(seg_str: str) -> bool:
        """Return True if any multi-char segment is not a CEDICT word."""
        for word in seg_str.split():
            if len(word) > 1:
                # Multi-char segments must be pure CJK and in CEDICT
                if not _CJK_RE.match(word) or word not in word_set:
                    return True
        return False

    # Load all old span-scorer source data
    old_pairs: dict[str, str] = {}  # sentence -> space-delimited segmentation
    filtered = 0
    for name, tasks_dir, results_dir in SOURCES:
        pairs = load_batch_pairs(tasks_dir, results_dir)
        source_filtered = 0
        for sent, seg in pairs:
            # Convert pipe-delimited to space-delimited
            words = [w for w in seg.split("|") if w]
            seg_str = " ".join(words)
            if has_invalid_segments(seg_str):
                source_filtered += 1
                continue
            old_pairs[sent] = seg_str
        filtered += source_filtered
        print(f"  {name}: {len(pairs):,} sentences, {source_filtered:,} filtered")

    print(f"  Total old (deduped within, after filter): {len(old_pairs):,} ({filtered:,} rejected)")

    # Read existing v2 dataset keys for dedup (preserve file content as-is)
    # Normalize quotes for dedup key so curly/ASCII variants match
    _QUOTE_DEDUP = str.maketrans({
        '"': '\u201c', '\u201d': '\u201c',
        "'": '\u2018', '\u2019': '\u2018',
        '\u300c': '\u201c', '\u300d': '\u201c',
        '\u300e': '\u2018', '\u300f': '\u2018',
    })

    def dedup_key(text: str) -> str:
        return text.translate(_QUOTE_DEDUP)

    # Strip invisible Unicode control characters that confuse BERT tokenization
    import re
    _INVISIBLE_RE = re.compile(r'[\u200b-\u200f\u202a-\u202e\u2060-\u2064\ufeff\ue000-\uf8ff]')

    def strip_invisible(text: str) -> str:
        return _INVISIBLE_RE.sub('', text)

    existing_keys: set[str] = set()
    existing_row_count = 0
    if CWS_TRAINING_PATH.exists():
        with open(CWS_TRAINING_PATH, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t", 1)
                if len(parts) == 2:
                    existing_keys.add(dedup_key(parts[0]))
                    existing_row_count += 1
    print(f"  Existing v2 rows: {existing_row_count:,} ({len(existing_keys):,} unique)")

    # Append new wiki/subs rows (deduped against existing, upsampled)
    added = 0
    skipped_dup = 0
    stripped_count = 0
    with open(CWS_TRAINING_PATH, "a", encoding="utf-8") as f:
        for sent, seg in old_pairs.items():
            key = dedup_key(sent)
            if key in existing_keys:
                skipped_dup += 1
                continue
            existing_keys.add(key)
            clean_sent = strip_invisible(sent)
            clean_seg = strip_invisible(seg)
            if clean_sent != sent:
                stripped_count += 1
            line = f"{clean_sent}\t{clean_seg}\n"
            f.write(line * WIKI_SUBS_UPSAMPLE)
            added += 1

    if stripped_count:
        print(f"  Stripped invisible chars from {stripped_count:,} sentences")

    print(f"  Added {added:,} new sentences from wiki/subs ({WIKI_SUBS_UPSAMPLE}x upsampled)")
    print(f"  Skipped {skipped_dup:,} duplicates (v2 wins)")
    total_rows = existing_row_count + added * WIKI_SUBS_UPSAMPLE
    print(f"  → Combined total: {total_rows:,} rows → {CWS_TRAINING_PATH.name}")


def build_span_scorer_dataset():
    """Convert CWS TSV → span-scorer JSONL (one example per ambiguous position).

    For each CJK position in a sentence, enumerates all CEDICT words starting
    there plus the single character. Where ≥2 candidates exist, emits a training
    example with the gold word and all candidates.
    """
    import re
    from cws.cedict_lookup import load_cedict_words, words_at

    SPAN_SCORER_OUTPUT = _DATA / "span_scorer_dataset_v2.jsonl"
    _IS_CJK = re.compile(r'[\u4e00-\u9fff]')

    # Collapse all quote variants to one form so sentence/segmentation align.
    _QUOTE_NORM = str.maketrans({
        '"': '\u201c', '\u201d': '\u201c',
        "'": '\u2018', '\u2019': '\u2018',
        '\u300c': '\u201c', '\u300d': '\u201c',
        '\u300e': '\u2018', '\u300f': '\u2018',
    })

    def normalize_quotes(text: str) -> str:
        return text.translate(_QUOTE_NORM)

    print("  Loading CEDICT words...")
    word_set, max_len = load_cedict_words()
    print(f"  {len(word_set):,} multi-char words, max length {max_len}")

    if not CWS_TRAINING_PATH.exists():
        sys.exit(f"  Error: {CWS_TRAINING_PATH} not found. Run steps 6-7 first.")

    total_sentences = 0
    total_examples = 0
    skipped_mismatch = 0

    with open(CWS_TRAINING_PATH, encoding="utf-8") as f_in, \
         open(SPAN_SCORER_OUTPUT, "w", encoding="utf-8") as f_out:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t", 1)
            if len(parts) != 2:
                continue
            sentence, seg_str = parts
            seg_words = seg_str.split()

            # Normalize quotes to single canonical form
            sentence = normalize_quotes(sentence)
            seg_words = [normalize_quotes(w) for w in seg_words]

            # Validate round-trip
            if "".join(seg_words) != sentence:
                skipped_mismatch += 1
                continue

            total_sentences += 1
            pos = 0
            for word in seg_words:
                start = pos
                pos += len(word)

                # Only CJK positions are real segmentation decisions
                if not _IS_CJK.match(sentence[start]):
                    continue

                # All CEDICT words at this position + the single char
                candidates = words_at(sentence, start, word_set, max_len)
                single_char = sentence[start]
                if single_char not in candidates:
                    candidates.append(single_char)

                # Gold word must be in candidates (should always be true
                # after step 7 filtering, but safety check)
                if word not in candidates:
                    candidates.append(word)

                # Only ambiguous positions (≥2 candidates) are useful
                if len(candidates) < 2:
                    continue

                f_out.write(json.dumps({
                    "sentence": sentence,
                    "start": start,
                    "gold_word": word,
                    "candidates": candidates,
                }, ensure_ascii=False) + "\n")
                total_examples += 1

    print(f"  Sentences: {total_sentences:,}, skipped {skipped_mismatch} (mismatch)")
    print(f"  → {total_examples:,} training examples → {SPAN_SCORER_OUTPUT.name}")


def _convert_v1_if_needed():
    """Convert v1 WSD data to v2 cluster ordering if not already done."""
    from collections import defaultdict

    v1_tsv = _ROOT / "data" / "wsd_dataset_merged.tsv"
    output = _DATA / "wsd_v1.jsonl"

    if output.exists():
        return  # already converted

    if not v1_tsv.exists():
        print("  (v1 dataset not found, skipping)")
        return

    print("  Converting v1 dataset to v2 cluster ordering...")
    import csv

    entries = json.loads(ENTRIES_PATH.read_text(encoding="utf-8"))
    cache_path = _ROOT / "data" / "translation_cache.json"
    with open(cache_path, encoding="utf-8") as f:
        translation_cache = json.load(f)

    # Build word -> [(cluster_idx, sense_zh), ...]
    by_word = defaultdict(list)
    for e in entries:
        by_word[e["word"]].append((e["pinyin"], e["clusters"]))

    word_cluster_senses: dict[str, list[tuple[int, str]]] = {}
    for word, entry_list in by_word.items():
        clusters_with_zh = []
        idx = 0
        for pinyin, clusters in entry_list:
            for cluster in clusters:
                idx += 1
                zh_parts = []
                for sense_en in cluster["senses"]:
                    zh = translation_cache.get(f"{word}|{pinyin}|{sense_en}")
                    if zh:
                        zh_parts.append(zh)
                clusters_with_zh.append((idx, "；".join(zh_parts) if zh_parts else ""))
        word_cluster_senses[word] = clusters_with_zh

    with open(v1_tsv, encoding="utf-8") as f:
        v1_rows = [r for r in csv.DictReader(f, delimiter="\t") if float(r["label"]) == 1.0]

    results = []
    skipped = 0
    for i, row in enumerate(v1_rows):
        word, context, v1_zh = row["word"], row["context"], row["sense_zh"]
        cluster_senses = word_cluster_senses.get(word, [])
        if len(cluster_senses) < 2:
            continue

        matched_idx = None
        for cid, sense_zh in cluster_senses:
            if sense_zh == v1_zh:
                matched_idx = cid
                break
        if matched_idx is None:
            # Fuzzy: compare as sets of parts
            v1_parts = {p.strip() for p in v1_zh.replace("；", "\n").replace(";", "\n").split("\n") if p.strip()}
            for cid, sense_zh in cluster_senses:
                cand_parts = {p.strip() for p in sense_zh.replace("；", "\n").replace(";", "\n").split("\n") if p.strip()}
                if v1_parts and cand_parts and v1_parts == cand_parts:
                    matched_idx = cid
                    break
        if matched_idx is None:
            skipped += 1
            continue

        star_pos = context.find("★")
        if star_pos == -1:
            skipped += 1
            continue
        clean = context.replace("★", "")

        results.append({
            "id": f"wsd_v1_{i}",
            "source": "v1_llm",
            "sentence": clean,
            "words": [{"word": word, "pos": star_pos, "sense": matched_idx, "n_clusters": len(cluster_senses)}],
        })

    with open(output, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  Converted {len(results)} v1 examples ({skipped} skipped) → {output.name}")


def assemble_wsd():
    """Assemble WSD training data from LLM results + v1 data into training TSV.

    Reads:
      - wsd_results.jsonl (LLM-annotated ebook sentences)
      - wsd_v1.jsonl (v1 LLM-generated + MICS, auto-converted if needed)
      - entries_after_merging.json (cluster structure)
      - translation_cache.json (Chinese translations per raw sense)

    Outputs:
      - wsd_dataset_v2.tsv (columns: word, context, cluster_id, sense_zh, label)
        compatible with train_wsd.py

    sense_zh is built by concatenating the Chinese translations of all raw
    senses in a cluster (from translation_cache.json), separated by "；".
    """
    _convert_v1_if_needed()

    from collections import defaultdict
    import csv

    # Load merged entries for cluster structure
    entries = json.loads(ENTRIES_PATH.read_text(encoding="utf-8"))

    # Load translation cache: "word|pinyin|english" -> chinese
    cache_path = _ROOT / "data" / "translation_cache.json"
    with open(cache_path, encoding="utf-8") as f:
        translation_cache = json.load(f)

    # Build: word -> [(pinyin, clusters), ...] in order (same as build_tasks.py)
    by_word: dict[str, list[tuple[str, list[dict]]]] = defaultdict(list)
    for e in entries:
        by_word[e["word"]].append((e["pinyin"], e["clusters"]))

    def get_sense_zh(word: str, cluster_idx_1based: int) -> str:
        """Build sense_zh by concatenating Chinese translations of senses in cluster."""
        idx = 0
        for pinyin, clusters in by_word.get(word, []):
            for cluster in clusters:
                idx += 1
                if idx == cluster_idx_1based:
                    # Look up each raw sense in translation cache
                    zh_parts = []
                    for sense_en in cluster["senses"]:
                        cache_key = f"{word}|{pinyin}|{sense_en}"
                        zh = translation_cache.get(cache_key)
                        if zh:
                            zh_parts.append(zh)
                    if zh_parts:
                        return "；".join(zh_parts)
                    # Fallback: use cluster zh/en if translation cache misses
                    if cluster.get("zh"):
                        return cluster["zh"]
                    return "; ".join(cluster["senses"])
        return ""

    def get_n_clusters(word: str) -> int:
        total = 0
        for _, clusters in by_word.get(word, []):
            total += len(clusters)
        return total

    # Load results from all sources
    # Note: v1_llm already contains MICLS data, so we don't include wsd_micls.jsonl separately
    wsd_sources = [
        (WSD_RESULTS_PATH, "ebooks"),
        (_DATA / "wsd_v1.jsonl", "v1_llm"),
    ]

    rows = []  # (word, context, cluster_id, sense_zh, label)
    stats = defaultdict(int)

    for source_path, source_name in wsd_sources:
        if not source_path.exists():
            print(f"  WARNING: {source_path.name} not found, skipping")
            continue

        source_count = 0
        with open(source_path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                sentence = r["sentence"]

                for w in r["words"]:
                    word = w["word"]
                    pos = w["pos"]
                    correct_cluster = w["sense"]  # 1-based
                    n_clusters = get_n_clusters(word)

                    if n_clusters < 2:
                        stats["skipped_mono"] += 1
                        continue

                    # Insert ★ markers around target word
                    marked = sentence[:pos] + "★" + word + "★" + sentence[pos + len(word):]

                    # Emit one row per cluster (positive + negatives)
                    for cid in range(1, n_clusters + 1):
                        sense_zh = get_sense_zh(word, cid)
                        if not sense_zh:
                            stats["skipped_no_zh"] += 1
                            continue
                        label = 1.0 if cid == correct_cluster else 0.0
                        rows.append((word, marked, cid, sense_zh, label))

                    source_count += 1

        stats[source_name] = source_count
        print(f"  {source_name}: {source_count:,} word disambiguations")

    # Write TSV
    WSD_TSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(WSD_TSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["word", "context", "cluster_id", "sense_zh", "label"])
        for row in rows:
            writer.writerow(row)

    n_positive = sum(1 for r in rows if r[4] == 1.0)
    n_negative = sum(1 for r in rows if r[4] == 0.0)
    print(f"  Total rows: {len(rows):,} ({n_positive:,} positive, {n_negative:,} negative)")
    if stats.get("skipped_mono"):
        print(f"  Skipped (monosemous): {stats['skipped_mono']:,}")
    if stats.get("skipped_no_zh"):
        print(f"  Skipped (no sense_zh): {stats['skipped_no_zh']:,}")
    print(f"  → {WSD_TSV_PATH}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

STEPS = [
    ("Extract sentences from corpus", extract),
    ("Build CWS tasks from sentences", cws_tasks),
    ("LLM-annotate CWS segmentation", cws_llm),
    ("Extract ICWB2 (free + tasks)", icwb2_extract),
    ("LLM-annotate ICWB2 tasks", icwb2_llm),
    ("Assemble CWS training data", assemble_all),
    ("Merge wiki/subs data (Opus)", merge_old_span_scorer),
    ("Build span-scorer dataset (JSONL)", build_span_scorer_dataset),
    ("Segment ebooks with span scorer for WSD", wsd_segment),
    ("Build WSD tasks from segments", wsd_tasks),
    ("LLM-annotate WSD labels", wsd_llm),
    ("Assemble WSD training data", assemble_wsd),
]

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Build CWS + WSD training data from ebook corpus via LLM annotation.",
        epilog="Examples:\n"
               "  python build_dataset.py 3 --model claude-opus-4.6 --verbose\n"
               "  python build_dataset.py 3 --model deepseek-v4-pro-thinking -c 4 -o results_test.jsonl\n"
               "  python build_dataset.py 1 2 3\n",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("steps", nargs="*", type=int, help="Step numbers to run (1-indexed). All if omitted.")
    parser.add_argument("--model", "-m", default=None, help="Override LLM model for annotation steps")
    parser.add_argument("--concurrency", "-c", type=int, default=8, help="Concurrent API calls (default: 8)")
    parser.add_argument("--tasks-per-call", type=int, default=5, help="Sentences batched per API call (default: 5)")
    parser.add_argument("--batch", action="store_true", help="Use Anthropic Batch API (50%% discount, requires 'anthropic' provider models)")
    parser.add_argument("--max-sentences", type=int, default=None, help="Limit sentences for WSD segmentation step (step 9)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print per-task details")
    parser.add_argument("--output", "-o", default=None, help="Override output filename for CWS results (in data/dataset_v2/)")

    args = parser.parse_args()

    # Apply overrides to globals used by step functions
    if args.model:
        CWS_MODEL = args.model
        WSD_MODEL = args.model
    if args.output:
        EBOOKS_CWS_RESULTS_PATH = _DATA / args.output

    wsd_max_sentences = args.max_sentences

    # Monkey-patch annotate calls to pass concurrency/verbose
    _orig_cws_llm = cws_llm
    _orig_icwb2_llm = icwb2_llm

    def cws_llm():
        from cws.annotate import annotate_cws
        annotate_cws(EBOOKS_CWS_TASKS_PATH, EBOOKS_CWS_RESULTS_PATH, model=CWS_MODEL,
                     concurrency=args.concurrency, verbose=args.verbose,
                     tasks_per_call=args.tasks_per_call, batch=args.batch)

    def icwb2_llm():
        from cws.annotate_merge import annotate_merges
        annotate_merges(ICWB2_CWS_TASKS_PATH, ICWB2_CWS_RESULTS_PATH, model=ICWB2_MODEL,
                        concurrency=args.concurrency, verbose=args.verbose,
                        batch=args.batch)

    def wsd_llm():
        from wsd.annotate import annotate_wsd
        annotate_wsd(WSD_TASKS_PATH, WSD_RESULTS_PATH, model=WSD_MODEL,
                     concurrency=args.concurrency, verbose=args.verbose,
                     tasks_per_call=args.tasks_per_call, batch=args.batch)

    requested = [x - 1 for x in args.steps] if args.steps else list(range(len(STEPS)))
    # Re-bind steps that were overridden
    STEPS[2] = ("LLM-annotate CWS segmentation", cws_llm)
    STEPS[4] = ("LLM-annotate ICWB2 tasks", icwb2_llm)
    STEPS[10] = ("LLM-annotate WSD labels", wsd_llm)

    for i in requested:
        if i < 0 or i >= len(STEPS):
            sys.exit(f"Unknown step {i+1}. Valid: 1-{len(STEPS)}")
        name, fn = STEPS[i]
        print(f"\n=== {i+1}. {name} ===")
        fn()
    print("\nDone.")
