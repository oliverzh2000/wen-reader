#!/usr/bin/env python3
"""
Build CWS fine-tuning dataset.

Usage:
    python build_dataset.py gen-tasks   # Generate LLM task files from MSR data
    python build_dataset.py assemble    # Build final dataset from LLM results
"""
import csv
import re
import shutil
import argparse
from pathlib import Path
from collections import defaultdict

_ROOT = Path(__file__).parent.parent.parent
DEFAULT_CEDICT = _ROOT / "data" / "cedict_ts.u8"
DEFAULT_MSR = _ROOT / "data" / "icwb2-data" / "training" / "msr_training.utf8"
DEFAULT_LLM_DIR = _ROOT / "data" / "cws_llm"

LINE_RE = re.compile(r"^\S+\s+(\S+)\s+\[.+?]\s+/.*/", re.VERBOSE)


def load_cedict_vocab(cedict_path: str | Path = DEFAULT_CEDICT) -> set[str]:
    """Load simplified Chinese vocabulary from cedict."""
    vocab: set[str] = set()
    with open(cedict_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            m = LINE_RE.match(line)
            if m:
                vocab.add(m.group(1))
    return vocab


def load_segmented_sentences(path: str | Path = DEFAULT_MSR) -> list[list[str]]:
    """Load MSR segmented sentences as lists of tokens."""
    sentences: list[list[str]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            words = [w for w in line.split() if w]
            if words:
                sentences.append(words)
    return sentences


def find_mergeable_spans(
    words: list[str],
    cedict: set[str],
    max_merge_len: int = 8,
) -> list[tuple[int, int, str, list[str]]]:
    """
    Find spans of consecutive tokens that merge into a cedict word.

    Returns list of (start_idx, end_idx, merged_word, original_tokens)
    where words[start_idx:end_idx] are the original tokens.

    Greedy maximal: longest span preferred, subsumed shorter spans discarded.
    Threshold: merged word must be 2+ characters (lowered from 4+ in the
    experimental script).
    """
    n = len(words)
    spans: list[tuple[int, int, str, list[str]]] = []

    for start in range(n):
        merged = ""
        for end in range(start + 1, min(start + max_merge_len, n + 1)):
            merged += words[end - 1]

            # Need at least 2 tokens to form a merge
            if end - start < 2:
                continue

            # Bail if merged string is too long
            if len(merged) > 16:
                break

            # Merged word must be 2+ chars and in cedict
            if len(merged) >= 2 and merged in cedict:
                original_tokens = words[start:end]
                # Each individual token must be a single char or a cedict word
                all_in_cedict = all(
                    len(t) == 1 or t in cedict for t in original_tokens
                )
                if all_in_cedict:
                    spans.append((start, end, merged, original_tokens))

    # Prefer longer spans; discard subsumed shorter spans
    spans.sort(key=lambda x: -(x[1] - x[0]))
    kept: list[tuple[int, int, str, list[str]]] = []
    used: set[int] = set()
    for start, end, merged, tokens in spans:
        if any(i in used for i in range(start, end)):
            continue
        kept.append((start, end, merged, tokens))
        for i in range(start, end):
            used.add(i)

    kept.sort(key=lambda x: x[0])
    return kept


# ---------------------------------------------------------------------------
# Subcommands (stubs — implementations added in later tasks)
# ---------------------------------------------------------------------------

def format_sentence_with_marker(
    words: list[str], span_start: int, span_end: int
) -> str:
    """Format a segmented sentence with #...# markers around a target span.

    Tokens inside the span are joined with spaces and wrapped in ``#...#``.
    All parts are joined with ``  `` (two spaces) following the WSD convention.
    """
    parts: list[str] = []
    i = 0
    while i < len(words):
        if i == span_start:
            inner = " ".join(words[span_start:span_end])
            parts.append(f"#{inner}#")
            i = span_end
        else:
            parts.append(words[i])
            i += 1
    return "  ".join(parts)


def classify_sentences(
    sentences: list[list[str]],
    cedict: set[str],
    freq_cap: int,
) -> tuple[
    dict[str, list[int]],
    list[tuple[int, int, int, str, list[str]]],
    dict[str, int],
]:
    """Classify every MSR sentence and collect LLM-eligible span rows.

    Returns:
        categories: mapping of category name -> list of sentence indices
        llm_rows: list of (sentence_idx, span_start, span_end, merged_word, original_tokens)
                  for 2-3 char spans within freq cap
        word_freq: final per-merged-word frequency counts
    """
    categories: dict[str, list[int]] = defaultdict(list)
    llm_rows: list[tuple[int, int, int, str, list[str]]] = []
    word_freq: dict[str, int] = defaultdict(int)

    for idx, words in enumerate(sentences):
        spans = find_mergeable_spans(words, cedict)

        if not spans:
            categories["clean"].append(idx)
            continue

        # Check frequency cap for ALL spans first
        over_cap = False
        for _, _, merged, _ in spans:
            if word_freq[merged] >= freq_cap:
                over_cap = True
                break

        if over_cap:
            categories["dropped"].append(idx)
            continue

        # Classify by char length of spans
        has_short = False  # 2-3 char span
        for _, _, merged, _ in spans:
            if len(merged) <= 3:
                has_short = True
                break

        if has_short:
            categories["llm_eligible"].append(idx)
        else:
            categories["blind_merge"].append(idx)

        # Increment frequency counts for all merged words in this sentence
        for start, end, merged, tokens in spans:
            word_freq[merged] += 1

        # Collect LLM rows for 2-3 char spans
        if has_short:
            for start, end, merged, tokens in spans:
                if len(merged) <= 3:
                    llm_rows.append((idx, start, end, merged, tokens))

    return dict(categories), llm_rows, dict(word_freq)


def gen_tasks(args: argparse.Namespace) -> None:
    """Generate LLM task batches from MSR data."""
    cedict = load_cedict_vocab(args.cedict_path)
    sentences = load_segmented_sentences(args.msr_path)

    categories, llm_rows, word_freq = classify_sentences(
        sentences, cedict, args.freq_cap
    )

    # Prepare output directory
    tasks_dir = args.llm_dir / "llm_tasks"
    if tasks_dir.exists():
        shutil.rmtree(tasks_dir)
    tasks_dir.mkdir(parents=True)

    # Write batch TSV files
    num_batches = 0
    for batch_start in range(0, len(llm_rows), args.batch_size):
        batch = llm_rows[batch_start : batch_start + args.batch_size]
        num_batches += 1
        path = tasks_dir / f"batch_{num_batches:04d}.tsv"
        with open(path, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f, delimiter="\t")
            w.writerow(["id", "segmented_sentence", "merged_word"])
            for local_id, (sent_idx, start, end, merged, tokens) in enumerate(
                batch, 1
            ):
                marked = format_sentence_with_marker(sentences[sent_idx], start, end)
                w.writerow([local_id, marked, merged])

    # Summary stats
    total = len(sentences)
    clean = len(categories.get("clean", []))
    blind = len(categories.get("blind_merge", []))
    eligible = len(categories.get("llm_eligible", []))
    dropped = len(categories.get("dropped", []))

    print(f"Total sentences: {total}")
    print(f"  Clean (no spans):    {clean}")
    print(f"  Blind merge (4+):    {blind}")
    print(f"  LLM eligible (2-3):  {eligible}")
    print(f"  Dropped (over cap):  {dropped}")
    print(f"  LLM task rows:       {len(llm_rows)}")
    print(f"  Batch files written: {num_batches}")



def apply_merges(
    words: list[str], merges: list[tuple[int, int, str]]
) -> list[str]:
    """Apply merges to a token list.

    Args:
        words: original token list
        merges: [(start, end, merged_word), ...] sorted by start index
    """
    result: list[str] = []
    i = 0
    merge_idx = 0
    while i < len(words):
        if merge_idx < len(merges) and i == merges[merge_idx][0]:
            start, end, merged = merges[merge_idx]
            result.append(merged)
            i = end
            merge_idx += 1
        else:
            result.append(words[i])
            i += 1
    return result


def load_llm_decisions(
    results_dir: Path, batch_size: int
) -> dict[tuple[str, int], str]:
    """Load LLM decisions from result batch files.

    Returns mapping of (batch_filename, local_id) -> decision ("merge"/"split").
    Unparseable rows are omitted (caller treats missing keys as discard).
    """
    decisions: dict[tuple[str, int], str] = {}
    if not results_dir.exists():
        return decisions

    for path in sorted(results_dir.glob("batch_*.tsv")):
        fname = path.name
        with open(path, encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            for row in reader:
                # Skip header
                if row and row[0] == "id":
                    continue
                # Validate row: must have exactly 2 columns
                if len(row) != 2:
                    continue
                raw_id, decision = row[0].strip(), row[1].strip().lower()
                if not raw_id or not decision:
                    continue
                # Validate decision value
                if decision not in ("merge", "split"):
                    continue
                try:
                    local_id = int(raw_id)
                except ValueError:
                    continue
                decisions[(fname, local_id)] = decision

    return decisions


def assemble(args: argparse.Namespace) -> None:
    """Assemble final dataset from LLM results."""
    cedict = load_cedict_vocab(args.cedict_path)
    sentences = load_segmented_sentences(args.msr_path)

    # Step 1: Re-classify sentences (same logic as gen-tasks)
    categories, llm_rows, word_freq = classify_sentences(
        sentences, cedict, args.freq_cap
    )

    # Step 2: Load LLM results
    results_dir = args.llm_dir / "llm_results"
    decisions = load_llm_decisions(results_dir, args.batch_size)

    # Build mapping: llm_row_index -> (batch_filename, local_id)
    # llm_rows are in the same order as written to batch files by gen_tasks
    row_to_batch: dict[int, tuple[str, int]] = {}
    for row_idx in range(len(llm_rows)):
        batch_num = (row_idx // args.batch_size) + 1
        local_id = (row_idx % args.batch_size) + 1
        batch_fname = f"batch_{batch_num:04d}.tsv"
        row_to_batch[row_idx] = (batch_fname, local_id)

    # Step 3: Group llm_rows by sentence index
    # A sentence may have multiple 2-3 char spans (multiple llm_rows)
    sent_llm_rows: dict[int, list[tuple[int, int, int, str, list[str], int]]] = defaultdict(list)
    for row_idx, (sent_idx, start, end, merged, tokens) in enumerate(llm_rows):
        sent_llm_rows[sent_idx].append((sent_idx, start, end, merged, tokens, row_idx))

    # Step 4: Load CWS model for filtering clean sentences
    from transformers import BertTokenizerFast, BertForTokenClassification
    import torch
    from cws import load_cedict_vocab as _load_cv, segment_sentence

    _device = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"
    _ckip_id = "ckiplab/bert-base-chinese-ws"
    print(f"Loading CWS model ({_ckip_id}) for clean sentence filtering...")
    ckip_tokenizer = BertTokenizerFast.from_pretrained(_ckip_id)
    ckip_model = BertForTokenClassification.from_pretrained(_ckip_id).to(_device)
    ckip_model.eval()
    _cedict_vocab = _load_cv(str(cedict_path))
    print("CWS model loaded")

    # Step 5: Process each category and collect output sentences
    output_lines: list[str] = []
    stats = {
        "clean_included": 0,
        "clean_excluded": 0,
        "blind_merged": 0,
        "llm_corrected": 0,
        "discarded_llm": 0,
        "dropped": len(categories.get("dropped", [])),
    }

    # Process clean sentences: filter unchanged
    clean_indices = categories.get("clean", [])
    for i, sent_idx in enumerate(clean_indices):
        words = sentences[sent_idx]
        raw_text = "".join(words)
        # BERT max 512 tokens — skip inference for long sentences, include as-is
        if len(raw_text) > 500:
            stats["clean_included"] += 1
            output_lines.append(" ".join(words))
            continue
        model_output = segment_sentence(
            raw_text, ckip_tokenizer, ckip_model,
            device=_device, cedict_vocab=_cedict_vocab,
        )
        msr_segmentation = "-".join(words)
        if model_output == msr_segmentation:
            stats["clean_excluded"] += 1
        else:
            stats["clean_included"] += 1
            output_lines.append(" ".join(words))
        if (i + 1) % 500 == 0:
            print(f"  Filtering clean sentences: {i + 1}/{len(clean_indices)} ({stats['clean_excluded']} excluded, {stats['clean_included']} kept)", end="\r", flush=True)
    if clean_indices:
        print()  # newline after carriage returns

    # Process blind_merge sentences: apply all 4+ char merges
    for sent_idx in categories.get("blind_merge", []):
        words = sentences[sent_idx]
        spans = find_mergeable_spans(words, cedict)
        merges = sorted(
            [(s, e, m) for s, e, m, _ in spans if len(m) >= 4],
            key=lambda x: x[0],
        )
        merged_words = apply_merges(words, merges)
        output_lines.append(" ".join(merged_words))
        stats["blind_merged"] += 1

    # Process llm_eligible sentences
    for sent_idx in categories.get("llm_eligible", []):
        words = sentences[sent_idx]
        spans = find_mergeable_spans(words, cedict)

        # Separate 4+ char spans (blind merge) and 2-3 char spans (LLM)
        blind_merges: list[tuple[int, int, str]] = []
        llm_spans: list[tuple[int, int, str]] = []
        for s, e, m, _ in spans:
            if len(m) >= 4:
                blind_merges.append((s, e, m))
            else:
                llm_spans.append((s, e, m))

        # Look up LLM decisions for 2-3 char spans
        discard = False
        approved_merges: list[tuple[int, int, str]] = list(blind_merges)

        rows_for_sent = sent_llm_rows.get(sent_idx, [])
        for _, start, end, merged, _, row_idx in rows_for_sent:
            key = row_to_batch.get(row_idx)
            if key is None:
                discard = True
                break
            decision = decisions.get(key)
            if decision is None:
                # Missing or unparseable -> discard entire sentence
                discard = True
                break
            if decision == "merge":
                approved_merges.append((start, end, merged))

        if discard:
            stats["discarded_llm"] += 1
            continue

        approved_merges.sort(key=lambda x: x[0])
        merged_words = apply_merges(words, approved_merges)
        output_lines.append(" ".join(merged_words))
        stats["llm_corrected"] += 1

    # Step 6: Write output
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for line in output_lines:
            f.write(line + "\n")

    # Step 7: Print summary stats
    total = len(sentences)
    final = len(output_lines)
    print(f"\nAssembly complete:")
    print(f"Total MSR sentences:       {total}")
    print(f"  Clean included:          {stats['clean_included']}")
    print(f"  Clean excluded (same):   {stats['clean_excluded']}")
    print(f"  Blind merged (4+):       {stats['blind_merged']}")
    print(f"  LLM corrected (2-3):     {stats['llm_corrected']}")
    print(f"  Discarded (bad LLM):     {stats['discarded_llm']}")
    print(f"  Dropped (over cap):      {stats['dropped']}")
    print(f"Final dataset size:        {final}")
    print(f"Wrote {args.output}")



# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Build CWS fine-tuning dataset")
    p.add_argument("--msr-path", type=Path, default=DEFAULT_MSR)
    p.add_argument("--cedict-path", type=Path, default=DEFAULT_CEDICT)
    p.add_argument("--llm-dir", type=Path, default=DEFAULT_LLM_DIR)
    p.add_argument("--freq-cap", type=int, default=10,
                   help="Max examples per unique merged word")
    p.add_argument("--batch-size", type=int, default=50,
                   help="Rows per LLM task batch file")

    sub = p.add_subparsers(dest="cmd", required=True)
    gt = sub.add_parser("gen-tasks", help="Generate LLM task files")
    gt.set_defaults(func=gen_tasks)

    asm = sub.add_parser("assemble", help="Build final dataset from LLM results")
    asm.add_argument("--output", type=Path,
                     default=_ROOT / "data" / "cws_dataset.txt")
    asm.set_defaults(func=assemble)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
