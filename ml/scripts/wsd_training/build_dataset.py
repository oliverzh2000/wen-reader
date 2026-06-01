#!/usr/bin/env python3
"""
Build WSD training dataset.

Usage:
    python build_dataset.py gen-tasks   # Generate LLM task files
    python build_dataset.py assemble    # Build final dataset from LLM results
"""
import csv
import json
import re
import argparse
from pathlib import Path
from collections import defaultdict

from wordfreq import zipf_frequency


def zipf_to_examples(word: str, min_ex: int, max_ex: int) -> int:
    """
    Map word frequency to examples needed. Uses zipf scale (0-7 for Chinese).
    Zipf 6+ (的,是,了) -> max_ex, Zipf 1- (rare) -> min_ex, linear between.
    """
    zipf = zipf_frequency(word, "zh")
    if zipf <= 1:
        return min_ex
    if zipf >= 6:
        return max_ex
    return min_ex + int((zipf - 1) / 5 * (max_ex - min_ex)) # linearly interpolate


class DatasetBuilder:
    def __init__(self, cache_path: Path, mappings_path: Path):
        with open(cache_path) as f:
            self.cache: dict[str, str] = json.load(f)
        self.word_to_senses = self._build_word_index()
        with open(mappings_path, newline="", encoding="utf-8") as f:
            self.mappings = list(csv.DictReader(f, delimiter="\t"))
        self.sense_counts = defaultdict(int)
        for m in self.mappings:
            self.sense_counts[m["cedict_key"]] += 1

    def _build_word_index(self) -> dict[str, list[tuple[str, str]]]:
        index = defaultdict(list)
        for key, chinese in self.cache.items():
            word = key.split("|")[0]
            index[word].append((key, chinese))
        return dict(index)

    def get_senses_needing_llm(self, min_ex: int, max_ex: int, max_senses: int | None) -> list[tuple[str, str, int, float]]:
        """Returns [(cedict_key, chinese, num_needed, zipf), ...] sorted by frequency desc."""
        needs_gen = []
        skipped_mono = 0
        for key, chinese in self.cache.items():
            word = key.split("|")[0]
            # Skip monosemous words — no negatives to contrast against
            if len(self.word_to_senses.get(word, [])) < 2:
                skipped_mono += 1
                continue
            zipf = zipf_frequency(word, "zh")
            target = zipf_to_examples(word, min_ex, max_ex)
            needed = max(0, target - self.sense_counts.get(key, 0))
            if needed > 0:
                needs_gen.append((key, chinese, needed, zipf))
        if skipped_mono:
            print(f"Skipped {skipped_mono} monosemous senses (no negatives possible)")
        needs_gen.sort(key=lambda x: -x[3])
        if max_senses:
            needs_gen = needs_gen[:max_senses]
        return needs_gen

    def generate_examples(self, word: str, context: str, correct_key: str, 
                          correct_sense: str, source: str) -> list[tuple]:
        """Generate (word, context, sense, sense_en, label, source) tuples with hard negatives."""
        ctx = re.sub(r"#([^#]+)#", r"★\1★", context)
        correct_en = correct_key.split("|", 2)[2] if "|" in correct_key else ""
        results = [(word, ctx, correct_sense, correct_en, 1.0, source)]
        for key, chinese in self.word_to_senses.get(word, []):
            if key != correct_key:
                neg_en = key.split("|", 2)[2] if "|" in key else ""
                results.append((word, ctx, chinese, neg_en, 0.0, source))
        return results


def gen_tasks(args):
    builder = DatasetBuilder(args.cache, args.mappings)
    needs_llm = builder.get_senses_needing_llm(args.min_examples, args.max_examples, args.max_senses)
    total_examples = sum(n for _, _, n, _ in needs_llm)
    
    print(f"Loaded {len(builder.cache)} senses, {len(builder.mappings)} existing examples")
    print(f"Generating: {len(needs_llm)} senses, {total_examples} examples")
    print(f"  (min={args.min_examples}, max={args.max_examples} examples per sense)")
    
    import shutil
    tasks_dir = args.llm_dir / "llm_tasks"
    if tasks_dir.exists():
        shutil.rmtree(tasks_dir)
    tasks_dir.mkdir(parents=True)
    
    num_files = 0
    for i in range(0, len(needs_llm), args.senses_per_file):
        batch = needs_llm[i:i + args.senses_per_file]
        num_files += 1
        path = tasks_dir / f"batch_{num_files:04d}.tsv"
        with open(path, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f, delimiter='\t')
            w.writerow(["word", "pinyin", "sense_en", "sense_zh", "num_sentences_needed"])
            for key, chinese, needed, _ in batch:
                parts = key.split("|", 2)
                w.writerow([parts[0], parts[1] if len(parts) > 1 else "", 
                           parts[2] if len(parts) > 2 else "", chinese, needed])
    
    print(f"Wrote {num_files} batch files to {tasks_dir}/")


def check_word_boundary(sentence: str, word: str, segment_fn) -> bool:
    """
    Check if marked word aligns with CWS segment boundaries.
    Returns True if word is a valid segment, False if it's part of a larger word.
    """
    # Remove markers and find word position
    clean = sentence.replace("#", "")
    start = sentence.find("#")
    if start == -1:
        return True
    # Adjust for removed first #
    word_start = start
    word_end = word_start + len(word)
    
    # Segment the clean sentence
    segments = segment_fn(clean).split("-")
    
    # Check if word matches a segment exactly
    pos = 0
    for seg in segments:
        seg_end = pos + len(seg)
        # Word starts inside this segment
        if pos <= word_start < seg_end:
            # Word should match this segment exactly
            if word_start == pos and word_end == seg_end:
                return True
            # Word is a substring of this segment - bad
            return False
        pos = seg_end
    return True


def assemble(args):
    builder = DatasetBuilder(args.cache, args.mappings)
    
    # Load CWS model if filtering enabled
    segment_fn = None
    if args.filter_cws:
        print("Loading finetuned CWS model for boundary filtering...")
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "cws_training"))
        from cws import load_cedict_vocab as _load_cv, segment_sentence
        from transformers import BertTokenizerFast, BertForTokenClassification
        import torch

        _device = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"
        _ml_dir = Path(__file__).parent.parent.parent
        _cedict_path = str(_ml_dir / "data" / "cedict_ts.u8")
        _cedict_vocab = _load_cv(_cedict_path)

        # Use finetuned CWS model (same as production app)
        _ckpt_path = _ml_dir / "models" / "cws_finetuned" / "final"
        if _ckpt_path.exists():
            _model_path = str(_ckpt_path)
            print(f"  Using finetuned checkpoint: {_model_path}")
        else:
            raise ValueError("finetuned checkpoint not found")

        _tok = BertTokenizerFast.from_pretrained(_model_path)
        _mod = BertForTokenClassification.from_pretrained(_model_path).to(_device)
        _mod.eval()
        segment_fn = lambda s: segment_sentence(s, _tok, _mod, device=_device, cedict_vocab=_cedict_vocab)
        print("CWS model loaded")
    
    examples = []
    filtered_count = 0
    
    # From mappings (MICLS) - assume these are already validated
    for m in builder.mappings:
        examples.extend(builder.generate_examples(
            m["word"], m["context"], m["cedict_key"], m["cedict_sense_zh"], "micls"))
    micls_count = len(examples)
    
    # From LLM results
    llm_sentences = 0
    skipped_sense = 0
    results_dir = args.llm_dir / "llm_results"
    if results_dir.exists():
        batch_files = sorted(results_dir.glob("*.tsv"))
        for batch_idx, path in enumerate(batch_files, 1):
            print(f"  Processing batch {batch_idx}/{len(batch_files)}: {path.name}", end="\r")
            
            with open(path, encoding="utf-8") as f:
                for row in csv.reader(f, delimiter='\t'):
                    if len(row) < 4:
                        continue
                    word, pinyin, sense_zh = row[0], row[1], row[2]
                    if word == "word":  # skip header
                        continue
                    # Build key for hard negative lookup (need to find matching entry in cache)
                    key = None
                    for k in builder.word_to_senses.get(word, []):
                        if k[1] == sense_zh:  # k is (cedict_key, chinese)
                            key = k[0]
                            break
                    if not key:
                        skipped_sense += 1
                        continue  # skip rows with unrecognized Chinese sense
                    for ctx in row[3:]:
                        if ctx.count("#") != 2:
                            continue
                        # Filter by CWS boundary if enabled
                        if segment_fn and not check_word_boundary(ctx, word, segment_fn):
                            filtered_count += 1
                            continue
                        llm_sentences += 1
                        examples.extend(builder.generate_examples(word, ctx, key, sense_zh, "llm"))
        print()  # newline after progress
    
    print(f"MICLS: {micls_count} examples")
    print(f"LLM: {llm_sentences} sentences -> {len(examples) - micls_count} examples (with hard negatives)")
    if skipped_sense:
        print(f"Skipped: {skipped_sense} rows (unrecognized Chinese sense)")
    if filtered_count:
        print(f"Filtered: {filtered_count} sentences (word not at segment boundary)")
    
    # Dedup: collapse by context, keep exactly one positive per context
    by_ctx = defaultdict(list)
    for ex in examples:
        by_ctx[ex[1]].append(ex)  # group by context (index 1)
    pre_filter = len(examples)
    deduped = []
    dup_dropped = 0
    mono_dropped = 0
    for ctx, group in by_ctx.items():
        if len(group) < 2:
            mono_dropped += len(group)
            continue  # monosemous — no negatives to rank against
        positives = [ex for ex in group if ex[4] == 1.0]
        negatives = [ex for ex in group if ex[4] == 0.0]
        if len(positives) > 1:
            dup_dropped += len(positives) - 1
            positives = positives[:1]  # keep first positive only
        deduped.extend(positives)
        deduped.extend(negatives)
    examples = deduped
    if dup_dropped:
        print(f"Deduped: {dup_dropped} extra positive rows (duplicate contexts in mappings)")
    if mono_dropped:
        print(f"Dropped: {mono_dropped} monosemous rows (context with no negatives)")
    
    print(f"Total: {len(examples)} examples")
    
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["word", "context", "sense_zh", "sense_en", "label", "source"])
        for ex in examples:
            w.writerow(ex)
    print(f"Wrote {args.output}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mappings", type=Path, default=Path("data/mappings.tsv"))
    p.add_argument("--cache", type=Path, default=Path("data/translation_cache.json"))
    p.add_argument("--llm-dir", type=Path, default=Path("data/wsd_llm"))
    p.add_argument("--output", type=Path, default=Path("data/wsd_dataset.tsv"))
    p.add_argument("--min-examples", type=int, default=2, help="Examples per sense for rare words")
    p.add_argument("--max-examples", type=int, default=10, help="Examples per sense for common words")
    p.add_argument("--max-senses", type=int, default=None, help="Cap total senses (prioritizes common words)")
    p.add_argument("--senses-per-file", type=int, default=50)
    p.add_argument("--filter-cws", action="store_true", help="Filter LLM examples by CWS boundary alignment")
    
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("gen-tasks")
    sub.add_parser("assemble")
    
    args = p.parse_args()
    if args.cmd == "gen-tasks":
        gen_tasks(args)
    else:
        assemble(args)


if __name__ == "__main__":
    main()
