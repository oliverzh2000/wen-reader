"""
Sense Mapper: Maps MiCLS definitions to cedict Chinese translations via semantic similarity.

Usage:
    python map_senses_to_cedict.py [--threshold 0.7] [--output ml/data/mappings.tsv]
    python map_senses_to_cedict.py --debug --limit 20  # inspect small sample
"""

import csv
import json
import argparse
from pathlib import Path
from dataclasses import dataclass, asdict, fields
from collections import defaultdict

import torch
from datasets import load_dataset
from sentence_transformers import SentenceTransformer


@dataclass
class SenseMapping:
    word: str
    context: str
    micls_definition: str
    cedict_key: str
    cedict_sense_zh: str
    similarity_score: float


class SenseMapper:
    def __init__(self, cache_path: str, model_name: str = "BAAI/bge-large-zh-v1.5", batch_size: int = 256):
        with open(cache_path) as f:
            self.cache = json.load(f)
        self.word_to_senses = self._build_word_index()
        self.model = SentenceTransformer(model_name)
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.model.to(self.device)
        self.batch_size = batch_size

    def _build_word_index(self) -> dict[str, list[tuple[str, str]]]:
        index = defaultdict(list)
        for key, chinese in self.cache.items():
            word = key.split("|")[0]
            index[word].append((key, chinese))
        return dict(index)

    def map_batch(self, examples: list[dict], threshold: float, debug: bool = False) -> list[SenseMapping | None]:
        """Map a batch of examples. Returns list aligned with input (None for failures)."""
        # Collect all texts to encode
        all_texts = []
        text_ranges = []  # (start_idx, def_idx, sense_indices) for each example
        
        for ex in examples:
            word = ex["word"]
            if word not in self.word_to_senses:
                text_ranges.append(None)
                continue
                
            senses = self.word_to_senses[word]
            definition = ex["definition"]
            
            start = len(all_texts)
            all_texts.append(definition)
            def_idx = start
            
            sense_start = len(all_texts)
            for key, chinese in senses:
                all_texts.append(f"{word}：{chinese}")
            
            text_ranges.append((def_idx, sense_start, sense_start + len(senses), senses))
        
        if not all_texts:
            return [None] * len(examples)
        
        # Batch encode everything
        embeddings = self.model.encode(all_texts, convert_to_tensor=True, batch_size=self.batch_size)
        
        # Compute similarities and build results
        results = []
        for i, ex in enumerate(examples):
            if text_ranges[i] is None:
                results.append(None)
                continue
                
            def_idx, sense_start, sense_end, senses = text_ranges[i]
            def_emb = embeddings[def_idx]
            sense_embs = embeddings[sense_start:sense_end]
            
            similarities = torch.nn.functional.cosine_similarity(def_emb.unsqueeze(0), sense_embs)
            best_idx = similarities.argmax().item()
            best_score = similarities[best_idx].item()
            
            if debug:
                word = ex["word"]
                print(f"\n--- {word} ---")
                print(f"Context: {ex['context'][:60]}...")
                print(f"MiCLS: {ex['definition']}")
                print("Cedict senses (sorted by score):")
                scored = [(senses[j][0], senses[j][1], similarities[j].item()) for j in range(len(senses))]
                scored.sort(key=lambda x: x[2], reverse=True)
                for key, chinese, score in scored:
                    marker = " <-- SELECTED" if score == best_score and best_score >= threshold else ""
                    if score == best_score and best_score < threshold:
                        marker = " <-- BELOW THRESHOLD"
                    print(f"  [{score:.3f}] {key} -> {chinese}{marker}")
            
            if best_score < threshold:
                results.append(None)
            else:
                best_key, best_chinese = senses[best_idx]
                results.append(SenseMapping(
                    ex["word"], ex["context"], ex["definition"],
                    best_key, best_chinese, best_score
                ))
        
        return results


def process_split(mapper: SenseMapper, examples: list, threshold: float, limit: int | None, debug: bool):
    positives = [ex for ex in examples if ex["label"] == 1]
    if limit:
        positives = positives[:limit]

    mappings = []
    skipped = 0
    failed = 0
    
    # Process in batches (batch_size controls both grouping and encode)
    batch_size = mapper.batch_size if not debug else len(positives)
    
    for batch_start in range(0, len(positives), batch_size):
        batch = positives[batch_start:batch_start + batch_size]
        results = mapper.map_batch(batch, threshold, debug)
        
        for ex, result in zip(batch, results):
            if ex["word"] not in mapper.word_to_senses:
                skipped += 1
            elif result is None:
                failed += 1
            else:
                mappings.append(asdict(result))
        
        if not debug and batch_start > 0:
            print(f"    {batch_start + len(batch)}/{len(positives)} - {len(mappings)} mapped")

    return mappings, skipped, failed, len(positives)


def write_tsv(path: Path, mappings: list):
    if not mappings:
        return
    fieldnames = [f.name for f in fields(SenseMapping)]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(mappings)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=0.7)
    parser.add_argument("--output", type=Path, default=Path("data/mappings.tsv"))
    parser.add_argument("--cache", type=str, default="data/translation_cache.json")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    print("Loading model and cache...")
    mapper = SenseMapper(args.cache, batch_size=args.batch_size)

    print("Loading MiCLS dataset...")
    ds = load_dataset("wyy209/MiCLS")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    all_mappings = []
    for split in ["train", "validation", "test"]:
        if split not in ds:
            continue

        print(f"\nProcessing {split}...")
        mappings, skipped, failed, total = process_split(mapper, ds[split], args.threshold, args.limit, args.debug)
        all_mappings.extend(mappings)

        attempted = total - skipped
        pct = 100 * len(mappings) / attempted if attempted > 0 else 0
        print(f"  {skipped} skipped (not in cache), {len(mappings)}/{attempted} mapped ({pct:.1f}%), {failed} below threshold")

    write_tsv(args.output, all_mappings)
    print(f"\nDone. {len(all_mappings)} mappings -> {args.output}")


if __name__ == "__main__":
    main()
