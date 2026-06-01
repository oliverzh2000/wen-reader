#!/usr/bin/env python3
"""Train bge-reranker-base for WSD using sentence-transformers v4.

Default: ListNet (listwise ranking loss) — groups all senses per context
and optimizes the ranking directly. Better than pointwise BCE for WSD
since we care about ranking the correct sense above incorrect ones.

Usage:
    python scripts/wsd_training/train_wsd_crossencoder.py sense_en              # ListNet (default)
    python scripts/wsd_training/train_wsd_crossencoder.py sense_zh              # ListNet with Chinese senses
    python scripts/wsd_training/train_wsd_crossencoder.py sense_en --bce        # fallback to pointwise BCE
"""
import csv
import random
import sys
import torch
from datetime import datetime
from pathlib import Path
from datasets import Dataset
from sentence_transformers import CrossEncoder
from sentence_transformers.cross_encoder import CrossEncoderTrainer, CrossEncoderTrainingArguments
from sentence_transformers.cross_encoder.losses import BinaryCrossEntropyLoss, LambdaLoss
from sentence_transformers.cross_encoder.evaluation import CrossEncoderRerankingEvaluator

DATA_PATH = Path(__file__).parent.parent.parent / "data" / "wsd_dataset.tsv"
OUTPUT_BASE = Path(__file__).parent.parent.parent / "models"
MODEL_NAME = "hfl/chinese-roberta-wwm-ext"

# Hyperparams - matching FlagEmbedding official settings
# Official: batch_size=2, train_group_size=8, nproc=2 → 32 pairs/step, LR=6e-5
# We use sentence-transformers (no train_group_size), so scale by pairs/step
# Uncomment ONE pair:

# --- Hyperparams ---
# ListNet: each sample = 1 context with ~4.4 senses (1 pos + ~3.4 neg).
# batch=64 contexts × ~4.4 senses ≈ 282 forward passes/step.
# LR sqrt-scaled from base: batch=16 @ 6e-5

# BATCH_SIZE, LR = 8, 4.2e-5     # 6e-5 * sqrt(8/16)
# BATCH_SIZE, LR = 16, 6.0e-5    # base
# BATCH_SIZE, LR = 32, 8.5e-5      # 6e-5 * sqrt(32/16)
# BATCH_SIZE, LR = 64, 1.2e-4    # 6e-5 * sqrt(64/16)
# BATCH_SIZE, LR = 128, 1.7e-4   # 6e-5 * sqrt(128/16)
# BATCH_SIZE, LR = 256, 2.4e-4   # 6e-5 * sqrt(256/16)

# For chinese-roberta
BATCH_SIZE, LR = 256, 8e-5 # 2e-5 × sqrt(256/16) = 8e-5

EPOCHS = 10
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01
MAX_LENGTH = 128
EVAL_STEPS = 200
TEST_SPLIT = 0.005


def load_data(sense_col: str):
    """Load TSV and group by context for listwise training."""
    from collections import defaultdict
    by_context = defaultdict(lambda: {"docs": [], "labels": []})
    with open(DATA_PATH, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            ctx = row["context"]
            by_context[ctx]["docs"].append(row[sense_col])
            by_context[ctx]["labels"].append(float(row["label"]))

    return [
        {"query": ctx, "documents": data["docs"], "labels": data["labels"]}
        for ctx, data in by_context.items()
    ]


def split_by_context(samples, test_ratio):
    """Split grouped samples into train/test."""
    random.shuffle(samples)
    split_idx = int(len(samples) * (1 - test_ratio))
    return samples[:split_idx], samples[split_idx:]


def make_eval_samples(samples):
    """Convert grouped samples to reranking evaluator format."""
    eval_samples = []
    for s in samples:
        positive = [d for d, l in zip(s["documents"], s["labels"]) if l == 1.0]
        negative = [d for d, l in zip(s["documents"], s["labels"]) if l == 0.0]
        if positive and negative:
            eval_samples.append({"query": s["query"], "positive": positive, "negative": negative})
    return eval_samples


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ("sense_en", "sense_zh"):
        print("Usage: python scripts/wsd_training/train_wsd_crossencoder.py <sense_en|sense_zh> [--bce]", file=sys.stderr)
        sys.exit(1)

    sense_col = sys.argv[1]
    use_bce = "--bce" in sys.argv
    suffix = "zh" if sense_col == "sense_zh" else "en"
    OUTPUT_DIR = OUTPUT_BASE / f"wsd_finetuned_crossencoder_{suffix}"

    loss_name = "BCE (pointwise)" if use_bce else "ListNet (listwise)"
    print(f"=== Training with {sense_col} senses, loss={loss_name} ===")
    print(f"Output: {OUTPUT_DIR}")

    print("Loading data...")
    samples = load_data(sense_col)

    train_samples, test_samples = split_by_context(samples, TEST_SPLIT)
    print(f"Train: {len(train_samples)} contexts, Test: {len(test_samples)} contexts")

    if use_bce:
        # Flatten grouped data back to pairs for BCE
        flat_train = []
        for s in train_samples:
            for doc, label in zip(s["documents"], s["labels"]):
                flat_train.append({"sentence1": s["query"], "sentence2": doc, "label": label})
        random.shuffle(flat_train)
        train_dataset = Dataset.from_list(flat_train)
        print(f"Flattened to {len(flat_train)} pairs for BCE")
    else:
        train_dataset = Dataset.from_list([
            {"query": s["query"], "documents": s["documents"], "labels": s["labels"]}
            for s in train_samples
        ])

    evaluator = CrossEncoderRerankingEvaluator(
        samples=make_eval_samples(test_samples),
        name="wsd-test"
    )

    print(f"Loading model: {MODEL_NAME}")
    model = CrossEncoder(MODEL_NAME, max_length=MAX_LENGTH)

    if use_bce:
        loss = BinaryCrossEntropyLoss(model)
    else:
        loss = OOMSafeLambdaLoss(model)

    args = CrossEncoderTrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        learning_rate=LR,
        warmup_ratio=WARMUP_RATIO,
        weight_decay=WEIGHT_DECAY,
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,
        save_steps=EVAL_STEPS,
        logging_steps=100,
        logging_dir=str(OUTPUT_DIR / "runs" / datetime.now().strftime("%Y%m%d_%H%M%S")),
        report_to="tensorboard",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="wsd-test_mrr@10",
        greater_is_better=True,
        bf16=True,
        dataloader_num_workers=4,
    )

    trainer = CrossEncoderTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        loss=loss,
        evaluator=evaluator,
    )

    print("Training...")
    trainer.train()
    trainer.save_model(str(OUTPUT_DIR / "final"))
    print(f"Done! Model saved to {OUTPUT_DIR}")


class OOMSafeLambdaLoss(LambdaLoss):
    """LambdaLoss that catches CUDA OOM and retries with smaller mini-batches."""

    def forward(self, inputs, labels):
        self.mini_batch_size = None
        while True:
            try:
                return super().forward(inputs, labels)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                current = self.mini_batch_size
                if current is None:
                    total_pairs = sum(len(docs) for docs in inputs[1])
                    self.mini_batch_size = max(total_pairs // 2, 1)
                elif current <= 1:
                    raise
                else:
                    self.mini_batch_size = max(current // 2, 1)
                print(f"  ⚠ OOM caught, retrying with mini_batch_size={self.mini_batch_size}")


if __name__ == "__main__":
    main()
