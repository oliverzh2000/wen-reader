#!/usr/bin/env python3
"""Fine-tune ckiplab/bert-base-chinese-ws on a refined CWS dataset.

Reads space-separated segmented text and converts to character-level B/I labels
(B = word start, I = word continuation). Uses HuggingFace Trainer with standard
cross-entropy loss only — no lattice, Viterbi, or OOV penalty in training.
"""

import logging
import random
import warnings
from pathlib import Path

import torch
from torch.utils.data import Dataset as TorchDataset
from transformers import (
    BertForTokenClassification,
    BertTokenizerFast,
    Trainer,
    TrainingArguments,
)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================
MODEL_NAME = "ckiplab/bert-base-chinese-ws"

NUM_EPOCHS = 3
BATCH_SIZE = 32
LEARNING_RATE = 2e-5
WARMUP_RATIO = 0.1
EVAL_SPLIT = 0.02  # 2% held out for eval
EVAL_STEPS = 250
SEED = 42

SCRIPT_DIR = Path(__file__).parent
DATASET_PATH = SCRIPT_DIR.parent.parent / "data" / "cws_dataset.txt"
OUTPUT_DIR = SCRIPT_DIR.parent.parent / "models" / "cws_finetuned"
# =============================================================================

LABEL_B = 0  # word start
LABEL_I = 1  # word continuation
IGNORE_LABEL = -100  # ignored by cross-entropy loss (special tokens)


def text_to_bi_labels(segmented_line: str) -> tuple[list[str], list[int]]:
    """Convert a space-separated segmented line to characters and B/I labels.

    Example:
        "中华人民共和国 是 一个" → (['中','华','人','民','共','和','国','是','一','个'],
                                     [B, I, I, I, I, I, I, B, B, I])
    """
    chars: list[str] = []
    labels: list[int] = []
    for word in segmented_line.split():
        if not word:
            continue
        for i, ch in enumerate(word):
            chars.append(ch)
            labels.append(LABEL_B if i == 0 else LABEL_I)
    return chars, labels


class CWSDataset(TorchDataset):
    """Dataset that reads space-separated segmented text and produces
    tokenized inputs with aligned B/I labels for BertForTokenClassification."""

    def __init__(self, path: Path, tokenizer: BertTokenizerFast, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples: list[tuple[list[str], list[int]]] = []

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                chars, labels = text_to_bi_labels(line)
                if chars:
                    self.samples.append((chars, labels))

        logger.info(f"Loaded {len(self.samples):,} sentences from {path}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        chars, char_labels = self.samples[idx]

        # Tokenize character-by-character with offset mapping for alignment.
        # is_split_into_words=True treats each element as a pre-tokenized token.
        encoding = self.tokenizer(
            chars,
            is_split_into_words=True,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        # Align labels with tokenizer output using word_ids.
        # word_ids maps each token position to the original word (char) index,
        # or None for special tokens ([CLS], [SEP], [PAD]).
        word_ids = encoding.word_ids(batch_index=0)
        aligned_labels: list[int] = []
        prev_word_id = None
        for word_id in word_ids:
            if word_id is None:
                # Special token — ignore in loss
                aligned_labels.append(IGNORE_LABEL)
            elif word_id != prev_word_id:
                # First subword token of this character — use its label
                aligned_labels.append(char_labels[word_id])
            else:
                # Subsequent subword token of same character — ignore in loss
                aligned_labels.append(IGNORE_LABEL)
            prev_word_id = word_id

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(aligned_labels, dtype=torch.long),
        }


def main():
    # Device selection
    if torch.cuda.is_available():
        device_msg = f"CUDA ({torch.cuda.get_device_name(0)})"
    elif torch.backends.mps.is_available():
        device_msg = "MPS (Apple Silicon)"
    else:
        device_msg = "CPU"
        warnings.warn(
            "No GPU detected — training will run on CPU and may be very slow.",
            stacklevel=2,
        )
    logger.info(f"Device: {device_msg}")

    # Load tokenizer and model
    logger.info(f"Loading pretrained model: {MODEL_NAME}")
    tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)
    model = BertForTokenClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        id2label={0: "B", 1: "I"},
        label2id={"B": 0, "I": 1},
    )

    # Load dataset and split train/eval
    logger.info(f"Loading dataset: {DATASET_PATH}")
    full_dataset = CWSDataset(DATASET_PATH, tokenizer)

    n = len(full_dataset)
    indices = list(range(n))
    random.seed(SEED)
    random.shuffle(indices)
    split = int(n * EVAL_SPLIT)
    eval_indices, train_indices = indices[:split], indices[split:]

    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    eval_dataset = torch.utils.data.Subset(full_dataset, eval_indices)
    logger.info(f"Train: {len(train_dataset):,}, Eval: {len(eval_dataset):,}")

    # Training arguments
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        warmup_ratio=WARMUP_RATIO,
        seed=SEED,
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,
        save_strategy="steps",
        save_steps=EVAL_STEPS,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=50,
        report_to=["tensorboard"],
        logging_dir=str(OUTPUT_DIR / "logs"),
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    logger.info("Starting training...")
    trainer.train()

    # Save final checkpoint
    final_dir = OUTPUT_DIR / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    logger.info(f"Done! Saved fine-tuned model to {final_dir}")


if __name__ == "__main__":
    main()
