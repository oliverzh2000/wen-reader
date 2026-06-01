#!/usr/bin/env python3
"""Fine-tune Chinese ELECTRA/BERT as a span scorer for CWS.

Reads JSONL from build_dataset.py assemble, groups examples by sentence,
batches the encoder forward pass, and trains the span head with
cross-entropy over candidate spans at each ambiguous position.

Uses HuggingFace Trainer with a custom model that computes its own loss.

Usage:
    python train_span_scorer.py small          # ELECTRA small (12M)
    python train_span_scorer.py base           # ELECTRA base (102M)
    python train_span_scorer.py large          # ELECTRA large (324M)
    python train_span_scorer.py small --epochs 2 --lr 3e-5
"""

import argparse
import json
import logging
import os
import random
import warnings
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

# Disable MPS high watermark to avoid OOM on Apple Silicon
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import (
    BertModel,
    BertTokenizerFast,
    ElectraModel,
    Trainer,
    TrainingArguments,
)
from transformers.utils import ModelOutput

from span_scorer import SpanScoringHead, MAX_WORD_LEN

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

# =============================================================================
# MODEL CONFIGS
# =============================================================================
CONFIGS = {
    "small": {
        "encoder": "hfl/chinese-electra-180g-small-discriminator",
        "output_dir": "span_scorer_electra_small2",
        "batch_size": 128,
        "encoder_lr": 1e-4,
        "head_lr": 5e-4,
        "eval_steps": 350,
        "epochs": 1,
        "dataset": "v2",
    },
    "small-v1": {
        "encoder": "hfl/chinese-electra-180g-small-discriminator",
        "output_dir": "cws_span_scorer_electra_small",
        "batch_size": 128,
        "encoder_lr": 1e-4,
        "head_lr": 3e-4,
        "eval_steps": 125,
        "epochs": 4,
        "dataset": "v1",
    },
    "base": {
        "encoder": "hfl/chinese-electra-180g-base-discriminator",
        "output_dir": "span_scorer_electra_base2",
        "batch_size": 64,
        "encoder_lr": 1e-4,
        "head_lr": 5e-4,
        "eval_steps": 700,
        "epochs": 1,
        "dataset": "v2",
    },
    "base-v1": {
        "encoder": "hfl/chinese-electra-180g-base-discriminator",
        "output_dir": "cws_span_scorer_electra_base",
        "batch_size": 64,
        "encoder_lr": 5e-5,
        "head_lr": 5e-4,
        "eval_steps": 100,
        "epochs": 2,
        "dataset": "v1",
    },
    "large": {
        "encoder": "hfl/chinese-electra-180g-large-discriminator",
        "output_dir": "span_scorer_electra_large2",
        "batch_size": 32,
        "encoder_lr": 5e-5,
        "head_lr": 5e-4,
        "eval_steps": 1400,
        "epochs": 1,
        "dataset": "v2",
    },
}

# =============================================================================
# SHARED DEFAULTS
# =============================================================================
WARMUP_RATIO = 0.2
WEIGHT_DECAY = 0.05
EVAL_SPLIT = 0.02
SEED = 42
FREEZE_ENCODER = False

SCRIPT_DIR = Path(__file__).parent
DATASETS = {
    "v2": SCRIPT_DIR.parent.parent / "data" / "dataset_v2" / "span_scorer_dataset_v2.jsonl",
    "v1": SCRIPT_DIR.parent.parent / "data" / "span_scorer_dataset.jsonl",
}
# =============================================================================


@dataclass
class SpanScorerOutput(ModelOutput):
    loss: torch.Tensor | None = None
    n_correct: torch.Tensor | None = None
    n_total: torch.Tensor | None = None


class SpanScorerForTraining(torch.nn.Module):
    """
    Wraps encoder + span head into a single module that computes loss.

    Batched: encoder runs on (batch, seq_len), then we loop over sentences
    to score their variable-length span candidates and compute CE loss.
    """

    def __init__(self, encoder: BertModel, head: SpanScoringHead):
        super().__init__()
        self.encoder = encoder
        self.head = head

    def forward(
        self,
        input_ids,          # (batch, seq_len)
        attention_mask,     # (batch, seq_len)
        span_starts,        # (batch, max_spans) — padded with 0
        span_ends,          # (batch, max_spans)
        span_widths,        # (batch, max_spans)
        span_mask,          # (batch, max_spans) — 1 for real spans, 0 for padding
        gold_indices,       # (batch, max_positions)
        position_offsets,   # (batch, max_offsets)
        n_positions,        # (batch,) — actual number of positions per sentence
        n_spans,            # (batch,) — actual number of spans per sentence
        **kwargs,
    ):
        # Batched encoder forward
        hidden = self.encoder(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state  # (batch, seq_len, hidden_dim)

        batch_size = hidden.shape[0]
        total_loss = torch.tensor(0.0, device=input_ids.device)
        total_positions = 0
        total_correct = 0

        for b in range(batch_size):
            ns = n_spans[b].item()
            np_ = n_positions[b].item()
            if np_ == 0:
                continue

            h = hidden[b]  # (seq_len, hidden_dim)
            s_starts = span_starts[b, :ns]
            s_ends = span_ends[b, :ns]
            s_widths = span_widths[b, :ns]

            scores = self.head(h, s_starts, s_ends, s_widths)  # (ns,)

            offsets = position_offsets[b]
            golds = gold_indices[b]
            for p in range(np_):
                lo = offsets[p].item()
                hi = offsets[p + 1].item()
                target = golds[p].item() - lo
                total_loss = total_loss + F.cross_entropy(
                    scores[lo:hi].unsqueeze(0),
                    torch.tensor([target], device=scores.device),
                )
                pred = scores[lo:hi].argmax().item()
                if pred == target:
                    total_correct += 1
            total_positions += np_

        loss = total_loss / max(total_positions, 1)
        return SpanScorerOutput(
            loss=loss,
            n_correct=torch.tensor(total_correct),
            n_total=torch.tensor(total_positions),
        )


class SpanDataset(Dataset):
    """Each item is one sentence with all its ambiguous positions and
    candidate spans, pre-tokenized and packed into tensors."""

    def __init__(self, items: list[dict], tokenizer: BertTokenizerFast):
        self.items = items
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        chars = list(item["sentence"])

        encoding = self.tokenizer(
            chars,
            is_split_into_words=True,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=False,
        )

        # Effective character limit: seq_len includes [CLS] and [SEP],
        # so max char index is seq_len - 2 (0-indexed).
        seq_len = encoding["input_ids"].shape[1]
        max_char_idx = seq_len - 2  # chars that fit within truncated window

        # MacBERT is char-level: token index = char index + 1 (for [CLS]).
        # Data must be clean (no invisible Unicode chars) — enforced by
        # build_dataset.py assemble.
        all_starts, all_ends, all_widths = [], [], []
        gold_indices = []
        position_offsets = [0]

        for pos in item["positions"]:
            start = pos["start"]
            gold_word = pos["gold_word"]
            candidates = pos["candidates"]
            if len(candidates) < 2:
                continue

            # Skip positions that fall outside the truncated window
            if start >= max_char_idx:
                continue

            local_gold = -1
            for word in candidates:
                w_len = len(word)
                if w_len < 1 or w_len > MAX_WORD_LEN:
                    continue
                # Skip candidates whose end falls outside truncated window
                if start + w_len - 1 >= max_char_idx:
                    continue
                if word == gold_word:
                    local_gold = len(all_starts)
                all_starts.append(start + 1)
                all_ends.append(start + w_len - 1 + 1)
                all_widths.append(w_len)

            if local_gold < 0:
                trim = len(all_starts) - position_offsets[-1]
                del all_starts[-trim:]
                del all_ends[-trim:]
                del all_widths[-trim:]
                continue

            gold_indices.append(local_gold)
            position_offsets.append(len(all_starts))

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "span_starts": torch.tensor(all_starts, dtype=torch.long),
            "span_ends": torch.tensor(all_ends, dtype=torch.long),
            "span_widths": torch.tensor(all_widths, dtype=torch.long),
            "gold_indices": torch.tensor(gold_indices, dtype=torch.long),
            "position_offsets": torch.tensor(position_offsets, dtype=torch.long),
        }


def collate_fn(batch):
    """Pad variable-length span tensors into a proper batch."""
    # Pad input_ids and attention_mask
    max_seq = max(item["input_ids"].shape[0] for item in batch)
    input_ids = torch.zeros(len(batch), max_seq, dtype=torch.long)
    attention_mask = torch.zeros(len(batch), max_seq, dtype=torch.long)
    for i, item in enumerate(batch):
        seq_len = item["input_ids"].shape[0]
        input_ids[i, :seq_len] = item["input_ids"]
        attention_mask[i, :seq_len] = item["attention_mask"]

    # Pad span arrays
    max_spans = max(item["span_starts"].shape[0] for item in batch)
    max_spans = max(max_spans, 1)  # avoid zero-size tensors
    max_positions = max(item["gold_indices"].shape[0] for item in batch)
    max_positions = max(max_positions, 1)
    max_offsets = max(item["position_offsets"].shape[0] for item in batch)
    max_offsets = max(max_offsets, 1)

    span_starts = torch.zeros(len(batch), max_spans, dtype=torch.long)
    span_ends = torch.zeros(len(batch), max_spans, dtype=torch.long)
    span_widths = torch.ones(len(batch), max_spans, dtype=torch.long)  # 1 to avoid 0-index in width_embed
    span_mask = torch.zeros(len(batch), max_spans, dtype=torch.bool)
    gold_indices = torch.zeros(len(batch), max_positions, dtype=torch.long)
    position_offsets = torch.zeros(len(batch), max_offsets, dtype=torch.long)
    n_positions = torch.zeros(len(batch), dtype=torch.long)
    n_spans = torch.zeros(len(batch), dtype=torch.long)

    for i, item in enumerate(batch):
        ns = item["span_starts"].shape[0]
        np_ = item["gold_indices"].shape[0]
        no = item["position_offsets"].shape[0]

        span_starts[i, :ns] = item["span_starts"]
        span_ends[i, :ns] = item["span_ends"]
        span_widths[i, :ns] = item["span_widths"]
        span_mask[i, :ns] = True
        gold_indices[i, :np_] = item["gold_indices"]
        position_offsets[i, :no] = item["position_offsets"]
        n_positions[i] = np_
        n_spans[i] = ns

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "span_starts": span_starts,
        "span_ends": span_ends,
        "span_widths": span_widths,
        "span_mask": span_mask,
        "gold_indices": gold_indices,
        "position_offsets": position_offsets,
        "n_positions": n_positions,
        "n_spans": n_spans,
    }


def load_grouped_data(path: Path) -> list[dict]:
    """Load JSONL and group examples by sentence."""
    grouped: dict[str, list[dict]] = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            grouped[ex["sentence"]].append(ex)

    items = []
    for sent, examples in grouped.items():
        positions = [
            {"start": ex["start"], "gold_word": ex["gold_word"],
             "candidates": ex["candidates"]}
            for ex in examples
        ]
        items.append({"sentence": sent, "positions": positions})
    return items


class SpanScorerTrainer(Trainer):
    """Trainer subclass with discriminative learning rates and eval accuracy."""

    def __init__(self, encoder_lr: float, head_lr: float, **kwargs):
        self.encoder_lr = encoder_lr
        self.head_lr = head_lr
        super().__init__(**kwargs)
        # Tell Trainer these keys are "labels" so it doesn't skip
        # loss computation during eval (default expects a 'labels' key).
        self.label_names = ["gold_indices", "position_offsets", "n_positions",
                            "n_spans", "span_mask"]

    def create_optimizer(self):
        self.optimizer = torch.optim.AdamW([
            {"params": self.model.encoder.parameters(), "lr": self.encoder_lr},
            {"params": self.model.head.parameters(), "lr": self.head_lr},
        ])
        return self.optimizer

    def evaluation_loop(self, *args, **kwargs):
        output = super().evaluation_loop(*args, **kwargs)
        # Aggregate accuracy from model outputs collected during eval.
        # The parent class stores per-batch losses but not custom fields,
        # so we run a quick pass ourselves.
        model = self.model
        model.eval()
        dataloader = self.get_eval_dataloader()
        total_correct = 0
        total_positions = 0
        with torch.no_grad():
            for batch in dataloader:
                batch = {k: v.to(self.args.device) for k, v in batch.items()}
                out = model(**batch)
                total_correct += out.n_correct.item()
                total_positions += out.n_total.item()
        acc = total_correct / max(total_positions, 1)
        output.metrics["eval_accuracy"] = acc
        logger.info(f"Eval accuracy: {acc:.4f} ({total_correct}/{total_positions})")
        return output


def save_checkpoint(model, tokenizer, save_dir: Path):
    """Save encoder, head weights, and tokenizer."""
    save_dir.mkdir(parents=True, exist_ok=True)
    model.encoder.save_pretrained(str(save_dir))
    tokenizer.save_pretrained(str(save_dir))
    torch.save(model.head.state_dict(), save_dir / "span_head.pt")


def main():
    parser = argparse.ArgumentParser(
        description="Train span scorer with model presets",
        epilog="Examples:\n"
               "  python train_span_scorer.py small\n"
               "  python train_span_scorer.py base --epochs 3 --lr 3e-5\n"
               "  python train_span_scorer.py large --batch-size 16\n",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("config", choices=CONFIGS.keys(),
                        help="Model config preset (small/base/large)")
    parser.add_argument("--epochs", type=int, default=None, help="Override num epochs")
    parser.add_argument("--lr", type=float, default=None, help="Override encoder LR")
    parser.add_argument("--head-lr", type=float, default=None, help="Override head LR")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--eval-steps", type=int, default=None, help="Override eval interval")
    parser.add_argument("--freeze-encoder", action="store_true", help="Freeze encoder, train head only")
    parser.add_argument("--dataset", choices=DATASETS.keys(), default=None,
                        help="Override dataset (v1=old, v2=new ICWB2)")
    args = parser.parse_args()

    cfg = CONFIGS[args.config]
    encoder_name = cfg["encoder"]
    output_dir = SCRIPT_DIR.parent.parent / "models" / cfg["output_dir"]
    batch_size = args.batch_size or cfg["batch_size"]
    encoder_lr = args.lr or cfg["encoder_lr"]
    head_lr = args.head_lr or cfg["head_lr"]
    num_epochs = args.epochs or cfg["epochs"]
    eval_steps = args.eval_steps or cfg["eval_steps"]
    freeze_encoder = args.freeze_encoder or FREEZE_ENCODER
    dataset_key = args.dataset or cfg["dataset"]
    dataset_path = DATASETS[dataset_key]

    logger.info(f"Config: {args.config} | {encoder_name}")
    logger.info(f"  dataset={dataset_key}, epochs={num_epochs}, batch={batch_size}, enc_lr={encoder_lr}, head_lr={head_lr}, eval_steps={eval_steps}")

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
    logger.info(f"Loading pretrained encoder: {encoder_name}")
    tokenizer = BertTokenizerFast.from_pretrained(encoder_name)
    # Support both BertModel and ElectraModel
    try:
        encoder = ElectraModel.from_pretrained(encoder_name)
    except Exception:
        try:
            encoder = BertModel.from_pretrained(encoder_name)
        except Exception:
            from transformers import BertForTokenClassification
            full_model = BertForTokenClassification.from_pretrained(encoder_name)
            encoder = full_model.bert
    head = SpanScoringHead(hidden_dim=encoder.config.hidden_size)
    model = SpanScorerForTraining(encoder, head)

    if freeze_encoder:
        logger.info("Freezing encoder — training span head only")
        for p in model.encoder.parameters():
            p.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"Parameters: {trainable:,} trainable / {total:,} total")

    # Load dataset and split train/eval
    logger.info(f"Loading dataset: {dataset_path}")
    all_items = load_grouped_data(dataset_path)
    logger.info(f"Loaded {len(all_items):,} sentences")

    n = len(all_items)
    random.seed(SEED)
    random.shuffle(all_items)
    split = max(1, int(n * EVAL_SPLIT))
    eval_items, train_items = all_items[:split], all_items[split:]

    train_dataset = SpanDataset(train_items, tokenizer)
    eval_dataset = SpanDataset(eval_items, tokenizer)
    logger.info(f"Train: {len(train_dataset):,}, Eval: {len(eval_dataset):,}")

    # Training arguments
    output_dir.mkdir(parents=True, exist_ok=True)
    from datetime import datetime
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=encoder_lr,
        warmup_ratio=WARMUP_RATIO,
        weight_decay=WEIGHT_DECAY,
        seed=SEED,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=eval_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=50,
        report_to=["tensorboard"],
        logging_dir=str(output_dir / "logs" / run_name),
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=0,
        remove_unused_columns=False,
    )

    trainer = SpanScorerTrainer(
        encoder_lr=head_lr if freeze_encoder else encoder_lr,
        head_lr=head_lr,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collate_fn,
    )

    logger.info("Starting training...")
    trainer.train()

    # Save final checkpoint
    final_dir = output_dir / "final"
    save_checkpoint(model, tokenizer, final_dir)
    logger.info(f"Done! Saved fine-tuned model to {final_dir}")


if __name__ == "__main__":
    main()
