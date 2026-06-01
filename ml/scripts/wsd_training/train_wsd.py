#!/usr/bin/env python3
"""Train a WSD bi-encoder with grouped cross-entropy loss.

For each batch, contexts are grouped by word. Each word's cluster-level sense
definitions are encoded once and reused across all contexts for that word.
Loss is cross-entropy over cosine similarities against the gold label distribution.
"""

import csv
import logging
import math
import os
import random
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, Sampler
from torch.utils.tensorboard import SummaryWriter
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================
CONFIGS = {
    "small": {
        "encoder": "thenlper/gte-small-zh",
        "output_dir": "wsd_finetuned_biencoder_gte_v2_small",
        "batch_size": 64,
        "lr": 5e-5,
        "epochs": 5,
        "eval_steps": 125,
        "temperature": 0.1,
    },
    "base": {
        "encoder": "thenlper/gte-base-zh",
        "output_dir": "wsd_biencoder_gte_base",
        "batch_size": 32,
        "lr": 2e-5,
        "epochs": 5,
        "eval_steps": 250,
        "temperature": 0.1,
    },
    "large": {
        "encoder": "thenlper/gte-large-zh",
        "output_dir": "wsd_finetuned_biencoder_gte_v2_large",
        "batch_size": 16,
        "lr": 1e-5,
        "epochs": 5,
        "eval_steps": 500,
        "temperature": 0.1,
    },
    "bge-base": {
        "encoder": "BAAI/bge-base-zh-v1.5",
        "output_dir": "wsd_finetuned_biencoder_bge_v2_base",
        "batch_size": 32,
        "lr": 2e-5,
        "epochs": 5,
        "eval_steps": 250,
        "temperature": 0.1,
    },
    "bge-large": {
        "encoder": "BAAI/bge-large-zh-v1.5",
        "output_dir": "wsd_finetuned_biencoder_bge_v2_large",
        "batch_size": 16,
        "lr": 1e-5,
        "epochs": 5,
        "eval_steps": 500,
        "temperature": 0.1,
    },
    "small-h100": {
        "encoder": "thenlper/gte-small-zh",
        "output_dir": "wsd_finetuned_biencoder_gte_v2_small",
        "batch_size": 512,
        "lr": 5e-5,
        "epochs": 10,
        "eval_steps": 62,
        "temperature": 0.1,
    },
    "base-h100": {
        "encoder": "thenlper/gte-base-zh",
        "output_dir": "wsd_biencoder_gte_base",
        "batch_size": 256,
        "lr": 2e-5,
        "epochs": 10,
        "eval_steps": 125,
        "temperature": 0.1,
    },
    "large-h100": {
        "encoder": "thenlper/gte-large-zh",
        "output_dir": "wsd_finetuned_biencoder_gte_v2_large",
        "batch_size": 128,
        "lr": 1e-5,
        "epochs": 10,
        "eval_steps": 250,
        "temperature": 0.1,
    },
    "bge-small-h100": {
        "encoder": "BAAI/bge-small-zh-v1.5",
        "output_dir": "wsd_finetuned_biencoder_bge_v2_small",
        "batch_size": 512,
        "lr": 5e-5,
        "epochs": 10,
        "eval_steps": 62,
        "temperature": 0.1,
    },
    "bge-base-h100": {
        "encoder": "BAAI/bge-base-zh-v1.5",
        "output_dir": "wsd_finetuned_biencoder_bge_v2_base",
        "batch_size": 256,
        "lr": 2e-5,
        "epochs": 10,
        "eval_steps": 125,
        "temperature": 0.1,
    },
    "bge-large-h100": {
        "encoder": "BAAI/bge-large-zh-v1.5",
        "output_dir": "wsd_finetuned_biencoder_bge_v2_large",
        "batch_size": 128,
        "lr": 1e-5,
        "epochs": 10,
        "eval_steps": 250,
        "temperature": 0.1,
    },
}

WARMUP_RATIO = 0.1
LOGGING_STEPS = 100
SEED = 42

SCRIPT_DIR = Path(__file__).parent
DATA_PATH = SCRIPT_DIR.parent.parent / "data" / "dataset_v2" / "wsd_dataset_v2.tsv"
# =============================================================================


# =============================================================================
# DATA
# =============================================================================

class WSDExample:
    """A single (word, context) with its cluster-level sense labels."""
    __slots__ = ("word", "context", "senses", "labels")

    def __init__(self, word: str, context: str, senses: list[str], labels: list[float]):
        self.word = word
        self.context = context
        self.senses = senses  # one text per cluster
        self.labels = labels  # 1.0 for correct cluster(s), 0.0 otherwise


def load_data(path: Path) -> tuple[list[WSDExample], list[WSDExample]]:
    """Load cluster-level TSV into WSDExamples.

    Each row is one (word, context, cluster_id) with sense_zh and label.
    We group rows by (word, context) to build one example per context.
    """
    # word -> {cluster_id: sense_zh} (stable sense list per word)
    word_clusters: dict[str, dict[int, str]] = defaultdict(dict)
    # (word, context) -> {cluster_id: label}
    context_labels: dict[tuple[str, str], dict[int, float]] = defaultdict(dict)

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            word, context = row["word"], row["context"]
            cid = int(row["cluster_id"])
            word_clusters[word][cid] = row["sense_zh"]
            context_labels[(word, context)][cid] = float(row["label"])

    examples = []
    for (word, context), cid_labels in context_labels.items():
        cids = sorted(word_clusters[word].keys())
        senses = [word_clusters[word][c] for c in cids]
        labels = [cid_labels.get(c, 0.0) for c in cids]
        n_pos = sum(1 for l in labels if l == 1.0)
        if n_pos == 0 or n_pos == len(labels):
            continue
        assert n_pos == 1, (
            f"Expected 1 positive cluster for ({word}, {context[:40]}...), "
            f"got {n_pos}. Labels: {labels}"
        )
        examples.append(WSDExample(word, context, senses, labels))

    logger.info(f"Loaded {len(examples):,} examples, "
                f"{len(word_clusters):,} words, "
                f"{sum(len(v) for v in word_clusters.values()):,} clusters")

    random.seed(SEED)
    random.shuffle(examples)
    split_idx = int(len(examples) * 0.98)
    return examples[:split_idx], examples[split_idx:]


# =============================================================================
# DATASET & SAMPLER
# =============================================================================

class WSDDataset(Dataset):
    def __init__(self, examples: list[WSDExample]):
        self.examples = examples
        self.word_to_indices: dict[str, list[int]] = defaultdict(list)
        for i, ex in enumerate(examples):
            self.word_to_indices[ex.word].append(i)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


class WordGroupedSampler(Sampler):
    """Yields batches where examples sharing a word are grouped together.

    Maximizes sense-embedding reuse: if a word has N contexts in a batch,
    its K sense definitions are encoded once instead of N times.
    """

    def __init__(self, dataset: WSDDataset, batch_size: int):
        self.dataset = dataset
        self.batch_size = batch_size

    def __iter__(self):
        words = list(self.dataset.word_to_indices.keys())
        random.shuffle(words)
        buffer = []
        for word in words:
            indices = self.dataset.word_to_indices[word][:]
            random.shuffle(indices)
            buffer.extend(indices)
            while len(buffer) >= self.batch_size:
                yield buffer[: self.batch_size]
                buffer = buffer[self.batch_size :]
        if buffer:
            yield buffer

    def __len__(self):
        return math.ceil(len(self.dataset) / self.batch_size)


# =============================================================================
# LOSS COMPUTATION
# =============================================================================

def encode_with_grad(model: SentenceTransformer, texts: list[str]) -> torch.Tensor:
    """Encode texts with gradient tracking (model.encode() uses no_grad)."""
    tokenized = model.tokenize(texts)
    tokenized = {k: v.to(model.device) for k, v in tokenized.items()}
    return model.forward(tokenized)["sentence_embedding"]


def compute_loss(model: SentenceTransformer, batch: list[WSDExample], temperature: float) -> torch.Tensor:
    """Grouped cross-entropy loss.

    All texts (contexts + senses) are encoded in a single forward pass,
    then sliced back per word for the per-word softmax.
    """
    device = model.device

    # Group by word
    word_groups: dict[str, list[WSDExample]] = defaultdict(list)
    for ex in batch:
        word_groups[ex.word].append(ex)

    # Build one big text list: [all_contexts..., all_senses...]
    # Track slicing info per word
    all_texts = []
    word_meta = []  # (word, ctx_start, ctx_count, sense_start, sense_count, examples)
    for word, examples in word_groups.items():
        contexts = [ex.context for ex in examples]
        senses = examples[0].senses

        ctx_start = len(all_texts)
        all_texts.extend(contexts)
        sense_start = len(all_texts)
        all_texts.extend(senses)

        word_meta.append((word, ctx_start, len(contexts), sense_start, len(senses), examples))

    # Single forward pass for everything
    all_embs = encode_with_grad(model, all_texts)
    all_embs = F.normalize(all_embs, p=2, dim=1)

    # Compute per-word cross-entropy
    losses = []
    for word, ctx_start, ctx_count, sense_start, sense_count, examples in word_meta:
        ctx_embs = all_embs[ctx_start : ctx_start + ctx_count]
        sense_embs = all_embs[sense_start : sense_start + sense_count]

        logits = (ctx_embs @ sense_embs.T) / temperature

        label_matrix = torch.tensor(
            [ex.labels for ex in examples], dtype=torch.float, device=device
        )
        label_matrix = label_matrix / label_matrix.sum(dim=1, keepdim=True)

        log_probs = F.log_softmax(logits, dim=1)
        loss = -(label_matrix * log_probs).sum(dim=1).mean()
        losses.append(loss)

    return torch.stack(losses).mean()


# =============================================================================
# EVALUATION
# =============================================================================

@torch.no_grad()
def evaluate(model: SentenceTransformer, dataset: WSDDataset, temperature: float) -> dict[str, float]:
    """Evaluate accuracy and loss on the dataset."""
    correct = 0
    total = 0
    total_loss = 0.0
    num_words = 0

    word_groups: dict[str, list[WSDExample]] = defaultdict(list)
    for ex in dataset.examples:
        word_groups[ex.word].append(ex)

    device = model.device
    for word, examples in word_groups.items():
        senses = examples[0].senses
        contexts = [ex.context for ex in examples]

        sense_embs = model.encode(senses, convert_to_tensor=True, show_progress_bar=False)
        ctx_embs = model.encode(contexts, convert_to_tensor=True, show_progress_bar=False)
        sense_embs = F.normalize(sense_embs, p=2, dim=1)
        ctx_embs = F.normalize(ctx_embs, p=2, dim=1)

        # Accuracy
        preds = (ctx_embs @ sense_embs.T).argmax(dim=1)
        for i, ex in enumerate(examples):
            if ex.labels[preds[i].item()] == 1.0:
                correct += 1
            total += 1

        # Loss
        logits = (ctx_embs @ sense_embs.T) / temperature
        label_matrix = torch.tensor(
            [ex.labels for ex in examples], dtype=torch.float, device=device
        )
        label_matrix = label_matrix / label_matrix.sum(dim=1, keepdim=True)
        log_probs = F.log_softmax(logits, dim=1)
        loss = -(label_matrix * log_probs).sum(dim=1).mean()
        total_loss += loss.item()
        num_words += 1

    return {
        "accuracy": correct / total if total else 0.0,
        "loss": total_loss / max(num_words, 1),
        "correct": correct,
        "total": total,
    }


# =============================================================================
# TRAINING LOOP
# =============================================================================

def get_lr(step: int, total_steps: int, warmup_steps: int, max_lr: float) -> float:
    """Linear warmup then cosine decay."""
    if step < warmup_steps:
        return max_lr * step / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return max_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Train WSD bi-encoder with model presets",
        epilog="Examples:\n"
               "  python train_wsd.py small\n"
               "  python train_wsd.py base\n"
               "  python train_wsd.py large-h100\n"
               "  python train_wsd.py bge-base-h100\n",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("config", choices=CONFIGS.keys(),
                        help="Model config preset")
    args = parser.parse_args()

    cfg = CONFIGS[args.config]
    _train_one(args.config, cfg)


def _train_one(config_name: str, cfg: dict):
    os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")

    base_model = cfg["encoder"]
    batch_size = cfg["batch_size"]
    learning_rate = cfg["lr"]
    num_epochs = cfg["epochs"]
    eval_steps = cfg["eval_steps"]
    temperature = cfg["temperature"]
    output_dir = SCRIPT_DIR.parent.parent / "models" / cfg["output_dir"]

    logger.info(f"=== Config: {config_name} | {base_model} ===")
    logger.info(f"  batch={batch_size}, lr={learning_rate}, epochs={num_epochs}, "
                f"eval_steps={eval_steps}, temp={temperature}")

    logger.info("Loading data...")
    train_examples, eval_examples = load_data(DATA_PATH)
    train_dataset = WSDDataset(train_examples)
    eval_dataset = WSDDataset(eval_examples)
    logger.info(f"Train: {len(train_dataset):,}, Eval: {len(eval_dataset):,}")
    logger.info(f"Train words: {len(train_dataset.word_to_indices):,}, "
                f"Eval words: {len(eval_dataset.word_to_indices):,}")

    logger.info(f"Loading model: {base_model}")
    model = SentenceTransformer(base_model)
    if torch.backends.mps.is_available():
        model = model.float()

    # CUDA optimizations
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        logger.info("  CUDA: bf16 autocast + cudnn.benchmark enabled")

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    sampler = WordGroupedSampler(train_dataset, batch_size)
    total_steps = len(sampler) * num_epochs
    warmup_steps = int(total_steps * WARMUP_RATIO)

    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(output_dir / "logs" / run_name))
    best_acc = 0.0
    global_step = 0

    logger.info(f"Total steps: {total_steps:,}, Warmup: {warmup_steps:,}")
    logger.info("Starting training...")

    use_amp = torch.cuda.is_available()

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        recent_losses = deque(maxlen=50)

        pbar = tqdm(iter(sampler), desc=f"Epoch {epoch+1}/{num_epochs}", total=len(sampler))
        for batch_indices in pbar:
            batch = [train_dataset[i] for i in batch_indices]

            lr = get_lr(global_step, total_steps, warmup_steps, learning_rate)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            optimizer.zero_grad()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                loss = compute_loss(model, batch, temperature)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            loss_val = loss.item()
            epoch_loss += loss_val
            num_batches += 1
            global_step += 1
            recent_losses.append(loss_val)

            if global_step % LOGGING_STEPS == 0:
                avg_loss = sum(recent_losses) / len(recent_losses)
                pbar.set_postfix(loss=f"{avg_loss:.4f}", lr=f"{lr:.2e}", step=global_step)
                writer.add_scalar("train/loss", loss_val, global_step)
                writer.add_scalar("train/loss_avg50", avg_loss, global_step)
                writer.add_scalar("train/lr", lr, global_step)

            if global_step % eval_steps == 0:
                model.eval()
                metrics = evaluate(model, eval_dataset, temperature)
                logger.info(f"Step {global_step}: eval_acc={metrics['accuracy']:.4f} "
                            f"eval_loss={metrics['loss']:.4f} "
                            f"({metrics['correct']}/{metrics['total']})")
                writer.add_scalar("eval/accuracy", metrics["accuracy"], global_step)
                writer.add_scalar("eval/loss", metrics["loss"], global_step)
                if metrics["accuracy"] > best_acc:
                    best_acc = metrics["accuracy"]
                    model.save_pretrained(str(run_dir / "best"))
                    logger.info(f"New best model saved")
                model.train()

        logger.info(f"Epoch {epoch+1} done. Avg loss: {epoch_loss / max(num_batches, 1):.4f}")

        # End-of-epoch eval
        model.eval()
        metrics = evaluate(model, eval_dataset, temperature)
        logger.info(f"Epoch {epoch+1} eval: acc={metrics['accuracy']:.4f} "
                    f"loss={metrics['loss']:.4f} ({metrics['correct']}/{metrics['total']})")
        writer.add_scalar("eval/accuracy", metrics["accuracy"], global_step)
        writer.add_scalar("eval/loss", metrics["loss"], global_step)
        if metrics["accuracy"] > best_acc:
            best_acc = metrics["accuracy"]
            model.save_pretrained(str(run_dir / "best"))
            logger.info(f"New best model saved")

    model.eval()
    metrics = evaluate(model, eval_dataset, temperature)
    logger.info(f"Final eval_acc={metrics['accuracy']:.4f} eval_loss={metrics['loss']:.4f}")
    # Save best model as final
    best_path = run_dir / "best"
    final_path = run_dir / "final"
    if best_path.exists():
        import shutil
        shutil.copytree(str(best_path), str(final_path), dirs_exist_ok=True)
        logger.info(f"Copied best model to {final_path}")
    else:
        model.save_pretrained(str(final_path))
    logger.info(f"Done! Best acc: {best_acc:.4f}. Run dir: {run_dir}")
    writer.close()


if __name__ == "__main__":
    main()
