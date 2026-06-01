---
license: mit
language:
- zh
tags:
- chinese
- word-segmentation
- word-sense-disambiguation
- coreml
- ios
- epub-reader
datasets:
- wyy209/MiCLS
base_model:
- hfl/chinese-electra-180g-base-discriminator
- hfl/chinese-electra-180g-small-discriminator
- thenlper/gte-base-zh
---

# Wen Reader — On-Device Chinese NLP Models

Model weights for [Wen Reader](https://github.com/oliverzh2000/wen-reader), an open-source Chinese EPUB reader for iOS with popup dictionary.

These models run on-device via CoreML to provide context-aware Chinese word segmentation and word sense disambiguation.

## Models

### cws-span-scorer-electra-base

Chinese Word Segmentation model using span scoring.

- **Architecture:** ELECTRA-base (12 layers, 768 hidden, 12 heads) + span scoring MLP head
- **Base model:** [hfl/chinese-electra-180g-base-discriminator](https://huggingface.co/hfl/chinese-electra-180g-base-discriminator)
- **Parameters:** ~102M (encoder) + span head
- **Task:** Given a sentence, score candidate word spans to find optimal segmentation
- **Eval:** 90/97 perfect segmentation on hand-curated test set (93/97 including acceptable over-splits)

### cws-span-scorer-electra-small

Smaller/faster variant of the above for lower-end devices.

- **Architecture:** ELECTRA-small (12 layers, 256 hidden, 4 heads) + span scoring MLP head
- **Base model:** [hfl/chinese-electra-180g-small-discriminator](https://huggingface.co/hfl/chinese-electra-180g-small-discriminator)
- **Parameters:** ~12M (encoder) + span head
- **Eval:** 85/97 perfect (91/97 including acceptable over-splits)

### wsd-biencoder-gte-base

Word Sense Disambiguation bi-encoder for selecting the correct dictionary definition in context.

- **Architecture:** GTE-base-zh fine-tuned as a bi-encoder (SentenceTransformers)
- **Base model:** [thenlper/gte-base-zh](https://huggingface.co/thenlper/gte-base-zh)
- **Parameters:** ~102M
- **Task:** Encode context sentence and candidate sense labels, rank by cosine similarity
- **Eval:** Top-1 accuracy 88.7%, Top-3 99.6%, MRR 0.936 on 239 hand-curated disambiguation examples

## Usage

These models are used by the Wen Reader iOS app via CoreML conversion.

To build the app from a fresh clone:

```bash
cd ml

# 1. Download model weights from HF Hub
uv run python scripts/download_models.py

# 2. Export to CoreML
uv run python scripts/export_coreml.py span --model-dir models/cws_span_scorer_electra_base/final
uv run python scripts/export_coreml.py wsd --model-dir models/wsd_biencoder_gte_base/final

# 3. Bundle into app resources (vocab, CoreML packages, CEDICT database)
./scripts/run_pipeline.sh bundle
```

Or use the pipeline script:

```bash
./scripts/run_pipeline.sh download
./scripts/run_pipeline.sh export
./scripts/run_pipeline.sh bundle
```

## Training

### CWS Span Scorer

Fine-tuned on a custom dataset built from:
- [ICWB2](https://github.com/yuikns/icwb2-data) segmentation corpus (MSR, auto-matched where greedy segmenter agrees with gold, plus LLM-annotated disagreement cases)
- Chinese [Wikipedia](https://huggingface.co/datasets/wikimedia/wikipedia) and [OpenSubtitles](https://huggingface.co/datasets/FradSer/OpenSubtitles-en-zh-cn-20m) sentences, LLM-annotated (Claude Opus)

Sentences with ambiguous segmentation boundaries (where multiple valid CEDICT words overlap) are
identified, then an LLM annotates the correct segmentation. The model scores all candidate word
spans at each ambiguous position using a span scoring MLP head (taking span boundary token
representations + learned width embeddings as input). A dynamic programming decoder finds the
optimal segmentation at inference time. Training uses cross-entropy loss over candidate spans at
each ambiguous position. Encoder and head use discriminative learning rates.

### WSD Bi-encoder

Fine-tuned on CC-CEDICT sense clusters with data from:
- [MiCLS](https://huggingface.co/datasets/wyy209/MiCLS) WSD corpus (mapped to CEDICT senses via BGE embedding similarity)
- LLM-generated context examples for each sense cluster (Claude Opus and Sonnet)
- LLM-annotated ebook sentences segmented by the CWS span scorer (Claude Sonnet)

Sense clusters are derived from CC-CEDICT entries with related senses merged using LLM-assisted
clustering. Chinese sense labels are LLM-translated from the English CEDICT definitions.

Training uses grouped cross-entropy loss: contexts and sense labels are encoded by the same model,
cosine similarity is computed between context embeddings and sense embeddings, and the loss is
cross-entropy over the similarity scores (with temperature scaling at τ=0.1). Contexts are grouped
by word in each batch so sense embeddings are encoded once and reused across all contexts for that word.