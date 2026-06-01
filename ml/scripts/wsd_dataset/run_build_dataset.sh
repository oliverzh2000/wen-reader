#!/bin/bash
set -e
cd "$(dirname "$0")/../.."

# Phase 1: Generate LLM task batches (sense merging for polysemous entries)
# python scripts/wsd_dataset/build_dataset.py \
#     --batch-size 50 \
#     gen-tasks

# Phase 1: Assemble entries_after_merging.json (run after LLM generation is complete)
# python scripts/wsd_dataset/build_dataset.py assemble

# Phase 2: Scan corpora → sentence_index.json
python scripts/wsd_dataset/build_sentence_index.py \
    --cws-batch-size 64

# Merge old LLM-generated wsd_dataset.tsv with sense merging results
python scripts/wsd_dataset/merge_wsd_dataset.py
