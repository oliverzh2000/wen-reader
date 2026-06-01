#!/bin/bash
set -e
cd "$(dirname "$0")/../.."
source .venv/bin/activate

# Generate LLM task batches (2-3 char spans go to LLM, 4+ char blind-merged)
# python scripts/cws_training/build_dataset.py \
#     --freq-cap 10 \
#     --batch-size 50 \
#     gen-tasks

# Assemble final dataset (run after LLM generation is complete)
python scripts/cws_training/build_dataset.py \
    --freq-cap 10 \
    assemble
