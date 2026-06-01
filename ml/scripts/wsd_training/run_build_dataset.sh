#!/bin/bash
set -e
cd "$(dirname "$0")/../.."

# # Generate LLM task files
# python scripts/wsd_training/build_dataset.py \
#     --min-examples 2 \
#     --max-examples 10 \
#     --max-senses 30000 \
#     --senses-per-file 10 \
#     gen-tasks

# Assemble final dataset (run after LLM generation is complete)
python scripts/wsd_training/build_dataset.py --filter-cws assemble
