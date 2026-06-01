#!/bin/bash
set -e
cd "$(dirname "$0")/../.."

# Phase 1a: Harvest wiki/subs corpora
# python scripts/span_scorer/build_dataset.py gen-tasks corpora --wiki-target 40000 --subs-target 20000 --coverage-weight 0.6

# Phase 1b: Harvest ICWB2 MSR hard cases
# python scripts/span_scorer/build_dataset.py gen-tasks icwb2

# Phase 2: Assemble final dataset (run after LLM generation is complete)
# python scripts/span_scorer/build_dataset.py assemble
