#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ML_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_ROOT="$(cd "$ML_DIR/.." && pwd)"
APP_RESOURCES="$PROJECT_ROOT/WenReader/Resources"

# ── Model paths (single source of truth) ──
SPAN_MODEL="models/cws_span_scorer_electra_small/final"
WSD_MODEL="models/wsd_distilled_gte_v2_small/20260529_021557/final"

cd "$ML_DIR"
source .venv/bin/activate

case "${1:-}" in
    download)
        python scripts/download_models.py
        ;;
    gen-tasks)
        python scripts/dataset_v2/build_dataset.py 1 2
        ;;
    annotate-cws)
        python scripts/dataset_v2/build_dataset.py 3 4 5
        ;;
    assemble-cws)
        python scripts/dataset_v2/build_dataset.py 6 7 8
        ;;
    annotate-wsd)
        python scripts/dataset_v2/build_dataset.py 9 10 11
        ;;
    assemble-wsd)
        python scripts/dataset_v2/build_dataset.py 12
        ;;
    train-cws)
        python scripts/span_scorer/train_span_scorer.py base-v1
        ;;
    train-wsd)
        python scripts/wsd_training/train_wsd.py base
        ;;
    export)
        python scripts/export_coreml.py span --model-dir "$SPAN_MODEL"
        python scripts/export_coreml.py wsd --model-dir "$WSD_MODEL"
        ;;
    bundle)
        # ── 1. Extract vocab files ──
        echo "==> Extracting vocab files..."
        python scripts/extract_vocab.py "$SPAN_MODEL" "$APP_RESOURCES/cws_vocab.txt"
        python scripts/extract_vocab.py "$WSD_MODEL" "$APP_RESOURCES/wsd_vocab.txt"

        # ── 2. Copy CoreML models ──
        echo "==> Copying CoreML models..."
        rm -rf "$APP_RESOURCES/Models"
        mkdir -p "$APP_RESOURCES/Models"
        cp -R output/coreml/span_scorer_encoder.mlpackage "$APP_RESOURCES/Models/"
        cp -R output/coreml/wsd_encoder.mlpackage "$APP_RESOURCES/Models/"

        # ── 3. Copy span head weights ──
        echo "==> Copying span head weights..."
        cp output/coreml/span_head_weights.bin "$APP_RESOURCES/span_head_weights.bin"

        # ── 4. Rebuild CEDICT database with embeddings ──
        echo "==> Rebuilding CEDICT database..."
        python scripts/cedict_to_sql.py --wsd-model "$WSD_MODEL"

        echo ""
        echo "==> Bundle complete. App is ready to compile."
        ;;
    eval-coreml)
        shift
        python scripts/eval_coreml.py "$@"
        ;;
    *)
        echo "Usage: $0 <command>" >&2
        echo "" >&2
        echo "Commands:" >&2
        echo "  download       Download model weights from HF Hub" >&2
        echo "  gen-tasks      Extract sentences and build task files" >&2
        echo "  annotate-cws   Run LLM annotation for CWS (steps 3-5)" >&2
        echo "  assemble-cws   Assemble CWS training data (steps 6-8)" >&2
        echo "  annotate-wsd   Run WSD pipeline and LLM annotation (steps 9-11)" >&2
        echo "  assemble-wsd   Assemble WSD training data (step 12)" >&2
        echo "  train-cws      Train CWS span scorer model" >&2
        echo "  train-wsd      Train WSD bi-encoder model" >&2
        echo "  export         Export models to CoreML" >&2
        echo "  bundle         Copy CoreML models + vocab to app resources" >&2
        echo "  eval-coreml    Evaluate exported CoreML models on test cases" >&2
        exit 1
        ;;
esac
