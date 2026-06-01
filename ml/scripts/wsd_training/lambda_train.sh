#!/bin/bash
# Train WSD biencoder on Lambda Labs H100
#
# Usage: bash ml/scripts/wsd_training/lambda_train.sh <command> <host_ip>
#
# Commands:
#   upload    - Upload code and data to the instance
#   train     - Install deps and run training
#   download  - Download trained models
#   all       - upload + train + download (original behavior)
#
# Examples:
#   bash ml/scripts/wsd_training/lambda_train.sh all 129.146.xx.xx
#   bash ml/scripts/wsd_training/lambda_train.sh download 129.146.xx.xx
#   bash ml/scripts/wsd_training/lambda_train.sh upload 129.146.xx.xx

set -euo pipefail

COMMAND="${1:?Usage: lambda_train.sh <upload|train|download|all> <host_ip>}"
HOST_IP="${2:?Usage: lambda_train.sh <upload|train|download|all> <host_ip>}"
LAMBDA_HOST="ubuntu@${HOST_IP}"
SSH="ssh -o StrictHostKeyChecking=no -o ServerAliveInterval=30 -o ServerAliveCountMax=10"
SCP="scp -o StrictHostKeyChecking=no -o ServerAliveInterval=30 -o ServerAliveCountMax=10"
REMOTE_DIR="~/wen-reader-train"
ML_DIR="$(cd "$(dirname "$0")/../.." && pwd)"

do_upload() {
    echo "=== Uploading project files and dataset to $LAMBDA_HOST ==="
    $SSH $LAMBDA_HOST "mkdir -p $REMOTE_DIR/ml/{scripts/wsd_training,data/dataset_v2}"

    # Project files
    $SCP "$ML_DIR/pyproject.toml" "$ML_DIR/uv.lock" "$ML_DIR/.python-version" \
         "$LAMBDA_HOST:$REMOTE_DIR/ml/"

    # Dataset
    $SCP "$ML_DIR/data/dataset_v2/wsd_dataset_v2.tsv" \
         "$LAMBDA_HOST:$REMOTE_DIR/ml/data/dataset_v2/"
}

do_train() {
    echo "=== Installing uv + deps (from lockfile) ==="
    $SSH $LAMBDA_HOST "curl -LsSf https://astral.sh/uv/install.sh | sh && \
        source \$HOME/.local/bin/env && \
        cd $REMOTE_DIR/ml && uv sync --frozen"

    echo "=== Uploading training script to $LAMBDA_HOST ==="
    # Training script
    $SCP "$ML_DIR/scripts/wsd_training/train_wsd.py" \
         "$LAMBDA_HOST:$REMOTE_DIR/ml/scripts/wsd_training/"

#     echo "=== Training GTE small ==="
#     $SSH $LAMBDA_HOST "source \$HOME/.local/bin/env && cd $REMOTE_DIR/ml && uv run python scripts/wsd_training/train_wsd.py small-h100"

#     echo "=== Training GTE base ==="
#     $SSH $LAMBDA_HOST "source \$HOME/.local/bin/env && cd $REMOTE_DIR/ml && uv run python scripts/wsd_training/train_wsd.py base-h100"

    echo "=== Training GTE large ==="
    $SSH $LAMBDA_HOST "source \$HOME/.local/bin/env && cd $REMOTE_DIR/ml && uv run python scripts/wsd_training/train_wsd.py large-h100"

    echo "=== Training BGE base ==="
    $SSH $LAMBDA_HOST "source \$HOME/.local/bin/env && cd $REMOTE_DIR/ml && uv run python scripts/wsd_training/train_wsd.py bge-base-h100"

    echo "=== Training BGE large ==="
    $SSH $LAMBDA_HOST "source \$HOME/.local/bin/env && cd $REMOTE_DIR/ml && uv run python scripts/wsd_training/train_wsd.py bge-large-h100"

    echo "=== Training BGE small ==="
    $SSH $LAMBDA_HOST "source \$HOME/.local/bin/env && cd $REMOTE_DIR/ml && uv run python scripts/wsd_training/train_wsd.py bge-small-h100"
}

do_download() {
    echo "=== Downloading models ==="
#     mkdir -p "$ML_DIR/models/wsd_finetuned_biencoder_gte_v2_small"
#     mkdir -p "$ML_DIR/models/wsd_biencoder_gte_base"
    mkdir -p "$ML_DIR/models/wsd_finetuned_biencoder_gte_v2_large"
    mkdir -p "$ML_DIR/models/wsd_finetuned_biencoder_bge_v2_base"
    mkdir -p "$ML_DIR/models/wsd_finetuned_biencoder_bge_v2_large"
    mkdir -p "$ML_DIR/models/wsd_finetuned_biencoder_bge_v2_small"

    # Download only the final (best) checkpoint
#     $SCP -r "$LAMBDA_HOST:$REMOTE_DIR/ml/models/wsd_finetuned_biencoder_gte_v2_small/*/final" \
#          "$ML_DIR/models/wsd_finetuned_biencoder_gte_v2_small/final"
#     $SCP -r "$LAMBDA_HOST:$REMOTE_DIR/ml/models/wsd_biencoder_gte_base/*/final" \
#          "$ML_DIR/models/wsd_biencoder_gte_base/final"
    $SCP -r "$LAMBDA_HOST:$REMOTE_DIR/ml/models/wsd_finetuned_biencoder_gte_v2_large/*/final" \
         "$ML_DIR/models/wsd_finetuned_biencoder_gte_v2_large/final"
    $SCP -r "$LAMBDA_HOST:$REMOTE_DIR/ml/models/wsd_finetuned_biencoder_bge_v2_base/*/final" \
         "$ML_DIR/models/wsd_finetuned_biencoder_bge_v2_base/final"
    $SCP -r "$LAMBDA_HOST:$REMOTE_DIR/ml/models/wsd_finetuned_biencoder_bge_v2_large/*/final" \
         "$ML_DIR/models/wsd_finetuned_biencoder_bge_v2_large/final"
    $SCP -r "$LAMBDA_HOST:$REMOTE_DIR/ml/models/wsd_finetuned_biencoder_bge_v2_small/*/final" \
         "$ML_DIR/models/wsd_finetuned_biencoder_bge_v2_small/final"

    echo ""
    echo "=== Done! Models downloaded to ml/models/ ==="
    echo "⚠️  Don't forget to TERMINATE your Lambda instance: ${HOST_IP}"
}

case "$COMMAND" in
    upload)   do_upload ;;
    train)    do_train ;;
    download) do_download ;;
    all)      do_upload; do_train; do_download ;;
    *)        echo "Unknown command: $COMMAND"; echo "Usage: lambda_train.sh <upload|train|download|all> <host_ip>"; exit 1 ;;
esac
