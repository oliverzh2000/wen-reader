#!/usr/bin/env python3
"""Download model weights from Hugging Face Hub.

Usage:
    uv run python scripts/download_models.py
"""

from pathlib import Path
from huggingface_hub import snapshot_download

REPO_ID = "oliverzh2000/wen-reader"
ML_DIR = Path(__file__).parent.parent

# (remote folder on HF, local destination relative to ML_DIR)
MODELS = [
    ("cws-span-scorer-electra-base", "models/cws_span_scorer_electra_base/final"),
    ("cws-span-scorer-electra-small", "models/cws_span_scorer_electra_small/final"),
    ("wsd-biencoder-gte-base", "models/wsd_biencoder_gte_base/final"),
    ("wsd-biencoder-gte-small-distilled", "models/wsd_distilled_gte_v2_small/20260529_021557/final"),
]


def main():
    # Download entire repo to a cache dir, then symlink/copy to expected paths
    print(f"Downloading models from {REPO_ID} ...")
    local = snapshot_download(repo_id=REPO_ID, repo_type="model")
    local = Path(local)
    print(f"  Cached at: {local}")

    for remote_folder, local_path in MODELS:
        src = local / remote_folder
        dest = ML_DIR / local_path
        if not src.exists():
            print(f"  WARNING: {src} not found in download, skipping")
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        # Create symlink to HF cache (avoids duplicating files on disk)
        if dest.exists() or dest.is_symlink():
            print(f"  {dest.name} already exists, skipping")
            continue
        dest.symlink_to(src)
        print(f"  {remote_folder} -> {dest}")

    print("\nDone. Models ready for export.")


if __name__ == "__main__":
    main()
