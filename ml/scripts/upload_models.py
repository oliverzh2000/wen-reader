#!/usr/bin/env python3
"""Upload final model checkpoints to Hugging Face Hub.

Usage:
    uv run python scripts/upload_models.py
"""

from pathlib import Path
from huggingface_hub import HfApi, login, upload_folder

ML_DIR = Path(__file__).parent.parent
REPO_ID = "oliverzh2000/wen-reader"

login()

api = HfApi()
api.create_repo(repo_id=REPO_ID, repo_type="model", private=False, exist_ok=True)

# Upload model card
api.upload_file(
    path_or_fileobj=str(ML_DIR / "scripts" / "MODEL_CARD.md"),
    path_in_repo="README.md",
    repo_id=REPO_ID,
    repo_type="model",
)

for local, remote in [
    ("models/cws_span_scorer_electra_base/final", "cws-span-scorer-electra-base"),
    ("models/cws_span_scorer_electra_small/final", "cws-span-scorer-electra-small"),
    ("models/wsd_biencoder_gte_base/final", "wsd-biencoder-gte-base"),
    ("models/wsd_distilled_gte_v2_small/20260529_021557/final", "wsd-biencoder-gte-small-distilled"),
]:
    path = ML_DIR / local
    if not path.exists():
        print(f"WARNING: {path} not found, skipping")
        continue
    print(f"Uploading {path} -> {remote} ...")
    upload_folder(folder_path=str(path), path_in_repo=remote, repo_id=REPO_ID, repo_type="model")
    print(f"  Done")

print(f"\nAll done! https://huggingface.co/{REPO_ID}")
