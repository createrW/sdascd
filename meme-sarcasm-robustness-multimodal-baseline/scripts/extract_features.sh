#!/usr/bin/env bash
# Run the frozen encoders once and cache features in data/processed/features/
# (src/features/feature_engineering.py; CPU is fine, see docs/multimodal_baseline.md).
#   bash scripts/extract_features.sh text                 # DistilBERT [CLS] + mean pooling, all splits
#   bash scripts/extract_features.sh text --model_dir hf_local/distilbert-base-uncased
#   bash scripts/extract_features.sh image                # CLIP ViT-B/32 pooler_output, train + validation
#   bash scripts/extract_features.sh image --limit 128    # separate smoke cache
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-python}"
if [[ $# -lt 1 ]]; then
  echo "usage: bash scripts/extract_features.sh image|text [options]" >&2
  exit 2
fi
exec "$PYTHON" -m src.features.feature_engineering "$@"
