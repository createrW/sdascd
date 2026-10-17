#!/usr/bin/env bash
# Train a head on cached frozen-encoder features (src/models/trainer.py).
#   bash scripts/train.sh                                                   # multimodal baseline
#   bash scripts/train.sh configs/train/image_baseline.yaml --run_id my_image_run
#   bash scripts/train.sh configs/train/multimodal_text_only.yaml --seed 43
#   CONFIG=configs/train/multimodal_mean_pooling.yaml bash scripts/train.sh
# Needs MEMOTION_DATA_DIR (or the dataset in data/raw/) and the caches in data/processed/features/.
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-python}"
config="${CONFIG:-configs/train/multimodal_baseline.yaml}"
if [[ $# -gt 0 && "$1" != -* ]]; then config="$1"; shift; fi
exec "$PYTHON" -m src.models.trainer --config "$config" "$@"
