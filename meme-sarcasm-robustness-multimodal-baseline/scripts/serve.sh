#!/usr/bin/env bash
# Serve the multimodal model over HTTP (FastAPI + uvicorn; contract in docs/api_spec.md).
#   bash scripts/serve.sh                                # configs/deploy/serving.yaml
#   CHECKPOINT=checkpoints/multimodal/my_run/best.pt bash scripts/serve.sh      # your own head
#   SERVING_CONFIG=my_serving.yaml PORT=9000 bash scripts/serve.sh
#   TEXT_MODEL_DIR=hf_local/distilbert-base-uncased CLIP_MODEL_DIR=hf_local/clip-vit-base-patch32 \
#       bash scripts/serve.sh                            # offline: local copies of the pinned encoders
# The default checkpoint comes from Multimodal_Assets_20261016.zip (docs/Multimodal_Assets.md).
# Then: curl -s localhost:8000/health
#       curl -s -X POST localhost:8000/predict -H 'Content-Type: application/json' \
#            -d @data/samples/predict_request.json
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-python}"
export SERVING_CONFIG="${SERVING_CONFIG:-configs/deploy/serving.yaml}"
args=(--serving_config "$SERVING_CONFIG")
if [[ -n "${CHECKPOINT:-}" ]]; then args+=(--checkpoint "$CHECKPOINT"); fi
if [[ -n "${TEXT_MODEL_DIR:-}" ]]; then args+=(--text_model_dir "$TEXT_MODEL_DIR"); fi
if [[ -n "${CLIP_MODEL_DIR:-}" ]]; then args+=(--clip_model_dir "$CLIP_MODEL_DIR"); fi
if [[ -n "${DEVICE:-}" ]]; then args+=(--device "$DEVICE"); fi
if [[ -n "${HOST:-}" ]]; then args+=(--host "$HOST"); fi
if [[ -n "${PORT:-}" ]]; then args+=(--port "$PORT"); fi
exec "$PYTHON" -m src.serving.app "${args[@]}" "$@"
