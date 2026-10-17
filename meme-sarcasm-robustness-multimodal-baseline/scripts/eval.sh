#!/usr/bin/env bash
# Re-score a saved head (src/evaluation/evaluate.py). Validation is compared with the run's
# reported metrics (Macro-F1 tolerance 0.01) and exits nonzero outside it.
#   bash scripts/eval.sh my_multimodal_run                        # a run you trained -> <run>/reeval/
#   bash scripts/eval.sh 20261016_distilbert_clip_seed42 --out_dir outputs/multimodal/20261016_distilbert_clip_seed42/reeval_mine
#   bash scripts/eval.sh 20261006_code_fixes_validation_v2 \
#       --features_dir data/processed/features/20261006_code_fixes_validation_v2 --out_dir outputs/image/quick_check
#   bash scripts/eval.sh 20261016_distilbert_clip_seed42_reproduction_cpu --end_to_end \
#       --report outputs/multimodal/20261016_distilbert_clip_seed42_reproduction_cpu/checks/end_to_end_mine.json \
#       --text_model_dir hf_local/distilbert-base-uncased --clip_model_dir hf_local/clip-vit-base-patch32
# The shipped reference runs already contain reeval/ and checks/; existing files are never
# overwritten, so give them a new --out_dir / --report. --end_to_end runs raw captions + raw images
# through the serving predictor (src/evaluation/end_to_end.py). Test is scored only with
# --split test, once, at the very end.
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-python}"
if [[ $# -lt 1 || "$1" == -* ]]; then
  echo "usage: bash scripts/eval.sh <run_id> [--end_to_end] [options]" >&2
  exit 2
fi
run_id="$1"; shift
if [[ "${1:-}" == "--end_to_end" ]]; then
  shift
  exec "$PYTHON" -m src.evaluation.end_to_end --run_id "$run_id" "$@"
fi
exec "$PYTHON" -m src.evaluation.evaluate --run_id "$run_id" "$@"
