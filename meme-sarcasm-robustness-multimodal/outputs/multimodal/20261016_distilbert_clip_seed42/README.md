# 20261016_distilbert_clip_seed42

distilbert_clip_frozen: modalities text + image, frozen encoders, cached features (text `first_token`, image `pooler_output`), L2-normalized, MLP 1536 -> 512 -> 2, seed 42, device cpu.

## Reproduce

```
# Set MEMOTION_DATA_DIR and choose an unused run_id.
python -m src.multimodal.train --config "configs/multimodal_baseline.json" --text_features_dir "features/distilbert_first_token" --image_features_dir "features/20261006_code_fixes_validation_v2" --seed 42 --run_id 20261016_distilbert_clip_seed42_reproduction
python -m scripts.verify_predictions outputs/multimodal/20261016_distilbert_clip_seed42_reproduction/predictions_validation.csv
python -m src.multimodal.evaluate --run_id 20261016_distilbert_clip_seed42   # reload the saved head
```

## Validation (best epoch by Macro-F1)

| model | Macro-F1 | Accuracy | F1 class 0 | F1 class 1 | best epoch | train time |
|---|---:|---:|---:|---:|---:|---:|
| majority (always 1) | 0.4349 | 0.7697 | 0.0000 | 0.8698 | - | - |
| distilbert_clip_frozen | 0.5336 | 0.6567 | 0.2941 | 0.7732 | 25 | 43.5s |

Confusion matrix (rows true 0/1, cols predicted 0/1): [[50, 111], [129, 409]]

Checkpoint (not in git): `checkpoints/multimodal/20261016_distilbert_clip_seed42/best.pt`
