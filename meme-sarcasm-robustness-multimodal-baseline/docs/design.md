# Technical design / 技术方案设计

Binary sarcasm detection on Memotion memes (image + caption) with frozen pretrained encoders and
small trained heads. This page describes the system as built: data contract, models, training,
evaluation, serving and the repository layout. Results are in
[Team_Review/multimodal/RESULTS_REPORT.md](../Team_Review/multimodal/RESULTS_REPORT.md); the
decision log is [decisions.md](decisions.md); run history is [experiment_log.md](experiment_log.md).

基于冻结预训练编码器和小型可训练分类头的梗图讽刺二分类。本页说明数据约定、模型、训练、评估、服务和仓库结构；结果见结果报告，决策见 decisions.md，实验记录见 experiment_log.md。

## 1. Goals and non-goals / 目标与边界

| | |
|---|---|
| Goal | Reproducible baselines for the team: image-only (CLIP), multimodal (DistilBERT + CLIP fusion), with unimodal ablations under the same head design. |
| Goal | One code path from cached features (training) to raw inputs (serving), checked numerically. |
| Goal | Everything runs on CPU; GPU is optional (feature extraction is the only expensive step). |
| Non-goal | Fine-tuning the encoders, tuning the threshold, or choosing anything on Test (team README 3.2). |
| Non-goal | The Text Group's fine-tuned DistilBERT classifier (separate branch). |

## 2. Data contract / 数据约定

| Item | Value |
|---|---|
| Source | `MMSoc_Memotion_exported/{train,validation,test}/metadata.csv` + `images/`; root from `MEMOTION_DATA_DIR`, else `data/raw/MMSoc_Memotion_exported/` |
| Sizes | train 5,593 · validation 699 · test 700 (checked by `scripts/verify_dataset.py`) |
| Label | `sarcasm` (4 classes) → `sarcasm_binary`: `not_sarcastic` = 0, `general` / `twisted_meaning` / `very_twisted` = 1; train 1,226 / 4,367 |
| Caption | `text_input` (= `text_corrected`, OCR fallback); missing or blank → `[NO_TEXT]` (`src/data/preprocess.py`) |
| Image | any mode → RGB (palette images via RGBA) → official CLIP processor (resize 224, center crop, normalize) |
| Use of splits | train: fit; validation: epoch selection, all comparisons; test: once, at the end |
| Primary metric | Macro-F1 at threshold 0.5 (`src/evaluation/metrics.py`); majority baseline 0.4349 |

## 3. Models / 模型

```text
                    frozen DistilBERT (distilbert-base-uncased @ 1c4513b)
caption ──WordPiece──► last_hidden_state[:, 0]  ([CLS], 768) ──L2──┐
                                                                   ├─ concat [text | image] (1536)
image ──CLIP proc.──► frozen CLIP ViT-B/32 (@ 3d74acf)             │        │
                      vision pooler_output (768) ──────────L2──────┘        ▼
                                                      Linear 1536→512 · ReLU · Dropout 0.2 · Linear 512→2
image-only baseline:  pooler_output (768) ─────────────► Linear 768→2 (linear probe)
```

| Model (config) | Trainable head | Inputs | Class in `src/models/` |
|---|---|---|---|
| Image baseline (`image_baseline.yaml`) | linear 768→2 (or 1-hidden-layer MLP if `head.hidden_dim > 0`) | CLIP `pooler_output` | `vit_classifier.ImageHead` |
| Multimodal baseline (`multimodal_baseline.yaml`) | MLP 1536→512→2, 787,970 parameters | L2(text) ‖ L2(image) | `multimodal_classifier.FusionHead` |
| Ablations (`multimodal_{text_only,image_only,mean_pooling}.yaml`) | same MLP on one modality, or mean-pooled text | | `FusionHead(modalities=...)` |
| End-to-end / serving | the trained `FusionHead.classifier` weights | raw caption + raw image | `multimodal_classifier.MultimodalClassifier` |

Every head derives from `base_model.FeatureHead` and takes cached tensors by modality name
(`head(text=..., image=...)`), so one trainer and one evaluator serve all of them. The base
class owns no parameters: the image group's shipped checkpoints load unchanged.

Why frozen `[CLS]`: DistilBERT has no pooler; the final-layer `[CLS]` state is what
`DistilBertForSequenceClassification` classifies from (there after fine-tuning). Masked mean
pooling, the usual frozen sentence vector, is kept as an ablation (D13–D15).

## 4. Pipeline / 流程

| Stage | Command | Module | Output |
|---|---|---|---|
| 0. data check | `python -m scripts.verify_dataset` | `src/data/dataset.py` | `DATASET OK` |
| 1. features (once) | `bash scripts/extract_features.sh text\|image` | `src/features/feature_engineering.py` | `data/processed/features/<name>/{split}.pt` + `meta.json` |
| 2. train | `bash scripts/train.sh <config>` | `src/models/trainer.py` | `outputs/<task>/<run_id>/`, `checkpoints/<task>/<run_id>/best.pt` |
| 3. evaluate / replay | `bash scripts/eval.sh <run_id>` (image heads: `--features_dir` if the run's config.json does not name the cache) | `src/evaluation/evaluate.py` | `reeval/` (or `--out_dir`) + pass/fail vs reported metrics |
| 4. end-to-end check | `bash scripts/eval.sh <run_id> --end_to_end` | `src/evaluation/end_to_end.py` | raw-input path == cached path |
| 5. analysis / report | `python -m src.evaluation.results_report` | `src/evaluation/{results_report,error_analysis,review_samples}.py` | `Team_Review/multimodal/report/` |
| 6. serve | `bash scripts/serve.sh` | `src/serving/{predictor,app}.py` | HTTP API ([api_spec.md](api_spec.md)) |

**Feature caches.** One file per split: `{"sample_id": list[str], "features": float32[N, 768],
"labels": int64[N]}`, rows in `metadata.csv` order, plus `meta.json` with the encoder id and
revision (requested and resolved), weight file and hash for local copies (D20), settings, versions, device,
timing and the SHA-256 of every split file. `src/features/cache.py` rejects a cache unless the
provenance matches the config, it is complete (no `--limit` smoke cache), IDs, order and labels
match the metadata, the dimension and dtype are right, values are finite and the hash matches.
`src/data/loader.py` then joins modalities on `sample_id`, never on row number.

## 5. Training / 训练

| Setting | Value | Source |
|---|---|---|
| Loss | cross-entropy with class weights from **train only**, w_c = N / (2 n_c) → w0 2.281, w1 0.640 | README 5.2 (D9) |
| Optimizer | AdamW, lr 1e-3, weight decay 1e-4 | README 5.1 |
| Batch / epochs | 64 / 30 (image), 16 / 30 (multimodal) | configs |
| Selection | best epoch by validation Macro-F1; that state is the checkpoint | README 5.1 (D10) |
| Determinism | `set_seed(42)` before head init; shuffling from `torch.Generator().manual_seed(seed)`; CPU runs repeat bit-identically | experiment_log |
| Threshold | 0.5, fixed | README 3.2 (D10) |

The training loop (`src/models/trainer.py::fit`) is shared by both tasks; the config's `task:`
key (inferred for configs and checkpoints recorded before it existed) chooses the head and which
caches to load. Outputs follow README 6: `config.json` (resolved config, versions, feature
provenance, command), `metrics_validation.json`, `predictions_validation.csv`,
`confusion_matrix_validation.png`, `training_log.csv`, `README.md`. Existing run or checkpoint
folders are never overwritten (`src/utils/safety.py`).

## 6. Evaluation and reproduction / 评估与复现

* **Replay.** `evaluate.py` reloads a head and its caches, recomputes metrics and compares
  Macro-F1 with the reported value (absolute tolerance 0.01, D11); a mismatch exits with code 1.
  Reload on the same machine must give identical predictions (`scripts/verify_reproduction.py
  --tolerance 0 --reference_predictions ...`).
* **End to end.** `end_to_end.py` sends raw captions and images through the serving predictor
  and compares features with the caches and probabilities with the run's predictions. On CPU
  caches the gap is 6.4e-7. The image group's cache was extracted on CUDA with TF32 convolution
  (mean cosine 0.99997 to float32), which the end-to-end check reports rather than hides (D19).
* **Statistics.** `results_report.py` adds bootstrap CIs, paired bootstrap and McNemar tests,
  label-permutation tests against a label-independent chance level (≈ 0.50), ROC/PR-AUC,
  calibration (ECE, Brier), linear probes and a modality-reliance diagnostic, all on validation.
* **Refactor equivalence (2026-10-17).** After the move to this layout, retraining the image
  baseline and the four seed-42 multimodal configs gave byte-identical predictions and training
  logs to the recorded runs, and every shipped head replayed identically ([experiment_log.md](experiment_log.md)).

## 7. Serving / 服务

`MultimodalPredictor` (`src/serving/predictor.py`) loads the frozen encoders, tokenizer and CLIP
processor once and the trained `FusionHead` weights from a checkpoint; preprocessing is the
training-time code in `src/data/preprocess.py`. `src/serving/app.py` wraps it in FastAPI
(`create_app` factory; `/health`, `/model`, `/predict`, `/predict/batch`). It checks every upload
before decoding (body and base64 size, image format, pixel count, aspect ratio), then decodes and
preprocesses one image at a time behind a lock, releasing each decoded raster before the next, so
the decoded pixels of only one image (plus its RGB copy) are in memory at a time; the compressed
uploads of a batch are held until the response. Settings live in
`configs/deploy/serving.yaml` (checkpoint, local encoder copies, device, limits, host/port). The
`Dockerfile` builds a CPU image with code and configs only; the checkpoint and (offline) encoder
weights are mounted at run time, so no dataset-derived file is baked into the image.

The served model is the validation-selected multimodal baseline. Its validation Macro-F1 is 0.53,
only a little above chance; the API is an integration target for the team, not a production
classifier.

## 8. Configuration / 配置

`configs/train/*.yaml` (read by `src/utils/config.py`, JSON also accepted):

| Key | Meaning |
|---|---|
| `task` | `image` or `multimodal` (inferred from `fusion` when absent) |
| `model_name`, `seed`, `threshold` | recorded in every metrics/predictions file |
| `clip_model_id`, `clip_revision`, `feature` / `image_feature` | image encoder and the vector taken from it |
| `text_model_id`, `text_revision`, `text_feature`, `text_max_length` | text encoder, `first_token` or `mean_pooling`, truncation |
| `modalities`, `normalize_embeddings`, `fusion.{input_dim,hidden_dim,dropout,num_classes}` | multimodal head |
| `head.{hidden_dim,dropout}` | image head (`hidden_dim: 0` = linear probe) |
| `features.{text,image}` | default cache folders (command-line flags override) |
| `loss`, `optimizer`, `lr`, `weight_decay`, `batch_size`, `epochs` | training |

## 9. Repository layout / 目录

| Folder | Role | Former location |
|---|---|---|
| `src/data/` | dataset contract, preprocessing, joint cached-feature loader | `src/common/{data,labels}.py`, `src/multimodal/features.py` |
| `src/features/` | encoder runs and cache validation | `src/image/{extract_features,cache}.py`, `src/text/*` |
| `src/models/` | heads, end-to-end model, trainer | `src/image/{model,train}.py`, `src/multimodal/{model,train}.py` |
| `src/evaluation/` | metrics, report writers, replay, reproduction checks, analysis | `src/common/{metrics,reproduction}.py`, `src/image/{evaluate,error_analysis}.py`, `src/multimodal/{evaluate,check_end_to_end,review_samples}.py`, `scripts/make_multimodal_report.py` |
| `src/serving/` | predictor and HTTP service | `src/multimodal/smoke_test.py` (now the predictor's `--smoke`) |
| `src/utils/` | config loading, seeding/devices, path guards | `src/common/reproducibility.py`, `src/image/safety.py` |
| `configs/train/`, `configs/deploy/` | YAML settings | `configs/*.json`, `configs/ablations/*.json` |
| `data/{raw,processed,samples}` | dataset, caches (both git-ignored), synthetic samples | `$MEMOTION_DATA_DIR`, `features/` |

## 10. Known limitations / 已知限制

* The frozen text features carry little signal (text-only 0.486 Macro-F1 vs majority 0.435);
  fusion does not beat image-only on validation (RESULTS_REPORT §5–6).
* The heads fit the training set almost perfectly (train ROC-AUC ≈ 1 vs validation ≈ 0.54). A linear
  head or a 10× lower learning rate shrinks that gap; neither improves validation scores or fusion vs
  image only ([overfitting diagnostic](../Team_Review/multimodal/overfitting/README.md)).
* The image group's reference cache was extracted with TF32; CPU caches differ slightly (D19).
* Test has not been scored. The image group's cache has no test split; use the CPU cache.
* Text and multimodal caches/checkpoints are not in Git (size); see
  [Multimodal_Assets.md](Multimodal_Assets.md).
* The Docker image was written but not built in the CPU session that produced this layout
  (no Docker daemon there).
