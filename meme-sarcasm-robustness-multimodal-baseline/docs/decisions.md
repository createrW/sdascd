# Decisions

Important team decisions (README 9.5). Newest at the bottom. Add the date, who decided, and why.
Each group appends its own entries; shared ones affect everyone.

## 2026-10-01 · image group (Yuxiang)

| # | decision | why | affects |
|---|---|---|---|
| D1 | Environment: Python 3.12, torch 2.11.0 (cu128), transformers 5.18.0; one shared `requirements-lock.txt` with macOS markers for torch/torchvision | one tested environment for all groups; Python 3.8 cannot install torch >= 2.7 needed for RTX 50xx GPUs | all |
| D2 | `requirements-lock.txt` saved as UTF-8 with `--extra-index-url .../whl/cu128` on line 1 | PowerShell `>` writes UTF-16 (shows as binary on GitHub); `+cu128` wheels are not on PyPI | all |
| D3 | CLIP revision `3d74acf` (short hash) is used as given | it resolves and loads with transformers 5.18 | image, multimodal |
| D4 | `ImageFile.LOAD_TRUNCATED_IMAGES = True` in the shared loader | `train_004578.png` is truncated but readable (800x800); README forbids dropping samples | all |
| D5 | Every image goes through `load_image()`: palette (P) -> RGBA -> RGB, everything else -> RGB | dataset mixes RGB, P, RGBA, L, CMYK and GIF; CLIP needs RGB | image, multimodal |
| D6 | Empty or blank `text_input` is replaced with `[NO_TEXT]` in the shared loader | README 3.4; 1 train row and 3 test rows are empty | text, multimodal |
| D7 | Image feature = CLIP `pooler_output` (768-d, after post-LayerNorm), not the 512-d projected `image_embeds` | pre-projection feature for a linear probe; **to confirm with the multimodal group** | image, multimodal |
| D8 | Frozen-encoder features are computed once and cached in `data/processed/features/` (formerly `features/`; git-ignored) | the encoder never changes, so recomputing it every epoch is wasted time | image (multimodal may reuse) |
| D9 | Class weights `w_c = N / (2 n_c)` from train only: w0 = 2.281, w1 = 0.640 | README 5.2 | all |
| D10 | Model selection: best validation Macro-F1 over all epochs, threshold fixed at 0.5 | README 5.1 | all |
| D11 | Reproduction passes if validation Macro-F1 differs by <= 0.01 | CPU / CUDA / MPS give ~1e-4 feature differences, so exact equality is not expected | all |
| D12 | Shared metric code in `src/evaluation/metrics.py` (formerly `src/common/metrics.py`); format check in `scripts/verify_predictions.py` | one implementation for the three groups (README 9.5) | all |
| D13 | Multimodal baseline uses normalized DistilBERT first-token feature (768-d) and the image group's normalized CLIP `pooler_output` (768-d), concatenated to 1536-d before a `1536 -> 512 -> 2` MLP | matches the fixed README model revisions, preserves both modality representations, and confirms the image/multimodal interface | image, multimodal |

## 2026-10-16 · multimodal group (CPU run)

| # | decision | why | affects |
|---|---|---|---|
| D14 | Text feature = final-layer `[CLS]` state `last_hidden_state[:, 0]` of frozen `DistilBertModel` (768-d), cached raw; L2 normalization happens inside the fusion head | DistilBERT has no pooler (Sanh et al., 2019); HF `DistilBertForSequenceClassification` classifies from this vector; confirms D13 | text, multimodal |
| D15 | Masked mean pooling is cached alongside as an ablation only | standard frozen-encoder sentence vector; 3-seed validation Macro-F1 0.5280 vs 0.5364 for [CLS], so the baseline is unchanged | multimodal |
| D16 | Tokenization: the model's uncased WordPiece, `[CLS] ... [SEP]`, truncation at 128, padding to the batch maximum with `attention_mask`; `[NO_TEXT]` is encoded as ordinary text | README 4.1 / 3.4; 4 train + 1 test captions exceed 128 tokens | text, multimodal |
| D17 | The fusion MLP trains on cached features (`FusionHead`); its `classifier.*` weights load unchanged into `MultimodalClassifier` | encoders are frozen; end-to-end check matches to 6.4e-7 | multimodal |
| D18 | Multimodal baseline images = the image group's reference cache (train/validation); test and reproduction use a CPU re-extraction with the same weights (`data/processed/features/clip_vitb32_pooler_cpu`) | reuse the agreed features; the cache has no test split | image, multimodal |
| D19 | The image group's CUDA cache differs from float32 by TF32 in the patch-embedding convolution (mean cosine 0.99997, min 0.981); downstream Macro-F1 differences stay within D11. Suggest `cudnn.allow_tf32 = False` for future CUDA extraction | explained exactly by TF32 emulation; see docs/Multimodal_Test_Results.md | image, multimodal |
| D20 | Machines without Hugging Face access may pass `--model_dir` / `--text_model_dir` / `--clip_model_dir` with a local copy of the pinned weights; the source and weight hash are recorded in each cache `meta.json` (CLIP caches record the hash from 2026-10-17; the earlier `clip_vitb32_pooler_cpu` records the source only, its weights are identified in docs/multimodal_baseline.md) | this CPU environment blocked huggingface.co; default behaviour unchanged | all |
| D21 | Ablations (text only, image only, mean pooling) and seeds 43/44 are supplementary and selected on validation only; Test not scored | README 3.2 / 5.1 | multimodal |

## 2026-10-17 · repository layout (merge of the image and multimodal branches)

| # | decision | why | affects |
|---|---|---|---|
| D22 | Code follows the team layout: `configs/{train,deploy}/*.yaml`, `data/{raw,processed,samples}`, `src/{data,features,models,evaluation,serving,utils}`, `tests/`, `scripts/*.sh`; one trainer (`src/models/trainer.py`) and one evaluator (`src/evaluation/evaluate.py`) for image and multimodal heads, selected by the config's `task:` | team code convention; removes the duplicated image/multimodal train and evaluate scripts. Retraining and replays are byte-identical to the old code (experiment_log, 2026-10-17) | all |
| D23 | Feature caches live in `data/processed/features/`; recorded `features/<name>` paths resolve to it and back (`resolve_features_dir`), so shipped heads and old commands keep working | no re-recording of checkpoints or run configs | all |
| D24 | Serving (`src/serving/`, `configs/deploy/serving.yaml`) uses the validation-selected multimodal head, threshold 0.5 from training, and the training preprocessing; no Test-based choice | README 3.2. The end-to-end check confirms the serving path equals the cached path: within 6.4e-7 for heads trained on CPU caches. The default head was trained on the image group's TF32 cache, so its served probabilities differ from its recorded ones by up to 0.013 (D19) | multimodal |
