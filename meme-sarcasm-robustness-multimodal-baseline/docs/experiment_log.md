# Experiment log

Every formal experiment (README 9.5). One row per run; newest at the bottom. The A/B summary
below compares models on validation; statistics (CIs, paired tests) are in the
[results report](../Team_Review/multimodal/RESULTS_REPORT.md).

实验记录（README 9.5）：每次正式实验一行，最新在最下方。下方 A/B 汇总在 Validation 上比较各模型，统计检验见结果报告。

## A/B summary (validation Macro-F1) / A/B 汇总

| Comparison / 对比 | A | B | Δ (B − A) | Verdict / 结论 |
|---|---:|---:|---:|---|
| majority → image linear probe (image group) | 0.4349 | 0.5276 | +0.093 | image features carry signal |
| image linear probe → multimodal fusion (seed 42) | 0.5276 | 0.5336 | +0.006 | within run-to-run noise |
| fusion → image only, same MLP (3 seeds) | 0.5364 ± 0.0067 | 0.5459 ± 0.0108 | +0.010 | text does not help the fused head |
| fusion → text only, same MLP (3 seeds) | 0.5364 ± 0.0067 | 0.4862 ± 0.0079 | −0.050 | frozen text alone is weak (majority 0.435, chance ≈ 0.50) |
| text `[CLS]` → mean pooling, fused (3 seeds) | 0.5364 ± 0.0067 | 0.5280 ± 0.0032 | −0.008 | keep `[CLS]` (D15) |
| CUDA (TF32) image cache → CPU image cache, fused, seed 42 | 0.5336 | 0.5435 | +0.010 | within D11 tolerance |
| old code layout → new layout (retrain, seed 42, 5 configs) | identical | identical | 0 | byte-identical predictions and logs (D22) |

## Runs / 实验

| date | run_id | group | model / change | seed | val Macro-F1 | val Acc | machine | who | notes |
|---|---|---|---|---:|---:|---:|---|---|---|
| 2026-10-01 | 20261001_clip_vitb32_seed42 | image | frozen CLIP ViT-B/32 + linear head, lr 1e-3, 30 epochs, best epoch 7 | 42 | 0.5266 | 0.6252 | Windows, RTX 5090, torch 2.11 cu128 | Yuxiang | majority baseline 0.4349; val curve noisy (0.43-0.53) |
| 2026-10 | 20261001_clip_vitb32_seed42_zhihan_rtx3080 | image | independent reproduction of the baseline (head kept in `checkpoints/image/`) | 42 | 0.5276 | n/a (only the head is kept) | RTX 3080 | Zhihan (image group) | best epoch 7; within D11 of the reference |
| 2026-10-05 | mac_test | image | Mac reproduction of 20261001_clip_vitb32_seed42 (same config, features re-extracted on MPS) | 42 | 0.5266 | 0.6252 | MacBook Pro, Apple Silicon, MPS, torch 2.11 | Yuxiang | identical to Windows run: best epoch 7, same confusion matrix |
| 2026-10-06 | 20261006_code_fixes_validation_v2 | image | **final image reference**: same model after the code fixes, features re-extracted (`Image_Features_CLIP_ViTB32_20261006.zip`) | 42 | 0.5276 | 0.6266 | Windows, CUDA, torch 2.11.0+cu128 | image group | best epoch 7; confusion matrix [[59, 102], [159, 379]]; used in the A/B summary |
| 2026-10-16 | 20261016_distilbert_clip_seed42 | multimodal | frozen DistilBERT [CLS] + image group CLIP cache, L2, MLP 1536-512-2, lr 1e-3, batch 16, 30 epochs, best epoch 25 | 42 | 0.5336 | 0.6567 | Linux, 4 vCPU, CPU only, torch 2.11 | Claude (multimodal) | majority 0.4349; overfits (val loss 0.69 -> 1.7); 43.5 s |
| 2026-10-16 | 20261016_distilbert_clip_seed42_reproduction_cpu | multimodal | reproduction: re-extracted text cache + CPU CLIP cache, best epoch 29 | 42 | 0.5435 | 0.7067 | same | Claude (multimodal) | diff 0.0098 <= 0.01 -> REPRODUCTION OK; end-to-end check 6.4e-7 |
| 2026-10-16 | 20261016_distilbert_clip_seed42_rerun_same_inputs | multimodal | same command and inputs as the main run | 42 | 0.5336 | 0.6567 | same | Claude (multimodal) | identical predictions (deterministic on CPU) |
| 2026-10-16 | 20261016_distilbert_clip_frozen_seed43 / _seed44 | multimodal | baseline, extra seeds | 43/44 | 0.5315 / 0.5441 | 0.7039 / 0.6853 | same | Claude (multimodal) | 3-seed mean 0.5364 ± 0.0067 |
| 2026-10-16 | 20261016_distilbert_meanpool_clip_frozen_seed42/43/44 | multimodal | ablation: mean-pooled text | 42-44 | 0.5280 ± 0.0032 | 0.6891 | same | Claude (multimodal) | not better than [CLS] |
| 2026-10-16 | 20261016_clip_frozen_mlp_image_only_seed42/43/44 | multimodal | ablation: CLIP only, same MLP | 42-44 | 0.5459 ± 0.0108 | 0.7144 | same | Claude (multimodal) | image alone >= fusion |
| 2026-10-16 | 20261016_distilbert_frozen_mlp_text_only_seed42/43/44 | multimodal | ablation: DistilBERT [CLS] only, same MLP | 42-44 | 0.4862 ± 0.0079 | 0.6052 | same | Claude (multimodal) | frozen text features near majority baseline |
| 2026-10-17 | equivalence check (runs not kept) | all | code moved to the configs/ data/ src/{data,features,models,evaluation,serving} layout; retrained image baseline (image group cache) and multimodal baseline, text-only, image-only, mean-pooling (seed 42) with the new code; replayed the 3 image heads and 14 multimodal heads; re-extracted 8-row smoke caches; reran end-to-end checks and the results report | 42 | 0.5276 / 0.5336 / 0.4903 / 0.5568 / 0.5311 | same as recorded | Linux, 4 vCPU, CPU only, torch 2.11 | Claude (multimodal) | predictions, training logs, checkpoints (image) and caches byte-identical to the old code / recorded runs |
