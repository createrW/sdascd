# Experiment log

Every formal experiment (README 9.5). One row per run; newest at the bottom.

| date | run_id | group | model / change | seed | val Macro-F1 | val Acc | machine | who | notes |
|---|---|---|---|---:|---:|---:|---|---|---|
| 2026-10-01 | 20261001_clip_vitb32_seed42 | image | frozen CLIP ViT-B/32 + linear head, lr 1e-3, 30 epochs, best epoch 7 | 42 | 0.5266 | 0.6252 | Windows, RTX 5090, torch 2.11 cu128 | Yuxiang | majority baseline 0.4349; val curve noisy (0.43-0.53) |
| 2026-10-05 | mac_test | image | Mac reproduction of 20261001_clip_vitb32_seed42 (same config, features re-extracted on MPS) | 42 | 0.5266 | 0.6252 | MacBook Pro, Apple Silicon, MPS, torch 2.11 | Yuxiang | identical to Windows run: best epoch 7, same confusion matrix |
| 2026-10-16 | 20261016_distilbert_clip_seed42 | multimodal | frozen DistilBERT [CLS] + image group CLIP cache, L2, MLP 1536-512-2, lr 1e-3, batch 16, 30 epochs, best epoch 25 | 42 | 0.5336 | 0.6567 | Linux, 4 vCPU, CPU only, torch 2.11 | Claude (multimodal) | majority 0.4349; overfits (val loss 0.69 -> 1.7); 43.5 s |
| 2026-10-16 | 20261016_distilbert_clip_seed42_reproduction_cpu | multimodal | reproduction: re-extracted text cache + CPU CLIP cache, best epoch 29 | 42 | 0.5435 | 0.7067 | same | Claude (multimodal) | diff 0.0098 <= 0.01 -> REPRODUCTION OK; end-to-end check 6.4e-7 |
| 2026-10-16 | 20261016_distilbert_clip_seed42_rerun_same_inputs | multimodal | same command and inputs as the main run | 42 | 0.5336 | 0.6567 | same | Claude (multimodal) | identical predictions (deterministic on CPU) |
| 2026-10-16 | 20261016_distilbert_clip_frozen_seed43 / _seed44 | multimodal | baseline, extra seeds | 43/44 | 0.5315 / 0.5441 | 0.7039 / 0.6853 | same | Claude (multimodal) | 3-seed mean 0.5364 ± 0.0067 |
| 2026-10-16 | 20261016_distilbert_meanpool_clip_frozen_seed42/43/44 | multimodal | ablation: mean-pooled text | 42-44 | 0.5280 ± 0.0032 | 0.6891 | same | Claude (multimodal) | not better than [CLS] |
| 2026-10-16 | 20261016_clip_frozen_mlp_image_only_seed42/43/44 | multimodal | ablation: CLIP only, same MLP | 42-44 | 0.5459 ± 0.0108 | 0.7144 | same | Claude (multimodal) | image alone >= fusion |
| 2026-10-16 | 20261016_distilbert_frozen_mlp_text_only_seed42/43/44 | multimodal | ablation: DistilBERT [CLS] only, same MLP | 42-44 | 0.4862 ± 0.0079 | 0.6052 | same | Claude (multimodal) | frozen text features near majority baseline |
