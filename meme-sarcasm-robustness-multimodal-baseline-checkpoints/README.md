# Checkpoints of the merged repository / 合并后仓库的全部分类头

Unzip **at the repository root** (`meme-sarcasm-robustness-multimodal-baseline/`); files land in
`checkpoints/image/<run_id>/best.pt` and `checkpoints/multimodal/<run_id>/best.pt`, where the
evaluator, the serving config and the results report expect them. Feature caches are a separate
archive (`Feature_Caches_merged_20261018.zip`, unzip at the repository root too; it contains
`data/processed/features/...`).

在**仓库根目录**解压，文件会放到 `checkpoints/image/<run_id>/best.pt` 和 `checkpoints/multimodal/<run_id>/best.pt`，即评估脚本、服务配置和结果报告默认读取的位置。特征缓存是另一个压缩包（`Feature_Caches_merged_20261018.zip`，同样在仓库根目录解压）。

| Group / 组 | Heads | What |
|---|---:|---|
| image group (already in Git) | 3 | linear CLIP heads: original baseline, Zhihan's reproduction, final reference `20261006_code_fixes_validation_v2` |
| multimodal 2026-10-16 | 14 | fusion baseline seeds 42/43/44, CPU-feature rerun, same-input rerun, text-only / image-only / mean-pooling ablations × 3 seeds |
| overfitting diagnostic 2026-10-18 | 42 | head-size / regularization sweep, fusion and image only × 7 variants × 3 seeds (`Team_Review/multimodal/overfitting/`) |

`MANIFEST.csv` lists every file with task, modalities, seed, selected epoch, validation Macro-F1,
head, learning rate, weight decay, parameter count, the feature caches it was trained on and
SHA-256 (`SHA256SUMS` for `sha256sum -c`). Every checkpoint is a dict with `state_dict`, `config`,
`epoch`, `val_macro_f1`, `class_weights` (+ `modalities`, feature dirs for multimodal heads); load
with `torch.load(path, map_location="cpu", weights_only=True)`.

```bash
sha256sum -c SHA256SUMS
python -m src.evaluation.evaluate --run_id 20261016_distilbert_clip_seed42 --out_dir outputs/multimodal/20261016_distilbert_clip_seed42/reeval_mine
CHECKPOINT=checkpoints/multimodal/20261016_distilbert_clip_seed42/best.pt bash scripts/serve.sh
```

Selection used validation only; Test was never scored. Diagnostic-sweep heads are for analysis, not
candidates for the final model (16 configurations compared on one validation set).
