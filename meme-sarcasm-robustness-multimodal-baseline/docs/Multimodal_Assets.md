# External assets / 外部资料

Checkpoints, feature caches and run folders are not in Git (team README 9.4). They are four zip
archives, kept outside the repository by its owner, on the team Drive (`Meme_Project/Model_ckpt/`)
or as release assets of a **private** repository. This page gives no download link: the owner adds
the link here after uploading. Any copy is valid if its size and SHA-256 match the table below.

检查点、特征缓存和实验目录不进 Git（团队 README 9.4），以四个压缩包形式由仓库负责人保存在仓库之外：团队 Drive
（`Meme_Project/Model_ckpt/`）或**私有**仓库的 Release。本页暂无下载链接，上传后由负责人补上；任何副本只要大小与
SHA-256 与下表一致即可使用。

The archives contain no encoder weights and no dataset image files. `Run_Outputs_20261018_v2.zip` has
per-sample predictions (`sample_id`, dataset label, predicted probability) in every run folder; the
dataset-derived manual review of the main run (`review/`: meme grid and caption excerpts) is only in Git
(see README "Rules").

压缩包不含编码器权重和数据集图片文件。`Run_Outputs_20261018_v2.zip` 的每个实验目录含逐样本预测（`sample_id`、数据集标签、预测概率）；主实验的人工核对目录 `review/`（梗图图集与文字摘录）由数据集生成，只在 Git 中（见 README "Rules"）。

**Unzip every archive at the repository root with `unzip -n`** (never overwrite). The archives
already contain their destination folders and write only into `checkpoints/`,
`data/processed/features/`, `outputs/` and `logs/`. These folders are git-ignored; files that are
already in Git are byte-identical to the archive copies and are skipped.

**所有压缩包都在仓库根目录用 `unzip -n` 解压**（不覆盖已有文件）。压缩包已包含目标目录，只写入
`checkpoints/`、`data/processed/features/`、`outputs/`、`logs/`；这些目录被 Git 忽略，已在 Git 中的文件与压缩包中的逐字节相同，会被跳过。

| Archive / 压缩包 | Bytes / 字节 | Creates / 生成 | Needed for / 用途 |
|---|---:|---|---|
| `Checkpoints_merged_20261018_v2.zip` | 32357210 | `checkpoints/image/` (3 image-group heads), `checkpoints/multimodal/20261016_*` (14 multimodal and ablation heads), and `checkpoints/{README,MANIFEST,SHA256SUMS}_merged.*` | evaluation, serving, results report |
| `Checkpoints_overfitting_diagnostic_20261018_v2.zip` | 51308414 | `checkpoints/multimodal/20261018_overfit_*` (42 sweep heads), and `checkpoints/{README,MANIFEST,SHA256SUMS}_overfitting_diagnostic.*` | optional: re-evaluating the diagnostic sweep |
| `Feature_Caches_merged_20261018_v2.zip` | 97754750 | `data/processed/features/`: `distilbert_first_token/`, `distilbert_mean_pooling/`, `reproduction_distilbert_first_token/` (train/validation/test), `20261006_code_fixes_validation_v2/` (image group, CUDA, train/validation), `clip_vitb32_pooler_cpu/` (CPU, train/validation/test), each with `meta.json` | training and evaluating without downloading the encoders |
| `Run_Outputs_20261018_v2.zip` | 1557165 | `outputs/multimodal/` (56 run folders: 14 of 2026-10-16 + 42 of the diagnostic sweep), `outputs/image/20261006_code_fixes_validation_v2_cpu_features_check/`, `logs/` (27 steps: command, start and end time, exit code, output) | reference metrics for `evaluate.py`, results report |

`reproduction_distilbert_first_token/` is the re-extracted text cache that the
`20261016_distilbert_clip_seed42_reproduction_cpu` head records; its tensors are byte-identical to
`distilbert_first_token/` (only the timings in `meta.json` differ). In the step logs the
machine-specific prefixes are shortened to `python`, `hf_local/` and `<repo>`.

`reproduction_distilbert_first_token/` 是 `_reproduction_cpu` 分类头记录的重新提取的文字缓存，张量与
`distilbert_first_token/` 逐字节相同（仅 `meta.json` 中的耗时不同）。逐步日志中与机器相关的路径前缀已缩写为 `python`、`hf_local/` 和 `<repo>`。

What each checkpoint is, which one to use and how to load it: [checkpoints.md](checkpoints.md).

每个 checkpoint 的含义、推荐使用哪个以及读取方法见 [checkpoints.md](checkpoints.md)。

## SHA-256

```text
99687a70b7e2431378ab2b98fd86f7924b80b38520e168e1d379354cae61ce8b  Checkpoints_merged_20261018_v2.zip
c0330822ffe496e293084180fe7e0118e30098ad090f28ba355352153afcdc2a  Checkpoints_overfitting_diagnostic_20261018_v2.zip
6b69e05d29352e52bd309d111978612bd22c1b9f1e57020bd5c6a9321c7510dd  Feature_Caches_merged_20261018_v2.zip
8a2e0a7e557f168a5a3b85eb93ec356d47381320f35225602d081d77c32b0ecb  Run_Outputs_20261018_v2.zip
```

## Fast route / 快速路线

No encoder download is needed for training or evaluating heads on the caches.

在缓存上训练或评估分类头无需下载编码器。

```bash
unzip -n Checkpoints_merged_20261018_v2.zip
unzip -n Feature_Caches_merged_20261018_v2.zip
unzip -n Run_Outputs_20261018_v2.zip
sha256sum -c checkpoints/SHA256SUMS_merged.txt
export MEMOTION_DATA_DIR=/path/to/MMSoc_Memotion_exported   # the caches are checked against metadata.csv
python -m src.evaluation.evaluate --run_id 20261016_distilbert_clip_seed42 \
    --out_dir outputs/multimodal/20261016_distilbert_clip_seed42/quick_check
python -m src.models.trainer --config configs/train/multimodal_baseline.yaml --run_id 20261016_distilbert_clip_seed42_my_reproduction
```

Write re-evaluations inside the run folder (as above), not directly under `outputs/multimodal/`:
`scripts/summarize_multimodal_runs.py` counts any folder there with a `metrics_validation.json` as a run (unless its name contains reproduction or rerun).

重新评估请写到实验目录内部（如上），不要直接放在 `outputs/multimodal/` 下：`scripts/summarize_multimodal_runs.py` 会把那里含 `metrics_validation.json` 的目录（名称含 reproduction 或 rerun 的除外）都当作一次实验。

The cache loaders check every file's SHA-256, sample IDs, order and labels against the dataset
metadata before use.

缓存读取程序在使用前会核对每个文件的 SHA-256、sample ID、顺序和标签是否与数据集元数据一致。

## Regenerating / 重新生成

Every multimodal file can be regenerated on a 4-vCPU CPU with the commands in
[multimodal_baseline.md](multimodal_baseline.md) and the
[overfitting diagnostic](../Team_Review/multimodal/overfitting/README.md); training on CPU is
deterministic. The image group's cache `20261006_code_fixes_validation_v2` and its three heads come
from the image group's CUDA run ([Image_Reproduction.md](Image_Reproduction.md)); a CPU re-extraction
differs slightly because of TF32 (D19), so that cache is shipped as is.

所有多模态文件都可以在 4 vCPU 的 CPU 上用 [multimodal_baseline.md](multimodal_baseline.md) 与
[过拟合诊断](../Team_Review/multimodal/overfitting/README.md) 的命令重新生成（CPU 训练可逐位复现）。图片组缓存
`20261006_code_fixes_validation_v2` 及其三个分类头来自图片组的 CUDA 运行（见 [Image_Reproduction.md](Image_Reproduction.md)）；
在 CPU 上重新提取会因 TF32 略有差异（D19），因此按原样提供。

## Earlier archives / 旧压缩包

The `_v2` archives replace all earlier ones. Do not mix the two sets.

`_v2` 压缩包取代下列所有旧压缩包，请不要混用。

| Earlier archive | Replaced by | Difference |
|---|---|---|
| `Checkpoints_merged_20261018.zip`, `Checkpoints_overfitting_diagnostic_20261018.zip` | the two `Checkpoints_*_v2` archives | same checkpoint files; the old ones had README.md / MANIFEST.csv / SHA256SUMS at the archive root, which collide with the repository README when unzipped at the root |
| `Feature_Caches_merged_20261018.zip` | `Feature_Caches_merged_20261018_v2.zip` | same four caches; `reproduction_distilbert_first_token/` added |
| `Run_Outputs_20261018.zip` | `Run_Outputs_20261018_v2.zip` | same run folders without the main run's `review/` folder (meme grid and `manual_review.csv`, both in Git); local path prefixes shortened in `logs/` and in two `checks/end_to_end_validation.json` |
| `Text_Features_DistilBERT_20261016.zip` | `Feature_Caches_merged_20261018_v2.zip` | same files, new location `data/processed/features/` |
| `Multimodal_Assets_20261016.zip` | `Checkpoints_merged_20261018_v2.zip`, `Feature_Caches_merged_20261018_v2.zip`, `Run_Outputs_20261018_v2.zip` | same 14 checkpoints and CPU CLIP cache; run folders and logs as in the row above |
| `Image_Features_CLIP_ViTB32_20261006.zip` (image group, [Image_Cloud_Assets.md](Image_Cloud_Assets.md)) | still valid; also inside `Feature_Caches_merged_20261018_v2.zip` | byte-identical cache |
