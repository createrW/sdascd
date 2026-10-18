# Image reproduction / 图片复现

Run all commands from the repository root after activating the environment and setting `MEMOTION_DATA_DIR` as described in the [root README](../README.md). Use a new run ID each time: existing results, checkpoints, and extraction folders are protected from overwrite. The commands below use Windows PowerShell; the Python module commands also work on other platforms.

所有命令从仓库根目录运行，先按[根目录 README](../README.md)启用环境并设置 `MEMOTION_DATA_DIR`。每次使用新 run ID，程序会保护已有结果、checkpoint 和提取目录，防止覆盖。以下示例使用 Windows PowerShell；Python 模块命令也适用于其他平台。

## Reference / 最终参考结果

Reference run: `20261006_code_fixes_validation_v2`. Train: 5,593; Validation: 699. Best epoch: 7 of 30. Macro-F1: 0.527606; accuracy: 0.626609. Confusion matrix (rows true 0/1, columns predicted 0/1): `[[59, 102], [159, 379]]`. No Test scoring was used to select settings.

最终参考实验：`20261006_code_fixes_validation_v2`。Train 5,593 张、Validation 699 张；30 个 epoch 中第 7 个最佳。Macro-F1 0.527606，准确率 0.626609。混淆矩阵行是真实类别、列是预测类别：`[[59, 102], [159, 379]]`。未使用 Test 分数调整设置。

Files / 文件：[reference outputs](../outputs/image/20261006_code_fixes_validation_v2/), [configuration](../configs/train/image_baseline.yaml), [final head](../checkpoints/image/20261006_code_fixes_validation_v2/best.pt).

The three retained `best.pt` files contain only small classifier heads: the original team baseline (`20261001_clip_vitb32_seed42`), Zhihan's earlier independent reproduction (`20261001_clip_vitb32_seed42_zhihan_rtx3080`), and the final reference run. They use the same frozen CLIP features. Zhihan's file is byte-identical to the final reference head (same SHA-256, as received from the image group), so the two paths hold one head. Historical results are in the external audit archive, not duplicated here.

保留的三个 `best.pt` 都只包含小型分类头，分别来自原始团队基线、Zhihan 的较早独立复现和最终参考实验。它们使用相同的冻结 CLIP 特征；其中 Zhihan 的文件与最终参考分类头逐字节相同（SHA-256 相同，图片组提供时即如此），两个路径是同一个分类头。历史结果在外部完整审计包中，此处不重复保存。

## A. Downloaded cache / 下载缓存快速复现

Get the cache from `Feature_Caches_merged_20261018_v2.zip` ([Multimodal_Assets.md](Multimodal_Assets.md); `unzip -n` at the repository root), or download `Image_Features_CLIP_ViTB32_20261006.zip` from the [cloud manifest](Image_Cloud_Assets.md) once its link is available, check its size and SHA-256 and extract it into `data/processed/`. It creates `data/processed/features/20261006_code_fixes_validation_v2/` with `train.pt`, `validation.pt`, and `meta.json`. The tensors contain 768-dimensional features, sample IDs, and labels; metadata and data alignment are checked before use. The original metadata for all three splits is still required for validation; this route does not read dataset image files. This route needs no encoder download or feature extraction.

从 `Feature_Caches_merged_20261018_v2.zip` 获取缓存（见 [Multimodal_Assets.md](Multimodal_Assets.md)，在仓库根目录 `unzip -n`）；或待[云端清单](Image_Cloud_Assets.md)提供链接后下载 `Image_Features_CLIP_ViTB32_20261006.zip`，核对大小和 SHA-256，解压到 `data/processed/`。将生成 `data/processed/features/20261006_code_fixes_validation_v2/`，包含 `train.pt`、`validation.pt`、`meta.json`。缓存包含 768 维特征、sample ID 和标签，使用前会检查元数据及样本一致性。仍需原始三个划分的元数据；此方式不读取数据集图片、不下载编码器、不重新提取特征。

```powershell
Get-FileHash 'D:\Downloads\Feature_Caches_merged_20261018_v2.zip' -Algorithm SHA256   # compare with docs/Multimodal_Assets.md
Expand-Archive -LiteralPath 'D:\Downloads\Feature_Caches_merged_20261018_v2.zip' -DestinationPath .   # creates data/processed/features/...
# alternative, once the image archive is uploaded:
# Expand-Archive -LiteralPath 'D:\Downloads\Image_Features_CLIP_ViTB32_20261006.zip' -DestinationPath data/processed
python -m src.evaluation.evaluate --run_id 20261006_code_fixes_validation_v2 --features_dir data/processed/features/20261006_code_fixes_validation_v2 --out_dir outputs/image/quick_checkpoint_check
python -m scripts.verify_predictions outputs/image/quick_checkpoint_check/predictions_validation.csv
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/reproduce_image.ps1 -RunId my_cached_reproduction_01 -FeaturesDir data/processed/features/20261006_code_fixes_validation_v2
```

The quick check evaluates the shipped final head. The script additionally trains a fresh head and checks checkpoint reload consistency. If paths already exist, choose new output/run names. `Expand-Archive` is shown without `-Force` to protect existing cache files.

快速检查评估随仓库提供的最终分类头。脚本还会重新训练分类头并检查 checkpoint 重载一致性。路径已存在时请换新名字；解压示例未使用 `-Force`，以保护已有缓存。

## B. Extract features again / 重新提取特征完整复现

No feature archive is needed. This route downloads the pinned public CLIP model on first use, extracts the complete Train/Validation cache, trains a fresh head, verifies predictions, and checks the saved head after reload. Full dataset images are required; limited `--limit` caches are only for smoke checks and are rejected for formal training/evaluation. The script honors an existing `HF_HOME`; otherwise it uses local `hf_cache/`.

此方式无需特征包。首次运行下载固定版本的公开 CLIP 模型，重新提取完整 Train/Validation 特征、训练分类头、验证预测并检查保存后的重载结果。需要完整数据集图片；带 `--limit` 的缓存仅用于 smoke 检查，正式训练和评估会拒绝使用。脚本优先使用已有 `HF_HOME`，否则使用本地 `hf_cache/`。

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/reproduce_image.ps1 -RunId my_full_reproduction_01 -Reextract
```

On Linux/macOS (no PowerShell) the same steps are / 在 Linux/macOS 上等价的步骤：

```bash
python -m src.features.feature_engineering image --splits train validation --out_dir data/processed/features/my_full_reproduction_01
python -m src.models.trainer --config configs/train/image_baseline.yaml --features_dir data/processed/features/my_full_reproduction_01 --run_id my_full_reproduction_01
python -m scripts.verify_predictions outputs/image/my_full_reproduction_01/predictions_validation.csv
python -m src.evaluation.evaluate --run_id my_full_reproduction_01 --features_dir data/processed/features/my_full_reproduction_01
python -m scripts.verify_reproduction --reference_metrics outputs/image/20261006_code_fixes_validation_v2/metrics_validation.json --candidate_metrics outputs/image/my_full_reproduction_01/metrics_validation.json
```

Results are saved to `outputs/image/<RunId>/`, checkpoints to `checkpoints/image/<RunId>/`, and step logs to `outputs/reproduction/<RunId>/`. Full extraction writes `data/processed/features/<RunId>/`. Fresh training is compared to the final reference with absolute Macro-F1 tolerance 0.01. Reload must match the newly trained predictions exactly on the same device. Differences outside tolerance cause a nonzero exit code. Hardware variation is allowed within the metric tolerance; exact reference probabilities across devices are not assumed.

结果保存在 `outputs/image/<RunId>/`，分类头在 `checkpoints/image/<RunId>/`，逐步日志在 `outputs/reproduction/<RunId>/`，完整提取的特征在 `data/processed/features/<RunId>/`。新训练与最终参考的 Macro-F1 绝对容差为 0.01；同一设备重载后的预测必须与新训练完全一致。超出容差返回非零退出码。不同硬件允许指标在容差内变化，不假设跨设备概率完全相同。

## Tests / 测试

```powershell
python -m unittest discover -s tests -v
python -m pip check
```

The suite checks overwrite protection, cache completeness and alignment, failure propagation, and CSV precision at threshold boundaries. See [current verification results](Image_Test_Results.md) for the image group's clean-branch runs (original team repository). Neither reproduction route scores Test.

测试覆盖覆盖保护、缓存完整性与样本对齐、失败状态传递和阈值附近的 CSV 精度。图片组精简分支（原团队仓库）的实际运行见[检测结果](Image_Test_Results.md)。两种复现方式都不评分 Test。
