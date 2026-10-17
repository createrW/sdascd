# External image assets / 图片外部资料

**Status: prepared locally; not uploaded.** The team plans to use Google Drive, but no destination link has been provided yet. The download URL fields in [Image_Cloud_Assets.csv](Image_Cloud_Assets.csv) are intentionally empty. After upload, add a working file download link for each archive and keep its recorded size and SHA-256 unchanged. The full extraction reproduction route works without these archives.

**状态：已在本地准备，尚未上传。** 团队计划使用 Google Drive，但尚未提供目标地址。[Image_Cloud_Assets.csv](Image_Cloud_Assets.csv) 的下载地址暂时留空。实际上传后为每个压缩包填写有效文件下载链接，保留记录的大小与 SHA-256。完整特征提取复现不依赖这些压缩包。

All archives already contain the destination folders listed below. Extract feature archives (top-level `features/`) **into `data/processed/`**, and the others **from the repository root**. Do not select the listed inner folder as the extraction root. Archives contain no dataset images or downloaded encoder weights. `data/processed/`, `features/` and `external/` are ignored by Git; a cache left at the old `features/<name>` location is still found.

包内已包含下表目标目录。特征包（顶层为 `features/`）**解压到 `data/processed/`**，其他包**从仓库根目录解压**，不要把目标子目录再次作为解压根目录。包中不包含数据集图片或下载的编码器权重。`data/processed/`、`features/` 和 `external/` 已被 Git 忽略；仍放在旧位置 `features/<name>` 的缓存也能被找到。

| Filename / 文件名 | Bytes / 字节 | MiB | Creates / 生成目录 | Download / 下载 |
|---|---:|---:|---|---|
| `Image_Baseline_Full_Audit_51c6dbd.zip` | 16408026 | 15.648 | `external/image_audit/image-baseline-51c6dbd/` | Pending upload / 待上传 |
| `Image_Features_CLIP_ViTB32_20261006.zip` | 17939572 | 17.109 | `data/processed/features/20261006_code_fixes_validation_v2/` | Pending upload / 待上传 |
| `Image_Baseline_Clean_Verification_20261009.zip` | 557951 | 0.532 | `external/image_validation/20261009/` | Pending upload / 待上传 |

## SHA-256

```text
865b0d724fad6b31790af22322deea2be3cc36d423815282257fda703d827823  Image_Baseline_Full_Audit_51c6dbd.zip
7a02d46bab9e7bd4d15b5ebb6884d2805d94cb5e03c0170c75a056e0588e5662  Image_Features_CLIP_ViTB32_20261006.zip
4a1a93ff2d2c37b2819637f8f99c2f862d2d55400bf0eacd83e22aebeb20ec63  Image_Baseline_Clean_Verification_20261009.zip
```

The audit archive is an exact file-tree snapshot of the preserved `image-baseline` commit `51c6dbd`, including earlier experiments, full logs, Excel tables, repeated grids, and code snapshots. Git history remains in that branch. Files inside it are historical evidence; follow the current reproduction guide for runnable instructions.

完整审计包是保留分支 `image-baseline` 的 `51c6dbd` 文件树快照，包含早期实验、完整日志、Excel 表格、重复图集与代码快照。Git 历史仍保留在该分支。审计包内容是历史证据，运行程序请使用当前复现说明。

The feature archive contains complete Train/Validation tensors (768-dimensional `pooler_output`, sample IDs and labels) plus metadata. It is used only for the fast reproduction route. The verification archive contains the fresh checks performed on the clean branch. None of these ZIP files or feature tensors is tracked in this branch.

特征包包含完整 Train/Validation 张量（768 维 `pooler_output`、sample ID、标签）及元数据，用于快速复现。检测包包含精简分支本次实际验证的证据。本分支不跟踪这些 ZIP 或特征张量。

Before extraction, compare `Get-FileHash <archive-path> -Algorithm SHA256` with this list. Then, from the repository root and without `-Force`, run `Expand-Archive -LiteralPath <archive-path> -DestinationPath data/processed` for the feature archive and `-DestinationPath .` for the two others. (A feature archive expanded at the root, into `features/`, is still found.)

解压前，使用 `Get-FileHash <archive-path> -Algorithm SHA256` 核对清单；再从仓库根目录、不使用 `-Force`：特征包运行 `Expand-Archive -LiteralPath <archive-path> -DestinationPath data/processed`，另外两个包使用 `-DestinationPath .`。（解压到根目录 `features/` 的特征包也能被找到。）
