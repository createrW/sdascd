# Meme Sarcasm Robustness

CSCI 566 baselines for binary meme sarcasm detection: text (DistilBERT), image (frozen CLIP ViT-B/32 + a linear head), and multimodal (frozen text/image encoders + a fusion MLP).

CSCI 566 梗图讽刺二分类项目：文字模型 DistilBERT；图片模型使用冻结的 CLIP ViT-B/32 和线性分类头；多模态模型组合文字、图片特征和融合 MLP。

## Start here / 从这里开始

| Goal / 目的 | Open / 查看 |
|---|---|
| Multimodal results, ablations, review / 多模态结果、对照与人工核对 | [Team_Review/multimodal/README.md](Team_Review/multimodal/README.md) |
| Multimodal pipeline and commands, DistilBERT text features / 多模态流程、命令与 DistilBERT 文字特征 | [src/multimodal/README.md](src/multimodal/README.md) |
| Multimodal verification / 多模态检测记录 | [docs/Multimodal_Test_Results.md](docs/Multimodal_Test_Results.md) |
| Image results and 32 reviewed samples / 图片结果和 32 个人工检查样本 | [Team_Review/image/README.md](Team_Review/image/README.md) |
| Cached or complete image reproduction / 缓存复现或完整图片复现 | [docs/Image_Reproduction.md](docs/Image_Reproduction.md) |
| Latest clean-branch checks / 精简分支检测结果 | [docs/Image_Test_Results.md](docs/Image_Test_Results.md) |
| External archives and cache download status / 云端审计资料和缓存下载状态 | [docs/Image_Cloud_Assets.md](docs/Image_Cloud_Assets.md) |
| Shared decisions and experiment history / 共同决策与实验记录 | [docs/decisions.md](docs/decisions.md), [docs/experiment_log.md](docs/experiment_log.md) |

This branch keeps one final image reference run, three small classifier heads, and the current image code. The full `image-baseline` branch remains available for historical audits. Datasets, feature tensors, downloaded encoders, and new experiment outputs stay outside Git. Google Drive archives are prepared locally; download links will be added after upload.

本分支保留一套最终图片 reference run、三个小型分类头和当前图片代码。完整 `image-baseline` 分支继续保留历史审计。数据集、特征缓存、下载的编码器和新实验输出不提交到 Git。Google Drive 资料已在本地准备，实际上传后补充下载链接。

The multimodal additions keep two reference runs in Git: `outputs/multimodal/20261016_distilbert_clip_seed42` and its independent reproduction. Text feature caches, multimodal checkpoints, all other runs and step logs are packaged as zips; see [docs/Multimodal_Assets.md](docs/Multimodal_Assets.md).

多模态部分在 Git 中保留两套参考实验：`outputs/multimodal/20261016_distilbert_clip_seed42` 及其独立复现。文字特征缓存、多模态 checkpoint、其余实验和逐步日志打包为 zip，见 [docs/Multimodal_Assets.md](docs/Multimodal_Assets.md)。

## Status / 当前状态

Shared dataset validation, the image baseline and the multimodal baseline are complete. The multimodal pipeline (frozen DistilBERT `[CLS]` + frozen CLIP `pooler_output`, L2, concat, MLP 1536-512-2) has text feature caching, training, evaluation, reproduction, an end-to-end check and a manual review, all run on CPU. Validation Macro-F1 is 0.5336 for seed 42 and 0.5364 ± 0.0067 over 3 seeds. The Text Group's DistilBERT classifier is not part of this branch.

共同数据验证、图片基线和多模态基线已完成。多模态流程（冻结 DistilBERT `[CLS]` + 冻结 CLIP `pooler_output`、L2、拼接、MLP 1536-512-2）包含文字特征缓存、训练、评估、复现、端到端检查和人工核对，全部在 CPU 上运行。Validation Macro-F1：seed 42 为 0.5336，3 个 seed 为 0.5364 ± 0.0067。文字组的 DistilBERT 分类器不在本分支中。

## Environment / 环境

Run every command below from the repository root. Use Python 3.12, PyTorch 2.11.0, torchvision 0.26.0, and Transformers 5.18.0. The lock file targets Windows/NVIDIA CUDA 12.8 and includes Apple Silicon platform markers. For different hardware, use its compatible PyTorch build with `requirements.txt` and record the installed versions.

以下命令均从仓库根目录运行。使用 Python 3.12、PyTorch 2.11.0、torchvision 0.26.0 和 Transformers 5.18.0。锁定依赖面向 Windows/NVIDIA CUDA 12.8，包含 Apple Silicon 平台条件。其他硬件请安装兼容的 PyTorch 和 `requirements.txt`，并记录实际版本。

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-lock.txt
python -m pip check
python -m scripts.show_environment
```

For the tested PyTorch wheel, an up-to-date NVIDIA driver is sufficient; a separate CUDA Toolkit or cuDNN installation is unnecessary.

对于已测试的 PyTorch wheel，只需要适配的 NVIDIA 驱动，无需另装 CUDA Toolkit 或 cuDNN。

## Dataset / 数据集

Obtain the team dataset separately. Its root contains `train/`, `validation/`, and `test/`, each with `metadata.csv` and the referenced images. Set `MEMOTION_DATA_DIR` in the terminal used for all commands. The code validates this data without changing it. Validation selects checkpoints; Test is reserved for the final evaluation.

数据集由组员单独获取；根目录包含 `train/`、`validation/`、`test/`，每个划分都有 `metadata.csv` 和相应图片。所有运行统一在当前终端设置 `MEMOTION_DATA_DIR`。验证程序不修改数据；Validation 用于选择 checkpoint，Test 留到最终评估。

```powershell
$env:MEMOTION_DATA_DIR = 'D:\path\to\MMSoc_Memotion_exported'
python -m scripts.verify_dataset
```

Expected / 预期：`DATASET OK`.

## Image baseline / 图片基线

Read [the image reproduction guide](docs/Image_Reproduction.md) for both cache-based and full extraction routes. The image encoder receives the original RGB meme through CLIP's official resize/center crop/normalization, including any visible text pixels. Do not deliberately mask text. `pooler_output` is 768-dimensional; the frozen encoder is followed by a 768-to-2 head, seed 42, threshold 0.5. Reference validation Macro-F1 is 0.527606 and accuracy is 0.626609.

[图片复现说明](docs/Image_Reproduction.md)包含缓存快速复现和重新提取特征两种方式。编码器接收原始 RGB 梗图，使用 CLIP 官方缩放、中心裁剪与归一化，包含图中可见的文字像素；不故意遮挡文字。使用 768 维 `pooler_output` 和 768 到 2 的分类头，seed 42、阈值 0.5。最终 reference run 的 Validation Macro-F1 为 0.527606，准确率为 0.626609。

## Multimodal baseline / 多模态基线

Commands for text features, training, evaluation, reproduction and the end-to-end check are in [src/multimodal/README.md](src/multimodal/README.md); results are in [Team_Review/multimodal/README.md](Team_Review/multimodal/README.md). Minimal route with the image group's cache unzipped at the repository root:

文字特征、训练、评估、复现与端到端检查的命令见 [src/multimodal/README.md](src/multimodal/README.md)，结果见 [Team_Review/multimodal/README.md](Team_Review/multimodal/README.md)。将图片组缓存解压到仓库根目录后的最简流程：

```powershell
python -m src.text.extract_features
python -m src.multimodal.train --run_id 20261016_distilbert_clip_seed42_my_copy
python -m scripts.verify_predictions outputs/multimodal/20261016_distilbert_clip_seed42_my_copy/predictions_validation.csv
```

## Multimodal smoke test / 多模态 smoke test

The existing smoke test uses the pinned public text revision `1c4513b2eedbda136f57676a34eea67aba266e5c` and image revision `3d74acf`. Its first run downloads the encoders locally. See [src/multimodal/README.md](src/multimodal/README.md).

现有 smoke test 使用固定文字模型 revision `1c4513b2eedbda136f57676a34eea67aba266e5c` 和图片 revision `3d74acf`。首次运行会在本地下载编码器。详见 [src/multimodal/README.md](src/multimodal/README.md)。

```powershell
python -m src.multimodal.smoke_test --batch_size 2
```

Expected / 预期：text `(2, 768)`, image `(2, 768)`, fused `(2, 1536)`, logits `(2, 2)`; `MULTIMODAL SMOKE TEST OK`.
