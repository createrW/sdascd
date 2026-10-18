# Meme Sarcasm Robustness

CSCI 566 baselines for binary meme sarcasm detection on Memotion: an image model (frozen CLIP ViT-B/32 + a linear head), a multimodal model (frozen DistilBERT + frozen CLIP + a fusion MLP) with unimodal ablations, and an HTTP service for the multimodal model. Everything runs on CPU.

CSCI 566 梗图讽刺二分类项目（Memotion）：图片模型（冻结 CLIP ViT-B/32 + 线性分类头）、多模态模型（冻结 DistilBERT + 冻结 CLIP + 融合 MLP）及单模态对照，以及多模态模型的 HTTP 服务。全部可在 CPU 上运行。

## Goals / 目标

* Reproducible team baselines under one data contract: fixed splits, `[NO_TEXT]` for blank captions, train-only class weights, best epoch by validation Macro-F1, threshold 0.5, Test reserved for the end.
* One code path from cached features (training) to raw inputs (serving), checked numerically.
* Clear evidence: statistics, figures and conclusions in the [results report](Team_Review/multimodal/RESULTS_REPORT.md).

* 统一数据约定下可复现的基线：固定划分、空文字用 `[NO_TEXT]`、类别权重只来自 train、按 Validation Macro-F1 选 epoch、阈值 0.5、Test 留到最后。
* 从缓存特征（训练）到原始输入（服务）只有一条代码路径，并有数值检查。
* 统计、图表和结论见[结果报告](Team_Review/multimodal/RESULTS_REPORT.md)。

## Results at a glance / 结果概览 (validation, 699 memes)

| Model / 模型 | Macro-F1 | Notes / 说明 |
|---|---:|---|
| Majority class / 多数类 | 0.4349 | always "sarcastic" |
| Image baseline: CLIP + linear head (image group) | 0.5276 | `outputs/image/20261006_code_fixes_validation_v2` |
| **Multimodal baseline: DistilBERT `[CLS]` + CLIP, MLP** | **0.5336** (3 seeds 0.5364 ± 0.0067) | `outputs/multimodal/20261016_distilbert_clip_seed42` |
| Image only, same MLP / 仅图片 | 0.5459 ± 0.0108 | ablation, 3 seeds |
| Text only, same MLP / 仅文字 | 0.4862 ± 0.0079 | ablation, 3 seeds |
| Mean-pooled text + CLIP | 0.5280 ± 0.0032 | ablation, 3 seeds |

A label-independent classifier scores about 0.50. The signal comes from the image features; frozen text features add little ([report](Team_Review/multimodal/RESULTS_REPORT.md)). Test has not been scored.

不看标签的分类器约为 0.50；信号主要来自图片特征，冻结的文字特征贡献很小。Test 尚未评分。

## Quick start / 快速启动

Run from the repository root. / 以下命令均在仓库根目录运行。

```bash
# 1. environment (CPU shown; see "Environment" below for the locked CUDA setup)
python3.12 -m venv .venv && source .venv/bin/activate
pip install torch torchvision                 # or the build for your GPU
pip install -r requirements.txt               # optional: pip install -e ".[serving,dev]"

# 2. dataset (never committed): point to it, or unpack it into data/raw/MMSoc_Memotion_exported/
export MEMOTION_DATA_DIR=/path/to/MMSoc_Memotion_exported
python -m scripts.verify_dataset              # -> DATASET OK

# 3. features -> data/processed/features/ (or unzip the shared caches into data/processed/, see data/README.md)
bash scripts/extract_features.sh text         # DistilBERT [CLS] + mean pooling, ~1.5 min on CPU
#    image: unzip the image group's cache (the multimodal configs' default, data/processed/features/20261006_code_fixes_validation_v2)
#    or extract your own (~4 min on 4 CPU cores):
bash scripts/extract_features.sh image --splits train validation test --out_dir data/processed/features/clip_vitb32_pooler_cpu

# 4. train and evaluate (feature folders default to the config's `features:` block)
bash scripts/train.sh configs/train/multimodal_baseline.yaml --run_id my_multimodal_run \
     --image_features_dir data/processed/features/clip_vitb32_pooler_cpu   # omit with the image group's cache
bash scripts/train.sh configs/train/image_baseline.yaml --run_id my_image_run \
     --features_dir data/processed/features/clip_vitb32_pooler_cpu
bash scripts/eval.sh my_multimodal_run        # reload; compares with the reported metrics

# 5. serve (FastAPI; docs/api_spec.md). The default head (configs/deploy/serving.yaml) comes from
#    Multimodal_Assets_20261016.zip (docs/Multimodal_Assets.md); or serve the run you just trained:
CHECKPOINT=checkpoints/multimodal/my_multimodal_run/best.pt bash scripts/serve.sh
curl -s -X POST localhost:8000/predict -H 'Content-Type: application/json' -d @data/samples/predict_request.json

# 6. tests (test_serving needs no dataset; the others read MEMOTION_DATA_DIR)
python -m unittest discover -s tests -v

# 7. notebooks (aggregate statistics only)
pip install jupyterlab && jupyter lab notebooks/
```

On Windows use the same `python -m ...` commands that the `.sh` scripts wrap (each script is a few lines), with `$env:MEMOTION_DATA_DIR = 'D:\path\to\MMSoc_Memotion_exported'`.

Windows 上直接运行 `.sh` 脚本中对应的 `python -m ...` 命令，并用 `$env:MEMOTION_DATA_DIR = ...` 设置数据路径。

## Directory guide / 目录介绍

```text
configs/
  train/            image_baseline.yaml, multimodal_baseline.yaml, multimodal_{text_only,image_only,mean_pooling}.yaml
  deploy/           serving.yaml                      inference / API settings
data/               (git-ignored except README.md and samples/)  see data/README.md
  raw/              the exported dataset, read-only
  processed/        feature caches: features/<name>/{train,validation,test}.pt + meta.json
  samples/          synthetic 4-row splits + API request bodies (committed)
docs/               design.md, experiment_log.md, api_spec.md, decisions.md, image/multimodal guides
notebooks/          eda.ipynb (data exploration), poc_model.ipynb (model PoC on cached features)
src/
  data/             dataset.py (contract, labels), preprocess.py ([NO_TEXT], RGB, tokenize), loader.py (joint cache loader)
  features/         feature_engineering.py (CLIP / DistilBERT extraction), cache.py (cache validation)
  models/           base_model.py, vit_classifier.py (image head), multimodal_classifier.py, trainer.py
  evaluation/       metrics.py, report.py, evaluate.py, reproduction.py, end_to_end.py,
                    error_analysis.py, review_samples.py, results_report.py
  serving/          predictor.py (inference wrapper + smoke test), app.py (FastAPI)
  utils/            config.py (YAML/JSON), reproducibility.py (seeds, device), safety.py (no overwrites)
tests/              test_data_loader.py, test_model.py, test_serving.py (+ helpers.py)
scripts/            train.sh, eval.sh, serve.sh, extract_features.sh, reproduce_image.ps1, verify_*.py, ...
outputs/            run folders (curated reference runs only in Git)
checkpoints/        trained heads (only the image group's three small heads in Git)
Team_Review/        curated results, manual reviews, the multimodal results report
requirements.txt, requirements-lock.txt, setup.py, Dockerfile, .dockerignore, .gitignore
```

## Documents / 文档

| Goal / 目的 | Open / 查看 |
|---|---|
| **Multimodal results report (statistics, figures, conclusions) / 多模态结果报告** | [Team_Review/multimodal/RESULTS_REPORT.md](Team_Review/multimodal/RESULTS_REPORT.md) |
| Overfitting diagnostic (same-parameter train/val scoring, head-size sweep) / 过拟合诊断 | [Team_Review/multimodal/overfitting/README.md](Team_Review/multimodal/overfitting/README.md) |
| Technical design / 技术方案设计 | [docs/design.md](docs/design.md) |
| Model API / 模型接口 | [docs/api_spec.md](docs/api_spec.md) |
| Experiment log and A/B comparisons / 实验记录 | [docs/experiment_log.md](docs/experiment_log.md) |
| Shared decisions / 共同决策 | [docs/decisions.md](docs/decisions.md) |
| Multimodal pipeline, commands, DistilBERT text features / 多模态流程与命令 | [docs/multimodal_baseline.md](docs/multimodal_baseline.md) |
| Multimodal results, ablations, review / 多模态结果与人工核对 | [Team_Review/multimodal/README.md](Team_Review/multimodal/README.md) |
| Multimodal verification and assets / 多模态检测记录与资料 | [docs/Multimodal_Test_Results.md](docs/Multimodal_Test_Results.md), [docs/Multimodal_Assets.md](docs/Multimodal_Assets.md) |
| Image baseline, results and 32 reviewed samples / 图片基线与人工检查 | [docs/image_baseline.md](docs/image_baseline.md), [Team_Review/image/README.md](Team_Review/image/README.md) |
| Cached or complete image reproduction / 图片复现 | [docs/Image_Reproduction.md](docs/Image_Reproduction.md), [docs/Image_Test_Results.md](docs/Image_Test_Results.md), [docs/Image_Cloud_Assets.md](docs/Image_Cloud_Assets.md) |
| Data folders and cache archives / 数据目录与缓存包 | [data/README.md](data/README.md) |

## Migration / 旧命令对照

The code moved from `src/{common,image,text,multimodal}/` to the layout above on 2026-10-17 (decision D22). Behaviour is unchanged: retraining with the new code reproduced the recorded runs byte for byte, and every shipped head replays identically ([experiment log](docs/experiment_log.md)). Run folders recorded before the move still name the old commands; use this table.

2026-10-17 起代码由 `src/{common,image,text,multimodal}/` 迁移到上述结构（D22），行为不变：用新代码重新训练与已记录结果逐字节一致，所有已提供的分类头重放结果一致。旧实验记录中的命令按下表对应。

| Before / 之前 | Now / 现在 |
|---|---|
| `python -m src.image.extract_features ...` | `python -m src.features.feature_engineering image ...` |
| `python -m src.text.extract_features ...` | `python -m src.features.feature_engineering text ...` |
| `python -m src.image.train [--config configs/image_baseline.json]` | `python -m src.models.trainer --config configs/train/image_baseline.yaml` |
| `python -m src.multimodal.train [--config configs/ablations/<x>.json]` | `python -m src.models.trainer --config configs/train/multimodal_baseline.yaml` (`multimodal_<x>.yaml`) |
| `python -m src.image.evaluate` / `python -m src.multimodal.evaluate` | `python -m src.evaluation.evaluate` (task detected from the checkpoint) |
| `python -m src.image.error_analysis` | `python -m src.evaluation.error_analysis` |
| `python -m src.multimodal.check_end_to_end` | `python -m src.evaluation.end_to_end` |
| `python -m src.multimodal.review_samples` | `python -m src.evaluation.review_samples` |
| `python -m src.multimodal.smoke_test` | `python -m src.serving.predictor --smoke` (`--train_config ...` for an untrained head) |
| `python -m scripts.make_multimodal_report` | `python -m src.evaluation.results_report` |
| `features/<name>/` | `data/processed/features/<name>/` (old paths still resolve, both ways) |
| `configs/*.json`, `configs/ablations/*.json` | `configs/train/*.yaml` (JSON still loads) |

## Status / 当前状态

Shared dataset validation, the image baseline and the multimodal baseline are complete. The multimodal pipeline has text feature caching, training and evaluation, all run on CPU, plus a same-agent rerun; the teammate's second-member reproduction is still pending. The end-to-end check passes on CPU caches; the main run's CUDA-cache head misses the 0.01 probability tolerance because of TF32 (D19). The manual review is a first pass awaiting a teammate's confirmation. The Text Group's fine-tuned DistilBERT classifier is not part of this branch.

共同数据验证、图片基线和多模态基线已完成。多模态流程的文字特征缓存、训练和评估都在 CPU 上完成，另有一次同一 agent 的重跑；组员的第二名成员复现仍待完成。端到端检查在 CPU 缓存上通过；主实验基于 CUDA 缓存的分类头因 TF32 未达到 0.01 的概率容差（D19）。人工核对是初步结果，需要组员确认。文字组微调的 DistilBERT 分类器不在本分支中。

This branch keeps one final image reference run, three small image heads and two multimodal reference runs (`outputs/multimodal/20261016_distilbert_clip_seed42` and its rerun on CPU image features). Datasets, feature tensors, downloaded encoders, other checkpoints and new outputs stay outside Git; the text caches, multimodal checkpoints and remaining runs are packaged as zips ([docs/Multimodal_Assets.md](docs/Multimodal_Assets.md), upload pending).

本分支保留一套最终图片 reference run、三个小型图片分类头和两套多模态参考实验。数据集、特征、下载的编码器、其他 checkpoint 和新输出不提交到 Git；文字特征、多模态 checkpoint 和其余实验已打包（见 Multimodal_Assets.md，待上传）。

## Environment / 环境

Python 3.12, PyTorch 2.11.0, torchvision 0.26.0, Transformers 5.18.0. `requirements-lock.txt` is the image group's tested environment (Windows/NVIDIA CUDA 12.8, with Apple Silicon markers); add `fastapi` and `uvicorn` for serving. On other hardware install the matching PyTorch build and `requirements.txt`, and record the versions (`python -m scripts.show_environment`).

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-lock.txt
python -m pip check
python -m scripts.show_environment
```

For the tested PyTorch wheel an up-to-date NVIDIA driver is enough; no separate CUDA Toolkit or cuDNN is needed. Machines without Hugging Face access can pass local copies of the pinned encoders (`--model_dir`, `--text_model_dir`, `--clip_model_dir`, or `*_model_dir` in `configs/deploy/serving.yaml`; D20).

## Rules / 约定

* Never commit the dataset or its zip, feature caches, encoder weights, Hugging Face caches or model checkpoints (team README 9.4, `.gitignore`). Deliberate exceptions, kept by the groups for review and reproduction:
  * dataset-derived review evidence: `Team_Review/image/images/*_review.png` (grids of 32 image-group samples), `Team_Review/image/manual_review/*.csv` (their sample IDs, labels and caption excerpts) and `outputs/multimodal/20261016_distilbert_clip_seed42/review/` (16 validation memes with captions); every `predictions_*.csv` also lists the dataset label of each `sample_id`;
  * the image group's three 8.6 KB linear heads in `checkpoints/image/`.
  Remove the review evidence before making the repository public. New outputs, including `error_analysis/` and re-evaluation folders inside the reference runs, stay git-ignored.
* Validation selects epochs and compares models; Test is scored once, at the end (`--split test`).
* Existing run, checkpoint and cache folders are never overwritten: choose a new `--run_id` / output folder.
* Record every formal run in [docs/experiment_log.md](docs/experiment_log.md) and team-wide choices in [docs/decisions.md](docs/decisions.md).
