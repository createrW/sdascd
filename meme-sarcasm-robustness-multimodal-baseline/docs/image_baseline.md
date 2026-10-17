# Image baseline / 图片基线

Frozen `openai/clip-vit-base-patch32` at revision `3d74acf`, official RGB preprocessing, 768-dimensional vision `pooler_output`, and a trainable linear 768-to-2 classification head. Seed 42, weighted cross-entropy, AdamW, batch size 64, 30 epochs, threshold 0.5; checkpoint selection uses validation Macro-F1.

固定版本 `3d74acf` 的冻结 CLIP ViT-B/32，官方 RGB 预处理、768 维 vision `pooler_output`，训练 768 到 2 的线性分类头。seed 42、加权交叉熵、AdamW、batch size 64、30 epochs、阈值 0.5；按 Validation Macro-F1 选择 checkpoint。

Use [the current reproduction guide](Image_Reproduction.md) from the repository root for both cache-based and complete extraction commands. View [curated results and manual review](../Team_Review/image/README.md). Dataset access uses `MEMOTION_DATA_DIR`.

从仓库根目录使用[当前复现说明](Image_Reproduction.md)，其中包含缓存和完整提取命令。[精简结果与人工检查](../Team_Review/image/README.md)可直接查看。数据集统一通过 `MEMOTION_DATA_DIR` 访问。

| Step / 步骤 | Command / 命令 | Code / 代码 |
|---|---|---|
| Extract features / 提取特征 | `python -m src.features.feature_engineering image` | `src/features/feature_engineering.py` (writes), `src/features/cache.py` (validates) |
| Train the head / 训练分类头 | `python -m src.models.trainer --config configs/train/image_baseline.yaml` | `src/models/vit_classifier.py` (`ImageHead`), `src/models/trainer.py` |
| Replay a saved head / 重放分类头 | `python -m src.evaluation.evaluate --run_id <run_id> --features_dir <cache>` | `src/evaluation/evaluate.py` |
| Diagnostic examples / 诊断样本 | `python -m src.evaluation.error_analysis --run_id <run_id>` | `src/evaluation/error_analysis.py` |

Before the restructure these were `src/image/{extract_features,cache,model,train,evaluate,error_analysis}.py`; `src/utils/safety.py` (formerly `src/image/safety.py`) protects existing paths. The three shipped heads load unchanged (same `net.*` weight names). Text remains in the original meme pixels; there is no separate OCR/text input to the image model.

重构前对应 `src/image/{extract_features,cache,model,train,evaluate,error_analysis}.py`；`src/utils/safety.py`（原 `src/image/safety.py`）保护已有路径。三个保留的分类头无需修改即可加载（权重名 `net.*` 不变）。文字保留在原始图片像素中，图片模型没有独立 OCR 或文本输入。
