# Image baseline / 图片基线

Frozen `openai/clip-vit-base-patch32` at revision `3d74acf`, official RGB preprocessing, 768-dimensional vision `pooler_output`, and a trainable linear 768-to-2 classification head. Seed 42, weighted cross-entropy, AdamW, batch size 64, 30 epochs, threshold 0.5; checkpoint selection uses validation Macro-F1.

固定版本 `3d74acf` 的冻结 CLIP ViT-B/32，官方 RGB 预处理、768 维 vision `pooler_output`，训练 768 到 2 的线性分类头。seed 42、加权交叉熵、AdamW、batch size 64、30 epochs、阈值 0.5；按 Validation Macro-F1 选择 checkpoint。

Use [the current reproduction guide](../../docs/Image_Reproduction.md) from the repository root for both cache-based and complete extraction commands. View [curated results and manual review](../../Team_Review/image/README.md). Dataset access uses `MEMOTION_DATA_DIR`.

从仓库根目录使用[当前复现说明](../../docs/Image_Reproduction.md)，其中包含缓存和完整提取命令。[精简结果与人工检查](../../Team_Review/image/README.md)可直接查看。数据集统一通过 `MEMOTION_DATA_DIR` 访问。

`extract_features.py` writes a new cache, `cache.py` validates it, `train.py` trains the head, `evaluate.py` replays saved heads, and `error_analysis.py` produces diagnostic examples. `safety.py` protects existing paths. Text remains in the original meme pixels; there is no separate OCR/text input to the image model.

`extract_features.py` 写入新缓存，`cache.py` 验证缓存，`train.py` 训练分类头，`evaluate.py` 评估已有分类头，`error_analysis.py` 生成诊断样本，`safety.py` 保护已有路径。文字保留在原始图片像素中，图片模型没有独立 OCR 或文本输入。
