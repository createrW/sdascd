# Final image reference / 最终图片参考实验

Run: `20261006_code_fixes_validation_v2`; seed 42, frozen CLIP ViT-B/32 (`3d74acf`), 768-dimensional `pooler_output`, linear head. Train 5,593 / Validation 699. Best epoch 7/30. Validation Macro-F1 0.527606, accuracy 0.626609; confusion matrix `[[59, 102], [159, 379]]` (rows true, columns predicted).

实验 seed 42，冻结 CLIP ViT-B/32（`3d74acf`）、768 维 `pooler_output` 和线性分类头。Train 5,593 / Validation 699，最佳 epoch 7/30，Macro-F1 0.527606、准确率 0.626609，混淆矩阵行是真实类别、列是预测类别。

The final classifier head is tracked at `checkpoints/image/20261006_code_fixes_validation_v2/best.pt`. Other files in this directory preserve the archived reference bytes. Run commands from the repository root using [the reproduction guide](../../../docs/Image_Reproduction.md).

最终分类头已纳入仓库：`checkpoints/image/20261006_code_fixes_validation_v2/best.pt`。本目录其他文件保留参考实验的原始字节。运行命令请从仓库根目录使用[复现说明](../../../docs/Image_Reproduction.md)。
