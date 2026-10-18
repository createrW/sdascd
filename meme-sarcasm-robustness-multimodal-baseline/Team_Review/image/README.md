# Image group review / 图片组资料

For a quick review: inspect the final results below, then the English or Chinese manual review CSV with its matching image grids. For running code, use the [reproduction guide](../../docs/Image_Reproduction.md).

快速查看：先看下方最终结果，再打开中英文人工检查 CSV 和对应图集。运行程序请使用[复现说明](../../docs/Image_Reproduction.md)。

## Final reference / 最终参考实验

Run `20261006_code_fixes_validation_v2`: frozen CLIP ViT-B/32, 768-dimensional `pooler_output`, linear head, seed 42, threshold 0.5. Validation Macro-F1 **0.527606**, accuracy **0.626609**, best epoch **7/30**.

实验 `20261006_code_fixes_validation_v2`：冻结 CLIP ViT-B/32，768 维 `pooler_output`、线性分类头、seed 42、阈值 0.5。Validation Macro-F1 **0.527606**，准确率 **0.626609**，最佳 epoch **7/30**。

| Artifact / 文件 | Link / 链接 |
|---|---|
| Metrics / 指标 | [metrics_validation.json](../../outputs/image/20261006_code_fixes_validation_v2/metrics_validation.json) |
| 699 validation predictions / 699 条预测 | [predictions_validation.csv](../../outputs/image/20261006_code_fixes_validation_v2/predictions_validation.csv) |
| Epoch log / 训练日志 | [training_log.csv](../../outputs/image/20261006_code_fixes_validation_v2/training_log.csv) |
| Confusion matrix / 混淆矩阵 | [confusion_matrix_validation.png](../../outputs/image/20261006_code_fixes_validation_v2/confusion_matrix_validation.png) |
| Independent checks / 独立检测 | [Image_Test_Results.md](../../docs/Image_Test_Results.md) |

## Manual review / 人工检查

Reviewer: **Zhihan Zhu**. These are the final 32 manually reviewed samples from the earlier run `20261001_clip_vitb32_seed42_zhihan_rtx3080`, with 8 samples each of false positive, false negative, true positive, and true negative. The table probabilities/categories belong to that earlier run, not to the newer reference above. Dataset labels, model predictions, and human judgments are separate columns; humor alone does not establish sarcasm.

检查人：**Zhihan Zhu**。最终人工表包含较早实验 `20261001_clip_vitb32_seed42_zhihan_rtx3080` 的 32 个样本，FP、FN、TP、TN 各 8 个。表中概率和分类对应该较早实验，不是上方的新 reference run。数据集标签、模型预测与人工判断分别记录；幽默不一定意味着讽刺。

- [Manual_Review_English.csv](manual_review/Manual_Review_English.csv)
- [Manual_Review_Chinese.csv](manual_review/Manual_Review_Chinese.csv)
- [False positive / 误报图集](images/false_positive_review.png)
- [False negative / 漏判图集](images/false_negative_review.png)
- [True positive / 正确有讽刺图集](images/true_positive_review.png)
- [True negative / 正确无讽刺图集](images/true_negative_review.png)

Match samples by `sample_id`. Each green square shows the approximate center crop for that individual meme; pixels outside it are removed. Remaining text can still be visible to the image encoder. Crop-related explanations in the manual table are hypotheses, not proven causes; the suggested follow-up checks were not all performed. [Processor evidence](images/processor_evidence.json) records the preprocessing details.

使用 `sample_id` 对照样本。每张梗图的绿框分别表示近似中心裁剪范围，框外像素被裁掉，框内文字仍可能被编码器看到。人工表中的裁剪原因属于待验证假设，不代表已证明的因果关系；建议的后续实验并非全部已做。[处理器记录](images/processor_evidence.json)说明预处理细节。

Historical full logs, Excel copies, duplicate grids, earlier results, and audit snapshots are preserved in the original team repository's `image-baseline` branch and the [external audit archive](../../docs/Image_Cloud_Assets.md). They are not needed for the quick review above.

完整历史日志、Excel 副本、重复图集、早期结果和审计快照保留在原团队仓库的 `image-baseline` 分支及[外部完整审计包](../../docs/Image_Cloud_Assets.md)中；快速审查只需本页资料。
