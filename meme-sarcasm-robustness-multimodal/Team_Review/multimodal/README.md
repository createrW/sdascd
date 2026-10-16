# Multimodal results and review / 多模态结果与核对

Validation split only (699 samples), threshold 0.5, model selection by validation Macro-F1, weighted cross-entropy with train-only class weights (w0 = 2.281, w1 = 0.640). Test was not scored. All multimodal numbers were produced on CPU.

仅 Validation（699 个样本）；阈值 0.5；按 Validation Macro-F1 选择模型；加权交叉熵，类别权重只来自 Train（w0 = 2.281，w1 = 0.640）。Test 未评分。所有多模态结果都在 CPU 上得到。

## October 14 comparison table / 10月14日比较表

| Model | Macro-F1 | Accuracy | F1 class 0 | F1 class 1 | Training time | Status |
|---|---:|---:|---:|---:|---:|---|
| Majority baseline (always 1) | 0.4349 | 0.7697 | 0.0000 | 0.8698 | N/A | done |
| DistilBERT text-only | 0.4903* | 0.6123* | 0.2409* | 0.7397* | 28.0 s* | Text Group pipeline not delivered; *frozen-feature probe below |
| CLIP image-only (image group, linear head) | 0.5276 | 0.6266 | 0.3113 | 0.7439 | 2.7 s (CUDA) | done (image group) |
| **DistilBERT + CLIP multimodal** (`20261016_distilbert_clip_seed42`) | **0.5336** | **0.6567** | **0.2941** | **0.7732** | 43.5 s (CPU, head only) | done |

*Text-only row: frozen DistilBERT `[CLS]` + the same MLP head (`configs/ablations/text_only.json`). It does not replace the Text Group's DistilBERT classifier. Training time excludes feature extraction (DistilBERT 83 s for all splits; CLIP ~3 min per 5.6k images on CPU).

*文字行：冻结 DistilBERT `[CLS]` + 同样的 MLP 分类头（`configs/ablations/text_only.json`），不能代替文字组的 DistilBERT 分类器。训练时间不含特征提取（DistilBERT 全部划分 83 秒；CLIP 在 CPU 上每 5.6k 张约 3 分钟）。

Main run / 主实验: confusion matrix `[[50, 111], [129, 409]]` (rows true 0/1, columns predicted 0/1), best epoch 25/30, validation loss 1.429. Files: [outputs/multimodal/20261016_distilbert_clip_seed42/](../../outputs/multimodal/20261016_distilbert_clip_seed42/).

## Ablations and seeds (supplementary) / 对照实验与多 seed（补充）

Same settings (MLP 512, dropout 0.2, AdamW 1e-3, batch 16, 30 epochs), image group cache for images. Per-run rows: [validation_runs.csv](validation_runs.csv).

设置相同（MLP 512、dropout 0.2、AdamW 1e-3、batch 16、30 epochs），图片使用图片组缓存。逐次结果见 [validation_runs.csv](validation_runs.csv)。

| Model / 模型 | Seeds 42, 43, 44 Macro-F1 | Mean ± std | Accuracy mean |
|---|---|---:|---:|
| DistilBERT [CLS] + CLIP (baseline) | 0.5336, 0.5315, 0.5441 | 0.5364 ± 0.0067 | 0.6819 |
| DistilBERT mean pooling + CLIP | 0.5311, 0.5247, 0.5282 | 0.5280 ± 0.0032 | 0.6891 |
| CLIP only, same MLP | 0.5568, 0.5352, 0.5457 | 0.5459 ± 0.0108 | 0.7144 |
| DistilBERT [CLS] only, same MLP | 0.4903, 0.4771, 0.4912 | 0.4862 ± 0.0079 | 0.6052 |

## What the numbers say / 结论

1. The pipeline works end to end and is reproducible: checkpoint reload and same-input reruns are identical; an independent rerun with re-extracted features stays within the 0.01 tolerance; raw-input inference through `MultimodalClassifier` matches the cached path to 6.4e-7 ([test report](../../docs/Multimodal_Test_Results.md)).
2. Frozen DistilBERT text features carry little sarcasm signal here: text-only is 0.486, close to the majority baseline 0.435. The visual features carry more.
3. Adding text to the image features does not help yet: the multimodal model (0.536 ± 0.007) is not better than the same MLP on CLIP alone (0.546 ± 0.011). With a 699-sample validation set and per-epoch Macro-F1 swinging between 0.42 and 0.53, differences of about 0.01 are within noise.
4. The MLP overfits: train loss reaches 0.06 while validation loss rises from 0.69 to 1.7. The best epoch is the maximum of a noisy curve, so single-seed validation numbers are optimistic.
5. Mean pooling is not better than [CLS], so the paper/HF-aligned [CLS] vector stays the baseline.

1. 流程端到端可运行且可复现：checkpoint 重载与相同输入重跑完全一致；重新提取特征的独立复现落在 0.01 容差内；原始输入经 `MultimodalClassifier` 的推理与缓存路径相差 6.4e-7（[检测报告](../../docs/Multimodal_Test_Results.md)）。
2. 冻结的 DistilBERT 文字特征在此几乎不含讽刺信号：纯文字 0.486，接近多数类基线 0.435；视觉特征信息更多。
3. 目前加入文字没有带来提升：多模态（0.536 ± 0.007）不优于同一 MLP 只用 CLIP（0.546 ± 0.011）。Validation 只有 699 个样本，逐 epoch 的 Macro-F1 在 0.42–0.53 间波动，约 0.01 的差异属于噪声范围。
4. MLP 过拟合：train loss 降到 0.06，validation loss 从 0.69 升到 1.7；最佳 epoch 是噪声曲线的最大值，单 seed 的 Validation 结果偏乐观。
5. mean pooling 不比 [CLS] 好，因此保留与论文/HF 一致的 [CLS] 作为基线。

This matches the proposal's question: a frozen-feature "multimodal" model can look competitive without using the text. The robustness tests after October 14 (image removal or substitution, OCR noise) will show which modality the model relies on.

这正对应开题报告的问题：冻结特征的"多模态"模型可能在不真正利用文字的情况下得到看似不错的分数；10月14日之后的鲁棒性测试（去除/替换图片、OCR 噪声）将揭示模型依赖哪一种模态。

## Manual review / 人工核对

[review_grid.png](../../outputs/multimodal/20261016_distilbert_clip_seed42/review/review_grid.png) and [manual_review.csv](../../outputs/multimodal/20261016_distilbert_clip_seed42/review/manual_review.csv): 4 TP, 4 TN, 4 FP and 4 FN validation samples, selected with seed 42.

- In every sample the image shows the caption stored for its `sample_id`, and the label equals `sarcasm_binary` in the metadata. The image-text-label triples are aligned.
- 4 of the 16 labels are debatable: satire or political mockery labelled `not_sarcastic` (`validation_000524`, `000508`, `000544`), and absurd humour labelled `twisted_meaning` (`000300`). Label noise in this direction caps class-0 F1.
- Captions include OCR errors and site watermarks (`makeameme.org`, `quickmeme.com`, `memecrunch.com`), which reach the text encoder unchanged.
- `validation_000242` is offensive (blackface image).
- The first pass was done by Claude, as noted in the CSV. A teammate should confirm it and fill in their name.

- 每个样本的图片都显示其 `sample_id` 对应的文字，标签与元数据 `sarcasm_binary` 一致，图片—文字—标签对齐无误。
- 16 个标签中 4 个存疑：讽刺新闻/政治嘲讽被标为 `not_sarcastic`（`validation_000524`、`000508`、`000544`），荒诞幽默被标为 `twisted_meaning`（`000300`）。这类标签噪声限制了 class 0 的 F1。
- 文字包含 OCR 错误和网站水印（`makeameme.org`、`quickmeme.com`、`memecrunch.com`），会原样进入文字编码器。
- `validation_000242` 含冒犯性内容（黑脸图片）。
- 初步核对由 Claude 完成（CSV 中已注明），需要组员确认并填写姓名。

## Known problems and unfinished tasks / 已知问题与未完成事项

- A teammate still has to do the human second-member reproduction. The reproduction here was a separate run on fresh text and CPU image caches, by the same agent.
- The encoders here were local copies of the pinned weights, because Hugging Face was blocked. On a machine with Hub access, re-running `python -m src.text.extract_features` and comparing with `scripts/compare_feature_caches.py` will confirm the DistilBERT copy (expected max diff ≈ 0).
- Test is not scored. For the final evaluation use `features/clip_vitb32_pooler_cpu`, because the image group cache has no test split.
- The image group's CUDA cache uses TF32 (see the test report). The 0.01 Macro-F1 tolerance absorbs it, but probability-level checks against that cache are looser than float precision.
- Validation is small and noisy. Report the 3-seed mean for comparisons. Early stopping on validation loss or a lower learning rate would reduce overfitting; leave that for the post-Oct 14 work.
- Checkpoints and caches are not in Git. They are in `Multimodal_Assets_20261016.zip` and `Text_Features_DistilBERT_20261016.zip` ([asset list](../../docs/Multimodal_Assets.md)).

- 组员仍需完成人工的"第二名成员复现"；本次复现由同一 agent 用新的文字缓存和 CPU 图片缓存单独运行。
- 由于无法访问 Hugging Face，本次编码器使用固定权重的本地副本；在能访问 Hub 的机器上重新运行 `python -m src.text.extract_features` 并用 `scripts/compare_feature_caches.py` 比较，即可确认 DistilBERT 副本（预期最大差异 ≈ 0）。
- Test 未评分；最终评估需使用 `features/clip_vitb32_pooler_cpu`，因为图片组缓存不含 test。
- 图片组 CUDA 缓存使用了 TF32（见检测报告）；0.01 的 Macro-F1 容差可以吸收，但针对该缓存的概率级检查达不到浮点精度。
- Validation 规模小、噪声大，比较时请用 3 个 seed 的均值；按 validation loss 早停或降低学习率可以缓解过拟合，留待10月14日之后的工作。
- Checkpoint 和缓存不在 Git 中，见 `Multimodal_Assets_20261016.zip` 和 `Text_Features_DistilBERT_20261016.zip`（[资料清单](../../docs/Multimodal_Assets.md)）。
