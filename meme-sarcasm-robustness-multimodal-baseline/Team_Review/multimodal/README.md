# Multimodal results and review / 多模态结果与核对

Validation split only (699 samples), threshold 0.5, model selection by validation Macro-F1, weighted cross-entropy with train-only class weights (w0 = 2.281, w1 = 0.640). Test was not scored. All multimodal training ran on CPU; the baseline's image features are the image group's CUDA cache.

仅 Validation（699 个样本）；阈值 0.5；按 Validation Macro-F1 选择模型；加权交叉熵，类别权重只来自 Train（w0 = 2.281，w1 = 0.640）。Test 未评分。所有多模态训练都在 CPU 上完成；基线的图片特征是图片组的 CUDA 缓存。

## October 14 comparison table / 10月14日比较表

| Model | Macro-F1 | Accuracy | F1 class 0 | F1 class 1 | Training time | Status |
|---|---:|---:|---:|---:|---:|---|
| Majority baseline (always 1) | 0.4349 | 0.7697 | 0.0000 | 0.8698 | N/A | done |
| DistilBERT text-only | 0.4903* | 0.6123* | 0.2409* | 0.7397* | 28.0 s* | Text Group pipeline not delivered; *frozen-feature probe below |
| CLIP image-only (image group, linear head) | 0.5276 | 0.6266 | 0.3113 | 0.7439 | 2.7 s (CUDA) | done (image group) |
| **DistilBERT + CLIP multimodal** (`20261016_distilbert_clip_seed42`) | **0.5336** | **0.6567** | **0.2941** | **0.7732** | 43.5 s (CPU, head only) | done |

*Text-only row: frozen DistilBERT `[CLS]` + the same MLP head (`configs/train/multimodal_text_only.yaml`). It does not replace the Text Group's DistilBERT classifier. Training time excludes feature extraction. On this 4-vCPU machine, DistilBERT took 83 s for all splits. CLIP took 13.5 min for all splits, but the train split ran while another job shared the CPU; validation and test alone ran at about 32 images/s.

*文字行：冻结 DistilBERT `[CLS]` + 同样的 MLP 分类头（`configs/train/multimodal_text_only.yaml`），不能代替文字组的 DistilBERT 分类器。训练时间不含特征提取。在这台 4 vCPU 机器上，DistilBERT 全部划分用时 83 秒；CLIP 全部划分用时 13.5 分钟，但 train 划分运行时有其他任务占用 CPU，validation 与 test 单独运行约每秒 32 张。

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

**Full statistics, figures and conclusions: [RESULTS_REPORT.md](RESULTS_REPORT.md).** That covers chance tests, bootstrap and paired tests, the modality diagnostic, linear probes, calibration and reproducibility.

**完整的统计、图片与结论见 [RESULTS_REPORT.md](RESULTS_REPORT.md)**，包括随机水平检验、bootstrap 与配对检验、模态诊断、线性探针、校准和可复现性。

1. The pipeline works end to end and is reproducible on one machine.
   - Checkpoint reload and same-input reruns are identical.
   - A same-agent rerun on CPU image features stays within the 0.01 tolerance (0.5435 vs 0.5336).
   - Raw-input inference through `MultimodalClassifier` (the serving predictor) matches the cached path to 6.4e-7 on CPU caches ([test report](../../docs/Multimodal_Test_Results.md)).
2. All frozen-feature models are at or barely above a label-independent classifier, which scores about 0.50 Macro-F1. The always-1 majority baseline (0.435) is a degenerate floor.
   - The baseline's label-permutation p is 0.015–0.048 across seeds, uncorrected for epoch selection.
   - Frozen DistilBERT text features carry no detectable signal: text-only 0.486, ROC-AUC 0.48 (3-seed mean), permutation p 0.68–0.87.
   - Image features carry a weak signal (ROC-AUC ≈ 0.54).
3. With this concat-MLP head, adding text shows no measurable gain: 0.536 ± 0.007 vs 0.546 ± 0.011 for CLIP alone. The paired 3-seed difference is −0.010 [−0.037, +0.017]. On 699 validation memes, differences below about 0.03–0.04 cannot be resolved.
4. The modality diagnostic is done (report §9). Shuffling captions between memes leaves Macro-F1 unchanged but flips 18% of predictions. Shuffling images drops Macro-F1 to 0.489. Replacing images with their mean makes the model predict "sarcastic" for 99.8% of memes.
5. The heads fit the training set almost perfectly (train ROC-AUC ≥ 0.999 at the selected epoch, same parameters, `eval()` mode) while validation ROC-AUC is 0.52–0.55, for fusion and image only alike: they largely memorize it. Validation loss is lowest at epoch 1 (≈ ln 2) and rises through over-confident errors, so it cannot guide selection; validation ranking and Macro-F1 do not decline. Of the smaller and regularized heads tried, only a linear head and a 10× lower learning rate reduce the gap; they lower validation Macro-F1 and leave fusion below image only (fusion higher in 1 of 24 seed pairs overall). Within this scope, reducing overfitting does not make fusion gain ([overfitting/README.md](overfitting/README.md)). The selected epoch overstates Macro-F1 by about 0.015–0.02.
6. Mean pooling is not better than [CLS], so [CLS] stays the baseline. [CLS] is the same token position HF's `DistilBertForSequenceClassification` uses, but here the encoder is frozen, not fine-tuned.

1. 流程端到端可运行，并且在同一台机器上可复现：
   - checkpoint 重载与相同输入重跑完全一致；
   - 同一 agent 在 CPU 图片特征上的重跑落在 0.01 容差内（0.5435 对 0.5336）；
   - 在 CPU 缓存上，原始输入经 `MultimodalClassifier` 的推理与缓存路径相差 6.4e-7（[检测报告](../../docs/Multimodal_Test_Results.md)）。
2. 所有冻结特征模型都处于或仅略高于不看标签的分类器（Macro-F1 约 0.50）；"总是预测 1"的多数类基线（0.435）是退化的下限。
   - 基线的标签置换 p 为 0.015–0.048（未校正 epoch 选择）。
   - 冻结 DistilBERT 文字特征没有可检测的信号：纯文字 0.486，ROC-AUC 0.48（3 个 seed 均值），置换 p 0.68–0.87。
   - 图像特征有微弱信号（ROC-AUC ≈ 0.54）。
3. 在拼接 + MLP 分类头下，加入文字没有可测量的增益：0.536 ± 0.007，只用 CLIP 为 0.546 ± 0.011；3 个 seed 的配对差为 −0.010 [−0.037, +0.017]。在 699 张 Validation 梗图上，约 0.03–0.04 以下的差异无法分辨。
4. 模态诊断已完成（报告 §9）：在梗图之间打乱文字，Macro-F1 不变，但 18% 的预测改变；打乱图片则降到 0.489；把图片换成平均向量后，模型对 99.8% 的梗图预测"讽刺"。
5. 分类头几乎完美拟合训练集（所选 epoch 训练 ROC-AUC ≥ 0.999，同一组参数、`eval()` 模式），而验证 ROC-AUC 为 0.52–0.55，融合与纯图像相同：即在很大程度上记住了训练集。验证 loss 在第 1 个 epoch 最低（≈ ln 2），之后因高置信错误上升，因此不能用来选择；验证排序能力和 Macro-F1 并未下降。尝试过的更小、正则更强的分类头中，只有线性头和学习率降为 1/10 能缩小差距；它们降低了验证 Macro-F1，融合仍低于纯图像（全部 24 对 seed 中只有 1 对融合更高）。在这个检验范围内，减少过拟合并不能让融合获益（[overfitting/README.md](overfitting/README.md)）。所选 epoch 使 Macro-F1 偏高约 0.015–0.02。
6. mean pooling 不比 [CLS] 好，因此保留 [CLS] 作为基线。[CLS] 与 HF `DistilBertForSequenceClassification` 使用的 token 位置相同，但这里编码器是冻结的，没有微调。

This answers the proposal's question for the frozen baseline: its "multimodal" score is carried by the image features. The remaining robustness work after October 14 is OCR-noise perturbations, plus repeating the diagnostic on new models (for example, a fine-tuned text encoder).

这回答了开题报告针对冻结基线的问题：其"多模态"分数由图像特征支撑。10月14日之后剩下的鲁棒性工作是 OCR 噪声扰动，以及在新模型（如微调后的文字编码器）上重复该诊断。

## Manual review / 人工核对

[review_grid.png](../../outputs/multimodal/20261016_distilbert_clip_seed42/review/review_grid.png) and [manual_review.csv](../../outputs/multimodal/20261016_distilbert_clip_seed42/review/manual_review.csv): 4 TP, 4 TN, 4 FP and 4 FN validation samples, selected with seed 42.

- In every sample the image shows the caption stored for its `sample_id`, and the label equals `sarcasm_binary` in the metadata. The image-text-label triples are aligned.
- 4 of the 16 labels are debatable: satire, irony or political mockery labelled `not_sarcastic` (`validation_000524`, `000508`, `000544`), and absurd humour labelled `twisted_meaning` (`000300`). Label noise of this kind may limit class-0 F1.
- Captions include OCR errors and site watermarks (`makeameme.org`, `quickmeme.com`, `memecrunch.com`), which reach the text encoder unchanged.
- `validation_000242` is offensive (blackface image).
- The first pass was done by Claude, as noted in the CSV. A teammate should confirm it and fill in their name.

- 每个样本的图片都显示其 `sample_id` 对应的文字，标签与元数据 `sarcasm_binary` 一致，图片—文字—标签对齐无误。
- 16 个标签中 4 个存疑：讽刺新闻、反语或政治嘲讽被标为 `not_sarcastic`（`validation_000524`、`000508`、`000544`），荒诞幽默被标为 `twisted_meaning`（`000300`）。这类标签噪声可能限制 class 0 的 F1。
- 文字包含 OCR 错误和网站水印（`makeameme.org`、`quickmeme.com`、`memecrunch.com`），会原样进入文字编码器。
- `validation_000242` 含冒犯性内容（黑脸图片）。
- 初步核对由 Claude 完成（CSV 中已注明），需要组员确认并填写姓名。

## Known problems and unfinished tasks / 已知问题与未完成事项

- A teammate still has to do the human second-member reproduction. The rerun here was done by the same agent on re-extracted (bit-identical) text features and CPU image features.
- The encoders here were local copies of the pinned weights, because Hugging Face was blocked. On a machine with Hub access, re-running `python -m src.features.feature_engineering text` and comparing with `scripts/compare_feature_caches.py` will confirm the DistilBERT copy (expected max diff ≈ 0).
- Test is not scored. For the final evaluation use `data/processed/features/clip_vitb32_pooler_cpu`, because the image group cache has no test split.
- The image group's CUDA cache uses TF32 (see the test report). The 0.01 Macro-F1 tolerance absorbs it, but probability-level checks against that cache are looser than float precision.
- Validation is small and noisy. Report the 3-seed mean plus paired bootstrap and label-permutation tests for comparisons. Validation loss is lowest at epoch 1 in every image-based run, so it cannot be used for early stopping. A lower learning rate or stronger regularization, with a fixed epoch budget or a smoothed Macro-F1 criterion, is post-Oct 14 work (a team decision if it replaces D10).
- Checkpoints and caches are not in Git. They are in `Multimodal_Assets_20261016.zip` and `Text_Features_DistilBERT_20261016.zip` ([asset list](../../docs/Multimodal_Assets.md)).

- 组员仍需完成人工的"第二名成员复现"；本次重跑由同一 agent 使用重新提取（逐位相同）的文字特征和 CPU 图片特征完成。
- 由于无法访问 Hugging Face，本次编码器使用固定权重的本地副本；在能访问 Hub 的机器上重新运行 `python -m src.features.feature_engineering text` 并用 `scripts/compare_feature_caches.py` 比较，即可确认 DistilBERT 副本（预期最大差异 ≈ 0）。
- Test 未评分；最终评估需使用 `data/processed/features/clip_vitb32_pooler_cpu`，因为图片组缓存不含 test。
- 图片组 CUDA 缓存使用了 TF32（见检测报告）；0.01 的 Macro-F1 容差可以吸收，但针对该缓存的概率级检查达不到浮点精度。
- Validation 规模小、噪声大，比较时请用 3 个 seed 的均值，并配合配对 bootstrap 与标签置换检验。所有使用图像的实验中 validation loss 都在第 1 个 epoch 最低，不能用来早停。降低学习率或加强正则化，并改用固定 epoch 数或平滑后的 Macro-F1 准则，属于10月14日之后的工作（若替代 D10，需团队决定）。
- Checkpoint 和缓存不在 Git 中，见 `Multimodal_Assets_20261016.zip` 和 `Text_Features_DistilBERT_20261016.zip`（[资料清单](../../docs/Multimodal_Assets.md)）。
