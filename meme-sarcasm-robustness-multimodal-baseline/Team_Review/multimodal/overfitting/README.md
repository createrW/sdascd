# Overfitting diagnostic / 过拟合诊断 (validation only)

Question: the fusion head fits the training set ever better while validation loss rises. Is that
overfitting, and is it why text + image does not beat image alone? This folder separates what was
**observed**, what was **tested**, and what is still **inference**. Test was not read.

问题：融合分类头越来越好地拟合训练集，而验证 loss 上升。这是否是过拟合？它是否是"文字 + 图像不优于纯图像"的原因？本文件把**观察到的**、**检验过的**和**仍属推断的**分开写。未读取 Test。

```
python -m src.evaluation.overfitting_diagnostic     # ~30 min on 4 CPUs; writes this folder
```

## Method / 方法

* **Same parameters, same definition, both splits.** At the end of every epoch the head is put in
  `eval()` mode and scored on the whole train split and the whole validation split with one loss
  definition: PyTorch weighted cross-entropy, `sum_i w(y_i) * l_i / sum_i w(y_i)`, train-only class
  weights (w0 2.281, w1 0.640). Also recorded: unweighted NLL, Macro-F1 at 0.5, ROC-AUC,
  PR-AUC (class 0), ECE (10 bins), errors and errors made with confidence ≥ 0.9.
* **The three train-loss numbers are kept apart:** the logged training loss (dropout on, batch means
  averaged by batch size while the weights move), the full-train loss with dropout on at the
  end-of-epoch weights, and the full-train loss in `eval()` mode.
* **Reruns are exact.** The 9 recorded runs (fusion, image only, text only × seeds 42, 43, 44) were
  retrained with this scoring hooked in. Every `training_log.csv` is byte-identical and every
  selected `best.pt` bit-identical to the recorded one ([reproduction_check.csv](reproduction_check.csv)),
  so the scoring did not change training and the numbers describe the shipped heads.
* **Sweep.** Same caches, seeds 42–44, same selection rule (best validation Macro-F1, D10), fusion
  and image only: hidden 512 (baseline), 256, 128, 64, linear; hidden 512 with dropout 0.5,
  weight decay 1e-2, or learning rate 1e-4. 42 new runs; heads in
  `checkpoints/multimodal/20261018_overfit_*` (not in Git).

## 1. Observed / 观察到的

![Seed 42](fig_o1_seed42_unified_eval.png)
![Recorded runs](fig_o2_recorded_runs.png)

Seed 42 baseline (fusion), every row from the same parameters:

| Epoch | logged train loss | train, dropout on | train, eval() | validation, eval() | train Macro-F1 / AUC | validation Macro-F1 / AUC | validation errors (conf. ≥ 0.9) |
|---:|---:|---:|---:|---:|---|---|---|
| 1 | 0.6931 | 0.6883 | 0.6882 | 0.6945 | 0.568 / 0.612 | 0.523 / 0.500 | 228 (0) |
| 10 | 0.5523 | 0.5128 | 0.5060 | 0.7422 | 0.666 / 0.878 | 0.486 / 0.542 | 332 (3) |
| **25 (selected)** | 0.0923 | 0.0773 | 0.0577 | 1.4294 | 0.989 / 0.9997 | 0.534 / 0.549 | 240 (92) |
| 30 | 0.0614 | 0.0489 | 0.0311 | 1.7443 | 0.996 / 0.9994 | 0.512 / 0.534 | 230 (113) |

1. **The heads memorize the training set.** At the selected epoch, train Macro-F1 is 0.978–0.995 and
   train ROC-AUC ≥ 0.999 for the fusion head (seeds 42–44), against validation Macro-F1 0.53–0.54 and
   ROC-AUC 0.53–0.55. The train−validation ROC-AUC gap is 0.45–0.47. **Image only behaves the
   same** (train AUC ≥ 0.999, validation 0.52–0.55, gap 0.45–0.47): this is not specific to fusion.
2. **Validation loss gets worse through confident mistakes, not through worse ranking.** Validation
   weighted CE is lowest at epoch 1 in all six fusion and image-only runs (0.695–0.698 ≈ ln 2, the
   loss of a uniform prediction) and reaches 1.16–1.64 (fusion) and 1.73–2.37 (image only) at the
   selected epoch. Over the same epochs validation ROC-AUC does not fall (≈ 0.50 at epoch 1,
   0.52–0.55 at the selected epoch) and the number of errors does not grow (seed 42: 228 → 240), but the errors made
   with ≥ 90 % confidence go from 0 to 92 (113 at epoch 30). ECE at the selected epoch is 0.15–0.21.
3. **"Training longer makes classification worse" is not what the data show.** Validation Macro-F1
   peaks late (selected epochs 19–29) and fluctuates; what deteriorates monotonically is probability
   quality (loss, calibration).
4. **The logged losses use different definitions on the two sides.** The logged validation loss *is*
   the full-split `eval()` weighted CE (1.4294 both ways), but the logged training loss is higher than
   the `eval()` train loss (0.0923 vs 0.0577 at epoch 25) because of dropout (0.0773 with dropout on)
   and because it averages batch means while the weights change. Comparing "0.06 vs 1.74" therefore
   overstates the gap's precision; on one definition it is 0.031 vs 1.744 at epoch 30.
5. **Text only fits slowly and does not transfer.** Train ROC-AUC reaches only 0.61–0.63 at its
   selected epoch (0.73–0.78 at epoch 30) while validation ROC-AUC falls to 0.43–0.45 by epoch 30.

## 2. Tested: less capacity / more regularization / 检验：减小容量、加强正则

![Sweep](fig_o3_capacity_sweep.png)

Mean of seeds 42–44 ([fusion_vs_image.csv](fusion_vs_image.csv), per run: [run_summary.csv](run_summary.csv)):

| Head | train−val AUC gap, fusion / image | val weighted CE at selected epoch, fusion / image | val Macro-F1 (selected), fusion / image | val Macro-F1 (last 10 epochs), fusion / image | val AUC (last 10), fusion / image |
|---|---|---|---|---|---|
| MLP 512 (baseline) | 0.46 / 0.46 | 1.41 / 2.11 | 0.536 / 0.546 | 0.519 / 0.532 | 0.531 / 0.535 |
| MLP 256 | 0.47 / 0.45 | 1.23 / 1.60 | 0.529 / 0.553 | 0.515 / 0.539 | 0.526 / 0.538 |
| MLP 128 | 0.46 / 0.43 | 1.19 / 1.31 | 0.523 / 0.555 | 0.512 / 0.540 | 0.536 / 0.551 |
| MLP 64 | 0.40 / 0.42 | 0.95 / 1.12 | 0.521 / 0.542 | 0.507 / 0.529 | 0.535 / 0.540 |
| linear | **0.20 / 0.17** | **0.71 / 0.71** | 0.517 / 0.525 | 0.502 / 0.512 | 0.528 / 0.536 |
| MLP 512, dropout 0.5 | 0.41 / 0.46 | 0.97 / 1.64 | 0.528 / 0.554 | 0.511 / 0.537 | 0.527 / 0.539 |
| MLP 512, weight decay 1e-2 | 0.35 / 0.45 | 1.25 / 1.82 | 0.528 / 0.566 | 0.513 / 0.539 | 0.533 / 0.545 |
| MLP 512, lr 1e-4 | 0.33 / 0.31 | 0.76 / 0.74 | 0.525 / 0.531 | 0.512 / 0.516 | 0.538 / 0.546 |

6. **Overfitting can be reduced.** A linear head cuts the train−validation AUC gap from 0.46 to 0.20
   (fusion) and keeps validation loss near ln 2; a 10× lower learning rate gives 0.33.
7. **Reducing it does not improve validation.** No variant raises the fusion head's validation
   Macro-F1 or ROC-AUC beyond run-to-run noise. The sweep's 21 fusion runs reach Macro-F1 0.515–0.537
   at their selected epoch (baseline seeds: 0.532–0.544) and ROC-AUC 0.517–0.541 over their last 10
   epochs (baseline: 0.526–0.539); the variant means stay at or below the baseline.
8. **Fusion never overtakes image only.** In all 8 variants the 3-seed mean of fusion is below image
   only on every validation measure (selected Macro-F1 −0.006 to −0.038, last-10 Macro-F1 −0.003 to
   −0.028, last-10 ROC-AUC −0.004 to −0.015). Seed by seed, fusion is higher in 1 of 24 pairs
   (selected and last-10 Macro-F1) and 2 of 24 (last-10 ROC-AUC). The pairs share one validation set,
   so this is a description, not an independent test.

## 3. What follows / 结论分级

| Level | Statement |
|---|---|
| **Observed (strong)** | The fusion and image-only heads memorize the training set; late in training, validation loss rises through over-confident errors while validation ranking stays flat and Macro-F1 does not decline. 观察到（强）：两类分类头都记住了训练集；训练后期验证 loss 因高置信错误上升，而验证排序能力持平、Macro-F1 没有下降。 |
| **Tested, not supported** | "Overfitting is the main reason fusion shows no gain." Every tested way of reducing overfitting left fusion at or below image only, and none improved validation. 已检验、不支持："过拟合是融合没有提升的主要原因"——所有减轻过拟合的设置下，融合都不高于纯图像，验证指标也没有改善。 |
| **Consistent, not proven** | Fusion shows no gain because the frozen DistilBERT `[CLS]` features carry no signal that transfers to validation (text-only ROC-AUC ≤ 0.50; linear probes ≤ 0.50, RESULTS_REPORT §7); the head can only fit those 768 dimensions to training noise, which would also explain why fusion sits slightly below image only in every variant. 一致但未证实：冻结 DistilBERT `[CLS]` 特征没有能迁移到验证集的信号；分类头只能把这 768 维拟合成训练噪声，这也能解释为什么各设置下融合都略低于纯图像。 |
| **Open** | Whether the small fusion < image-only deficit is real (it is below the ≈ 0.03–0.04 Macro-F1 resolution of 699 memes); whether other regularizers (modality dropout, feature noise, early stopping on held-out training data), another text encoder or fine-tuning change the picture; label noise. 未解决：融合略低于纯图像是否真实（小于 699 张验证集约 0.03–0.04 的分辨率）；其他正则方法、其他文字编码器或微调是否改变结论；标签噪声。 |

**Limits.** Validation only; 16 configurations were compared on the same 699 memes, so the best
values of the sweep (for example image only with weight decay 1e-2, 0.566) are selection-optimistic
and were not adopted. Three seeds per configuration. The sweep is not exhaustive. Test untouched.

**局限**：只用 Validation；16 个配置在同一批 699 张梗图上比较，因此扫描中的最好值（例如纯图像 + weight decay 1e-2 的 0.566）带有选择偏乐观，未被采用。每个配置 3 个 seed；扫描并不完备；未使用 Test。

## Files / 文件

| File | Content |
|---|---|
| `per_epoch.csv` | every run × epoch: the three train losses, and for train and validation: weighted CE, NLL, Macro-F1, accuracy, ROC-AUC, PR-AUC (class 0), ECE, mean confidence, errors, confident errors |
| `run_summary.csv` | one row per run: selected epoch, best and last-10 validation metrics, train metrics and gaps at the selected epoch |
| `fusion_vs_image.csv` | per head variant: 3-seed mean/sd for fusion and image only, paired differences |
| `reproduction_check.csv` | the 9 recorded runs reproduced byte for byte |
| `fig_o1_*.png`, `fig_o2_*.png`, `fig_o3_*.png` | figures above |
