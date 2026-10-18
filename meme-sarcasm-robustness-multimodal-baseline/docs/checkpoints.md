# Checkpoints 说明 / Checkpoint guide

本仓库全部 **59 个训练好的分类头**的说明：图片组 3 个、多模态 14 个、过拟合诊断 42 个。编码器（DistilBERT、CLIP）是冻结的公开权重，**不在** checkpoint 里；checkpoint 只包含在缓存特征上训练的小分类头。逐个文件的元数据见 [checkpoints_manifest.csv](checkpoints_manifest.csv)，校验和见 [checkpoints_SHA256SUMS.txt](checkpoints_SHA256SUMS.txt)。

All 59 trained heads of this repository (image group 3, multimodal 14, overfitting diagnostic 42). The frozen encoders are not included. Per-file metadata: [checkpoints_manifest.csv](checkpoints_manifest.csv); checksums: [checkpoints_SHA256SUMS.txt](checkpoints_SHA256SUMS.txt).

所有 checkpoint 都**只按 Validation 选择**（30 个 epoch 中验证 Macro-F1 最高的一个，阈值 0.5），**Test 从未评分**。

## 1. 获取与放置 / Where to get them

团队 README 9.4 规定 checkpoint 不提交到 GitHub，因此除图片组 3 个 8.6 KB 的线性头（已在 Git 中）外，其余以压缩包分发。两个 checkpoint 压缩包都**在仓库根目录用 `unzip -n` 解压**（不覆盖已有文件），只写入 `checkpoints/`，文件会落到代码默认读取的位置。特征缓存与实验记录压缩包、获取方式与旧压缩包见 [Multimodal_Assets.md](Multimodal_Assets.md)。

Checkpoints are not in Git (team README 9.4), except the three 8.6 KB image-group heads. Unzip both checkpoint archives **at the repository root with `unzip -n`**; they write only into `checkpoints/`. Feature caches, run folders, where to get the archives and the earlier archives: [Multimodal_Assets.md](Multimodal_Assets.md).

| 压缩包 / Archive | 字节 / Bytes | 内容 | SHA-256 |
|---|---:|---|---|
| `Checkpoints_merged_20261018_v2.zip` | 32357210 | `checkpoints/image/`（3）+ `checkpoints/multimodal/20261016_*`（14） | `99687a70b7e2431378ab2b98fd86f7924b80b38520e168e1d379354cae61ce8b` |
| `Checkpoints_overfitting_diagnostic_20261018_v2.zip` | 51308414 | `checkpoints/multimodal/20261018_overfit_*`（42，只用于分析） | `c0330822ffe496e293084180fe7e0118e30098ad090f28ba355352153afcdc2a` |

每个压缩包在 `checkpoints/` 下附带自己的 `README_<bundle>.md`、`MANIFEST_<bundle>.csv` 与 `SHA256SUMS_<bundle>.txt`（只列本包的文件）；本目录的 [checkpoints_manifest.csv](checkpoints_manifest.csv) 与 [checkpoints_SHA256SUMS.txt](checkpoints_SHA256SUMS.txt) 列出全部 59 个。

**已被取代 / Superseded:** 不带 `_v2` 的 `Checkpoints_*_20261018.zip` 把 README.md 放在压缩包根目录，会与仓库的 README.md 冲突；请改用 `_v2`（checkpoint 文件逐字节相同）。所有旧压缩包的对应关系见 [Multimodal_Assets.md](Multimodal_Assets.md#earlier-archives--旧压缩包)。 The archives without `_v2` put README.md at the archive root, where it collides with the repository README; the checkpoint files are byte-identical.

```text
checkpoints/image/<run_id>/best.pt        # 图片组线性头
checkpoints/multimodal/<run_id>/best.pt   # 多模态、对照与诊断分类头
data/processed/features/<cache>/          # 特征缓存
outputs/<task>/<run_id>/                  # 实验记录
```

## 2. 推荐使用 / Which one to use

| 用途 | checkpoint | 验证 Macro-F1 |
|---|---|---:|
| 多模态模型：评估、服务、演示 | ★ `checkpoints/multimodal/20261016_distilbert_clip_seed42/best.pt` | 0.5336（seed 42，epoch 25） |
| 图片模型参考 | ★ `checkpoints/image/20261006_code_fixes_validation_v2/best.pt` | 0.5276（epoch 7） |
| 服务输出需要与记录的验证概率一致时 | `checkpoints/multimodal/20261016_distilbert_clip_seed42_reproduction_cpu/best.pt` | 0.5435（epoch 29） |

参考水平：多数类基线 0.4349，不看标签的分类器约 0.50。多模态基线 3 个 seed 为 0.5364 ± 0.0067，纯图像 0.5459 ± 0.0108，融合没有可测量的增益（[结果报告](../Team_Review/multimodal/RESULTS_REPORT.md)）。

主实验分类头在图片组的 CUDA（TF32）CLIP 缓存上训练；服务时在 CPU 上由原始图片计算特征，概率与记录值最多相差 0.013（决定 D19、D24）。需要服务输出与记录值一致时用 `_reproduction_cpu`（在 CPU 缓存上训练，端到端最大差异 6.4e-7）。

## 3. 文件格式 / Format

每个 `best.pt` 是一个 dict，用 `torch.load(path, map_location="cpu", weights_only=True)` 读取：

| 键 | 含义 |
|---|---|
| `state_dict` | 分类头权重 |
| `config` | 训练配置（编码器 id 与 revision、`text_feature`、`text_max_length`、分类头、学习率、seed、阈值 0.5 等） |
| `epoch` | 被选中的 epoch（验证 Macro-F1 最高） |
| `val_macro_f1` | 该 epoch 的验证 Macro-F1（在训练所用的特征上） |
| `class_weights` | 只用训练集计算的类别权重 `[2.281, 0.640]` |
| `modalities`, `text_features_dir`, `image_features_dir` | 仅多模态分类头：使用的模态；2026-10-16 的分类头还记录了训练时的特征目录 |

| 分类头类型 | 结构 | `state_dict` 键 | 参数量 |
|---|---|---|---:|
| 图片组线性头（`ImageHead`） | Dropout → Linear 768→2 | `net.1.weight`, `net.1.bias` | 1,538 |
| 多模态基线（`FusionHead`） | L2(文字) ‖ L2(图像) → Linear 1536→512 → ReLU → Dropout 0.2 → Linear 512→2 | `classifier.0.*`, `classifier.3.*` | 787,970 |
| 单模态对照（`FusionHead`，一种模态） | L2 → Linear 768→512 → ReLU → Dropout 0.2 → Linear 512→2 | `classifier.0.*`, `classifier.3.*` | 394,754 |
| 诊断中的线性头（`FusionHead`，hidden 0） | L2 → Dropout → Linear →2 | `classifier.1.*` | 3,074（融合）/ 1,538（图像） |

多模态分类头的 `classifier.*` 权重可直接装入端到端模型 `MultimodalClassifier`（`load_fusion_head`），服务代码就是这样加载的。

**共同训练设置**：冻结 DistilBERT `distilbert/distilbert-base-uncased@1c4513b2eedbda136f57676a34eea67aba266e5c`（`[CLS]` = `last_hidden_state[:, 0]`，最长 128 token）与 CLIP `openai/clip-vit-base-patch32@3d74acf`（vision `pooler_output`，768 维）；加权交叉熵（只用训练集算权重）；AdamW lr 1e-3、weight decay 1e-4；多模态 batch 16、图片组 batch 64；30 个 epoch；CPU 上训练可逐位复现。

## 4. 全部清单 / Full list

`val_macro_f1` 是 checkpoint 里记录的值。表中 SHA-256 只列前 12 位，完整值见 [checkpoints_SHA256SUMS.txt](checkpoints_SHA256SUMS.txt)。

### 4.1 图片组（3 个）

3 个线性头使用相同的冻结 CLIP 特征。在最终缓存 `20261006_code_fixes_validation_v2` 上重放：原始基线 0.529162，Zhihan 复现 0.527606，最终参考 0.527606（[Image_Test_Results.md](Image_Test_Results.md)）。**注意**：图片组仓库中 `20261001_clip_vitb32_seed42_zhihan_rtx3080/best.pt` 与 `20261006_code_fixes_validation_v2/best.pt` 是逐字节相同的同一个文件（SHA-256 相同），这里按原样保留两个路径。两个 `20261001_*` 分类头没有实验目录（`outputs/image/20261001_*` 不存在），它们也没有记录特征目录，用 `evaluate.py` 重放时需同时指定特征缓存和参考指标文件：`python -m src.evaluation.evaluate --run_id 20261001_clip_vitb32_seed42 --features_dir data/processed/features/20261006_code_fixes_validation_v2 --reference_metrics outputs/image/20261006_code_fixes_validation_v2/metrics_validation.json --out_dir outputs/image/replay_20261001`（Zhihan 的分类头同理，另换 `--out_dir`，如 `outputs/image/replay_20261001_zhihan`），或直接与上面的数值比较。

| 标注 | run_id | seed | epoch | 验证 Macro-F1 | 分类头 | SHA-256 |
|---|---|---|---|---|---|---|
| 图片组原始基线 | `20261001_clip_vitb32_seed42` | 42 | 7 | 0.5266 | 线性 768→2 | `66fcccc2c533` |
| 图片组 Zhihan 复现（文件与最终参考逐字节相同） | `20261001_clip_vitb32_seed42_zhihan_rtx3080` | 42 | 7 | 0.5276 | 线性 768→2 | `413cc9e91d29` |
| ★ 图片组最终参考 | `20261006_code_fixes_validation_v2` | 42 | 7 | 0.5276 | 线性 768→2 | `413cc9e91d29` |

### 4.2 多模态及对照（14 个，2026-10-16）

纯文字 / 纯图像对照使用与基线相同的 MLP（只换输入模态）；mean pooling 对照只把文字向量换成 masked mean pooling。特征目录按训练时记录（`features/<name>`），代码会自动解析到 `data/processed/features/<name>`。

| 标注 | run_id | 模态 | seed | epoch | 验证 Macro-F1 | 训练用特征缓存 | SHA-256 |
|---|---|---|---|---|---|---|---|
| ★ 多模态主实验（服务默认加载） | `20261016_distilbert_clip_seed42` | 文字+图像 | 42 | 25 | 0.5336 | `distilbert_first_token` + `20261006_code_fixes_validation_v2` | `b4020a38ca8b` |
| 多模态基线，seed 43 | `20261016_distilbert_clip_frozen_seed43` | 文字+图像 | 43 | 27 | 0.5315 | `distilbert_first_token` + `20261006_code_fixes_validation_v2` | `783322d17423` |
| 多模态基线，seed 44 | `20261016_distilbert_clip_frozen_seed44` | 文字+图像 | 44 | 21 | 0.5441 | `distilbert_first_token` + `20261006_code_fixes_validation_v2` | `0c11007c0b8c` |
| 多模态复现：重新提取文字 + CPU 图片特征 | `20261016_distilbert_clip_seed42_reproduction_cpu` | 文字+图像 | 42 | 29 | 0.5435 | `reproduction_distilbert_first_token` + `clip_vitb32_pooler_cpu` | `9f2640780438` |
| 多模态相同输入重跑（文件与主实验逐字节相同） | `20261016_distilbert_clip_seed42_rerun_same_inputs` | 文字+图像 | 42 | 25 | 0.5336 | `distilbert_first_token` + `20261006_code_fixes_validation_v2` | `b4020a38ca8b` |
| 对照：纯图像 | `20261016_clip_frozen_mlp_image_only_seed42` | 图像 | 42 | 29 | 0.5568 | `20261006_code_fixes_validation_v2` | `fb1909a059a2` |
| 对照：纯图像 | `20261016_clip_frozen_mlp_image_only_seed43` | 图像 | 43 | 23 | 0.5352 | `20261006_code_fixes_validation_v2` | `9e915f57e953` |
| 对照：纯图像 | `20261016_clip_frozen_mlp_image_only_seed44` | 图像 | 44 | 19 | 0.5457 | `20261006_code_fixes_validation_v2` | `d5ec6d5bb8e8` |
| 对照：纯文字 | `20261016_distilbert_frozen_mlp_text_only_seed42` | 文字 | 42 | 9 | 0.4903 | `distilbert_first_token` | `8589759247e3` |
| 对照：纯文字 | `20261016_distilbert_frozen_mlp_text_only_seed43` | 文字 | 43 | 12 | 0.4771 | `distilbert_first_token` | `0768ad1c855f` |
| 对照：纯文字 | `20261016_distilbert_frozen_mlp_text_only_seed44` | 文字 | 44 | 11 | 0.4912 | `distilbert_first_token` | `d3a1c5e64620` |
| 对照：文字 mean pooling + 图像 | `20261016_distilbert_meanpool_clip_frozen_seed42` | 文字+图像 | 42 | 29 | 0.5311 | `distilbert_mean_pooling` + `20261006_code_fixes_validation_v2` | `32fd801f6fb1` |
| 对照：文字 mean pooling + 图像 | `20261016_distilbert_meanpool_clip_frozen_seed43` | 文字+图像 | 43 | 28 | 0.5247 | `distilbert_mean_pooling` + `20261006_code_fixes_validation_v2` | `ce7433295332` |
| 对照：文字 mean pooling + 图像 | `20261016_distilbert_meanpool_clip_frozen_seed44` | 文字+图像 | 44 | 23 | 0.5282 | `distilbert_mean_pooling` + `20261006_code_fixes_validation_v2` | `db2575799ca6` |

### 4.3 过拟合诊断（42 个，2026-10-18）

由 `python -m src.evaluation.overfitting_diagnostic` 训练：融合（文字+图像）与纯图像 × 7 种分类头 × seed 42–44，缓存与选择规则和基线相同（配置默认的 `distilbert_first_token` + `20261006_code_fixes_validation_v2`；这些 checkpoint 没有记录特征目录，评估时自动用配置默认值）。**只用于分析**：16 个配置在同一份验证集上比较，最好值带选择偏乐观，不作最终模型。分析与结论见 [过拟合诊断](../Team_Review/multimodal/overfitting/README.md)。

"训练−验证 AUC 差距" 是最后 10 个 epoch（21–30）的平均，衡量记忆训练集的程度（基线：融合 0.468、纯图像 0.464）；只有线性头和学习率 1e-4 明显减小了它。

| 分类头 | 模态 | seed | epoch | 验证 Macro-F1 | 训练−验证 AUC 差距（后 10 epoch） | 参数量 | run_id |
|---|---|---|---|---|---|---|---|
| MLP 256 | 文字+图像 | 42 | 20 | 0.5370 | 0.459 | 393,986 | `20261018_overfit_mlp256_fusion_seed42` |
| MLP 256 | 文字+图像 | 43 | 30 | 0.5283 | 0.482 | 393,986 | `20261018_overfit_mlp256_fusion_seed43` |
| MLP 256 | 文字+图像 | 44 | 24 | 0.5223 | 0.480 | 393,986 | `20261018_overfit_mlp256_fusion_seed44` |
| MLP 256 | 图像 | 42 | 21 | 0.5534 | 0.450 | 197,378 | `20261018_overfit_mlp256_image_seed42` |
| MLP 256 | 图像 | 43 | 21 | 0.5527 | 0.466 | 197,378 | `20261018_overfit_mlp256_image_seed43` |
| MLP 256 | 图像 | 44 | 16 | 0.5527 | 0.470 | 197,378 | `20261018_overfit_mlp256_image_seed44` |
| MLP 128 | 文字+图像 | 42 | 30 | 0.5279 | 0.462 | 196,994 | `20261018_overfit_mlp128_fusion_seed42` |
| MLP 128 | 文字+图像 | 43 | 24 | 0.5209 | 0.455 | 196,994 | `20261018_overfit_mlp128_fusion_seed43` |
| MLP 128 | 文字+图像 | 44 | 28 | 0.5186 | 0.445 | 196,994 | `20261018_overfit_mlp128_fusion_seed44` |
| MLP 128 | 图像 | 42 | 25 | 0.5539 | 0.443 | 98,690 | `20261018_overfit_mlp128_image_seed42` |
| MLP 128 | 图像 | 43 | 25 | 0.5659 | 0.442 | 98,690 | `20261018_overfit_mlp128_image_seed43` |
| MLP 128 | 图像 | 44 | 11 | 0.5437 | 0.462 | 98,690 | `20261018_overfit_mlp128_image_seed44` |
| MLP 64 | 文字+图像 | 42 | 28 | 0.5299 | 0.450 | 98,498 | `20261018_overfit_mlp64_fusion_seed42` |
| MLP 64 | 文字+图像 | 43 | 15 | 0.5178 | 0.417 | 98,498 | `20261018_overfit_mlp64_fusion_seed43` |
| MLP 64 | 文字+图像 | 44 | 26 | 0.5153 | 0.439 | 98,498 | `20261018_overfit_mlp64_fusion_seed44` |
| MLP 64 | 图像 | 42 | 24 | 0.5447 | 0.462 | 49,346 | `20261018_overfit_mlp64_image_seed42` |
| MLP 64 | 图像 | 43 | 24 | 0.5396 | 0.463 | 49,346 | `20261018_overfit_mlp64_image_seed43` |
| MLP 64 | 图像 | 44 | 10 | 0.5415 | 0.451 | 49,346 | `20261018_overfit_mlp64_image_seed44` |
| 线性 | 文字+图像 | 42 | 25 | 0.5153 | 0.198 | 3,074 | `20261018_overfit_linear_fusion_seed42` |
| 线性 | 文字+图像 | 43 | 26 | 0.5197 | 0.199 | 3,074 | `20261018_overfit_linear_fusion_seed43` |
| 线性 | 文字+图像 | 44 | 26 | 0.5170 | 0.199 | 3,074 | `20261018_overfit_linear_fusion_seed44` |
| 线性 | 图像 | 42 | 20 | 0.5289 | 0.165 | 1,538 | `20261018_overfit_linear_image_seed42` |
| 线性 | 图像 | 43 | 28 | 0.5235 | 0.167 | 1,538 | `20261018_overfit_linear_image_seed43` |
| 线性 | 图像 | 44 | 26 | 0.5225 | 0.166 | 1,538 | `20261018_overfit_linear_image_seed44` |
| MLP 512 + dropout 0.5 | 文字+图像 | 42 | 12 | 0.5347 | 0.469 | 787,970 | `20261018_overfit_mlp512_dropout0.5_fusion_seed42` |
| MLP 512 + dropout 0.5 | 文字+图像 | 43 | 28 | 0.5264 | 0.467 | 787,970 | `20261018_overfit_mlp512_dropout0.5_fusion_seed43` |
| MLP 512 + dropout 0.5 | 文字+图像 | 44 | 30 | 0.5224 | 0.470 | 787,970 | `20261018_overfit_mlp512_dropout0.5_fusion_seed44` |
| MLP 512 + dropout 0.5 | 图像 | 42 | 27 | 0.5564 | 0.469 | 394,754 | `20261018_overfit_mlp512_dropout0.5_image_seed42` |
| MLP 512 + dropout 0.5 | 图像 | 43 | 30 | 0.5510 | 0.463 | 394,754 | `20261018_overfit_mlp512_dropout0.5_image_seed43` |
| MLP 512 + dropout 0.5 | 图像 | 44 | 27 | 0.5553 | 0.451 | 394,754 | `20261018_overfit_mlp512_dropout0.5_image_seed44` |
| MLP 512 + weight decay 1e-2 | 文字+图像 | 42 | 1 | 0.5221 | 0.465 | 787,970 | `20261018_overfit_mlp512_wd1e-2_fusion_seed42` |
| MLP 512 + weight decay 1e-2 | 文字+图像 | 43 | 28 | 0.5314 | 0.470 | 787,970 | `20261018_overfit_mlp512_wd1e-2_fusion_seed43` |
| MLP 512 + weight decay 1e-2 | 文字+图像 | 44 | 28 | 0.5295 | 0.466 | 787,970 | `20261018_overfit_mlp512_wd1e-2_fusion_seed44` |
| MLP 512 + weight decay 1e-2 | 图像 | 42 | 26 | 0.5802 | 0.447 | 394,754 | `20261018_overfit_mlp512_wd1e-2_image_seed42` |
| MLP 512 + weight decay 1e-2 | 图像 | 43 | 16 | 0.5604 | 0.460 | 394,754 | `20261018_overfit_mlp512_wd1e-2_image_seed43` |
| MLP 512 + weight decay 1e-2 | 图像 | 44 | 19 | 0.5574 | 0.456 | 394,754 | `20261018_overfit_mlp512_wd1e-2_image_seed44` |
| MLP 512 + 学习率 1e-4 | 文字+图像 | 42 | 20 | 0.5293 | 0.350 | 787,970 | `20261018_overfit_mlp512_lr1e-4_fusion_seed42` |
| MLP 512 + 学习率 1e-4 | 文字+图像 | 43 | 21 | 0.5216 | 0.368 | 787,970 | `20261018_overfit_mlp512_lr1e-4_fusion_seed43` |
| MLP 512 + 学习率 1e-4 | 文字+图像 | 44 | 25 | 0.5250 | 0.362 | 787,970 | `20261018_overfit_mlp512_lr1e-4_fusion_seed44` |
| MLP 512 + 学习率 1e-4 | 图像 | 42 | 17 | 0.5283 | 0.381 | 394,754 | `20261018_overfit_mlp512_lr1e-4_image_seed42` |
| MLP 512 + 学习率 1e-4 | 图像 | 43 | 16 | 0.5312 | 0.382 | 394,754 | `20261018_overfit_mlp512_lr1e-4_image_seed43` |
| MLP 512 + 学习率 1e-4 | 图像 | 44 | 20 | 0.5348 | 0.375 | 394,754 | `20261018_overfit_mlp512_lr1e-4_image_seed44` |

## 5. 使用方法 / How to use

```bash
# 在仓库根目录
unzip -n Checkpoints_merged_20261018_v2.zip                  # -> checkpoints/（17 个）
unzip -n Checkpoints_overfitting_diagnostic_20261018_v2.zip  # 可选：-> checkpoints/multimodal/20261018_overfit_*（42 个）
unzip -n Feature_Caches_merged_20261018_v2.zip                 # -> data/processed/features/
unzip -n Run_Outputs_20261018_v2.zip                           # 可选：-> outputs/、logs/（结果报告需要）
sha256sum -c checkpoints/SHA256SUMS_merged.txt                 # 校验已解压的 checkpoint（诊断包：SHA256SUMS_overfitting_diagnostic.txt）
sha256sum -c docs/checkpoints_SHA256SUMS.txt --ignore-missing  # 或者：按仓库中的完整清单校验
export MEMOTION_DATA_DIR=/path/to/MMSoc_Memotion_exported        # 缓存校验需要数据集的 metadata.csv

# 重放一个分类头（与记录的指标比较，容差 0.01）。仓库中的参考实验已有 reeval/，请给新目录；
# 其他 run 的参考指标 outputs/<task>/<run_id>/metrics_validation.json 在 Run_Outputs_20261018_v2.zip 中
# （两个 20261001_* 图片头没有实验目录，需加 --features_dir 与 --reference_metrics，见 4.1）
python -m src.evaluation.evaluate --run_id 20261016_distilbert_clip_seed42 \
    --out_dir outputs/multimodal/20261016_distilbert_clip_seed42/reeval_mine
python -m src.evaluation.evaluate --run_id 20261006_code_fixes_validation_v2 \
    --features_dir data/processed/features/20261006_code_fixes_validation_v2 \
    --out_dir outputs/image/20261006_code_fixes_validation_v2/quick_check

# 用某个分类头启动服务（FastAPI，接口见 docs/api_spec.md）
CHECKPOINT=checkpoints/multimodal/20261016_distilbert_clip_seed42/best.pt bash scripts/serve.sh
```

```python
import torch
from src.models.trainer import build_head
ckpt = torch.load("checkpoints/multimodal/20261016_distilbert_clip_seed42/best.pt",
                  map_location="cpu", weights_only=True)
head = build_head(ckpt["config"], {"text": torch.zeros(1, 768), "image": torch.zeros(1, 768)})
head.load_state_dict(ckpt["state_dict"]); head.eval()
prob = head(text=text_features, image=image_features).softmax(-1)[:, 1]   # P(sarcastic)
```

`text_features` / `image_features` 是缓存里的原始 768 维特征（分类头内部做 L2 归一化）。从原始文字和图片推理请用 `src.serving.predictor.MultimodalPredictor.from_checkpoint(...)`。

结果报告（`python -m src.evaluation.results_report`）需要全部 14 个 2026-10-16 多模态 checkpoint、`Run_Outputs_20261018_v2.zip` 中的实验目录和 `Feature_Caches_merged_20261018_v2.zip` 中的特征缓存（包括 `reproduction_distilbert_first_token`）。

## 6. 注意事项 / Caveats

* 验证集只有 699 张，约 0.03–0.04 以下的 Macro-F1 差异无法分辨；使用图像的分类头，单次实验所选 epoch 的值约偏高 0.015–0.02（纯文字约 0.046）。
* 2026-10-16 的融合基线与纯图像分类头（MLP 512，seed 42–44）在所选 epoch 几乎完美拟合训练集（训练 ROC-AUC 0.999–0.9997，验证 0.525–0.550；mean pooling 与 `_reproduction_cpu` 融合头相近），验证 loss 因高置信错误上升，概率偏过度自信，不宜当作校准后的概率使用。纯文字（训练 0.61–0.63）、诊断中的线性头（0.69–0.73）和学习率 1e-4（0.84–0.90）拟合明显更弱（[过拟合诊断](../Team_Review/multimodal/overfitting/README.md)）。
* 纯文字分类头接近或低于随机水平：冻结 DistilBERT 特征在这个任务上没有可测量的信号。
* Test 从未评分；最终评估只做一次，且图片组缓存没有 test 划分，需要使用 `clip_vitb32_pooler_cpu`。
