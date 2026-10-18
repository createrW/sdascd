# sdascd — CSCI 566 meme sarcasm project storage / 项目存储

| Path / 路径 | What it is / 内容 |
|---|---|
| `Meme_Project-20261016T002724Z-1-001.zip` | Drive export: README (EN/中文), proposal, dataset `MMSoc_Memotion_exported.zip`, image group cloud assets |
| `meme-sarcasm-robustness-main.zip`, `meme-sarcasm-robustness-image-baseline-clean.zip` | team repository snapshots (`main`, `image-baseline-clean`) |
| **`meme-sarcasm-robustness-multimodal-baseline/`** | **current version to submit**: the uploaded `multimodal-baseline` branch merged with the multimodal stage and restructured to the team code convention (`configs/{train,deploy}`, `data/{raw,processed,samples}`, `docs/{design,experiment_log,api_spec}.md`, `notebooks/`, `src/{data,features,models,evaluation,serving}`, `tests/`, `scripts/*.sh`, `setup.py`, `Dockerfile`), with a FastAPI service. Start with its `README.md` |
| `meme-sarcasm-robustness-multimodal-baseline-patches/` | the same as `git am` patches for the team repository (`0001` multimodal stage, `0002` restructure + serving, `0003` overfitting diagnostic + fixes) |
| `meme-sarcasm-robustness-multimodal-baseline-checkpoints/` | **all trained heads of the merged version** (59: image group 3, multimodal 14, overfitting diagnostic 42) as `checkpoints/...` with `MANIFEST.csv` and `SHA256SUMS`; copy into the repository root. Plain files, not LFS (lfs.github.com is blocked here); the same content and the feature caches were also sent as zips |
| `meme-sarcasm-robustness-multimodal/` | earlier version (old layout, superseded by the folder above); **multimodal stage implemented**: `image-baseline-clean` + DistilBERT text features + multimodal training/evaluation, run on CPU (teammate reproduction pending). Start with **`Team_Review/multimodal/RESULTS_REPORT.md`** (statistics, figures, conclusions) |
| `meme-sarcasm-robustness-multimodal.patch` | the same changes as one patch for the team repository: `git apply meme-sarcasm-robustness-multimodal.patch` on `image-baseline-clean` (reference run folders under `outputs/multimodal/` are copied separately from the folder above) |
| `Meme_Project_updates/Model_ckpt/text/` | `Text_Features_DistilBERT_20261016.zip` → upload to Drive `Meme_Project/Model_ckpt/text/` |
| `Meme_Project_updates/Model_ckpt/multimodal/` | `Multimodal_Assets_20261016.zip` (fusion checkpoints, CPU CLIP cache with test, all runs, logs) → Drive `Meme_Project/Model_ckpt/multimodal/` |

The two `Meme_Project_updates` zips are **pending upload**: the CPU session could not reach `lfs.github.com`, so only their manifests and SHA-256 are committed. Their contents can be regenerated on a 4-vCPU CPU in about 15–25 minutes (`meme-sarcasm-robustness-multimodal/docs/Multimodal_Assets.md`).

`Meme_Project_updates` 中的两个 zip **尚未上传**：CPU 会话无法访问 `lfs.github.com`，因此只提交了清单和 SHA-256；其内容可在 4 vCPU 的 CPU 上约 15–25 分钟重新生成（见 `meme-sarcasm-robustness-multimodal/docs/Multimodal_Assets.md`）。

Validation (seed 42): multimodal Macro-F1 0.5336, accuracy 0.6567; 3 seeds 0.5364 ± 0.0067; majority baseline 0.4349; a label-independent classifier scores ≈ 0.50. The signal comes from the image features: frozen DistilBERT text adds no measurable gain. Test not scored.

Validation（seed 42）：多模态 Macro-F1 0.5336，准确率 0.6567；3 个 seed 0.5364 ± 0.0067；多数类基线 0.4349；不看标签的分类器约 0.50。信号来自图像特征，冻结 DistilBERT 文字没有可测量的增益。Test 未评分。

**2026-10-17 restructure.** Behaviour is unchanged: retraining with the new code and replaying all 17 shipped heads give byte-identical predictions; the results-report statistics regenerate byte-identically; 54 tests pass (one PowerShell test skipped). Serving was checked live over HTTP: 20 validation memes match the recorded probabilities within 9.5e-7. The Docker image was written but not built (no Docker daemon in this session).

**2026-10-17 目录重构。** 行为不变：用新代码重新训练、重放全部 17 个分类头，预测逐字节一致；结果报告统计逐字节一致；54 项测试通过（1 项 PowerShell 测试跳过）。在线接口已实际验证：20 张 validation 梗图的概率与记录一致（差异 ≤ 9.5e-7）。Dockerfile 已编写，但本会话无 Docker，未构建镜像。

**2026-10-18 overfitting diagnostic.** The recorded runs were retrained byte-identically while every epoch's parameters were scored on train and validation in `eval()` mode. The heads memorize the training set (train ROC-AUC ≥ 0.999 vs validation ≈ 0.54), validation loss rises through over-confident errors while validation ranking and Macro-F1 do not decline, and smaller or more regularized heads shrink the gap without improving validation or letting fusion beat image only (fusion higher in 1 of 24 seed pairs). So overfitting is observed, but it is not supported as the main reason fusion shows no gain. Details: `meme-sarcasm-robustness-multimodal-baseline/Team_Review/multimodal/overfitting/README.md`.

**2026-10-18 过拟合诊断。** 重新训练（与记录逐字节一致），并在每个 epoch 用同一组参数、`eval()` 模式同时评估训练集与验证集：分类头记住了训练集（训练 ROC-AUC ≥ 0.999，验证约 0.54）；验证 loss 因高置信错误上升，而验证排序能力和 Macro-F1 没有下降；更小或正则更强的分类头能缩小差距，但不能提高验证分数，也不能让融合超过纯图像（24 对 seed 中 1 对）。因此过拟合是观察到的事实，但不支持它是融合没有提升的主要原因。
