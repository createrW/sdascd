# sdascd — CSCI 566 meme sarcasm project storage / 项目存储

| Path / 路径 | What it is / 内容 |
|---|---|
| `Meme_Project-20261016T002724Z-1-001.zip` | Drive export: README (EN/中文), proposal, dataset `MMSoc_Memotion_exported.zip`, image group cloud assets |
| `meme-sarcasm-robustness-main.zip`, `meme-sarcasm-robustness-image-baseline-clean.zip` | team repository snapshots (`main`, `image-baseline-clean`) |
| `meme-sarcasm-robustness-multimodal/` | **multimodal stage complete**: `image-baseline-clean` + DistilBERT text features + multimodal training/evaluation/reproduction, run on CPU. Start with its `README.md` → `Team_Review/multimodal/README.md` |
| `meme-sarcasm-robustness-multimodal.patch` | the same changes as one patch for the team repository: `git apply meme-sarcasm-robustness-multimodal.patch` on `image-baseline-clean` (reference run folders under `outputs/multimodal/` are copied separately from the folder above) |
| `Meme_Project_updates/Model_ckpt/text/` | `Text_Features_DistilBERT_20261016.zip` → upload to Drive `Meme_Project/Model_ckpt/text/` |
| `Meme_Project_updates/Model_ckpt/multimodal/` | `Multimodal_Assets_20261016.zip` (fusion checkpoints, CPU CLIP cache with test, all runs, logs) → Drive `Meme_Project/Model_ckpt/multimodal/` |

The two `Meme_Project_updates` zips are **pending upload**: the CPU session could not reach `lfs.github.com`, so only their manifests and SHA-256 are committed. Their contents can be regenerated on CPU in about 6 minutes (`meme-sarcasm-robustness-multimodal/src/multimodal/README.md`).

`Meme_Project_updates` 中的两个 zip **尚未上传**：CPU 会话无法访问 `lfs.github.com`，因此只提交了清单和 SHA-256；其内容可在 CPU 上约 6 分钟重新生成（见 `meme-sarcasm-robustness-multimodal/src/multimodal/README.md`）。

Validation (seed 42): multimodal Macro-F1 0.5336, accuracy 0.6567; 3 seeds 0.5364 ± 0.0067; majority baseline 0.4349. Test not scored.

Validation（seed 42）：多模态 Macro-F1 0.5336，准确率 0.6567；3 个 seed 0.5364 ± 0.0067；多数类基线 0.4349。Test 未评分。
