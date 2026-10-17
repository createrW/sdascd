# Multimodal verification / 多模态检测记录

Verified on 2026-10-16 from the repository root. Environment: Linux, 4 vCPU, no GPU, Python 3.12.3, PyTorch 2.11.0, torchvision 0.26.0, Transformers 5.18.0. Dataset `MMSoc_Memotion_exported.zip` (SHA-256 `9d019fb4…330c`). Step logs are in `logs/` inside `Multimodal_Assets_20261016.zip` ([asset list](Multimodal_Assets.md)).

2026-10-16 在仓库根目录检测。环境：Linux、4 vCPU、无 GPU、Python 3.12.3、PyTorch 2.11.0、torchvision 0.26.0、Transformers 5.18.0。数据集 `MMSoc_Memotion_exported.zip`（SHA-256 `9d019fb4…330c`）。逐步日志在 `Multimodal_Assets_20261016.zip` 的 `logs/` 中（[资料清单](Multimodal_Assets.md)）。

| Check / 检查 | Result / 结果 |
|---|---|
| Dataset / 数据集 | `DATASET OK`: 5,593 / 699 / 700 rows, train labels 1,226 / 4,367, all 6,992 images readable |
| Image group cache / 图片组缓存 | `Image_Features_CLIP_ViTB32_20261006.zip` SHA-256 matches the manifest; cache checks (IDs, order, labels, dims, hashes) pass |
| Text extraction / 文字特征提取 | train 67.0 s, validation 7.6 s, test 8.3 s on CPU; 768-d, finite; `[NO_TEXT]` 1 train / 3 test; >128 tokens 4 train / 1 test; median 20 tokens |
| Text re-extraction / 文字特征重提取 | fresh extraction into new folders is bit-identical on all three splits (max abs diff 0) |
| CLIP on CPU vs image group CUDA cache / CPU 与 CUDA 图片特征 | mean cosine 0.99997 (train and validation); mean abs diff 1.8e-3 (feature mean abs 0.73); min cosine 0.9815 (`train_004233`), 0.9854 (`validation_000550`) |
| Cause of that gap / 差异原因 | `scripts/check_tf32_emulation.py` (output `Team_Review/multimodal/report/tf32_emulation.json`) rounds the patch-embedding convolution to TF32, PyTorch's CUDA default (`cudnn.allow_tf32=True`). On the first 64 validation images the mean abs diff drops from 1.7e-3 to 2.9e-5. The 10 worst train and 10 worst validation images go from cosine 0.981–0.985 to ≥ 0.99999995. Weights and preprocessing agree; the CUDA cache is the approximate one. |
| Image head on CPU features / 图片分类头使用 CPU 特征 | image group's final head: Macro-F1 0.528634 vs reported 0.527606 (diff 0.0010, within D11) |
| Unit tests / 单元测试 | `tests/test_multimodal_pipeline.py`: 9/9 pass. Image suite: 15/16 pass; `test_reproduction_script_requires_dataset_and_one_cache_mode` needs `powershell`, which Linux does not have (not related to this change) |
| Smoke test / Smoke test | text (2, 768), image (2, 768), fused (2, 1536), logits (2, 2); `MULTIMODAL SMOKE TEST OK` |
| Main run / 主实验 | `20261016_distilbert_clip_seed42`: Macro-F1 0.533637, accuracy 0.656652, best epoch 25/30, 43.5 s; `PREDICTIONS OK` (699 rows) |
| Checkpoint reload / Checkpoint 重载 | identical metrics and prediction CSV (tolerance 0) |
| Same-input rerun / 相同输入重跑 | `..._rerun_same_inputs`: identical metrics and predictions (CPU training is deterministic) |
| Same-agent rerun / 同一 agent 重跑 | re-extracted (bit-identical) text cache + CPU image cache: Macro-F1 0.543468 vs 0.533637, diff 0.009831 <= 0.01 -> `REPRODUCTION OK` (best epoch 29 instead of 25; margin is small) |
| End-to-end, CPU caches / 端到端（CPU 缓存） | 64 validation samples through `MultimodalClassifier` from raw text and images: text diff 2.4e-7, image diff 0, probability diff 6.4e-7, labels 64/64 -> `END-TO-END CHECK OK` |
| End-to-end, main run / 端到端（主实验） | same check against the CUDA-cache head: labels 64/64 agree, but probability diff 0.0131 > 0.01 because of `validation_000550` (TF32 gap above) -> reported as `FAILED` at that tolerance; not hidden or loosened |
| Manual review / 人工核对 | 16 validation samples (4 TP/TN/FP/FN): every image shows the caption stored for its `sample_id`; 4 labels debatable (listed in the CSV) |
| Test split / Test | not scored |

Recommendation for future CUDA extraction: set `torch.backends.cudnn.allow_tf32 = False` (and `torch.backends.cuda.matmul.allow_tf32 = False`) so CPU, MPS and CUDA caches agree to float precision. This change was not made to the image group's code.

对今后 CUDA 提取的建议：设置 `torch.backends.cudnn.allow_tf32 = False`（以及 `torch.backends.cuda.matmul.allow_tf32 = False`），使 CPU、MPS、CUDA 缓存达到浮点精度一致。本次未修改图片组代码。
