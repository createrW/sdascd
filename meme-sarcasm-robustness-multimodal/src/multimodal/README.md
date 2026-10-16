# Multimodal baseline: frozen DistilBERT + frozen CLIP / 多模态基线

```text
text_input -> frozen DistilBERT -> [CLS] last_hidden_state[:, 0] (768) -> L2 \
                                                                              concat [text | image] (1536)
image      -> frozen CLIP ViT-B/32 vision -> pooler_output (768)      -> L2 /        |
                                                                                       v
                                                     Linear 1536->512 -> ReLU -> Dropout 0.2 -> Linear 512->2
```

Only the fusion MLP (787,970 parameters) is trained. Both encoders are frozen, so their outputs are computed once and cached; the MLP then trains in about 45 s on a 4-core CPU. `FusionHead` (cached features) and `MultimodalClassifier` (raw text + raw image) share the same normalization, concatenation order and MLP, and a trained head loads into the end-to-end model unchanged (`check_end_to_end.py` verifies this).

只训练融合 MLP（787,970 个参数）。两个编码器都冻结，输出只需计算一次并缓存；之后在 4 核 CPU 上训练 MLP 约 45 秒。`FusionHead`（缓存特征）和 `MultimodalClassifier`（原始文字 + 原始图片）使用同一套归一化、拼接顺序和 MLP，训练好的分类头可直接装入端到端模型（由 `check_end_to_end.py` 验证）。

Results, reproduction evidence and the manual review are in [Team_Review/multimodal/README.md](../../Team_Review/multimodal/README.md).

结果、复现证据和人工核对见 [Team_Review/multimodal/README.md](../../Team_Review/multimodal/README.md)。

## Text features: how DistilBERT is used / 文字特征：DistilBERT 的用法

| Item / 项目 | Setting / 设置 | Why / 依据 |
|---|---|---|
| Model / 模型 | `distilbert/distilbert-base-uncased` @ `1c4513b2eedbda136f57676a34eea67aba266e5c`, `DistilBertModel` (MLM head unused) | README 4.1 |
| Input / 输入 | `text_input` from the shared loader (= `text_corrected`; empty -> `[NO_TEXT]`) | README 3.4; the proposal encodes the corrected text |
| Tokenizer / 分词 | the model's own uncased WordPiece, `[CLS] text [SEP]`, `truncation=True`, `max_length=128`, padding to the longest caption in the batch with `attention_mask` | README 4.1; only 4 train + 1 test captions exceed 128 tokens (median 20) |
| Sentence vector / 句向量 | **`first_token`**: final-layer `[CLS]` state `last_hidden_state[:, 0]` (768-d) | DistilBERT (Sanh et al., 2019) removes BERT's pooler, so there is no `pooler_output`; Hugging Face's `DistilBertForSequenceClassification`, written by the paper's authors, classifies from exactly this vector. Team decision D13. |
| Ablation / 对照 | `mean_pooling`: mean of `last_hidden_state` over real tokens (mask excludes padding) | Usual sentence vector for a *frozen* encoder (Sentence-BERT, Reimers & Gurevych, 2019) |
| Mode / 模式 | `eval()`, `no_grad`, float32, raw (un-normalized) vectors cached; L2 normalization happens inside the fusion head | identical code path for cached and end-to-end inference |

[CLS] stays the baseline. Mean pooling was not better on validation (3 seeds: 0.5280 ± 0.0032 vs 0.5364 ± 0.0067 Macro-F1). Both vectors are cached in one pass (`src/text/extract_features.py`). `src/text/encoder.py` holds the pooling code shared with `MultimodalClassifier`.

[CLS] 仍作为基线；mean pooling 在 Validation 上没有更好（3 个 seed：Macro-F1 0.5280 ± 0.0032 对 0.5364 ± 0.0067）。两种向量在一次运行中同时缓存（`src/text/extract_features.py`）；池化代码在 `src/text/encoder.py`，与 `MultimodalClassifier` 共用。

## Image features / 图片特征

The baseline uses the image group's reference cache `Image_Features_CLIP_ViTB32_20261006.zip` (SHA-256 `7a02d46b…5662`, train + validation, 768-d `pooler_output`, CUDA). It has no test split. The same encoder was therefore re-run on CPU for train/validation/test (`features/clip_vitb32_pooler_cpu`). Both caches agree to mean cosine 0.99997. The few larger gaps (minimum cosine 0.981) all come from TF32 in the CUDA patch-embedding convolution, as [the test report](../../docs/Multimodal_Test_Results.md) shows.

基线使用图片组的参考缓存 `Image_Features_CLIP_ViTB32_20261006.zip`（SHA-256 `7a02d46b…5662`，train + validation，768 维 `pooler_output`，CUDA 提取），其中没有 test。因此在 CPU 上用同一编码器重新提取了 train/validation/test（`features/clip_vitb32_pooler_cpu`）。两份缓存的平均余弦相似度为 0.99997；少数较大差异（最小 0.981）完全来自 CUDA patch-embedding 卷积的 TF32，见[检测报告](../../docs/Multimodal_Test_Results.md)。

## Commands / 命令

Run from the repository root with `MEMOTION_DATA_DIR` set. Existing outputs, checkpoints and caches are protected; use new run IDs and directories. Everything below was run on CPU.

从仓库根目录运行并设置 `MEMOTION_DATA_DIR`。已有输出、checkpoint 和缓存受覆盖保护，请使用新的 run ID 和目录。以下全部在 CPU 上运行过。

```bash
# 0. data check / 数据检查
python -m scripts.verify_dataset

# 1. text features (~1.5 min CPU): features/distilbert_first_token + features/distilbert_mean_pooling
python -m src.text.extract_features

# 2. image features: unzip the image group's cache at the repo root (train/validation) ...
#    or extract all three splits yourself (~3-4 min CPU per 5.6k images):
python -m src.image.extract_features --splits train validation test --out_dir features/clip_vitb32_pooler_cpu

# 3. train the fusion MLP (README 6 outputs + checkpoints/multimodal/<run_id>/best.pt)
python -m src.multimodal.train --run_id 20261016_distilbert_clip_seed42
python -m scripts.verify_predictions outputs/multimodal/20261016_distilbert_clip_seed42/predictions_validation.csv

# 4. reload the saved head (compares with the reported metrics, tolerance 0.01)
python -m src.multimodal.evaluate --run_id 20261016_distilbert_clip_seed42

# 5. second-member reproduction: fresh text cache + CPU image cache, new run id
python -m src.text.extract_features --out_prefix features/reproduction_distilbert
python -m src.multimodal.train --text_features_dir features/reproduction_distilbert_first_token \
    --image_features_dir features/clip_vitb32_pooler_cpu --run_id 20261016_distilbert_clip_seed42_reproduction_cpu
python -m scripts.verify_reproduction \
    --reference_metrics outputs/multimodal/20261016_distilbert_clip_seed42/metrics_validation.json \
    --candidate_metrics outputs/multimodal/20261016_distilbert_clip_seed42_reproduction_cpu/metrics_validation.json

# 6. end-to-end check (raw text + raw image through MultimodalClassifier) and smoke test
python -m src.multimodal.check_end_to_end --run_id 20261016_distilbert_clip_seed42_reproduction_cpu
python -m src.multimodal.smoke_test --batch_size 2

# 7. manual review sheet (4 TP / TN / FP / FN)
python -m src.multimodal.review_samples --run_id 20261016_distilbert_clip_seed42

# 8. ablations and extra seeds (supplementary, validation only)
python -m src.multimodal.train --config configs/ablations/text_only.json      # also image_only, mean_pooling
python -m src.multimodal.train --seed 43
python -m scripts.summarize_multimodal_runs

# 9. FINAL ONLY, once: test split (needs an image cache that contains test)
python -m src.multimodal.evaluate --run_id 20261016_distilbert_clip_seed42 --split test \
    --image_features_dir features/clip_vitb32_pooler_cpu
```

Test was not scored. No setting was chosen with test data.

Test 尚未评分，所有设置都没有使用 test 选择。

## Offline machines / 无法访问 Hugging Face 的机器

The default path loads the pinned Hub revisions. If `huggingface.co` is blocked, as it was in the CPU environment used here, every script accepts a local copy of the same weights: `--model_dir` for the extractors, `--text_model_dir` / `--clip_model_dir` for the smoke test and end-to-end check.

默认从 Hub 加载固定 revision。如果无法访问 `huggingface.co`（本次 CPU 环境即如此），所有脚本都可以改用同一权重的本地副本：提取脚本用 `--model_dir`，smoke test 和端到端检查用 `--text_model_dir` / `--clip_model_dir`。

| Encoder / 编码器 | Local copy used here / 本次使用的本地副本 | Check / 校验 |
|---|---|---|
| DistilBERT | Hugging Face's legacy S3 files `distilbert-base-uncased-{pytorch_model.bin,config.json,tokenizer.json}` + `bert-base-uncased-vocab.txt`, plus `tokenizer_config.json` `{"do_lower_case": true, "model_max_length": 512}` | weights SHA-256 `e60d7161…ea7c` (recorded in every cache `meta.json`); 66,362,880 parameters; all 100 encoder tensors load; MLM sanity check sensible |
| CLIP ViT-B/32 | OpenAI `ViT-B-32.pt`, converted by `scripts/convert_openai_clip_to_hf.py` | SHA-256 `40d36571…0af` = hash in OpenAI's official URL; reproduces the image group's CUDA features once TF32 is emulated (cosine 1.00000) |

## Files / 文件

| File / 文件 | Purpose / 用途 |
|---|---|
| `src/text/encoder.py` | load frozen DistilBERT, tokenize, `first_token` / `mean_pooling` |
| `src/text/extract_features.py`, `src/text/cache.py` | write / validate text caches (same layout and checks as the image cache) |
| `src/multimodal/features.py` | joint loader: validates both caches, joins on `sample_id` |
| `src/multimodal/model.py` | `FusionHead`, `MultimodalClassifier`, `load_fusion_head` |
| `src/multimodal/train.py`, `evaluate.py` | training with README 6 outputs; checkpoint replay / test scoring |
| `src/multimodal/check_end_to_end.py` | cached path == raw-input path |
| `src/multimodal/review_samples.py` | manual-review grid and CSV |
| `src/multimodal/smoke_test.py` | real-data forward pass (shapes) |
| `configs/multimodal_baseline.json`, `configs/ablations/*.json` | settings |
| `scripts/compare_feature_caches.py`, `scripts/convert_openai_clip_to_hf.py`, `scripts/summarize_multimodal_runs.py` | cache comparison, offline CLIP, run table |
| `tests/test_multimodal_pipeline.py` | 9 tests (cache checks, sample_id alignment, fusion order, head transfer, pooling, overwrite protection, train/evaluate round trip) |
