# External multimodal assets / 多模态外部资料

**Status: prepared locally; upload pending.** The CPU session that produced them could not reach `lfs.github.com` (network policy), so the two archives are not in the repository yet. Every file in them can be regenerated on CPU in about 6 minutes with the commands in [src/multimodal/README.md](../src/multimodal/README.md); after upload, the sizes and SHA-256 below must match.

**状态：已在本地准备，尚未上传。** 生成它们的 CPU 会话无法访问 `lfs.github.com`（网络策略限制），两个压缩包暂未进入仓库。其中所有文件都可以按 [src/multimodal/README.md](../src/multimodal/README.md) 的命令在 CPU 上约 6 分钟重新生成；上传后大小与 SHA-256 应与下表一致。

These files are not in Git. In the `sdascd` storage repository they are under `Meme_Project_updates/Model_ckpt/{text,multimodal}/` (Git LFS). They match the empty `Meme_Project/Model_ckpt/text/` and `Meme_Project/Model_ckpt/multimodal/` folders on the team Drive. Extract each archive **from the repository root**; the archives already contain the destination folders. They contain no dataset images and no encoder weights.

这些文件不在 Git 中。它们位于 `sdascd` 存储仓库的 `Meme_Project_updates/Model_ckpt/{text,multimodal}/`（Git LFS），对应团队 Drive 中空着的 `Meme_Project/Model_ckpt/text/` 和 `Meme_Project/Model_ckpt/multimodal/`。请**从仓库根目录**解压，压缩包已包含目标目录；不含数据集图片和编码器权重。

| Filename / 文件名 | Bytes / 字节 | Creates / 生成目录 | Contents / 内容 |
|---|---:|---|---|
| `Text_Features_DistilBERT_20261016.zip` | 39871951 | `features/distilbert_first_token/`, `features/distilbert_mean_pooling/` | frozen DistilBERT train/validation/test caches (768-d `[CLS]` and mean pooling) + `meta.json` |
| `Multimodal_Assets_20261016.zip` | 56613660 | `features/clip_vitb32_pooler_cpu/`, `checkpoints/multimodal/`, `outputs/multimodal/`, `outputs/image/20261006_code_fixes_validation_v2_cpu_features_check/`, `logs/` | CPU CLIP cache incl. test; all 14 fusion-head checkpoints; all run folders (main, reproduction, rerun, ablations, seeds); image head on CPU features; step logs with timing and exit codes |

## SHA-256

```text
2c6916f6d4fa575961d655332276aa9cdf1d1b663a489b9ad3e994409213200b  Text_Features_DistilBERT_20261016.zip
3559ac223b6c8d754e3e6cb1ccba95a44bd02d46e05591d2a360a275ffce1860  Multimodal_Assets_20261016.zip
```

Fast route (no encoder download): unzip the image group's `Image_Features_CLIP_ViTB32_20261006.zip` and the two archives above at the repository root, then:

快速路线（无需下载编码器）：在仓库根目录解压图片组的 `Image_Features_CLIP_ViTB32_20261006.zip` 和上面两个包，然后：

```bash
python -m src.multimodal.evaluate --run_id 20261016_distilbert_clip_seed42 --out_dir outputs/multimodal/quick_check
python -m src.multimodal.train --run_id 20261016_distilbert_clip_seed42_my_reproduction
```

The cache loaders check every file's SHA-256, sample IDs, order and labels against the dataset metadata before use.

缓存读取程序在使用前会核对每个文件的 SHA-256、sample ID、顺序和标签是否与数据集元数据一致。
