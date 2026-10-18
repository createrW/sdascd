# Clean image branch verification / 图片精简分支检测

*Historical record from the original team repository: the branches and commits named here (`image-baseline-clean`, `main` `ff79624`, `image-baseline` `51c6dbd`) are not part of this repository's history.*

*原团队仓库的历史记录：文中的分支与提交（`image-baseline-clean`、`main` 的 `ff79624`、`image-baseline` 的 `51c6dbd`）不在本仓库历史中。*

Verified on 2026-10-09 from the repository root, on `image-baseline-clean` based directly on `main` commit `ff79624`. The full audit branch remains at `51c6dbd`. Dataset access used `MEMOTION_DATA_DIR`. Environment: Windows, Python 3.12.14, PyTorch 2.11.0+cu128, torchvision 0.26.0+cu128, Transformers 5.18.0, NVIDIA RTX 3080.

2026-10-09 在仓库根目录检测 `image-baseline-clean`，直接基于 `main` 的 `ff79624`，完整审计分支仍为 `51c6dbd`。数据访问使用 `MEMOTION_DATA_DIR`。环境为 Windows、Python 3.12.14、PyTorch 2.11.0+cu128、torchvision 0.26.0+cu128、Transformers 5.18.0、NVIDIA RTX 3080。

| Check / 检查 | Result / 结果 |
|---|---|
| Complete regression suite / 完整回归套件 | 16 tests passed / 16 项通过 |
| Dependencies and Python compilation / 依赖与编译 | `pip check` and `compileall` passed / 均通过 |
| Dataset integrity / 数据完整性 | Train 5,593, Validation 699, Test 700; all loadable / 全部可读 |
| Full extraction route / 完整提取复现 | 11 steps passed; full Train/Validation features / 11 步通过，完整特征 |
| Cache route / 缓存复现 | 10 steps passed using extracted ZIP and metadata only; empty encoder cache, offline / 解压 ZIP 后仅用元数据、无编码器缓存、离线通过 |
| Independent training / 独立训练 | Both routes: Macro-F1 0.527606, accuracy 0.626609, best epoch 7/30 / 两种路线一致 |
| Checkpoint reload / 分类头重载 | Both routes: zero metric difference and identical prediction CSV contents / 两种路线指标差为零，预测内容完全一致 |
| Three retained heads / 三个保留分类头 | Original 0.529162; earlier Zhihan 0.527606; final 0.527606; all within 0.01 of final reference / 均在最终参考容差内 |
| Real out-of-tolerance evaluation / 实际超容差评估 | Deliberately different reference returned exit code 1 / 使用故意不一致的参考指标，返回 1 |
| Review/reference byte integrity / 人工表与参考文件完整性 | 32 rows in each CSV, reviewer Zhihan Zhu; CSVs, grids, and reference artifacts preserved / 32 行、检查人正确，文件字节保留 |

Regression coverage includes protected output/checkpoint paths, path traversal rejection, separate smoke caches, cache counts/IDs/labels/dimensions/dtypes/nonfinite values/hashes, threshold-boundary probability round trips, exact tolerance boundaries, failure propagation through logging, and explicit dataset/cache-mode requirements.

回归覆盖结果与 checkpoint 覆盖保护、路径穿越拒绝、独立 smoke 缓存、缓存数量/ID/标签/维度/类型/非有限值/哈希、阈值附近概率的 CSV 往返、容差边界、日志失败状态传递，以及数据环境变量和缓存模式要求。

Test images were checked for readability only; no Test predictions or scores were generated. These results cover the image pipeline on the environment above. Text/multimodal files from `main` are preserved; their training and cross-platform execution are outside this verification. Hardware variation may change probabilities while remaining within the agreed 0.01 Macro-F1 tolerance.

Test 图片只检查可读性，未生成 Test 预测或分数。本次验证覆盖上述环境的图片流程，保留了 `main` 的文字/多模态文件；不包含这些模型的训练或其他平台的执行。不同硬件的概率可能变化，复现按约定的 0.01 Macro-F1 容差判断。

Full step logs, environment details, negative checks, and fresh validation outputs are in `Image_Baseline_Clean_Verification_20261009.zip`, listed in [the cloud manifest](Image_Cloud_Assets.md).

完整逐步日志、环境记录、失败场景检测和新验证输出在[云端清单](Image_Cloud_Assets.md)中的 `Image_Baseline_Clean_Verification_20261009.zip`。
