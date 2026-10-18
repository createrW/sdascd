# Patches for the team repository / 团队仓库补丁

Apply on top of the `multimodal-baseline` branch as uploaded (`meme-sarcasm-robustness-multimodal-baseline.zip`):

在上传的 `multimodal-baseline` 分支之上按顺序应用：

```bash
git checkout multimodal-baseline
git am meme-sarcasm-robustness-multimodal-baseline-patches/*.patch
```

| Patch | Content / 内容 |
|---|---|
| `0001` | multimodal stage: DistilBERT text features, fusion MLP, ablations, results report, two reference runs |
| `0002` | restructure to the team code convention (configs/ data/ docs/ notebooks/ src/{data,features,models,evaluation,serving} tests/ scripts/*.sh setup.py Dockerfile), FastAPI serving, first review fixes |
| `0003` | overfitting diagnostic (`Team_Review/multimodal/overfitting/`, same-parameter train/validation scoring and head-size sweep) and fixes from the second verification round |
| `0004` | corrections to the diagnostic's claims after an independent check (memorization compared at fixed epochs; scoped conclusions) and small code fixes |
| `0005` | checkpoint guide `docs/checkpoints.md` (+ manifest, SHA-256); docs point at the 2026-10-18 asset archives |
| `0006`–`0008` | fixes from three fresh-copy verifications of the final package: `_v2` archives (adds the `reproduction_distilbert_first_token` cache), no absolute paths in reports, corrected docs and commands, packaging (`pydantic`, `.gitattributes`, `.dockerignore`) |

Apply only the patches you do not have yet (e.g. `0005`–`0008` if `0001`–`0004` are applied). Checked: the eight patches apply with `git am` to the uploaded zip, and the result is byte-identical to `../meme-sarcasm-robustness-multimodal-baseline/`.

只需应用尚未应用的补丁（例如已应用 `0001`–`0004` 时只应用 `0005`–`0008`）。已验证：八个补丁可用 `git am` 应用到上传的 zip，结果与 `../meme-sarcasm-robustness-multimodal-baseline/` 逐字节一致。
