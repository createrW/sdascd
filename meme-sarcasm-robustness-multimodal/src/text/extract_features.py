"""Run the frozen DistilBERT encoder over every `text_input` ONCE and cache 768-d sentence vectors.

Same idea as src/image/extract_features.py: the encoder is frozen, so a caption's vector never
changes, and the fusion MLP can then be trained in seconds on CPU. One pass writes both pooled
vectors described in src/text/encoder.py into two caches:

    <out_prefix>_first_token/<split>.pt    [CLS] state      (baseline, decisions D13)
    <out_prefix>_mean_pooling/<split>.pt   masked mean      (ablation)
        -> {"sample_id": list[str], "features": FloatTensor[N, 768], "labels": LongTensor[N]}
    <out_prefix>_<feature>/meta.json       model revision, tokenizer settings, truncation counts

Tokenization follows README 4.1: the model's own uncased WordPiece tokenizer, [CLS] ... [SEP],
max length 128 with truncation; `[NO_TEXT]` (shared loader) is encoded like any other caption.

Usage (repo root, MEMOTION_DATA_DIR set):
    python -m src.text.extract_features                               # train validation test
    python -m src.text.extract_features --limit 64 --splits validation    # separate smoke cache
    python -m src.text.extract_features --model_dir hf_local/distilbert-base-uncased   # offline copy
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import torch
import transformers
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.common.data import NO_TEXT, SPLITS, load_split  # noqa: E402
from src.common.reproducibility import get_device, set_seed  # noqa: E402
from src.image.cache import file_sha256  # noqa: E402
from src.image.safety import require_new_paths  # noqa: E402
from src.text.encoder import (TEXT_FEATURES, load_text_encoder, pool_text_features,  # noqa: E402
                              tokenize)


def weights_sha256(model_dir):
    """Hash of the local weight file, recorded so an offline copy can be traced."""
    for name in ("model.safetensors", "pytorch_model.bin"):
        path = Path(model_dir) / name
        if path.is_file():
            with path.open("rb") as stream:
                return name, hashlib.file_digest(stream, "sha256").hexdigest()
    return None, None


@torch.no_grad()
def extract_split(split, tokenizer, model, device, batch_size, max_length, data_dir, limit=None):
    df = load_split(split, data_dir)
    if limit:
        df = df.head(limit)
    texts = df["text_input"].tolist()

    pooled = {how: [] for how in TEXT_FEATURES}
    lengths = []
    for start in tqdm(range(0, len(texts), batch_size), desc=f"{split:>10}"):
        batch = texts[start:start + batch_size]
        enc = tokenize(tokenizer, batch, max_length)
        hidden = model(input_ids=enc["input_ids"].to(device),
                       attention_mask=enc["attention_mask"].to(device)).last_hidden_state
        mask = enc["attention_mask"].to(device)
        for how in TEXT_FEATURES:
            pooled[how].append(pool_text_features(hidden, mask, how).float().cpu())
        # untruncated length (with [CLS]/[SEP]) to report how many captions lose tokens at 128
        lengths += [len(ids) for ids in tokenizer(batch, truncation=False)["input_ids"]]

    labels = torch.tensor(df["sarcasm_binary"].to_numpy(), dtype=torch.long)
    lengths_t = torch.tensor(lengths, dtype=torch.float32)
    stats = {
        "no_text": int((df["text_input"] == NO_TEXT).sum()),
        "truncated_at_max_length": int((lengths_t > max_length).sum()),
        "tokens_mean": round(float(lengths_t.mean()), 2),
        "tokens_median": float(lengths_t.median()),
        "tokens_max": int(lengths_t.max()),
    }
    results = {how: {"sample_id": df["sample_id"].tolist(), "features": torch.cat(chunks),
                     "labels": labels} for how, chunks in pooled.items()}
    return results, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/multimodal_baseline.json")
    ap.add_argument("--data_dir", default=None)
    ap.add_argument("--model_dir", default=None,
                    help="local copy of distilbert-base-uncased; default loads the pinned Hub revision")
    ap.add_argument("--out_prefix", default=None,
                    help="caches go to <out_prefix>_first_token and <out_prefix>_mean_pooling "
                         "(must be new); default features/distilbert")
    ap.add_argument("--splits", nargs="+", choices=SPLITS, default=list(SPLITS))
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--limit", type=int, default=None, help="only the first N rows (smoke test)")
    args = ap.parse_args()
    if args.batch_size <= 0 or (args.limit is not None and args.limit <= 0):
        ap.error("batch_size and limit must be positive")
    if len(set(args.splits)) != len(args.splits):
        ap.error("splits must not contain duplicates")
    prefix = args.out_prefix or "features/distilbert"
    if args.limit is not None and args.out_prefix is None:
        prefix += f"_smoke_{args.limit}"
    out_dirs = {how: Path(f"{prefix}_{how}") for how in TEXT_FEATURES}
    require_new_paths(*out_dirs.values())  # fail before loading DistilBERT

    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    set_seed(cfg["seed"])
    device = get_device()
    max_length = cfg["text_max_length"]
    source = args.model_dir or f"{cfg['text_model_id']}@{cfg['text_revision']}"
    print(f"device={device}  model={source}  max_length={max_length}")

    tokenizer, model = load_text_encoder(cfg, device, args.model_dir)
    weight_file, weight_hash = weights_sha256(args.model_dir) if args.model_dir else (None, None)
    for out_dir in out_dirs.values():
        out_dir.mkdir(parents=True, exist_ok=False)

    base_meta = {
        "text_model_id": cfg["text_model_id"],
        "text_revision_requested": cfg["text_revision"],
        "text_commit_resolved": getattr(model.config, "_commit_hash", None),
        "text_weights_source": "local_dir" if args.model_dir else "hub",
        "text_weights_file": weight_file,
        "text_weights_sha256": weight_hash,
        "tokenizer": type(tokenizer).__name__,
        "do_lower_case": bool(getattr(tokenizer, "do_lower_case", True)),
        "max_length": max_length,
        "truncation": True,
        "padding": "longest_in_batch",
        "batch_size": args.batch_size,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "device": str(device),
        "limit": args.limit,
    }
    metas = {how: {**base_meta, "text_feature": how, "splits": {}} for how in TEXT_FEATURES}
    for split in args.splits:
        t0 = time.time()
        results, stats = extract_split(split, tokenizer, model, device, args.batch_size,
                                       max_length, args.data_dir, args.limit)
        secs = time.time() - t0
        for how, res in results.items():
            f = res["features"]
            assert len(res["sample_id"]) == f.shape[0] == res["labels"].shape[0]
            assert torch.isfinite(f).all(), f"{split}/{how}: NaN/inf in features"
            path = out_dirs[how] / f"{split}.pt"
            torch.save(res, path)
            metas[how]["splits"][split] = {"n": f.shape[0], "dim": f.shape[1],
                                           "seconds": round(secs, 1), "sha256": file_sha256(path),
                                           **stats}
            print(f"  saved {path}  shape={tuple(f.shape)}")
        print(f"  {split}: {secs:.1f}s  {stats}")

    for how, out_dir in out_dirs.items():
        (out_dir / "meta.json").write_text(json.dumps(metas[how], indent=2), encoding="utf-8")
    print("done. meta ->", ", ".join(str(d / "meta.json") for d in out_dirs.values()))


if __name__ == "__main__":
    main()
