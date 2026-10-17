"""Feature engineering: run the frozen encoders over every sample ONCE and cache the vectors.

Why cache? Both encoders are frozen, so a sample's vector never changes. Computing it once means
training a classification/fusion head later takes seconds on CPU instead of re-running the encoders
every epoch. Caches are validated by src/features/cache.py before use.

Image (CLIP ViT-B/32, README 4.2): the official CLIP processor (resize, center crop, normalize) on the
RGB image, vision tower `pooler_output` (768-d, after post-LayerNorm; `image_embeds` = 512-d projection).

Text (DistilBERT, README 4.1): the model's own uncased WordPiece tokenizer, [CLS] ... [SEP], max length
128 with truncation; one pass writes two sentence vectors:
  first_token   last_hidden_state[:, 0], the final-layer [CLS] state. DistilBERT (Sanh et al., 2019)
                removed BERT's pooler, so there is no `pooler_output`; HF's
                DistilBertForSequenceClassification classifies from this token position (after
                fine-tuning and a pre-classifier layer). Team baseline feature (docs/decisions.md D13).
  mean_pooling  mean of last_hidden_state over real tokens (padding excluded via attention_mask), the
                usual sentence vector for a frozen encoder (Sentence-BERT); kept as an ablation.

Usage (repo root, MEMOTION_DATA_DIR set; outputs go to data/processed/features/ by default):
    python -m src.features.feature_engineering image                        # train + validation
    python -m src.features.feature_engineering image --splits train validation test --out_dir ...
    python -m src.features.feature_engineering image --limit 128            # separate smoke cache
    python -m src.features.feature_engineering text                         # train validation test
    python -m src.features.feature_engineering text --model_dir hf_local/distilbert-base-uncased

Output, one file per split:  <dir>/<split>.pt -> {"sample_id": list[str],
                                                 "features": FloatTensor[N, D], "labels": LongTensor[N]}
                             <dir>/meta.json   -> model revision, settings, versions, timing, SHA-256
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import torch
import transformers
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel, AutoTokenizer, CLIPModel, CLIPProcessor

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.dataset import NO_TEXT, SPLITS, load_split  # noqa: E402
from src.data.preprocess import load_image, tokenize  # noqa: E402
from src.features.cache import file_sha256  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.reproducibility import get_device, set_seed  # noqa: E402
from src.utils.safety import require_new_paths  # noqa: E402

FEATURES_ROOT = "data/processed/features"
TEXT_FEATURES = ("first_token", "mean_pooling")


# ----------------------------------------------------------------------------- image (CLIP)
def load_clip(model_id: str, revision: str, device: torch.device, model_dir: str | None = None):
    """Download (first time) and load CLIP, then freeze it.

    model_dir: optional local copy of the same weights (e.g. from scripts/convert_openai_clip_to_hf.py
    on a machine without Hugging Face access); only the image processor is needed then.
    """
    if model_dir:
        model = CLIPModel.from_pretrained(model_dir)
        processor = AutoImageProcessor.from_pretrained(model_dir)
    else:
        model = CLIPModel.from_pretrained(model_id, revision=revision)
        processor = CLIPProcessor.from_pretrained(model_id, revision=revision)
    model.eval().to(device)                 # eval(): inference mode; to(device): move to GPU
    for p in model.parameters():
        p.requires_grad_(False)             # frozen: no parameter will ever be updated
    return model, processor


@torch.no_grad()
def extract_image_split(split, model, processor, device, batch_size, data_dir, feature, limit=None):
    df = load_split(split, data_dir)
    if limit:
        df = df.head(limit)
    paths = df["image_abspath"].tolist()

    chunks = []
    for start in tqdm(range(0, len(paths), batch_size), desc=f"{split:>10}"):
        images = [load_image(p) for p in paths[start:start + batch_size]]
        inputs = processor(images=images, return_tensors="pt")           # [B, 3, 224, 224]
        out = model.vision_model(pixel_values=inputs["pixel_values"].to(device))
        vec = out.pooler_output                                           # [B, 768]
        if feature == "image_embeds":       # optional 512-d projected version
            vec = model.visual_projection(vec)
        chunks.append(vec.float().cpu())    # back to CPU memory so the GPU stays free

    features = torch.cat(chunks)            # stack all batches -> [N, 768]
    return {
        "sample_id": df["sample_id"].tolist(),
        "features": features,
        "labels": torch.tensor(df["sarcasm_binary"].to_numpy(), dtype=torch.long),
    }


def run_image(args, parser):
    if len(set(args.splits)) != len(args.splits):
        parser.error("splits must not contain duplicates")
    default_dir = f"{FEATURES_ROOT}/clip_vitb32_pooler"
    if args.limit is not None:
        default_dir += f"_smoke_{args.limit}"
    out_dir = Path(args.out_dir or default_dir)
    require_new_paths(out_dir)  # fail before loading CLIP or touching cached files

    cfg = load_config(args.config)
    revision = args.revision or cfg["clip_revision"]
    set_seed(cfg["seed"])
    device = get_device()
    print(f"device={device}  model={cfg['clip_model_id']}@{revision}  feature={cfg['feature']}")

    model, processor = load_clip(cfg["clip_model_id"], revision, device, args.model_dir)
    weight_file, weight_hash = weights_sha256(args.model_dir) if args.model_dir else (None, None)
    out_dir.mkdir(parents=True, exist_ok=False)

    meta = {
        "clip_model_id": cfg["clip_model_id"],
        "clip_revision_requested": revision,
        "clip_commit_resolved": getattr(model.config, "_commit_hash", None),
        "clip_weights_source": "local_dir" if args.model_dir else "hub",
        "clip_weights_file": weight_file,
        "clip_weights_sha256": weight_hash,
        "image_processor": type(processor).__name__,
        "feature": cfg["feature"],
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "device": str(device),
        "limit": args.limit,
        "splits": {},
    }
    for split in args.splits:
        t0 = time.time()
        res = extract_image_split(split, model, processor, device, args.batch_size,
                                  args.data_dir, cfg["feature"], args.limit)
        f = res["features"]
        assert len(res["sample_id"]) == f.shape[0] == res["labels"].shape[0]
        assert torch.isfinite(f).all(), f"{split}: NaN/inf in features"
        torch.save(res, out_dir / f"{split}.pt")
        secs = time.time() - t0
        meta["splits"][split] = {"n": f.shape[0], "dim": f.shape[1], "seconds": round(secs, 1),
                                 "sha256": file_sha256(out_dir / f"{split}.pt")}
        print(f"  saved {out_dir / (split + '.pt')}  shape={tuple(f.shape)}  {secs:.1f}s")

    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("done. meta ->", out_dir / "meta.json")


# ----------------------------------------------------------------------------- text (DistilBERT)
def text_source(cfg: dict, model_dir: str | None = None) -> tuple[str, dict]:
    """Hub id + pinned revision by default; a local copy of the same weights when offline."""
    if model_dir:
        return str(model_dir), {}
    return cfg["text_model_id"], {"revision": cfg["text_revision"]}


def load_text_encoder(cfg: dict, device: torch.device, model_dir: str | None = None):
    """Tokenizer + frozen DistilBertModel in eval mode (MLM-head weights are not loaded)."""
    source, kwargs = text_source(cfg, model_dir)
    tokenizer = AutoTokenizer.from_pretrained(source, **kwargs)
    model = AutoModel.from_pretrained(source, **kwargs)
    model.eval().to(device)
    model.requires_grad_(False)
    return tokenizer, model


def pool_text_features(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor,
                       how: str) -> torch.Tensor:
    if how == "first_token":
        return last_hidden_state[:, 0, :]
    if how == "mean_pooling":
        mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
        return (last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
    raise ValueError(f"unknown text feature {how!r}; expected one of {TEXT_FEATURES}")


def weights_sha256(model_dir):
    """Hash of the local weight file, recorded so an offline copy can be traced."""
    for name in ("model.safetensors", "pytorch_model.bin"):
        path = Path(model_dir) / name
        if path.is_file():
            with path.open("rb") as stream:
                return name, hashlib.file_digest(stream, "sha256").hexdigest()
    return None, None


@torch.no_grad()
def extract_text_split(split, tokenizer, model, device, batch_size, max_length, data_dir, limit=None):
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


def run_text(args, parser):
    if len(set(args.splits)) != len(args.splits):
        parser.error("splits must not contain duplicates")
    prefix = args.out_prefix or f"{FEATURES_ROOT}/distilbert"
    if args.limit is not None and args.out_prefix is None:
        prefix += f"_smoke_{args.limit}"
    out_dirs = {how: Path(f"{prefix}_{how}") for how in TEXT_FEATURES}
    require_new_paths(*out_dirs.values())  # fail before loading DistilBERT

    cfg = load_config(args.config)
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
        results, stats = extract_text_split(split, tokenizer, model, device, args.batch_size,
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


# ----------------------------------------------------------------------------- CLI
def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="modality", required=True)

    img = sub.add_parser("image", help="frozen CLIP ViT-B/32 image features")
    img.add_argument("--config", default="configs/train/image_baseline.yaml")
    img.add_argument("--data_dir", default=None)
    img.add_argument("--out_dir", default=None,
                     help=f"must be a new directory; default {FEATURES_ROOT}/clip_vitb32_pooler")
    img.add_argument("--splits", nargs="+", choices=SPLITS, default=["train", "validation"])
    img.add_argument("--batch_size", type=int, default=64)
    img.add_argument("--revision", default=None, help="override the revision in the config")
    img.add_argument("--limit", type=int, default=None, help="only the first N rows (smoke test)")
    img.add_argument("--model_dir", default=None,
                     help="local copy of openai/clip-vit-base-patch32 (offline machines)")

    txt = sub.add_parser("text", help="frozen DistilBERT text features ([CLS] and mean pooling)")
    txt.add_argument("--config", default="configs/train/multimodal_baseline.yaml")
    txt.add_argument("--data_dir", default=None)
    txt.add_argument("--model_dir", default=None,
                     help="local copy of distilbert-base-uncased; default loads the pinned Hub revision")
    txt.add_argument("--out_prefix", default=None,
                     help="caches go to <out_prefix>_first_token and <out_prefix>_mean_pooling "
                          f"(must be new); default {FEATURES_ROOT}/distilbert")
    txt.add_argument("--splits", nargs="+", choices=SPLITS, default=list(SPLITS))
    txt.add_argument("--batch_size", type=int, default=32)
    txt.add_argument("--limit", type=int, default=None, help="only the first N rows (smoke test)")
    return ap


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.batch_size <= 0 or (args.limit is not None and args.limit <= 0):
        parser.error("batch_size and limit must be positive")
    (run_image if args.modality == "image" else run_text)(args, parser)


if __name__ == "__main__":
    main()
