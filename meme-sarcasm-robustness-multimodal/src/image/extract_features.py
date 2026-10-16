"""Step 4: run the frozen CLIP ViT-B/32 image encoder over every image ONCE and cache the vectors.

Why cache? The encoder is frozen, so an image's vector never changes. Computing it once
means training the head later takes seconds instead of re-running CLIP every epoch.

Usage (from the repo root, with MEMOTION_DATA_DIR set):
    python -m src.image.extract_features
    python -m src.image.extract_features --splits validation      # just one split
    python -m src.image.extract_features --limit 128               # separate smoke cache

Output (git-ignored), one file per split:
    features/clip_vitb32_pooler/<split>.pt  ->  {"sample_id": list[str],
                                                 "features": FloatTensor[N, 768],
                                                 "labels":   LongTensor[N]}
    features/clip_vitb32_pooler/meta.json   ->  model revision, versions, timing
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch
import transformers
from tqdm import tqdm
from transformers import AutoImageProcessor, CLIPModel, CLIPProcessor

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.common.data import SPLITS, load_image, load_split  # noqa: E402
from src.common.reproducibility import get_device, set_seed  # noqa: E402
from src.image.cache import file_sha256  # noqa: E402
from src.image.safety import require_new_paths  # noqa: E402


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


@torch.no_grad()                            # same as the `with torch.no_grad():` you tried in the REPL
def extract_split(split, model, processor, device, batch_size, data_dir, feature, limit=None):
    df = load_split(split, data_dir)
    if limit:
        df = df.head(limit)
    paths = df["image_abspath"].tolist()

    chunks = []
    for start in tqdm(range(0, len(paths), batch_size), desc=f"{split:>10}"):
        images = [load_image(p) for p in paths[start:start + batch_size]]

        # ---- the three lines you typed by hand in the REPL, now on a batch of 64 images ----
        inputs = processor(images=images, return_tensors="pt")           # [B, 3, 224, 224]
        out = model.vision_model(pixel_values=inputs["pixel_values"].to(device))
        vec = out.pooler_output                                           # [B, 768]
        # -------------------------------------------------------------------------------------

        if feature == "image_embeds":       # optional 512-d projected version
            vec = model.visual_projection(vec)
        chunks.append(vec.float().cpu())    # back to CPU memory so the GPU stays free

    features = torch.cat(chunks)            # stack all batches -> [N, 768]
    return {
        "sample_id": df["sample_id"].tolist(),
        "features": features,
        "labels": torch.tensor(df["sarcasm_binary"].to_numpy(), dtype=torch.long),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/image_baseline.json")
    ap.add_argument("--data_dir", default=None)
    ap.add_argument("--out_dir", default=None, help="must be a new directory")
    ap.add_argument("--splits", nargs="+", choices=SPLITS, default=["train", "validation"])
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--revision", default=None, help="override the revision in the config")
    ap.add_argument("--limit", type=int, default=None, help="only the first N rows (smoke test)")
    ap.add_argument("--model_dir", default=None,
                    help="local copy of openai/clip-vit-base-patch32 (offline machines)")
    args = ap.parse_args()
    if args.batch_size <= 0 or (args.limit is not None and args.limit <= 0):
        ap.error("batch_size and limit must be positive")
    if len(set(args.splits)) != len(args.splits):
        ap.error("splits must not contain duplicates")
    default_dir = "features/clip_vitb32_pooler"
    if args.limit is not None:
        default_dir += f"_smoke_{args.limit}"
    out_dir = Path(args.out_dir or default_dir)
    require_new_paths(out_dir)  # fail before loading CLIP or touching cached files

    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    revision = args.revision or cfg["clip_revision"]
    set_seed(cfg["seed"])
    device = get_device()
    print(f"device={device}  model={cfg['clip_model_id']}@{revision}  feature={cfg['feature']}")

    model, processor = load_clip(cfg["clip_model_id"], revision, device, args.model_dir)
    out_dir.mkdir(parents=True, exist_ok=False)

    meta = {
        "clip_model_id": cfg["clip_model_id"],
        "clip_revision_requested": revision,
        "clip_commit_resolved": getattr(model.config, "_commit_hash", None),
        "clip_weights_source": "local_dir" if args.model_dir else "hub",
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
        res = extract_split(split, model, processor, device, args.batch_size,
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


if __name__ == "__main__":
    main()
