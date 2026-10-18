"""Check that TF32 explains the gap between CPU and image-group CUDA CLIP features.

PyTorch enables TF32 for CUDA convolutions by default (torch.backends.cudnn.allow_tf32 = True).
This script recomputes CLIP pooler_output on CPU twice — in float32, and with the patch-embedding
convolution's inputs and weights rounded to TF32 (10-bit mantissa) — and compares both with the
image group's CUDA cache.

    python -m scripts.check_tf32_emulation --clip_model_dir hf_local/clip-vit-base-patch32 \
        --out Team_Review/multimodal/report/tf32_emulation.json
"""
import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor, CLIPModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.dataset import load_image, load_split  # noqa: E402
from src.features.cache import resolve_features_dir  # noqa: E402


def round_to_tf32(t):
    """Round float32 to the nearest value with a 10-bit mantissa (TF32)."""
    i = t.float().contiguous().view(torch.int32)
    return ((i + 0x1000) & ~0x1FFF).view(torch.float32)


@torch.no_grad()
def features(model, pixels, tf32):
    conv = model.vision_model.embeddings.patch_embedding
    weight = conv.weight.data.clone()
    if tf32:
        conv.weight.data = round_to_tf32(weight)
        pixels = round_to_tf32(pixels)
    try:
        return model.vision_model(pixel_values=pixels).pooler_output
    finally:
        conv.weight.data = weight


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip_model_dir", default=None, help="default: the pinned Hub revision")
    ap.add_argument("--reference_dir", default="data/processed/features/20261006_code_fixes_validation_v2")
    ap.add_argument("--n_first", type=int, default=64)
    ap.add_argument("--n_worst", type=int, default=10)
    ap.add_argument("--data_dir", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    source = args.clip_model_dir or "openai/clip-vit-base-patch32"
    kwargs = {} if args.clip_model_dir else {"revision": "3d74acf"}
    model = CLIPModel.from_pretrained(source, **kwargs).eval()
    processor = AutoImageProcessor.from_pretrained(source, **kwargs)
    # a local copy is recorded by folder name only (no machine-specific paths)
    report = {"clip_source": f"local_dir:{Path(source).name}" if args.clip_model_dir else source,
              "groups": {}}

    reference_dir = resolve_features_dir(args.reference_dir)
    cpu_dir = resolve_features_dir("data/processed/features/clip_vitb32_pooler_cpu")
    for split in ("validation", "train"):
        ref = torch.load(reference_dir / f"{split}.pt", weights_only=True)
        cpu = torch.load(cpu_dir / f"{split}.pt", weights_only=True)
        assert ref["sample_id"] == cpu["sample_id"]
        cos = F.cosine_similarity(ref["features"].double(), cpu["features"].double())
        frame = load_split(split, args.data_dir)
        groups = {f"{split}_worst{args.n_worst}": torch.argsort(cos)[:args.n_worst].tolist()}
        if split == "validation":
            groups[f"validation_first{args.n_first}"] = list(range(args.n_first))
        for name, rows in groups.items():
            images = [load_image(frame.image_abspath[i]) for i in rows]
            pixels = processor(images=images, return_tensors="pt")["pixel_values"]
            r = ref["features"][rows]
            out = {}
            for mode in ("float32", "tf32_conv"):
                f = features(model, pixels, tf32=(mode == "tf32_conv"))
                c = F.cosine_similarity(f.double(), r.double())
                out[mode] = {"mean_abs_diff": float((f - r).abs().mean()),
                             "max_abs_diff": float((f - r).abs().max()),
                             "min_cosine": float(c.min()), "mean_cosine": float(c.mean())}
            report["groups"][name] = {"n": len(rows), "sample_ids": [frame.sample_id[i] for i in rows],
                                      **out}
            print(f"{name:22s} float32 mean|d| {out['float32']['mean_abs_diff']:.2e} "
                  f"min cos {out['float32']['min_cosine']:.5f}  ->  TF32 conv mean|d| "
                  f"{out['tf32_conv']['mean_abs_diff']:.2e} min cos {out['tf32_conv']['min_cosine']:.6f}")
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
