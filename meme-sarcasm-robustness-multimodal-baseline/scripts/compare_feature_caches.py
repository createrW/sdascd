"""Compare two feature caches of the same split row by row (same sample IDs, same order).

Used to check that features extracted on another machine (or from a local copy of the pinned
weights) match a reference cache, e.g. CPU vs the image group's CUDA cache:

    python -m scripts.compare_feature_caches data/processed/features/20261006_code_fixes_validation_v2 \
        data/processed/features/clip_vitb32_pooler_cpu --splits train validation

Prints max/mean absolute difference and the minimum cosine similarity per split; exits nonzero
if sample IDs, labels or shapes differ, or if the minimum cosine is below --min_cosine.
"""
import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F


def compare(reference_dir, candidate_dir, split):
    ref = torch.load(Path(reference_dir) / f"{split}.pt", map_location="cpu", weights_only=True)
    cand = torch.load(Path(candidate_dir) / f"{split}.pt", map_location="cpu", weights_only=True)
    if ref["sample_id"] != cand["sample_id"]:
        raise ValueError(f"{split}: sample IDs or order differ")
    if not torch.equal(ref["labels"], cand["labels"]):
        raise ValueError(f"{split}: labels differ")
    a, b = ref["features"].float(), cand["features"].float()
    if a.shape != b.shape:
        raise ValueError(f"{split}: shapes {tuple(a.shape)} vs {tuple(b.shape)}")
    diff = (a - b).abs()
    cos = F.cosine_similarity(a, b, dim=1)
    return {"n": a.shape[0], "max_abs": diff.max().item(), "mean_abs": diff.mean().item(),
            "mean_abs_reference": a.abs().mean().item(), "min_cosine": cos.min().item(),
            "mean_cosine": cos.mean().item()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("reference_dir")
    ap.add_argument("candidate_dir")
    ap.add_argument("--splits", nargs="+", default=["train", "validation"])
    ap.add_argument("--min_cosine", type=float, default=0.999)
    args = ap.parse_args()
    ok = True
    for split in args.splits:
        r = compare(args.reference_dir, args.candidate_dir, split)
        print(f"{split}: n={r['n']}  max_abs={r['max_abs']:.3e}  mean_abs={r['mean_abs']:.3e} "
              f"(reference mean |x| {r['mean_abs_reference']:.3f})  "
              f"min_cos={r['min_cosine']:.6f}  mean_cos={r['mean_cosine']:.6f}")
        ok &= r["min_cosine"] >= args.min_cosine
    print("FEATURES MATCH" if ok else "FEATURES DIFFER")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
