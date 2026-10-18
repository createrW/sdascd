"""Check that the cached-feature pipeline equals the end-to-end model of README 4.3.

Loads a trained fusion head into the serving predictor (src/serving/predictor.py: frozen
DistilBERT + frozen CLIP + the same MLP), runs it on raw captions and raw images for some
validation samples, and compares with
  * the cached text/image feature rows of the same sample_id (alignment + preprocessing check)
  * probability_1 in the run's predictions_validation.csv (training path == inference path).

    python -m src.evaluation.end_to_end --run_id <run_id>
    python -m src.evaluation.end_to_end --run_id 20261016_distilbert_clip_seed42_reproduction_cpu \
        --report outputs/multimodal/20261016_distilbert_clip_seed42_reproduction_cpu/checks/end_to_end_mine.json \
        --text_model_dir hf_local/distilbert-base-uncased \
        --clip_model_dir hf_local/clip-vit-base-patch32          # offline copies of the pinned weights

The shipped reference runs already have checks/end_to_end_validation.json; give them a new --report.

Passes when every predicted label agrees and max |probability difference| <= --tolerance.
The image group's cache was extracted on CUDA (TF32 convolution), so image features differ from
CPU by ~1e-3; this is reported, not hidden.
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.dataset import load_split  # noqa: E402
from src.data.loader import load_split_features  # noqa: E402
from src.evaluation.report import save_json  # noqa: E402
from src.serving.predictor import MultimodalPredictor  # noqa: E402
from src.utils.reproducibility import get_device, set_seed  # noqa: E402
from src.utils.safety import require_new_paths, validate_run_id  # noqa: E402


def pick_rows(frame, n):
    """n rows spread evenly over the split (deterministic)."""
    step = max(1, len(frame) // n)
    return frame.iloc[::step].head(n)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_id", required=True)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--text_model_dir", default=None)
    ap.add_argument("--clip_model_dir", default=None)
    ap.add_argument("--data_dir", default=None)
    ap.add_argument("--tolerance", type=float, default=0.01)
    ap.add_argument("--report", default=None,
                    help="default: outputs/multimodal/<run_id>/checks/end_to_end_validation.json")
    args = ap.parse_args(argv)
    validate_run_id(args.run_id)

    run_dir = Path("outputs/multimodal") / args.run_id
    report_path = Path(args.report or run_dir / "checks" / "end_to_end_validation.json")
    require_new_paths(report_path)
    ckpt_path = Path("checkpoints/multimodal") / args.run_id / "best.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    cfg = ckpt["config"]
    if tuple(ckpt["modalities"]) != ("text", "image"):
        raise ValueError("end-to-end check needs a text+image fusion run")
    set_seed(cfg["seed"])
    device = get_device()

    frame = pick_rows(load_split("validation", args.data_dir), args.n)
    ids = frame["sample_id"].tolist()
    reported = pd.read_csv(run_dir / "predictions_validation.csv",
                           float_precision="round_trip").set_index("sample_id").loc[ids]

    predictor = MultimodalPredictor.from_checkpoint(ckpt_path, args.text_model_dir,
                                                    args.clip_model_dir, str(device))

    outs = {"text_features": [], "image_features": [], "prob": []}
    for start in range(0, len(frame), args.batch_size):
        part = frame.iloc[start:start + args.batch_size]
        out = predictor.forward(part["text_input"].tolist(), part["image_abspath"].tolist())
        outs["text_features"].append(out["text_features"])
        outs["image_features"].append(out["image_features"])
        outs["prob"].append(out["probability_1"])
    outs = {k: torch.cat(v) for k, v in outs.items()}

    _, cached, _ = load_split_features(cfg, "validation", ckpt["text_features_dir"],
                                       ckpt["image_features_dir"], args.data_dir)
    row = {sid: i for i, sid in enumerate(load_split("validation", args.data_dir)["sample_id"])}
    index = torch.tensor([row[s] for s in ids])
    report = {"run_id": args.run_id, "n": len(ids), "device": str(device),
              "text_features_dir": ckpt["text_features_dir"],
              "image_features_dir": ckpt["image_features_dir"],
              # local copies are recorded by folder name only (no machine-specific paths)
              "text_model": (f"local_dir:{Path(args.text_model_dir).name}" if args.text_model_dir
                             else f"{cfg['text_model_id']}@{cfg['text_revision']}"),
              "clip_model": (f"local_dir:{Path(args.clip_model_dir).name}" if args.clip_model_dir
                             else f"{cfg['clip_model_id']}@{cfg['clip_revision']}")}
    for name, key in (("text", "text_features"), ("image", "image_features")):
        ref = F.normalize(cached[name][index], dim=-1)
        diff = (outs[key] - ref).abs()
        report[f"{name}_max_abs_diff"] = diff.max().item()
        report[f"{name}_min_cosine"] = F.cosine_similarity(outs[key], ref, dim=1).min().item()
    prob_ref = torch.tensor(reported["probability_1"].to_numpy(), dtype=torch.float32)
    pred_now = (outs["prob"] >= cfg["threshold"]).long()
    pred_ref = torch.tensor(reported["predicted_label"].to_numpy())
    report["probability_max_abs_diff"] = (outs["prob"] - prob_ref).abs().max().item()
    report["label_agreement"] = (pred_now == pred_ref).float().mean().item()
    report["passed"] = bool(report["label_agreement"] == 1.0
                            and report["probability_max_abs_diff"] <= args.tolerance)

    report_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(report, report_path)
    print(json.dumps(report, indent=2))
    print("END-TO-END CHECK OK" if report["passed"] else "END-TO-END CHECK FAILED")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
