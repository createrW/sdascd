"""Evaluate a saved multimodal fusion head on one split with the shared metrics.

    python -m src.multimodal.evaluate --run_id 20261016_distilbert_clip_seed42                 # validation
    python -m src.multimodal.evaluate --run_id 20261016_distilbert_clip_seed42 --split test \
        --image_features_dir features/clip_vitb32_pooler_cpu                                 # FINAL ONLY

Validation re-evaluation is compared with the run's reported metrics (Macro-F1 tolerance 0.01,
docs/decisions.md D11) and exits nonzero outside it. The image group's cache has no test split;
pass a cache that has one. Never use test results to choose settings (README 3.2).
"""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.common.metrics import (compute_metrics, get_confusion_matrix, metrics_json,  # noqa: E402
                                predictions_frame, save_confusion_matrix_png, save_json)
from src.common.reproducibility import get_device  # noqa: E402
from src.common.reproduction import compare_metrics  # noqa: E402
from src.image.safety import require_new_paths, validate_run_id  # noqa: E402
from src.multimodal.features import load_split_features  # noqa: E402
from src.multimodal.model import build_fusion_head  # noqa: E402
from src.multimodal.train import predict  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_id", required=True)
    ap.add_argument("--split", default="validation", choices=["validation", "test"])
    ap.add_argument("--text_features_dir", default=None, help="default: the dir used for training")
    ap.add_argument("--image_features_dir", default=None, help="default: the dir used for training")
    ap.add_argument("--checkpoint", default=None,
                    help="default: checkpoints/multimodal/<run_id>/best.pt")
    ap.add_argument("--out_dir", default=None,
                    help="default: outputs/multimodal/<run_id>/reeval (validation) or "
                         "outputs/multimodal/<run_id> (test)")
    ap.add_argument("--data_dir", default=None)
    ap.add_argument("--reference_metrics", default=None,
                    help="validation defaults to the reported run metrics")
    ap.add_argument("--tolerance", type=float, default=0.01)
    args = ap.parse_args()
    validate_run_id(args.run_id)

    if args.split == "test":
        print("!! Scoring the TEST split. Do this once, at the end, never to pick settings.")

    run_dir = Path(f"outputs/multimodal/{args.run_id}")
    ckpt_path = Path(args.checkpoint or f"checkpoints/multimodal/{args.run_id}/best.pt")
    out_dir = Path(args.out_dir or (run_dir / "reeval" if args.split == "validation" else run_dir))
    tag = args.split
    require_new_paths(*(out_dir / f"{name}_{tag}.{ext}" for name, ext in
                        (("predictions", "csv"), ("metrics", "json"), ("confusion_matrix", "png"))))
    reported = Path(args.reference_metrics) if args.reference_metrics else (
        run_dir / f"metrics_{tag}.json" if tag == "validation" else None)
    if reported is not None and not reported.is_file():
        raise FileNotFoundError(f"Missing reproduction reference: {reported}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    cfg = ckpt["config"]
    device = get_device()

    text_dir = args.text_features_dir or ckpt.get("text_features_dir")
    image_dir = args.image_features_dir or ckpt.get("image_features_dir")
    ids, features, y = load_split_features(cfg, args.split, text_dir, image_dir, args.data_dir)
    head = build_fusion_head(cfg).to(device)
    head.load_state_dict(ckpt["state_dict"])

    prob = predict(head, features, device)
    pred = (prob >= cfg["threshold"]).long()
    m = compute_metrics(y.numpy(), pred.numpy())
    cm = get_confusion_matrix(y.numpy(), pred.numpy())

    candidate = metrics_json(cfg["model_name"], tag, cfg["seed"], cfg["threshold"], m, 0.0,
                             checkpoint_epoch=ckpt["epoch"])
    passed = compare_metrics(reported, candidate, args.tolerance) if reported is not None else True
    out_dir.mkdir(parents=True, exist_ok=True)
    predictions_frame(ids, y.numpy(), pred.numpy(), prob.numpy(), cfg["model_name"], tag,
                      cfg["seed"]).to_csv(out_dir / f"predictions_{tag}.csv", index=False)
    save_json(candidate, out_dir / f"metrics_{tag}.json")
    save_confusion_matrix_png(cm, out_dir / f"confusion_matrix_{tag}.png",
                              f"{cfg['model_name']} | {tag} | epoch {ckpt['epoch']}")

    print(f"{tag}: Macro-F1 {m['macro_f1']:.4f}  Accuracy {m['accuracy']:.4f}  "
          f"F1_0 {m['f1_0']:.4f}  F1_1 {m['f1_1']:.4f}")
    print("confusion matrix [rows true 0/1, cols pred 0/1]:", cm.tolist())
    print("written to", out_dir)

    if not passed:
        print("REPRODUCTION FAILED: Macro-F1 exceeds the allowed tolerance.")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
