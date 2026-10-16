"""Evaluate a saved image-head checkpoint on one split with the shared metrics.

    python -m src.image.evaluate --run_id 20261005_clip_vitb32_seed42                  # validation
    python -m src.image.evaluate --run_id 20261005_clip_vitb32_seed42 --split test     # FINAL ONLY

Use this to check a teammate's checkpoint reproduces the reported numbers, or (once, at the very
end) to score the test split. Never use test results to choose settings (README 3.2).
"""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.common.metrics import (compute_metrics, get_confusion_matrix, metrics_json,  # noqa: E402
                                predictions_frame, save_confusion_matrix_png, save_json)
from src.common.reproducibility import get_device  # noqa: E402
from src.image.model import ImageHead  # noqa: E402
from src.image.train import predict  # noqa: E402
from src.image.cache import load_features  # noqa: E402
from src.image.safety import require_new_paths, validate_run_id  # noqa: E402
from src.common.reproduction import compare_metrics  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_id", required=True)
    ap.add_argument("--split", default="validation", choices=["validation", "test"])
    ap.add_argument("--features_dir", default="features/clip_vitb32_pooler")
    ap.add_argument("--checkpoint", default=None, help="default: checkpoints/image/<run_id>/best.pt")
    ap.add_argument("--out_dir", default=None,
                    help="default: outputs/image/<run_id>/reeval (validation) or "
                         "outputs/image/<run_id> (test)")
    ap.add_argument("--data_dir", default=None)
    ap.add_argument("--reference_metrics", default=None, help="validation defaults to the reported run metrics")
    ap.add_argument("--tolerance", type=float, default=0.01)
    args = ap.parse_args()
    validate_run_id(args.run_id)

    if args.split == "test":
        print("!! Scoring the TEST split. Do this once, at the end, never to pick settings.")

    run_dir = Path(f"outputs/image/{args.run_id}")
    ckpt_path = Path(args.checkpoint or f"checkpoints/image/{args.run_id}/best.pt")
    # validation re-evaluation goes to a subfolder so it never overwrites the training run's files
    out_dir = Path(args.out_dir or (run_dir / "reeval" if args.split == "validation" else run_dir))
    tag = args.split
    require_new_paths(*(out_dir / f"{name}_{tag}.{ext}" for name, ext in
                        (("predictions", "csv"), ("metrics", "json"), ("confusion_matrix", "png"))))
    reported = Path(args.reference_metrics) if args.reference_metrics else (
        run_dir / f"metrics_{tag}.json" if tag == "validation" else None)
    if reported is not None and not reported.is_file():
        raise FileNotFoundError(f"Missing reproduction reference: {reported}")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = ckpt["config"]
    device = get_device()

    ids, x, y = load_features(Path(args.features_dir), args.split, cfg, args.data_dir)
    head = ImageHead(in_dim=x.shape[1], hidden_dim=cfg["head"]["hidden_dim"],
                     dropout=cfg["head"]["dropout"]).to(device)
    head.load_state_dict(ckpt["state_dict"])

    prob = predict(head, x, device)
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
