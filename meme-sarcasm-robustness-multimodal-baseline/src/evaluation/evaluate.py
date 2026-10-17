"""Evaluate a saved head checkpoint (image or multimodal) on one split with the shared metrics.

    python -m src.evaluation.evaluate --run_id <run_id>                                  # validation
    python -m src.evaluation.evaluate --run_id 20261016_distilbert_clip_seed42 \
        --out_dir outputs/multimodal/20261016_distilbert_clip_seed42/reeval_mine          # shipped run
    python -m src.evaluation.evaluate --run_id 20261006_code_fixes_validation_v2 --out_dir outputs/image/quick_check \
        --features_dir data/processed/features/20261006_code_fixes_validation_v2         # image head
    python -m src.evaluation.evaluate --run_id 20261016_distilbert_clip_seed42 --split test \
        --image_features_dir data/processed/features/clip_vitb32_pooler_cpu                  # FINAL ONLY

The task is read from where the checkpoint lives (checkpoints/image/ or checkpoints/multimodal/),
or from --task, and must match the checkpoint. Feature caches: command line > the dirs recorded in
the checkpoint (multimodal) or the run's config.json (image) > the config's `features:` block.
Validation output goes to <run>/reeval/ by default; the shipped reference runs already have one,
so pass a new --out_dir for them (existing files are never overwritten). Validation re-evaluation is compared with the run's reported metrics (Macro-F1
tolerance 0.01, docs/decisions.md D11) and exits nonzero outside it. The image group's reference
cache has no test split; pass a cache that has one. Never use test results to choose settings
(README 3.2).
"""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.loader import load_split_features  # noqa: E402
from src.evaluation.metrics import (compute_metrics, get_confusion_matrix, metrics_json,  # noqa: E402
                                    predictions_frame)
from src.evaluation.report import save_confusion_matrix_png, save_json  # noqa: E402
from src.evaluation.reproduction import compare_metrics  # noqa: E402
from src.models.trainer import build_head, predict  # noqa: E402
from src.utils.config import infer_task  # noqa: E402
from src.utils.reproducibility import get_device  # noqa: E402
from src.utils.safety import require_new_paths, validate_run_id  # noqa: E402

TASKS = ("image", "multimodal")


def detect_task(run_id: str, checkpoint=None) -> str:
    """'image' or 'multimodal': from the checkpoint itself when given, else its location
    (checkpoints/<task>/<run_id>/best.pt)."""
    if checkpoint is not None:
        return infer_task(torch.load(checkpoint, map_location="cpu", weights_only=True)["config"])
    found = [t for t in TASKS if Path(f"checkpoints/{t}/{run_id}/best.pt").is_file()]
    if len(found) != 1:
        raise SystemExit(f"cannot tell the task of {run_id!r} (found {found or 'no checkpoint'}); "
                         "pass --task image|multimodal")
    return found[0]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_id", required=True)
    ap.add_argument("--task", choices=TASKS, default=None,
                    help="default: inferred from checkpoints/<task>/<run_id>/")
    ap.add_argument("--split", default="validation", choices=["validation", "test"])
    ap.add_argument("--features_dir", default=None,
                    help="image feature cache (alias of --image_features_dir)")
    ap.add_argument("--image_features_dir", default=None,
                    help="default: the dir recorded for the run (checkpoint or config.json), "
                         "else the config's features.image")
    ap.add_argument("--text_features_dir", default=None, help="default: the dir used for training")
    ap.add_argument("--checkpoint", default=None, help="default: checkpoints/<task>/<run_id>/best.pt")
    ap.add_argument("--out_dir", default=None,
                    help="default: outputs/<task>/<run_id>/reeval (validation) or "
                         "outputs/<task>/<run_id> (test)")
    ap.add_argument("--data_dir", default=None)
    ap.add_argument("--reference_metrics", default=None,
                    help="validation defaults to the reported run metrics")
    ap.add_argument("--tolerance", type=float, default=0.01)
    args = ap.parse_args(argv)
    validate_run_id(args.run_id)
    task = args.task or detect_task(args.run_id, args.checkpoint)

    if args.split == "test":
        print("!! Scoring the TEST split. Do this once, at the end, never to pick settings.")

    ckpt_path = Path(args.checkpoint or f"checkpoints/{task}/{args.run_id}/best.pt")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    actual = infer_task(ckpt["config"])  # inferred for checkpoints saved before `task:` existed
    if actual != task:
        raise SystemExit(f"{ckpt_path}: checkpoint task is '{actual}', not '{task}'; use --task {actual}")
    cfg = {**ckpt["config"], "task": actual}

    run_dir = Path(f"outputs/{task}/{args.run_id}")
    # validation re-evaluation goes to a subfolder so it never overwrites the training run's files
    out_dir = Path(args.out_dir or (run_dir / "reeval" if args.split == "validation" else run_dir))
    tag = args.split
    require_new_paths(*(out_dir / f"{name}_{tag}.{ext}" for name, ext in
                        (("predictions", "csv"), ("metrics", "json"), ("confusion_matrix", "png"))))
    reported = Path(args.reference_metrics) if args.reference_metrics else (
        run_dir / f"metrics_{tag}.json" if tag == "validation" else None)
    if reported is not None and not reported.is_file():
        raise FileNotFoundError(f"Missing reproduction reference: {reported}")

    device = get_device()

    # Feature caches: command line > the dirs recorded for the run > the config's features block.
    record = run_dir / "config.json"
    recorded_image = ckpt.get("image_features_dir") or (
        json.loads(record.read_text(encoding="utf-8")).get("features_dir")
        if task == "image" and record.is_file() else None)
    recorded_text = ckpt.get("text_features_dir")
    recorded_image, recorded_text = (d.replace("\\", "/") if d else d  # runs recorded on Windows
                                     for d in (recorded_image, recorded_text))
    image_dir = args.image_features_dir or args.features_dir or recorded_image
    text_dir = args.text_features_dir or recorded_text
    ids, features, y = load_split_features(cfg, args.split, text_dir, image_dir, args.data_dir,
                                           modalities=ckpt.get("modalities"))
    head = build_head(cfg, features).to(device)
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
