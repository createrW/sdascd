"""Training logic for every head on cached frozen-encoder features.

One loop serves both tasks (the config's `task:` key decides):
  image       frozen CLIP ViT-B/32 pooler_output -> ImageHead (linear probe or small MLP)
  multimodal  frozen DistilBERT [CLS] + frozen CLIP, each L2-normalized, concatenated -> fusion MLP

Weighted cross-entropy with train-only class weights (README 5.2), AdamW, seed-controlled shuffling,
validation every epoch, and the best epoch by validation Macro-F1 is kept (README 5.1).

Extract the features first (src/features/feature_engineering.py, or unpack the cache zips into
data/processed/). Then, from the repo root (or with scripts/train.sh):
    python -m src.models.trainer --config configs/train/multimodal_baseline.yaml
    python -m src.models.trainer --config configs/train/image_baseline.yaml --run_id 20261005_clip_vitb32_seed42
    python -m src.models.trainer --config configs/train/multimodal_text_only.yaml --seed 43

Writes
    outputs/<task>/<run_id>/config.json
    outputs/<task>/<run_id>/metrics_validation.json
    outputs/<task>/<run_id>/predictions_validation.csv
    outputs/<task>/<run_id>/confusion_matrix_validation.png
    outputs/<task>/<run_id>/training_log.csv
    outputs/<task>/<run_id>/README.md
    checkpoints/<task>/<run_id>/best.pt       (NOT for GitHub -> copy to Google Drive)
"""
import argparse
import datetime as dt
import json
import platform
import sys
import time
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.loader import features_dirs, load_split_features  # noqa: E402
from src.evaluation.metrics import (class_weights_from_labels, compute_metrics,  # noqa: E402
                                    get_confusion_matrix, majority_baseline, metrics_json,
                                    predictions_frame)
from src.evaluation.report import save_confusion_matrix_png, save_json  # noqa: E402
from src.features.cache import resolve_features_dir  # noqa: E402
from src.models.multimodal_classifier import build_fusion_head  # noqa: E402
from src.models.vit_classifier import build_image_head  # noqa: E402
from src.utils.config import infer_task, load_config  # noqa: E402
from src.utils.reproducibility import get_device, set_seed  # noqa: E402
from src.utils.safety import require_new_paths, validate_run_id  # noqa: E402


def build_head(config: dict, features: dict):
    """The trainable head for this config, sized from the loaded features."""
    if infer_task(config) == "image":
        return build_image_head(config, in_dim=features["image"].shape[1])
    return build_fusion_head(config)


def as_inputs(head, tensors):
    """Positional cached tensors (in head.modalities order) -> keyword inputs for the head."""
    return dict(zip(head.modalities, tensors))


@torch.no_grad()
def predict(head, features, device, batch_size=1024):
    """Return P(class 1) for every row."""
    head.eval()
    xs = [features[m] for m in head.modalities]
    probs = []
    for i in range(0, len(xs[0]), batch_size):
        logits = head(**as_inputs(head, [x[i:i + batch_size].to(device) for x in xs]))
        probs.append(torch.softmax(logits, dim=1)[:, 1].cpu())   # softmax -> probabilities
    return torch.cat(probs)


@torch.no_grad()
def eval_loss(head, features, y, loss_fn, device):
    head.eval()
    logits = head(**as_inputs(head, [features[m].to(device) for m in head.modalities]))
    return loss_fn(logits, y.to(device)).item()


def fit(head, f_train, y_train, f_val, y_val, cfg, device, ckpt_path, ckpt_extra=None):
    """Train for cfg['epochs']; save the best epoch (validation Macro-F1) to ckpt_path.

    Returns (per-epoch log, best-epoch record with probabilities/predictions, seconds, class weights).
    """
    seed, threshold = cfg["seed"], cfg["threshold"]
    weights = class_weights_from_labels(y_train.numpy())         # from TRAIN only
    print(f"class weights (train only): w0={weights[0]:.3f}  w1={weights[1]:.3f}")
    g = torch.Generator().manual_seed(seed)                      # reproducible shuffling
    train_loader = DataLoader(TensorDataset(*[f_train[m] for m in head.modalities], y_train),
                              batch_size=cfg["batch_size"], shuffle=True, generator=g)
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))
    optimizer = torch.optim.AdamW(head.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])

    log, best = [], {"macro_f1": -1.0}
    t0 = time.time()
    for epoch in range(1, cfg["epochs"] + 1):
        head.train()
        running, n_seen = 0.0, 0
        for *xs, yb in train_loader:
            yb = yb.to(device)
            logits = head(**as_inputs(head, [x.to(device) for x in xs]))   # 1. guess
            loss = loss_fn(logits, yb)        # 2. how wrong (weighted cross-entropy)
            optimizer.zero_grad()             #    clear last batch's gradients
            loss.backward()                   # 3. backprop
            optimizer.step()                  # 4. move every weight a small step
            running += loss.item() * len(yb)
            n_seen += len(yb)

        prob_val = predict(head, f_val, device)
        pred_val = (prob_val >= threshold).long()
        m = compute_metrics(y_val.numpy(), pred_val.numpy())
        val_loss = eval_loss(head, f_val, y_val, loss_fn, device)
        log.append({"epoch": epoch, "train_loss": round(running / n_seen, 6),
                    "val_loss": round(val_loss, 6), "val_macro_f1": round(m["macro_f1"], 6),
                    "val_accuracy": round(m["accuracy"], 6)})
        flag = ""
        if m["macro_f1"] > best["macro_f1"]:     # keep the best epoch (= early stopping)
            best = {**m, "epoch": epoch, "val_loss": val_loss, "prob": prob_val.clone(),
                    "pred": pred_val.clone()}
            torch.save({"state_dict": head.state_dict(), "config": cfg, "epoch": epoch,
                        "val_macro_f1": m["macro_f1"], "class_weights": weights,
                        **(ckpt_extra or {})}, ckpt_path)
            flag = "  <- best so far, saved"
        print(f"epoch {epoch:>3}  train_loss {running / n_seen:.4f}  val_loss {val_loss:.4f}  "
              f"val_macroF1 {m['macro_f1']:.4f}  val_acc {m['accuracy']:.4f}{flag}")
    return log, best, time.time() - t0, weights


def feature_meta(directory):
    path = Path(directory) / "meta.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def run_readme(task, run_id, cfg, head, device, best, maj, cm, train_secs, config_path, dirs, ckpt):
    seed, model_name = cfg["seed"], cfg["model_name"]
    if task == "image":
        description = (f"Image-only baseline: frozen CLIP ViT-B/32 ({cfg['clip_revision']}) + "
                       f"{'linear head' if not cfg['head']['hidden_dim'] else 'MLP head'}, seed {seed}.")
        reproduce = ("# Set MEMOTION_DATA_DIR and choose unused run/cache names.\n"
                     f'python -m src.features.feature_engineering image --config "{config_path}" '
                     f'--splits train validation --out_dir "data/processed/features/{run_id}_reproduction"\n'
                     f'python -m src.models.trainer --config "{config_path}" '
                     f'--features_dir "data/processed/features/{run_id}_reproduction" '
                     f"--run_id {run_id}_reproduction\n"
                     "python -m scripts.verify_predictions "
                     f"outputs/image/{run_id}_reproduction/predictions_validation.csv\n")
    else:
        description = (f"{model_name}: modalities {' + '.join(head.modalities)}, frozen encoders, "
                       f"cached features (text `{cfg['text_feature']}`, image `{cfg['image_feature']}`), "
                       f"{'L2-normalized, ' if cfg['normalize_embeddings'] else ''}"
                       f"MLP {head.input_dim} -> {cfg['fusion']['hidden_dim']} -> "
                       f"{cfg['fusion']['num_classes']}, seed {seed}, device {device}.")
        text_arg = f' --text_features_dir "{dirs["text"]}"' if "text" in head.modalities else ""
        image_arg = f' --image_features_dir "{dirs["image"]}"' if "image" in head.modalities else ""
        reproduce = ("# Set MEMOTION_DATA_DIR and choose an unused run_id.\n"
                     f'python -m src.models.trainer --config "{config_path}"{text_arg}{image_arg} '
                     f"--seed {seed} --run_id {run_id}_reproduction\n"
                     "python -m scripts.verify_predictions "
                     f"outputs/multimodal/{run_id}_reproduction/predictions_validation.csv\n"
                     f"python -m src.evaluation.evaluate --run_id {run_id}   # reload the saved head\n")
    return (f"# {run_id}\n\n{description}\n\n## Reproduce\n\n```\n{reproduce}```\n\n"
            "## Validation (best epoch by Macro-F1)\n\n"
            "| model | Macro-F1 | Accuracy | F1 class 0 | F1 class 1 | best epoch | train time |\n"
            "|---|---:|---:|---:|---:|---:|---:|\n"
            f"| majority (always 1) | {maj['macro_f1']:.4f} | {maj['accuracy']:.4f} | "
            f"{maj['f1_0']:.4f} | {maj['f1_1']:.4f} | - | - |\n"
            f"| {model_name} | {best['macro_f1']:.4f} | {best['accuracy']:.4f} | {best['f1_0']:.4f} | "
            f"{best['f1_1']:.4f} | {best['epoch']} | {train_secs:.1f}s |\n\n"
            f"Confusion matrix (rows true 0/1, cols predicted 0/1): {cm.tolist()}\n\n"
            f"Checkpoint (not in git): `{ckpt}`\n")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/train/multimodal_baseline.yaml")
    ap.add_argument("--features_dir", default=None,
                    help="image feature cache (alias of --image_features_dir)")
    ap.add_argument("--image_features_dir", default=None, help="default: the config's features.image")
    ap.add_argument("--text_features_dir", default=None, help="default: the config's features.text")
    ap.add_argument("--data_dir", default=None, help="dataset metadata for cache validation")
    ap.add_argument("--run_id", default=None, help="default: YYYYMMDD_<model>_seed<seed>")
    ap.add_argument("--seed", type=int, default=None, help="override the config seed")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    if args.seed is not None:
        cfg["seed"] = args.seed
    task = infer_task(cfg)
    seed, threshold, model_name = cfg["seed"], cfg["threshold"], cfg["model_name"]
    default_name = "clip_vitb32" if task == "image" else model_name
    run_id = args.run_id or f"{dt.date.today():%Y%m%d}_{default_name}_seed{seed}"
    validate_run_id(run_id)
    out_dir = Path("outputs") / task / run_id
    ckpt_dir = Path("checkpoints") / task / run_id
    require_new_paths(out_dir, ckpt_dir)

    set_seed(seed)
    device = get_device()
    # resolved once, so loading, the checkpoint and config.json all name the folder actually used
    dirs = {k: str(resolve_features_dir(v)) for k, v in
            features_dirs(cfg, args.text_features_dir, args.image_features_dir or args.features_dir).items()}

    # ---------------- data: cached frozen-encoder vectors, joined on sample_id ----------------
    _, f_train, y_train = load_split_features(cfg, "train", dirs.get("text"), dirs["image"],
                                              args.data_dir)
    val_ids, f_val, y_val = load_split_features(cfg, "validation", dirs.get("text"), dirs["image"],
                                                args.data_dir)
    # Exclusive creation also protects simultaneous commands with the same ID.
    ckpt_dir.mkdir(parents=True, exist_ok=False)
    try:
        out_dir.mkdir(parents=True, exist_ok=False)
    except OSError:
        ckpt_dir.rmdir()  # remove only this invocation's empty reservation
        raise
    shapes = {m: tuple(f_train[m].shape) for m in f_train}
    print(f"device={device}  task={task}  train={shapes}  validation={len(val_ids)}")

    head = build_head(cfg, f_train).to(device)
    n_params = sum(p.numel() for p in head.parameters() if p.requires_grad)
    print(f"head: {type(head).__name__} over {', '.join(head.modalities)}  "
          f"trainable parameters {n_params:,}")
    ckpt_extra = {} if task == "image" else {
        "modalities": list(head.modalities),
        "text_features_dir": dirs.get("text") if "text" in head.modalities else None,
        "image_features_dir": dirs["image"] if "image" in head.modalities else None}
    log, best, train_secs, weights = fit(head, f_train, y_train, f_val, y_val, cfg, device,
                                         ckpt_dir / "best.pt", ckpt_extra)

    # ---------------- outputs (README 6) ----------------
    cm = get_confusion_matrix(y_val.numpy(), best["pred"].numpy())
    pd.DataFrame(log).to_csv(out_dir / "training_log.csv", index=False)
    preds = predictions_frame(val_ids, y_val.numpy(), best["pred"].numpy(), best["prob"].numpy(),
                              model_name, "validation", seed)
    preds.to_csv(out_dir / "predictions_validation.csv", index=False)
    save_json(metrics_json(model_name, "validation", seed, threshold, best, train_secs,
                           validation_loss=round(best["val_loss"], 6), best_epoch=best["epoch"]),
              out_dir / "metrics_validation.json")
    save_confusion_matrix_png(cm, out_dir / "confusion_matrix_validation.png",
                              f"{model_name} | validation | epoch {best['epoch']}")

    import transformers  # only for recording the version
    record = {**cfg, "run_id": run_id, "task": task, "class_weights": weights,
              "device": str(device), "python": platform.python_version(), "torch": torch.__version__,
              "transformers": transformers.__version__, "torch_threads": torch.get_num_threads(),
              "trainable_parameters": n_params, "best_epoch": best["epoch"],
              "checkpoint": str(ckpt_dir / "best.pt"),
              "command": "python -m src.models.trainer " + " ".join(sys.argv[1:] if argv is None else argv)}
    if task == "image":
        record.update(features_dir=dirs["image"], feature_meta=feature_meta(dirs["image"]))
    else:
        record.update(
            text_features_dir=dirs.get("text") if "text" in head.modalities else None,
            image_features_dir=dirs["image"] if "image" in head.modalities else None,
            text_feature_meta=feature_meta(dirs["text"]) if "text" in head.modalities else None,
            image_feature_meta=feature_meta(dirs["image"]) if "image" in head.modalities else None)
    save_json(record, out_dir / "config.json")

    maj = majority_baseline(y_val.numpy())
    (out_dir / "README.md").write_text(
        run_readme(task, run_id, cfg, head, device, best, maj, cm, train_secs, args.config, dirs,
                   ckpt_dir / "best.pt"), encoding="utf-8")

    print("\n=== best epoch", best["epoch"], "===")
    print(f"validation  Macro-F1 {best['macro_f1']:.4f}  Accuracy {best['accuracy']:.4f}  "
          f"F1_0 {best['f1_0']:.4f}  F1_1 {best['f1_1']:.4f}")
    print(f"majority    Macro-F1 {maj['macro_f1']:.4f}  Accuracy {maj['accuracy']:.4f}")
    print("confusion matrix [rows true 0/1, cols pred 0/1]:", cm.tolist())
    print(f"training time {train_secs:.1f}s")
    print("outputs ->", out_dir, "   checkpoint ->", ckpt_dir / "best.pt")


if __name__ == "__main__":
    main()
