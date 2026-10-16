"""Train the multimodal fusion MLP on cached frozen features, keep the best epoch by validation
Macro-F1, and write every output README section 6 asks for.

    frozen DistilBERT [CLS] (768) --L2--\
                                         concat (1536) -> Linear 512 -> ReLU -> Dropout -> Linear 2
    frozen CLIP pooler_output (768) -L2-/

Run the two feature extractors first (src/text/extract_features.py, src/image/extract_features.py,
or the image group's cache zip). Then, from the repo root:
    python -m src.multimodal.train
    python -m src.multimodal.train --run_id 20261016_distilbert_clip_seed42
    python -m src.multimodal.train --config configs/ablations/text_only.json --run_id ...

Writes
    outputs/multimodal/<run_id>/config.json
    outputs/multimodal/<run_id>/metrics_validation.json
    outputs/multimodal/<run_id>/predictions_validation.csv
    outputs/multimodal/<run_id>/confusion_matrix_validation.png
    outputs/multimodal/<run_id>/training_log.csv
    outputs/multimodal/<run_id>/README.md
    checkpoints/multimodal/<run_id>/best.pt      (NOT for GitHub -> copy to Google Drive)
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

from src.common.metrics import (class_weights_from_labels, compute_metrics,  # noqa: E402
                                get_confusion_matrix, majority_baseline, metrics_json,
                                predictions_frame, save_confusion_matrix_png, save_json)
from src.common.reproducibility import get_device, set_seed  # noqa: E402
from src.image.safety import require_new_paths, validate_run_id  # noqa: E402
from src.multimodal.features import (DEFAULT_IMAGE_FEATURES_DIR,  # noqa: E402
                                     default_text_features_dir, load_split_features)
from src.multimodal.model import build_fusion_head  # noqa: E402


def as_inputs(head, tensors):
    """Positional cached tensors (in head.modalities order) -> keyword inputs for FusionHead."""
    return dict(zip(head.modalities, tensors))


@torch.no_grad()
def predict(head, features, device, batch_size=1024):
    """Return P(class 1) for every row."""
    head.eval()
    xs = [features[m] for m in head.modalities]
    probs = []
    for i in range(0, len(xs[0]), batch_size):
        logits = head(**as_inputs(head, [x[i:i + batch_size].to(device) for x in xs]))
        probs.append(torch.softmax(logits, dim=1)[:, 1].cpu())
    return torch.cat(probs)


@torch.no_grad()
def eval_loss(head, features, y, loss_fn, device):
    head.eval()
    logits = head(**as_inputs(head, [features[m].to(device) for m in head.modalities]))
    return loss_fn(logits, y.to(device)).item()


def feature_meta(directory):
    path = Path(directory) / "meta.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/multimodal_baseline.json")
    ap.add_argument("--text_features_dir", default=None,
                    help="default: features/distilbert_<text_feature>")
    ap.add_argument("--image_features_dir", default=DEFAULT_IMAGE_FEATURES_DIR)
    ap.add_argument("--data_dir", default=None, help="dataset metadata for cache validation")
    ap.add_argument("--run_id", default=None, help="default: YYYYMMDD_<model_name>_seed<seed>")
    ap.add_argument("--seed", type=int, default=None, help="override the config seed")
    args = ap.parse_args()

    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    if args.seed is not None:
        cfg["seed"] = args.seed
    seed, threshold = cfg["seed"], cfg["threshold"]
    model_name = cfg["model_name"]
    run_id = args.run_id or f"{dt.date.today():%Y%m%d}_{model_name}_seed{seed}"
    validate_run_id(run_id)
    out_dir = Path("outputs/multimodal") / run_id
    ckpt_dir = Path("checkpoints/multimodal") / run_id
    require_new_paths(out_dir, ckpt_dir)

    set_seed(seed)
    device = get_device()
    modalities = cfg.get("modalities", ["text", "image"])
    text_dir = Path(args.text_features_dir or default_text_features_dir(cfg))
    image_dir = Path(args.image_features_dir)

    # ---------------- data: cached frozen-encoder vectors, joined on sample_id ----------------
    _, f_train, y_train = load_split_features(cfg, "train", text_dir, image_dir, args.data_dir)
    val_ids, f_val, y_val = load_split_features(cfg, "validation", text_dir, image_dir,
                                                args.data_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=False)
    try:
        out_dir.mkdir(parents=True, exist_ok=False)
    except OSError:
        ckpt_dir.rmdir()  # remove only this invocation's empty reservation
        raise
    shapes = {m: tuple(f_train[m].shape) for m in f_train}
    print(f"device={device}  modalities={modalities}  train={shapes}  validation={len(val_ids)}")

    weights = class_weights_from_labels(y_train.numpy())  # from TRAIN only (README 5.2)
    print(f"class weights (train only): w0={weights[0]:.3f}  w1={weights[1]:.3f}")

    head = build_fusion_head(cfg).to(device)
    g = torch.Generator().manual_seed(seed)  # reproducible shuffling
    train_loader = DataLoader(TensorDataset(*[f_train[m] for m in head.modalities], y_train),
                              batch_size=cfg["batch_size"], shuffle=True, generator=g)
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))
    optimizer = torch.optim.AdamW(head.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    n_params = sum(p.numel() for p in head.parameters() if p.requires_grad)
    print(f"fusion head: input {head.input_dim} -> {cfg['fusion']['hidden_dim']} -> "
          f"{cfg['fusion']['num_classes']}  trainable parameters {n_params:,}")

    # ---------------- training loop ----------------
    log, best = [], {"macro_f1": -1.0}
    t0 = time.time()
    for epoch in range(1, cfg["epochs"] + 1):
        head.train()
        running, n_seen = 0.0, 0
        for *xs, yb in train_loader:
            yb = yb.to(device)
            logits = head(**as_inputs(head, [x.to(device) for x in xs]))
            loss = loss_fn(logits, yb)            # weighted cross-entropy
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
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
        if m["macro_f1"] > best["macro_f1"]:     # keep the best epoch (README 5.1)
            best = {**m, "epoch": epoch, "val_loss": val_loss, "prob": prob_val.clone(),
                    "pred": pred_val.clone()}
            torch.save({"state_dict": head.state_dict(), "config": cfg, "epoch": epoch,
                        "val_macro_f1": m["macro_f1"], "class_weights": weights,
                        "modalities": list(head.modalities),
                        "text_features_dir": str(text_dir) if "text" in head.modalities else None,
                        "image_features_dir": str(image_dir) if "image" in head.modalities else None},
                       ckpt_dir / "best.pt")
            flag = "  <- best so far, saved"
        print(f"epoch {epoch:>3}  train_loss {running / n_seen:.4f}  val_loss {val_loss:.4f}  "
              f"val_macroF1 {m['macro_f1']:.4f}  val_acc {m['accuracy']:.4f}{flag}")
    train_secs = time.time() - t0

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
    save_json({**cfg, "run_id": run_id,
               "text_features_dir": str(text_dir) if "text" in modalities else None,
               "image_features_dir": str(image_dir) if "image" in modalities else None,
               "text_feature_meta": feature_meta(text_dir) if "text" in modalities else None,
               "image_feature_meta": feature_meta(image_dir) if "image" in modalities else None,
               "class_weights": weights, "trainable_parameters": n_params,
               "device": str(device), "python": platform.python_version(),
               "torch": torch.__version__, "transformers": transformers.__version__,
               "torch_threads": torch.get_num_threads(), "best_epoch": best["epoch"],
               "checkpoint": str(ckpt_dir / "best.pt"),
               "command": "python -m src.multimodal.train " + " ".join(sys.argv[1:])},
              out_dir / "config.json")

    maj = majority_baseline(y_val.numpy())
    text_arg = f' --text_features_dir "{text_dir}"' if "text" in modalities else ""
    image_arg = f' --image_features_dir "{image_dir}"' if "image" in modalities else ""
    (out_dir / "README.md").write_text(
        f"# {run_id}\n\n"
        f"{model_name}: modalities {' + '.join(head.modalities)}, frozen encoders, cached features "
        f"(text `{cfg['text_feature']}`, image `{cfg['image_feature']}`), "
        f"{'L2-normalized, ' if cfg['normalize_embeddings'] else ''}"
        f"MLP {head.input_dim} -> {cfg['fusion']['hidden_dim']} -> {cfg['fusion']['num_classes']}, "
        f"seed {seed}, device {device}.\n\n"
        "## Reproduce\n\n```\n"
        "# Set MEMOTION_DATA_DIR and choose an unused run_id.\n"
        f'python -m src.multimodal.train --config "{args.config}"{text_arg}{image_arg} '
        f"--seed {seed} --run_id {run_id}_reproduction\n"
        "python -m scripts.verify_predictions "
        f"outputs/multimodal/{run_id}_reproduction/predictions_validation.csv\n"
        f"python -m src.multimodal.evaluate --run_id {run_id}   # reload the saved head\n```\n\n"
        "## Validation (best epoch by Macro-F1)\n\n"
        "| model | Macro-F1 | Accuracy | F1 class 0 | F1 class 1 | best epoch | train time |\n"
        "|---|---:|---:|---:|---:|---:|---:|\n"
        f"| majority (always 1) | {maj['macro_f1']:.4f} | {maj['accuracy']:.4f} | "
        f"{maj['f1_0']:.4f} | {maj['f1_1']:.4f} | - | - |\n"
        f"| {model_name} | {best['macro_f1']:.4f} | {best['accuracy']:.4f} | {best['f1_0']:.4f} | "
        f"{best['f1_1']:.4f} | {best['epoch']} | {train_secs:.1f}s |\n\n"
        f"Confusion matrix (rows true 0/1, cols predicted 0/1): {cm.tolist()}\n\n"
        f"Checkpoint (not in git): `{ckpt_dir / 'best.pt'}`\n", encoding="utf-8")

    print("\n=== best epoch", best["epoch"], "===")
    print(f"validation  Macro-F1 {best['macro_f1']:.4f}  Accuracy {best['accuracy']:.4f}  "
          f"F1_0 {best['f1_0']:.4f}  F1_1 {best['f1_1']:.4f}")
    print(f"majority    Macro-F1 {maj['macro_f1']:.4f}  Accuracy {maj['accuracy']:.4f}")
    print("confusion matrix [rows true 0/1, cols pred 0/1]:", cm.tolist())
    print(f"training time {train_secs:.1f}s")
    print("outputs ->", out_dir, "   checkpoint ->", ckpt_dir / "best.pt")


if __name__ == "__main__":
    main()
