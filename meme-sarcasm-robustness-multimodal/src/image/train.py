"""Step 5: train the image head on cached CLIP features, keep the best epoch by validation Macro-F1,
and write every output README section 6 asks for.

Run step 4 first (src/image/extract_features.py). Then, from the repo root:
    python -m src.image.train
    python -m src.image.train --config configs/image_baseline.json --run_id 20261005_clip_vitb32_seed42

Writes
    outputs/image/<run_id>/config.json
    outputs/image/<run_id>/metrics_validation.json
    outputs/image/<run_id>/predictions_validation.csv
    outputs/image/<run_id>/confusion_matrix_validation.png
    outputs/image/<run_id>/training_log.csv
    outputs/image/<run_id>/README.md
    checkpoints/image/<run_id>/best.pt         (NOT for GitHub -> copy to Google Drive)
"""
import argparse
import datetime as dt
import json
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
from src.image.model import ImageHead  # noqa: E402
from src.image.cache import load_features  # noqa: E402
from src.image.safety import require_new_paths, validate_run_id  # noqa: E402


@torch.no_grad()
def predict(head, x, device, batch_size=1024):
    """Return P(class 1) for every row of x."""
    head.eval()
    probs = []
    for i in range(0, len(x), batch_size):
        logits = head(x[i:i + batch_size].to(device))
        probs.append(torch.softmax(logits, dim=1)[:, 1].cpu())   # softmax -> probabilities
    return torch.cat(probs)


@torch.no_grad()
def eval_loss(head, x, y, loss_fn, device):
    head.eval()
    return loss_fn(head(x.to(device)), y.to(device)).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/image_baseline.json")
    ap.add_argument("--features_dir", default="features/clip_vitb32_pooler")
    ap.add_argument("--data_dir", default=None, help="dataset metadata for cache validation")
    ap.add_argument("--run_id", default=None, help="default: YYYYMMDD_clip_vitb32_seed<seed>")
    args = ap.parse_args()

    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    seed, threshold = cfg["seed"], cfg["threshold"]
    model_name = cfg["model_name"]
    run_id = args.run_id or f"{dt.date.today():%Y%m%d}_clip_vitb32_seed{seed}"
    validate_run_id(run_id)
    out_dir = Path("outputs/image") / run_id
    ckpt_dir = Path("checkpoints/image") / run_id
    require_new_paths(out_dir, ckpt_dir)

    set_seed(seed)
    device = get_device()
    features_dir = Path(args.features_dir)

    # ---------------- data: cached CLIP vectors ----------------
    _, x_train, y_train = load_features(features_dir, "train", cfg, args.data_dir)
    val_ids, x_val, y_val = load_features(features_dir, "validation", cfg, args.data_dir)
    # Exclusive creation also protects simultaneous commands with the same ID.
    ckpt_dir.mkdir(parents=True, exist_ok=False)
    try:
        out_dir.mkdir(parents=True, exist_ok=False)
    except OSError:
        ckpt_dir.rmdir()  # remove only this invocation's empty reservation
        raise
    print(f"device={device}  train={tuple(x_train.shape)}  validation={tuple(x_val.shape)}")

    weights = class_weights_from_labels(y_train.numpy())       # from TRAIN only
    print(f"class weights (train only): w0={weights[0]:.3f}  w1={weights[1]:.3f}")

    g = torch.Generator().manual_seed(seed)                     # reproducible shuffling
    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=cfg["batch_size"],
                              shuffle=True, generator=g)

    # ---------------- model, loss, optimizer ----------------
    head = ImageHead(in_dim=x_train.shape[1], hidden_dim=cfg["head"]["hidden_dim"],
                     dropout=cfg["head"]["dropout"]).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))
    optimizer = torch.optim.AdamW(head.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])

    # ---------------- training loop ----------------
    log, best = [], {"macro_f1": -1.0}
    t0 = time.time()
    for epoch in range(1, cfg["epochs"] + 1):
        head.train()
        running, n_seen = 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = head(xb)                 # 1. guess
            loss = loss_fn(logits, yb)        # 2. how wrong (weighted cross-entropy)
            optimizer.zero_grad()             #    clear last batch's gradients
            loss.backward()                   # 3. backprop: which way should each weight move
            optimizer.step()                  # 4. move every weight a small step
            running += loss.item() * len(yb)
            n_seen += len(yb)

        # ---- evaluate on validation after every epoch ----
        prob_val = predict(head, x_val, device)
        pred_val = (prob_val >= threshold).long()
        m = compute_metrics(y_val.numpy(), pred_val.numpy())
        val_loss = eval_loss(head, x_val, y_val, loss_fn, device)
        log.append({"epoch": epoch, "train_loss": round(running / n_seen, 6),
                    "val_loss": round(val_loss, 6), "val_macro_f1": round(m["macro_f1"], 6),
                    "val_accuracy": round(m["accuracy"], 6)})
        flag = ""
        if m["macro_f1"] > best["macro_f1"]:  # keep the best epoch (= early stopping)
            best = {**m, "epoch": epoch, "val_loss": val_loss, "prob": prob_val.clone(),
                    "pred": pred_val.clone()}
            torch.save({"state_dict": head.state_dict(), "config": cfg, "epoch": epoch,
                        "val_macro_f1": m["macro_f1"], "class_weights": weights},
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

    try:
        import transformers  # only for recording the version
        tf_version = transformers.__version__
    except ImportError:
        tf_version = None
    meta = json.loads((features_dir / "meta.json").read_text(encoding="utf-8")) \
        if (features_dir / "meta.json").exists() else {}
    save_json({**cfg, "run_id": run_id, "features_dir": str(features_dir),
               "feature_meta": meta, "class_weights": weights, "device": str(device),
               "torch": torch.__version__, "transformers": tf_version,
               "best_epoch": best["epoch"], "checkpoint": str(ckpt_dir / "best.pt"),
               "command": "python -m src.image.train " + " ".join(sys.argv[1:])},
              out_dir / "config.json")

    maj = majority_baseline(y_val.numpy())
    (out_dir / "README.md").write_text(
        f"# {run_id}\n\n"
        f"Image-only baseline: frozen CLIP ViT-B/32 ({cfg['clip_revision']}) + "
        f"{'linear head' if not cfg['head']['hidden_dim'] else 'MLP head'}, seed {seed}.\n\n"
        "## Reproduce\n\n```\n"
        "# Set MEMOTION_DATA_DIR and choose unused run/cache names.\n"
        f'python -m src.image.extract_features --config "{args.config}" '
        f'--splits train validation --out_dir "features/{run_id}_reproduction"\n'
        f'python -m src.image.train --config "{args.config}" '
        f'--features_dir "features/{run_id}_reproduction" --run_id {run_id}_reproduction\n'
        "python -m scripts.verify_predictions "
        f"outputs/image/{run_id}_reproduction/predictions_validation.csv\n```\n\n"
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
    print("outputs ->", out_dir, "   checkpoint ->", ckpt_dir / "best.pt")


if __name__ == "__main__":
    main()
