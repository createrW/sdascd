"""Overfitting diagnostic for the frozen-feature heads (validation only; Test is never read).

Questions it answers, with one definition of every quantity:
  A. For the recorded runs (baseline, image only, text only; seeds 42-44): at the end of every epoch,
     score the SAME parameters on train and validation in eval() mode with the same loss definition
     (PyTorch weighted cross-entropy over the whole split: sum_i w_{y_i} l_i / sum_i w_{y_i}, train
     class weights), plus unweighted NLL, Macro-F1, ROC-AUC, ECE and confident errors. The logged
     train loss (dropout on, batch averages while parameters move) and the train-mode loss at the
     end-of-epoch parameters are recorded next to it, so the three loss accountings can be compared.
     Each rerun must reproduce the recorded training_log.csv and best.pt exactly, which shows the
     per-epoch scoring does not perturb training.
  B. Capacity / regularization sweep with the same caches, seeds and selection rule: does a smaller or
     more regularized head shrink the train-validation gap, and does fusion then beat image only?

    python -m src.evaluation.overfitting_diagnostic                       # A + B, ~35 min on 4 CPUs
    python -m src.evaluation.overfitting_diagnostic --parts A --seeds 42 --out_dir /tmp/diag  # quick
    python -m src.evaluation.overfitting_diagnostic --report_only         # redraw from the CSVs

A partial run (other --parts/--seeds/--variants) must name its own --out_dir, so it cannot replace the
committed full results with a subset.

Writes Team_Review/multimodal/overfitting/ (CSV + figures) and, for the sweep, the selected heads to
checkpoints/multimodal/20261018_overfit_<variant>_<modalities>_seed<seed>/best.pt with run folders
outputs/multimodal/<same id>/ (metrics_validation.json, predictions_validation.csv, training_log.csv,
per_epoch.csv, config.json). Those folders are new and must not exist yet.
"""
import argparse
import copy
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.loader import load_split_features  # noqa: E402
from src.evaluation.metrics import (class_weights_from_labels, compute_metrics,  # noqa: E402
                                    metrics_json, predictions_frame)
from src.evaluation.report import save_json  # noqa: E402
from src.models.trainer import as_inputs, build_head, fit  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.reproducibility import set_seed  # noqa: E402
from src.utils.safety import require_new_paths  # noqa: E402

DEVICE = torch.device("cpu")
CONFIGS = {"fusion": "configs/train/multimodal_baseline.yaml",
           "image": "configs/train/multimodal_image_only.yaml",
           "text": "configs/train/multimodal_text_only.yaml"}
RECORDED = {("fusion", 42): "20261016_distilbert_clip_seed42",
            ("fusion", 43): "20261016_distilbert_clip_frozen_seed43",
            ("fusion", 44): "20261016_distilbert_clip_frozen_seed44",
            **{("image", s): f"20261016_clip_frozen_mlp_image_only_seed{s}" for s in (42, 43, 44)},
            **{("text", s): f"20261016_distilbert_frozen_mlp_text_only_seed{s}" for s in (42, 43, 44)}}
# name -> changes to the config (the baseline is MLP 512, dropout 0.2, AdamW lr 1e-3, wd 1e-4)
VARIANTS = {
    "mlp512": {},
    "mlp256": {"hidden_dim": 256},
    "mlp128": {"hidden_dim": 128},
    "mlp64": {"hidden_dim": 64},
    "linear": {"hidden_dim": 0, "dropout": 0.0},
    "mlp512_dropout0.5": {"dropout": 0.5},
    "mlp512_wd1e-2": {"weight_decay": 0.01},
    "mlp512_lr1e-4": {"lr": 0.0001},
}


def variant_config(base: dict, changes: dict) -> dict:
    cfg = copy.deepcopy(base)
    for key in ("hidden_dim", "dropout"):
        if key in changes:
            cfg["fusion"][key] = changes[key]
    for key in ("weight_decay", "lr"):
        if key in changes:
            cfg[key] = changes[key]
    return cfg


def ece(prob1: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    """Expected calibration error of the predicted class's confidence (equal-width bins)."""
    pred = (prob1 >= 0.5).astype(int)
    conf = np.where(pred == 1, prob1, 1 - prob1)
    edges = np.linspace(0.5, 1.0, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf >= lo) & ((conf < hi) if hi < 1.0 else (conf <= hi))
        if mask.any():
            total += mask.mean() * abs((pred[mask] == y[mask]).mean() - conf[mask].mean())
    return float(total)


@torch.no_grad()
def split_scores(head, features, y, weights, threshold) -> dict:
    """Every quantity for one split, eval() mode, whole split at once."""
    head.eval()
    logits = head(**as_inputs(head, [features[m].to(DEVICE) for m in head.modalities]))
    yt = y.to(DEVICE)
    prob1 = logits.softmax(-1)[:, 1].cpu().numpy().astype(np.float64)
    pred = (prob1 >= threshold).astype(int)
    labels = y.numpy()
    m = compute_metrics(labels, pred)
    conf = np.where(pred == 1, prob1, 1 - prob1)
    wrong = pred != labels
    return {
        "weighted_ce": F.cross_entropy(logits, yt, weight=weights).item(),
        "nll": F.cross_entropy(logits, yt).item(),
        "macro_f1": m["macro_f1"], "accuracy": m["accuracy"],
        "roc_auc": roc_auc_score(labels, prob1),
        "pr_auc_class0": average_precision_score(1 - labels, 1 - prob1),
        "ece": ece(prob1, labels),
        "mean_confidence": float(conf.mean()),
        "errors": int(wrong.sum()),
        "confident_errors_90": int((wrong & (conf >= 0.9)).sum()),
    }


@torch.no_grad()
def train_mode_loss(head, features, y, weights) -> float:
    """Weighted CE on train with dropout ON at the end-of-epoch parameters; uses a private RNG
    state (fork_rng) so the training trajectory is unchanged."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        head.train()
        logits = head(**as_inputs(head, [features[m].to(DEVICE) for m in head.modalities]))
        loss = F.cross_entropy(logits, y.to(DEVICE), weight=weights).item()
    head.eval()
    return loss


def run_one(cfg, data, ckpt_path, label):
    """Train exactly like src.models.trainer and score every epoch; returns (log, best, per_epoch)."""
    (_, f_train, y_train), (val_ids, f_val, y_val) = data
    weights = torch.tensor(class_weights_from_labels(y_train.numpy()), dtype=torch.float32)
    rows = []

    def score(head, epoch):
        row = {"run": label, "epoch": epoch}
        for split, feats, y in (("train", f_train, y_train), ("validation", f_val, y_val)):
            row.update({f"{split}_{k}": v for k, v in
                        split_scores(head, feats, y, weights, cfg["threshold"]).items()})
        row["train_weighted_ce_dropout_on"] = train_mode_loss(head, f_train, y_train, weights)
        rows.append(row)

    set_seed(cfg["seed"])                      # same order as the trainer CLI: seed -> head -> fit
    head = build_head(cfg, f_train).to(DEVICE)
    log, best, secs, _ = fit(head, f_train, y_train, f_val, y_val, cfg, DEVICE, ckpt_path,
                             ckpt_extra={"modalities": list(head.modalities)}, epoch_callback=score)
    per_epoch = pd.DataFrame(rows)
    per_epoch.insert(2, "logged_train_loss", [r["train_loss"] for r in log])
    per_epoch.insert(3, "logged_val_loss", [r["val_loss"] for r in log])
    return log, best, per_epoch, secs, sum(p.numel() for p in head.parameters())


def load_data(cfg):
    return (load_split_features(cfg, "train"), load_split_features(cfg, "validation"))


def part_a(seeds, out, logs):
    """Recorded configs, unified per-epoch scoring, exact reproduction check."""
    per_epoch, checks = [], []
    for modset, path in CONFIGS.items():
        base = load_config(path)
        data = load_data(base)
        for seed in seeds:
            cfg = dict(copy.deepcopy(base), seed=seed)
            recorded = RECORDED[(modset, seed)]
            with tempfile.TemporaryDirectory() as tmp:
                log, best, pe, secs, n = run_one(cfg, data, Path(tmp) / "best.pt", recorded)
                new = torch.load(Path(tmp) / "best.pt", weights_only=True)["state_dict"]
            old = torch.load(f"checkpoints/multimodal/{recorded}/best.pt", weights_only=True)["state_dict"]
            ref_log = Path(f"outputs/multimodal/{recorded}/training_log.csv").read_text(encoding="utf-8")
            checks.append({"run": recorded, "modalities": modset, "seed": seed,
                           # same text as the CSV the trainer wrote for the recorded run (any line ending)
                           "training_log_identical": pd.DataFrame(log).to_csv(index=False, lineterminator="\n")
                           == ref_log.replace("\r\n", "\n"),
                           "best_checkpoint_identical": all(torch.equal(new[k], old[k]) for k in old),
                           "best_epoch": best["epoch"], "seconds": round(secs, 1)})
            pe.insert(1, "modalities", modset); pe.insert(2, "variant", "mlp512"); pe.insert(3, "seed", seed)
            per_epoch.append(pe)
            print(f"[A] {recorded}: reproduced={checks[-1]}", file=logs, flush=True)
    return pd.concat(per_epoch, ignore_index=True), pd.DataFrame(checks)


def part_b(seeds, variants, out, logs, run_date="20261018"):
    """Sweep: fusion vs image only under each variant, same seeds and selection (best val Macro-F1)."""
    per_epoch = []
    for modset in ("fusion", "image"):
        base = load_config(CONFIGS[modset])
        data = load_data(base)
        for name in variants:
            if name == "mlp512":
                continue  # = part A (recorded runs)
            for seed in seeds:
                cfg = dict(variant_config(base, VARIANTS[name]), seed=seed,
                           model_name=f"{base['model_name']}_{name}")
                run_id = f"{run_date}_overfit_{name}_{modset}_seed{seed}"
                ckpt_dir, out_dir = Path("checkpoints/multimodal") / run_id, Path("outputs/multimodal") / run_id
                require_new_paths(ckpt_dir, out_dir)
                ckpt_dir.mkdir(parents=True); out_dir.mkdir(parents=True)
                log, best, pe, secs, n = run_one(cfg, data, ckpt_dir / "best.pt", run_id)
                val_ids = data[1][0]
                y_val = data[1][2].numpy()
                pd.DataFrame(log).to_csv(out_dir / "training_log.csv", index=False)
                pe.to_csv(out_dir / "per_epoch.csv", index=False)
                predictions_frame(val_ids, y_val, best["pred"].numpy(), best["prob"].numpy(),
                                  cfg["model_name"], "validation", seed).to_csv(
                    out_dir / "predictions_validation.csv", index=False)
                save_json(metrics_json(cfg["model_name"], "validation", seed, cfg["threshold"], best, secs,
                                       validation_loss=round(best["val_loss"], 6), best_epoch=best["epoch"]),
                          out_dir / "metrics_validation.json")
                save_json({**cfg, "run_id": run_id, "task": "multimodal", "variant": name,
                           "trainable_parameters": n, "best_epoch": best["epoch"],
                           "checkpoint": str(ckpt_dir / "best.pt"),
                           "command": "python -m src.evaluation.overfitting_diagnostic"},
                          out_dir / "config.json")
                pe.insert(1, "modalities", modset); pe.insert(2, "variant", name); pe.insert(3, "seed", seed)
                per_epoch.append(pe)
                print(f"[B] {run_id}: best epoch {best['epoch']} val Macro-F1 {best['macro_f1']:.4f} "
                      f"({secs:.0f}s, {n:,} params)", file=logs, flush=True)
    return pd.concat(per_epoch, ignore_index=True) if per_epoch else pd.DataFrame()


def summarize(per_epoch: pd.DataFrame) -> pd.DataFrame:
    """One row per run: the selected epoch (best validation Macro-F1, ties -> first) and late epochs."""
    rows = []
    for (modset, variant, seed), g in per_epoch.groupby(["modalities", "variant", "seed"], sort=False):
        g = g.sort_values("epoch")
        b = g.loc[g["validation_macro_f1"].idxmax()]
        last10 = g[g.epoch > g.epoch.max() - 10]
        first, final = g.iloc[0], g.iloc[-1]
        rows.append({
            "modalities": modset, "variant": variant, "seed": seed, "best_epoch": int(b.epoch),
            "val_macro_f1_best": b.validation_macro_f1, "val_macro_f1_last10": last10.validation_macro_f1.mean(),
            "val_roc_auc_best": b.validation_roc_auc, "val_roc_auc_last10": last10.validation_roc_auc.mean(),
            "train_macro_f1_best": b.train_macro_f1, "train_roc_auc_best": b.train_roc_auc,
            "gap_macro_f1_best": b.train_macro_f1 - b.validation_macro_f1,
            "gap_roc_auc_best": b.train_roc_auc - b.validation_roc_auc,
            # fixed epochs, not affected by which epoch happened to be selected
            "gap_roc_auc_last10": (last10.train_roc_auc - last10.validation_roc_auc).mean(),
            "train_roc_auc_epoch30": final.train_roc_auc,
            # signed calibration: mean confidence minus accuracy (< 0 under-, > 0 over-confident)
            "val_overconfidence_epoch1": first.validation_mean_confidence - first.validation_accuracy,
            "val_overconfidence_epoch30": final.validation_mean_confidence - final.validation_accuracy,
            "train_weighted_ce_best": b.train_weighted_ce, "val_weighted_ce_best": b.validation_weighted_ce,
            "val_weighted_ce_min": g.validation_weighted_ce.min(),
            "val_weighted_ce_min_epoch": int(g.loc[g.validation_weighted_ce.idxmin()].epoch),
            "val_ece_best": b.validation_ece, "val_confident_errors_90_best": int(b.validation_confident_errors_90),
            "train_weighted_ce_epoch30": g.iloc[-1].train_weighted_ce,
            "val_weighted_ce_epoch30": g.iloc[-1].validation_weighted_ce,
            "val_roc_auc_epoch1": g.iloc[0].validation_roc_auc,
        })
    return pd.DataFrame(rows)


COLORS = {"fusion": "#2a78d6", "image": "#eb6834", "text": "#1baf7a"}   # same slots as the results report
NAMES = {"fusion": "Text+Image (fusion)", "image": "Image only", "text": "Text only"}
INK, MUTED, GRID = "#1f2328", "#6b7280", "#e5e7eb"


def _axes_style(ax, title, ylabel):
    ax.set_title(title, fontsize=10, color=INK, loc="left")
    ax.set_ylabel(ylabel, fontsize=9, color=INK)
    ax.set_xlabel("epoch", fontsize=9, color=INK)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.tick_params(labelsize=8, colors=MUTED)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def make_report(out: Path):
    """Figures + comparison table from per_epoch.csv / run_summary.csv."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pe = pd.read_csv(out / "per_epoch.csv")
    summary = pd.read_csv(out / "run_summary.csv")

    # O1: one run, every loss accounting + both splits, same parameters per epoch
    run = pe[(pe.modalities == "fusion") & (pe.variant == "mlp512") & (pe.seed == 42)].sort_values("epoch")
    if len(run):
        best = int(run.loc[run.validation_macro_f1.idxmax()].epoch)
        c = COLORS["fusion"]
        fig, axes = plt.subplots(1, 4, figsize=(17, 3.8))
        ax = axes[0]
        ax.plot(run.epoch, run.validation_weighted_ce, color=c, lw=2, label="validation, eval()")
        ax.plot(run.epoch, run.train_weighted_ce, color=c, lw=2, ls="--", label="train, eval()")
        ax.plot(run.epoch, run.train_weighted_ce_dropout_on, color=c, lw=1.2, ls=(0, (1, 1.5)),
                label="train, dropout on (end-of-epoch weights)")
        ax.plot(run.epoch, run.logged_train_loss, color=MUTED, lw=1.2, ls="-.", label="train, as logged")
        ax.axhline(np.log(2), color=MUTED, lw=0.8, ls=":")
        ax.text(1, np.log(2) + 0.03, "ln 2 (uniform)", fontsize=7, color=MUTED)
        _axes_style(ax, "Weighted cross-entropy (one definition)", "loss")
        ax.legend(fontsize=7, frameon=False)
        for ax, key, title in ((axes[1], "macro_f1", "Macro-F1 at threshold 0.5"),
                               (axes[2], "roc_auc", "ROC-AUC (threshold-free)")):
            ax.plot(run.epoch, run[f"validation_{key}"], color=c, lw=2, label="validation")
            ax.plot(run.epoch, run[f"train_{key}"], color=c, lw=2, ls="--", label="train")
            ax.axhline(0.5, color=MUTED, lw=0.8, ls=":")
            ax.text(30, 0.505, "0.5: label-independent level", fontsize=7, color=MUTED, ha="right")
            ax.set_ylim(0.4, 1.02)
            _axes_style(ax, title, {"macro_f1": "Macro-F1", "roc_auc": "ROC-AUC", "weighted_ce": "loss"}[key])
            ax.legend(fontsize=7, frameon=False, loc="center right")
        ax = axes[3]
        ax.plot(run.epoch, run.validation_errors, color=c, lw=2, label="all errors")
        ax.plot(run.epoch, run.validation_confident_errors_90, color=c, lw=2, ls="--",
                label="errors with confidence >= 0.9")
        _axes_style(ax, "Validation errors (of 699)", "count")
        ax.legend(fontsize=7, frameon=False)
        for ax in axes:
            ax.axvline(best, color=INK, lw=0.8, ls=":")
        lo, hi = axes[3].get_ylim()
        axes[3].text(best - 0.4, lo + 0.04 * (hi - lo), f"selected epoch {best}", fontsize=7, color=INK,
                     ha="right", va="bottom")
        fig.suptitle("Baseline fusion head, seed 42: the same parameters scored on train and validation every epoch",
                     fontsize=11, color=INK, x=0.01, ha="left")
        fig.tight_layout()
        fig.savefig(out / "fig_o1_seed42_unified_eval.png", dpi=150)
        plt.close(fig)

    # O2: recorded configs, 3 seeds (mean line, min-max band), train dashed / validation solid
    rec = pe[pe.variant == "mlp512"]
    if len(rec):
        fig, axes = plt.subplots(1, 3, figsize=(15, 3.8))
        for ax, key, title in ((axes[0], "weighted_ce", "Weighted cross-entropy"),
                               (axes[1], "macro_f1", "Macro-F1"), (axes[2], "roc_auc", "ROC-AUC")):
            for modset in ("fusion", "image", "text"):
                g = rec[rec.modalities == modset].groupby("epoch")
                if not len(g):
                    continue
                for split, ls in (("train", "--"), ("validation", "-")):
                    col = f"{split}_{key}"
                    mean, lo, hi = g[col].mean(), g[col].min(), g[col].max()
                    ax.plot(mean.index, mean, color=COLORS[modset], lw=1.8, ls=ls,
                            label=f"{NAMES[modset]}, {split}")
                    ax.fill_between(mean.index, lo, hi, color=COLORS[modset], alpha=0.12, lw=0)
            if key != "weighted_ce":
                ax.axhline(0.5, color=MUTED, lw=0.8, ls=":")
                ax.text(30, 0.505, "0.5: label-independent level", fontsize=7, color=MUTED, ha="right")
            _axes_style(ax, title, {"macro_f1": "Macro-F1", "roc_auc": "ROC-AUC", "weighted_ce": "loss"}[key])
        axes[0].legend(fontsize=7, frameon=False)
        fig.suptitle("Recorded configurations, seeds 42-44 (line = mean, band = min-max); dashed = train, solid = validation",
                     fontsize=11, color=INK, x=0.01, ha="left")
        fig.tight_layout()
        fig.savefig(out / "fig_o2_recorded_runs.png", dpi=150)
        plt.close(fig)

    # O3 + table: capacity / regularization sweep, fusion vs image only
    sw = summary[summary.modalities.isin(["fusion", "image"])]
    order = [v for v in VARIANTS if v in set(sw.variant)]
    if len(order) > 1:
        rows = []
        for v in order:
            f = sw[(sw.variant == v) & (sw.modalities == "fusion")].set_index("seed")
            i = sw[(sw.variant == v) & (sw.modalities == "image")].set_index("seed")
            seeds = sorted(set(f.index) & set(i.index))
            row = {"variant": v, "seeds": len(seeds)}
            for name, frame in (("fusion", f), ("image", i)):
                for col in ("val_macro_f1_best", "val_macro_f1_last10", "val_roc_auc_last10",
                            "gap_roc_auc_best", "gap_roc_auc_last10", "train_roc_auc_epoch30",
                            "gap_macro_f1_best", "val_weighted_ce_best", "best_epoch"):
                    row[f"{name}_{col}_mean"] = frame.loc[seeds, col].mean()
                    row[f"{name}_{col}_sd"] = frame.loc[seeds, col].std(ddof=1) if len(seeds) > 1 else np.nan
            for col in ("val_macro_f1_best", "val_macro_f1_last10", "val_roc_auc_last10"):
                d = f.loc[seeds, col] - i.loc[seeds, col]
                row[f"delta_{col}_mean"], row[f"delta_{col}_min"], row[f"delta_{col}_max"] = d.mean(), d.min(), d.max()
            rows.append(row)
        table = pd.DataFrame(rows)
        table.to_csv(out / "fusion_vs_image.csv", index=False, float_format="%.4f")
        # same capacity: fusion with hidden H has about as many weights as image only with 2H
        means = sw.groupby(["modalities", "variant"])[["val_macro_f1_best", "val_macro_f1_last10"]].mean()
        pairs = [("mlp256", "mlp512"), ("mlp128", "mlp256"), ("mlp64", "mlp128")]
        pd.DataFrame([{"fusion_head": f, "image_head": i,
                       **{f"delta_{c}": means.loc[("fusion", f), c] - means.loc[("image", i), c]
                          for c in ("val_macro_f1_best", "val_macro_f1_last10")}}
                      for f, i in pairs if ("fusion", f) in means.index and ("image", i) in means.index]
                     ).to_csv(out / "parameter_matched.csv", index=False, float_format="%.4f")

        fig, axes = plt.subplots(1, 4, figsize=(21, 4.4))
        x = np.arange(len(order))
        for ax, col, title in ((axes[0], "val_macro_f1_last10", "Validation Macro-F1, mean of last 10 epochs"),
                               (axes[1], "val_roc_auc_last10", "Validation ROC-AUC, mean of last 10 epochs"),
                               (axes[2], "gap_roc_auc_last10", "Train minus validation ROC-AUC, last 10 epochs")):
            for k, modset in enumerate(("fusion", "image")):
                pos = x + (k - 0.5) * 0.3
                vals = [sw[(sw.variant == v) & (sw.modalities == modset)][col].to_numpy() for v in order]
                ax.scatter(np.repeat(pos, [len(a) for a in vals]), np.concatenate(vals), s=14,
                           color=COLORS[modset], alpha=0.55, linewidths=0)
                ax.scatter(pos, [a.mean() for a in vals], s=46, marker="D", color=COLORS[modset],
                           edgecolors="white", linewidths=1.5, label=NAMES[modset])
            if col != "gap_roc_auc_last10":
                ax.axhline(0.5, color=MUTED, lw=0.8, ls=":")
            ax.set_xticks(x, order, rotation=30, ha="right", fontsize=8)
            ax.set_ylabel({"val_macro_f1_last10": "Macro-F1", "val_roc_auc_last10": "ROC-AUC",
                           "gap_roc_auc_last10": "ROC-AUC gap"}[col], fontsize=9, color=INK)
            ax.set_title(title, fontsize=10, color=INK, loc="left")
            ax.grid(True, axis="y", color=GRID, linewidth=0.6)
            ax.tick_params(labelsize=8, colors=MUTED)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
        ax = axes[3]
        f = sw[sw.modalities == "fusion"].set_index(["variant", "seed"])
        i = sw[sw.modalities == "image"].set_index(["variant", "seed"])
        ax.axhspan(-0.03, 0.03, color=GRID, alpha=0.7, lw=0)
        ax.text(len(order) - 0.5, 0.027, "about the resolution of 699 memes", fontsize=7, color=MUTED, ha="right", va="top")
        ax.axhline(0, color=INK, lw=0.8)
        for k, v in enumerate(order):
            d = (f.loc[v, "val_macro_f1_best"] - i.loc[v, "val_macro_f1_best"]).dropna()
            ax.scatter(np.full(len(d), k), d, s=14, color=COLORS["fusion"], alpha=0.55, linewidths=0)
            ax.scatter([k], [d.mean()], s=46, marker="D", color=COLORS["fusion"], edgecolors="white", linewidths=1.5)
        ax.set_xticks(x, order, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("fusion minus image only", fontsize=9, color=INK)
        ax.set_title("Paired difference, Macro-F1 at the selected epoch", fontsize=10, color=INK, loc="left")
        ax.grid(True, axis="y", color=GRID, linewidth=0.6)
        ax.tick_params(labelsize=8, colors=MUTED)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        axes[0].legend(fontsize=8, frameon=False)
        fig.suptitle("Head capacity and regularization sweep (diamond = mean of seeds 42-44, dots = seeds; validation only)",
                     fontsize=11, color=INK, x=0.01, ha="left")
        fig.tight_layout()
        fig.savefig(out / "fig_o3_capacity_sweep.png", dpi=150)
        plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", nargs="+", choices=["A", "B"], default=["A", "B"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    ap.add_argument("--variants", nargs="+", choices=list(VARIANTS), default=list(VARIANTS))
    ap.add_argument("--out_dir", default="Team_Review/multimodal/overfitting")
    ap.add_argument("--report_only", action="store_true", help="redraw figures/tables from the CSVs")
    args = ap.parse_args(argv)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if args.report_only:
        summarize(pd.read_csv(out / "per_epoch.csv")).to_csv(out / "run_summary.csv", index=False,
                                                            float_format="%.6f")
        make_report(out)
        return
    if out == Path(ap.get_default("out_dir")) and (
            sorted(args.parts) != ["A", "B"] or sorted(args.seeds) != [42, 43, 44]
            or sorted(args.variants) != sorted(VARIANTS)):
        ap.error("a partial run needs its own --out_dir (the default folder holds the full results)")
    t0 = time.time()
    logs = sys.stdout
    frames = []
    if "A" in args.parts:
        pe_a, checks = part_a(args.seeds, out, logs)
        checks.to_csv(out / "reproduction_check.csv", index=False)
        frames.append(pe_a)
    if "B" in args.parts:
        frames.append(part_b(args.seeds, args.variants, out, logs))
    per_epoch = pd.concat([f for f in frames if len(f)], ignore_index=True)
    per_epoch.to_csv(out / "per_epoch.csv", index=False, float_format="%.6f")
    summarize(per_epoch).to_csv(out / "run_summary.csv", index=False, float_format="%.6f")
    make_report(out)
    print(f"done in {time.time() - t0:.0f}s -> {out}")


if __name__ == "__main__":
    main()
