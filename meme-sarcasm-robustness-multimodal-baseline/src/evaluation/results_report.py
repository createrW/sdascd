"""Statistics and figures for the multimodal results report (validation split only).

Reads the finished runs in outputs/multimodal/, their checkpoints and the cached features, and
writes everything the report cites to Team_Review/multimodal/report/:

    metrics_by_run.csv          extended metrics per run (AUCs, Brier, ECE, selection optimism)
    seed_summary.csv            mean / std / min / max over seeds 42, 43, 44 per model
    bootstrap_ci.csv            95% bootstrap CI of Macro-F1 / accuracy / F1 class 0 (seed 42)
    paired_comparisons.csv      paired bootstrap (seed 42 and 3-seed mean) + exact McNemar
    modality_reliance.csv       text / image features shuffled or replaced at inference
    dataset_stats.csv           split sizes, labels, caption lengths
    feature_agreement.json      CPU vs image-group CUDA CLIP cache
    (tf32_emulation.json is written separately by scripts/check_tf32_emulation.py)
    feature_geometry.csv        exact pairwise cosine / anisotropy of the cached text and image features
    chance_tests.csv            label-permutation tests (Macro-F1, ROC-AUC) and AUC bootstrap CIs
    linear_probes.csv           standardized logistic-regression probes per feature set (diagnostic)
    stats_summary.json          every number above in one file
    fig1..fig7 *.png

    python -m src.evaluation.results_report

Deterministic: bootstrap and permutation generators are seeded. No model is evaluated on Test;
only Test's label counts and caption lengths are described (dataset_stats.csv).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import binomtest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve, precision_recall_curve
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.dataset import load_split  # noqa: E402
from src.data.loader import FEATURES_ROOT, load_split_features  # noqa: E402
from src.features.cache import resolve_features_dir  # noqa: E402
from src.evaluation.metrics import compute_metrics  # noqa: E402
from src.models.multimodal_classifier import build_fusion_head  # noqa: E402
from src.models.trainer import predict  # noqa: E402

RUNS = Path("outputs/multimodal")


def cache_dir(name):
    """data/processed/features/<name>, or the old features/<name> if the cache was left there."""
    return resolve_features_dir(f"{FEATURES_ROOT}/{name}")
# Fixed model order = fixed colour slot (dataviz reference palette, light mode, validated).
MODELS = [
    dict(key="mm_cls", label="DistilBERT [CLS] + CLIP (baseline)", short="Text+Image [CLS]",
         color="#2a78d6", runs={42: "20261016_distilbert_clip_seed42",
                                43: "20261016_distilbert_clip_frozen_seed43",
                                44: "20261016_distilbert_clip_frozen_seed44"}),
    dict(key="image_mlp", label="CLIP only (same MLP)", short="Image only",
         color="#eb6834", runs={s: f"20261016_clip_frozen_mlp_image_only_seed{s}" for s in (42, 43, 44)}),
    dict(key="text_mlp", label="DistilBERT [CLS] only (same MLP)", short="Text only",
         color="#1baf7a", runs={s: f"20261016_distilbert_frozen_mlp_text_only_seed{s}" for s in (42, 43, 44)}),
    dict(key="mm_mean", label="DistilBERT mean pooling + CLIP", short="Text+Image [mean]",
         color="#eda100", runs={s: f"20261016_distilbert_meanpool_clip_frozen_seed{s}" for s in (42, 43, 44)}),
]
IMAGE_GROUP_RUN = Path("outputs/image/20261006_code_fixes_validation_v2")
REPRODUCTION_RUN = "20261016_distilbert_clip_seed42_reproduction_cpu"

INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
SURFACE, GRID, AXIS = "#fcfcfb", "#e1e0d9", "#c3c2b7"
GRAY_RAMP = ["#f0efec", "#c3c2b7", "#898781", "#52514e", "#2c2c2a"]
B_BOOT, N_PERM = 10000, 20


# ----------------------------------------------------------------------------- metrics
def read_preds(path):
    df = pd.read_csv(path, float_precision="round_trip")
    return df["sample_id"].tolist(), df["true_label"].to_numpy(), df["probability_1"].to_numpy()


def ece(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    total = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            total += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(total)


def extended_metrics(y, p, threshold=0.5):
    pred = (p >= threshold).astype(int)
    m = compute_metrics(y, pred)
    m["balanced_accuracy"] = (m["recall_0"] + m["recall_1"]) / 2
    m["roc_auc"] = float(roc_auc_score(y, p))
    m["pr_auc_1"] = float(average_precision_score(y, p))
    m["pr_auc_0"] = float(average_precision_score(1 - y, 1 - p))
    m["brier"] = float(np.mean((p - y) ** 2))
    m["ece_10"] = ece(y, p)
    m["predicted_1_rate"] = float(pred.mean())
    m["mean_p1"] = float(p.mean())
    m["oracle_threshold_macro_f1"], m["oracle_threshold"] = oracle_threshold(y, p)
    m["share_p_above_0.9"] = float((p > 0.9).mean())
    m["share_p_below_0.1"] = float((p < 0.1).mean())
    m["chance_macro_f1_at_rate"] = chance_macro_f1(float(y.mean()), m["predicted_1_rate"])
    return m


def oracle_threshold(y, p):
    """Best Macro-F1 over all thresholds ON VALIDATION. Diagnostic only; never used for selection."""
    best, best_t = -1.0, 0.5
    for t in np.unique(np.concatenate([p, [0.5]])):
        f = compute_metrics(y, (p >= t).astype(int))["macro_f1"]
        if f > best:
            best, best_t = f, float(t)
    return float(best), best_t


def chance_macro_f1(prior1, rate1):
    """Expected Macro-F1 (large-n limit) of a label-independent classifier predicting 1 at rate1."""
    f1_1 = 2 * prior1 * rate1 / (prior1 + rate1) if prior1 + rate1 else 0.0
    p0, r0 = 1 - prior1, 1 - rate1
    f1_0 = 2 * p0 * r0 / (p0 + r0) if p0 + r0 else 0.0
    return (f1_0 + f1_1) / 2


def auc_by_ranks(ranks_pos_sum, n1, n0):
    return (ranks_pos_sum - n1 * (n1 + 1) / 2) / (n1 * n0)


def permutation_tests(y, p, b=B_BOOT, seed=7):
    """Label-permutation null for Macro-F1 (predictions fixed) and ROC-AUC (scores fixed)."""
    from scipy.stats import rankdata
    rng = np.random.default_rng(seed)
    order = np.argsort(rng.random((b, len(y))), axis=1)
    yperm = y[order]
    pred = (p >= 0.5).astype(int)
    f1_null, _ = macro_f1_batch(yperm, np.broadcast_to(pred, yperm.shape))
    ranks = rankdata(p)
    n1 = int(y.sum())
    n0 = len(y) - n1
    auc_null = auc_by_ranks((ranks[None, :] * yperm).sum(1), n1, n0)
    f1_obs = compute_metrics(y, pred)["macro_f1"]
    auc_obs = float(roc_auc_score(y, p))
    return {"macro_f1": f1_obs, "macro_f1_null_mean": float(f1_null.mean()),
            "macro_f1_null_q95": float(np.percentile(f1_null, 95)),
            "macro_f1_perm_p": float((1 + (f1_null >= f1_obs).sum()) / (b + 1)),
            "roc_auc": auc_obs, "roc_auc_null_q95": float(np.percentile(auc_null, 95)),
            "roc_auc_perm_p": float((1 + (auc_null >= auc_obs).sum()) / (b + 1))}


def auc_bootstrap(y, p, idx):
    """Bootstrap ROC-AUC via pair-comparison matrix and resample multiplicities."""
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    k = (p[pos][:, None] > p[neg][None, :]) + 0.5 * (p[pos][:, None] == p[neg][None, :])
    counts = np.stack([np.bincount(r, minlength=len(y)) for r in idx]).astype(float)
    cp, cn = counts[:, pos], counts[:, neg]
    num = np.einsum("bi,ij,bj->b", cp, k, cn)
    auc = num / (cp.sum(1) * cn.sum(1))
    return auc


def macro_f1_batch(y, pred):
    """Macro-F1 for each row of (B, n) arrays of labels/predictions (0/1)."""
    tp1 = ((y == 1) & (pred == 1)).sum(1)
    fp1 = ((y == 0) & (pred == 1)).sum(1)
    fn1 = ((y == 1) & (pred == 0)).sum(1)
    tp0 = ((y == 0) & (pred == 0)).sum(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        f1_1 = np.where(2 * tp1 + fp1 + fn1 > 0, 2 * tp1 / (2 * tp1 + fp1 + fn1), 0.0)
        f1_0 = np.where(2 * tp0 + fn1 + fp1 > 0, 2 * tp0 / (2 * tp0 + fn1 + fp1), 0.0)
    return (f1_0 + f1_1) / 2, f1_0


def bootstrap_indices(n, seed=42):
    return np.random.default_rng(seed).integers(0, n, size=(B_BOOT, n))


def ci(values):
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def paired_p(diff):
    """Two-sided bootstrap p-value for H0: difference = 0."""
    return float(min(1.0, 2 * min((diff <= 0).mean(), (diff >= 0).mean())))


# ----------------------------------------------------------------------------- figures
def style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9.5, "text.color": INK,
        "axes.labelcolor": INK2, "axes.edgecolor": AXIS, "axes.linewidth": 1.0,
        "axes.facecolor": SURFACE, "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 1.0, "grid.linestyle": "-",
        "axes.axisbelow": True, "xtick.color": MUTED, "ytick.color": MUTED,
        "xtick.labelcolor": INK2, "ytick.labelcolor": INK2, "axes.spines.top": False,
        "axes.spines.right": False, "legend.frameon": False, "lines.linewidth": 2,
        "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
        "axes.titlesize": 10.5, "axes.titleweight": "semibold", "axes.titlecolor": INK,
        "axes.titlelocation": "left",
    })
    return plt


def line_handles(models):
    from matplotlib.lines import Line2D
    return [Line2D([0], [0], color=m["color"], lw=2, label=m["label"]) for m in models]


def dot_handle(label):
    from matplotlib.lines import Line2D
    return Line2D([0], [0], color=MUTED, marker="o", lw=0, markersize=7, markeredgecolor=SURFACE,
                  markeredgewidth=1.5, label=label)


def fig_macro_f1(plt, seed_rows, boot, refs, path, image_group):
    from matplotlib.lines import Line2D
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    ys = np.arange(1, len(MODELS) + 1)[::-1]
    offsets = {42: 0.14, 43: 0.0, 44: -0.14}
    for y, m in zip(ys, MODELS):
        lo, hi = boot[m["key"]]["macro_f1_ci"]
        ax.plot([lo, hi], [y + offsets[42]] * 2, color=m["color"], lw=2, alpha=0.35,
                solid_capstyle="round", zorder=2)
        rows = [r for r in seed_rows if r["model"] == m["key"]]
        mean = float(np.mean([r["macro_f1"] for r in rows]))
        ax.plot([mean, mean], [y - 0.26, y + 0.26], color=INK, lw=1.5, zorder=3)
        for r in rows:
            ax.scatter([r["macro_f1"]], [y + offsets[r["seed"]]], s=52 if r["seed"] == 42 else 44,
                       marker="D" if r["seed"] == 42 else "o", color=m["color"], edgecolor=SURFACE,
                       linewidth=1.8, zorder=4)
        ax.annotate(f"mean {mean:.3f}", (hi, y), xytext=(8, 0), textcoords="offset points",
                    va="center", fontsize=9, color=INK2)
    # the image group's linear head: one run, its own row (slot 5, magenta)
    lo, hi = boot["image_group_linear"]["macro_f1_ci"]
    ax.plot([lo, hi], [0, 0], color="#e87ba4", lw=2, alpha=0.4, solid_capstyle="round", zorder=2)
    ax.scatter([image_group], [0], s=52, marker="D", color="#e87ba4", edgecolor=SURFACE, linewidth=1.8,
               zorder=4)
    ax.annotate(f"one run {image_group:.3f}", (hi, 0), xytext=(8, 0), textcoords="offset points",
                va="center", fontsize=9, color=INK2)
    top = ys.max() + 0.55
    for value, name in refs:
        ax.axvline(value, color=MUTED, lw=1, zorder=1)
        ax.annotate(name, (value, top), xytext=(3, 0), textcoords="offset points",
                    fontsize=8.5, color=INK2, va="center")
    ax.set_yticks(list(ys) + [0], [m["label"] for m in MODELS] + ["CLIP linear head (image group)"])
    ax.set_ylim(-0.6, len(MODELS) + 0.85)
    ax.set_xlim(0.40, 0.64)
    ax.set_xlabel("Validation Macro-F1 (threshold 0.5)")
    ax.grid(axis="y", visible=False)
    ax.legend(handles=[Line2D([0], [0], marker="D", lw=0, color=MUTED, markersize=6, label="seed 42"),
                       Line2D([0], [0], marker="o", lw=0, color=MUTED, markersize=6, label="seeds 43, 44"),
                       Line2D([0], [0], color=INK, lw=1.5, label="3-seed mean"),
                       Line2D([0], [0], color=MUTED, lw=2, alpha=0.5, label="seed-42 bootstrap 95% CI")],
              loc="upper center", bbox_to_anchor=(0.42, -0.15), fontsize=8.5, ncol=4)
    ax.set_title("Validation Macro-F1 by model")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fig_training(plt, logs, path):
    from matplotlib.lines import Line2D
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.6))
    ax = axes[0]
    for m in MODELS:
        log = logs[m["key"]]
        ax.plot(log.epoch, log.val_macro_f1, color=m["color"], lw=2, label=m["label"], zorder=2)
        best = log.loc[log.val_macro_f1.idxmax()]
        ax.scatter([best.epoch], [best.val_macro_f1], s=50, color=m["color"], edgecolor=SURFACE,
                   linewidth=2, zorder=5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Macro-F1")
    ax.set_title("A  Validation Macro-F1 per epoch (seed 42)")
    ax.legend(handles=line_handles(MODELS) + [dot_handle("selected epoch (max)")],
              loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, fontsize=8.8)
    ax = axes[1]
    log = logs["mm_cls"]
    ax.plot(log.epoch, log.train_loss, color=MUTED, lw=2)
    ax.plot(log.epoch, log.val_loss, color="#2a78d6", lw=2)
    ax.axhline(np.log(2), color=AXIS, lw=1)
    ax.annotate("ln 2 = uniform prediction", (30, np.log(2)), xytext=(0, -13), textcoords="offset points",
                ha="right", fontsize=9, color=INK2,
                bbox=dict(boxstyle="square,pad=0.1", fc=SURFACE, ec="none"))
    ax.legend(handles=[Line2D([0], [0], color=MUTED, lw=2, label="train loss"),
                       Line2D([0], [0], color="#2a78d6", lw=2, label="validation loss")],
              loc="upper left", fontsize=9)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Weighted cross-entropy")
    ax.set_title("B  Baseline losses (seed 42)")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fig_confusion(plt, cms, path):
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("gray_seq", GRAY_RAMP)
    fig, axes = plt.subplots(2, 3, figsize=(10.0, 6.6))
    for ax, (title, cm) in zip(axes.ravel(), cms):
        rate = cm / cm.sum(1, keepdims=True)
        ax.imshow(rate, cmap=cmap, vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]}\n{rate[i, j]:.0%} of row", ha="center", va="center",
                        fontsize=11, color="white" if rate[i, j] > 0.5 else INK)
        ax.set_xticks([0, 1], ["0 not sarc.", "1 sarcastic"], fontsize=10)
        ax.set_yticks([0, 1], ["0 not sarc.", "1 sarcastic"], fontsize=10)
        ax.set_xlabel("predicted", fontsize=10)
        ax.set_ylabel("true", fontsize=10)
        ax.grid(False)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title(title, fontsize=10.5)
    note = axes.ravel()[-1]
    note.axis("off")
    note.text(0.0, 0.6, "Validation, threshold 0.5.\nMultimodal-group heads:\nseed 42.\n"
              "Image group: its reference\nrun (20261006, CUDA).\n\n"
              "Shade = share of the\ntrue-class row.",
              fontsize=10, color=INK2, va="center")
    fig.suptitle("Confusion matrices", x=0.01, ha="left", fontsize=12, fontweight="semibold", color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_roc_pr(plt, curves, y, path):
    from matplotlib.lines import Line2D
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.6))
    roc_handles, pr_handles = [], []
    for m in MODELS:
        yy, p = curves[m["key"]]
        fpr, tpr, _ = roc_curve(yy, p)
        axes[0].plot(fpr, tpr, color=m["color"], lw=2)
        prec, rec, _ = precision_recall_curve(1 - yy, 1 - p)
        axes[1].plot(rec[:-1], prec[:-1], color=m["color"], lw=2)  # drop sklearn's synthetic (0, 1) end point
        roc_handles.append(Line2D([0], [0], color=m["color"], lw=2,
                                  label=f"{m['label']}  AUC {roc_auc_score(yy, p):.3f}"))
        pr_handles.append(Line2D([0], [0], color=m["color"], lw=2,
                                 label=f"{m['label']}  PR-AUC {average_precision_score(1 - yy, 1 - p):.3f}"))
    axes[0].plot([0, 1], [0, 1], color=MUTED, lw=1)
    roc_handles.append(Line2D([0], [0], color=MUTED, lw=1, label="chance (AUC 0.5)"))
    axes[0].set_xlabel("False positive rate (positive class = 1 sarcastic)")
    axes[0].set_ylabel("True positive rate")
    axes[0].set_title("A  ROC curves (seed 42)")
    axes[0].legend(handles=roc_handles, loc="lower right", fontsize=8.6)
    prior0 = float((y == 0).mean())
    axes[1].axhline(prior0, color=MUTED, lw=1)
    axes[1].annotate(f"chance = class-0 share {prior0:.3f}", (1.0, prior0), xytext=(0, -12),
                     textcoords="offset points", ha="right", fontsize=8, color=INK2)
    axes[1].set_xlabel("Recall of class 0 (not sarcastic)")
    axes[1].set_ylabel("Precision of class 0")
    axes[1].set_title("B  Precision-recall, minority class 0 (seed 42)")
    axes[1].set_ylim(0, 0.6)
    axes[1].text(1.0, 0.02, "y-axis cut at 0.6: the first 1-2 memes of\nimage only / text only exceed it",
                 fontsize=8, color=INK2, va="bottom", ha="right")
    axes[1].legend(handles=pr_handles, loc="upper right", bbox_to_anchor=(1.0, 0.93), fontsize=8.6)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fig_probabilities(plt, y, p, path):
    fig, ax = plt.subplots(figsize=(7.6, 3.3))
    bins = np.linspace(0, 1, 26)
    for cls, color, name in ((0, INK, "true 0 (not sarcastic)"), (1, MUTED, "true 1 (sarcastic)")):
        w = np.ones((y == cls).sum()) / (y == cls).sum()
        ax.hist(p[y == cls], bins=bins, weights=w, histtype="step", lw=2, color=color, label=name)
    ax.axvline(0.5, color=MUTED, lw=1)
    ax.annotate("threshold 0.5", (0.5, ax.get_ylim()[1] * 0.92), xytext=(4, 0),
                textcoords="offset points", fontsize=8, color=INK2)
    ax.set_xlabel("Predicted P(sarcastic)")
    ax.set_ylabel("Share of the true class")
    ax.set_title("Baseline (seed 42): P(sarcastic) by true class")
    ax.legend(loc="upper left", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fig_reliance(plt, rel, path):
    conds = ["clean", "text_shuffled", "text_mean", "image_shuffled", "image_mean"]
    names = ["clean", "text\nshuffled", "text ->\ntrain mean", "image\nshuffled", "image ->\ntrain mean"]
    agg = rel.groupby("condition").agg(f1=("macro_f1", "mean"), change=("prediction_change_rate", "mean"),
                                       dp=("mean_abs_delta_p", "mean"))
    per_seed = rel.groupby(["condition", "seed"]).macro_f1.mean()
    lo = per_seed.groupby("condition").min()
    hi = per_seed.groupby("condition").max()
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.9))
    x = np.arange(len(conds))
    ax = axes[0]  # dots: Macro-F1 needs no zero baseline; line = range over seeds x permutations
    for xi, c in zip(x, conds):
        ax.plot([xi, xi], [lo[c], hi[c]], color="#2a78d6", lw=2, alpha=0.35)
        ax.scatter([xi], [agg.loc[c, "f1"]], s=46, color="#2a78d6", edgecolor=SURFACE, linewidth=2,
                   zorder=3)
        ax.annotate(f"{agg.loc[c, 'f1']:.3f}", (xi, hi[c]), xytext=(0, 5), textcoords="offset points",
                    ha="center", fontsize=9.5, color=INK2)
    for value, name, dy in ((0.4349, "majority 0.435", 3), (0.5, "label-independent chance 0.500", -11)):
        ax.axhline(value, color=MUTED, lw=1)
        ax.annotate(name, (x[0] - 0.25, value), xytext=(0, dy), textcoords="offset points",
                    ha="left", fontsize=8.5, color=INK2)
    ax.set_xlim(x[0] - 0.35, x[-1] + 0.35)
    ax.set_ylim(0.40, float(hi.max()) + 0.03)
    ax.set_ylabel("Validation Macro-F1")
    ax.set_title("A  Macro-F1 (dot = mean, line = range of 3 seed means)")
    ax = axes[1]  # bars from zero: a rate
    vals = [agg.loc[c, "change"] for c in conds]
    bars = ax.bar(x, vals, width=0.5, color="#2a78d6")
    for b, v in zip(bars, vals):
        ax.annotate(f"{v:.0%}", (b.get_x() + b.get_width() / 2, v), xytext=(0, 3),
                    textcoords="offset points", ha="center", fontsize=9.5, color=INK2)
    ax.set_ylim(0, max(vals) * 1.18 + 0.01)
    ax.set_ylabel("Share of predictions that flip")
    ax.set_title("B  Predictions that flip vs clean")
    for ax in axes:
        ax.set_xticks(x, names)
        ax.grid(axis="x", visible=False)
    fig.suptitle("Baseline fusion model with one modality broken at inference: seeds 42/43/44, "
                 "20 permutations per shuffle", x=0.01, ha="left", fontsize=9, color=INK2)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fig_features(plt, cos, path):
    fig, ax = plt.subplots(figsize=(7.8, 3.4))
    d = np.clip(1 - cos, 1e-9, None)
    bins = np.logspace(-9, -1, 41)
    ax.hist(d, bins=bins, color="#2a78d6", edgecolor=SURFACE, linewidth=0.6)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("1 - cosine(CPU float32 feature, image-group CUDA feature), computed in float64")
    ax.set_ylabel("Images (log scale)")
    ax.set_title("CLIP features: CPU re-extraction vs image-group CUDA cache (6,292 images)")
    n_bad = int((cos < 0.999).sum())
    ax.axvspan(1e-3, bins[-1], color="#eb6834", alpha=0.08, lw=0)
    ax.text(1.15e-3, 250, f"shaded: {n_bad} images\nwith cosine < 0.999\n(worst 1 - cos\n= {float(d.max()):.4f})",
            fontsize=8.5, color=INK2, va="center", ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="Team_Review/multimodal/report")
    ap.add_argument("--data_dir", default=None)
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    plt = style()
    summary = {"bootstrap_resamples": B_BOOT, "permutations": N_PERM}

    # ---------- dataset statistics
    rows = []
    for split in ("train", "validation", "test"):
        df = load_split(split, args.data_dir)
        meta = json.loads(Path(f"{cache_dir('distilbert_first_token')}/meta.json").read_text())["splits"][split]
        counts4 = df["sarcasm"].value_counts().to_dict()
        rows.append({"split": split, "n": len(df), "class_0": int((df.sarcasm_binary == 0).sum()),
                     "class_1": int((df.sarcasm_binary == 1).sum()),
                     "share_class_1": round(float(df.sarcasm_binary.mean()), 4),
                     **{f"sarcasm_{k}": int(v) for k, v in sorted(counts4.items())},
                     "no_text": meta["no_text"], "tokens_mean": meta["tokens_mean"],
                     "tokens_median": meta["tokens_median"], "tokens_max": meta["tokens_max"],
                     "truncated_at_128": meta["truncated_at_max_length"]})
    pd.DataFrame(rows).to_csv(out / "dataset_stats.csv", index=False)
    summary["dataset"] = rows

    # ---------- per-run metrics
    truth = load_split("validation", args.data_dir)
    y = truth["sarcasm_binary"].to_numpy()
    run_rows, seed_rows, preds = [], [], {}
    for m in MODELS:
        for seed, run in m["runs"].items():
            ids, yt, p = read_preds(RUNS / run / "predictions_validation.csv")
            assert ids == truth.sample_id.tolist() and (yt == y).all(), run
            log = pd.read_csv(RUNS / run / "training_log.csv")
            reported = json.loads((RUNS / run / "metrics_validation.json").read_text())
            em = extended_metrics(y, p)
            assert abs(em["macro_f1"] - reported["macro_f1"]) < 1e-6, run
            row = {"model": m["key"], "label": m["label"], "seed": seed, "run_id": run, **em,
                   "best_epoch": reported["best_epoch"], "training_seconds": reported["training_seconds"],
                   "val_f1_last_epoch": float(log.val_macro_f1.iloc[-1]),
                   "val_f1_mean_all_epochs": float(log.val_macro_f1.mean()),
                   "val_f1_mean_last10": float(log.val_macro_f1.iloc[-10:].mean()),
                   "val_f1_epoch_min": float(log.val_macro_f1.min()),
                   "train_loss_final": float(log.train_loss.iloc[-1]),
                   "val_loss_min": float(log.val_loss.min()), "val_loss_final": float(log.val_loss.iloc[-1])}
            row["selection_gap"] = row["macro_f1"] - row["val_f1_mean_last10"]
            steps = log.val_macro_f1.diff().abs().iloc[1:]
            row["val_f1_epoch_step_mean_abs"] = float(steps.mean())
            row["val_f1_epoch_step_max_abs"] = float(steps.max())
            run_rows.append(row)
            seed_rows.append(row)
            preds[(m["key"], seed)] = p
    ids, yt, p_img = read_preds(IMAGE_GROUP_RUN / "predictions_validation.csv")
    assert ids == truth.sample_id.tolist()
    img_row = {"model": "image_group_linear", "label": "CLIP linear head (image group, CUDA)",
               "seed": 42, "run_id": IMAGE_GROUP_RUN.name, **extended_metrics(y, p_img)}
    run_rows.append(img_row)
    preds[("image_group_linear", 42)] = p_img
    maj = extended_metrics(y, np.full(len(y), 1.0))
    maj.update(roc_auc=0.5, pr_auc_1=float(y.mean()), pr_auc_0=float(1 - y.mean()))
    run_rows.append({"model": "majority", "label": "Majority (always 1)", "seed": None,
                     "run_id": None, **maj})
    ids, _, p_rep = read_preds(RUNS / REPRODUCTION_RUN / "predictions_validation.csv")
    run_rows.append({"model": "mm_cls_reproduction", "label": "Baseline reproduction (CPU features)",
                     "seed": 42, "run_id": REPRODUCTION_RUN, **extended_metrics(y, p_rep)})
    runs_df = pd.DataFrame(run_rows)
    runs_df.to_csv(out / "metrics_by_run.csv", index=False, float_format="%.6f")
    summary["runs"] = runs_df.replace({np.nan: None}).to_dict(orient="records")

    sdf = pd.DataFrame(seed_rows)
    cols = ["macro_f1", "accuracy", "balanced_accuracy", "f1_0", "f1_1", "roc_auc", "pr_auc_0",
            "brier", "ece_10", "selection_gap", "best_epoch"]
    agg = sdf.groupby("model", sort=False)[cols].agg(["mean", "std", "min", "max"])
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    agg.insert(0, "label", [m["label"] for m in MODELS])
    agg.to_csv(out / "seed_summary.csv", float_format="%.6f")
    summary["seed_summary"] = json.loads(agg.reset_index().to_json(orient="records"))

    # ---------- bootstrap CIs (seed 42)
    idx = bootstrap_indices(len(y))
    yb = y[idx]
    boot, boot_rows = {}, []
    for key, p in [(m["key"], preds[(m["key"], 42)]) for m in MODELS] + [
            ("image_group_linear", p_img), ("majority", np.full(len(y), 1.0))]:
        pred = (p >= 0.5).astype(int)
        f1s, f10 = macro_f1_batch(yb, pred[idx])
        acc = (yb == pred[idx]).mean(1)
        boot[key] = {"macro_f1_ci": ci(f1s), "accuracy_ci": ci(acc), "f1_0_ci": ci(f10),
                     "macro_f1_boot_sd": float(f1s.std())}
        boot_rows.append({"model": key, "macro_f1_ci_low": boot[key]["macro_f1_ci"][0],
                          "macro_f1_ci_high": boot[key]["macro_f1_ci"][1],
                          "macro_f1_boot_sd": boot[key]["macro_f1_boot_sd"],
                          "accuracy_ci_low": boot[key]["accuracy_ci"][0],
                          "accuracy_ci_high": boot[key]["accuracy_ci"][1],
                          "f1_0_ci_low": boot[key]["f1_0_ci"][0], "f1_0_ci_high": boot[key]["f1_0_ci"][1]})
    pd.DataFrame(boot_rows).to_csv(out / "bootstrap_ci.csv", index=False, float_format="%.6f")
    summary["bootstrap_ci"] = boot_rows

    # ---------- chance tests: label permutations (all runs) + AUC bootstrap (seed 42)
    chance_rows = []
    for (key, seed), p in preds.items():
        row = {"model": key, "seed": seed, **permutation_tests(y, p),
               "chance_macro_f1_at_rate": chance_macro_f1(float(y.mean()), float((p >= 0.5).mean()))}
        if seed == 42:
            aucs = auc_bootstrap(y, p, idx)
            row.update(roc_auc_ci_low=ci(aucs)[0], roc_auc_ci_high=ci(aucs)[1],
                       roc_auc_boot_share_le_0_5=float((aucs <= 0.5).mean()))
        chance_rows.append(row)
    chance = pd.DataFrame(chance_rows)
    chance.to_csv(out / "chance_tests.csv", index=False, float_format="%.6f")
    summary["chance_tests"] = chance.replace({np.nan: None}).to_dict(orient="records")
    summary["base_rate_constant_reference"] = {"p": float(y.mean()), "brier": float(y.mean() * (1 - y.mean())),
                                               "ece_10": 0.0, "macro_f1_label_independent_max": 0.5}

    # ---------- paired comparisons vs the baseline
    comp_rows = []
    base_pred = {s: (preds[("mm_cls", s)] >= 0.5).astype(int) for s in (42, 43, 44)}
    for other in ["image_mlp", "text_mlp", "mm_mean", "image_group_linear", "majority"]:
        if other == "majority":
            other_pred42 = np.ones(len(y), dtype=int)
            other_seeds = None
        elif other == "image_group_linear":
            other_pred42 = (p_img >= 0.5).astype(int)
            other_seeds = None
        else:
            other_pred42 = (preds[(other, 42)] >= 0.5).astype(int)
            other_seeds = {s: (preds[(other, s)] >= 0.5).astype(int) for s in (42, 43, 44)}
        a, _ = macro_f1_batch(yb, base_pred[42][idx])
        b, _ = macro_f1_batch(yb, other_pred42[idx])
        d42 = a - b
        point42 = compute_metrics(y, base_pred[42])["macro_f1"] - compute_metrics(y, other_pred42)["macro_f1"]
        ca, cb = base_pred[42] == y, other_pred42 == y
        n10, n01 = int((ca & ~cb).sum()), int((~ca & cb).sum())
        mc = binomtest(min(n10, n01), n10 + n01, 0.5).pvalue if n10 + n01 else 1.0
        row = {"baseline": "mm_cls", "other": other,
               "delta_macro_f1_seed42": point42, "ci_low_seed42": ci(d42)[0], "ci_high_seed42": ci(d42)[1],
               "bootstrap_p_seed42": paired_p(d42),
               "baseline_only_correct": n10, "other_only_correct": n01, "mcnemar_exact_p_seed42": float(mc)}
        if other_seeds:
            ma = np.mean([macro_f1_batch(yb, base_pred[s][idx])[0] for s in (42, 43, 44)], axis=0)
            mb = np.mean([macro_f1_batch(yb, other_seeds[s][idx])[0] for s in (42, 43, 44)], axis=0)
            dm = ma - mb
            point = np.mean([compute_metrics(y, base_pred[s])["macro_f1"] for s in (42, 43, 44)]) - \
                np.mean([compute_metrics(y, other_seeds[s])["macro_f1"] for s in (42, 43, 44)])
            row.update(delta_macro_f1_seedmean=float(point), ci_low_seedmean=ci(dm)[0],
                       ci_high_seedmean=ci(dm)[1], bootstrap_p_seedmean=paired_p(dm))
        comp_rows.append(row)
    comp = pd.DataFrame(comp_rows)
    comp.to_csv(out / "paired_comparisons.csv", index=False, float_format="%.6f")
    summary["paired_comparisons"] = comp.replace({np.nan: None}).to_dict(orient="records")

    # ---------- agreement between models (seed 42)
    agree = {}
    keys = [m["key"] for m in MODELS] + ["image_group_linear"]
    for i, ka in enumerate(keys):
        for kb in keys[i + 1:]:
            pa = (preds[(ka, 42)] >= 0.5).astype(int)
            pb = (preds[(kb, 42)] >= 0.5).astype(int)
            agree[f"{ka}__{kb}"] = {"label_agreement": float((pa == pb).mean()),
                                    "prob_pearson": float(np.corrcoef(preds[(ka, 42)], preds[(kb, 42)])[0, 1])}
    summary["agreement_seed42"] = agree

    # ---------- reproduction vs main run (same 699 memes)
    pm, pr_ = (preds[("mm_cls", 42)] >= 0.5).astype(int), (p_rep >= 0.5).astype(int)
    d_rep = macro_f1_batch(yb, pr_[idx])[0] - macro_f1_batch(yb, pm[idx])[0]
    t_main = torch.load(f"{cache_dir('distilbert_first_token')}/train.pt", weights_only=True)["features"]
    t_rep_path = Path(f"{cache_dir('reproduction_distilbert_first_token')}/train.pt")
    text_identical = (bool(torch.equal(t_main, torch.load(t_rep_path, weights_only=True)["features"]))
                      if t_rep_path.exists() else None)
    checks = {}
    for run in (MODELS[0]["runs"][42], REPRODUCTION_RUN):
        path = RUNS / run / "checks" / "end_to_end_validation.json"
        if path.exists():
            c = json.loads(path.read_text())
            checks[run] = {k: c[k] for k in ("n", "image_features_dir", "text_max_abs_diff",
                                             "image_min_cosine", "probability_max_abs_diff",
                                             "label_agreement", "passed")}
    summary["reproduction"] = {
        "delta_macro_f1": compute_metrics(y, pr_)["macro_f1"] - compute_metrics(y, pm)["macro_f1"],
        "paired_bootstrap_sd": float(d_rep.std()), "paired_bootstrap_p": paired_p(d_rep),
        "label_agreement": float((pm == pr_).mean()),
        "prob_pearson": float(np.corrcoef(preds[("mm_cls", 42)], p_rep)[0, 1]),
        "text_cache_bit_identical": text_identical, "end_to_end_checks": checks}

    # ---------- modality reliance (baseline checkpoints, cached features)
    rel_rows = []
    for seed, run in MODELS[0]["runs"].items():
        ckpt = torch.load(Path("checkpoints/multimodal") / run / "best.pt", map_location="cpu",
                          weights_only=True)
        cfg = ckpt["config"]
        head = build_fusion_head(cfg)
        head.load_state_dict(ckpt["state_dict"])
        _, ftr, _ = load_split_features(cfg, "train", ckpt["text_features_dir"], ckpt["image_features_dir"],
                                        args.data_dir)
        _, fv, yv = load_split_features(cfg, "validation", ckpt["text_features_dir"],
                                        ckpt["image_features_dir"], args.data_dir)
        clean = predict(head, fv, torch.device("cpu")).numpy()
        ids, _, p_ref = read_preds(RUNS / run / "predictions_validation.csv")
        assert np.abs(clean - p_ref).max() < 1e-6, run
        clean_pred = (clean >= 0.5).astype(int)

        def record(cond, feats, rep):
            p = predict(head, feats, torch.device("cpu")).numpy()
            pr = (p >= 0.5).astype(int)
            flipped = pr != clean_pred
            rel_rows.append({"seed": seed, "condition": cond, "repeat": rep,
                             "macro_f1": compute_metrics(y, pr)["macro_f1"],
                             "accuracy": compute_metrics(y, pr)["accuracy"],
                             "prediction_change_rate": float(flipped.mean()),
                             "flips_now_correct_share": float((pr[flipped] == y[flipped]).mean())
                             if flipped.any() else float("nan"),
                             "mean_abs_delta_p": float(np.abs(p - clean).mean()),
                             "predicted_1_rate": float(pr.mean())})
        record("clean", fv, 0)
        for mod in ("text", "image"):
            for r in range(N_PERM):
                perm = torch.from_numpy(np.random.default_rng(1000 + r).permutation(len(y)))
                record(f"{mod}_shuffled", {**fv, mod: fv[mod][perm]}, r)
            mean_vec = ftr[mod].mean(0, keepdim=True).expand_as(fv[mod])
            record(f"{mod}_mean", {**fv, mod: mean_vec}, 0)
    rel = pd.DataFrame(rel_rows)
    rel.to_csv(out / "modality_reliance.csv", index=False, float_format="%.6f")
    rel_agg = rel.groupby("condition").agg(macro_f1_mean=("macro_f1", "mean"), macro_f1_sd=("macro_f1", "std"),
                                           change_rate_mean=("prediction_change_rate", "mean"),
                                           mean_abs_delta_p=("mean_abs_delta_p", "mean"),
                                           flips_now_correct_share=("flips_now_correct_share", "mean"),
                                           predicted_1_rate=("predicted_1_rate", "mean"))
    summary["modality_reliance"] = json.loads(rel_agg.reset_index().to_json(orient="records"))

    # ---------- feature geometry (train split): how much do L2-normalized vectors differ?
    geo_rows = []
    for name, path in (("text_first_token", f"{cache_dir('distilbert_first_token')}/train.pt"),
                       ("text_mean_pooling", f"{cache_dir('distilbert_mean_pooling')}/train.pt"),
                       ("image_pooler_output", f"{cache_dir('20261006_code_fixes_validation_v2')}/train.pt")):
        x = F.normalize(torch.load(path, weights_only=True)["features"].double(), dim=-1)
        gram = x @ x.T
        off = gram[~torch.eye(len(x), dtype=torch.bool)]  # exact: all ordered pairs i != j
        sv = torch.linalg.svdvals(x - x.mean(0))
        geo_rows.append({"feature": name, "mean_pairwise_cosine": float(off.mean()),
                         "sd_pairwise_cosine": float(off.std()),
                         "norm_of_mean_vector": float(x.mean(0).norm()),
                         "top1_pc_variance_share": float(sv[0] ** 2 / (sv ** 2).sum())})
    pd.DataFrame(geo_rows).to_csv(out / "feature_geometry.csv", index=False, float_format="%.6f")
    summary["feature_geometry"] = geo_rows

    # ---------- standardized logistic-regression probes (diagnostic; every C is reported, none chosen)
    def feats(path, split):
        return F.normalize(torch.load(Path(path) / f"{split}.pt", weights_only=True)["features"], dim=-1).numpy()
    sets = {"text_first_token": [str(cache_dir("distilbert_first_token"))],
            "text_mean_pooling": [str(cache_dir("distilbert_mean_pooling"))],
            "image_pooler_output": [str(cache_dir("20261006_code_fixes_validation_v2"))],
            "text_first_token+image": [str(cache_dir("distilbert_first_token")),
                                       str(cache_dir("20261006_code_fixes_validation_v2"))]}
    ytr = load_split("train", args.data_dir)["sarcasm_binary"].to_numpy()
    probe_rows = []
    for name, dirs in sets.items():
        xtr = np.concatenate([feats(d, "train") for d in dirs], axis=1)
        xva = np.concatenate([feats(d, "validation") for d in dirs], axis=1)
        sc = StandardScaler().fit(xtr)
        xtr, xva = sc.transform(xtr), sc.transform(xva)
        for c in (0.001, 0.01, 0.1, 1.0):
            lr = LogisticRegression(C=c, class_weight="balanced", max_iter=5000).fit(xtr, ytr)
            ptr, pva = lr.predict_proba(xtr)[:, 1], lr.predict_proba(xva)[:, 1]
            probe_rows.append({"features": name, "C": c, "train_roc_auc": float(roc_auc_score(ytr, ptr)),
                               "val_roc_auc": float(roc_auc_score(y, pva)),
                               "val_macro_f1": compute_metrics(y, (pva >= 0.5).astype(int))["macro_f1"]})
    probes = pd.DataFrame(probe_rows)
    probes.to_csv(out / "linear_probes.csv", index=False, float_format="%.6f")
    summary["linear_probes"] = probe_rows

    # ---------- CPU vs CUDA CLIP features
    cos_all = []
    feat = {}
    for split in ("train", "validation"):
        a = torch.load(f"{cache_dir('20261006_code_fixes_validation_v2')}/{split}.pt", weights_only=True)
        b = torch.load(f"{cache_dir('clip_vitb32_pooler_cpu')}/{split}.pt", weights_only=True)
        assert a["sample_id"] == b["sample_id"]
        c = F.cosine_similarity(a["features"].double(), b["features"].double()).numpy()
        cos_all.append(c)
        feat[split] = {"n": int(len(c)), "mean_cosine": float(c.mean()), "min_cosine": float(c.min()),
                       "share_cos_below_0.999": float((c < 0.999).mean()),
                       "share_cos_below_0.9999": float((c < 0.9999).mean()),
                       "mean_abs_diff": float((a["features"] - b["features"]).abs().mean())}
    cos_all = np.concatenate(cos_all)
    (out / "feature_agreement.json").write_text(json.dumps(feat, indent=2), encoding="utf-8")
    summary["feature_agreement"] = feat

    # ---------- figures
    refs = [(float(runs_df.loc[runs_df.model == "majority", "macro_f1"].iloc[0]), "majority 0.435"),
            (0.5, "label-independent chance 0.500")]
    fig_macro_f1(plt, seed_rows, boot, refs, out / "fig1_macro_f1_by_model.png", float(img_row["macro_f1"]))
    logs = {m["key"]: pd.read_csv(RUNS / m["runs"][42] / "training_log.csv") for m in MODELS}
    fig_training(plt, logs, out / "fig2_training_curves.png")
    cms = []
    for m in MODELS:
        pr = (preds[(m["key"], 42)] >= 0.5).astype(int)
        cms.append((m["label"].replace(" (", "\n("), np.array([[((y == i) & (pr == j)).sum() for j in (0, 1)]
                                                              for i in (0, 1)])))
    pr = (p_img >= 0.5).astype(int)
    cms.append(("CLIP linear head\n(image group, CUDA)", np.array([[((y == i) & (pr == j)).sum() for j in (0, 1)] for i in (0, 1)])))
    fig_confusion(plt, cms, out / "fig3_confusion_matrices.png")
    fig_roc_pr(plt, {m["key"]: (y, preds[(m["key"], 42)]) for m in MODELS}, y, out / "fig4_roc_pr.png")
    fig_probabilities(plt, y, preds[("mm_cls", 42)], out / "fig5_probability_distributions.png")
    fig_reliance(plt, rel, out / "fig6_modality_reliance.png")
    fig_features(plt, cos_all, out / "fig7_cpu_vs_cuda_clip_features.png")

    (out / "stats_summary.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    print(f"wrote {len(list(out.iterdir()))} files to {out}")


if __name__ == "__main__":
    main()
