"""Shared evaluation code (README 5.3 / 6). All three groups must use these functions.

Primary metric: Macro-F1. Everything is computed for the two classes 0 (not sarcastic)
and 1 (sarcastic) at a fixed threshold of 0.5 on probability_1.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

PRED_COLUMNS = ["sample_id", "true_label", "predicted_label", "probability_1",
                "model_name", "split", "seed"]
EXPECTED_ROWS = {"validation": 699, "test": 700}


def class_weights_from_labels(labels) -> list[float]:
    """README 5.2: weights from the TRAIN split only.  w_c = N / (2 * n_c).

    train: N=5593, n0=1226, n1=4367  ->  w0 ~= 2.281, w1 ~= 0.640
    A mistake on the rare class (0) costs ~3.6x more than one on class 1.
    """
    labels = np.asarray(labels)
    n = len(labels)
    return [n / (2 * int((labels == c).sum())) for c in (0, 1)]


def compute_metrics(y_true, y_pred) -> dict:
    """Macro-F1, accuracy, and per-class precision / recall / F1."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], average=None, zero_division=0)
    return {
        "macro_f1": float(f.mean()),            # average of the two per-class F1 scores
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_0": float(p[0]), "recall_0": float(r[0]), "f1_0": float(f[0]),
        "precision_1": float(p[1]), "recall_1": float(r[1]), "f1_1": float(f[1]),
    }


def get_confusion_matrix(y_true, y_pred) -> np.ndarray:
    """Rows = true label (0, 1); columns = predicted label (0, 1)."""
    return confusion_matrix(np.asarray(y_true).astype(int), np.asarray(y_pred).astype(int),
                            labels=[0, 1])


def majority_baseline(y_true) -> dict:
    """What 'always predict sarcastic (1)' scores. A real model must beat this."""
    y_true = np.asarray(y_true).astype(int)
    return compute_metrics(y_true, np.ones_like(y_true))


def metrics_json(model_name: str, split: str, seed: int, threshold: float,
                 metrics: dict, training_seconds: float, **extra) -> dict:
    """Exact README 6.2 layout first; optional extras (e.g. validation_loss) appended after."""
    out = {"model_name": model_name, "split": split, "seed": seed, "threshold": threshold}
    for k in ["macro_f1", "accuracy", "precision_0", "recall_0", "f1_0",
              "precision_1", "recall_1", "f1_1"]:
        out[k] = round(metrics[k], 6)
    out["training_seconds"] = round(float(training_seconds), 2)
    out.update(extra)
    return out


def save_json(obj: dict, path: Path) -> None:
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def predictions_frame(sample_ids, y_true, y_pred, prob1, model_name: str,
                      split: str, seed: int) -> pd.DataFrame:
    """README 6.1 prediction CSV, one row per sample, original order kept."""
    df = pd.DataFrame({
        "sample_id": list(sample_ids),
        "true_label": np.asarray(y_true).astype(int),
        "predicted_label": np.asarray(y_pred).astype(int),
        # Preserve inference precision near the decision threshold.
        "probability_1": np.asarray(prob1, dtype=float),
        "model_name": model_name,
        "split": split,
        "seed": int(seed),
    })
    return df[PRED_COLUMNS]


def save_confusion_matrix_png(cm: np.ndarray, path: Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    ax.imshow(cm, cmap="Blues")
    names = ["0 not sarcastic", "1 sarcastic"]
    ax.set_xticks([0, 1], names)
    ax.set_yticks([0, 1], names)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    total = cm.sum()
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]}\n({cm[i, j] / total:.1%})", ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=10)
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
