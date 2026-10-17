"""Run-level report artifacts (README section 6): metrics JSON, prediction CSV, confusion matrix.

Every training or evaluation run writes, under outputs/<task>/<run_id>/:
    config.json  metrics_<split>.json  predictions_<split>.csv  confusion_matrix_<split>.png
    training_log.csv  README.md
The multimodal results report (statistics and figures across runs) is in results_report.py.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def save_json(obj: dict, path: Path) -> None:
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


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
