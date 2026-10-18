"""Make a manual-review sheet of image-text-label triples (README 7.3 "manually verify").

Picks validation samples from a run's predictions (default 4 each of TP, TN, FP, FN, seeded),
draws image + caption + labels in one grid, and writes a CSV with empty columns for the
reviewer to fill in.

    python -m src.evaluation.review_samples --run_id 20261016_distilbert_clip_seed42 \
        --out_dir outputs/multimodal/20261016_distilbert_clip_seed42/review_mine

Writes outputs/multimodal/<run_id>/review/{review_grid.png, manual_review.csv}, or --out_dir
(the shipped reference run already has review/; existing files are never overwritten).
"""
import argparse
import sys
import textwrap
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.dataset import load_image, load_split  # noqa: E402
from src.utils.safety import require_new_paths, validate_run_id  # noqa: E402

GROUPS = (("TP", 1, 1), ("TN", 0, 0), ("FP", 0, 1), ("FN", 1, 0))  # name, true, predicted


def select(preds, per_group, seed):
    parts = []
    for name, true, pred in GROUPS:
        pool = preds[(preds.true_label == true) & (preds.predicted_label == pred)]
        take = pool.sample(n=min(per_group, len(pool)), random_state=seed).sort_index()
        parts.append(take.assign(group=name))
    return pd.concat(parts)


def draw(rows, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cols = 4
    n_rows = (len(rows) + cols - 1) // cols
    fig, axes = plt.subplots(n_rows, cols, figsize=(4.2 * cols, 5.6 * n_rows))
    for ax in axes.ravel():
        ax.axis("off")
    for ax, (_, r) in zip(axes.ravel(), rows.iterrows()):
        image = load_image(r.image_abspath)
        image.thumbnail((360, 360))
        ax.imshow(image)
        caption = textwrap.shorten(r.text_input, width=200, placeholder=" ...")
        ax.set_title(f"{r.group}  {r.sample_id}\ntrue {r.true_label} ({r.sarcasm})  "
                     f"pred {r.predicted_label}  p1={r.probability_1:.3f}", fontsize=9)
        ax.text(0.0, -0.02, "\n".join(textwrap.wrap(caption, 52)[:5]), transform=ax.transAxes,
                fontsize=7.5, va="top", ha="left", wrap=True)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_id", required=True)
    ap.add_argument("--per_group", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data_dir", default=None)
    ap.add_argument("--out_dir", default=None, help="default: outputs/multimodal/<run_id>/review")
    args = ap.parse_args(argv)
    validate_run_id(args.run_id)

    run_dir = Path("outputs/multimodal") / args.run_id
    out_dir = Path(args.out_dir) if args.out_dir else run_dir / "review"
    require_new_paths(out_dir)
    preds = pd.read_csv(run_dir / "predictions_validation.csv", float_precision="round_trip")
    meta = load_split("validation", args.data_dir).set_index("sample_id")
    rows = select(preds, args.per_group, args.seed).set_index("sample_id")
    rows = rows.join(meta[["text_input", "sarcasm", "sarcasm_binary", "image_path",
                           "image_abspath"]]).reset_index()
    if (rows.sarcasm_binary != rows.true_label).any():
        raise ValueError("prediction CSV labels disagree with metadata")

    out_dir.mkdir(parents=True)
    draw(rows, out_dir / "review_grid.png",
         f"{args.run_id} | validation | {args.per_group} per TP/TN/FP/FN (seed {args.seed})")
    sheet = rows[["group", "sample_id", "image_path", "text_input", "sarcasm", "true_label",
                  "predicted_label", "probability_1"]].copy()
    for column in ("image_matches_sample_id", "caption_matches_image_text", "label_plausible",
                   "reviewer", "notes"):
        sheet[column] = ""
    sheet.to_csv(out_dir / "manual_review.csv", index=False, encoding="utf-8")
    print(f"{len(rows)} samples -> {out_dir}")


if __name__ == "__main__":
    main()
