"""Step 7: look at what the image model gets right and wrong.

Picks the most CONFIDENT examples in each cell of the confusion matrix and draws them as
contact sheets. Each thumbnail is the ORIGINAL image with a box showing the square that
CLIP's center crop actually keeps, so you can see when caption text was cut off.
Also prints a few simple checks (tall images, long captions) to test hunches with numbers.

Usage (repo root, MEMOTION_DATA_DIR set):
    python -m src.image.error_analysis --run_id 20261001_clip_vitb32_seed42
    python -m src.image.error_analysis --run_id 20261001_clip_vitb32_seed42 --n 12

Writes outputs/image/<run_id>/error_analysis/:
    false_positive.png   true 0, predicted 1  (model sure it is sarcastic, it is not)
    false_negative.png   true 1, predicted 0  (model sure it is NOT sarcastic, it is)
    true_positive.png    true 1, predicted 1  (most confident correct sarcastic)
    true_negative.png    true 0, predicted 0  (most confident correct not-sarcastic)
    examples.csv         every example shown, with its numbers
    summary.md           the simple checks below, ready to paste into notes
"""
import argparse
import sys
import warnings
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as patches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image, ImageFile  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.common.data import load_image, load_split  # noqa: E402

ImageFile.LOAD_TRUNCATED_IMAGES = True
warnings.filterwarnings("ignore", message="Glyph .* missing from font")  # non-Latin captions

# name, true label, predicted label, sort so the most confident come first
GROUPS = [
    ("false_positive", 0, 1, "probability_1", False),  # highest P(sarcastic) first
    ("false_negative", 1, 0, "probability_1", True),   # lowest  P(sarcastic) first
    ("true_positive", 1, 1, "probability_1", False),
    ("true_negative", 0, 0, "probability_1", True),
]
TITLES = {
    "false_positive": "False positive: NOT sarcastic, model said sarcastic",
    "false_negative": "False negative: sarcastic, model said NOT sarcastic",
    "true_positive": "True positive: sarcastic, model agreed (most confident)",
    "true_negative": "True negative: NOT sarcastic, model agreed (most confident)",
}


def image_sizes(paths):
    """Width and height of every image (header only, fast)."""
    sizes = []
    for p in paths:
        with Image.open(p) as im:
            sizes.append(im.size)
    return sizes


def crop_box(w, h):
    """The square CLIP keeps after 'resize short side + center crop', in original pixels."""
    s = min(w, h)
    return (w - s) / 2, (h - s) / 2, s


def contact_sheet(rows: pd.DataFrame, title: str, path: Path, cols: int = 4):
    n = len(rows)
    if n == 0:
        return
    r = int(np.ceil(n / cols))
    fig, axes = plt.subplots(r, cols, figsize=(4.0 * cols, 4.9 * r))
    axes = np.atleast_1d(axes).ravel()
    for ax, (_, row) in zip(axes, rows.iterrows()):
        img = load_image(row["image_abspath"])
        w, h = img.size
        ax.imshow(img)
        x0, y0, s = crop_box(w, h)
        ax.add_patch(patches.Rectangle((x0, y0), s, s, fill=False, edgecolor="lime",
                                       linewidth=2, linestyle="--"))
        text = textwrap.shorten(str(row["text_input"]), width=110, placeholder=" ...")
        ax.set_title(f"{row['sample_id']}\ntrue={row['true_label']}  P(sarc)={row['probability_1']:.2f}"
                     f"  {w}x{h}", fontsize=9)
        ax.text(0.5, -0.03, textwrap.fill(text, 48), transform=ax.transAxes, ha="center", va="top",
                fontsize=7.5, wrap=True)
        ax.axis("off")
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle(title + "   (green box = what CLIP sees)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=110)
    plt.close(fig)


def rate_table(df: pd.DataFrame, by: str) -> pd.DataFrame:
    """Error rate and Macro-F1-relevant counts per bucket."""
    g = df.groupby(by, observed=True)
    out = pd.DataFrame({
        "n": g.size(),
        "share_sarcastic": g["true_label"].mean().round(3),
        "error_rate": g["wrong"].mean().round(3),
        "mean_P_sarc": g["probability_1"].mean().round(3),
    })
    return out


def md(table: pd.DataFrame) -> str:
    """DataFrame -> markdown table without extra packages."""
    cols = [table.index.name or ""] + list(table.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for idx, *vals in table.itertuples():
        out.append("| " + " | ".join([str(idx)] + [str(v) for v in vals]) + " |")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_id", required=True)
    ap.add_argument("--n", type=int, default=8, help="examples per group")
    ap.add_argument("--data_dir", default=None)
    args = ap.parse_args()

    run_dir = Path("outputs/image") / args.run_id
    out_dir = run_dir / "error_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    pred = pd.read_csv(run_dir / "predictions_validation.csv")
    meta = load_split("validation", args.data_dir)[["sample_id", "image_abspath", "text_input"]]
    df = pred.merge(meta, on="sample_id", how="left", validate="one_to_one")
    df["wrong"] = (df["true_label"] != df["predicted_label"]).astype(int)

    # ---------- simple measurable hunches ----------
    sizes = image_sizes(df["image_abspath"])
    df["width"] = [s[0] for s in sizes]
    df["height"] = [s[1] for s in sizes]
    df["h_over_w"] = (df["height"] / df["width"]).round(3)
    df["shape"] = pd.cut(df["h_over_w"], [0, 0.8, 1.25, 100],
                         labels=["wide (h/w<0.8)", "squareish", "tall (h/w>1.25)"])
    df["kept_by_crop"] = (np.minimum(df["width"], df["height"]) ** 2 /
                          (df["width"] * df["height"])).round(3)
    n_words = df["text_input"].str.split().str.len()
    df["text_len"] = pd.cut(n_words, [-1, 8, 20, 10_000],
                            labels=["short (<=8 words)", "medium (9-20)", "long (>20)"])

    shape_tab = rate_table(df, "shape")
    text_tab = rate_table(df, "text_len")
    conf = df.assign(confidence=(df["probability_1"] - 0.5).abs())
    conf["confidence_bin"] = pd.cut(conf["confidence"], [-0.01, 0.1, 0.25, 0.5],
                                    labels=["unsure (0.4-0.6)", "medium", "sure (<0.25 or >0.75)"])
    conf_tab = rate_table(conf, "confidence_bin")

    # ---------- contact sheets ----------
    shown = []
    for name, t, p, col, ascending in GROUPS:
        sub = df[(df["true_label"] == t) & (df["predicted_label"] == p)]
        sub = sub.sort_values(col, ascending=ascending).head(args.n)
        contact_sheet(sub, f"{TITLES[name]}  [{len(df[(df.true_label == t) & (df.predicted_label == p)])} total]",
                      out_dir / f"{name}.png")
        shown.append(sub.assign(group=name))
        print(f"{name:>15}: {len(sub)} shown -> {out_dir / (name + '.png')}")
    cols = ["group", "sample_id", "true_label", "predicted_label", "probability_1", "width", "height",
            "h_over_w", "kept_by_crop", "text_input"]
    pd.concat(shown)[cols].to_csv(out_dir / "examples.csv", index=False)

    # ---------- summary ----------
    lines = [
        f"# Error analysis: {args.run_id} (validation, {len(df)} memes)",
        "",
        f"Overall error rate: {df['wrong'].mean():.3f}",
        "",
        "## By image shape (center crop cuts tall/wide images)",
        "", md(shape_tab), "",
        f"Average share of the image CLIP keeps after cropping: {df['kept_by_crop'].mean():.2f} "
        f"(tall images: {df.loc[df['shape'] == 'tall (h/w>1.25)', 'kept_by_crop'].mean():.2f})",
        "",
        "## By caption length",
        "", md(text_tab), "",
        "## By model confidence (is the model wrong less often when it is sure?)",
        "", md(conf_tab), "",
        "## My observations (fill in after looking at the four PNGs)",
        "",
        "1. ",
        "2. ",
        "3. ",
        "",
        "Caution: groups with few memes give unreliable rates; treat these as hints, not results.",
    ]
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n" + "\n".join(lines[:-7]))
    print("\nfull summary ->", out_dir / "summary.md")


if __name__ == "__main__":
    main()
