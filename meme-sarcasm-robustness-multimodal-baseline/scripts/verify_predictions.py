"""Check a prediction CSV follows README 6.1 exactly. Works for all three groups.

    python -m scripts.verify_predictions outputs/image/<run_id>/predictions_validation.csv
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.dataset import load_split  # noqa: E402
from src.evaluation.metrics import EXPECTED_ROWS, PRED_COLUMNS  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--data_dir", default=None)
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()

    df = pd.read_csv(args.csv, float_precision="round_trip")
    problems = []

    if list(df.columns) != PRED_COLUMNS:
        problems.append(f"columns {list(df.columns)} != {PRED_COLUMNS}")
        print("\n".join("!! " + p for p in problems)); sys.exit(1)

    splits = df["split"].unique().tolist()
    if len(splits) != 1 or splits[0] not in EXPECTED_ROWS:
        problems.append(f"split column must be one of {list(EXPECTED_ROWS)}, got {splits}")
        print("\n".join("!! " + p for p in problems)); sys.exit(1)
    split = splits[0]

    if len(df) != EXPECTED_ROWS[split]:
        problems.append(f"{len(df)} rows, expected {EXPECTED_ROWS[split]}")
    if not df["sample_id"].is_unique:
        problems.append("duplicate sample_id")
    if df["seed"].nunique() != 1 or df["model_name"].nunique() != 1:
        problems.append("seed and model_name must be constant")
    if not df["probability_1"].between(0, 1).all():
        problems.append("probability_1 outside [0, 1]")
    if not set(df["true_label"]).issubset({0, 1}) or not set(df["predicted_label"]).issubset({0, 1}):
        problems.append("labels must be 0/1")
    expected_pred = (df["probability_1"] >= args.threshold).astype(int)
    if (expected_pred != df["predicted_label"]).any():
        n = int((expected_pred != df["predicted_label"]).sum())
        problems.append(f"{n} rows where predicted_label != (probability_1 >= {args.threshold})")

    meta = load_split(split, args.data_dir)
    if set(df["sample_id"]) != set(meta["sample_id"]):
        problems.append("sample_id set differs from the dataset split")
    else:
        if list(df["sample_id"]) != list(meta["sample_id"]):
            print("note: rows are not in the original sample_id order (allowed, but preferred)")
        truth = meta.set_index("sample_id")["sarcasm_binary"]
        if (df.set_index("sample_id")["true_label"] != truth.loc[df["sample_id"]]).any():
            problems.append("true_label disagrees with sarcasm_binary in the dataset")

    if problems:
        print("\n".join("!! " + p for p in problems))
        print("PREDICTIONS CHECK FAILED"); sys.exit(1)
    print(f"{args.csv}: {len(df)} rows, split={split}, model={df['model_name'].iloc[0]}")
    print("PREDICTIONS OK")


if __name__ == "__main__":
    main()
