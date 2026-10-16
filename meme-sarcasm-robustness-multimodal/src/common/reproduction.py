"""Strict comparisons used by both evaluation and the reproduction pipeline."""
from decimal import Decimal
import json
import math
from pathlib import Path

import pandas as pd

from src.common.metrics import PRED_COLUMNS


def compare_metrics(reference_path, candidate, tolerance=0.01):
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("Reproduction tolerance must be finite and non-negative")
    reference = json.loads(Path(reference_path).read_text(encoding="utf-8"))
    for key in ("model_name", "split", "seed", "threshold"):
        if key not in reference or key not in candidate or reference[key] != candidate[key]:
            raise ValueError(f"Reproduction metadata mismatch: {key}")
    values = (reference["macro_f1"], candidate["macro_f1"])
    if not all(isinstance(v, (int, float)) and math.isfinite(v) and 0 <= v <= 1 for v in values):
        raise ValueError("Macro-F1 must be finite and within [0, 1]")
    diff = abs(values[0] - values[1])
    # Compare the recorded decimal values exactly at the tolerance boundary.
    passed = abs(Decimal(str(values[0])) - Decimal(str(values[1]))) <= Decimal(str(tolerance))
    print(f"reported Macro-F1 {values[0]:.6f} vs now {values[1]:.6f}; "
          f"difference={diff:.6f}, tolerance={tolerance:g} -> {'MATCH' if passed else 'DIFFERENT'}")
    return passed


def compare_predictions(reference_path, candidate_path):
    frames = [pd.read_csv(p, float_precision="round_trip")
              for p in (reference_path, candidate_path)]
    for frame in frames:
        if list(frame.columns) != PRED_COLUMNS or frame.empty or frame.isna().any().any():
            raise ValueError("Invalid reproduction prediction CSV")
        if not frame["sample_id"].is_unique:
            raise ValueError("Duplicate reproduction sample IDs")
    a, b = [f.set_index("sample_id").sort_index() for f in frames]
    passed = a.equals(b)
    print(f"Checkpoint reload predictions -> {'IDENTICAL' if passed else 'DIFFERENT'}")
    return passed
