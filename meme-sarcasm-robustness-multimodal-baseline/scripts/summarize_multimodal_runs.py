"""Collect validation metrics of multimodal runs into one table (per seed and mean +/- std).

    python -m scripts.summarize_multimodal_runs --pattern '20261016_*' --out outputs/validation_runs_mine.csv
    cmp outputs/validation_runs_mine.csv Team_Review/multimodal/validation_runs.csv
    (the pattern keeps the 2026-10-16 runs of the committed table; '*' also lists the 20261018_overfit_* sweep;
    --out never overwrites an existing file)

Reads outputs/multimodal/<run_id>/metrics_validation.json for every run whose id matches
--pattern; reproduction/rerun folders are skipped unless --include_reproductions is given.
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.safety import require_new_paths  # noqa: E402

KEYS = ["macro_f1", "accuracy", "f1_0", "f1_1", "best_epoch", "training_seconds"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_dir", default="outputs/multimodal")
    ap.add_argument("--pattern", default="*")
    ap.add_argument("--out", default=None, help="optional CSV of per-run rows")
    ap.add_argument("--include_reproductions", action="store_true")
    args = ap.parse_args()

    rows = []
    for path in sorted(Path(args.runs_dir).glob(f"{args.pattern}/metrics_validation.json")):
        run_id = path.parent.name
        if not args.include_reproductions and ("reproduction" in run_id or "rerun" in run_id):
            continue
        m = json.loads(path.read_text(encoding="utf-8"))
        rows.append({"run_id": run_id, "model_name": m["model_name"], "seed": m["seed"],
                     **{k: m.get(k) for k in KEYS}})
    if not rows:
        raise SystemExit("no runs found")
    df = pd.DataFrame(rows).sort_values(["model_name", "seed"])
    if args.out:
        require_new_paths(Path(args.out))
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.out, index=False)

    print("| model | seed | Macro-F1 | Accuracy | F1 class 0 | F1 class 1 | best epoch | train time |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for _, r in df.iterrows():
        print(f"| {r.model_name} | {r.seed} | {r.macro_f1:.4f} | {r.accuracy:.4f} | {r.f1_0:.4f} | "
              f"{r.f1_1:.4f} | {r.best_epoch} | {r.training_seconds:.1f}s |")
    print()
    print("| model | seeds | Macro-F1 mean ± std | Accuracy mean ± std | F1 class 0 mean ± std |")
    print("|---|---:|---:|---:|---:|")
    for name, g in df.groupby("model_name", sort=False):
        def ms(col):
            return f"{g[col].mean():.4f} ± {g[col].std(ddof=1) if len(g) > 1 else 0:.4f}"
        print(f"| {name} | {len(g)} | {ms('macro_f1')} | {ms('accuracy')} | {ms('f1_0')} |")


if __name__ == "__main__":
    main()
