"""Check the local dataset copy matches the team spec (README 3).

Usage:  python -m scripts.verify_dataset --data_dir <path>
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.dataset import SPLITS, load_split, load_image  # noqa: E402
from src.data.dataset import (EXPECTED_SPLIT_SIZES, EXPECTED_TOTAL_LABEL_COUNTS,  # noqa: E402
                               EXPECTED_TRAIN_LABEL_COUNTS, SARCASM_TO_BINARY)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=None)
    ap.add_argument("--skip_images", action="store_true")
    args = ap.parse_args()

    ok = True
    total = Counter()
    for split in SPLITS:
        df = load_split(split, args.data_dir)
        n = len(df)
        counts = df["sarcasm_binary"].value_counts().to_dict()
        total.update(counts)
        print(f"[{split}] rows={n} labels={dict(sorted(counts.items()))}")

        if n != EXPECTED_SPLIT_SIZES[split]:
            print(f"  !! expected {EXPECTED_SPLIT_SIZES[split]} rows"); ok = False
        if not df["sample_id"].is_unique:
            print("  !! duplicate sample_id"); ok = False
        mapped = df["sarcasm"].map(SARCASM_TO_BINARY)
        if (mapped != df["sarcasm_binary"]).any():
            print("  !! sarcasm_binary disagrees with label mapping"); ok = False
        if split == "train" and counts != EXPECTED_TRAIN_LABEL_COUNTS:
            print(f"  !! expected train counts {EXPECTED_TRAIN_LABEL_COUNTS}"); ok = False

        if not args.skip_images:
            bad = []
            for sid, p in zip(df["sample_id"], df["image_abspath"]):
                try:
                    load_image(p)
                except Exception as e:  # noqa: BLE001
                    bad.append((sid, repr(e)))
            print(f"  images unreadable: {len(bad)}")
            for b in bad[:10]:
                print("   ", b)
            ok &= not bad

    if dict(total) != EXPECTED_TOTAL_LABEL_COUNTS:
        print(f"!! total labels {dict(total)} != {EXPECTED_TOTAL_LABEL_COUNTS}"); ok = False
    print("DATASET OK" if ok else "DATASET CHECK FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
