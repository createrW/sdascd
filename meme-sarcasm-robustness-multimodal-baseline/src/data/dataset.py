"""Dataset contract (team README section 3). Never hard-code a path: use MEMOTION_DATA_DIR or --data_dir.

All groups use the fixed MMSoc_Memotion_exported split and the binary sarcasm mapping below.
"""
import os
from pathlib import Path

import pandas as pd

from src.data.preprocess import NO_TEXT, clean_text_column, load_image  # noqa: F401  (re-exported)

SPLITS = ("train", "validation", "test")

# README 3.3: 4-class `sarcasm` -> binary `sarcasm_binary`
SARCASM_TO_BINARY = {
    "not_sarcastic": 0,
    "general": 1,
    "twisted_meaning": 1,
    "very_twisted": 1,
}
LABEL_NAMES = {0: "not_sarcastic", 1: "sarcastic"}

# README 3.2 / 3.3 / 5.2: expected counts, used by scripts/verify_dataset.py
EXPECTED_SPLIT_SIZES = {"train": 5593, "validation": 699, "test": 700}
EXPECTED_TRAIN_LABEL_COUNTS = {0: 1226, 1: 4367}
EXPECTED_TOTAL_LABEL_COUNTS = {0: 1544, 1: 5448}

# Repository-relative fallback when MEMOTION_DATA_DIR is not set (see data/README.md).
DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "MMSoc_Memotion_exported"


def get_data_dir(data_dir: str | None = None) -> Path:
    d = data_dir or os.environ.get("MEMOTION_DATA_DIR")
    if not d:
        if (DEFAULT_DATA_DIR / "train" / "metadata.csv").exists():
            return DEFAULT_DATA_DIR
        raise ValueError("Set MEMOTION_DATA_DIR or pass --data_dir "
                         "(the folder that contains train/ validation/ test/), "
                         f"or unpack the dataset to {DEFAULT_DATA_DIR}.")
    p = Path(d)
    if not (p / "train" / "metadata.csv").exists():
        raise FileNotFoundError(f"{p} does not contain train/metadata.csv")
    return p


def load_split(split: str, data_dir: str | None = None) -> pd.DataFrame:
    """Read one split's metadata in original order, with an absolute image path column."""
    assert split in SPLITS, split
    root = get_data_dir(data_dir)
    df = pd.read_csv(root / split / "metadata.csv", encoding="utf-8-sig")
    df["sarcasm_binary"] = df["sarcasm_binary"].astype(int)
    df["text_input"] = clean_text_column(df["text_input"])
    df["image_abspath"] = [str(root / split / p) for p in df["image_path"]]
    return df
