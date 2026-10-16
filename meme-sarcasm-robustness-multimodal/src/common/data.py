"""Shared data loading. Never hard-code a path: use MEMOTION_DATA_DIR or --data_dir."""
import os
from pathlib import Path

import pandas as pd
from PIL import Image, ImageFile

# One training image (train_004578.png) is truncated; this lets PIL still load it.
ImageFile.LOAD_TRUNCATED_IMAGES = True

SPLITS = ("train", "validation", "test")
NO_TEXT = "[NO_TEXT]"


def get_data_dir(data_dir: str | None = None) -> Path:
    d = data_dir or os.environ.get("MEMOTION_DATA_DIR")
    if not d:
        raise ValueError("Set MEMOTION_DATA_DIR or pass --data_dir "
                         "(the folder that contains train/ validation/ test/).")
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
    df["text_input"] = df["text_input"].fillna(NO_TEXT).astype(str)
    df.loc[df["text_input"].str.strip() == "", "text_input"] = NO_TEXT
    df["image_abspath"] = [str(root / split / p) for p in df["image_path"]]
    return df


def load_image(path: str) -> Image.Image:
    """Always return RGB (dataset has P, RGBA, L, CMYK and GIF images)."""
    with Image.open(path) as im:
        if im.mode == "P":
            im = im.convert("RGBA")  # keeps palette transparency correct
        return im.convert("RGB")
