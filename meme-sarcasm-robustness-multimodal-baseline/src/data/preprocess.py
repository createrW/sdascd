"""Input preprocessing shared by training-time feature extraction and serving.

Text: missing or blank captions become `[NO_TEXT]` (README 3.4); the frozen DistilBERT tokenizer
adds [CLS]/[SEP], truncates to the configured length and pads to the longest caption in a batch.
Images: every image is converted to RGB before the official CLIP processor (README 4.2).
"""
from __future__ import annotations

import pandas as pd
from PIL import Image, ImageFile

# One training image (train_004578.png) is truncated; this lets PIL still load it.
ImageFile.LOAD_TRUNCATED_IMAGES = True

NO_TEXT = "[NO_TEXT]"


def clean_text(value) -> str:
    """Caption as used by every model: missing/blank -> [NO_TEXT]."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return NO_TEXT
    text = str(value)
    return text if text.strip() else NO_TEXT


def clean_text_column(series: pd.Series) -> pd.Series:
    out = series.fillna(NO_TEXT).astype(str)
    out[out.str.strip() == ""] = NO_TEXT
    return out


def to_rgb(im: Image.Image) -> Image.Image:
    """Dataset has P, RGBA, L, CMYK and GIF images; CLIP needs RGB."""
    if im.mode == "P":
        im = im.convert("RGBA")  # keeps palette transparency correct
    return im.convert("RGB")


def load_image(path: str) -> Image.Image:
    """Always return an RGB copy of the image at `path`."""
    with Image.open(path) as im:
        return to_rgb(im)


def tokenize(tokenizer, texts: list[str], max_length: int) -> dict:
    """[CLS] text [SEP], WordPiece (uncased), truncated to max_length, padded to the batch maximum."""
    return tokenizer(texts, padding=True, truncation=True, max_length=max_length,
                     return_tensors="pt")
