"""Validate complete frozen-DistilBERT feature caches (same layout and checks as src/image/cache.py)."""
import json
from pathlib import Path

import torch

from src.common.data import load_split
from src.common.labels import EXPECTED_SPLIT_SIZES
from src.image.cache import file_sha256

TEXT_DIM = 768


def load_text_features(features_dir, split, config, data_dir=None):
    """Return (sample_ids, FloatTensor[N, 768], LongTensor[N]) after checking provenance and alignment.

    `config` needs text_model_id, text_feature and text_max_length (configs/multimodal_baseline.json).
    """
    features_dir = Path(features_dir)
    meta = json.loads((features_dir / "meta.json").read_text(encoding="utf-8"))
    if meta.get("limit") is not None:
        raise ValueError("Smoke/limited caches cannot be used for formal training or evaluation.")
    for key, expected in (("text_model_id", config["text_model_id"]),
                          ("text_feature", config["text_feature"]),
                          ("max_length", config["text_max_length"])):
        if meta.get(key) != expected:
            raise ValueError(f"Cache {key}={meta.get(key)!r}; expected {expected!r}")
    count = EXPECTED_SPLIT_SIZES[split]
    record = meta.get("splits", {}).get(split, {})
    if record.get("n") != count or record.get("dim") != TEXT_DIM:
        raise ValueError(f"Incomplete or incompatible {split} cache metadata: {record}")
    path = features_dir / f"{split}.pt"
    if record.get("sha256") and file_sha256(path) != record["sha256"]:
        raise ValueError(f"{split}: feature cache SHA-256 mismatch")
    cached = torch.load(path, map_location="cpu", weights_only=True)
    ids, x, y = cached["sample_id"], cached["features"], cached["labels"]
    if not isinstance(ids, list) or len(ids) != count or any(not isinstance(i, str) for i in ids):
        raise ValueError(f"{split}: expected {count} string sample IDs")
    if len(set(ids)) != count:
        raise ValueError(f"{split}: duplicate sample IDs")
    if not isinstance(x, torch.Tensor) or x.shape != (count, TEXT_DIM) or x.dtype != torch.float32:
        raise ValueError(f"{split}: expected float32 features of shape ({count}, {TEXT_DIM})")
    if not torch.isfinite(x).all():
        raise ValueError(f"{split}: NaN/inf in features")
    if not isinstance(y, torch.Tensor) or y.shape != (count,) or y.dtype != torch.int64:
        raise ValueError(f"{split}: expected int64 labels of shape ({count},)")
    truth = load_split(split, data_dir)
    if len(truth) != count or ids != truth["sample_id"].tolist():
        raise ValueError(f"{split}: sample IDs/order differ from the complete dataset")
    expected_y = torch.tensor(truth["sarcasm_binary"].to_numpy(), dtype=torch.long)
    if not torch.equal(y, expected_y):
        raise ValueError(f"{split}: labels differ from dataset metadata")
    return ids, x.float(), y
