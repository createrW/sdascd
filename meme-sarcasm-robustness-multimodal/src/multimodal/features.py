"""Joint loader for cached text + image features, aligned by sample_id (README 3.2 / 7.3).

Both caches are validated first (counts, IDs and order against metadata.csv, labels, dims, dtype,
NaN/inf, SHA-256) by the same checks the image group uses, then joined on sample_id.
"""
from pathlib import Path

import torch

from src.image.cache import load_features as load_image_features
from src.text.cache import load_text_features

# The image group's reference cache (Image_Features_CLIP_ViTB32_20261006.zip, train/validation).
DEFAULT_IMAGE_FEATURES_DIR = "features/20261006_code_fixes_validation_v2"


def default_text_features_dir(config: dict) -> str:
    return f"features/distilbert_{config['text_feature']}"


def image_cache_config(config: dict) -> dict:
    """Keys src/image/cache.py checks, taken from the multimodal config."""
    return {"clip_model_id": config["clip_model_id"], "clip_revision": config["clip_revision"],
            "feature": config["image_feature"]}


def align_by_sample_id(ids, x, y, target_ids):
    """Reorder (x, y) from `ids` order into `target_ids` order; every ID must be present once."""
    position = {sid: i for i, sid in enumerate(ids)}
    if len(position) != len(ids) or set(position) != set(target_ids):
        raise ValueError("text and image caches do not contain the same sample IDs")
    index = torch.tensor([position[sid] for sid in target_ids], dtype=torch.long)
    return x[index], y[index]


def load_split_features(config, split, text_features_dir=None, image_features_dir=None,
                        data_dir=None, modalities=None):
    """Return (sample_ids, {"text": [N, 768], "image": [N, 768]}, labels[N]) for one split.

    Order follows the split's metadata.csv; only the requested modalities are loaded.
    """
    modalities = tuple(modalities or config.get("modalities", ("text", "image")))
    loaded = {}
    if "text" in modalities:
        loaded["text"] = load_text_features(
            Path(text_features_dir or default_text_features_dir(config)), split, config, data_dir)
    if "image" in modalities:
        loaded["image"] = load_image_features(
            Path(image_features_dir or DEFAULT_IMAGE_FEATURES_DIR), split,
            image_cache_config(config), data_dir)
    if not loaded:
        raise ValueError("no modality requested")

    ids, _, labels = next(iter(loaded.values()))
    features = {}
    for name, (m_ids, x, y) in loaded.items():
        x, y = align_by_sample_id(m_ids, x, y, ids)
        if not torch.equal(y, labels):
            raise ValueError(f"{split}: {name} labels disagree after sample_id alignment")
        features[name] = x
    return list(ids), features, labels
