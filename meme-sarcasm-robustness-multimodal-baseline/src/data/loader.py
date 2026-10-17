"""Joint loader for cached features, aligned by sample_id (README 3.2 / 7.3).

Every cache is validated first (counts, IDs and order against metadata.csv, labels, dims, dtype,
NaN/inf, SHA-256) by src/features/cache.py, then the modalities are joined on sample_id.
Feature directories come from the run config (`features:` block in configs/train/*.yaml) unless a
command-line override is given.
"""
from pathlib import Path

import torch

from src.features.cache import load_image_features, load_text_features
from src.utils.config import infer_task

FEATURES_ROOT = "data/processed/features"
# The image group's reference cache (Image_Features_CLIP_ViTB32_20261006.zip, train/validation);
# the multimodal runs were trained on it.
DEFAULT_IMAGE_FEATURES_DIR = f"{FEATURES_ROOT}/20261006_code_fixes_validation_v2"
# Where `feature_engineering image` writes a full extraction; the image-only baseline's default.
DEFAULT_IMAGE_TASK_FEATURES_DIR = f"{FEATURES_ROOT}/clip_vitb32_pooler"


def default_text_features_dir(config: dict) -> str:
    return f"{FEATURES_ROOT}/distilbert_{config['text_feature']}"


def features_dirs(config: dict, text_dir=None, image_dir=None) -> dict:
    """{"text": dir, "image": dir}: command-line override > config `features:` > default."""
    configured = config.get("features") or {}
    task = infer_task(config)
    default_image = DEFAULT_IMAGE_TASK_FEATURES_DIR if task == "image" else DEFAULT_IMAGE_FEATURES_DIR
    dirs = {"image": image_dir or configured.get("image") or default_image}
    if task == "multimodal":
        dirs["text"] = text_dir or configured.get("text") or default_text_features_dir(config)
    return {k: str(v) for k, v in dirs.items()}


def image_cache_config(config: dict) -> dict:
    """Keys src/features/cache.py checks for a CLIP cache, from an image or multimodal config."""
    return {"clip_model_id": config["clip_model_id"], "clip_revision": config["clip_revision"],
            "feature": config.get("image_feature", config.get("feature"))}


def align_by_sample_id(ids, x, y, target_ids):
    """Reorder (x, y) from `ids` order into `target_ids` order; every ID must be present once."""
    position = {sid: i for i, sid in enumerate(ids)}
    if len(position) != len(ids) or set(position) != set(target_ids):
        raise ValueError("text and image caches do not contain the same sample IDs")
    index = torch.tensor([position[sid] for sid in target_ids], dtype=torch.long)
    return x[index], y[index]


def load_split_features(config, split, text_features_dir=None, image_features_dir=None,
                        data_dir=None, modalities=None):
    """Return (sample_ids, {"text": [N, 768], "image": [N, D]}, labels[N]) for one split.

    Order follows the split's metadata.csv. For an image-task config only "image" is loaded; for a
    multimodal config, the modalities listed in the config (or `modalities`).
    """
    dirs = features_dirs(config, text_features_dir, image_features_dir)
    if infer_task(config) == "image":
        modalities = ("image",)
    modalities = tuple(modalities or config.get("modalities", ("text", "image")))
    loaded = {}
    if "text" in modalities:
        loaded["text"] = load_text_features(Path(dirs["text"]), split, config, data_dir)
    if "image" in modalities:
        loaded["image"] = load_image_features(Path(dirs["image"]), split,
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
