"""Shared fixtures for the unit tests: synthetic feature caches and tiny random encoders.

Nothing here downloads a model or reads a dataset image. Caches follow the real format
(src/features/cache.py) and use the real metadata.csv sample IDs and labels, so the tests that
build them need MEMOTION_DATA_DIR. The tiny encoders keep the real 768-d widths but have one layer
and random weights.
"""
import json
import os
import tempfile
from pathlib import Path

import torch

from src.data.dataset import EXPECTED_SPLIT_SIZES, load_split
from src.features.cache import file_sha256
from src.utils.config import load_config

ROOT = Path(__file__).resolve().parents[1]
IMAGE_CFG = load_config(ROOT / "configs/train/image_baseline.yaml")
MM_CFG = load_config(ROOT / "configs/train/multimodal_baseline.yaml")
SAMPLES = ROOT / "data/samples/synthetic_memotion"


def work_dir(test, prefix):
    """A fresh directory for one test (kept under REVIEW_TEST_DIR when that is set)."""
    base = os.environ.get("REVIEW_TEST_DIR")
    if base:
        base = Path(base).resolve()
        base.mkdir(parents=True, exist_ok=True)
        return Path(tempfile.mkdtemp(prefix=test._testMethodName + "_", dir=base))
    temp = tempfile.TemporaryDirectory(prefix=prefix)
    test.addCleanup(temp.cleanup)
    return Path(temp.name).resolve()


def text_meta(splits, cfg=MM_CFG, **changes):
    meta = dict(text_model_id=cfg["text_model_id"], text_revision_requested=cfg["text_revision"],
                text_feature=cfg["text_feature"], max_length=cfg["text_max_length"], limit=None,
                splits={s: {"n": EXPECTED_SPLIT_SIZES[s], "dim": 768} for s in splits})
    meta.update(changes)
    return meta


def image_meta(splits, cfg=MM_CFG):
    return dict(clip_model_id=cfg["clip_model_id"], clip_revision_requested=cfg["clip_revision"],
                feature=cfg.get("image_feature", cfg.get("feature")), limit=None,
                splits={s: {"n": EXPECTED_SPLIT_SIZES[s], "dim": 768} for s in splits})


def write_cache(directory, split, features, meta, order=None):
    """<directory>/<split>.pt with the split's real IDs/labels, and meta.json with its SHA-256."""
    truth = load_split(split)
    ids = truth.sample_id.tolist()
    labels = torch.tensor(truth.sarcasm_binary.to_numpy(), dtype=torch.long)
    if order is not None:
        ids, labels, features = [ids[i] for i in order], labels[order], features[order]
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    torch.save(dict(sample_id=ids, features=features, labels=labels), directory / f"{split}.pt")
    meta["splits"][split]["sha256"] = file_sha256(directory / f"{split}.pt")
    (directory / "meta.json").write_text(json.dumps(meta), encoding="utf-8")


def write_train_validation_caches(root, modalities=("text", "image"), cfg=MM_CFG, seed=0):
    """Random 768-d train + validation caches under root/text and root/image."""
    generator = torch.Generator().manual_seed(seed)
    for split in ("train", "validation"):
        n = EXPECTED_SPLIT_SIZES[split]
        for name in modalities:
            directory = Path(root) / name
            meta = ((text_meta if name == "text" else image_meta)(["train", "validation"], cfg)
                    if split == "train" else
                    json.loads((directory / "meta.json").read_text(encoding="utf-8")))
            write_cache(directory, split, torch.randn(n, 768, generator=generator), meta)


def make_tiny_encoders(directory):
    """Save random one-layer DistilBERT (+ tokenizer) and CLIP (+ image processor) checkpoints.

    Returns (text_model_dir, clip_model_dir). Widths are the real 768 so MultimodalClassifier's
    dimension checks pass; everything else is minimal so the test runs in seconds on CPU.
    """
    from transformers import (CLIPConfig, CLIPImageProcessor, CLIPModel, DistilBertConfig,
                              DistilBertModel, DistilBertTokenizerFast)

    directory = Path(directory)
    text_dir, clip_dir = directory / "tiny_distilbert", directory / "tiny_clip"
    words = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "[", "]", "_", "no", "text", "oh",
             "great", "another", "monday", "i", "love", "waiting", "best", "day", "ever", "tea",
             "a", "book", "and", "an", "early", "night", "sunday", "plans", ".", ",", ":"]
    text_dir.mkdir(parents=True)
    (text_dir / "vocab.txt").write_text("\n".join(words) + "\n", encoding="utf-8")
    DistilBertTokenizerFast(vocab_file=str(text_dir / "vocab.txt"),
                            do_lower_case=True).save_pretrained(text_dir)
    torch.manual_seed(0)
    DistilBertModel(DistilBertConfig(vocab_size=len(words), dim=768, n_layers=1, n_heads=2,
                                     hidden_dim=64, max_position_embeddings=128)
                    ).save_pretrained(text_dir)
    config = CLIPConfig(
        text_config=dict(vocab_size=64, hidden_size=32, intermediate_size=32, num_hidden_layers=1,
                         num_attention_heads=2, max_position_embeddings=16),
        vision_config=dict(hidden_size=768, intermediate_size=64, num_hidden_layers=1,
                           num_attention_heads=2, image_size=64, patch_size=32),
        projection_dim=16)
    CLIPModel(config).save_pretrained(clip_dir)
    CLIPImageProcessor(size={"shortest_edge": 64}, crop_size={"height": 64, "width": 64}
                       ).save_pretrained(clip_dir)
    return str(text_dir), str(clip_dir)
