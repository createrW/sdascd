"""Inference wrapper: raw caption + raw image -> P(sarcastic) from a trained multimodal head.

MultimodalPredictor loads the frozen DistilBERT and CLIP encoders, the tokenizer and the CLIP image
processor once, and puts a trained fusion head (a FusionHead checkpoint from src/models/trainer.py)
on top: MultimodalClassifier in src/models/multimodal_classifier.py. The preprocessing is the
training-time one (src/data/preprocess.py): blank caption -> [NO_TEXT], RGB conversion, the
official CLIP processor, DistilBERT WordPiece truncated to text_max_length. It is what
src/serving/app.py serves and what src/evaluation/end_to_end.py checks against the cached-feature
pipeline.

    python -m src.serving.predictor --smoke                         # 2 validation samples
    python -m src.serving.predictor --text "nice weather" --image path/to/meme.jpg
    python -m src.serving.predictor --smoke --train_config configs/train/multimodal_baseline.yaml
        # untrained head: checks the encoders, preprocessing and shapes before any training

Paths and model locations come from configs/deploy/serving.yaml (command-line flags override).
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoTokenizer, CLIPProcessor

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.dataset import LABEL_NAMES, load_split  # noqa: E402
from src.data.preprocess import clean_text, load_image, to_rgb, tokenize  # noqa: E402
from src.features.feature_engineering import text_source  # noqa: E402
from src.models.multimodal_classifier import build_model, load_fusion_head  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.reproducibility import get_device, set_seed  # noqa: E402

DEFAULT_SERVING_CONFIG = "configs/deploy/serving.yaml"


def as_rgb_image(image) -> Image.Image:
    """PIL image, file path or encoded bytes -> RGB PIL image (same conversion as training)."""
    if isinstance(image, Image.Image):
        return image if image.mode == "RGB" else to_rgb(image)  # RGB already: no extra full-size copy
    if isinstance(image, (bytes, bytearray)):
        with Image.open(io.BytesIO(image)) as im:
            return to_rgb(im)
    return load_image(str(image))


class MultimodalPredictor:
    """Frozen DistilBERT + frozen CLIP + trained fusion MLP, ready for batched inference."""

    def __init__(self, config: dict, head_state_dict: dict | None = None,
                 text_model_dir: str | None = None, clip_model_dir: str | None = None,
                 device: str | None = None, info: dict | None = None) -> None:
        if tuple(config.get("modalities", ("text", "image"))) != ("text", "image"):
            raise ValueError("serving needs a text+image fusion head")
        self.config = config
        self.threshold = float(config["threshold"])
        self.device = torch.device(device) if device else get_device()
        self.model = build_model(config, text_model_dir, clip_model_dir).to(self.device).eval()
        self.trained = head_state_dict is not None
        if self.trained:
            load_fusion_head(self.model, head_state_dict)
        source, kwargs = text_source(config, text_model_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(source, **kwargs)
        self.processor = (
            AutoImageProcessor.from_pretrained(clip_model_dir) if clip_model_dir else
            CLIPProcessor.from_pretrained(config["clip_model_id"], revision=config["clip_revision"]))
        self.info = {
            "model_name": config["model_name"],
            "trained_head": self.trained,
            "threshold": self.threshold,
            "labels": {str(k): v for k, v in LABEL_NAMES.items()},
            "text_model": text_model_dir or f"{config['text_model_id']}@{config['text_revision']}",
            "text_feature": config["text_feature"],
            "text_max_length": config["text_max_length"],
            "clip_model": clip_model_dir or f"{config['clip_model_id']}@{config['clip_revision']}",
            "image_feature": config["image_feature"],
            "device": str(self.device),
            **(info or {}),
        }

    @classmethod
    def from_checkpoint(cls, checkpoint, text_model_dir=None, clip_model_dir=None, device=None):
        """Load checkpoints/multimodal/<run_id>/best.pt (config + fusion head weights)."""
        if not Path(checkpoint).is_file():
            raise FileNotFoundError(
                f"{checkpoint} not found. The reference heads are in Multimodal_Assets_20261016.zip "
                "(docs/Multimodal_Assets.md); to serve your own run pass --checkpoint "
                "checkpoints/multimodal/<run_id>/best.pt or set `checkpoint:` in the serving config.")
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
        info = {"checkpoint": str(checkpoint), "checkpoint_epoch": ckpt["epoch"],
                "checkpoint_val_macro_f1": ckpt["val_macro_f1"]}
        return cls(ckpt["config"], ckpt["state_dict"], text_model_dir, clip_model_dir, device, info)

    @classmethod
    def from_serving_config(cls, path=DEFAULT_SERVING_CONFIG, **overrides):
        """Build from configs/deploy/serving.yaml; keyword overrides that are not None win."""
        cfg = {**load_config(path), **{k: v for k, v in overrides.items() if v is not None}}
        return cls.from_checkpoint(cfg["checkpoint"], cfg.get("text_model_dir"),
                                   cfg.get("clip_model_dir"), cfg.get("device"))

    def pixel_values(self, images) -> torch.Tensor:
        """CLIP processor output [N, 3, 224, 224]; images are decoded and resized one at a time
        (the processor treats every image independently), so only one full-size image is in memory."""
        return torch.cat([self.processor(images=[as_rgb_image(im)], return_tensors="pt")["pixel_values"]
                          for im in images])

    @torch.no_grad()
    def forward(self, texts, images=None, pixel_values=None) -> dict[str, torch.Tensor]:
        """Batched forward pass; returns the encoder features, logits and P(class 1), on CPU.

        Give either raw `images` (paths, bytes or PIL images) or precomputed `pixel_values`.
        """
        if images is None and pixel_values is None:
            raise ValueError("give images or pixel_values")
        n_images = len(images if pixel_values is None else pixel_values)
        if len(texts) != n_images or not texts:
            raise ValueError("need the same, nonzero number of texts and images")
        pixels = self.pixel_values(images) if pixel_values is None else pixel_values
        enc = tokenize(self.tokenizer, [clean_text(t) for t in texts],
                       self.config["text_max_length"])
        out = self.model(input_ids=enc["input_ids"].to(self.device),
                         attention_mask=enc["attention_mask"].to(self.device),
                         pixel_values=pixels.to(self.device), return_features=True)
        out = {k: v.cpu() for k, v in out.items()}
        out["probability_1"] = out["logits"].softmax(dim=-1)[:, 1]
        return out

    def predict(self, texts, images=None, pixel_values=None) -> list[dict]:
        """One {probability_sarcastic, predicted_label, label} per (text, image) pair."""
        prob = self.forward(texts, images, pixel_values)["probability_1"]
        pred = (prob >= self.threshold).long()
        return [{"probability_sarcastic": float(p), "predicted_label": int(y),
                 "label": LABEL_NAMES[int(y)]} for p, y in zip(prob.tolist(), pred.tolist())]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--serving_config", default=DEFAULT_SERVING_CONFIG)
    ap.add_argument("--checkpoint", default=None, help="default: the serving config's checkpoint")
    ap.add_argument("--train_config", default=None,
                    help="use this training config with an UNTRAINED head (shape check only)")
    ap.add_argument("--text_model_dir", default=None, help="local copy of the pinned DistilBERT")
    ap.add_argument("--clip_model_dir", default=None, help="local copy of the pinned CLIP")
    ap.add_argument("--device", default=None)
    ap.add_argument("--smoke", action="store_true", help="run on the first samples of --split")
    ap.add_argument("--split", default="validation", choices=("train", "validation", "test"))
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--data_dir", default=None)
    ap.add_argument("--text", action="append", default=[], help="caption (repeat with --image)")
    ap.add_argument("--image", action="append", default=[], help="image path (repeat with --text)")
    args = ap.parse_args(argv)

    serving = load_config(args.serving_config) if Path(args.serving_config).is_file() else {}
    model_dirs = {"text_model_dir": args.text_model_dir or serving.get("text_model_dir"),
                  "clip_model_dir": args.clip_model_dir or serving.get("clip_model_dir"),
                  "device": args.device or serving.get("device")}
    if args.train_config:
        config = load_config(args.train_config)
        set_seed(config["seed"])  # the untrained head's initialization
        predictor = MultimodalPredictor(config, None, **model_dirs)
        print("!! untrained fusion head: probabilities are meaningless (shape check only)")
    else:
        predictor = MultimodalPredictor.from_checkpoint(args.checkpoint or serving["checkpoint"],
                                                        **model_dirs)

    if args.smoke:
        frame = load_split(args.split, args.data_dir).head(args.batch_size)
        texts, images, ids = frame["text_input"].tolist(), frame["image_abspath"].tolist(), \
            frame["sample_id"].tolist()
    else:
        if not args.text or len(args.text) != len(args.image):
            ap.error("give --smoke, or matching --text/--image pairs")
        texts, images, ids = args.text, args.image, list(range(len(args.text)))

    out = predictor.forward(texts, images)
    n = len(texts)
    expected = {"text_features": (n, 768), "image_features": (n, 768),
                "fused_features": (n, 1536), "logits": (n, 2)}
    for name, shape in expected.items():
        actual = tuple(out[name].shape)
        assert actual == shape, f"{name}: expected {shape}, got {actual}"
        assert torch.isfinite(out[name]).all(), f"{name} contains NaN/inf"

    model = predictor.model
    print(json.dumps(predictor.info, indent=2))
    print(f"trainable_parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"frozen_parameters: {sum(p.numel() for p in model.parameters() if not p.requires_grad):,}")
    for name in expected:
        print(f"{name}: {tuple(out[name].shape)}")
    prob = out["probability_1"]
    for sid, p in zip(ids, prob.tolist()):
        y = int(p >= predictor.threshold)
        print(f"{sid}: probability_sarcastic={p:.6f}  predicted={y} ({LABEL_NAMES[y]})")
    print("PREDICTOR SMOKE TEST OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
