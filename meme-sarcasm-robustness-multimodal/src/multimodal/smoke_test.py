"""Run a real multimodal forward pass on a few dataset samples.

This is intentionally not a training script. It verifies that the fixed model
revisions, shared data loader, preprocessing, GPU path, and fusion dimensions
work together before the full training pipeline is built.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoImageProcessor, AutoTokenizer, CLIPProcessor

from src.common.data import load_image, load_split
from src.common.reproducibility import get_device, set_seed
from src.multimodal.model import build_model
from src.text.encoder import text_source


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/multimodal_baseline.json")
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--split", default="validation", choices=("train", "validation", "test"))
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--text_model_dir", default=None, help="local copy of the pinned DistilBERT")
    parser.add_argument("--clip_model_dir", default=None, help="local copy of the pinned CLIP")
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    set_seed(config["seed"])
    device = get_device()

    frame = load_split(args.split, args.data_dir).head(args.batch_size)
    texts = frame["text_input"].tolist()
    images = [load_image(path) for path in frame["image_abspath"]]

    source, kwargs = text_source(config, args.text_model_dir)
    tokenizer = AutoTokenizer.from_pretrained(source, **kwargs)
    clip_processor = (
        AutoImageProcessor.from_pretrained(args.clip_model_dir) if args.clip_model_dir else
        CLIPProcessor.from_pretrained(config["clip_model_id"], revision=config["clip_revision"])
    )
    text_inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=config["text_max_length"],
        return_tensors="pt",
    )
    image_inputs = clip_processor(images=images, return_tensors="pt")

    model = build_model(config, args.text_model_dir, args.clip_model_dir).to(device).eval()
    with torch.no_grad():
        output = model(
            input_ids=text_inputs["input_ids"].to(device),
            attention_mask=text_inputs["attention_mask"].to(device),
            pixel_values=image_inputs["pixel_values"].to(device),
            return_features=True,
        )
        probabilities = output["logits"].softmax(dim=-1)[:, 1]
        predictions = (probabilities >= config["threshold"]).long()

    expected_shapes = {
        "text_features": (args.batch_size, 768),
        "image_features": (args.batch_size, 768),
        "fused_features": (args.batch_size, 1536),
        "logits": (args.batch_size, 2),
    }
    for name, expected in expected_shapes.items():
        actual = tuple(output[name].shape)
        assert actual == expected, f"{name}: expected {expected}, got {actual}"
        assert torch.isfinite(output[name]).all(), f"{name} contains NaN/inf"

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"device: {device}")
    print(f"sample_ids: {frame['sample_id'].tolist()}")
    for name in expected_shapes:
        print(f"{name}: {tuple(output[name].shape)}")
    print(f"trainable_parameters: {trainable:,}")
    print(f"frozen_parameters: {frozen:,}")
    print(f"probability_1: {[round(x, 6) for x in probabilities.cpu().tolist()]}")
    print(f"predicted_label: {predictions.cpu().tolist()}")
    print("MULTIMODAL SMOKE TEST OK")


if __name__ == "__main__":
    main()
