"""Frozen DistilBERT + frozen CLIP vision encoder with a small fusion MLP.

Two ways to run the same classifier:

* MultimodalClassifier: raw text + raw image -> frozen encoders -> fusion MLP (end to end).
* FusionHead: cached encoder features -> the identical normalization/concatenation/MLP.
  Training uses this (src/multimodal/train.py) because the encoders never change; its
  `classifier.*` weights load directly into MultimodalClassifier (see load_fusion_head).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModel, CLIPModel

from src.text.encoder import pool_text_features

FEATURE_DIMS = {"text": 768, "image": 768}
MODALITY_ORDER = ("text", "image")  # concatenation order: [text | image]


def build_fusion_mlp(input_dim: int, hidden_dim: int = 512, dropout: float = 0.2,
                     num_classes: int = 2) -> nn.Sequential:
    """README 4.3 small MLP: Linear -> ReLU -> Dropout -> Linear (one hidden layer)."""
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, num_classes),
    )


class FusionHead(nn.Module):
    """Trainable part of the multimodal model, applied to cached 768-d features.

    modalities=("text", "image") is the baseline (1536-d input). A single modality gives the
    matching unimodal ablation with the same head design.
    """

    def __init__(self, modalities=MODALITY_ORDER, hidden_dim: int = 512, dropout: float = 0.2,
                 num_classes: int = 2, normalize_embeddings: bool = True) -> None:
        super().__init__()
        modalities = tuple(m for m in MODALITY_ORDER if m in modalities)
        if not modalities:
            raise ValueError("at least one modality is required")
        self.modalities = modalities
        self.normalize_embeddings = normalize_embeddings
        self.input_dim = sum(FEATURE_DIMS[m] for m in modalities)
        self.classifier = build_fusion_mlp(self.input_dim, hidden_dim, dropout, num_classes)

    def fuse(self, text: torch.Tensor | None = None,
             image: torch.Tensor | None = None) -> torch.Tensor:
        given = {"text": text, "image": image}
        parts = []
        for m in self.modalities:
            x = given[m]
            if x is None:
                raise ValueError(f"missing {m} features")
            parts.append(F.normalize(x, dim=-1) if self.normalize_embeddings else x)
        return torch.cat(parts, dim=-1)

    def forward(self, text: torch.Tensor | None = None,
                image: torch.Tensor | None = None) -> torch.Tensor:
        return self.classifier(self.fuse(text, image))


def build_fusion_head(config: dict) -> FusionHead:
    fusion = config["fusion"]
    head = FusionHead(
        modalities=config.get("modalities", list(MODALITY_ORDER)),
        hidden_dim=fusion["hidden_dim"],
        dropout=fusion["dropout"],
        num_classes=fusion["num_classes"],
        normalize_embeddings=config["normalize_embeddings"],
    )
    if fusion.get("input_dim", head.input_dim) != head.input_dim:
        raise ValueError(f"config fusion.input_dim={fusion['input_dim']} but modalities "
                         f"{head.modalities} give {head.input_dim}")
    return head


class MultimodalClassifier(nn.Module):
    """Fuse 768-d text and 768-d image features into binary logits.

    The pretrained encoders are frozen and kept in evaluation mode. Only the
    fusion/classification MLP is trainable.
    """

    def __init__(
        self,
        text_model_id: str,
        text_revision: str,
        clip_model_id: str,
        clip_revision: str,
        hidden_dim: int = 512,
        dropout: float = 0.2,
        num_classes: int = 2,
        normalize_embeddings: bool = True,
        text_feature: str = "first_token",
        text_model_dir: str | None = None,
        clip_model_dir: str | None = None,
    ) -> None:
        super().__init__()
        # *_model_dir: local copies of the same pinned weights for machines without Hub access.
        self.text_encoder = (AutoModel.from_pretrained(text_model_dir) if text_model_dir else
                             AutoModel.from_pretrained(text_model_id, revision=text_revision))
        self.clip = (CLIPModel.from_pretrained(clip_model_dir) if clip_model_dir else
                     CLIPModel.from_pretrained(clip_model_id, revision=clip_revision))
        self.normalize_embeddings = normalize_embeddings
        self.text_feature = text_feature

        for encoder in (self.text_encoder, self.clip):
            encoder.requires_grad_(False)
            encoder.eval()

        text_dim = int(self.text_encoder.config.dim)
        image_dim = int(self.clip.config.vision_config.hidden_size)
        fusion_dim = text_dim + image_dim
        if text_dim != 768 or image_dim != 768 or fusion_dim != 1536:
            raise ValueError(
                f"Unexpected feature dimensions: text={text_dim}, "
                f"image={image_dim}, fusion={fusion_dim}"
            )

        self.text_dim = text_dim
        self.image_dim = image_dim
        self.fusion_dim = fusion_dim
        self.classifier = build_fusion_mlp(fusion_dim, hidden_dim, dropout, num_classes)

    def train(self, mode: bool = True) -> "MultimodalClassifier":
        """Train the fusion head while frozen encoders stay deterministic."""
        super().train(mode)
        self.text_encoder.eval()
        self.clip.eval()
        return self

    def encode_text(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        with torch.no_grad():
            output = self.text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            features = pool_text_features(output.last_hidden_state, attention_mask,
                                          self.text_feature)
        if self.normalize_embeddings:
            features = F.normalize(features, dim=-1)
        return features

    def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            output = self.clip.vision_model(pixel_values=pixel_values)
            features = output.pooler_output
        if self.normalize_embeddings:
            features = F.normalize(features, dim=-1)
        return features

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor,
        return_features: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        text_features = self.encode_text(input_ids, attention_mask)
        image_features = self.encode_image(pixel_values)
        fused_features = torch.cat((text_features, image_features), dim=-1)
        logits = self.classifier(fused_features)

        if return_features:
            return {
                "text_features": text_features,
                "image_features": image_features,
                "fused_features": fused_features,
                "logits": logits,
            }
        return logits


def build_model(config: dict, text_model_dir: str | None = None,
                clip_model_dir: str | None = None) -> MultimodalClassifier:
    """Construct the baseline directly from configs/multimodal_baseline.json."""
    fusion = config["fusion"]
    return MultimodalClassifier(
        text_model_id=config["text_model_id"],
        text_revision=config["text_revision"],
        clip_model_id=config["clip_model_id"],
        clip_revision=config["clip_revision"],
        hidden_dim=fusion["hidden_dim"],
        dropout=fusion["dropout"],
        num_classes=fusion["num_classes"],
        normalize_embeddings=config["normalize_embeddings"],
        text_feature=config.get("text_feature", "first_token"),
        text_model_dir=text_model_dir,
        clip_model_dir=clip_model_dir,
    )


def load_fusion_head(model: MultimodalClassifier, head_state_dict: dict) -> None:
    """Put a FusionHead checkpoint (trained on cached features) into the end-to-end model."""
    prefix = "classifier."
    weights = {k[len(prefix):]: v for k, v in head_state_dict.items() if k.startswith(prefix)}
    model.classifier.load_state_dict(weights, strict=True)
