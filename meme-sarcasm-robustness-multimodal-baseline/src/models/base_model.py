"""Building blocks shared by every model in this project.

All trainable models are small heads on top of *frozen* encoder features (README 4.3 / 7):
they take cached feature tensors by modality name ("text", "image") as keyword arguments and
return 2-class logits [not sarcastic, sarcastic]. Softmax[:, 1] is P(sarcastic); the decision
threshold is 0.5 (README 5.1).
"""
from __future__ import annotations

from torch import nn

FEATURE_DIMS = {"text": 768, "image": 768}
MODALITY_ORDER = ("text", "image")  # fusion concatenation order: [text | image]


def build_mlp(input_dim: int, hidden_dim: int = 512, dropout: float = 0.2,
              num_classes: int = 2) -> nn.Sequential:
    """README 4.3 small MLP: Linear -> ReLU -> Dropout -> Linear (one hidden layer).

    hidden_dim 0 gives a linear classifier (Dropout -> Linear), the capacity-ablation variant.
    """
    if not hidden_dim:
        return nn.Sequential(nn.Dropout(dropout), nn.Linear(input_dim, num_classes))
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, num_classes),
    )


class FeatureHead(nn.Module):
    """Base class: a head over cached features. Subclasses set `modalities` and `forward(**inputs)`.

    The base class owns no parameters, so subclasses keep their original state_dict keys
    (shipped checkpoints load unchanged).
    """

    modalities: tuple[str, ...] = ()

    def inputs(self, features: dict) -> dict:
        """Select this head's modalities from a {modality: tensor} dict."""
        missing = [m for m in self.modalities if m not in features]
        if missing:
            raise ValueError(f"missing {', '.join(missing)} features")
        return {m: features[m] for m in self.modalities}
