"""Image-only model: a small head on top of frozen CLIP ViT-B/32 features (image group baseline)."""
import torch
from torch import nn

from src.models.base_model import FeatureHead


class ImageHead(FeatureHead):
    """768-d CLIP vector -> 2 scores (logits): [not sarcastic, sarcastic].

    hidden_dim == 0  -> a single linear layer: logits = W @ x + b, W is 2 x 768.
                        This is the linear classifier from Block 1 p229 / the CLIP
                        paper's "linear probe" (Block 4 p265).
    hidden_dim  > 0  -> a 1-hidden-layer MLP: Linear -> ReLU -> Dropout -> Linear.
    """

    modalities = ("image",)

    def __init__(self, in_dim: int = 768, hidden_dim: int = 0, dropout: float = 0.0,
                 num_classes: int = 2):
        super().__init__()
        if hidden_dim and hidden_dim > 0:
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes),
            )
        else:
            self.net = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_dim, num_classes))

    def forward(self, x: torch.Tensor | None = None, image: torch.Tensor | None = None) -> torch.Tensor:
        x = image if x is None else x
        return self.net(x)  # raw scores (logits); softmax turns them into probabilities


def build_image_head(config: dict, in_dim: int = 768) -> ImageHead:
    """From configs/train/image_baseline.yaml (`head: {hidden_dim, dropout}`)."""
    return ImageHead(in_dim=in_dim, hidden_dim=config["head"]["hidden_dim"],
                     dropout=config["head"]["dropout"])
