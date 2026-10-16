"""Frozen DistilBERT encoder and the sentence vectors we take from it.

Which 768-d vector represents one meme caption?

  first_token   last_hidden_state[:, 0], the final-layer state of the [CLS] token.
                DistilBERT (Sanh et al., 2019) removed BERT's pooler and token-type embeddings,
                so there is no `pooler_output`; Hugging Face's DistilBertForSequenceClassification
                classifies from exactly this vector (then pre_classifier -> ReLU -> classifier).
                This is the team baseline feature (docs/decisions.md D13).
  mean_pooling  mean of last_hidden_state over real tokens ([CLS] and [SEP] included,
                padding excluded via attention_mask). The usual sentence vector for a *frozen*
                encoder (Sentence-BERT, Reimers & Gurevych, 2019); kept as an ablation.
"""
from __future__ import annotations

import torch
from transformers import AutoModel, AutoTokenizer

TEXT_FEATURES = ("first_token", "mean_pooling")


def text_source(cfg: dict, model_dir: str | None = None) -> tuple[str, dict]:
    """Hub id + pinned revision by default; a local copy of the same weights when offline."""
    if model_dir:
        return str(model_dir), {}
    return cfg["text_model_id"], {"revision": cfg["text_revision"]}


def load_text_encoder(cfg: dict, device: torch.device, model_dir: str | None = None):
    """Tokenizer + frozen DistilBertModel in eval mode (MLM-head weights are not loaded)."""
    source, kwargs = text_source(cfg, model_dir)
    tokenizer = AutoTokenizer.from_pretrained(source, **kwargs)
    model = AutoModel.from_pretrained(source, **kwargs)
    model.eval().to(device)
    model.requires_grad_(False)
    return tokenizer, model


def tokenize(tokenizer, texts: list[str], max_length: int) -> dict:
    """[CLS] text [SEP], WordPiece (uncased), truncated to max_length, padded to the batch maximum."""
    return tokenizer(texts, padding=True, truncation=True, max_length=max_length,
                     return_tensors="pt")


def pool_text_features(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor,
                       how: str) -> torch.Tensor:
    if how == "first_token":
        return last_hidden_state[:, 0, :]
    if how == "mean_pooling":
        mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
        return (last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
    raise ValueError(f"unknown text feature {how!r}; expected one of {TEXT_FEATURES}")
