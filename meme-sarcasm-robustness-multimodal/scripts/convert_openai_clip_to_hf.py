"""Convert OpenAI's original CLIP ViT-B/32 checkpoint into a local Hugging Face CLIPModel folder.

Only needed on machines that cannot reach huggingface.co (the team default is
`CLIPModel.from_pretrained("openai/clip-vit-base-patch32", revision="3d74acf")`).
`openai/clip-vit-base-patch32` on the Hub is itself a conversion of this OpenAI file, so the
weights are the same; `scripts/compare_feature_caches.py` checks the result against the image
group's CUDA cache.

    python -m scripts.convert_openai_clip_to_hf --openai_pt ViT-B-32.pt --out_dir hf_local/clip-vit-base-patch32

The OpenAI file's SHA-256 must be 40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af
(the hash is part of OpenAI's official download URL).
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch
from transformers import CLIPConfig, CLIPModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.image.safety import require_new_paths  # noqa: E402

OPENAI_SHA256 = "40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af"

# preprocessor_config.json of openai/clip-vit-base-patch32 (official CLIP preprocessing).
PREPROCESSOR_CONFIG = {
    "crop_size": 224,
    "do_center_crop": True,
    "do_normalize": True,
    "do_resize": True,
    "feature_extractor_type": "CLIPFeatureExtractor",
    "image_mean": [0.48145466, 0.4578275, 0.40821073],
    "image_std": [0.26862954, 0.26130258, 0.27577711],
    "resample": 3,
    "size": 224,
}


def clip_config() -> CLIPConfig:
    """Architecture of openai/clip-vit-base-patch32 (QuickGELU, LayerNorm eps 1e-5)."""
    return CLIPConfig(
        text_config=dict(vocab_size=49408, hidden_size=512, intermediate_size=2048,
                         num_hidden_layers=12, num_attention_heads=8, max_position_embeddings=77,
                         hidden_act="quick_gelu", layer_norm_eps=1e-5,
                         bos_token_id=49406, eos_token_id=49407, pad_token_id=1),
        vision_config=dict(hidden_size=768, intermediate_size=3072, num_hidden_layers=12,
                           num_attention_heads=12, image_size=224, patch_size=32,
                           hidden_act="quick_gelu", layer_norm_eps=1e-5),
        projection_dim=512,
        logit_scale_init_value=2.6592,
    )


def convert_block(src: dict, src_prefix: str, dst_prefix: str, out: dict) -> None:
    """One residual attention block: fused in_proj -> q/k/v, c_fc/c_proj -> fc1/fc2."""
    q, k, v = src[f"{src_prefix}.attn.in_proj_weight"].chunk(3, dim=0)
    qb, kb, vb = src[f"{src_prefix}.attn.in_proj_bias"].chunk(3, dim=0)
    for name, w, b in (("q_proj", q, qb), ("k_proj", k, kb), ("v_proj", v, vb)):
        out[f"{dst_prefix}.self_attn.{name}.weight"] = w
        out[f"{dst_prefix}.self_attn.{name}.bias"] = b
    pairs = {
        "attn.out_proj": "self_attn.out_proj",
        "ln_1": "layer_norm1",
        "ln_2": "layer_norm2",
        "mlp.c_fc": "mlp.fc1",
        "mlp.c_proj": "mlp.fc2",
    }
    for s, d in pairs.items():
        out[f"{dst_prefix}.{d}.weight"] = src[f"{src_prefix}.{s}.weight"]
        out[f"{dst_prefix}.{d}.bias"] = src[f"{src_prefix}.{s}.bias"]


def convert_state_dict(src: dict, n_layers: int = 12) -> dict:
    out = {
        # vision tower
        "vision_model.embeddings.class_embedding": src["visual.class_embedding"],
        "vision_model.embeddings.patch_embedding.weight": src["visual.conv1.weight"],
        "vision_model.embeddings.position_embedding.weight": src["visual.positional_embedding"],
        "vision_model.pre_layrnorm.weight": src["visual.ln_pre.weight"],
        "vision_model.pre_layrnorm.bias": src["visual.ln_pre.bias"],
        "vision_model.post_layernorm.weight": src["visual.ln_post.weight"],
        "vision_model.post_layernorm.bias": src["visual.ln_post.bias"],
        "visual_projection.weight": src["visual.proj"].t(),
        # text tower
        "text_model.embeddings.token_embedding.weight": src["token_embedding.weight"],
        "text_model.embeddings.position_embedding.weight": src["positional_embedding"],
        "text_model.final_layer_norm.weight": src["ln_final.weight"],
        "text_model.final_layer_norm.bias": src["ln_final.bias"],
        "text_projection.weight": src["text_projection"].t(),
        "logit_scale": src["logit_scale"],
    }
    for i in range(n_layers):
        convert_block(src, f"visual.transformer.resblocks.{i}", f"vision_model.encoder.layers.{i}", out)
        convert_block(src, f"transformer.resblocks.{i}", f"text_model.encoder.layers.{i}", out)
    return {k: v.detach().clone().float().contiguous() for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--openai_pt", required=True)
    ap.add_argument("--out_dir", required=True, help="must be a new directory")
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    require_new_paths(out_dir)

    with open(args.openai_pt, "rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != OPENAI_SHA256:
        raise ValueError(f"{args.openai_pt}: SHA-256 {digest} is not the official ViT-B/32 file")

    src = torch.jit.load(args.openai_pt, map_location="cpu").state_dict()
    model = CLIPModel(clip_config()).eval()
    converted = convert_state_dict(src)
    missing, unexpected = model.load_state_dict(converted, strict=False)
    # position_ids buffers are created by the model itself; nothing else may be missing.
    missing = [k for k in missing if not k.endswith("position_ids")]
    if missing or unexpected:
        raise RuntimeError(f"conversion mismatch: missing={missing} unexpected={unexpected}")

    out_dir.mkdir(parents=True, exist_ok=False)
    model.save_pretrained(out_dir)
    (out_dir / "preprocessor_config.json").write_text(json.dumps(PREPROCESSOR_CONFIG, indent=2),
                                                     encoding="utf-8")
    (out_dir / "conversion.json").write_text(json.dumps({
        "source_file": Path(args.openai_pt).name, "source_sha256": digest,
        "equivalent_hub_model": "openai/clip-vit-base-patch32",
        "converted_tensors": len(converted),
    }, indent=2), encoding="utf-8")
    print(f"converted {len(converted)} tensors -> {out_dir}")


if __name__ == "__main__":
    main()
