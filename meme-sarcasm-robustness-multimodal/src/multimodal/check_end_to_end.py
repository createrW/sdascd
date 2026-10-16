"""Check that the cached-feature pipeline equals the end-to-end model of README 4.3.

Loads a trained fusion head into MultimodalClassifier (frozen DistilBERT + frozen CLIP + the same
MLP), runs it on raw captions and raw images for some validation samples, and compares with
  * the cached text/image feature rows of the same sample_id (alignment + preprocessing check)
  * probability_1 in the run's predictions_validation.csv (training path == inference path).

    python -m src.multimodal.check_end_to_end --run_id 20261016_distilbert_clip_seed42
    python -m src.multimodal.check_end_to_end --run_id ... --text_model_dir hf_local/distilbert-base-uncased \
        --clip_model_dir hf_local/clip-vit-base-patch32          # offline copies of the pinned weights

Passes when every predicted label agrees and max |probability difference| <= --tolerance.
The image group's cache was extracted on CUDA (TF32 convolution), so image features differ from
CPU by ~1e-3; this is reported, not hidden.
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoTokenizer, CLIPProcessor

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.common.data import load_image, load_split  # noqa: E402
from src.common.metrics import save_json  # noqa: E402
from src.common.reproducibility import get_device, set_seed  # noqa: E402
from src.image.safety import require_new_paths, validate_run_id  # noqa: E402
from src.multimodal.features import load_split_features  # noqa: E402
from src.multimodal.model import build_model, load_fusion_head  # noqa: E402
from src.text.encoder import text_source, tokenize  # noqa: E402


def pick_rows(frame, n):
    """n rows spread evenly over the split (deterministic)."""
    step = max(1, len(frame) // n)
    return frame.iloc[::step].head(n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_id", required=True)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--text_model_dir", default=None)
    ap.add_argument("--clip_model_dir", default=None)
    ap.add_argument("--data_dir", default=None)
    ap.add_argument("--tolerance", type=float, default=0.01)
    args = ap.parse_args()
    validate_run_id(args.run_id)

    run_dir = Path("outputs/multimodal") / args.run_id
    report_path = run_dir / "checks" / "end_to_end_validation.json"
    require_new_paths(report_path)
    ckpt = torch.load(Path("checkpoints/multimodal") / args.run_id / "best.pt",
                      map_location="cpu", weights_only=True)
    cfg = ckpt["config"]
    if tuple(ckpt["modalities"]) != ("text", "image"):
        raise ValueError("end-to-end check needs a text+image fusion run")
    set_seed(cfg["seed"])
    device = get_device()

    frame = pick_rows(load_split("validation", args.data_dir), args.n)
    ids = frame["sample_id"].tolist()
    reported = pd.read_csv(run_dir / "predictions_validation.csv",
                           float_precision="round_trip").set_index("sample_id").loc[ids]

    model = build_model(cfg, args.text_model_dir, args.clip_model_dir).to(device).eval()
    load_fusion_head(model, ckpt["state_dict"])
    source, kwargs = text_source(cfg, args.text_model_dir)
    tokenizer = AutoTokenizer.from_pretrained(source, **kwargs)
    processor = (AutoImageProcessor.from_pretrained(args.clip_model_dir) if args.clip_model_dir else
                 CLIPProcessor.from_pretrained(cfg["clip_model_id"], revision=cfg["clip_revision"]))

    outs = {"text_features": [], "image_features": [], "prob": []}
    with torch.no_grad():
        for start in range(0, len(frame), args.batch_size):
            part = frame.iloc[start:start + args.batch_size]
            enc = tokenize(tokenizer, part["text_input"].tolist(), cfg["text_max_length"])
            pixels = processor(images=[load_image(p) for p in part["image_abspath"]],
                               return_tensors="pt")["pixel_values"]
            out = model(input_ids=enc["input_ids"].to(device),
                        attention_mask=enc["attention_mask"].to(device),
                        pixel_values=pixels.to(device), return_features=True)
            outs["text_features"].append(out["text_features"].cpu())
            outs["image_features"].append(out["image_features"].cpu())
            outs["prob"].append(out["logits"].softmax(dim=-1)[:, 1].cpu())
    outs = {k: torch.cat(v) for k, v in outs.items()}

    _, cached, _ = load_split_features(cfg, "validation", ckpt["text_features_dir"],
                                       ckpt["image_features_dir"], args.data_dir)
    row = {sid: i for i, sid in enumerate(load_split("validation", args.data_dir)["sample_id"])}
    index = torch.tensor([row[s] for s in ids])
    report = {"run_id": args.run_id, "n": len(ids), "device": str(device),
              "text_features_dir": ckpt["text_features_dir"],
              "image_features_dir": ckpt["image_features_dir"],
              "text_model": args.text_model_dir or f"{cfg['text_model_id']}@{cfg['text_revision']}",
              "clip_model": args.clip_model_dir or f"{cfg['clip_model_id']}@{cfg['clip_revision']}"}
    for name, key in (("text", "text_features"), ("image", "image_features")):
        ref = F.normalize(cached[name][index], dim=-1)
        diff = (outs[key] - ref).abs()
        report[f"{name}_max_abs_diff"] = diff.max().item()
        report[f"{name}_min_cosine"] = F.cosine_similarity(outs[key], ref, dim=1).min().item()
    prob_ref = torch.tensor(reported["probability_1"].to_numpy(), dtype=torch.float32)
    pred_now = (outs["prob"] >= cfg["threshold"]).long()
    pred_ref = torch.tensor(reported["predicted_label"].to_numpy())
    report["probability_max_abs_diff"] = (outs["prob"] - prob_ref).abs().max().item()
    report["label_agreement"] = (pred_now == pred_ref).float().mean().item()
    report["passed"] = bool(report["label_agreement"] == 1.0
                            and report["probability_max_abs_diff"] <= args.tolerance)

    report_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(report, report_path)
    print(json.dumps(report, indent=2))
    print("END-TO-END CHECK OK" if report["passed"] else "END-TO-END CHECK FAILED")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
