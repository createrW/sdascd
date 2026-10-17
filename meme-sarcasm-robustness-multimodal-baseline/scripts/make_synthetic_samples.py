"""Write the committable synthetic sample set in data/samples/ (no dataset content).

The real Memotion images and captions are never committed (team README). These rows only mimic
the dataset's folder layout and metadata.csv columns, so the loader, the predictor and the API can
be smoke-tested on a machine without the dataset:

    python -m scripts.make_synthetic_samples              # rewrites data/samples/
    python -m scripts.make_synthetic_samples --out_dir /tmp/samples
    python -m src.serving.predictor --smoke --data_dir data/samples/synthetic_memotion --train_config configs/train/multimodal_baseline.yaml

Images are 64x64 PNG drawings (shapes and colours), captions are invented sentences.
"""
import argparse
import base64
import json
import sys
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.dataset import SARCASM_TO_BINARY, SPLITS  # noqa: E402

ROWS = [  # (sarcasm, caption)  -- invented, not from the dataset
    ("not_sarcastic", "Sunday plans: tea, a book and an early night."),
    ("general", "Oh great, another Monday. Exactly what I needed."),
    ("twisted_meaning", "I love waiting in line for three hours, best day ever."),
    ("not_sarcastic", ""),   # blank caption -> [NO_TEXT]
]
COLOURS = [(230, 90, 60), (60, 120, 220), (40, 170, 120), (235, 180, 30)]


def draw(i: int, split_index: int) -> Image.Image:
    im = Image.new("RGB", (64, 64), COLOURS[(i + split_index) % len(COLOURS)])
    d = ImageDraw.Draw(im)
    d.ellipse((8 + 4 * i, 8, 40 + 4 * i, 40), fill=(255, 255, 255))
    d.rectangle((4, 48, 60, 58), fill=(20, 20, 20))
    return im


def main(argv=None):
    ap = argparse.ArgumentParser(description="Write the synthetic sample set (no dataset content).")
    ap.add_argument("--out_dir", default="data/samples")
    out = Path(ap.parse_args(argv).out_dir)
    root = out / "synthetic_memotion"
    for s, split in enumerate(SPLITS):
        (root / split / "images").mkdir(parents=True, exist_ok=True)
        rows = []
        for i, (sarcasm, caption) in enumerate(ROWS):
            sid = f"{split}_{i:06d}"
            rel = f"images/{sid}.png"
            draw(i, s).save(root / split / rel, optimize=True)
            rows.append({"sample_id": sid, "image_path": rel, "text_ocr": caption,
                         "text_corrected": caption, "text_input": caption, "sarcasm": sarcasm,
                         "sarcasm_binary": SARCASM_TO_BINARY[sarcasm], "humor": "funny",
                         "offensive": "not_offensive", "motivational": "not_motivational",
                         "sentiment": "neutral", "split": split})
        pd.DataFrame(rows).to_csv(root / split / "metadata.csv", index=False, encoding="utf-8-sig")

    image_b64 = base64.b64encode((root / "validation/images/validation_000001.png").read_bytes()).decode()
    (out / "predict_request.json").write_text(json.dumps(
        {"text": ROWS[1][1], "image_base64": image_b64}, indent=2) + "\n", encoding="utf-8")
    items = [{"text": caption or None, "image_base64": base64.b64encode(
        (root / f"validation/images/validation_{i:06d}.png").read_bytes()).decode()}
        for i, (_, caption) in enumerate(ROWS)]
    (out / "predict_batch_request.json").write_text(json.dumps({"items": items}, indent=2) + "\n",
                                                    encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()
