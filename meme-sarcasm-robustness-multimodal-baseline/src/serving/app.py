"""FastAPI service for the multimodal sarcasm classifier (contract: docs/api_spec.md).

    bash scripts/serve.sh                                          # reads configs/deploy/serving.yaml
    python -m src.serving.app --serving_config configs/deploy/serving.yaml \
        --checkpoint checkpoints/multimodal/<run_id>/best.pt       # serve another trained head
    uvicorn src.serving.app:create_app --factory --host 0.0.0.0 --port 8000

Endpoints
    GET  /health          liveness + whether the model is loaded
    GET  /model           model card: encoders, revisions, threshold, checkpoint
    POST /predict         {"text": "...", "image_base64": "..."} -> probability + label
    POST /predict/batch   {"items": [{...}, ...]} -> one result per item, same order

The threshold is the training one (0.5, README 3.2). Uploads are checked cheaply first (body and
base64 size, image format, pixel count, aspect ratio), then decoded to RGB one at a time and passed
through the same CLIP processor and DistilBERT tokenizer as training (src/serving/predictor.py).
Decoding and inference run one request at a time behind a lock, and each image's decoded raster is
released before the next one is decoded, so memory holds one full-size image (plus its RGB copy) at
a time; the compressed uploads of a batch stay in memory until the response.
"""
from __future__ import annotations

import argparse
import base64
import binascii
import io
import math
import os
import sys
import threading
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel, Field, field_validator

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.preprocess import to_rgb  # noqa: E402
from src.serving.predictor import DEFAULT_SERVING_CONFIG  # noqa: E402
from src.utils.config import load_config  # noqa: E402

DEFAULTS = {"max_batch_size": 32, "max_image_bytes": 10 * 1024 * 1024, "max_text_chars": 2000,
            "max_image_pixels": 40_000_000, "max_aspect_ratio": 20, "host": "0.0.0.0", "port": 8000}
# Formats the API decodes. The dataset has JPEG, PNG and GIF; formats that Pillow renders through
# external programs (EPS via Ghostscript) or rarely see in memes are refused.
IMAGE_FORMATS = ("JPEG", "PNG", "GIF", "WEBP", "BMP")  # JPEG also opens multi-picture (MPO) files
Image.init()  # register every plugin, so Image.open(formats=...) can name WEBP
MODEL_KEYS = ("checkpoint", "text_model_dir", "clip_model_dir", "device")


class PredictRequest(BaseModel):
    text: str | None = Field(None, description="Meme caption; null or blank -> [NO_TEXT].")
    image_base64: str = Field(..., description="Base64-encoded image file (JPEG, PNG, GIF, WebP, "
                                               "BMP). A data URL prefix, line breaks, missing "
                                               "padding and the URL-safe alphabet are accepted.")

    @field_validator("text")
    @classmethod
    def text_must_be_valid_unicode(cls, value):
        if value is not None:
            try:
                value.encode("utf-8")
            except UnicodeEncodeError:  # e.g. a lone UTF-16 surrogate from a cut emoji
                raise ValueError("text is not valid Unicode (unpaired surrogate)")
        return value


class BatchRequest(BaseModel):
    items: list[PredictRequest] = Field(..., min_length=1)


class Prediction(BaseModel):
    probability_sarcastic: float = Field(..., description="Softmax P(class 1 = sarcastic).")
    predicted_label: int = Field(..., description="1 if probability_sarcastic >= threshold, else 0.")
    label: str = Field(..., description="not_sarcastic | sarcastic")
    threshold: float


class BatchResponse(BaseModel):
    predictions: list[Prediction]


def decode_base64(data: str, max_bytes: int) -> bytes:
    """Base64 text (data URL, wrapped lines, URL-safe or unpadded forms allowed) -> file bytes."""
    if data.startswith("data:"):
        data = data.split(",", 1)[-1]
    data = "".join(data.split())                       # line breaks / spaces from `base64` tools
    if len(data) > 4 * math.ceil(max_bytes / 3) + 4:   # reject before decoding anything
        raise HTTPException(status_code=413, detail=f"image larger than {max_bytes} bytes")
    if "-" in data or "_" in data:
        data = data.replace("-", "+").replace("_", "/")
    data += "=" * (-len(data) % 4)
    try:
        raw = base64.b64decode(data, validate=True)
    except (binascii.Error, ValueError):
        raise HTTPException(status_code=400, detail="image_base64 is not valid base64")
    if not raw:
        raise HTTPException(status_code=400, detail="image_base64 is empty")
    if len(raw) > max_bytes:
        raise HTTPException(status_code=413, detail=f"image larger than {max_bytes} bytes")
    return raw


def open_image(raw: bytes, settings: dict) -> Image.Image:
    """Open (header only, no pixel decoding) and check format, pixel count and aspect ratio."""
    try:
        image = Image.open(io.BytesIO(raw), formats=IMAGE_FORMATS)
    except Image.DecompressionBombError:  # Pillow's own cap (~179 MP) is hit before ours
        raise HTTPException(status_code=413,
                            detail=f"image has more than {settings['max_image_pixels']} pixels")
    except Exception:  # UnidentifiedImageError, OSError, ...
        raise HTTPException(status_code=400, detail="image_base64 is not a readable "
                                                    f"{'/'.join(IMAGE_FORMATS)} image")
    width, height = image.size
    if width < 1 or height < 1:
        raise HTTPException(status_code=400, detail="image has no pixels")
    if width * height > settings["max_image_pixels"]:
        raise HTTPException(status_code=413,
                            detail=f"image has more than {settings['max_image_pixels']} pixels")
    if max(width, height) / min(width, height) > settings["max_aspect_ratio"]:
        raise HTTPException(status_code=413,
                            detail=f"image aspect ratio above {settings['max_aspect_ratio']}")
    return image


def max_request_bytes(settings: dict) -> int:
    """Body-size limit implied by the per-item limits (a gateway should also enforce one).

    Per item: base64 of max_image_bytes, plus 5 bytes per 64 characters for the accepted line breaks
    (CRLF every 64 characters, sent as the 4-byte JSON escape) and escaped slashes, plus the caption
    as JSON (\\uXXXX escapes) and 1 KiB for keys and a data-URL prefix.
    """
    b64 = 4 * math.ceil(settings["max_image_bytes"] / 3)
    per_item = b64 + 5 * math.ceil(b64 / 64) + 12 * settings["max_text_chars"] + 1024
    return settings["max_batch_size"] * per_item + 64 * 1024


def create_app(predictor=None, serving_config: str | None = None, overrides: dict | None = None) -> FastAPI:
    """Build the service. `predictor` defaults to MultimodalPredictor.from_serving_config(...).

    serving_config defaults to $SERVING_CONFIG, then configs/deploy/serving.yaml. `overrides`
    (checkpoint, text_model_dir, clip_model_dir, device, limits) take precedence over the file.
    """
    config_path = serving_config or os.environ.get("SERVING_CONFIG", DEFAULT_SERVING_CONFIG)
    overrides = {k: v for k, v in (overrides or {}).items() if v is not None}
    settings = {**DEFAULTS, **(load_config(config_path) if Path(config_path).is_file() else {}),
                **overrides}
    lock = threading.Lock()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if app.state.predictor is None:
            from src.serving.predictor import MultimodalPredictor
            app.state.predictor = MultimodalPredictor.from_serving_config(
                config_path, **{k: overrides.get(k) for k in MODEL_KEYS})
        yield

    app = FastAPI(title="Meme sarcasm classifier", version="1.0.0", lifespan=lifespan,
                  description="Frozen DistilBERT + frozen CLIP ViT-B/32 with a trained fusion MLP.")
    app.state.predictor = predictor
    app.state.settings = settings
    body_limit = max_request_bytes(settings)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError):
        # type/loc/msg only: the default handler echoes the input back, which repeats whole
        # base64 images and fails on text that is not valid UTF-8 (lone surrogates)
        detail = [{"type": e.get("type"), "loc": list(e.get("loc", ())), "msg": e.get("msg")}
                  for e in exc.errors()]
        return JSONResponse(status_code=422, content={"detail": detail})

    @app.middleware("http")
    async def limit_body_size(request: Request, call_next):
        length = request.headers.get("content-length")
        if length is not None and length.isdigit() and int(length) > body_limit:
            return JSONResponse(status_code=413, content={"detail": f"request body larger than {body_limit} bytes"})
        return await call_next(request)

    def loaded_model():
        if app.state.predictor is None:
            raise HTTPException(status_code=503, detail="model not loaded")
        return app.state.predictor

    def run(items: list[PredictRequest]) -> list[Prediction]:
        model = loaded_model()
        for item in items:
            if item.text is not None and len(item.text) > settings["max_text_chars"]:
                raise HTTPException(status_code=422,
                                    detail=f"text longer than {settings['max_text_chars']} characters")
        # cheap checks for every item first, so a bad item fails the request before any decoding
        opened = [open_image(decode_base64(item.image_base64, settings["max_image_bytes"]), settings)
                  for item in items]
        texts = [item.text for item in items]
        with lock:
            pixels = []
            for i in range(len(opened)):
                image, opened[i] = opened[i], None   # drop the list's reference ...
                try:
                    rgb = to_rgb(image)
                except Exception:  # truncated or corrupt pixel data
                    raise HTTPException(status_code=400, detail="image_base64 could not be decoded")
                finally:
                    image.close()                    # ... and the decoded raster, before the next image
                    del image
                pixels.append(model.pixel_values([rgb]))
                del rgb
            results = model.predict(texts, pixel_values=torch.cat(pixels))
        return [Prediction(**r, threshold=model.threshold) for r in results]

    @app.get("/health")
    async def health():  # async: answered by the event loop even while inference threads queue
        return {"status": "ok", "model_loaded": app.state.predictor is not None}

    @app.get("/model")
    async def model_info():
        return loaded_model().info

    @app.post("/predict", response_model=Prediction)
    def predict(request: PredictRequest):
        return run([request])[0]

    @app.post("/predict/batch", response_model=BatchResponse)
    def predict_batch(request: BatchRequest):
        if len(request.items) > settings["max_batch_size"]:
            raise HTTPException(status_code=413,
                                detail=f"at most {settings['max_batch_size']} items per batch")
        return {"predictions": run(request.items)}

    return app


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--serving_config", default=os.environ.get("SERVING_CONFIG", DEFAULT_SERVING_CONFIG))
    ap.add_argument("--checkpoint", default=None, help="override the serving config's checkpoint")
    ap.add_argument("--text_model_dir", default=None, help="local copy of the pinned DistilBERT")
    ap.add_argument("--clip_model_dir", default=None, help="local copy of the pinned CLIP")
    ap.add_argument("--device", default=None)
    ap.add_argument("--host", default=None)
    ap.add_argument("--port", type=int, default=None)
    args = ap.parse_args(argv)
    import uvicorn

    app = create_app(serving_config=args.serving_config,
                     overrides={k: getattr(args, k) for k in MODEL_KEYS})
    settings = app.state.settings
    uvicorn.run(app, host=args.host or settings["host"], port=args.port or settings["port"])


if __name__ == "__main__":
    main()
