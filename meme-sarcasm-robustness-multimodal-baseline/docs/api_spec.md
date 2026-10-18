# Model API / 模型接口定义

HTTP interface of the multimodal sarcasm classifier (`src/serving/app.py`, FastAPI). The model
behind it is the validation-selected multimodal baseline: frozen DistilBERT `[CLS]` + frozen CLIP
ViT-B/32 `pooler_output`, L2-normalized and concatenated, MLP 1536→512→2
([design.md](design.md) §3, §7). Interactive OpenAPI docs are served at `/docs`, the schema at
`/openapi.json`.

多模态讽刺分类模型的 HTTP 接口（FastAPI）。服务的模型是按 Validation 选出的多模态基线。启动后 `/docs` 提供交互式 OpenAPI 文档。

## Run / 启动

```bash
pip install -r requirements.txt              # includes fastapi and uvicorn
bash scripts/serve.sh                        # reads configs/deploy/serving.yaml, listens on 0.0.0.0:8000
CHECKPOINT=checkpoints/multimodal/<run_id>/best.pt bash scripts/serve.sh      # serve another head
TEXT_MODEL_DIR=hf_local/distilbert-base-uncased CLIP_MODEL_DIR=hf_local/clip-vit-base-patch32 \
    bash scripts/serve.sh                    # offline: local copies of the pinned encoders
SERVING_CONFIG=my_serving.yaml PORT=9000 bash scripts/serve.sh
# or, with everything from the serving config:
uvicorn src.serving.app:create_app --factory --host 0.0.0.0 --port 8000
```

The default checkpoint is not in Git: it comes from `Checkpoints_merged_20261018_v2.zip`
([checkpoints.md](checkpoints.md)), or point `CHECKPOINT` at a head you trained.

`configs/deploy/serving.yaml`:

| Key | Default | Meaning |
|---|---|---|
| `checkpoint` | `checkpoints/multimodal/20261016_distilbert_clip_seed42/best.pt` | trained fusion head; its stored config fixes encoders, revisions, `text_max_length` and the threshold |
| `text_model_dir`, `clip_model_dir` | `null` | local copies of the pinned encoders; `null` downloads `distilbert/distilbert-base-uncased@1c4513b…` and `openai/clip-vit-base-patch32@3d74acf` |
| `device` | `cpu` | `cpu`, `cuda`, `mps`, or `null` for the first available |
| `host`, `port` | `0.0.0.0`, `8000` | bind address (`HOST` / `PORT` env vars override in `serve.sh`) |
| `max_batch_size` | 32 | items per `/predict/batch` request |
| `max_image_bytes` | 10485760 | image file size after base64 decoding |
| `max_image_pixels` | 40000000 | width × height (largest dataset image: 25 MP) |
| `max_aspect_ratio` | 20 | longer side / shorter side (dataset maximum: 4.35) |
| `max_text_chars` | 2000 | caption length limit (the model reads at most 128 WordPiece tokens) |

`serve.sh` maps the environment variables `CHECKPOINT`, `TEXT_MODEL_DIR`, `CLIP_MODEL_DIR`,
`DEVICE`, `HOST` and `PORT` to the matching command-line overrides of `python -m src.serving.app`.

The model loads before the port opens (about 6 s on CPU with local encoder copies; longer on a
first Hub download): until then connections are refused, so clients and health checks should
retry. Request bodies larger than the limits imply (`max_batch_size` × base64 of
`max_image_bytes`, plus captions) are refused with 413 when they declare a `Content-Length`;
a gateway in front of the service should enforce its own body limit as well. There is no
authentication: put the service behind your own gateway if it leaves a trusted network.

## Endpoints / 接口

### `GET /health`

```json
{"status": "ok", "model_loaded": true}
```

### `GET /model`

Model card of the loaded head.

```json
{
  "model_name": "distilbert_clip_frozen",
  "trained_head": true,
  "threshold": 0.5,
  "labels": {"0": "not_sarcastic", "1": "sarcastic"},
  "text_model": "distilbert/distilbert-base-uncased@1c4513b2eedbda136f57676a34eea67aba266e5c",
  "text_feature": "first_token",
  "text_max_length": 128,
  "clip_model": "openai/clip-vit-base-patch32@3d74acf",
  "image_feature": "pooler_output",
  "device": "cpu",
  "checkpoint": "checkpoints/multimodal/20261016_distilbert_clip_seed42/best.pt",
  "checkpoint_epoch": 25,
  "checkpoint_val_macro_f1": 0.5336372734348939
}
```

(`text_model` / `clip_model` show `local_dir:<folder name>` instead when `*_model_dir` is set; `checkpoint`
echoes the configured path, so prefer a repository-relative `CHECKPOINT`.)

### `POST /predict`

Request (`application/json`):

| Field | Type | Required | Notes |
|---|---|---|---|
| `text` | string or null | no | meme caption; null, missing or blank → `[NO_TEXT]` (as in training) |
| `image_base64` | string | yes | base64 of the image **file**: JPEG, PNG, GIF, WebP or BMP (other formats are refused). Standard or URL-safe alphabet; line breaks (e.g. `base64` output), missing padding and a `data:image/...;base64,` prefix are accepted. Any mode (palette, grayscale, CMYK, RGBA) is converted to RGB; an animated GIF uses its first frame. |

Response `200`:

| Field | Type | Meaning |
|---|---|---|
| `probability_sarcastic` | float in [0, 1] | softmax P(class 1) |
| `predicted_label` | 0 or 1 | 1 iff `probability_sarcastic >= threshold` |
| `label` | string | `not_sarcastic` or `sarcastic` |
| `threshold` | float | 0.5, the training threshold (never tuned) |

```bash
curl -s -X POST localhost:8000/predict -H 'Content-Type: application/json' \
     -d @data/samples/predict_request.json
# {"probability_sarcastic":0.7638478875160217,"predicted_label":1,"label":"sarcastic","threshold":0.5}
```

```python
import base64, json, urllib.request
body = {"text": "Oh great, another Monday.",
        "image_base64": base64.b64encode(open("meme.jpg", "rb").read()).decode()}
request = urllib.request.Request("http://localhost:8000/predict", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
print(json.load(urllib.request.urlopen(request, timeout=60)))
```

### `POST /predict/batch`

Request: `{"items": [<predict request>, ...]}` with 1 to `max_batch_size` items. Response:
`{"predictions": [<predict response>, ...]}` in the same order. Items are encoded together
(captions padded to the longest in the batch, which does not change any item's result beyond
float rounding).

```bash
curl -s -X POST localhost:8000/predict/batch -H 'Content-Type: application/json' \
     -d @data/samples/predict_batch_request.json
```

## Errors / 错误

| Status | When | Body |
|---|---|---|
| 400 | `image_base64` is not base64, is empty, is not a JPEG/PNG/GIF/WebP/BMP image, or its pixel data cannot be decoded | `{"detail": "<message>"}` |
| 413 | image file larger than `max_image_bytes`, more than `max_image_pixels` pixels (Pillow's own cap of about 179 MP applies on top), aspect ratio above `max_aspect_ratio`; batch larger than `max_batch_size`; request body above the implied limit | `{"detail": "<message>"}` |
| 422 | missing `image_base64`, wrong types, empty `items`, caption with an unpaired UTF-16 surrogate | `{"detail": [{"type": "...", "loc": [...], "msg": "..."}]}` (the input is not echoed back) |
| 422 | caption longer than `max_text_chars` | `{"detail": "<message>"}` (a string) |
| 503 | the app runs without its startup hook and no model is loaded (not the case under `serve.sh` / uvicorn) | `{"detail": "model not loaded"}` |

In a batch, one invalid item rejects the whole request; nothing is decoded until every item has
passed the size and format checks.

## Behaviour guarantees / 行为约定

* Same preprocessing and weights as training and evaluation: the response for a dataset sample
  equals `probability_1` in that run's `predictions_validation.csv` up to float rounding when the
  run was trained on CPU caches (live check, 20 validation memes through HTTP: max difference
  9.5e-7, all labels equal; `python -m src.evaluation.end_to_end` checks the same on 64). The
  default checkpoint was trained on the image group's CUDA (TF32) cache, so for it the gap can
  reach 0.013 (D19).
* Deterministic: the same request gives the same response on the same machine.
* One request is decoded and run at a time per process (`/health` and `/model` still answer
  while requests queue); scale out with more processes/containers. Measured on a
  4-vCPU CPU machine (two runs): model load 6–6.5 s; `/predict` median 74–98 ms; `/predict/batch`
  with 16 items 0.54–0.72 s.
* The model's validation Macro-F1 is 0.53 (chance level ≈ 0.50). Treat outputs as a baseline
  signal, not a decision ([results report](../Team_Review/multimodal/RESULTS_REPORT.md)).

## Container / 容器

```bash
docker build -t meme-sarcasm .
docker run --rm -p 8000:8000 -v "$PWD/checkpoints:/app/checkpoints:ro" meme-sarcasm
# offline: also mount local copies of the pinned encoders
docker run --rm -p 8000:8000 -v "$PWD/checkpoints:/app/checkpoints:ro" \
    -v "$PWD/hf_local:/app/hf_local:ro" -e TEXT_MODEL_DIR=hf_local/distilbert-base-uncased \
    -e CLIP_MODEL_DIR=hf_local/clip-vit-base-patch32 meme-sarcasm
# another head or port: -e CHECKPOINT=checkpoints/multimodal/<run_id>/best.pt -e PORT=9000 -p 9000:9000
```

The image contains code, configs and the synthetic samples only; the checkpoint (and, offline,
the encoders) are mounted. Inside a container set the port with `PORT` (the health check reads it),
not with `port:` in a mounted serving config.

## Versioning / 版本

`version` in the OpenAPI document is 1.0.0. A change of checkpoint is visible in `/model`
(`checkpoint`, `checkpoint_epoch`); a change of request or response fields bumps the version.
