"""Serving: the inference wrapper and the FastAPI service (no download, no dataset needed).

Uses one-layer random encoders saved to a temporary folder (tests/helpers.py) and the synthetic
samples in data/samples/, so it runs anywhere in a few seconds:
    python -m unittest tests.test_serving -v
"""
import base64
import io
import json
import os
import struct
import zlib
from pathlib import Path
import sys
import tempfile
import unittest

import torch
import yaml
from fastapi.testclient import TestClient
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

from helpers import MM_CFG, ROOT, SAMPLES, make_tiny_encoders  # noqa: E402
from src.data.dataset import NO_TEXT, load_split  # noqa: E402
from src.models.multimodal_classifier import build_fusion_head  # noqa: E402
from src.serving.app import create_app  # noqa: E402
from src.serving.predictor import MultimodalPredictor, as_rgb_image  # noqa: E402
from src.utils.config import load_config  # noqa: E402


def header_only_png(width, height):
    """A valid PNG signature + IHDR declaring width x height, with (almost) no pixel data."""
    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(b"\x00" * 4)) + chunk(b"IEND", b""))


def png_base64(color=(200, 30, 30), size=(48, 40), noise=False):
    buffer = io.BytesIO()
    image = (Image.frombytes("RGB", size, os.urandom(size[0] * size[1] * 3)) if noise else
             Image.new("RGB", size, color))
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


class TinyModelMixin:
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix="serving_")
        cls.work = Path(cls.temp.name)
        cls.text_dir, cls.clip_dir = make_tiny_encoders(cls.work)
        torch.manual_seed(1)
        cls.head = build_fusion_head(MM_CFG).eval()
        cls.ckpt_path = cls.work / "best.pt"
        torch.save({"state_dict": cls.head.state_dict(), "config": MM_CFG, "epoch": 3,
                    "val_macro_f1": 0.5, "class_weights": [2.0, 0.6],
                    "modalities": ["text", "image"]}, cls.ckpt_path)
        cls.serving_config = cls.work / "serving.yaml"
        cls.serving_config.write_text(yaml.safe_dump({
            "checkpoint": str(cls.ckpt_path), "text_model_dir": cls.text_dir,
            "clip_model_dir": cls.clip_dir, "device": "cpu", "max_batch_size": 3,
            "max_image_bytes": 50_000, "max_text_chars": 100}), encoding="utf-8")
        cls.predictor = MultimodalPredictor.from_serving_config(cls.serving_config)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()


class PredictorTests(TinyModelMixin, unittest.TestCase):
    def test_predictions_follow_the_training_threshold(self):
        frame = load_split("validation", SAMPLES)
        results = self.predictor.predict(frame.text_input.tolist(), frame.image_abspath.tolist())
        self.assertEqual(len(results), len(frame))
        for r in results:
            self.assertGreaterEqual(r["probability_sarcastic"], 0.0)
            self.assertLessEqual(r["probability_sarcastic"], 1.0)
            self.assertEqual(r["predicted_label"], int(r["probability_sarcastic"] >= 0.5))
            self.assertEqual(r["label"], ("not_sarcastic", "sarcastic")[r["predicted_label"]])
        self.assertTrue(self.predictor.trained)
        self.assertEqual(self.predictor.info["checkpoint_epoch"], 3)

    def test_end_to_end_equals_the_head_on_extracted_features(self):
        frame = load_split("train", SAMPLES)
        out = self.predictor.forward(frame.text_input.tolist(), frame.image_abspath.tolist())
        self.assertEqual(tuple(out["fused_features"].shape), (4, 1536))
        with torch.no_grad():
            head_prob = self.head(out["text_features"], out["image_features"]).softmax(-1)[:, 1]
        self.assertTrue(torch.allclose(head_prob, out["probability_1"], atol=1e-6))

    def test_blank_caption_is_no_text_and_inputs_are_checked(self):
        image = SAMPLES / "validation/images/validation_000000.png"
        blank = self.predictor.forward([None, "  "], [image, image])["probability_1"]
        explicit = self.predictor.forward([NO_TEXT, NO_TEXT], [image, image])["probability_1"]
        self.assertTrue(torch.equal(blank, explicit))
        with self.assertRaises(ValueError):
            self.predictor.forward(["a"], [])
        with self.assertRaises(ValueError):
            self.predictor.forward([], [])

    def test_image_inputs_paths_bytes_and_pil_agree(self):
        path = SAMPLES / "test/images/test_000002.png"
        palette = Image.open(path).convert("P")
        for image in (path, str(path), path.read_bytes(), Image.open(path), palette):
            with self.subTest(kind=type(image).__name__):
                self.assertEqual(as_rgb_image(image).mode, "RGB")
        probs = [self.predictor.forward(["x"], [image])["probability_1"]
                 for image in (path, path.read_bytes(), Image.open(path))]
        self.assertTrue(torch.equal(probs[0], probs[1]) and torch.equal(probs[0], probs[2]))

    def test_per_image_preprocessing_equals_the_batched_processor(self):
        paths = load_split("validation", SAMPLES).image_abspath.tolist()
        batched = self.predictor.processor(images=[as_rgb_image(p) for p in paths],
                                           return_tensors="pt")["pixel_values"]
        self.assertTrue(torch.equal(self.predictor.pixel_values(paths), batched))

    def test_missing_checkpoint_explains_where_to_get_it(self):
        with self.assertRaisesRegex(FileNotFoundError, "Multimodal_Assets"):
            MultimodalPredictor.from_checkpoint(self.work / "nope.pt", self.text_dir, self.clip_dir)

    def test_untrained_head_is_flagged(self):
        untrained = MultimodalPredictor(MM_CFG, None, self.text_dir, self.clip_dir, "cpu")
        self.assertFalse(untrained.info["trained_head"])
        with self.assertRaises(ValueError):
            MultimodalPredictor(dict(MM_CFG, modalities=["text"]), None, self.text_dir,
                                self.clip_dir, "cpu")

    def test_predictor_cli_smoke_on_synthetic_samples(self):
        from src.serving import predictor as module
        self.assertEqual(module.main(["--serving_config", str(self.serving_config), "--smoke",
                                      "--data_dir", str(SAMPLES), "--batch_size", "3"]), 0)


class AppTests(TinyModelMixin, unittest.TestCase):
    def setUp(self):
        self.client = TestClient(create_app(self.predictor, serving_config=str(self.serving_config)))
        self.sample = json.loads((ROOT / "data/samples/predict_request.json").read_text())

    def test_health_and_model_card(self):
        self.assertEqual(self.client.get("/health").json(), {"status": "ok", "model_loaded": True})
        card = self.client.get("/model").json()
        self.assertEqual(card["model_name"], MM_CFG["model_name"])
        self.assertEqual(card["threshold"], 0.5)
        self.assertEqual(card["text_feature"], "first_token")

    def test_predict_matches_the_predictor(self):
        response = self.client.post("/predict", json=self.sample)
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(set(body), {"probability_sarcastic", "predicted_label", "label", "threshold"})
        image = base64.b64decode(self.sample["image_base64"])
        direct = self.predictor.predict([self.sample["text"]], [image])[0]
        self.assertAlmostEqual(body["probability_sarcastic"], direct["probability_sarcastic"], places=6)
        self.assertEqual(body["predicted_label"], direct["predicted_label"])
        data_url = dict(self.sample, image_base64="data:image/png;base64," + self.sample["image_base64"])
        self.assertEqual(self.client.post("/predict", json=data_url).json(), body)
        no_text = self.client.post("/predict", json={"image_base64": self.sample["image_base64"]})
        self.assertEqual(no_text.status_code, 200, no_text.text)

    def test_batch_keeps_order(self):
        items = [{"text": "oh great", "image_base64": png_base64((10 * i, 50, 90))} for i in range(3)]
        response = self.client.post("/predict/batch", json={"items": items})
        self.assertEqual(response.status_code, 200, response.text)
        batch = response.json()["predictions"]
        singles = [self.client.post("/predict", json=item).json() for item in items]
        for a, b in zip(batch, singles):
            self.assertAlmostEqual(a["probability_sarcastic"], b["probability_sarcastic"], places=6)

    def test_bad_requests_are_rejected(self):
        not_image = base64.b64encode(b"definitely not an image").decode()
        eps = base64.b64encode(b"%!PS-Adobe-3.0 EPSF-3.0\n%%BoundingBox: 0 0 10 10\nshowpage\n").decode()
        b64 = lambda raw: base64.b64encode(raw).decode()  # noqa: E731
        thin = io.BytesIO(); Image.new("RGB", (100, 2)).save(thin, format="PNG")
        cases = [
            ("/predict", {"text": "x", "image_base64": eps}, 400),                     # EPS refused
            ("/predict", {"text": "x", "image_base64": b64(header_only_png(9000, 9000))}, 413),
            ("/predict", {"text": "x", "image_base64": b64(thin.getvalue())}, 413),   # aspect 50
            ("/predict", {"text": "x", "image_base64": "%%%not base64%%%"}, 400),
            ("/predict", {"text": "x", "image_base64": ""}, 400),
            ("/predict", {"text": "x", "image_base64": not_image}, 400),
            ("/predict", {"text": "x", "image_base64": png_base64(size=(200, 200), noise=True)}, 413),
            ("/predict", {"text": "x" * 101, "image_base64": png_base64()}, 422),
            ("/predict", {"text": "missing image"}, 422),
            ("/predict/batch", {"items": []}, 422),
            ("/predict/batch", {"items": [{"image_base64": png_base64()}] * 4}, 413),
        ]
        for path, body, status in cases:
            with self.subTest(path=path, status=status, body=str(body)[:60]):
                self.assertEqual(self.client.post(path, json=body).status_code, status)
        # lone UTF-16 surrogate (a cut emoji), sent as escaped JSON like a JavaScript client would
        surrogate = self.client.post("/predict", headers={"Content-Type": "application/json"},
                                     content=json.dumps({"text": "\ud800", "image_base64": png_base64()}))
        self.assertEqual(surrogate.status_code, 422, surrogate.text)
        too_big = self.client.post("/predict", content=b"{" + b" " * 400_000 + b"}",
                                   headers={"Content-Type": "application/json"})
        self.assertEqual(too_big.status_code, 413)

    def test_common_base64_forms_are_accepted(self):
        raw = base64.b64decode(self.sample["image_base64"])
        reference = self.client.post("/predict", json=self.sample).json()
        forms = {"wrapped": base64.encodebytes(raw).decode(),
                 "unpadded": base64.b64encode(raw).decode().rstrip("="),
                 "urlsafe": base64.urlsafe_b64encode(raw).decode()}
        for name, data in forms.items():
            with self.subTest(form=name):
                response = self.client.post("/predict", json={"text": self.sample["text"],
                                                              "image_base64": data})
                self.assertEqual(response.status_code, 200, response.text)
                self.assertEqual(response.json(), reference)

    def test_without_a_model_the_api_answers_503(self):
        client = TestClient(create_app(serving_config=str(self.serving_config)))  # no startup hook
        self.assertEqual(client.get("/health").json()["model_loaded"], False)
        self.assertEqual(client.get("/model").status_code, 503)
        self.assertEqual(client.post("/predict", json=self.sample).status_code, 503)

    def test_lazy_loading_from_the_serving_config(self):
        with TestClient(create_app(serving_config=str(self.serving_config))) as client:
            self.assertTrue(client.get("/health").json()["model_loaded"])
            self.assertEqual(client.post("/predict", json=self.sample).status_code, 200)
        # command-line style override of the checkpoint (serve.sh CHECKPOINT=...)
        broken = self.work / "broken.yaml"
        broken.write_text(self.serving_config.read_text().replace(str(self.ckpt_path), "missing.pt"))
        with TestClient(create_app(serving_config=str(broken),
                                   overrides={"checkpoint": str(self.ckpt_path)})) as client:
            self.assertEqual(client.get("/model").json()["checkpoint"], str(self.ckpt_path))

    def test_deploy_config_names_the_reference_run(self):
        cfg = load_config(ROOT / "configs/deploy/serving.yaml")
        self.assertEqual(cfg["checkpoint"],
                         "checkpoints/multimodal/20261016_distilbert_clip_seed42/best.pt")
        for key in ("text_model_dir", "clip_model_dir", "device", "host", "port", "max_batch_size",
                    "max_image_bytes", "max_image_pixels", "max_aspect_ratio", "max_text_chars"):
            self.assertIn(key, cfg)


if __name__ == "__main__":
    unittest.main()
