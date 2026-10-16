"""Tests for the DistilBERT text cache and the multimodal fusion pipeline (no encoder download).

Run from the repository root with MEMOTION_DATA_DIR pointing at the fixed dataset:
    python -m unittest discover -s tests -v
"""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F

from src.common.data import load_split
from src.common.labels import EXPECTED_SPLIT_SIZES
from src.image.cache import file_sha256
from src.multimodal.features import align_by_sample_id, load_split_features
from src.multimodal.model import (FusionHead, build_fusion_head, build_fusion_mlp,
                                  load_fusion_head)
from src.text.cache import load_text_features
from src.text.encoder import pool_text_features

ROOT = Path(__file__).resolve().parents[1]
CFG = json.loads((ROOT / "configs/multimodal_baseline.json").read_text(encoding="utf-8"))


def text_meta(splits, **changes):
    meta = dict(text_model_id=CFG["text_model_id"], text_revision_requested=CFG["text_revision"],
                text_feature=CFG["text_feature"], max_length=CFG["text_max_length"], limit=None,
                splits={s: {"n": EXPECTED_SPLIT_SIZES[s], "dim": 768} for s in splits})
    meta.update(changes)
    return meta


def image_meta(splits):
    return dict(clip_model_id=CFG["clip_model_id"], clip_revision_requested=CFG["clip_revision"],
                feature=CFG["image_feature"], limit=None,
                splits={s: {"n": EXPECTED_SPLIT_SIZES[s], "dim": 768} for s in splits})


def write_cache(directory, split, features, meta, order=None):
    truth = load_split(split)
    ids = truth.sample_id.tolist()
    labels = torch.tensor(truth.sarcasm_binary.to_numpy(), dtype=torch.long)
    if order is not None:
        ids, labels, features = [ids[i] for i in order], labels[order], features[order]
    directory.mkdir(parents=True, exist_ok=True)
    torch.save(dict(sample_id=ids, features=features, labels=labels), directory / f"{split}.pt")
    meta["splits"][split]["sha256"] = file_sha256(directory / f"{split}.pt")
    (directory / "meta.json").write_text(json.dumps(meta), encoding="utf-8")


class MultimodalPipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="multimodal_pipeline_")
        self.work = Path(self.temp.name).resolve()
        self.addCleanup(self.temp.cleanup)

    def test_text_cache_accepts_complete_and_rejects_wrong_provenance(self):
        directory = self.work / "text"
        n = EXPECTED_SPLIT_SIZES["validation"]
        write_cache(directory, "validation", torch.randn(n, 768), text_meta(["validation"]))
        ids, x, y = load_text_features(directory, "validation", CFG)
        self.assertEqual(ids, load_split("validation").sample_id.tolist())
        self.assertEqual(tuple(x.shape), (n, 768))
        for key, value in (("limit", 64), ("text_feature", "mean_pooling"), ("max_length", 512),
                           ("text_model_id", "bert-base-uncased")):
            with self.subTest(key=key):
                write_cache(directory, "validation", torch.randn(n, 768),
                            text_meta(["validation"], **{key: value}))
                with self.assertRaises(ValueError):
                    load_text_features(directory, "validation", CFG)

    def test_text_cache_rejects_shuffled_rows_and_changed_values(self):
        directory = self.work / "text"
        n = EXPECTED_SPLIT_SIZES["validation"]
        order = torch.randperm(n, generator=torch.Generator().manual_seed(0))
        write_cache(directory, "validation", torch.randn(n, 768), text_meta(["validation"]), order)
        with self.assertRaisesRegex(ValueError, "order"):
            load_text_features(directory, "validation", CFG)
        write_cache(directory, "validation", torch.randn(n, 768), text_meta(["validation"]))
        cached = torch.load(directory / "validation.pt", weights_only=True)
        cached["features"][0, 0] += 1
        torch.save(cached, directory / "validation.pt")
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            load_text_features(directory, "validation", CFG)

    def test_alignment_uses_sample_id_not_row_number(self):
        ids = ["a", "b", "c"]
        x = torch.tensor([[1.0], [2.0], [3.0]])
        y = torch.tensor([0, 1, 1])
        x2, y2 = align_by_sample_id(["c", "a", "b"], x[[2, 0, 1]], y[[2, 0, 1]], ids)
        self.assertTrue(torch.equal(x2, x))
        self.assertTrue(torch.equal(y2, y))
        with self.assertRaises(ValueError):
            align_by_sample_id(["a", "b", "d"], x, y, ids)

    def test_joint_loader_aligns_text_and_image(self):
        n = EXPECTED_SPLIT_SIZES["validation"]
        text, image = torch.randn(n, 768), torch.randn(n, 768)
        write_cache(self.work / "text", "validation", text, text_meta(["validation"]))
        write_cache(self.work / "image", "validation", image, image_meta(["validation"]))
        ids, feats, y = load_split_features(CFG, "validation", self.work / "text",
                                            self.work / "image")
        self.assertEqual(ids, load_split("validation").sample_id.tolist())
        self.assertTrue(torch.equal(feats["text"], text))
        self.assertTrue(torch.equal(feats["image"], image))

    def test_fusion_head_normalizes_then_concatenates_text_first(self):
        head = build_fusion_head(CFG)
        self.assertEqual(head.input_dim, 1536)
        text, image = torch.randn(5, 768) * 7, torch.randn(5, 768) * 0.1
        fused = head.fuse(text, image)
        self.assertTrue(torch.allclose(fused, torch.cat([F.normalize(text, dim=-1),
                                                         F.normalize(image, dim=-1)], dim=-1)))
        unimodal = FusionHead(modalities=["image"])
        self.assertEqual(unimodal.input_dim, 768)
        with self.assertRaises(ValueError):
            unimodal(text=text)
        wrong = copy.deepcopy(CFG)
        wrong["modalities"] = ["text"]
        with self.assertRaisesRegex(ValueError, "input_dim"):
            build_fusion_head(wrong)

    def test_head_checkpoint_loads_into_end_to_end_classifier(self):
        torch.manual_seed(0)
        head = build_fusion_head(CFG).eval()
        stand_in = SimpleNamespace(classifier=build_fusion_mlp(1536).eval())
        load_fusion_head(stand_in, head.state_dict())
        text, image = torch.randn(4, 768), torch.randn(4, 768)
        with torch.no_grad():
            self.assertTrue(torch.equal(stand_in.classifier(head.fuse(text, image)),
                                        head(text, image)))

    def test_pooling_ignores_padding(self):
        hidden = torch.randn(2, 5, 768)
        mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]])
        hidden_garbage = hidden.clone()
        hidden_garbage[0, 3:] = 1e6
        mean = pool_text_features(hidden_garbage, mask, "mean_pooling")
        self.assertTrue(torch.allclose(mean[0], hidden[0, :3].mean(0), atol=1e-5))
        self.assertTrue(torch.allclose(mean[1], hidden[1].mean(0), atol=1e-5))
        self.assertTrue(torch.equal(pool_text_features(hidden, mask, "first_token"), hidden[:, 0]))
        with self.assertRaises(ValueError):
            pool_text_features(hidden, mask, "pooler_output")

    def test_text_extraction_refuses_existing_cache_before_loading_model(self):
        from src.text import extract_features
        existing = self.work / "features/distilbert_first_token"
        existing.mkdir(parents=True)
        (existing / "train.pt").write_bytes(b"full cache")
        with patch.object(sys, "argv", ["extract", "--out_prefix",
                                        str(self.work / "features/distilbert")]), \
                patch.object(extract_features, "load_text_encoder") as loader, \
                self.assertRaises(FileExistsError):
            extract_features.main()
        loader.assert_not_called()
        self.assertEqual((existing / "train.pt").read_bytes(), b"full cache")

    def run_cli(self, *args):
        result = subprocess.run([sys.executable, *map(str, args)], cwd=self.work,
                                capture_output=True, text=True, encoding="utf-8",
                                errors="replace", timeout=600)
        return result

    def test_train_evaluate_round_trip_on_synthetic_caches(self):
        generator = torch.Generator().manual_seed(0)
        for split in ("train", "validation"):
            n = EXPECTED_SPLIT_SIZES[split]
            write_cache(self.work / "text", split, torch.randn(n, 768, generator=generator),
                        text_meta(["train", "validation"]) if split == "train" else
                        json.loads((self.work / "text/meta.json").read_text()))
            write_cache(self.work / "image", split, torch.randn(n, 768, generator=generator),
                        image_meta(["train", "validation"]) if split == "train" else
                        json.loads((self.work / "image/meta.json").read_text()))
        cfg = dict(CFG, epochs=2, batch_size=256)
        (self.work / "cfg.json").write_text(json.dumps(cfg), encoding="utf-8")
        train = self.run_cli(ROOT / "src/multimodal/train.py", "--config", "cfg.json",
                             "--text_features_dir", "text", "--image_features_dir", "image",
                             "--run_id", "synthetic")
        self.assertEqual(train.returncode, 0, train.stdout + train.stderr)
        run = self.work / "outputs/multimodal/synthetic"
        for name in ("config.json", "metrics_validation.json", "predictions_validation.csv",
                     "confusion_matrix_validation.png", "training_log.csv", "README.md"):
            self.assertTrue((run / name).is_file(), name)
        verify = self.run_cli(ROOT / "scripts/verify_predictions.py",
                              run / "predictions_validation.csv")
        self.assertEqual(verify.returncode, 0, verify.stdout + verify.stderr)
        reload = self.run_cli(ROOT / "src/multimodal/evaluate.py", "--run_id", "synthetic")
        self.assertEqual(reload.returncode, 0, reload.stdout + reload.stderr)
        compare = self.run_cli(ROOT / "scripts/verify_reproduction.py",
                               "--reference_metrics", run / "metrics_validation.json",
                               "--candidate_metrics", run / "reeval/metrics_validation.json",
                               "--tolerance", "0",
                               "--reference_predictions", run / "predictions_validation.csv",
                               "--candidate_predictions", run / "reeval/predictions_validation.csv")
        self.assertEqual(compare.returncode, 0, compare.stdout + compare.stderr)
        again = self.run_cli(ROOT / "src/multimodal/train.py", "--config", "cfg.json",
                             "--text_features_dir", "text", "--image_features_dir", "image",
                             "--run_id", "synthetic")
        self.assertNotEqual(again.returncode, 0)
        self.assertIn("Existing output", again.stderr)


if __name__ == "__main__":
    unittest.main()
