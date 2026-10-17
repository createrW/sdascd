"""Data, preprocessing, feature caches, the joint loader and the extraction CLI.

Covers the image group's cache / output / reproduction safety tests and the multimodal text-cache
and alignment tests. Run from the repository root with MEMOTION_DATA_DIR set:
    python -m unittest discover -s tests -v
Fixtures stay in REVIEW_TEST_DIR when set; otherwise temporary fixtures are removed.
"""
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

from helpers import (IMAGE_CFG, MM_CFG, ROOT, SAMPLES, image_meta, text_meta,  # noqa: E402
                     work_dir, write_cache)
from src.data.dataset import EXPECTED_SPLIT_SIZES, NO_TEXT, load_split  # noqa: E402
from src.data.loader import (DEFAULT_IMAGE_FEATURES_DIR, DEFAULT_IMAGE_TASK_FEATURES_DIR,  # noqa: E402
                             align_by_sample_id, features_dirs, load_split_features)
from src.data.preprocess import clean_text, to_rgb  # noqa: E402
from src.evaluation.metrics import predictions_frame  # noqa: E402
from src.evaluation.reproduction import compare_metrics, compare_predictions  # noqa: E402
from src.features import feature_engineering  # noqa: E402
from src.features.cache import (file_sha256, load_image_features, load_text_features,  # noqa: E402
                                resolve_features_dir)
from src.utils.config import infer_task, load_config  # noqa: E402
from src.utils.safety import require_new_paths, validate_run_id  # noqa: E402

CFG = IMAGE_CFG


class DataAndPreprocessTests(unittest.TestCase):
    def test_blank_and_missing_captions_become_no_text(self):
        for value in (None, float("nan"), "", "   "):
            self.assertEqual(clean_text(value), NO_TEXT)
        self.assertEqual(clean_text(" keep spacing "), " keep spacing ")
        self.assertEqual(clean_text(NO_TEXT), NO_TEXT)

    def test_palette_transparency_converts_to_rgb(self):
        im = Image.new("P", (4, 4))
        im.putpalette([255, 0, 0] * 256)
        im.info["transparency"] = 0
        self.assertEqual(to_rgb(im).mode, "RGB")
        self.assertEqual(to_rgb(Image.new("L", (2, 2))).mode, "RGB")

    def test_synthetic_samples_follow_the_dataset_layout(self):
        for split in ("train", "validation", "test"):
            frame = load_split(split, SAMPLES)
            self.assertEqual(len(frame), 4)
            self.assertTrue(frame.sample_id.is_unique)
            self.assertIn(NO_TEXT, frame.text_input.tolist())
            for path in frame.image_abspath:
                self.assertTrue(Path(path).is_file(), path)

    def test_every_training_config_loads_with_its_task(self):
        expected = {"image_baseline": "image", "multimodal_baseline": "multimodal",
                    "multimodal_text_only": "multimodal", "multimodal_image_only": "multimodal",
                    "multimodal_mean_pooling": "multimodal"}
        for name, task in expected.items():
            with self.subTest(config=name):
                cfg = load_config(ROOT / f"configs/train/{name}.yaml")
                self.assertEqual(cfg["task"], task)
                self.assertEqual(infer_task({k: v for k, v in cfg.items() if k != "task"}), task)
                self.assertEqual(cfg["threshold"], 0.5)
                self.assertIsInstance(cfg["lr"], float)
                self.assertIsInstance(cfg["weight_decay"], float)
        with self.assertRaises(ValueError):
            infer_task({"task": "audio"})

    def test_feature_dirs_precedence(self):
        self.assertEqual(features_dirs(CFG), {"image": CFG["features"]["image"]})
        no_block = {k: v for k, v in CFG.items() if k != "features"}
        self.assertEqual(features_dirs(no_block)["image"], DEFAULT_IMAGE_TASK_FEATURES_DIR)
        mm = {k: v for k, v in MM_CFG.items() if k != "features"}
        self.assertEqual(features_dirs(mm), {"image": DEFAULT_IMAGE_FEATURES_DIR,
                                             "text": "data/processed/features/distilbert_first_token"})
        self.assertEqual(features_dirs(MM_CFG, "t", "i"), {"image": "i", "text": "t"})


class FeatureCacheTests(unittest.TestCase):
    """The image group's cache checks (load_image_features) plus the text cache."""

    def setUp(self):
        self.work = work_dir(self, "feature_cache_")
        self.truth = load_split("validation")

    def cli(self, script, *args):
        result = subprocess.run([sys.executable, str(ROOT / script), *map(str, args)],
                                cwd=self.work, capture_output=True, text=True,
                                encoding="utf-8", errors="replace", timeout=60)
        (self.work / "command.log").write_text(result.stdout + result.stderr, encoding="utf-8")
        return result

    def cache(self):
        directory = self.work / "cache"
        directory.mkdir()
        data = dict(sample_id=self.truth.sample_id.tolist(),
                    features=torch.zeros(699, 768),
                    labels=torch.tensor(self.truth.sarcasm_binary.to_numpy(), dtype=torch.long))
        meta = dict(clip_model_id=CFG["clip_model_id"], clip_revision_requested=CFG["clip_revision"],
                    feature=CFG["feature"], limit=None, splits={"validation": {"n": 699, "dim": 768}})
        self.write_cache(directory, data, meta)
        return directory, data, meta

    @staticmethod
    def write_cache(directory, data, meta):
        torch.save(data, directory / "validation.pt")
        (directory / "meta.json").write_text(json.dumps(meta), encoding="utf-8")

    def test_complete_cache_is_accepted(self):
        directory, _, _ = self.cache()
        ids, features, labels = load_image_features(directory, "validation", CFG)
        self.assertEqual(ids, self.truth.sample_id.tolist())
        self.assertEqual(tuple(features.shape), (699, 768))
        self.assertTrue(torch.equal(labels, torch.tensor(self.truth.sarcasm_binary.to_numpy())))

    def test_cache_missing_metadata_is_rejected(self):
        directory, _, _ = self.cache()
        (directory / "meta.json").unlink()
        with self.assertRaises(FileNotFoundError):
            load_image_features(directory, "validation", CFG)

    def test_cache_provenance_and_partial_counts_are_rejected(self):
        directory, data, meta = self.cache()
        changes = (("limit", 128), ("clip_revision_requested", "wrong_revision"),
                   ("clip_model_id", "wrong_model"), ("feature", "image_embeds"))
        for key, value in changes:
            with self.subTest(key=key):
                changed = copy.deepcopy(meta)
                changed[key] = value
                self.write_cache(directory, data, changed)
                with self.assertRaises(ValueError):
                    load_image_features(directory, "validation", CFG)
        meta["splits"]["validation"]["n"] = 128
        self.write_cache(directory, data, meta)
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            load_image_features(directory, "validation", CFG)

    def test_incomplete_rows_rejected_even_if_meta_claims_complete(self):
        directory, data, meta = self.cache()
        for key in data:
            data[key] = data[key][:128]
        self.write_cache(directory, data, meta)
        with self.assertRaises(ValueError):
            load_image_features(directory, "validation", CFG)

    def test_cache_ids_labels_dimensions_and_nonfinite_values_are_rejected(self):
        directory, data, meta = self.cache()
        variants = []
        duplicate = copy.deepcopy(data); duplicate["sample_id"][0] = duplicate["sample_id"][1]; variants.append(duplicate)
        wrong_id = copy.deepcopy(data); wrong_id["sample_id"][0] = "other_sample"; variants.append(wrong_id)
        wrong_label = copy.deepcopy(data); wrong_label["labels"][0] = 1 - wrong_label["labels"][0]; variants.append(wrong_label)
        wrong_dim = copy.deepcopy(data); wrong_dim["features"] = torch.zeros(699, 512); variants.append(wrong_dim)
        wrong_dtype = copy.deepcopy(data); wrong_dtype["labels"] = wrong_dtype["labels"].float(); variants.append(wrong_dtype)
        wrong_features_dtype = copy.deepcopy(data); wrong_features_dtype["features"] = wrong_features_dtype["features"].double(); variants.append(wrong_features_dtype)
        for value in (float("nan"), float("inf")):
            changed = copy.deepcopy(data); changed["features"][0, 0] = value; variants.append(changed)
        for index, changed in enumerate(variants):
            with self.subTest(case=index):
                self.write_cache(directory, changed, meta)
                with self.assertRaises(ValueError):
                    load_image_features(directory, "validation", CFG)

    def test_new_cache_hash_detects_changed_feature_values(self):
        directory, data, meta = self.cache()
        meta["splits"]["validation"]["sha256"] = file_sha256(directory / "validation.pt")
        data["features"][0, 0] = 10
        self.write_cache(directory, data, meta)
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            load_image_features(directory, "validation", CFG)

    def test_text_cache_accepts_complete_and_rejects_wrong_provenance(self):
        directory = self.work / "text"
        n = EXPECTED_SPLIT_SIZES["validation"]
        write_cache(directory, "validation", torch.randn(n, 768), text_meta(["validation"]))
        ids, x, _ = load_text_features(directory, "validation", MM_CFG)
        self.assertEqual(ids, self.truth.sample_id.tolist())
        self.assertEqual(tuple(x.shape), (n, 768))
        for key, value in (("limit", 64), ("text_feature", "mean_pooling"), ("max_length", 512),
                           ("text_model_id", "bert-base-uncased")):
            with self.subTest(key=key):
                write_cache(directory, "validation", torch.randn(n, 768),
                            text_meta(["validation"], **{key: value}))
                with self.assertRaises(ValueError):
                    load_text_features(directory, "validation", MM_CFG)

    def test_text_cache_rejects_shuffled_rows_and_changed_values(self):
        directory = self.work / "text"
        n = EXPECTED_SPLIT_SIZES["validation"]
        order = torch.randperm(n, generator=torch.Generator().manual_seed(0))
        write_cache(directory, "validation", torch.randn(n, 768), text_meta(["validation"]), order)
        with self.assertRaisesRegex(ValueError, "order"):
            load_text_features(directory, "validation", MM_CFG)
        write_cache(directory, "validation", torch.randn(n, 768), text_meta(["validation"]))
        cached = torch.load(directory / "validation.pt", weights_only=True)
        cached["features"][0, 0] += 1
        torch.save(cached, directory / "validation.pt")
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            load_text_features(directory, "validation", MM_CFG)

    def test_legacy_and_new_cache_locations_resolve_both_ways(self):
        cwd = Path.cwd()
        try:
            os.chdir(self.work)
            Path("features/old_layout").mkdir(parents=True)
            Path("data/processed/features/new_layout").mkdir(parents=True)
            self.assertEqual(resolve_features_dir("data/processed/features/old_layout"),
                             Path("features/old_layout"))
            self.assertEqual(resolve_features_dir("features/new_layout"),
                             Path("data/processed/features/new_layout"))
            self.assertEqual(resolve_features_dir("features/missing"), Path("features/missing"))
            absolute = self.work / "elsewhere"
            self.assertEqual(resolve_features_dir(absolute), absolute)
        finally:
            os.chdir(cwd)

    def test_threshold_probabilities_survive_csv_and_verifier(self):
        probs = np.full(699, 0.7, dtype=np.float64)
        probs[:7] = [np.nextafter(0.5, 0), 0.5, np.nextafter(0.5, 1),
                     np.nextafter(np.float32(0.5), np.float32(0)),
                     np.nextafter(np.float32(0.5), np.float32(1)), 0, 1]
        frame = predictions_frame(self.truth.sample_id, self.truth.sarcasm_binary, (probs >= 0.5),
                                  probs, CFG["model_name"], "validation", 42)
        path = self.work / "boundary.csv"
        frame.to_csv(path, index=False)
        restored = pd.read_csv(path, float_precision="round_trip")
        np.testing.assert_array_equal(restored.probability_1.to_numpy(), probs)
        result = self.cli("scripts/verify_predictions.py", path)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


class JointLoaderTests(unittest.TestCase):
    def setUp(self):
        self.work = work_dir(self, "joint_loader_")

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
        ids, feats, _ = load_split_features(MM_CFG, "validation", self.work / "text",
                                            self.work / "image")
        self.assertEqual(ids, load_split("validation").sample_id.tolist())
        self.assertTrue(torch.equal(feats["text"], text))
        self.assertTrue(torch.equal(feats["image"], image))

    def test_image_task_loads_only_the_image_cache(self):
        n = EXPECTED_SPLIT_SIZES["validation"]
        image = torch.randn(n, 768)
        write_cache(self.work / "image", "validation", image, image_meta(["validation"], CFG))
        _, feats, _ = load_split_features(CFG, "validation", self.work / "no_text_needed",
                                          self.work / "image")
        self.assertEqual(set(feats), {"image"})
        self.assertTrue(torch.equal(feats["image"], image))


class ExtractionCliTests(unittest.TestCase):
    def setUp(self):
        self.work = work_dir(self, "extraction_cli_")
        self.truth = load_split("validation")

    def test_duplicate_and_invalid_extraction_arguments_are_rejected(self):
        for modality, loader_name in (("image", "load_clip"), ("text", "load_text_encoder")):
            for arguments in (["--limit", "0"], ["--limit", "-1"], ["--batch_size", "0"],
                              ["--splits", "train", "train"], ["--splits", "unknown"]):
                with self.subTest(modality=modality, arguments=arguments), \
                        patch.object(feature_engineering, loader_name) as loader, \
                        self.assertRaises(SystemExit) as error:
                    feature_engineering.main([modality, *arguments])
                self.assertEqual(error.exception.code, 2)
                loader.assert_not_called()
        with self.assertRaises(SystemExit):
            feature_engineering.main([])          # a modality is required

    def test_path_traversal_run_ids_are_rejected(self):
        for name in ("../old", "a/b", "a\\b", "", ".."):
            with self.subTest(name=name), self.assertRaises(ValueError):
                validate_run_id(name)
        require_new_paths(self.work / "new")
        with self.assertRaises(FileExistsError):
            require_new_paths(self.work)

    def test_existing_cache_rejected_before_loading_clip(self):
        cache = self.work / "formal_cache"
        cache.mkdir()
        marker = cache / "train.pt"
        marker.write_bytes(b"full cache")
        with patch.object(feature_engineering, "load_clip") as loader, \
                self.assertRaises(FileExistsError):
            feature_engineering.main(["image", "--out_dir", str(cache), "--limit", "2"])
        loader.assert_not_called()
        self.assertEqual(marker.read_bytes(), b"full cache")

    def test_text_extraction_refuses_existing_cache_before_loading_model(self):
        existing = self.work / "features/distilbert_first_token"
        existing.mkdir(parents=True)
        (existing / "train.pt").write_bytes(b"full cache")
        with patch.object(feature_engineering, "load_text_encoder") as loader, \
                self.assertRaises(FileExistsError):
            feature_engineering.main(["text", "--out_prefix", str(self.work / "features/distilbert")])
        loader.assert_not_called()
        self.assertEqual((existing / "train.pt").read_bytes(), b"full cache")

    def test_limited_default_cache_is_separate(self):
        original_cwd = Path.cwd()
        try:
            os.chdir(self.work)
            formal = self.work / "data/processed/features/clip_vitb32_pooler"
            formal.mkdir(parents=True)
            (formal / "train.pt").write_bytes(b"full cache")
            result = dict(sample_id=self.truth.sample_id.tolist()[:2], features=torch.zeros(2, 768),
                          labels=torch.tensor(self.truth.sarcasm_binary.to_numpy()[:2], dtype=torch.long))
            with patch.object(feature_engineering, "load_clip",
                              return_value=(SimpleNamespace(config=SimpleNamespace()), None)), \
                    patch.object(feature_engineering, "extract_image_split", return_value=result), \
                    patch.object(feature_engineering, "get_device", return_value=torch.device("cpu")):
                feature_engineering.main(["image", "--config", str(ROOT / "configs/train/image_baseline.yaml"),
                                          "--limit", "2", "--splits", "validation"])
            smoke = self.work / "data/processed/features/clip_vitb32_pooler_smoke_2"
            meta = json.loads((smoke / "meta.json").read_text())
            self.assertEqual(meta["limit"], 2)
            self.assertEqual(meta["splits"]["validation"]["sha256"], file_sha256(smoke / "validation.pt"))
            self.assertEqual((formal / "train.pt").read_bytes(), b"full cache")
            with self.assertRaisesRegex(ValueError, "Smoke/limited"):
                load_image_features(smoke, "validation", CFG)
        finally:
            os.chdir(original_cwd)


class ReproductionTests(unittest.TestCase):
    def setUp(self):
        self.work = work_dir(self, "reproduction_")
        self.truth = load_split("validation")

    def cli(self, script, *args):
        return subprocess.run([sys.executable, str(ROOT / script), *map(str, args)],
                              cwd=self.work, capture_output=True, text=True,
                              encoding="utf-8", errors="replace", timeout=60)

    def metrics(self, macro_f1=0.5):
        return dict(model_name=CFG["model_name"], split="validation", seed=42, threshold=0.5, macro_f1=macro_f1)

    def test_reproduction_tolerance_and_invalid_references(self):
        reference = self.work / "reference.json"
        reference.write_text(json.dumps(self.metrics()), encoding="utf-8")
        self.assertTrue(compare_metrics(reference, self.metrics(0.509), 0.01))
        self.assertTrue(compare_metrics(reference, self.metrics(0.51), 0.01))
        self.assertFalse(compare_metrics(reference, self.metrics(np.nextafter(0.51, 1)), 0.01))
        self.assertFalse(compare_metrics(reference, self.metrics(0.511), 0.01))
        self.assertTrue(compare_metrics(reference, self.metrics(), 0))
        for value in (float("nan"), float("inf"), -0.1, 1.1):
            with self.subTest(value=value), self.assertRaises(ValueError):
                compare_metrics(reference, self.metrics(value))
        for value in (-1, float("nan"), float("inf")):
            with self.subTest(tolerance=value), self.assertRaises(ValueError):
                compare_metrics(reference, self.metrics(), value)
        candidate = self.metrics(); candidate["seed"] = 43
        with self.assertRaises(ValueError):
            compare_metrics(reference, candidate)
        with self.assertRaises(FileNotFoundError):
            compare_metrics(self.work / "missing.json", self.metrics())

    def test_reproduction_failure_propagates_through_logged_command(self):
        reference, candidate = self.work / "ref.json", self.work / "candidate.json"
        reference.write_text(json.dumps(self.metrics()), encoding="utf-8")
        candidate.write_text(json.dumps(self.metrics(0.53)), encoding="utf-8")
        result = self.cli("scripts/run_logged.py", "--log-dir", self.work / "logs", "--name", "failed_reproduction",
                          "--", "python", ROOT / "scripts/verify_reproduction.py", "--reference_metrics", reference,
                          "--candidate_metrics", candidate)
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn("REPRODUCTION FAILED", result.stdout)
        receipt = json.loads((self.work / "logs/failed_reproduction.json").read_text())
        self.assertEqual(receipt["exit_code"], 1)

    @unittest.skipUnless(shutil.which("powershell"), "Windows PowerShell is not installed")
    def test_reproduction_script_requires_dataset_and_one_cache_mode(self):
        script = ROOT / 'scripts/reproduce_image.ps1'
        common = ['powershell', '-NoProfile', '-ExecutionPolicy', 'Bypass',
                  '-File', str(script), '-RunId', 'script_guard', '-PythonExe', sys.executable]
        env = dict(os.environ); env.pop('MEMOTION_DATA_DIR', None)
        result = subprocess.run(common + ['-Reextract'], env=env, cwd=ROOT,
                                capture_output=True, text=True, timeout=30)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Set MEMOTION_DATA_DIR', result.stderr)
        for args in ([], ['-Reextract', '-FeaturesDir', str(self.work)]):
            result = subprocess.run(common + args, cwd=ROOT, capture_output=True,
                                    text=True, timeout=30)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('Choose exactly one', result.stderr)

    def test_reload_comparison_detects_different_probabilities(self):
        frame = predictions_frame(self.truth.sample_id, self.truth.sarcasm_binary, np.ones(699),
                                  np.full(699, 0.7), CFG["model_name"], "validation", 42)
        a, b = self.work / "a.csv", self.work / "b.csv"
        frame.to_csv(a, index=False); frame.iloc[::-1].to_csv(b, index=False)
        self.assertTrue(compare_predictions(a, b))
        frame.loc[0, "probability_1"] = 0.71
        frame.to_csv(b, index=False)
        self.assertFalse(compare_predictions(a, b))


if __name__ == "__main__":
    unittest.main()
