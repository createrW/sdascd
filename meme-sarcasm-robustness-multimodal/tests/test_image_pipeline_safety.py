"""Regression tests for image output, cache, reproduction and probability safety.

Run from the repository root with MEMOTION_DATA_DIR pointing at the fixed dataset:
    python -m unittest discover -s tests -v
Fixtures stay in REVIEW_TEST_DIR when set; otherwise temporary fixtures are removed.
"""
import copy
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from src.common.data import load_split
from src.common.metrics import predictions_frame
from src.common.reproduction import compare_metrics, compare_predictions
from src.image.cache import file_sha256, load_features
from src.image.safety import require_new_paths, validate_run_id

ROOT = Path(__file__).resolve().parents[1]
CFG = json.loads((ROOT / "configs/image_baseline.json").read_text(encoding="utf-8"))


class PipelineSafetyTests(unittest.TestCase):
    def setUp(self):
        base = os.environ.get("REVIEW_TEST_DIR")
        if base:
            base = Path(base).resolve()
            base.mkdir(parents=True, exist_ok=True)
            self.work = Path(tempfile.mkdtemp(prefix=self._testMethodName + "_", dir=base))
        else:
            self.temp = tempfile.TemporaryDirectory(prefix="image_pipeline_safety_")
            self.work = Path(self.temp.name).resolve()
            self.addCleanup(self.temp.cleanup)
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

    def test_existing_output_and_checkpoint_are_protected(self):
        for parent in ("outputs/image", "checkpoints/image"):
            with self.subTest(parent=parent):
                run = "existing_" + parent.split("/")[0]
                directory = self.work / parent / run
                directory.mkdir(parents=True)
                marker = directory / "best.pt"
                marker.write_bytes(b"previous result")
                before = file_sha256(marker)
                result = self.cli("src/image/train.py", "--config", ROOT / "configs/image_baseline.json",
                                  "--run_id", run, "--features_dir", "nonexistent_cache")
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("Existing output", result.stderr)
                self.assertEqual(before, file_sha256(marker))
                other = "checkpoints/image" if parent == "outputs/image" else "outputs/image"
                self.assertFalse((self.work / other / run).exists())

    def test_duplicate_and_invalid_extraction_arguments_are_rejected(self):
        from src.image import extract_features
        for arguments in (["--limit", "0"], ["--limit", "-1"], ["--batch_size", "0"],
                          ["--splits", "train", "train"], ["--splits", "unknown"]):
            with self.subTest(arguments=arguments), \
                    patch.object(sys, "argv", ["extract", *arguments]), \
                    patch.object(extract_features, "load_clip") as loader, \
                    self.assertRaises(SystemExit) as error:
                extract_features.main()
            self.assertEqual(error.exception.code, 2)
            loader.assert_not_called()

    def test_path_traversal_run_ids_are_rejected(self):
        for name in ("../old", "a/b", "a\\b", "", ".."):
            with self.subTest(name=name), self.assertRaises(ValueError):
                validate_run_id(name)

    def test_existing_cache_rejected_before_loading_clip(self):
        from src.image import extract_features
        cache = self.work / "formal_cache"
        cache.mkdir()
        marker = cache / "train.pt"
        marker.write_bytes(b"full cache")
        with patch.object(sys, "argv", ["extract", "--out_dir", str(cache), "--limit", "2"]), \
                patch.object(extract_features, "load_clip") as loader, \
                self.assertRaises(FileExistsError):
            extract_features.main()
        loader.assert_not_called()
        self.assertEqual(marker.read_bytes(), b"full cache")

    def test_limited_default_cache_is_separate(self):
        from src.image import extract_features
        original_cwd = Path.cwd()
        try:
            os.chdir(self.work)
            formal = self.work / "features/clip_vitb32_pooler"
            formal.mkdir(parents=True)
            (formal / "train.pt").write_bytes(b"full cache")
            result = dict(sample_id=self.truth.sample_id.tolist()[:2], features=torch.zeros(2, 768),
                          labels=torch.tensor(self.truth.sarcasm_binary.to_numpy()[:2], dtype=torch.long))
            with patch.object(sys, "argv", ["extract", "--config", str(ROOT / "configs/image_baseline.json"),
                                           "--limit", "2", "--splits", "validation"]), \
                    patch.object(extract_features, "load_clip", return_value=(SimpleNamespace(config=SimpleNamespace()), None)), \
                    patch.object(extract_features, "extract_split", return_value=result), \
                    patch.object(extract_features, "get_device", return_value=torch.device("cpu")):
                extract_features.main()
            smoke = self.work / "features/clip_vitb32_pooler_smoke_2"
            meta = json.loads((smoke / "meta.json").read_text())
            self.assertEqual(meta["limit"], 2)
            self.assertEqual(meta["splits"]["validation"]["sha256"], file_sha256(smoke / "validation.pt"))
            self.assertEqual((formal / "train.pt").read_bytes(), b"full cache")
            with self.assertRaisesRegex(ValueError, "Smoke/limited"):
                load_features(smoke, "validation", CFG)
        finally:
            os.chdir(original_cwd)

    def test_complete_cache_is_accepted(self):
        directory, _, _ = self.cache()
        ids, features, labels = load_features(directory, "validation", CFG)
        self.assertEqual(ids, self.truth.sample_id.tolist())
        self.assertEqual(tuple(features.shape), (699, 768))
        self.assertTrue(torch.equal(labels, torch.tensor(self.truth.sarcasm_binary.to_numpy())))

    def test_cache_missing_metadata_is_rejected(self):
        directory, _, _ = self.cache()
        (directory / "meta.json").unlink()
        with self.assertRaises(FileNotFoundError):
            load_features(directory, "validation", CFG)

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
                    load_features(directory, "validation", CFG)
        meta["splits"]["validation"]["n"] = 128
        self.write_cache(directory, data, meta)
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            load_features(directory, "validation", CFG)

    def test_incomplete_rows_rejected_even_if_meta_claims_complete(self):
        directory, data, meta = self.cache()
        for key in data:
            data[key] = data[key][:128]
        self.write_cache(directory, data, meta)
        with self.assertRaises(ValueError):
            load_features(directory, "validation", CFG)

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
                    load_features(directory, "validation", CFG)

    def test_new_cache_hash_detects_changed_feature_values(self):
        directory, data, meta = self.cache()
        meta["splits"]["validation"]["sha256"] = file_sha256(directory / "validation.pt")
        data["features"][0, 0] = 10
        self.write_cache(directory, data, meta)
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            load_features(directory, "validation", CFG)

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
