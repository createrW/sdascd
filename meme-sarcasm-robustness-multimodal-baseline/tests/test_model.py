"""Model definitions, the shared trainer and checkpoint evaluation (no encoder download).

Run from the repository root with MEMOTION_DATA_DIR set:
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

import torch
import torch.nn.functional as F
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from helpers import (IMAGE_CFG, MM_CFG, ROOT, work_dir,  # noqa: E402
                     write_train_validation_caches)
from src.evaluation.evaluate import detect_task  # noqa: E402
from src.features.feature_engineering import pool_text_features  # noqa: E402
from src.features.cache import file_sha256  # noqa: E402
from src.models.base_model import FeatureHead, build_mlp  # noqa: E402
from src.models.multimodal_classifier import (FusionHead, build_fusion_head,  # noqa: E402
                                              build_fusion_mlp, load_fusion_head)
from src.models.trainer import build_head, predict  # noqa: E402
from src.models.vit_classifier import ImageHead, build_image_head  # noqa: E402
from src.utils.config import load_config  # noqa: E402

SHIPPED_IMAGE_HEADS = ("20261001_clip_vitb32_seed42", "20261001_clip_vitb32_seed42_zhihan_rtx3080",
                       "20261006_code_fixes_validation_v2")


class HeadTests(unittest.TestCase):
    def test_fusion_head_normalizes_then_concatenates_text_first(self):
        head = build_fusion_head(MM_CFG)
        self.assertEqual(head.input_dim, 1536)
        text, image = torch.randn(5, 768) * 7, torch.randn(5, 768) * 0.1
        fused = head.fuse(text, image)
        self.assertTrue(torch.allclose(fused, torch.cat([F.normalize(text, dim=-1),
                                                         F.normalize(image, dim=-1)], dim=-1)))
        unimodal = FusionHead(modalities=["image"])
        self.assertEqual(unimodal.input_dim, 768)
        with self.assertRaises(ValueError):
            unimodal(text=text)
        wrong = copy.deepcopy(MM_CFG)
        wrong["modalities"] = ["text"]
        with self.assertRaisesRegex(ValueError, "input_dim"):
            build_fusion_head(wrong)

    def test_head_checkpoint_loads_into_end_to_end_classifier(self):
        torch.manual_seed(0)
        head = build_fusion_head(MM_CFG).eval()
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

    def test_image_head_shapes_and_keyword_input(self):
        linear = build_image_head(IMAGE_CFG)
        self.assertEqual(set(linear.state_dict()), {"net.1.weight", "net.1.bias"})
        mlp = ImageHead(hidden_dim=16, dropout=0.1)
        self.assertEqual(set(mlp.state_dict()),
                         {"net.0.weight", "net.0.bias", "net.3.weight", "net.3.bias"})
        x = torch.randn(3, 768)
        linear.eval()
        self.assertTrue(torch.equal(linear(x), linear(image=x)))
        self.assertEqual(tuple(mlp(image=x).shape), (3, 2))
        for head in (linear, mlp, build_fusion_head(MM_CFG)):
            self.assertIsInstance(head, FeatureHead)
        self.assertEqual(linear.inputs({"image": x, "text": x}).keys(), {"image"})
        with self.assertRaises(ValueError):
            linear.inputs({"text": x})
        self.assertEqual(tuple(build_mlp(768, 8)(x).shape), (3, 2))

    def test_shipped_image_heads_load_into_the_new_head(self):
        for run in SHIPPED_IMAGE_HEADS:
            with self.subTest(run=run):
                ckpt = torch.load(ROOT / f"checkpoints/image/{run}/best.pt", map_location="cpu",
                                  weights_only=True)
                self.assertNotIn("task", ckpt["config"])  # recorded before the restructure
                head = build_head(ckpt["config"], {"image": torch.zeros(1, 768)})
                head.load_state_dict(ckpt["state_dict"])   # strict: keys unchanged
                prob = predict(head, {"image": torch.randn(4, 768)}, torch.device("cpu"))
                self.assertTrue(((prob >= 0) & (prob <= 1)).all())

    def test_every_training_config_builds_its_head(self):
        for path in sorted((ROOT / "configs/train").glob("*.yaml")):
            with self.subTest(config=path.name):
                cfg = load_config(path)
                head = build_head(cfg, {m: torch.zeros(1, 768) for m in ("text", "image")})
                self.assertEqual(head.modalities, tuple(cfg.get("modalities", ("image",))))

    def test_epoch_callback_scoring_does_not_change_training(self):
        """The overfitting diagnostic scores the head every epoch; training must be unaffected."""
        from src.evaluation.overfitting_diagnostic import split_scores, train_mode_loss
        from src.models.trainer import fit
        g = torch.Generator().manual_seed(0)
        feats = {m: torch.randn(200, 768, generator=g) for m in ("text", "image")}
        y = torch.randint(0, 2, (200,), generator=g)
        weights = torch.tensor([1.5, 0.75])
        cfg = dict(MM_CFG, epochs=3, batch_size=32, fusion=dict(MM_CFG["fusion"], hidden_dim=0, dropout=0.3))
        results = []
        with tempfile.TemporaryDirectory() as tmp:
            for callback in (None, lambda head, epoch: (split_scores(head, feats, y, weights, 0.5),
                                                        train_mode_loss(head, feats, y, weights))):
                torch.manual_seed(1)
                head = build_head(cfg, feats)
                log, _, _, _ = fit(head, feats, y, feats, y, cfg, torch.device("cpu"),
                                   Path(tmp) / "best.pt", epoch_callback=callback)
                results.append((log, {k: v.clone() for k, v in head.state_dict().items()}))
        (log_a, state_a), (log_b, state_b) = results
        self.assertEqual(log_a, log_b)
        self.assertTrue(all(torch.equal(state_a[k], state_b[k]) for k in state_a))
        self.assertEqual(set(state_a), {"classifier.1.weight", "classifier.1.bias"})  # linear head

    def test_task_detection_from_the_checkpoint(self):
        shipped = ROOT / f"checkpoints/image/{SHIPPED_IMAGE_HEADS[-1]}/best.pt"
        self.assertEqual(detect_task("any", shipped), "image")       # legacy config, no task key
        with tempfile.TemporaryDirectory() as tmp:
            # a multimodal head stored under a folder named image/ is still a multimodal head
            path = Path(tmp) / "image" / "best.pt"
            path.parent.mkdir()
            torch.save({"config": dict(MM_CFG), "state_dict": {}}, path)
            self.assertEqual(detect_task("any", path), "multimodal")
        with self.assertRaises(SystemExit):
            detect_task("no_such_run_anywhere")


class TrainEvaluateTests(unittest.TestCase):
    """Train -> verify -> reload on synthetic caches, through the real command-line entry points."""

    def setUp(self):
        self.work = work_dir(self, "train_evaluate_")

    def run_cli(self, *args):
        return subprocess.run([sys.executable, *map(str, args)], cwd=self.work,
                              capture_output=True, text=True, encoding="utf-8",
                              errors="replace", timeout=600)

    def round_trip(self, task, cfg, train_args, eval_args=()):
        (self.work / "cfg.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
        train = self.run_cli(ROOT / "src/models/trainer.py", "--config", "cfg.yaml", *train_args,
                             "--run_id", "synthetic")
        self.assertEqual(train.returncode, 0, train.stdout + train.stderr)
        run = self.work / f"outputs/{task}/synthetic"
        for name in ("config.json", "metrics_validation.json", "predictions_validation.csv",
                     "confusion_matrix_validation.png", "training_log.csv", "README.md"):
            self.assertTrue((run / name).is_file(), name)
        record = json.loads((run / "config.json").read_text(encoding="utf-8"))
        self.assertEqual(record["task"], task)
        self.assertTrue(record["command"].startswith("python -m src.models.trainer"))
        ckpt = torch.load(self.work / f"checkpoints/{task}/synthetic/best.pt", weights_only=True)
        self.assertEqual(ckpt["epoch"], record["best_epoch"])
        verify = self.run_cli(ROOT / "scripts/verify_predictions.py", run / "predictions_validation.csv")
        self.assertEqual(verify.returncode, 0, verify.stdout + verify.stderr)
        reload = self.run_cli(ROOT / "src/evaluation/evaluate.py", "--run_id", "synthetic", *eval_args)
        self.assertEqual(reload.returncode, 0, reload.stdout + reload.stderr)
        compare = self.run_cli(ROOT / "scripts/verify_reproduction.py",
                               "--reference_metrics", run / "metrics_validation.json",
                               "--candidate_metrics", run / "reeval/metrics_validation.json",
                               "--tolerance", "0",
                               "--reference_predictions", run / "predictions_validation.csv",
                               "--candidate_predictions", run / "reeval/predictions_validation.csv")
        self.assertEqual(compare.returncode, 0, compare.stdout + compare.stderr)
        again = self.run_cli(ROOT / "src/models/trainer.py", "--config", "cfg.yaml", *train_args,
                             "--run_id", "synthetic")
        self.assertNotEqual(again.returncode, 0)
        self.assertIn("Existing output", again.stderr)
        return run

    def test_multimodal_round_trip_on_synthetic_caches(self):
        write_train_validation_caches(self.work)
        cfg = dict(MM_CFG, epochs=2, batch_size=256, features={"text": "text", "image": "image"})
        run = self.round_trip("multimodal", cfg, ())
        record = json.loads((run / "config.json").read_text(encoding="utf-8"))
        self.assertEqual((record["text_features_dir"], record["image_features_dir"]), ("text", "image"))

    def test_image_round_trip_on_synthetic_caches(self):
        # cache left at the old features/<name> location, requested by its new data/processed path
        write_train_validation_caches(self.work / "features", modalities=("image",), cfg=IMAGE_CFG)
        cfg = dict(IMAGE_CFG, epochs=2, batch_size=256)
        run = self.round_trip("image", cfg, ("--features_dir", "data/processed/features/image"))
        record = json.loads((run / "config.json").read_text(encoding="utf-8"))
        self.assertEqual(record["features_dir"], "features/image")   # the folder actually used
        self.assertEqual(record["feature_meta"]["clip_model_id"], IMAGE_CFG["clip_model_id"])
        # evaluate found the cache through config.json (no --features_dir given) and checks the task
        wrong = self.run_cli(ROOT / "src/evaluation/evaluate.py", "--run_id", "synthetic",
                             "--task", "multimodal", "--checkpoint",
                             "checkpoints/image/synthetic/best.pt", "--out_dir", "wrong")
        self.assertNotEqual(wrong.returncode, 0)
        self.assertIn("checkpoint task is 'image'", wrong.stderr)

    def test_error_analysis_needs_an_existing_run(self):
        result = self.run_cli(ROOT / "src/evaluation/error_analysis.py", "--run_id", "missing_run")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("no unique outputs", result.stderr)
        self.assertFalse((self.work / "outputs").exists())

    def test_existing_output_and_checkpoint_are_protected(self):
        for parent in ("outputs/image", "checkpoints/image"):
            with self.subTest(parent=parent):
                run = "existing_" + parent.split("/")[0]
                directory = self.work / parent / run
                directory.mkdir(parents=True)
                marker = directory / "best.pt"
                marker.write_bytes(b"previous result")
                before = file_sha256(marker)
                result = self.run_cli(ROOT / "src/models/trainer.py", "--config",
                                      ROOT / "configs/train/image_baseline.yaml",
                                      "--run_id", run, "--features_dir", "nonexistent_cache")
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("Existing output", result.stderr)
                self.assertEqual(before, file_sha256(marker))
                other = "checkpoints/image" if parent == "outputs/image" else "outputs/image"
                self.assertFalse((self.work / other / run).exists())


if __name__ == "__main__":
    unittest.main()
