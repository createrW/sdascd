"""Read run configurations (configs/train/*.yaml, configs/deploy/*.yaml; JSON also accepted)."""
import json
from pathlib import Path

import yaml


def load_config(path) -> dict:
    """Return the configuration file at `path` as a dict (.yaml/.yml or .json)."""
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in (".yaml", ".yml"):
        cfg = yaml.safe_load(text)
    elif path.suffix.lower() == ".json":
        cfg = json.loads(text)
    else:
        raise ValueError(f"{path}: expected a .yaml, .yml or .json configuration file")
    if not isinstance(cfg, dict):
        raise ValueError(f"{path}: the configuration must be a mapping")
    return cfg


def infer_task(cfg: dict) -> str:
    """'image' or 'multimodal'; older configs (and checkpoints) have no explicit task key."""
    task = cfg.get("task") or ("multimodal" if "fusion" in cfg else "image")
    if task not in ("image", "multimodal"):
        raise ValueError(f"unknown task {task!r}")
    return task
