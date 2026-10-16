"""Output guards shared by image commands."""
import re
from pathlib import Path


def validate_run_id(run_id):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", run_id):
        raise ValueError("run_id must contain only letters, digits, underscores or hyphens")


def require_new_paths(*paths):
    for path in map(Path, paths):
        if path.exists():
            raise FileExistsError(f"Existing output: {path}. Choose a new run_id or directory.")
