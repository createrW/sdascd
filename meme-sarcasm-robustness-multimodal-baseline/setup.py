"""Optional packaging: pip install -e .  (adds the `src` package and console commands).

    pip install -e ".[serving,dev]"
    meme-train --config configs/train/multimodal_baseline.yaml
    meme-serve --serving_config configs/deploy/serving.yaml

Commands run from the repository root, like the python -m forms in README.md, because configs,
data/, outputs/ and checkpoints/ are repository-relative.
"""
from setuptools import find_packages, setup

setup(
    name="meme-sarcasm-robustness",
    version="0.2.0",
    description="Sarcasm detection on Memotion memes: frozen CLIP / DistilBERT baselines and a "
                "multimodal fusion head",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    python_requires=">=3.11",   # hashlib.file_digest; tested on 3.12
    packages=find_packages(include=["src", "src.*"]),
    install_requires=[
        "torch>=2.7", "torchvision>=0.22", "transformers>=5.0", "pandas>=2.2", "numpy",
        "scikit-learn>=1.5", "scipy>=1.11", "matplotlib>=3.8", "pillow>=10", "tqdm",
        "pyyaml>=6.0",
    ],
    extras_require={
        "serving": ["fastapi>=0.110", "uvicorn>=0.29"],
        "dev": ["httpx>=0.27", "nbclient>=0.10", "ipykernel>=6.29", "jupyterlab>=4.0"],
    },
    entry_points={
        "console_scripts": [
            "meme-extract=src.features.feature_engineering:main",
            "meme-train=src.models.trainer:main",
            "meme-eval=src.evaluation.evaluate:main",
            "meme-serve=src.serving.app:main",
        ],
    },
)
