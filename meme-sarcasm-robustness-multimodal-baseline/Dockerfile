# CPU inference image for the multimodal classifier (FastAPI on port 8000).
#
#   docker build -t meme-sarcasm .
#   docker run --rm -p 8000:8000 -v "$PWD/checkpoints:/app/checkpoints:ro" meme-sarcasm
#   # offline (no Hugging Face access): mount local copies of the pinned encoders as well
#   docker run --rm -p 8000:8000 -v "$PWD/checkpoints:/app/checkpoints:ro" \
#       -v "$PWD/hf_local:/app/hf_local:ro" -e TEXT_MODEL_DIR=hf_local/distilbert-base-uncased \
#       -e CLIP_MODEL_DIR=hf_local/clip-vit-base-patch32 meme-sarcasm
#   # other head / port: -e CHECKPOINT=checkpoints/multimodal/<run_id>/best.pt -e PORT=9000 -p 9000:9000
#
# The image holds code, configs and the synthetic samples only: no dataset, feature caches,
# checkpoints or encoder weights (team README). The checkpoint (and, offline, the encoders) are
# mounted at run time; scripts/serve.sh reads CHECKPOINT, TEXT_MODEL_DIR, CLIP_MODEL_DIR, PORT.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/hf_cache

WORKDIR /app

# CPU-only PyTorch wheels keep the image small; GPU users install torch for their CUDA instead.
RUN pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.7" "torchvision>=0.22"
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src/ src/
COPY configs/ configs/
COPY scripts/serve.sh scripts/serve.sh
COPY data/samples/ data/samples/

RUN useradd --create-home --uid 1000 app && mkdir -p /app/hf_cache && chown -R app /app
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s \
    CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:%s/health' % os.environ.get('PORT', '8000'))"
CMD ["bash", "scripts/serve.sh"]
