FROM python:3.10-slim

# System deps needed by torchvision/Pillow
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libsm6 libxext6 libxrender1 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /code

# Install Python deps first for better layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY app ./app
# optional checkpoint, app falls back to a pretrained model if it's not there
COPY artifacts ./artifacts

# HF Spaces (Docker SDK) expects the app on port 7860
ENV PORT=7860 \
    CAPTION_MODEL_MODE=auto \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/code/.cache/huggingface

# non-root user, some hosts (HF Spaces) require this
RUN useradd -m appuser && chown -R appuser /code
USER appuser

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
    CMD curl -f http://localhost:7860/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
