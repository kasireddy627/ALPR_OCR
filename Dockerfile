# ──────────────────────────────────────────────
#  ALPR System – Cross-Platform Dockerfile
#  Base: python:3.11-slim (Debian Bookworm)
# ──────────────────────────────────────────────
FROM python:3.11-slim

# ── System metadata ────────────────────────────
LABEL maintainer="Kasi Reddy"
LABEL description="Automatic License Plate Recognition System"
LABEL version="1.0"

# ── Build args (override at build time if needed)
ARG DEBIAN_FRONTEND=noninteractive

# ── System dependencies ────────────────────────
#    - tesseract-ocr      : OCR engine
#    - libgl1             : OpenCV headless still needs this on some distros
#    - libglib2.0-0       : glib runtime for OpenCV
#    - libsm6 libxrender1 : X11 stubs (headless safety)
#    - curl               : healthcheck / debug
RUN apt-get update && apt-get install -y --no-install-recommends \
        tesseract-ocr \
        tesseract-ocr-eng \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxrender1 \
        libxext6 \
        curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ──────────────────────────
WORKDIR /app

# ── Python dependencies ────────────────────────
#    Copy requirements first to leverage Docker layer cache
COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# ── Copy project files ─────────────────────────
COPY . .

# ── Create runtime directories ─────────────────
RUN mkdir -p models data/uploads data/db

# ── Environment variables ──────────────────────
#    Tesseract path is auto-detected on Linux (/usr/bin/tesseract)
#    Override TESSDATA_PREFIX if you mount custom traineddata
ENV TESSERACT_CMD=/usr/bin/tesseract \
    TESSDATA_PREFIX=/usr/share/tesseract-ocr/5/tessdata \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# ── Expose Streamlit port ──────────────────────
EXPOSE 8501

# ── Healthcheck ────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# ── Entrypoint ─────────────────────────────────
CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
