FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/backend

WORKDIR /app

# System deps (runtime + build only where needed)
RUN apt-get update && apt-get install --no-install-recommends -y \
    # build deps (needed for some pip wheels)
    build-essential \
    gcc \
    # postgres client libs
    libpq-dev \
    libpq5 \
    # pdf + media tooling
    poppler-utils \
    ghostscript \
    qpdf \
    pngquant \
    ffmpeg \
    # misc
    curl \
    ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# Install python requirements first (cache-friendly)
COPY backend/requirements.txt /tmp/requirements.txt

RUN pip install --upgrade pip \
 && pip install --no-cache-dir -r /tmp/requirements.txt \
 # cleanup build tools to reduce image size
 && apt-get purge -y --auto-remove build-essential gcc curl \
 && rm -rf /tmp/requirements.txt

# Copy backend source
COPY backend ./backend

# (Optional but recommended) non-root user
RUN useradd -m appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# IMPORTANT:
# - Keep workers=2 for c7i-flex.large (2 vCPU). Increase only if CPU allows.
# - timeout=600 for long tasks, but with job-based OCR you can reduce later.
CMD ["gunicorn","app.main:app","-k","uvicorn.workers.UvicornWorker","--workers","2","--timeout","600","--bind","0.0.0.0:8000","--access-logfile","-","--error-logfile","-"]
