FROM python:3.11-slim

# System deps for audio + parselmouth + models
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    portaudio19-dev \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy download script and pre-download models
# This ensures models are baked into the image
COPY download_models.py .
RUN python download_models.py

# Copy project
COPY . .

# Create required dirs
RUN mkdir -p outputs/tts_cache history/database history/records uploads cache

# Expose FastAPI port
EXPOSE 8000

ENV PORT=8000
ENV PYTHONUNBUFFERED=1

# Run with uvicorn
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
