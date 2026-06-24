FROM python:3.10-slim

LABEL maintainer="vrutulpatel25@gmail.com"
LABEL description="EchoGuard — real-time, privacy-first deepfake and voice clone detector"
LABEL version="0.1.0"

# System dependencies required by OpenCV and MediaPipe
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (cached layer if requirements.txt unchanged)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY configs/ ./configs/
COPY scripts/ ./scripts/
COPY setup.py .

# Create directories expected at runtime
RUN mkdir -p logs models

# Default: show help
ENTRYPOINT ["python", "src/main.py"]
CMD ["--help"]
