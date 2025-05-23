# PaddleOCR Training Environment for Pinball Score Extraction
FROM paddlepaddle/paddle:2.5.1-gpu-cuda11.7-cudnn8

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    unzip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgtk-3-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install PaddleOCR from source for latest features
RUN git clone https://github.com/PaddlePaddle/PaddleOCR.git /tmp/PaddleOCR && \
    cd /tmp/PaddleOCR && \
    pip install -e . && \
    rm -rf /tmp/PaddleOCR

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/data/paddleocr_training/images \
             /app/data/paddleocr_training/labels \
             /app/logs \
             /app/models/checkpoints \
             /app/models/pinball_ocr

# Set environment variables
ENV PYTHONPATH=/app:$PYTHONPATH
ENV CUDA_VISIBLE_DEVICES=0

# Default command runs the training script
CMD ["python", "scripts/train_model.py", "--config", "configs/pinball_ocr_config.yml"]