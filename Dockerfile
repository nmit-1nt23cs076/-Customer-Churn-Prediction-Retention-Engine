# Multi-stage production container for Customer Churn Engine
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000

# Install system dependencies (libgomp1 is required for LightGBM OpenMP parallelization)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first to leverage Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY . .

# Generate dataset and train the model inside the build stage so container boots ready
RUN python data/generate_dataset.py && \
    python src/train.py

# Expose port (Render overrides with dynamic $PORT)
EXPOSE 8000

# Healthcheck probe
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8000}/health || exit 1

# Start FastAPI application
CMD sh -c "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}"
