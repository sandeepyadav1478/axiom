# Dockerfile for Axiom Derivatives Platform
# Multi-stage build for optimal image size and performance

# =============================================================================
# Stage 1: Build stage with all dependencies
# =============================================================================
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 as builder

# Install Python and build tools
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3-pip \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Set working directory
WORKDIR /build

# Copy requirements
COPY requirements-derivatives.txt .

# Install PyTorch with CUDA support
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install all dependencies
RUN pip3 install -r requirements-derivatives.txt

# =============================================================================
# Stage 2: Runtime stage (smaller, production-optimized)
# =============================================================================
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Install Python runtime only
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN useradd -m -u 1000 axiom && \
    mkdir -p /app && \
    chown -R axiom:axiom /app

# Set working directory
WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/dist-packages /usr/local/lib/python3.11/dist-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY --chown=axiom:axiom axiom/ ./axiom/
COPY --chown=axiom:axiom demos/ ./demos/
COPY --chown=axiom:axiom requirements-derivatives.txt .

# Switch to non-root user
USER axiom

# Environment variables
ENV PYTHONPATH=/app
ENV CUDA_VISIBLE_DEVICES=0
ENV OMP_NUM_THREADS=4
ENV MKL_NUM_THREADS=4

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=40s --retries=3 \
    CMD python3 -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

# Run FastAPI server
CMD ["python3", "-m", "uvicorn", "axiom.derivatives.api.endpoints:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

# For production: Use gunicorn
# CMD ["gunicorn", "axiom.derivatives.api.endpoints:app", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "-b", "0.0.0.0:8000"]

# Build:
# docker build -t axiom/derivatives:latest -f axiom/derivatives/docker/Dockerfile .

# Run:
# docker run --gpus all -p 8000:8000 axiom/derivatives:latest

# Test:
# curl http://localhost:8000/health
# curl -X POST http://localhost:8000/greeks -H "Content-Type: application/json" -d '{"spot": 100, "strike": 100, "time_to_maturity": 1.0, "risk_free_rate": 0.03, "volatility": 0.25}'