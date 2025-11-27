FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy streaming requirements only
COPY axiom/streaming/requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create axiom package structure
RUN mkdir -p /app/axiom && touch /app/axiom/__init__.py

# Copy only streaming service code
COPY axiom/streaming/ /app/axiom/streaming/

# Expose ports
EXPOSE 8001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

# Run the streaming service
CMD ["uvicorn", "axiom.streaming.streaming_service:app", "--host", "0.0.0.0", "--port", "8001", "--workers", "1"]