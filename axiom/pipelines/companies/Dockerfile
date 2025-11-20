FROM python:3.13-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY axiom/pipelines/companies/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy shared utilities
COPY axiom/pipelines/shared /app/axiom/pipelines/shared

# Copy pipeline script
COPY axiom/pipelines/companies/company_graph_builder.py /app/pipeline.py

# Copy .env for configuration
COPY .env /app/.env

# Set environment
ENV PYTHONPATH=/app

# Health check
HEALTHCHECK --interval=60s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import sys; sys.exit(0)"

# Run the pipeline
CMD ["python", "/app/pipeline.py"]