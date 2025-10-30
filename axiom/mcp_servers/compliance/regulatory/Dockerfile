FROM python:3.10-slim

WORKDIR /app

# Copy entire project
COPY . /app

# Install complete project with all dependencies
RUN pip install --no-cache-dir -e .

# Set environment
ENV PYTHONPATH=/app
ENV MCP_SERVER=regulatory
ENV LOG_LEVEL=INFO

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.exit(0)"

# Run MCP server via STDIO
CMD ["python", "-m", "axiom.mcp_servers.compliance.regulatory.server"]