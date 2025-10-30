FROM python:3.10-slim

WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir -e .

ENV PYTHONPATH=/app
ENV MCP_SERVER=interface
ENV LOG_LEVEL=INFO

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.exit(0)"

CMD ["python", "-m", "axiom.mcp_servers.client.interface.server"]