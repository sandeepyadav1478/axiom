FROM python:3.10-slim

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends gcc
RUN apt-get clean

COPY axiom/mcp_servers/requirements-mcp-only.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY axiom/mcp_servers/shared /app/axiom/mcp_servers/shared
COPY axiom/mcp_servers/client/interface /app/axiom/mcp_servers/client/interface
RUN echo "# Minimal" > /app/axiom/__init__.py
COPY axiom/mcp_servers/__init__.py /app/axiom/mcp_servers/__init__.py
RUN mkdir -p /app/axiom/mcp_servers/client && touch /app/axiom/mcp_servers/client/__init__.py

ENV PYTHONPATH=/app
CMD ["python", "-m", "axiom.mcp_servers.client.interface.server"]