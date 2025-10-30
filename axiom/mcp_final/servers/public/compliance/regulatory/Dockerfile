FROM python:3.10-slim

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends gcc
RUN apt-get clean

COPY axiom/mcp_final/servers/requirements-mcp-only.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY axiom/mcp_final/servers/shared /app/axiom/mcp_final/servers/shared
COPY axiom/mcp_final/servers/public/compliance/regulatory /app/axiom/mcp_final/servers/compliance/regulatory
RUN echo "# Minimal" > /app/axiom/__init__.py
COPY axiom/mcp_final/servers/__init__.py /app/axiom/mcp_final/servers/__init__.py
RUN mkdir -p /app/axiom/mcp_final/servers/compliance && touch /app/axiom/mcp_final/servers/compliance/__init__.py

RUN mkdir -p /app/axiom/derivatives/compliance && \
    echo "class RegulatoryReporter:\n    def __init__(self):pass" > /app/axiom/derivatives/compliance/regulatory_reporting.py && \
    touch /app/axiom/derivatives/__init__.py /app/axiom/derivatives/compliance/__init__.py

ENV PYTHONPATH=/app
CMD ["python", "-m", "axiom.mcp_final.servers.compliance.regulatory.server"]