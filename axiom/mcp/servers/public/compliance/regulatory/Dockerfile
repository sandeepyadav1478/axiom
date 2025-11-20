FROM python:3.10-slim

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends gcc
RUN apt-get clean

COPY axiom/mcp/servers/public/analytics/performance/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY axiom/mcp/servers/shared /app/axiom/mcp/servers/shared
COPY axiom/mcp/servers/public/compliance/regulatory /app/axiom/mcp/servers/public/compliance/regulatory
RUN echo "# Minimal" > /app/axiom/__init__.py && \
    touch /app/axiom/mcp/__init__.py /app/axiom/mcp/servers/__init__.py /app/axiom/mcp/servers/public/__init__.py /app/axiom/mcp/servers/public/compliance/__init__.py

RUN mkdir -p /app/axiom/derivatives/compliance && \
    echo "class RegulatoryReporter:\n    def __init__(self):pass" > /app/axiom/derivatives/compliance/regulatory_reporting.py && \
    touch /app/axiom/derivatives/__init__.py /app/axiom/derivatives/compliance/__init__.py

ENV PYTHONPATH=/app
CMD ["python", "-m", "axiom.mcp.servers.public.compliance.regulatory.server"]