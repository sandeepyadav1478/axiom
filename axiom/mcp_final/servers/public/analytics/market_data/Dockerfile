FROM python:3.10-slim

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends gcc
RUN apt-get clean

COPY axiom/mcp_servers/requirements-mcp-only.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY axiom/mcp_final/servers/shared /app/axiom/mcp_final/servers/shared
COPY axiom/mcp_final/servers/public/analytics/market_data /app/axiom/mcp_final/servers/analytics/market_data
RUN echo "# Minimal" > /app/axiom/__init__.py
COPY axiom/mcp_final/servers/__init__.py /app/axiom/mcp_final/servers/__init__.py
RUN mkdir -p /app/axiom/mcp_final/servers/analytics && touch /app/axiom/mcp_final/servers/analytics/__init__.py

RUN mkdir -p /app/axiom/derivatives/mcp && \
    echo "class MarketDataAggregator:\n    def __init__(self):pass" > /app/axiom/derivatives/mcp/market_data_integrations.py && \
    touch /app/axiom/derivatives/__init__.py /app/axiom/derivatives/mcp/__init__.py

ENV PYTHONPATH=/app
CMD ["python", "-m", "axiom.mcp_final.servers.analytics.market_data.server"]