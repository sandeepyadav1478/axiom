FROM python:3.10-slim

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends gcc
RUN apt-get clean

# Use shared lightweight requirements
COPY axiom/mcp/servers/public/analytics/performance/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY axiom/mcp/servers/shared /app/axiom/mcp/servers/shared
COPY axiom/mcp/servers/public/analytics/market_data /app/axiom/mcp/servers/public/analytics/market_data
RUN echo "# Minimal" > /app/axiom/__init__.py && \
    touch /app/axiom/mcp/__init__.py /app/axiom/mcp/servers/__init__.py /app/axiom/mcp/servers/public/__init__.py /app/axiom/mcp/servers/public/analytics/__init__.py

RUN mkdir -p /app/axiom/mcp_clients && \
    echo "class MarketDataAggregator:\n    def __init__(self):pass" > /app/axiom/mcp_clients/market_data_integrations.py && \
    touch /app/axiom/mcp_clients/__init__.py

ENV PYTHONPATH=/app
CMD ["python", "-m", "axiom.mcp.servers.public.analytics.market_data.server"]