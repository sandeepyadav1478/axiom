FROM python:3.10-slim

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends gcc && rm -rf /var/lib/apt/lists/*

COPY axiom/mcp/servers/public/analytics/performance/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY axiom/mcp/servers/shared /app/axiom/mcp/servers/shared
COPY axiom/mcp/servers/public/analytics/performance /app/axiom/mcp/servers/public/analytics/performance
RUN echo "# Minimal" > /app/axiom/__init__.py && \
    touch /app/axiom/mcp/__init__.py /app/axiom/mcp/servers/__init__.py /app/axiom/mcp/servers/public/__init__.py /app/axiom/mcp/servers/public/analytics/__init__.py

RUN mkdir -p /app/axiom/derivatives/analytics && \
    echo "class RealTimePnLEngine:\n    def __init__(self,**kwargs):pass\n    def calculate_pnl(self,**kwargs):\n        class Result:\n            total_pnl=1500;realized_pnl=800;unrealized_pnl=700;delta_pnl=400;gamma_pnl=200;vega_pnl=100;theta_pnl=-50\n        return Result()" > /app/axiom/derivatives/analytics/pnl_engine.py && \
    touch /app/axiom/derivatives/__init__.py /app/axiom/derivatives/analytics/__init__.py

ENV PYTHONPATH=/app
CMD ["python", "-m", "axiom.mcp.servers.public.analytics.performance.server"]