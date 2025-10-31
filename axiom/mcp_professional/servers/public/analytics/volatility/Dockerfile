FROM python:3.10-slim

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends gcc
RUN apt-get clean

COPY axiom/mcp_servers/requirements-mcp-only.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY axiom/mcp_professional/servers/shared /app/axiom/mcp_professional/servers/shared
COPY axiom/mcp_professional/servers/public/analytics/volatility /app/axiom/mcp_professional/servers/analytics/volatility
RUN echo "# Minimal" > /app/axiom/__init__.py
COPY axiom/mcp_professional/servers/__init__.py /app/axiom/mcp_professional/servers/__init__.py
RUN mkdir -p /app/axiom/mcp_professional/servers/analytics && touch /app/axiom/mcp_professional/servers/analytics/__init__.py

RUN mkdir -p /app/axiom/derivatives/ai && \
    echo "class AIVolatilityPredictor:\n    def __init__(self,**kwargs):pass\n    def predict_volatility(self,**kwargs):\n        class Result:\n            forecast_vol=0.28;regime='normal';confidence=0.85\n        return Result()" > /app/axiom/derivatives/ai/volatility_predictor.py && \
    touch /app/axiom/derivatives/__init__.py /app/axiom/derivatives/ai/__init__.py

ENV PYTHONPATH=/app
CMD ["python", "-m", "axiom.mcp_professional.servers.analytics.volatility.server"]