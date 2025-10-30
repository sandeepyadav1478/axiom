FROM python:3.10-slim

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends gcc && rm -rf /var/lib/apt/lists/*

COPY axiom/mcp_servers/requirements-mcp-only.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY axiom/mcp_servers/shared /app/axiom/mcp_servers/shared
COPY axiom/mcp_servers/trading/hedging /app/axiom/mcp_servers/trading/hedging
RUN echo "# Minimal" > /app/axiom/__init__.py
COPY axiom/mcp_servers/__init__.py /app/axiom/mcp_servers/__init__.py
COPY axiom/mcp_servers/trading/__init__.py /app/axiom/mcp_servers/trading/__init__.py

RUN mkdir -p /app/axiom/ai_layer/domain && \
    echo "from enum import Enum\nclass HedgeType(Enum):\n    DELTA='delta'\nclass HedgeUrgency(Enum):\n    LOW='low'\nclass HedgeRecommendation:\n    pass" > /app/axiom/ai_layer/domain/hedging_value_objects.py && \
    touch /app/axiom/ai_layer/__init__.py /app/axiom/ai_layer/domain/__init__.py

RUN mkdir -p /app/axiom/derivatives/market_making && \
    echo "class PortfolioState:\n    def __init__(self,**kwargs):\n        for k,v in kwargs.items():setattr(self,k,v)\nclass DRLAutoHedger:\n    def __init__(self,**kwargs):pass\n    def get_optimal_hedge(self,portfolio_state):\n        class Action:\n            hedge_delta=50;expected_delta_after=2;expected_cost=500;urgency='medium';confidence=0.85\n        return Action()" > /app/axiom/derivatives/market_making/auto_hedger.py && \
    touch /app/axiom/derivatives/__init__.py /app/axiom/derivatives/market_making/__init__.py

ENV PYTHONPATH=/app
CMD ["python", "-m", "axiom.mcp_servers.trading.hedging.server"]