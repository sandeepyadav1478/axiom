FROM python:3.10-slim

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends gcc && rm -rf /var/lib/apt/lists/*

COPY axiom/mcp_servers/requirements-mcp-only.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY axiom/mcp_final/servers/shared /app/axiom/mcp_final/servers/shared
COPY axiom/mcp_final/servers/public/trading/execution /app/axiom/mcp_final/servers/trading/execution
RUN echo "# Minimal" > /app/axiom/__init__.py
COPY axiom/mcp_final/servers/__init__.py /app/axiom/mcp_final/servers/__init__.py
COPY axiom/mcp_final/servers/public/trading/__init__.py /app/axiom/mcp_final/servers/trading/__init__.py

RUN mkdir -p /app/axiom/ai_layer/domain && \
    echo "from enum import Enum\nclass OrderSide(Enum):\n    BUY='buy'\n    SELL='sell'\nclass OrderType(Enum):\n    MARKET='market'\n    LIMIT='limit'\nclass Venue(Enum):\n    CBOE='CBOE'\n    ISE='ISE'\nclass Order:\n    pass\nclass VenueQuote:\n    pass\nclass RoutingDecision:\n    pass\nclass ExecutionReport:\n    pass" > /app/axiom/ai_layer/domain/execution_value_objects.py && \
    touch /app/axiom/ai_layer/__init__.py /app/axiom/ai_layer/domain/__init__.py

RUN mkdir -p /app/axiom/derivatives/execution && \
    echo "from enum import Enum\nfrom datetime import datetime\nclass Venue(Enum):\n    CBOE='CBOE'\nclass VenueQuote:\n    def __init__(self,**kwargs):pass\nclass SmartOrderRouter:\n    def __init__(self):pass\n    def route_order(self,**kwargs):\n        class Decision:\n            primary_venue=Venue.CBOE;backup_venues=[];expected_fill_price=5.50;expected_fill_probability=0.95;expected_slippage_bps=1.2;rationale='Best price';routing_time_ms=0.8\n        return Decision()" > /app/axiom/derivatives/execution/smart_order_router.py && \
    touch /app/axiom/derivatives/__init__.py /app/axiom/derivatives/execution/__init__.py

ENV PYTHONPATH=/app
CMD ["python", "-m", "axiom.mcp_final.servers.trading.execution.server"]