FROM python:3.10-slim

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends gcc && rm -rf /var/lib/apt/lists/*

COPY axiom/mcp_servers/requirements-mcp-only.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY axiom/mcp_professional/servers/shared /app/axiom/mcp_professional/servers/shared
COPY axiom/mcp_professional/servers/public/trading/strategy_gen /app/axiom/mcp_professional/servers/trading/strategy_gen
RUN echo "# Minimal" > /app/axiom/__init__.py
COPY axiom/mcp_professional/servers/__init__.py /app/axiom/mcp_professional/servers/__init__.py
COPY axiom/mcp_professional/servers/public/trading/__init__.py /app/axiom/mcp_professional/servers/trading/__init__.py

RUN mkdir -p /app/axiom/ai_layer/domain && \
    echo "from enum import Enum\nclass MarketOutlook(Enum):\n    BULLISH='bullish'\n    BEARISH='bearish'\n    NEUTRAL='neutral'\nclass VolatilityView(Enum):\n    INCREASING='increasing'\n    STABLE='stable'\n    DECREASING='decreasing'\nclass TradingStrategy:\n    pass\nclass StrategyLeg:\n    pass" > /app/axiom/ai_layer/domain/strategy_value_objects.py && \
    touch /app/axiom/ai_layer/__init__.py /app/axiom/ai_layer/domain/__init__.py

RUN mkdir -p /app/axiom/derivatives/advanced && \
    echo "class AIStrategyGenerator:\n    def __init__(self,**kwargs):pass\n    def generate_strategy(self,**kwargs):\n        class Result:\n            strategy_name='Bull Call Spread';legs=[];entry_cost=500;max_profit=1500;max_loss=500;expected_return=0.3;risk_reward_ratio=3.0;probability_profit=0.65;rationale='Bullish outlook'\n        return Result()" > /app/axiom/derivatives/advanced/strategy_generator.py && \
    touch /app/axiom/derivatives/__init__.py /app/axiom/derivatives/advanced/__init__.py

ENV PYTHONPATH=/app
CMD ["python", "-m", "axiom.mcp_professional.servers.trading.strategy_gen.server"]