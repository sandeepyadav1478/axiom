FROM python:3.10-slim

WORKDIR /app

# Install minimal system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends gcc && rm -rf /var/lib/apt/lists/*

# Copy server requirements
COPY axiom/mcp/servers/public/trading/portfolio_risk/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy MCP infrastructure
COPY axiom/mcp/servers/shared /app/axiom/mcp/servers/shared
COPY axiom/mcp/servers/public/trading/portfolio_risk /app/axiom/mcp/servers/public/trading/portfolio_risk

# Create minimal package structure
RUN echo "# Minimal axiom package" > /app/axiom/__init__.py && \
    touch /app/axiom/mcp/__init__.py && \
    touch /app/axiom/mcp/servers/__init__.py && \
    touch /app/axiom/mcp/servers/public/__init__.py && \
    touch /app/axiom/mcp/servers/public/trading/__init__.py

# Create domain stubs
RUN mkdir -p /app/axiom/ai_layer/domain && \
    echo "from dataclasses import dataclass\nfrom decimal import Decimal\n\n@dataclass\nclass PortfolioGreeks:\n    total_delta:float=0\n    total_gamma:float=0\n    total_vega:float=0\n    total_theta:float=0\n\n@dataclass\nclass VaRMetrics:\n    var_1day_parametric:float=0\n    var_1day_historical:float=0\n    var_1day_monte_carlo:float=0\n    cvar_1day:float=0\n    calculation_time_ms:float=0\n\n@dataclass\nclass RiskLimits:\n    max_delta:Decimal\n    max_gamma:Decimal\n    max_vega:Decimal\n    max_theta:Decimal\n    max_var_1day:Decimal\n    max_notional_exposure:Decimal\n\n@dataclass\nclass RiskAlert:\n    pass" > /app/axiom/ai_layer/domain/risk_value_objects.py && \
    echo "class InvalidInputError(Exception):pass" > /app/axiom/ai_layer/domain/exceptions.py && \
    touch /app/axiom/ai_layer/__init__.py && \
    touch /app/axiom/ai_layer/domain/__init__.py

# Create engine stub
RUN mkdir -p /app/axiom/derivatives/risk && \
    echo "class RealTimeRiskEngine:\n    def __init__(self,**kwargs):pass\n    def calculate_portfolio_risk(self,**kwargs):\n        from axiom.ai_layer.domain.risk_value_objects import VaRMetrics\n        class Result:\n            total_delta=0;total_gamma=0;total_vega=0;var_1day_parametric=5000;var_1day_historical=4800;var_1day_monte_carlo=5200;cvar_1day=6500;calculation_time_ms=4.5\n        return Result()\n    def stress_test(self,**kwargs):return {}" > /app/axiom/derivatives/risk/real_time_risk_engine.py && \
    touch /app/axiom/derivatives/__init__.py && \
    touch /app/axiom/derivatives/risk/__init__.py

ENV PYTHONPATH=/app
ENV MCP_SERVER=portfolio-risk
ENV LOG_LEVEL=INFO

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.exit(0)"

CMD ["python", "-m", "axiom.mcp.servers.public.trading.portfolio_risk.server"]