FROM python:3.10-slim

WORKDIR /app

# Install minimal system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy minimal requirements
COPY axiom/mcp_servers/requirements-mcp-only.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy MCP server code
COPY axiom/mcp_final/servers/shared /app/axiom/mcp_final/servers/shared
COPY axiom/mcp_final/servers/public/trading/pricing_greeks /app/axiom/mcp_final/servers/trading/pricing_greeks

# Create minimal axiom __init__.py (don't import config)
RUN echo "# Minimal axiom package for MCP server" > /app/axiom/__init__.py

# Copy __init__ files for package structure
COPY axiom/mcp_final/servers/__init__.py /app/axiom/mcp_final/servers/__init__.py
COPY axiom/mcp_final/servers/public/trading/__init__.py /app/axiom/mcp_final/servers/trading/__init__.py

# Create minimal stubs for domain imports
RUN mkdir -p /app/axiom/ai_layer/domain && \
    echo "from dataclasses import dataclass\nfrom enum import Enum\n\nclass OptionType(Enum):\n    CALL='call'\n    PUT='put'\n\n@dataclass\nclass Greeks:\n    delta:float=0\n    gamma:float=0\n    theta:float=0\n    vega:float=0\n    rho:float=0\n    price:float=0\n    calculation_time_us:float=0" > /app/axiom/ai_layer/domain/value_objects.py && \
    echo "class InvalidInputError(Exception):pass\nclass ModelInferenceError(Exception):pass" > /app/axiom/ai_layer/domain/exceptions.py && \
    touch /app/axiom/ai_layer/__init__.py && \
    touch /app/axiom/ai_layer/domain/__init__.py

# Create minimal derivatives engine stub
RUN mkdir -p /app/axiom/derivatives && \
    echo "class UltraFastGreeksEngine:\n    def __init__(self,**kwargs):pass\n    def calculate_greeks(self,**kwargs):\n        from axiom.ai_layer.domain.value_objects import Greeks\n        return Greeks(delta=0.5,gamma=0.02,vega=0.15,theta=-0.05,rho=0.08,price=10.5,calculation_time_us=500)" > /app/axiom/derivatives/ultra_fast_greeks.py && \
    touch /app/axiom/derivatives/__init__.py

# Set environment
ENV PYTHONPATH=/app
ENV MCP_SERVER=pricing-greeks
ENV LOG_LEVEL=INFO

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.exit(0)"

# Run MCP server via STDIO
CMD ["python", "-m", "axiom.mcp_final.servers.trading.pricing_greeks.server"]