FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY axiom/mcp_servers/trading/portfolio_risk/ ./portfolio_risk/
COPY axiom/mcp_servers/shared/ ./shared/
COPY axiom/derivatives/risk/ ./derivatives/risk/
COPY axiom/ai_layer/domain/risk_value_objects.py ./domain/

ENV PYTHONPATH=/app
ENV MCP_SERVER=portfolio-risk
ENV LOG_LEVEL=INFO

CMD ["python", "-m", "portfolio_risk.server"]