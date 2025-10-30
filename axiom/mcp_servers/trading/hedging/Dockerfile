FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY axiom/mcp_servers/trading/hedging/ ./hedging/
COPY axiom/mcp_servers/shared/ ./shared/
COPY axiom/derivatives/market_making/ ./derivatives/
COPY axiom/ai_layer/domain/hedging_value_objects.py ./domain/
ENV PYTHONPATH=/app
CMD ["python", "-m", "hedging.server"]