FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY axiom/mcp_servers/trading/strategy_gen/ ./strategy_gen/
COPY axiom/mcp_servers/shared/ ./shared/
COPY axiom/derivatives/advanced/strategy_generator.py ./derivatives/
COPY axiom/ai_layer/domain/strategy_value_objects.py ./domain/

ENV PYTHONPATH=/app
ENV MCP_SERVER=strategy-generation
ENV LOG_LEVEL=INFO

CMD ["python", "-m", "strategy_gen.server"]