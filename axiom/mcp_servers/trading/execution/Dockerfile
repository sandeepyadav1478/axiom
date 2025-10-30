FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY axiom/mcp_servers/trading/execution/ ./execution/
COPY axiom/mcp_servers/shared/ ./shared/
COPY axiom/derivatives/execution/ ./derivatives/execution/
COPY axiom/ai_layer/domain/execution_value_objects.py ./domain/
ENV PYTHONPATH=/app
ENV MCP_SERVER=smart-execution
CMD ["python", "-m", "execution.server"]