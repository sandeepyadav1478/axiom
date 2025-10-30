FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY axiom/mcp_servers/analytics/performance/ ./performance/
COPY axiom/mcp_servers/shared/ ./shared/
COPY axiom/derivatives/analytics/ ./derivatives/
ENV PYTHONPATH=/app
CMD ["python", "-m", "performance.server"]