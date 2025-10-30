#!/bin/bash
# Performance Testing Script

echo "Axiom Platform Performance Tests"
echo "================================="

# Test model loading speed
echo -e "\n1. Model Loading Performance"
python -c "
import time
from axiom.models.base.model_cache import get_cached_model
from axiom.models.base.factory import ModelType

start = time.time()
model = get_cached_model(ModelType.PORTFOLIO_TRANSFORMER)
elapsed = (time.time() - start) * 1000
print(f'  Portfolio Transformer: {elapsed:.1f}ms')

start = time.time()
model2 = get_cached_model(ModelType.ANN_GREEKS_CALCULATOR)
elapsed = (time.time() - start) * 1000
print(f'  ANN Greeks: {elapsed:.1f}ms')
"

# Test API response time
echo -e "\n2. API Response Time"
curl -o /dev/null -s -w "  GET /models: %{time_total}s\n" http://localhost:8000/models

# Test batch processing
echo -e "\n3. Batch Processing Throughput"
python -c "
import time
import asyncio
from axiom.infrastructure.batch_inference_engine import BatchInferenceEngine

async def test():
    engine = BatchInferenceEngine()
    print(f'  Max workers: {engine.max_workers}')
    print(f'  Max batch size: {engine.max_batch_size}')

asyncio.run(test())
"

# Memory usage
echo -e "\n4. Memory Usage"
ps aux | grep python | grep axiom | awk '{print \"  Process:\", $11, \"Memory:\", $6/1024 \"MB\"}'

echo -e "\n✓ Performance tests complete"