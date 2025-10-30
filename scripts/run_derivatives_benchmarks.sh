#!/bin/bash
# Comprehensive Benchmarking Script for Derivatives Platform
# Runs all performance tests and generates report

echo "============================================================"
echo "AXIOM DERIVATIVES PLATFORM - COMPREHENSIVE BENCHMARKS"
echo "============================================================"
echo "Running all performance tests..."
echo ""

# Set Python path
export PYTHONPATH=/Users/sandeep.yadav/work/axiom:$PYTHONPATH

# Activate virtual environment if exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Create output directory
mkdir -p benchmarks/results

# 1. Ultra-Fast Greeks Benchmark
echo "→ Benchmarking Ultra-Fast Greeks Engine..."
pytest tests/derivatives/test_ultra_fast_greeks.py --benchmark-only \
    --benchmark-json=benchmarks/results/greeks_benchmark.json \
    --benchmark-min-rounds=1000 \
    -v

# 2. Exotic Options Benchmark
echo ""
echo "→ Benchmarking Exotic Options Pricing..."
pytest tests/derivatives/test_exotic_options.py --benchmark-only \
    --benchmark-json=benchmarks/results/exotic_benchmark.json \
    -v

# 3. Integration Tests
echo ""
echo "→ Running Integration Tests..."
pytest tests/derivatives/test_integration.py -v

# 4. Load Testing (if Locust installed)
if command -v locust &> /dev/null; then
    echo ""
    echo "→ Running Load Test (30 seconds)..."
    locust -f tests/derivatives/load_test.py \
        --users 100 \
        --spawn-rate 10 \
        --run-time 30s \
        --host http://localhost:8000 \
        --headless \
        --csv=benchmarks/results/load_test
fi

# 5. Generate benchmark report
echo ""
echo "→ Generating Benchmark Report..."
python3 <<EOF
import json
import pandas as pd
from datetime import datetime

print("\n" + "="*60)
print("BENCHMARK RESULTS SUMMARY")
print("="*60)

# Load Greeks benchmark
try:
    with open('benchmarks/results/greeks_benchmark.json') as f:
        greeks_data = json.load(f)
    
    greeks_bench = greeks_data['benchmarks'][0]['stats']
    mean_us = greeks_bench['mean'] * 1_000_000
    p95_us = greeks_bench.get('q_95', greeks_bench['mean']) * 1_000_000
    
    print(f"\n1. GREEKS CALCULATION:")
    print(f"   Mean: {mean_us:.2f} microseconds")
    print(f"   P95: {p95_us:.2f} microseconds")
    print(f"   Target <100us: {'✓ PASS' if mean_us < 100 else '✗ FAIL'}")
    print(f"   vs Bloomberg (100ms): {100_000 / mean_us:.0f}x faster")
except:
    print("\n1. GREEKS: Benchmark file not found")

# Load Exotic benchmark
try:
    with open('benchmarks/results/exotic_benchmark.json') as f:
        exotic_data = json.load(f)
    
    print(f"\n2. EXOTIC OPTIONS:")
    for bench in exotic_data['benchmarks']:
        name = bench['name']
        mean_ms = bench['stats']['mean'] * 1000
        print(f"   {name}: {mean_ms:.2f}ms")
except:
    print("\n2. EXOTIC: Benchmark file not found")

# Load test results
try:
    load_stats = pd.read_csv('benchmarks/results/load_test_stats.csv')
    print(f"\n3. LOAD TEST:")
    print(f"   Total requests: {load_stats['Request Count'].sum()}")
    print(f"   Avg response time: {load_stats['Average Response Time'].mean():.2f}ms")
    print(f"   Requests/sec: {load_stats['Requests/s'].mean():.0f}")
except:
    print("\n3. LOAD TEST: Results not found")

print("\n" + "="*60)
print("BENCHMARK COMPLETE")
print("="*60)
print(f"Timestamp: {datetime.now().isoformat()}")
print(f"Results saved to: benchmarks/results/")
print("")
EOF

echo "✓ All benchmarks complete"
echo "✓ Results in benchmarks/results/"
echo ""
echo "Run this script regularly to track performance over time."